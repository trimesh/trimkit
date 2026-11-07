"""
The scheduler in src/parallel.hpp, as a state machine checked in every
interleaving: each step below is one atomic operation of the C++, named for
the function it belongs to, and a breadth-first search visits every state
callers and workers can reach, asserting in each:

- S1: no thread touches a job after its caller has returned (it lives on
  the caller's stack)
- S2: no item runs twice
- L1: from every state, every call can still return, and does with nothing
  stopped having run each item once
- P1: no worker sleeps through a wake: asleep with `wake` past what it saw,
  and no wake on its way

Sleeps have no timeouts here (but idle exit), so a lost wakeup is a hang,
not a 50 ms delay. Then each mutation breaks one guard of the C++, and the
search must find what it breaks, so the checker is shown able to fail.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import pytest


class Slot(NamedTuple):
    epoch: int
    open: bool
    active: int
    job: int
    left: int


class Job(NamedTuple):
    count: int
    next: int
    stop: bool
    done: int  # a bit per item run
    returned: bool
    nest_at: int  # the item that calls run_job itself, or -1
    throw_at: int  # the item that throws, or -1


class Frame(NamedTuple):
    kind: str  # "caller" or "worker"
    pc: str
    job: int = -1  # a caller's own job, or the job a worker loaded
    slot: int = -1  # a caller's published slot, or the slot a worker is at
    tried: bool = False  # a caller has tried to publish
    held: int = -1  # the item claimed
    seen: int = 0  # a caller's `left` as read, or the epoch a worker saw
    epoch: int = 0  # the epoch a caller published under
    observed: int = 0  # `wake` as a worker saw it before scanning
    joined: bool = False  # a worker joined a job this scan
    drained: tuple = ()  # per slot, the epoch a worker last found empty


class Thread(NamedTuple):
    frames: tuple  # the innermost call last; empty: not started, or exited
    calls: int  # calls a caller thread has still to make
    caller: bool


class State(NamedTuple):
    slots: tuple
    jobs: tuple
    threads: tuple
    wake: int
    live: int
    wanted: int
    interrupt: bool  # Ctrl-C pending
    forked: bool


@dataclass(frozen=True)
class Scenario:
    name: str
    callers: int = 1
    calls: int = 2
    workers: int = 1
    slots: int = 2
    items: int = 2
    throw_at: int = -1
    nest_at: int = -1
    interrupt: bool = False
    spawn_fails: bool = False
    idle_exit: bool = False
    fork: bool = False


class Violation(Exception):
    pass


def new_job(scenario: Scenario, nested: bool = False) -> Job:
    # a top-level call's items, or the two of a call made from inside an item
    if nested:
        return Job(2, 0, False, 0, False, -1, -1)
    return Job(scenario.items, 0, False, 0, False, scenario.nest_at, scenario.throw_at)


def caller_frame(job: int) -> Frame:
    return Frame("caller", "loop", job=job)


def initial(scenario: Scenario) -> State:
    jobs = tuple(new_job(scenario) for _ in range(scenario.callers))
    threads = tuple(Thread((caller_frame(c),), scenario.calls - 1, True) for c in range(scenario.callers))
    threads += tuple(Thread((), 0, False) for _ in range(scenario.workers))
    slots = tuple(Slot(0, False, 0, -1, 0) for _ in range(scenario.slots))
    return State(slots, jobs, threads, 0, 0, 0, scenario.interrupt, False)


# -- small immutable updates ---------------------------------------------------


def put(items: tuple, index: int, value) -> tuple:
    return (*items[:index], value, *items[index + 1 :])


def set_frame(state: State, t: int, frame: Frame) -> State:
    thread = state.threads[t]
    return state._replace(threads=put(state.threads, t, thread._replace(frames=(*thread.frames[:-1], frame))))


def set_slot(state: State, k: int, **changes) -> State:
    return state._replace(slots=put(state.slots, k, state.slots[k]._replace(**changes)))


def set_job(state: State, j: int, **changes) -> State:
    return state._replace(jobs=put(state.jobs, j, state.jobs[j]._replace(**changes)))


def push(state: State, t: int, frame: Frame) -> State:
    thread = state.threads[t]
    return state._replace(threads=put(state.threads, t, thread._replace(frames=(*thread.frames, frame))))


def pop(state: State, t: int) -> State:
    thread = state.threads[t]
    return state._replace(threads=put(state.threads, t, thread._replace(frames=thread.frames[:-1])))


def touch(state: State, j: int, what: str) -> None:
    if state.jobs[j].returned:
        raise Violation(f"S1: {what} touched job {j} after its caller returned")


# -- the transitions, each one atomic step of src/parallel.hpp ----------------------


@dataclass(frozen=True)
class At:
    # one thread about to take its next step: its innermost frame
    state: State
    t: int
    frame: Frame
    scenario: Scenario
    mutations: frozenset

    @property
    def slot(self) -> Slot:
        return self.state.slots[self.frame.slot]

    @property
    def job(self) -> Job:
        return self.state.jobs[self.frame.job]

    def go(self, pc: str, state: State | None = None, **changes) -> State:
        # the thread at its next step, in `state` (by default as it is)
        return set_frame(
            self.state if state is None else state, self.t, self.frame._replace(pc=pc, **changes)
        )


def spawn_worker(state: State, scenario: Scenario):
    # live < cap: one more thread; creation may fail, which changes nothing
    if state.live >= scenario.workers:
        yield state
        return
    idle = next(t for t, thread in enumerate(state.threads) if not thread.caller and not thread.frames)
    worker = Frame("worker", "start", drained=(-1,) * scenario.slots)
    yield push(state._replace(live=state.live + 1), idle, worker)
    if scenario.spawn_fails:
        yield state


def wake_sleepers(state: State, asleep, pc: str) -> State:
    for t, thread in enumerate(state.threads):
        if thread.frames and asleep(thread.frames[-1]):
            state = set_frame(state, t, thread.frames[-1]._replace(pc=pc))
    return state


def claim_range(at: At):
    touch(at.state, at.frame.job, f"claim_range on thread {at.t}")
    job = at.job
    if job.stop or job.next >= job.count:
        if at.frame.kind == "worker":
            yield "claim_range (none left)", at.go("leave")
        else:
            yield "claim_range (none left)", at.go("close" if at.frame.slot >= 0 else "return")
    elif "claim_load_then_store" in at.mutations:
        yield "claim_range (load)", at.go("claim_store", held=job.next)
    else:
        yield "claim_range", at.go("run", set_job(at.state, at.frame.job, next=job.next + 1), held=job.next)


def claim_store(at: At):
    yield "claim_range (store)", at.go("run", set_job(at.state, at.frame.job, next=at.frame.held + 1))


def run_range(at: At):
    touch(at.state, at.frame.job, f"run_range on thread {at.t}")
    if at.job.nest_at == at.frame.held:
        # the item calls run_job itself: a nested caller on this thread
        state = at.state._replace(jobs=(*at.state.jobs, new_job(at.scenario, nested=True)))
        yield "run_range (nested call)", push(at.go("nested", state), at.t, caller_frame(len(state.jobs) - 1))
    else:
        yield "run_range", finish_item(at)


def nested_returned(at: At):
    yield "run_range (nested call returned)", finish_item(at)


def finish_item(at: At) -> State:
    touch(at.state, at.frame.job, f"run_range on thread {at.t}")
    job = at.job
    bit = 1 << at.frame.held
    if job.done & bit:
        raise Violation(f"S2: item {at.frame.held} of job {at.frame.job} ran twice")
    # publish_error: the item that throws stops the job
    state = set_job(
        at.state, at.frame.job, done=job.done | bit, stop=job.stop or job.throw_at == at.frame.held
    )
    return at.go("claim" if at.frame.kind == "worker" else "loop", state, held=-1)


def caller_loop(at: At):
    yield from claim_range(at)
    if not at.frame.tried and at.job.next < at.job.count:
        free = [k for k, slot in enumerate(at.state.slots) if not slot.open and slot.active == 0]
        if not free:
            yield "try_publish (every slot busy)", at.go("loop", tried=True)
        else:
            epoch = at.state.slots[free[0]].epoch + (0 if "publish_keeps_epoch" in at.mutations else 1)
            reserved = set_slot(at.state, free[0], epoch=epoch, active=1)
            yield "try_publish (reserve)", at.go("store_job", reserved, slot=free[0], tried=True, epoch=epoch)
    thread = at.state.threads[at.t]
    if at.state.interrupt and len(thread.frames) == 1 and thread.caller:
        stopped = set_job(at.state._replace(interrupt=False), at.frame.job, stop=True)
        yield "poll_interrupt", stopped


def store_job(at: At):
    yield "try_publish (store job)", at.go("open", set_slot(at.state, at.frame.slot, job=at.frame.job))


def open_slot(at: At):
    yield "try_publish (open)", at.go("wake_spawn", set_slot(at.state, at.frame.slot, open=True))


def wake_spawn(at: At):
    state = at.go("wake_bump", at.state._replace(wanted=at.scenario.workers))
    for spawned in spawn_worker(state, at.scenario):
        yield "wake_workers (spawn_worker)", spawned


def wake_bump(at: At):
    yield "wake_workers (bump)", at.go("wake_unpark", at.state._replace(wake=at.state.wake + 1))


def wake_unpark(at: At):
    asleep = lambda frame: frame.kind == "worker" and frame.pc == "sleep"  # noqa: E731
    yield "wake_workers (unpark)", wake_sleepers(at.go("loop"), asleep, "scan_start")


def close_job(at: At):
    before = at.slot.active
    state = set_slot(at.state, at.frame.slot, open=False, active=before - 1)
    waits = before != 1 and "close_without_waiting" not in at.mutations
    first = "wait_check_early" if "wait_checks_count_first" in at.mutations else "wait_load"
    yield "close_job", at.go(first if waits else "return", state)


def workers_gone(at: At) -> bool:
    # none left in this caller's job: none active, or the slot published
    # again since, which needs it free first
    moved_on = at.slot.epoch != at.frame.epoch and "wait_ignores_epoch" not in at.mutations
    return at.slot.active == 0 or moved_on


def wait_load(at: At):
    yield "wait_for_workers (load left)", at.go("wait_check", seen=at.slot.left)


def wait_check(at: At):
    yield "wait_for_workers (check)", at.go("return" if workers_gone(at) else "park")


def wait_check_early(at: At):
    yield "wait_for_workers (check first)", at.go("return" if workers_gone(at) else "wait_load_late")


def wait_load_late(at: At):
    yield "wait_for_workers (load left late)", at.go("park", seen=at.slot.left)


def caller_park(at: At):
    # the kernel compares and sleeps in one step
    yield "wait_for_workers (park)", at.go("wait_load" if at.slot.left != at.frame.seen else "sleep_left")


def caller_return(at: At):
    state = set_job(pop(at.state, at.t), at.frame.job, returned=True)
    thread = state.threads[at.t]
    if not thread.frames and thread.caller and thread.calls > 0:
        state = state._replace(jobs=(*state.jobs, new_job(at.scenario)))
        state = state._replace(threads=put(state.threads, at.t, thread._replace(calls=thread.calls - 1)))
        state = push(state, at.t, caller_frame(len(state.jobs) - 1))
    yield "run_job (return)", state


def worker_start(at: At):
    # the spawn chain: each new worker starts the next
    state = at.go("scan_start")
    if state.live < state.wanted:
        for spawned in spawn_worker(state, at.scenario):
            yield "worker_main (spawn_worker)", spawned
    else:
        yield "worker_main (start)", state


def scan_start(at: At):
    yield "worker_main (load wake)", at.go("load_word", observed=at.state.wake, slot=0, joined=False)


def load_word(at: At):
    k = at.frame.slot
    if k == at.scenario.slots:
        yield "worker_main (scanned)", at.go("scan_start" if at.frame.joined else "park")
    elif not at.slot.open or at.slot.epoch == at.frame.drained[k]:
        yield "worker_main (skip slot)", at.go("load_word", slot=k + 1)
    else:
        yield "worker_main (load word)", at.go("load_job", seen=at.slot.epoch)


def load_job(at: At):
    yield "worker_main (load job)", at.go("join", job=at.slot.job)


def join_job(at: At):
    slot = at.slot
    if slot.epoch == at.frame.seen and (slot.open or "join_ignores_open" in at.mutations):
        yield "join_job", at.go("claim", set_slot(at.state, at.frame.slot, active=slot.active + 1))
    else:
        yield "join_job (failed)", at.go("load_word", slot=at.frame.slot + 1)


def leave_job(at: At):
    k, slot = at.frame.slot, at.slot
    state = set_slot(at.state, k, active=slot.active - 1)
    drained = put(at.frame.drained, k, at.frame.seen)
    if not slot.open and slot.active == 1:
        yield "leave_job (last out)", at.go("bump_left", state, drained=drained)
    else:
        yield "leave_job", at.go("load_word", state, slot=k + 1, joined=True, drained=drained)


def bump_left(at: At):
    yield (
        "leave_job (bump left)",
        at.go("unpark_left", set_slot(at.state, at.frame.slot, left=at.slot.left + 1)),
    )


def unpark_left(at: At):
    k = at.frame.slot
    asleep = lambda frame: frame.kind == "caller" and frame.pc == "sleep_left" and frame.slot == k  # noqa: E731
    yield (
        "leave_job (unpark)",
        wake_sleepers(at.go("load_word", slot=k + 1, joined=True), asleep, "wait_load"),
    )


def worker_park(at: At):
    sleeps = at.state.wake == at.frame.observed or "park_ignores_wake" in at.mutations
    yield "park_worker", at.go("sleep" if sleeps else "scan_start")


def nothing(at: At):
    return iter(())


# each program counter's step, for callers and for workers; asleep, a
# thread steps only when another wakes it
CALLER = {
    "loop": caller_loop,
    "claim_store": claim_store,
    "run": run_range,
    "nested": nested_returned,
    "store_job": store_job,
    "open": open_slot,
    "wake_spawn": wake_spawn,
    "wake_bump": wake_bump,
    "wake_unpark": wake_unpark,
    "close": close_job,
    "wait_load": wait_load,
    "wait_check": wait_check,
    "wait_check_early": wait_check_early,
    "wait_load_late": wait_load_late,
    "park": caller_park,
    "sleep_left": nothing,
    "return": caller_return,
}
WORKER = {
    "start": worker_start,
    "scan_start": scan_start,
    "load_word": load_word,
    "load_job": load_job,
    "join": join_job,
    "claim": claim_range,
    "claim_store": claim_store,
    "run": run_range,
    "nested": nested_returned,
    "leave": leave_job,
    "bump_left": bump_left,
    "unpark_left": unpark_left,
    "park": worker_park,
    "sleep": nothing,
}


def child_after_fork(state: State, scenario: Scenario) -> State:
    # only the forking thread is left, outside any call, which then calls
    slots = tuple(Slot(s.epoch + 1, False, 0, -1, s.left) for s in state.slots)
    threads = tuple(Thread((), 0, thread.caller) for thread in state.threads)
    jobs = (*state.jobs, new_job(scenario))
    threads = put(threads, 0, Thread((caller_frame(len(jobs) - 1),), 0, True))
    return State(slots, jobs, threads, state.wake, 0, 0, False, True)


def successors(state: State, scenario: Scenario, mutations: frozenset):
    for t, thread in enumerate(state.threads):
        if not thread.frames:
            continue
        at = At(state, t, thread.frames[-1], scenario, mutations)
        steps = CALLER if at.frame.kind == "caller" else WORKER
        for name, following in steps[at.frame.pc](at):
            yield f"thread {t}: {name}", following
        exits = at.frame.pc == "sleep" or ("exit_while_joined" in mutations and at.frame.pc == "claim")
        if at.frame.kind == "worker" and scenario.idle_exit and exits:
            yield f"thread {t}: exit_idle", pop(state._replace(live=state.live - 1), t)
    if scenario.fork and not state.forked:
        yield "fork: child_after_fork", child_after_fork(state, scenario)


def terminal(state: State) -> bool:
    calls_left = any(thread.caller and thread.calls > 0 for thread in state.threads)
    calling = any(frame.kind == "caller" for thread in state.threads for frame in thread.frames)
    return not calls_left and not calling


def check_state(state: State) -> None:
    unparking = any(t.frames and t.frames[-1].pc == "wake_unpark" for t in state.threads)
    for thread in state.threads:
        for frame in thread.frames:
            if (
                frame.kind == "worker"
                and frame.pc == "sleep"
                and frame.observed != state.wake
                and not unparking
            ):
                raise Violation("P1: a worker sleeps through a wake")
    if terminal(state):
        for j, job in enumerate(state.jobs):
            if job.returned and not job.stop and job.done != (1 << job.count) - 1:
                raise Violation(f"L1: job {j} returned without running every item")


def explore(scenario: Scenario, mutations: frozenset = frozenset()) -> int:
    """
    Every state reachable from the start, checked; returns how many there are.
    """
    start = initial(scenario)
    parent = {start: None}
    edges: dict[State, list[State]] = {}
    queue = deque([start])

    def trace(state: State) -> str:
        steps = []
        while parent[state] is not None:
            state, name = parent[state]
            steps.append(name)
        return "\n  ".join(reversed(steps[-40:]))

    while queue:
        state = queue.popleft()
        try:
            check_state(state)
            following = list(successors(state, scenario, mutations))
        except Violation as violation:
            raise Violation(f"{violation} in {scenario.name}, after:\n  {trace(state)}") from None
        edges[state] = [nxt for _, nxt in following]
        if not following and not terminal(state):
            raise Violation(f"L1: deadlock in {scenario.name}, after:\n  {trace(state)}")
        for name, nxt in following:
            if nxt not in parent:
                parent[nxt] = (state, name)
                queue.append(nxt)
    # L1: every state can still reach the end, so nothing hangs or loops forever
    reverse: dict[State, list[State]] = {s: [] for s in edges}
    for state, following in edges.items():
        for nxt in following:
            reverse[nxt].append(state)
    finishing = {s for s in edges if terminal(s)}
    queue = deque(finishing)
    while queue:
        for before in reverse[queue.popleft()]:
            if before not in finishing:
                finishing.add(before)
                queue.append(before)
    stuck = [s for s in edges if s not in finishing]
    if stuck:
        raise Violation(f"L1: a call can never return in {scenario.name}, after:\n  {trace(stuck[0])}")
    return len(edges)


SCENARIOS = [
    Scenario("two sequential calls"),
    Scenario("two workers", calls=1, workers=2),
    Scenario("an item throws", calls=1, throw_at=0),
    Scenario("ctrl-c", calls=1, interrupt=True),
    Scenario("thread creation fails", calls=1, workers=2, spawn_fails=True),
    Scenario("workers exit idle, then start again", idle_exit=True),
    Scenario("fork", calls=1, fork=True),
    Scenario("a nested call", calls=1, nest_at=0),
    Scenario("two callers, one slot", callers=2, calls=1, slots=1),
    Scenario("two callers, two slots", callers=2, calls=1),
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_every_interleaving(scenario):
    assert explore(scenario) > 100


# each breaks one guard of src/parallel.hpp; the search must find a scenario
# where that goes wrong, and say which property broke
MUTATIONS = {
    "join_ignores_open": ("S1", Scenario("join closed", calls=1)),
    "close_without_waiting": ("S1|L1", Scenario("no wait", calls=1)),
    "publish_keeps_epoch": ("S1", Scenario("no epoch")),
    "claim_load_then_store": ("S2", Scenario("racy claim", calls=1)),
    "park_ignores_wake": ("P1", Scenario("lost wake")),
    "exit_while_joined": ("L1", Scenario("exit holding a job", calls=1, idle_exit=True)),
    "wait_checks_count_first": ("L1", Scenario("caller sleeps through the last leave", calls=1)),
    "wait_ignores_epoch": ("L1", Scenario("caller waits on the next job", callers=2, calls=1, slots=1)),
}


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_mutations_are_caught(mutation):
    expected, scenario = MUTATIONS[mutation]
    with pytest.raises(Violation, match=f"^({expected}):"):
        explore(scenario, frozenset([mutation]))
