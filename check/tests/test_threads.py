"""
Every function releases the GIL, so calls really do run at once: the same
inputs must give bit-identical answers on 8 threads as they do serially. And
the pool lives through any sequence of calls: sizes in any order, idle gaps
around its idle exit, one to four callers, forks, Ctrl-C and the exit.
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from helpers import CASES, call, cpus_allowed, run_python, runs_here, same_answers


@pytest.mark.parametrize("name", sorted(CASES))
def test_parallel_matches_serial(name):
    # each public call's small forms, the same answer to the bit on 8 threads at once
    runs_here(name)
    forms = CASES[name].arguments("small")
    expected = [call(name, form) for form in forms]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda k: call(name, forms[k % len(forms)]), range(32)))
    for k, result in enumerate(results):
        assert same_answers(result, expected[k % len(forms)])


def test_free_threaded_stays_free():
    # importing trimkit must not turn the GIL back on
    if not hasattr(sys, "_is_gil_enabled") or sys.flags.gil == 1:
        pytest.skip("not a free-threaded interpreter")
    assert not sys._is_gil_enabled()


def test_shared_scene():
    # one scene queried from many threads at once, with batches big enough
    # that each query also splits across its own worker threads
    from helpers import first_hits, sphere

    import trimkit

    vertices, faces = sphere(30, 60)
    origins, directions = np.random.default_rng(0).normal(size=(2, 20_000, 3))
    scene = trimkit.RayScene([(vertices, faces)])
    expected = first_hits(scene, origins, directions)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: first_hits(scene, origins, directions), range(8)))
    del scene
    assert all(same_answers(result, expected) for result in results)


def test_scheduler_stress():
    # eight threads at once, each mixing queries small enough to run alone
    # and big enough to wake the pool's workers: every caller shares the same
    # workers through parallel.hpp's slots, and still gets exactly the answer
    # it gets alone
    from helpers import first_hits, sphere

    import trimkit

    rng = np.random.default_rng(1)
    corners = rng.uniform(0, 100, (20_000, 3))
    tree = trimkit.Boxes(np.stack([corners, corners + 1.0], axis=1))
    scene = trimkit.RayScene([sphere(30, 60)])
    solids = trimkit.CollisionScene([sphere(8, 16)], convex=True)
    matrices = np.tile(np.eye(4), (120, 1, 1))
    matrices[:, :3, 3] = rng.uniform(0, 10, (120, 3))
    index = np.zeros(120, np.int64)
    queries = []
    for size in (10, 20_000):
        boxes = rng.uniform(0, 100, (size, 3))
        rays = rng.normal(size=(2, size, 3)) * 0.1
        queries += [
            lambda b=boxes: tree.intersection(np.stack([b, b + 3.0], axis=1)),
            lambda b=boxes: tree.nearest(np.stack([b, b + 3.0], axis=1), maximum_hits=4),
            lambda r=rays: first_hits(scene, *r),
            lambda r=rays: scene.intersect(*r),
        ]
    queries += [
        lambda: solids.distances(matrices, index),
        lambda: solids.collides(matrices, index),
        lambda: solids.collides(matrices, index, return_distance=True, maximum_contacts=4),
    ]
    expected = [query() for query in queries]

    def mixed(seed: int) -> bool:
        order = np.random.default_rng(seed).permutation(len(queries) * 3) % len(queries)
        return all(same_answers(queries[q](), expected[q]) for q in order)

    with ThreadPoolExecutor(8) as pool:
        assert all(pool.map(mixed, range(16)))


# A child forked while a call runs may only exec or exit, as POSIX says of a
# child of a threaded process (src/parallel.hpp, what it does not promise):
# Linux lets it run on, so only there is what it does next tested
runs_on_after_fork = pytest.mark.skipif(
    sys.platform != "linux", reason="a child forked during a call may only exec or exit (POSIX)"
)


@runs_on_after_fork
def test_fork_while_threads_swap_placements():
    # threads keep swapping one scene's remembered placement while the main
    # thread forks again and again: every child queries that same scene and
    # gets the parent's answer, never stuck on a lock a vanished thread held
    import threading

    from helpers import box, first_hits, forked_answer, sphere

    import trimkit

    rng = np.random.default_rng(11)
    meshes = [sphere(10, 20), box()]
    index = rng.integers(0, 2, 20)
    placements = []
    for _ in range(2):
        matrices = np.tile(np.eye(4), (20, 1, 1))
        matrices[:, :3, 3] = rng.uniform(-10, 10, (20, 3))
        placements.append(matrices)
    # few rays, and a child on one CPU, so it answers on its own thread
    # alone: the pool starts no worker there, at any speed (a sanitizer's
    # included, which can't follow threads a forked child starts)
    origins = rng.uniform(-15, 15, (32, 3))
    directions = placements[0][rng.integers(0, 20, 32), :3, 3] - origins
    scene = trimkit.RayScene(meshes)
    expected = first_hits(scene, origins, directions, placements[0], index)
    stop = threading.Event()

    def churn(k):
        while not stop.is_set():
            first_hits(scene, origins, directions, placements[k % 2], index)

    threads = [threading.Thread(target=churn, args=(k,)) for k in range(4)]
    for thread in threads:
        thread.start()
    try:

        def child():
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
            return first_hits(scene, origins, directions, placements[0], index)

        for _ in range(20):
            answer = forked_answer(child, expected)
            assert answer == b"1", "the child hung" if answer == b"" else "the child answered differently"
    finally:
        stop.set()
        for thread in threads:
            thread.join()


# One sequence of calls in a fresh interpreter, ours alone: every order of
# sizes, each step every caller at once and then an idle gap, the pool's
# threads counted throughout. It fails the moment a pool thread starts after
# one exited with no call since (helpers.PoolCensus), when they outlive twice
# the idle period after the last call, or a call answers otherwise than alone.
# Then a fork, Ctrl-C or the interpreter's exit lands inside a long call at
# `fraction` of it.
# Prints what it saw as JSON, and for an exit the moment it exits after.
_SEQUENCE = """
import _thread, json, os, signal, sys, threading, time
sys.path.insert(0, sys.argv[1])
from helpers import CASES, PoolCensus, call, cpus_allowed, long_call, pool_threads, same_answers

idle, gap, callers, slot = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
disruption, fraction = sys.argv[6], float(sys.argv[7])
ORDERS = {
    "ascending": ["empty", "small", "big"],
    "descending": ["big", "small", "empty"],
    "big then small": ["big", "small"],
    "alternating": ["small", "big", "small", "big"],
}
names = sorted(CASES)


def arguments(name, size):
    given = CASES[name].arguments(size)
    return given[0] if size == "small" else given


# each step one public name in turn, a different one in each slot, beside
# the query that wakes every worker the cap allows when it is big
steps, turn = [], slot * 7
for order in ORDERS.values():
    sizes = []
    for size in order:
        name = names[turn % len(names)]
        turn += 1
        chosen = [n for n in (name, "RayScene.intersect") if not isinstance(getattr(CASES[n], size), str)]
        sizes.append([(n, size) for n in dict.fromkeys(chosen)])
    steps.append(sizes)
expected = {key: call(key[0], arguments(*key)) for order in steps for step in order for key in step}
census, seen = PoolCensus(), set()


def quiet(seconds):
    # no call runs: no pool thread may start once one has exited
    end = time.monotonic() + seconds
    while True:
        seen.update(census.between_calls())
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(0.1, left))


def every_caller(step):
    # every caller makes the step's calls at once, each answering as alone
    given = [[arguments(*key) for key in step] for _ in range(callers)]
    together = threading.Barrier(callers)
    wrong = []

    def caller(k):
        together.wait()
        for key, arguments_given in zip(step, given[k]):
            if not same_answers(call(key[0], arguments_given), expected[key]):
                wrong.append(key)

    threads = [threading.Thread(target=caller, args=(k,)) for k in range(callers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not wrong, f"answered otherwise with {callers} callers: {wrong}"


for order in steps:
    for i, step in enumerate(order):
        every_caller(step)
        census.returned()
        quiet(gap * idle if i + 1 < len(order) else 0.0)
last = time.monotonic()
while pool_threads():
    quiet(0.1)
    assert time.monotonic() - last < 2 * idle, "pool threads outlived twice the idle period"
drained = time.monotonic() - last
# the calls woke workers, or this tested nothing: none exist only on one CPU
assert seen or cpus_allowed() == 1, "no pool thread was ever seen: still named trimkit?"

# a second here, however many cores, so what lands at a fraction of it lands inside
query, _ = long_call()
answer = query()
took = float("inf")
for _ in range(2):
    tic = time.perf_counter()
    query()
    took = min(took, time.perf_counter() - tic)
report = {"drained": drained, "seen": len(seen), "disruption": disruption}
if disruption == "ctrl-c":
    # raised, or answered whole, and the next call is the same
    timer = threading.Timer(fraction * took, _thread.interrupt_main)
    inside = True
    timer.start()
    try:
        assert same_answers(query(), answer), "Ctrl-C left part of an answer"
        inside = False
        # the call finished first: the interrupt lands here, between sleeps
        for _ in range(1000):
            time.sleep(0.01)
    except KeyboardInterrupt:
        report["landed"] = inside
    timer.join()
    assert same_answers(query(), answer), "the next call answered otherwise"
elif disruption == "fork":
    # a signal handler forks while the query polls for Ctrl-C, its workers
    # mid-range: the child's query raises or answers whole, and its pool
    # answers again; the parent's answer is the answer
    forked, child = [], []

    def fork(signum, frame):
        if not forked:
            forked.append(True)
            if os.fork() == 0:
                child.append(True)

    signal.signal(signal.SIGALRM, fork)
    # the first fork at this fraction of the fastest query; the timer
    # repeats, so one lands inside a faster one too, and a query the
    # machine ran faster than it measured is tried again, sooner
    for attempt in range(6):
        signal.setitimer(signal.ITIMER_REAL, fraction * took / 2**attempt, took / 20)
        try:
            outcome = "same" if same_answers(query(), answer) else "different"
        except RuntimeError:
            outcome = "raised"
        signal.setitimer(signal.ITIMER_REAL, 0)
        if forked:
            break
    if child:
        os.write(1, json.dumps({"child": outcome, "again": same_answers(query(), answer)}).encode() + b"\\n")
        os._exit(0)
    assert forked, "the timer never fired inside the query"
    assert outcome == "same", outcome
    _, status = os.wait()
    assert status == 0, f"the child ended with {os.waitstatus_to_exitcode(status)}"
print(json.dumps(report), flush=True)
if disruption == "exit":
    # the interpreter exits while a daemon thread is inside a query on the
    # pool: the atexit handler stops it at its next poll, before CPython
    # finalizes, so no worker is left inside a job on a vanished stack
    started = threading.Event()

    def forever():
        while True:
            started.set()
            query()
            os.write(1, b"answered\\n")

    threading.Thread(target=forever, daemon=True).start()
    started.wait()
    time.sleep(fraction * took)
    # at 10% to 50% of a query: one that answers after this line is one the
    # exit waited for, however fast or slow this machine is
    os.write(1, b"exiting\\n")
"""


def idle_exit_seconds() -> float:
    """
    How long a pool thread runs nothing before it exits: IDLE_EXIT_MS, as
    src/parallel.hpp sets it.
    """
    source = (Path(__file__).parents[2] / "src" / "parallel.hpp").read_text()
    milliseconds = re.search(r"IDLE_EXIT_MS\s*=[^;]*?([\d']+)\s*;", source)[1]
    return int(milliseconds.replace("'", "")) / 1000


@pytest.mark.timing
@pytest.mark.skipif(sys.platform != "linux", reason="counts threads in /proc; forks inside a call (POSIX)")
def test_sequences_of_calls():
    # the product of every order of sizes, idle gaps either side of the
    # pool's idle exit, and one to four callers at once, each a fresh
    # interpreter, four at once: each peaks near 140 MB on 16 cores, beside a
    # suite that may hold 1 GB (Mesa's Vulkan device, once a test opened it),
    # within the gate's 2 GB.
    # Then in each a fork, Ctrl-C or the exit lands in a long call at a
    # fraction of it

    idle = idle_exit_seconds()
    runs = [(gap, callers) for gap in (1.05, 0.95, 0.5, 0.0) for callers in (1, 2, 3, 4)]
    kinds = ("fork", "exit", "ctrl-c")

    def one(slot: int):
        gap, callers = runs[slot]
        kind = kinds[slot % 3]
        # an exit lands by half a query: a query that ran faster than the
        # fastest measured, and finished in the moment between "exiting" and
        # the exit's stop, would answer after it with nothing at fault
        fraction = (slot % 8 + 1) / 10 if kind != "exit" else (slot % 5 + 1) / 10
        done = run_python(
            "-W",
            "ignore",
            "-c",
            _SEQUENCE,
            str(Path(__file__).parent),
            str(idle),
            str(gap),
            str(callers),
            str(slot),
            kind,
            str(fraction),
            text=True,
            timeout=300,
        )
        return done

    with ThreadPoolExecutor(4) as pool:
        finished = list(pool.map(one, range(len(runs))))
    landed = {kind: [] for kind in kinds}
    for slot, done in enumerate(finished):
        assert done.returncode == 0, f"gap and callers {runs[slot]}:\n{done.stderr[-3000:]}"
        lines = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
        report = next(line for line in lines if "disruption" in line)
        children = [line for line in lines if "child" in line]
        if report["disruption"] == "fork":
            (child,) = children
            assert child["child"] in ("same", "raised"), "the child's query returned an answer missing items"
            assert child["again"], "the child's pool answered otherwise afterwards"
            landed["fork"].append(child["child"] == "raised")
        elif report["disruption"] == "exit":
            # the daemon's query stopped, not finished: nothing answered once
            # the exit began
            after = done.stdout.split("exiting\n", 1)
            assert len(after) == 2 and "answered" not in after[1], "the exit waited on the daemon's query"
            landed["exit"].append(True)
        else:
            landed["ctrl-c"].append(report["landed"])
    # at least one of each landed inside a call, or nothing here tested it
    several = cpus_allowed() > 1
    assert any(landed["ctrl-c"]) and landed["exit"], landed
    assert any(landed["fork"]) or not several, f"no fork landed inside a published job: {landed['fork']}"


@pytest.mark.timing
@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork")
def test_forked_child_exits_promptly():
    # a parent thread is inside a query when the main thread forks: the
    # child has none of the parent's calls, so its exit waits for none of
    # them (it once waited out the whole drain limit for a call that
    # existed only in the parent)
    import textwrap

    script = textwrap.dedent(
        """
        import os, sys, threading, time
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        from helpers import long_call

        # a second or more here, however many cores, so the forks below land
        # inside the call that follows the first one
        call, _ = long_call()
        busy = threading.Event()
        stop = threading.Event()

        def query():
            call()
            while not stop.is_set():
                busy.set()
                call()

        thread = threading.Thread(target=query)
        thread.start()
        busy.wait()
        time.sleep(0.05)
        slowest = 0.0
        for _ in range(3):
            tic = time.monotonic()
            pid = os.fork()
            if pid == 0:
                sys.exit(0)
            _, status = os.waitpid(pid, 0)
            slowest = max(slowest, time.monotonic() - tic)
            assert status == 0, status
        stop.set()
        thread.join()
        print(f"{slowest:.3f}")
        """
    )
    done = run_python(
        "-W",
        "ignore",
        "-c",
        script,
        str(Path(__file__).parent),
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert float(done.stdout.split()[-1]) < 2.0, f"a forked child took {done.stdout.split()[-1]} s to exit"


@pytest.mark.skipif(not hasattr(__import__("os"), "sched_setaffinity"), reason="needs CPU affinity")
def test_workers_follow_affinity_as_it_widens():
    # the first call runs on one CPU, so it has no workers; once the process
    # may use more CPUs, the next big call after the pool reads them again
    # starts workers for them (the worker count was once fixed at whatever the
    # first call saw)
    import textwrap

    if cpus_allowed() < 2:
        pytest.skip("needs two CPUs")
    script = textwrap.dedent(
        """
        import os, sys, time
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        from helpers import sphere
        import trimkit

        every = os.sched_getaffinity(0)
        # one CPU before anything uses the pool, so no worker has ever started
        os.sched_setaffinity(0, {min(every)})
        scene = trimkit.RayScene([sphere(60, 120)])
        origins, directions = np.random.default_rng(0).normal(size=(2, 400_000, 3))
        scene.intersect(origins, directions)
        alone = len(os.listdir("/proc/self/task"))
        os.sched_setaffinity(0, every)
        # the CPUs allowed are read at most every 100 ms (CAP_REFRESH_NS)
        time.sleep(0.15)
        scene.intersect(origins, directions)
        print(alone, len(os.listdir("/proc/self/task")))
        """
    )
    done = run_python(
        "-W",
        "ignore",
        "-c",
        script,
        str(Path(__file__).parent),
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    alone, widened = map(int, done.stdout.split())
    assert widened > alone, f"{widened} threads after widening, {alone} on one CPU"
