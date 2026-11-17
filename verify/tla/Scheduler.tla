---------------------------- MODULE Scheduler ----------------------------
(***************************************************************************)
(* The pool in src/parallel.hpp as a state machine, which TLC checks in    *)
(* every interleaving (`uv run scripts model`). Each action is one atomic  *)
(* step of the C++, named for the case of run_job's or worker_main's       *)
(* switch it belongs to (CallerStep, WorkerStep): `uv run scripts lint`    *)
(* fails if a step is in one and not the other. The slot word's arithmetic *)
(* is the C++'s own, generated from SlotWord into SlotWordTable.tla.       *)
(*                                                                         *)
(* What the C++ decides by measuring (how many helpers, how big a claim,   *)
(* whether to publish early) is a free choice here, so every promise is    *)
(* checked for every policy. Sleeps have no timeouts but the idle exit, so *)
(* a lost wake is a hang here, where the C++ would stall for a timeout.    *)
(*                                                                         *)
(* Checked in every state:                                                 *)
(*   S1  no thread touches a job after its call returned (it is on the     *)
(*       caller's stack)                                                   *)
(*   S2  no item runs twice                                                *)
(*   S3  no two workers hold one thread index                              *)
(*   P1  a worker counted as spinning by a waking caller never sleeps      *)
(*       through that caller's wake (the wake and spinning counts, both    *)
(*       sequentially consistent, pair)                                    *)
(*   P2  a job published, wanting help, with items left, has a live worker *)
(*       or one leaving that will start its replacement (the wake and live *)
(*       counts pair); but in a forked child, whose caller runs alone      *)
(*   R   a call that returned ran every item, or failed, or was stopped    *)
(*       by its caller's poll                                              *)
(*   X   no range is claimed once the interpreter's exit closed the gate   *)
(*       (here a claim is one step; in the C++ its look at the gate comes  *)
(*       just before its fetch-add, so one that read the gate open may     *)
(*       still take its range, and the call then fails by R's StoppedAtExit*)
(*       check rather than by stopping first)                              *)
(*   E   an epoch names one job, so a worker skips only a job it drained   *)
(* A step outside a SlotWord transition's precondition (which the C++'s    *)
(* checked() asserts) is outside its table's domain, and stops TLC too.    *)
(* and over every behaviour, each thread taking its steps fairly:          *)
(*   L1  every call returns                                                *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences, SlotWordTable

CONSTANTS
    Callers,     \* threads calling the pool
    Calls,       \* calls each makes, one after another
    Cap,         \* the most workers (worker_cap)
    NSlots,      \* slots jobs are shared from (SLOTS)
    Items,       \* items in each call a caller makes
    ThrowAt,     \* the item that fails, in every call, or 0 for none
    NestAt,      \* the item that calls the pool itself (two items), or 0
    Interrupt,   \* Ctrl-C arrives, once
    SpawnFails,  \* starting a thread may fail
    IdleExit,    \* idle workers may exit
    Fork,        \* the process may fork, twice, from any thread in a body or a caller's poll
    ThreadEnds,  \* a worker's thread may end inside a body (pthread_exit, a forced unwind)
    AtExit,      \* the interpreter may exit, once
    Mutation     \* one guard of the C++ broken, or "none"

Workers == Callers + 1 .. Callers + Cap + 1  \* one more than the cap: a leaving worker holds its own
Threads == 1 .. Callers + Cap + 1
SlotIds == 1 .. NSlots
MaxJobs == Callers * Calls * (IF NestAt > 0 THEN 2 ELSE 1)

Broken(m) == Mutation = m

VARIABLES
    slot,       \* each slot: its word (epoch, open, active) and job, and `left`
    job,        \* each call's items and how it ended
    jobs,       \* jobs made so far
    stack,      \* each thread's frames, its innermost call last
    calls,      \* calls each caller thread has still to make
    wake,       \* bumped to tell idle workers to look
    spinning,   \* workers counted as spinning
    live,       \* workers counted live
    wanted,     \* how far the spawn chain goes
    indexes,    \* thread indexes held, and by which thread
    promised,   \* per worker, the wake a caller counted it to see while it spun (0: none)
    gate,       \* the exit gate has closed
    interrupt,  \* Ctrl-C pending
    forked      \* how many times the process has forked: after one, this is a child

vars == <<slot, job, jobs, stack, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

NoDrain == [s \in SlotIds |-> 0]

CallerFrame(j, pc) ==
    [kind |-> "caller", pc |-> pc, j |-> j, k |-> 0, b |-> 0, e |-> 0, helpers |-> 0, want |-> 0,
     epoch |-> 0, seen |-> 0, observed |-> 0, joined |-> FALSE, drained |-> NoDrain, drained_job |-> NoDrain,
     closed |-> FALSE,
     spun |-> 0, bumped |-> 0, top |-> TRUE, ending |-> FALSE, held |-> 0]
WorkerFrame ==
    [kind |-> "worker", pc |-> "Start", j |-> 0, k |-> 1, b |-> 0, e |-> 0, helpers |-> 0, want |-> 0,
     epoch |-> 0, seen |-> 0, observed |-> 0, joined |-> FALSE, drained |-> NoDrain, drained_job |-> NoDrain,
     closed |-> FALSE,
     spun |-> 0, bumped |-> 0, top |-> FALSE, ending |-> FALSE, held |-> 0]

NewJob(count, nest, throw) ==
    [count |-> count, next |-> 1, stopped |-> FALSE, done |-> {}, twice |-> FALSE, returned |-> FALSE,
     failed |-> FALSE, interrupted |-> FALSE, forked |-> FALSE, touched_after |-> FALSE,
     claimed_after_exit |-> FALSE, nest |-> nest,
     throw |-> throw]

\* a call's first frame: claiming, or publishing at once (Coarse, wanting any
\* number of helpers, or with workers spinning, wanting none)
FirstFrames(j) ==
    {CallerFrame(j, "Claim")} \cup
    (IF Cap > 0 THEN {[CallerFrame(j, "Publish") EXCEPT !.want = w] : w \in 0 .. Cap} ELSE {})

Init ==
    /\ slot = [s \in SlotIds |-> [epoch |-> 0, open |-> FALSE, active |-> 0, job |-> 0, left |-> 0]]
    /\ jobs = Callers
    /\ job = [j \in 1 .. MaxJobs |-> NewJob(Items, NestAt, ThrowAt)]
    /\ \E first \in [1 .. Callers -> UNION {FirstFrames(c) : c \in 1 .. Callers}]:
           /\ \A c \in 1 .. Callers : first[c].j = c
           /\ stack = [t \in Threads |-> IF t <= Callers THEN <<first[t]>> ELSE <<>>]
    /\ calls = [t \in Threads |-> IF t <= Callers THEN Calls - 1 ELSE 0]
    /\ wake = 0 /\ spinning = 0 /\ live = 0 /\ wanted = 0
    /\ indexes = [w \in Workers |-> 0]
    /\ promised = [w \in Workers |-> 0]
    /\ gate = FALSE /\ interrupt = Interrupt /\ forked = 0

----------------------------------------------------------------------------
\* small updates

Top(t) == stack[t][Len(stack[t])]
SetTop(t, f) == [stack EXCEPT ![t] = [@ EXCEPT ![Len(@)] = f]]
Go(t, f) == stack' = SetTop(t, f)
Push(st, t, f) == [st EXCEPT ![t] = Append(@, f)]
Pop(st, t) == [st EXCEPT ![t] = SubSeq(@, 1, Len(@) - 1)]

Word(s) == [epoch |-> slot[s].epoch, open |-> slot[s].open, active |-> slot[s].active]
SetWord(sl, s, w) == [sl EXCEPT ![s].epoch = w.epoch, ![s].open = w.open, ![s].active = w.active]

\* a thread touching job j: once its call has returned, S1 is broken
Touch(j) == IF job[j].returned THEN [job EXCEPT ![j].touched_after = TRUE] ELSE job

\* one more worker, unless as many as the cap are live; starting it may fail
SpawnWorker(st, lv) ==
    LET idle == {w \in Workers : st[w] = <<>> /\ indexes[w] = 0}
    IN IF lv >= Cap \/ idle = {} THEN {<<st, lv>>}
       ELSE LET w == CHOOSE x \in idle : \A y \in idle : x <= y
                started == <<Push(st, w, WorkerFrame), lv + 1>>
            IN IF SpawnFails THEN {started, <<st, lv>>} ELSE {started}

----------------------------------------------------------------------------
\* claim_range and run_range, shared by callers and workers

Claimable(f) == ~job[f.j].stopped /\ ~(gate /\ ~Broken("claims_ignore_exit")) /\ job[f.j].next <= job[f.j].count

\* the next range, of any size: the policy's claim_items, as a free choice.
\* One fetch-add (claim_load_then_store: a load, then a store)
Claim(t, f, none) ==
    IF Claimable(f)
    THEN \E take \in 1 .. (job[f.j].count - job[f.j].next + 1):
             IF Broken("claim_load_then_store")
             THEN /\ job' = Touch(f.j)
                  /\ Go(t, [f EXCEPT !.pc = "ClaimStore", !.b = job[f.j].next, !.e = job[f.j].next + take])
             ELSE /\ job' = [Touch(f.j) EXCEPT ![f.j].next = job[f.j].next + take, ![f.j].claimed_after_exit = @ \/ gate]
                  /\ Go(t, [f EXCEPT !.pc = "Run", !.b = job[f.j].next, !.e = job[f.j].next + take])
    ELSE /\ job' = Touch(f.j)
         /\ Go(t, [f EXCEPT !.pc = none])

ClaimStore(t, f) ==
    /\ job' = [Touch(f.j) EXCEPT ![f.j].next = f.e]
    /\ Go(t, [f EXCEPT !.pc = "Run"])

\* one item of the range [b, e): a failure stops the job, the nested item calls the pool
RunItem(t, f, after) ==
    LET j == f.j
        i == f.b
        ran == [Touch(j) EXCEPT ![j].done = @ \cup {i}, ![j].twice = @ \/ i \in job[j].done,
                                ![j].stopped = @ \/ i = job[j].throw, ![j].failed = @ \/ i = job[j].throw]
        next == IF i + 1 = f.e THEN [f EXCEPT !.pc = after, !.b = i + 1] ELSE [f EXCEPT !.b = i + 1]
    IN IF i = job[j].nest /\ jobs < MaxJobs
       THEN /\ job' = [Touch(j) EXCEPT ![jobs + 1] = NewJob(2, 0, 0)]
            /\ jobs' = jobs + 1
            /\ \E first \in FirstFrames(jobs + 1):
                   stack' = Push(SetTop(t, [f EXCEPT !.pc = "RunNested"]), t, [first EXCEPT !.top = FALSE])
       ELSE /\ job' = ran
            /\ UNCHANGED jobs
            /\ Go(t, next)

----------------------------------------------------------------------------
\* the caller: run_job's CallerStep

\* none left: a call that never published has nothing to close
CallerClaim(t, f) == Claim(t, f, IF f.k = 0 THEN "Return" ELSE "Close") /\ UNCHANGED <<slot, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>
CallerClaimStore(t, f) == ClaimStore(t, f) /\ UNCHANGED <<slot, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

CallerRun(t, f) ==
    /\ RunItem(t, f, "Decide")
    /\ UNCHANGED <<slot, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* the nested call returned: the item it ran in is done
CallerRunNested(t, f) ==
    LET next == IF f.b + 1 = f.e THEN [f EXCEPT !.pc = "Decide", !.b = f.b + 1] ELSE [f EXCEPT !.pc = "Run", !.b = f.b + 1]
    IN /\ job' = [Touch(f.j) EXCEPT ![f.j].done = @ \cup {f.b}, ![f.j].twice = @ \/ f.b \in job[f.j].done]
       /\ Go(t, next)
       /\ UNCHANGED <<slot, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* wake_helpers' first step: the bump (wake_counts_live_first: the live count;
\* wake_reads_spinning_first: the spinning count)
FirstWakeStep == IF Broken("wake_counts_live_first") THEN "WakeSpawn"
                 ELSE IF Broken("wake_reads_spinning_first") THEN "WakeSpinning" ELSE "Wake"

\* poll_stop, then helpers_wanted, as a free choice: any number up to the cap
CallerDecide(t, f) ==
    LET polled == f.top /\ interrupt
    IN /\ job' = IF polled THEN [job EXCEPT ![f.j].stopped = TRUE, ![f.j].interrupted = TRUE] ELSE job
       /\ interrupt' = (interrupt /\ ~polled)
       /\ \E want \in 0 .. Cap:
              Go(t, IF want > f.helpers
                    THEN [f EXCEPT !.pc = IF f.k # 0 THEN FirstWakeStep ELSE "Publish", !.want = want]
                    ELSE [f EXCEPT !.pc = "Claim"])
       /\ UNCHANGED <<slot, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, forked>>

\* try_publish: reserve a free slot under the next epoch, the caller counted
CallerPublish(t, f) ==
    LET free == {s \in SlotIds : IsFree(Word(s))}
    IN /\ IF free = {}
          THEN Go(t, [f EXCEPT !.pc = "Claim"])
          ELSE \E s \in free:
                   \* publish_keeps_epoch: the epoch moves on to 1, then no further
                   LET w == IF Broken("publish_keeps_epoch") /\ Word(s).epoch > 0
                            THEN [Reserved(Word(s)) EXCEPT !.epoch = Word(s).epoch]
                            ELSE Reserved(Word(s))
                   IN /\ slot' = SetWord(slot, s, w)
                      /\ Go(t, [f EXCEPT !.pc = "PublishStore", !.k = s, !.epoch = w.epoch, !.held = w.epoch])
       /\ (free = {} => UNCHANGED slot)
       /\ UNCHANGED <<job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

CallerPublishStore(t, f) ==
    /\ slot' = [slot EXCEPT ![f.k].job = f.j]
    /\ Go(t, [f EXCEPT !.pc = "PublishOpen"])
    /\ UNCHANGED <<job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

CallerPublishOpen(t, f) ==
    /\ slot' = SetWord(slot, f.k, Opened(Word(f.k)))
    /\ Go(t, [f EXCEPT !.pc = FirstWakeStep])
    /\ UNCHANGED <<job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* wake_helpers: bump the wake, read the spinning count, start a worker if too
\* few are live, then wake as many sleepers as the spinning ones won't cover
CallerWake(t, f) ==
    /\ wake' = wake + 1
    /\ Go(t, [f EXCEPT !.bumped = wake + 1, !.pc = IF Broken("wake_reads_spinning_first") THEN "WakeSpawn"
                              ELSE IF Broken("wake_counts_live_first") THEN "WakeSome" ELSE "WakeSpinning"])
    /\ UNCHANGED <<slot, job, jobs, calls, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* every worker counted as spinning now is counted on to see this caller's
\* bump: the one it made, or (wake_reads_spinning_first) the one it will
Spinning(w) == stack[w] # <<>> /\ Top(w).pc \in {"ParkSpin", "ParkSpinSaw", "ParkSpinDone"}
CallerWakeSpinning(t, f) ==
    /\ promised' = [w \in Workers |-> IF Spinning(w) THEN (IF f.bumped # 0 THEN f.bumped ELSE wake + 1)
                                       ELSE promised[w]]
    /\ Go(t, [f EXCEPT !.pc = IF Broken("wake_reads_spinning_first") THEN "Wake"
                              ELSE IF Broken("wake_counts_live_first") THEN "Wake" ELSE "WakeSpawn", !.spun = spinning])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, gate, interrupt, forked>>

CallerWakeSpawn(t, f) ==
    LET after == IF Broken("wake_counts_live_first") THEN "Wake" ELSE "WakeSome"
        g == [f EXCEPT !.pc = after]
    IN /\ IF live < f.want
          THEN /\ wanted' = f.want
               /\ \E r \in SpawnWorker(SetTop(t, g), live): stack' = r[1] /\ live' = r[2]
          ELSE /\ Go(t, g) /\ UNCHANGED <<wanted, live>>
       /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, indexes, promised, gate, interrupt, forked>>

CallerWakeSome(t, f) ==
    LET n == (f.want - f.helpers) - f.spun
        asleep == {w \in Workers : stack[w] # <<>> /\ Top(w).pc = "ParkSleep"}
        woken(st, ws) == [w \in Threads |-> IF w \in ws THEN [st[w] EXCEPT ![Len(st[w])].pc = "ParkCompare"] ELSE st[w]]
    IN /\ \E ws \in SUBSET asleep:
              /\ Cardinality(ws) <= IF n > 0 THEN n ELSE 0  \* at most n: the kernel wakes what is asleep
              /\ stack' = woken(SetTop(t, [f EXCEPT !.pc = "Claim", !.helpers = f.want, !.bumped = 0]), ws)
       /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* close_job: no more joins, and the caller out; wait if workers are inside
CallerClose(t, f) ==
    LET before == Word(f.k)
        waits == before.active # 1 /\ ~Broken("close_without_waiting")
    IN /\ slot' = SetWord(slot, f.k, Closed(before))
       /\ Go(t, [f EXCEPT !.pc = IF waits THEN "CloseCheck" ELSE "Return", !.epoch = before.epoch,
                          !.closed = TRUE])
    /\ UNCHANGED <<job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

WorkersGone(f) == slot[f.k].active = 0 \/ (slot[f.k].epoch # f.epoch /\ ~Broken("wait_ignores_epoch"))

\* wait_for_workers: the loop's check, `left` read, the check again, then a
\* sleep that compares `left` first
CallerCloseCheck(t, f) ==
    /\ Go(t, [f EXCEPT !.pc = IF WorkersGone(f) THEN "Return" ELSE "CloseLoadLeft"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

CallerCloseLoadLeft(t, f) ==
    /\ Go(t, [f EXCEPT !.pc = IF Broken("wait_checks_count_first") THEN "ClosePark" ELSE "CloseRecheck",
                       !.seen = slot[f.k].left])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

CallerCloseRecheck(t, f) ==
    /\ Go(t, [f EXCEPT !.pc = IF WorkersGone(f) THEN "Return" ELSE "ClosePark"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

CallerClosePark(t, f) ==
    /\ Go(t, [f EXCEPT !.pc = IF slot[f.k].left # f.seen THEN "CloseCheck" ELSE "CloseSleep"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* the call returns what it ran into: the fork (T5), or the exit before every
\* item was claimed; then its caller thread makes its next call
CallerCloseReturn(t, f) ==
    LET j == f.j
        exited == gate /\ job[j].next <= job[j].count
        ended == [job EXCEPT ![j].returned = TRUE, ![j].failed = @ \/ job[j].forked \/ exited]
        popped == Pop(stack, t)
        again == popped[t] = <<>> /\ t <= Callers /\ calls[t] > 0
    IN /\ IF again
          THEN /\ job' = [ended EXCEPT ![jobs + 1] = NewJob(Items, NestAt, ThrowAt)]
               /\ jobs' = jobs + 1
               /\ calls' = [calls EXCEPT ![t] = @ - 1]
               /\ \E first \in FirstFrames(jobs + 1): stack' = Push(popped, t, first)
          ELSE /\ job' = ended /\ stack' = popped /\ UNCHANGED <<jobs, calls>>
       /\ UNCHANGED <<slot, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

----------------------------------------------------------------------------
\* the worker: worker_main's WorkerStep

\* take the lowest free thread index, then the next of the spawn chain
WorkerStart(t, f) ==
    LET held == {indexes[w] : w \in Workers} \ {0}
        free == {i \in 1 .. Cap : i \notin held}
        index == CHOOSE i \in free : \A x \in free : i <= x
        g == [f EXCEPT !.pc = "Scan"]
    IN /\ free # {}
       /\ indexes' = [indexes EXCEPT ![t] = index]
       /\ IF live < wanted
          THEN \E r \in SpawnWorker(SetTop(t, g), live): stack' = r[1] /\ live' = r[2]
          ELSE Go(t, g) /\ UNCHANGED live
       /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, wanted, promised, gate, interrupt, forked>>

\* the wake read before looking, then each slot's word
WorkerScan(t, f) ==
    /\ Go(t, [f EXCEPT !.pc = "ScanSlot", !.k = 1, !.observed = wake, !.joined = FALSE])
    /\ promised' = [promised EXCEPT ![t] = 0]
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, gate, interrupt, forked>>

WorkerScanSlot(t, f) ==
    /\ IF f.k > NSlots
       THEN Go(t, [f EXCEPT !.pc = IF f.joined THEN "Scan" ELSE "Park"])
       ELSE IF slot[f.k].open /\ slot[f.k].epoch # f.drained[f.k]
            THEN Go(t, [f EXCEPT !.pc = "Join", !.epoch = slot[f.k].epoch])
            ELSE Go(t, [f EXCEPT !.k = f.k + 1])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* join_job: one more inside the job published as the epoch seen, if still open
WorkerJoin(t, f) ==
    /\ IF slot[f.k].epoch = f.epoch /\ (slot[f.k].open \/ Broken("join_ignores_open"))
       THEN /\ slot' = SetWord(slot, f.k, Joined(Word(f.k)))
            /\ Go(t, [f EXCEPT !.pc = "JoinLoad"])
       ELSE /\ Go(t, [f EXCEPT !.pc = "ScanSlot", !.k = f.k + 1]) /\ UNCHANGED slot
    /\ UNCHANGED <<job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* counted, so the slot holds this epoch's job
WorkerJoinLoad(t, f) ==
    /\ Go(t, [f EXCEPT !.pc = "Claim", !.j = slot[f.k].job, !.joined = TRUE])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

WorkerClaim(t, f) == Claim(t, f, "Leave") /\ UNCHANGED <<slot, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>
WorkerClaimStore(t, f) == ClaimStore(t, f) /\ UNCHANGED <<slot, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

WorkerRun(t, f) ==
    /\ RunItem(t, f, "Claim")
    /\ UNCHANGED <<slot, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

WorkerRunNested(t, f) ==
    LET next == IF f.b + 1 = f.e THEN [f EXCEPT !.pc = "Claim", !.b = f.b + 1] ELSE [f EXCEPT !.pc = "Run", !.b = f.b + 1]
    IN /\ job' = [Touch(f.j) EXCEPT ![f.j].done = @ \cup {f.b}, ![f.j].twice = @ \/ f.b \in job[f.j].done]
       /\ Go(t, next)
       /\ UNCHANGED <<slot, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* leave_job: never touching the job again; the last out of a closed job
\* bumps `left` and wakes its caller
WorkerLeave(t, f) ==
    LET before == Word(f.k)
        last == ~before.open /\ before.active = 1
    IN /\ slot' = SetWord(slot, f.k, Left(before))
       /\ Go(t, [f EXCEPT !.pc = IF last THEN "LeaveBump" ELSE IF f.ending THEN "ExitEnding" ELSE "ScanSlot",
                          !.k = IF last \/ f.ending THEN f.k ELSE f.k + 1, !.j = 0, !.drained[f.k] = f.epoch,
                          !.drained_job[f.k] = f.j])
       /\ UNCHANGED <<job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

LeaveWakes(st, k) ==
    [c \in Threads |-> IF st[c] # <<>> /\ st[c][Len(st[c])].pc = "CloseSleep" /\ st[c][Len(st[c])].k = k
                       THEN [st[c] EXCEPT ![Len(st[c])].pc = "CloseCheck"] ELSE st[c]]

WorkerLeaveBump(t, f) ==
    /\ slot' = [slot EXCEPT ![f.k].left = @ + 1]
    /\ stack' = LeaveWakes(SetTop(t, IF f.ending THEN [f EXCEPT !.pc = "ExitEnding"]
                                       ELSE [f EXCEPT !.pc = "ScanSlot", !.k = f.k + 1]), f.k)
    /\ UNCHANGED <<job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* park_worker: counted spinning, it looks at the wake while it spins (or
\* not at all, out of budget), then out of the count, then a sleep that
\* compares the wake first
WorkerPark(t, f) ==
    /\ spinning' = spinning + 1
    /\ Go(t, [f EXCEPT !.pc = "ParkSpin"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, live, wanted, indexes, promised, gate, interrupt, forked>>

WorkerParkSpin(t, f) ==
    /\ \/ /\ wake # f.observed
          /\ Go(t, [f EXCEPT !.pc = "ParkSpinSaw"])
       \/ Go(t, [f EXCEPT !.pc = "ParkSpinDone"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

WorkerParkSpinSaw(t, f) ==
    /\ spinning' = spinning - 1
    /\ Go(t, [f EXCEPT !.pc = "Scan"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, live, wanted, indexes, promised, gate, interrupt, forked>>

WorkerParkSpinDone(t, f) ==
    /\ spinning' = spinning - 1
    /\ Go(t, [f EXCEPT !.pc = "ParkCompare"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, live, wanted, indexes, promised, gate, interrupt, forked>>

WorkerParkCompare(t, f) ==
    /\ \/ /\ wake = f.observed \/ Broken("park_ignores_wake")
          /\ Go(t, [f EXCEPT !.pc = "ParkSleep"])
       \/ /\ wake # f.observed
          /\ Go(t, [f EXCEPT !.pc = "Scan"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

\* idle long enough: its index given back, then out of the live count, then
\* (Holding's destructor) a last look at the wake, which a caller may have
\* bumped after counting it live, so it starts its replacement
WorkerExit(t, f) ==
    /\ indexes' = [indexes EXCEPT ![t] = 0]
    /\ live' = live - 1
    /\ Go(t, [f EXCEPT !.pc = "ExitRecheck"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, wanted, promised, gate, interrupt, forked>>

\* the same, for a thread ending inside a body, once it has left its job
WorkerExitEnding(t, f) == WorkerExit(t, f)

\* a worker's thread ending inside a body: the job fails, as the items it held
\* never run, and Holding's destructor leaves the job (leave_job), then exits
ThreadEndsInBody(t) ==
    LET f == Top(t) IN
    /\ ThreadEnds /\ Len(stack[t]) = 1 /\ f.kind = "worker" /\ f.pc = "Run"
    /\ job' = [Touch(f.j) EXCEPT ![f.j].stopped = TRUE, ![f.j].failed = TRUE]
    /\ Go(t, [f EXCEPT !.pc = IF Broken("end_without_leaving") THEN "ExitEnding" ELSE "Leave", !.ending = TRUE])
    /\ UNCHANGED <<slot, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

WorkerExitRecheck(t, f) ==
    /\ IF wake # f.observed /\ ~Broken("exit_without_recheck")
       THEN \E r \in SpawnWorker(Pop(stack, t), live): stack' = r[1] /\ live' = r[2]
       ELSE stack' = Pop(stack, t) /\ UNCHANGED live
    /\ promised' = [promised EXCEPT ![t] = 0]
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, wanted, indexes, gate, interrupt, forked>>

\* an idle worker may exit once asleep
MayExit(f) == IdleExit /\ f.pc = "ParkSleep"

----------------------------------------------------------------------------
\* the environment: fork, the interpreter's exit

\* child_after_fork, in the child of a fork by thread t from inside a body:
\* only t is left; every slot is freed under a new epoch, but one t published
\* and hasn't closed is opened again with only t counted, and one t is inside
\* as a worker is closed with only t counted; a job a vanished worker was
\* inside stops, and fails
ChildAfterFork(t) ==
    LET mine == {i \in 1 .. Len(stack[t]) : stack[t][i].kind = "caller" /\ stack[t][i].k # 0}
        \* workers of this very call inside: its caller is counted until it closes,
        \* and a closed slot may hold another caller's job since
        lost(i) == LET f == stack[t][i] IN
                   slot[f.k].epoch = f.held /\ slot[f.k].active > (IF f.closed THEN 0 ELSE 1)
                   /\ ~Broken("fork_ignores_workers")
        lostjobs == {stack[t][i].j : i \in {x \in mine : lost(x)}}
        worker == {i \in 1 .. Len(stack[t]) : stack[t][i].kind = "worker" /\ stack[t][i].j # 0}
        As(s, w, j) == [epoch |-> w.epoch, open |-> w.open, active |-> w.active, job |-> j, left |-> slot[s].left]
        reopened == [s \in SlotIds |->
                       IF \E i \in mine : stack[t][i].k = s /\ ~stack[t][i].closed
                       THEN LET i == CHOOSE x \in mine : stack[t][x].k = s /\ ~stack[t][x].closed
                            IN As(s, ReopenedByFork(Word(s)), stack[t][i].j)
                       ELSE IF \E i \in worker : stack[t][i].k = s
                            THEN As(s, KeptByFork(Word(s)), 0)
                            ELSE As(s, FreedByFork(Word(s)), 0)]
        stopped == {stack[t][i].j : i \in worker}
    IN /\ slot' = IF Broken("fork_keeps_slots") THEN slot ELSE reopened
       /\ job' = [j \in 1 .. MaxJobs |->
                    IF j \in lostjobs THEN [job[j] EXCEPT !.forked = TRUE, !.stopped = TRUE]
                    ELSE IF j \in stopped THEN [job[j] EXCEPT !.stopped = TRUE] ELSE job[j]]
       \* a call it reopened is held under the slot's new epoch, which a second
       \* fork compares with (HeldSlot's epoch; fork_keeps_held_epoch: the one
       \* it published under). Its wait, if it closes, reads the epoch afresh
       /\ stack' = [x \in Threads |->
                     IF x # t THEN <<>>
                     ELSE [i \in 1 .. Len(stack[t]) |->
                             IF i \in mine /\ ~stack[t][i].closed /\ ~Broken("fork_keeps_held_epoch")
                             THEN [stack[t][i] EXCEPT !.held = ReopenedByFork(Word(stack[t][i].k)).epoch]
                             ELSE stack[t][i]]]
       /\ calls' = [x \in Threads |-> IF x = t THEN calls[t] ELSE 0]
       /\ live' = IF t \in Workers THEN 1 ELSE 0
       /\ indexes' = [w \in Workers |-> IF w = t THEN indexes[w] ELSE 0]
       /\ promised' = [w \in Workers |-> 0]
       /\ spinning' = 0 /\ wanted' = 0 /\ gate' = FALSE /\ forked' = forked + 1
       /\ UNCHANGED <<jobs, wake, interrupt>>

\* inside a body (Run), or in a caller's poll: between ranges (Decide), or
\* while it waits for its workers (CloseCheck, after each sleep)
Forking(t) == Fork /\ forked < 2 /\ stack[t] # <<>> /\ Top(t).pc \in {"Run", "Decide", "CloseCheck"}

\* a sleep that ends with no wake: its timeout (POLL_MS, IDLE_EXIT_MS), or the
\* kernel's compare finding the word moved since the thread's own. Neither is
\* fair, so neither can stand in for a wake that never comes (L1)
ParkTimeout(t) ==
    /\ stack[t] # <<>> /\ Top(t).pc = "ParkSleep"
    /\ Go(t, [Top(t) EXCEPT !.pc = "ParkCompare"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>
CloseTimeout(t) ==
    /\ stack[t] # <<>> /\ Top(t).pc = "CloseSleep"
    /\ Go(t, [Top(t) EXCEPT !.pc = "CloseCheck"])
    /\ UNCHANGED <<slot, job, jobs, calls, wake, spinning, live, wanted, indexes, promised, gate, interrupt, forked>>

ExitGateClose == AtExit /\ ~gate /\ gate' = TRUE /\ UNCHANGED <<slot, job, jobs, stack, calls, wake, spinning, live, wanted, indexes, promised, interrupt, forked>>

----------------------------------------------------------------------------

Step(t) ==
    /\ stack[t] # <<>>
    /\ LET f == Top(t) IN
       IF f.kind = "caller" THEN
           CASE f.pc = "Claim" -> CallerClaim(t, f)
             [] f.pc = "ClaimStore" -> CallerClaimStore(t, f)
             [] f.pc = "Run" -> CallerRun(t, f)
             [] f.pc = "RunNested" -> CallerRunNested(t, f)
             [] f.pc = "Decide" -> CallerDecide(t, f)
             [] f.pc = "Publish" -> CallerPublish(t, f)
             [] f.pc = "PublishStore" -> CallerPublishStore(t, f)
             [] f.pc = "PublishOpen" -> CallerPublishOpen(t, f)
             [] f.pc = "Wake" -> CallerWake(t, f)
             [] f.pc = "WakeSpinning" -> CallerWakeSpinning(t, f)
             [] f.pc = "WakeSpawn" -> CallerWakeSpawn(t, f)
             [] f.pc = "WakeSome" -> CallerWakeSome(t, f)
             [] f.pc = "Close" -> CallerClose(t, f)
             [] f.pc = "CloseCheck" -> CallerCloseCheck(t, f)
             [] f.pc = "CloseLoadLeft" -> CallerCloseLoadLeft(t, f)
             [] f.pc = "CloseRecheck" -> CallerCloseRecheck(t, f)
             [] f.pc = "ClosePark" -> CallerClosePark(t, f)
             [] f.pc = "CloseSleep" -> FALSE  \* until the last worker out wakes it
             [] f.pc = "Return" -> CallerCloseReturn(t, f)
       ELSE
           CASE f.pc = "Start" -> WorkerStart(t, f)
             [] f.pc = "Scan" -> WorkerScan(t, f)
             [] f.pc = "ScanSlot" -> WorkerScanSlot(t, f)
             [] f.pc = "Join" -> WorkerJoin(t, f)
             [] f.pc = "JoinLoad" -> WorkerJoinLoad(t, f)
             [] f.pc = "Claim" -> WorkerClaim(t, f)
             [] f.pc = "ClaimStore" -> WorkerClaimStore(t, f)
             [] f.pc = "Run" -> WorkerRun(t, f)
             [] f.pc = "RunNested" -> WorkerRunNested(t, f)
             [] f.pc = "Leave" -> WorkerLeave(t, f)
             [] f.pc = "LeaveBump" -> WorkerLeaveBump(t, f)
             [] f.pc = "Park" -> WorkerPark(t, f)
             [] f.pc = "ParkSpin" -> WorkerParkSpin(t, f)
             [] f.pc = "ParkSpinSaw" -> WorkerParkSpinSaw(t, f)
             [] f.pc = "ParkSpinDone" -> WorkerParkSpinDone(t, f)
             [] f.pc = "ParkCompare" -> WorkerParkCompare(t, f)
             [] f.pc = "ParkSleep" -> FALSE  \* until a caller wakes it
             [] f.pc = "ExitEnding" -> WorkerExitEnding(t, f)
             [] f.pc = "ExitRecheck" -> WorkerExitRecheck(t, f)

Idle(t) == stack[t] # <<>> /\ MayExit(Top(t)) /\ WorkerExit(t, Top(t))

Next ==
    \/ \E t \in Threads : Step(t) \/ Idle(t)
    \/ \E t \in Threads : Forking(t) /\ ChildAfterFork(t)
    \/ \E t \in Workers : stack[t] # <<>> /\ ThreadEndsInBody(t)
    \/ \E t \in Threads : ParkTimeout(t) \/ CloseTimeout(t)
    \/ ExitGateClose

\* each thread's own steps are taken fairly; forks, exits and failures to
\* start a thread are the environment's, and need not happen
Spec == Init /\ [][Next]_vars /\ \A t \in Threads : WF_vars(Step(t))

----------------------------------------------------------------------------
\* the properties

S1 == \A j \in 1 .. MaxJobs : ~job[j].touched_after
S2 == \A j \in 1 .. MaxJobs : ~job[j].twice
S3 == /\ \A v, w \in Workers : v # w /\ indexes[v] # 0 => indexes[v] # indexes[w]
      /\ \A w \in Workers : stack[w] # <<>> /\ Top(w).pc = "Start" =>
             \E i \in 1 .. Cap : \A v \in Workers : indexes[v] # i

P1 == \A w \in Workers :
        stack[w] # <<>> /\ Top(w).pc = "ParkSleep" /\ promised[w] # 0 => Top(w).observed >= promised[w]

Waking == {"PublishStore", "PublishOpen", "Wake", "WakeSpinning", "WakeSpawn", "WakeSome"}
Leaving == \E w \in Workers : stack[w] # <<>> /\ Top(w).pc = "ExitRecheck"
P2 == SpawnFails \/ forked > 0 \/
      \A t \in Threads : \A i \in 1 .. Len(stack[t]) :
        LET f == stack[t][i] IN
        (f.kind = "caller" /\ f.k # 0 /\ ~f.closed /\ f.helpers > 0 /\ f.pc \notin Waking
         /\ job[f.j].next <= job[f.j].count /\ ~job[f.j].stopped)
        => live > 0 \/ Leaving

\* an epoch names one job: a worker skips an open slot whose epoch it found
\* drained only while that slot still holds the job it drained
E == \A w \in Workers : \A k \in SlotIds :
       stack[w] # <<>> /\ stack[w][1].drained[k] # 0 /\ slot[k].open /\ slot[k].epoch = stack[w][1].drained[k]
       => slot[k].job = stack[w][1].drained_job[k]

\* once the interpreter's exit closes the gate, no range is claimed
X == \A j \in 1 .. MaxJobs : ~job[j].claimed_after_exit

R == \A j \in 1 .. MaxJobs :
       job[j].returned => job[j].done = 1 .. job[j].count \/ job[j].failed \/ job[j].interrupted

Calling == \E t \in Threads : \E i \in 1 .. Len(stack[t]) : stack[t][i].kind = "caller"
L1 == <>(~Calling /\ \A t \in 1 .. Callers : calls[t] = 0)

=============================================================================
