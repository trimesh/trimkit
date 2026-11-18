--------------------------- MODULE SchedulerTrace ---------------------------
(***************************************************************************)
(* Does src/parallel.hpp do what Scheduler.tla says? A traced build of the *)
(* header (TRIMKIT_TRACE, run by check/verify/tla/trace.cpp on the real pool)    *)
(* logs every step of the protocol: its thread, the model action it is,    *)
(* what it took, and every slot word, `left`, and the wake, spinning and   *)
(* live counts just after it, all under one lock, so the log is one order  *)
(* the steps really took. TraceLog.tla is that log.                        *)
(*                                                                         *)
(* This spec walks the model along the log: each record must be the next  *)
(* step the model allows its thread (the action for that thread's step),   *)
(* and leave the model's shared state exactly as the C++ left it. Between  *)
(* records a thread may only run its items (local, and not logged), and a *)
(* sleep may end by its timeout. TLC checks every way the model's free     *)
(* choices could have been made; the log is accepted when some way reaches *)
(* its end, which TLC reports as Unmatched broken. A log no way follows    *)
(* means the code took a step the model doesn't allow: the last record     *)
(* matched is printed, and the next one is where the two part.             *)
(***************************************************************************)
EXTENDS Scheduler, TraceLog, TLC

VARIABLE l  \* the next record of the log to match
traceVars == <<vars, l>>

\* the model action a thread's next step is, by its frame
Action(f) == (IF f.kind = "caller" THEN "Caller" ELSE "Worker") \o (IF f.pc = "Return" THEN "CloseReturn" ELSE f.pc)

\* the shared state after a step, as the C++ logged it
SameState(r) ==
    /\ \A s \in SlotIds : /\ slot'[s].epoch = r.word[s][1]
                          /\ slot'[s].open = r.word[s][2]
                          /\ slot'[s].active = r.word[s][3]
                          /\ slot'[s].left = r.left[s]
    /\ wake' = r.wake
    /\ spinning' = r.spinning
    /\ live' = r.live

\* what the step took, where the C++ logged it: the range a claim took, the
\* helpers a decision wants, the slot a publish reserved
After(t) == stack'[t][Len(stack'[t])]
SameChoice(t, r) ==
    /\ r.action \in {"CallerClaim", "WorkerClaim"} /\ r.arg[1] # 0 =>
           After(t).pc = "Run" /\ After(t).b = r.arg[1] /\ After(t).e = r.arg[2]
    /\ r.action \in {"CallerClaim", "WorkerClaim"} /\ r.arg[1] = 0 => After(t).pc # "Run"
    /\ r.action = "CallerDecide" =>
           IF r.arg[1] > Top(t).helpers THEN After(t).want = r.arg[1] ELSE After(t).pc = "Claim"
    /\ r.action = "CallerPublish" /\ r.arg[1] # 0 => After(t).k = r.arg[1]
    /\ r.action = "CallerPublish" /\ r.arg[1] = 0 => After(t).pc = "Claim"
    \* a wake wakes as many sleepers as the system said it woke
    /\ r.action = "CallerWakeSome" =>
           Cardinality({w \in Workers : stack[w] # <<>> /\ Top(w).pc = "ParkSleep" /\ stack'[w] # <<>>
                                        /\ stack'[w][Len(stack'[w])].pc = "ParkCompare"}) = r.arg[1]

\* a thread's own steps between records: its items, or a sleep's timeout
Local(t) ==
    /\ stack[t] # <<>>
    /\ \/ Top(t).pc \in {"Run", "RunNested"} /\ Step(t)
       \/ ParkTimeout(t)
       \/ CloseTimeout(t)

Matched(r) ==
    LET t == r.thread IN
    /\ stack[t] # <<>>
    /\ \/ Action(Top(t)) = r.action /\ Step(t)
       \/ r.action = "WorkerExit" /\ Idle(t)
    /\ SameState(r)
    /\ (stack'[t] # <<>> => SameChoice(t, r))

TraceInit == Init /\ l = 1 /\ TLCSet(1, 0)

TraceNext ==
    /\ l <= Len(Log)
    /\ \/ Local(Log[l].thread) /\ UNCHANGED l
       \/ /\ Matched(Log[l])
          /\ l' = l + 1
          /\ IF l > TLCGet(1) THEN TLCSet(1, l) /\ PrintT(<<"matched", l>>) ELSE TRUE

TraceSpec == TraceInit /\ [][TraceNext]_traceVars

\* broken exactly when the whole log has been matched: TLC's way of saying yes
Unmatched == l <= Len(Log)
=============================================================================
