// parallel.hpp: a parallel for loop, for a C++ extension.
//
// WHAT IT IS
//
// One call runs a function over every index from 0 to n - 1 (each an
// "item"), on every CPU core the process may use: the idea of Rayon's
// `par_iter`, OpenMP's `parallel for`, or ThreadPoolExecutor.map over chunks.
// The body is handed ranges [begin, end) until every item has had its turn,
// and writes each item's output in its own place, so the answer never
// depends on which thread ran which range. Everything in trimkit that uses
// more than one core runs through here (ray queries, collisions, contours,
// V-HACD, xatlas), on one pool of worker threads per process, and nothing
// else in the extension starts a thread (a lint checks).
//
// USING IT
//
// Bindings call common.hpp's wrappers, which also release the GIL and check
// for Ctrl-C:
//
//     trimkit::in_parallel(rays, [&](std::size_t begin, std::size_t end) {
//         for (std::size_t i = begin; i < end; i++) {
//             hits[i] = trace(origins[i], directions[i]);
//         }
//     });
//
// Underneath is the one entry point, for_each_range, which returns what
// failed, if anything did. It can be told ItemCost::Coarse when each item is
// worth waking a thread for (a hull to split), a `min_range` when items are
// too cheap to claim one at a time, and a Stop to poll for Ctrl-C.
//
// WHY NOT JUST START THREADS
//
// Starting a thread per core costs tens of microseconds a call (about 60 on 8
// cores), more than many calls take, and an even split leaves every thread
// waiting on the slowest when items cost unevenly, as real meshes do. This
// pool keeps its threads, runs a call alone until it has run a wake-up's
// worth of items, wakes as many helpers as the work left pays for, and hands
// out smaller ranges near the end (the policy, below the constants). Idle
// workers sleep, using no CPU, and exit after 10 seconds. `scripts bench`
// fails if it loses to the serial loop or the best of three pools written by
// hand (check/bench/parallel.cpp), or uses much more CPU, in any case it measures.
//
// WHAT CHECKS IT, on every `scripts check`, each failing the run on any break
//
// The hard part is a job on the caller's stack handed to other threads: no
// thread may touch it after the call returns, no item may run twice, nothing
// may deadlock and no wake-up may be lost, through exceptions, Ctrl-C, fork()
// and interpreter exit, on machines that reorder memory.
//   - check/verify/genmc: GenMC runs this header (platform.hpp's TRIMKIT_GENMC) in
//     every execution the C++ memory model allows, for one call of a few
//     items with its worker: T1 (a missing happens-before is a race on the
//     body's plain outputs), T2, T3, T4 for a failing body and a stop, a call
//     as the interpreter exits, the thread indexes, Exclusively, ChunkList and
//     the exit gate. Each harness explores at least a recorded number of
//     executions, and each memory order here weakened one step fails some
//     harness or is listed with why it can't. Two jobs at once are too many
//     executions (some 170 million): two callers, nesting and slot reuse are
//     the model's.
//   - check/verify/tla/Scheduler.tla: TLC checks the protocol in every
//     interleaving at small sizes, with every choice the policy makes free:
//     publish, join, leave, close and wait; the wake, spinning and live
//     counts; idle exit and its replacement; the thread indexes; up to two
//     forks, from a body or a poll; a thread ending in a body; the
//     interpreter's exit; and P3, that no call wakes or starts more workers
//     than it asked for but to replace one it lost, and that a worker roused
//     for a call sleeps again only once that call has nothing left to hand
//     out (or stopped, or returned). Each mutation breaks one
//     guard and must be found as the promise it breaks. Its steps are this
//     file's (a lint holds CallerStep, WorkerStep and every TraceStep to its
//     actions), and its slot arithmetic is SlotWord's, generated each run.
//   - check/verify/tla/SchedulerTrace.tla: a traced build (TRIMKIT_TRACE: each
//     TraceStep logs its action and the shared state after it) runs real
//     calls on the real pool, an idle exit included, and TLC must find the
//     model allows every step of every log, each wake waking as many as the
//     system said it did, and P3 holding along it. Three
//     reorderings of this file's steps must each be rejected.
//   - Checked builds (TRIMKIT_CHECKED: every sanitized build and fuzzer):
//     each slot transition asserts the state it found, and each call W1.
//   - check/fuzz/parallel.cpp, under ASan: any count, cost, ItemCost, min_range and
//     number of callers, with items that throw, nest and sleep; the policy's
//     rules over grids of inputs; and fixed checks for a poll that throws, a
//     thread ending in an item, exit, fork from a worker and a held slot, W1
//     on real calls, a min_range past the count, and too many items.
//   - check/tests/test_threads.py, under ThreadSanitizer too: fork inside a call, a
//     forked child's exit, exit during a call, the workers following
//     affinity.
//   - A lint over all the extension compiles: only this header and
//     platform.hpp start, wait on or sleep a thread, but common.hpp's poll.
// The checkers are exhaustive only at small sizes; at real sizes the fuzzer
// and sanitizers run what the machine schedules. GenMC's waits return at once
// (a wait may), so it checks every wait ends; TLC's sleeps have no timeouts,
// so it checks no wake is lost.
//
// WHAT A CALL PROMISES, however it ends (return, exception, Ctrl-C, fork, exit)
//
//   T1 Once for_each_range returns, or an exception unwinds out of it, no other
//      thread touches the job or the body again, and everything the body
//      wrote is visible to the caller: the job, the body and the outputs can
//      all live on the caller's stack.
//   T2 Each item runs at most once; if nothing stopped the call, exactly once.
//   T3 It always returns. The calling thread runs items too, so the call ends
//      even if no worker ever starts; it waits only for workers already
//      running one of its ranges, never for one to start or wake.
//   T4 If the body throws, the job stops handing out ranges, and
//      for_each_range returns one of the exceptions. A Stop that says stop
//      (Ctrl-C) stops it the same way, but no failure is returned: the caller
//      knows it asked, and that some items may not have run.
//   T5 A fork() during a call never gives the child a wrong answer: if a
//      worker was in the middle of the job, the child's call fails instead.
//      As the interpreter exits, every job stops, and fails rather than
//      return part of an answer.
//   And the scheduling policy's W1 to W3 and C1, stated with it.
//
// WHAT IT DOES NOT PROMISE
//
// Which thread runs which range, in what order, how big the ranges are, or
// how many threads help: these change from call to call, and only the speed
// depends on them, unless the body lets its answer depend on them.
//
// SHARP EDGES: rules for the body
//
//   - Items must not wait on each other or on anything the caller holds: the
//     caller waits for workers inside its ranges, so that is a deadlock. The
//     GIL is the one exception, and only on the calling thread, to poll for
//     Ctrl-C (xatlas's progress callback): every caller releases it before it
//     waits on the pool. Checked builds catch Exclusively taken in a body, and
//     a body calling back into the pool from Python.
//   - The body runs on worker threads too, with 8 MB stacks and every signal
//     blocked. Per-thread scratch is indexed by thread_index(), which no two
//     threads of one call share.
//   - The body must not fork, start threads, longjmp out, or end its thread;
//     a thread that ends inside it anyway fails the call (run_range).
//   - A body that throws stops the job early: check the failure before using
//     any output.
//   - A body may call for_each_range itself. At most SLOTS jobs share the
//     workers at once; a call that finds every slot taken runs alone, and
//     tries again after each of its ranges.
//   - `count` is at most MAX_ITEMS, 2^53 (more fails with TooManyItems). The
//     Stop's poll runs only on the calling thread, between its ranges.
//   - Only Linux runs the fuzzers, ThreadSanitizer and lints. The Windows and
//     macOS sleep and wake code builds there, and every wheel runs the tests.
//
// FOR REVIEWERS: the slot protocol
//
// The state, checked in every interleaving by check/verify/tla/Scheduler.tla (each
// transition a function of the same name below, and SlotWord's arithmetic):
//   JobSlot.word = epoch (32 bits) | OPEN | active (31 bits): active counts
//   the caller and the workers that joined; free when OPEN and active are 0.
//   try_publish   free slot: epoch + 1, active 1, store the job, then OPEN
//   join_job      CAS from the exact word seen (OPEN, that epoch) to active + 1
//   leave_job     active - 1; the last out of a closed job wakes the caller
//   close_job     clear OPEN, active - 1; then wait for active == 0
// A worker touches a job only between a successful join and its leave, a
// join succeeds only while that job's epoch is open, and a caller returns
// only after closing and seeing active reach 0: so no worker touches a job
// after its caller has returned (T1). The word changes only by
// read-modify-writes, so checked builds check each transition's old word as
// it happens, holding the protocol to its states at real sizes.
//
// Threads sleep by waiting on an address (futex, WaitOnAddress, __ulock),
// which compares and sleeps in one step, so no wake-up is lost, and the pool
// holds no lock a fork could leave held (Exclusively is the extension's own,
// whose users refuse a device the parent opened): the child frees every slot
// but those its one thread is still inside (child_after_fork), and starts its
// own workers when it needs them.
#pragma once

#include "platform.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <utility>
#if defined(TRIMKIT_TRACE)
#include <string_view>
#endif

namespace trimkit::parallel {

// The lesser and the greater of two values: <algorithm> is one header GenMC
// can't compile (check/verify/genmc), and these are all this file needs of it.
// For the same reason every compare-exchange here is the strong one: GenMC
// models a weak one failing for no reason, which makes any loop retrying it
// unbounded, and on x86 the two are one instruction
template <typename T>
constexpr T least(T a, T b) {
    return b < a ? b : a;
}
template <typename T>
constexpr T most(T a, T b) {
    return a < b ? b : a;
}

// -- The entry point ----------------------------------------------------------

// What the caller knows about what one item costs
enum class ItemCost : std::uint8_t {
    // Nothing: the pool times the first items itself to find out. Use this
    // unless the next one is known to be true
    Measured,
    // Each item is worth waking a thread for (a hull to split, a mesh to
    // chart): the job is shared at once, with as many workers as items, and
    // nothing is spent measuring first
    Coarse,
};

// How a call can be stopped from outside it: `poll(context)` runs on the
// calling thread between its ranges, and while it waits for workers, and
// true stops the call (Ctrl-C). Null never stops it
struct Stop {
    bool (*poll)(void* context) = nullptr;
    void* context = nullptr;
};

template <typename Body>
platform::Failure for_each_range(std::size_t count, const Body& body, ItemCost cost = ItemCost::Measured,
                                 std::size_t min_range = 1, Stop stop = {});

// -- Constants, each with where its value comes from --------------------------

// A sleeping worker costs no CPU, so a long idle exit keeps the pool warm
// across interactive use (Python then warns on os.fork() while threads live).
// Under GenMC an idle worker exits at once, so every thread of a checked
// execution ends, and a caller left waiting shows as a run that never ends
constexpr std::uint32_t IDLE_EXIT_MS = platform::MODEL_CHECKING ? 0 : 10'000;
constexpr std::uint32_t POLL_MS = 50;  // a waiting caller polls for Ctrl-C this often
// jobs shared at once; a caller finding none runs alone. Two under GenMC, as
// each slot a worker scans multiplies the executions it explores
constexpr std::size_t SLOTS = platform::MODEL_CHECKING ? 2 : 4;
constexpr std::size_t WORKER_STACK = std::size_t{8} << 20;  // a main thread's stack: musl's 128 KB has overflowed
constexpr std::uint32_t MAX_THREAD_SLOTS = 1024;            // thread indexes there can be (thread_slots)
constexpr std::int64_t CAP_REFRESH_NS = 100'000'000;        // the CPUs this process may use, read at most this often
// the most items one call takes: each thread in a job (at most MAX_THREAD_SLOTS)
// may move `next` on by up to `count` after it last saw items left, so this
// keeps `next` from wrapping (claim_range), at 2^53 items on 64 bits
constexpr std::size_t MAX_ITEMS = SIZE_MAX / (2 * std::size_t{MAX_THREAD_SLOTS});

// -- The scheduling policy: every choice the pool makes, as pure functions ----

// When a call wakes helpers and how many, how long an idle thread spins, and
// how big a claim is: functions of plain values, no clock and no atomic. The
// protocol below measures, asks, and acts, and never trusts an answer for T1
// to T5 (check/verify/tla leaves every one of these choices free). C, WAKE_COST_NS,
// is the CPU one wake costs: a system call and two context switches, 4 to 6 us
// measured on x86, which `scripts bench` reports on each machine. Nothing
// predicts when a woken worker arrives: one that comes late finds the counter
// empty and leaves.
//   W1 A Measured call wakes no one until its caller has run items for C; till
//      then only workers still spinning from an earlier call help.
//   W2 Then it wants one helper per 2C of the work left, at the mean cost of
//      what its caller ran, at most the cap and the items left but its own. A
//      helper costs C to wake and at most C spinning after (W3), so helpers
//      cost at most the work they share, and a call at most cap x 2C (a dear
//      or stalled first range can overstate the work). Decided after each of
//      the caller's ranges, from what is left then. Coarse wants every helper
//      the items can use, at once.
//   W3 An idle thread spins at most C, then sleeps: spin-then-block.
//   C1 A claim is at least min_range and 1, and at most that or half an even
//      share of what is left: guided self-scheduling, so the last claims are
//      short. Within those it aims to take as long as its thread has already
//      run the job, MIN_CLAIM_NS to MAX_CLAIM_NS, at its last range's cost an
//      item. With no stealing, a claim too big would idle the helpers: a range
//      shorter than TIMED_NS is counted as TIMED_NS, which only ever makes the
//      next claim smaller than its aim.
// The cap is the CPUs this process may use (affinity, narrowed by any cgroup
// quota) less the caller, read at most every CAP_REFRESH_NS (worker_cap). P3
// (check/verify/tla) holds the protocol to waking no more than this asks
namespace policy {

constexpr double WAKE_COST_NS = 5'000;   // C
constexpr double TIMED_NS = 2'000;       // the shortest range timed as itself: 100 reads of a fast clock
constexpr double MIN_CLAIM_NS = 4'000;   // a contended claim costs about 100 ns, 2.5% of this
constexpr double MAX_CLAIM_NS = 50'000;  // claims grow to this, so they are rare

// What one thread has run of one job: its ranges' time and items, and its
// last range's time per item (0: none yet), an overstatement when it was short
struct Ran {
    double ns = 0;
    std::size_t items = 0;
    double item_ns = 0;
};

// What a thread has run after one more range of `items` that took `took_ns`.
// `most` puts the constant first, so NaN gives TIMED_NS, and a time below 0
// adds none
constexpr Ran after_range(Ran ran, double took_ns, std::size_t items) {
    return Ran{ran.ns + most(0.0, took_ns), ran.items + items, most(TIMED_NS, took_ns) / static_cast<double>(items)};
}

// How many items the thread's next claim takes (C1)
constexpr std::size_t claim_items(Ran ran, std::size_t left, std::uint32_t sharing, std::size_t min_range) {
    const std::size_t share = most<std::size_t>(1, left / (2 * std::size_t{most(1U, sharing)}));
    std::size_t take = 1;
    if (ran.item_ns > 0) {
        const double by_time = least(MAX_CLAIM_NS, most(MIN_CLAIM_NS, ran.ns)) / ran.item_ns;
        take = by_time >= static_cast<double>(share) ? share : most<std::size_t>(1, static_cast<std::size_t>(by_time));
    }
    return most(take, min_range);
}

// How many helpers a call wants now, its caller having run `caller` (W1, W2)
constexpr std::uint32_t helpers_wanted(ItemCost cost, Ran caller, std::size_t left, std::uint32_t cap) {
    const auto room = static_cast<std::uint32_t>(least<std::size_t>(cap, left > 0 ? left - 1 : 0));
    if (cost == ItemCost::Coarse || room == 0) {
        return room;
    }
    if (!(caller.ns >= WAKE_COST_NS) || caller.items == 0) {
        return 0;  // and NaN
    }
    const double work_left = caller.ns / static_cast<double>(caller.items) * static_cast<double>(left);
    return static_cast<std::uint32_t>(least(static_cast<double>(room), work_left / (2 * WAKE_COST_NS)));
}

// Whether a call shares its job before running anything: Coarse, or workers
// already spinning, which join for no wake (wanting none, by W1)
constexpr bool share_at_once(ItemCost cost, std::uint32_t spinning, std::uint32_t cap) {
    return cap > 0 && (cost == ItemCost::Coarse || spinning > 0);
}

// How long an idle worker spins before it sleeps (W3), and a caller before it
// sleeps waiting on workers inside their last claims, two first claims long.
// None under GenMC, whose clock stands still while a thread waits
constexpr double spin_ns() { return platform::MODEL_CHECKING ? 0.0 : WAKE_COST_NS; }
constexpr double close_spin_ns() { return platform::MODEL_CHECKING ? 0.0 : 2 * MIN_CLAIM_NS; }

static_assert(helpers_wanted(ItemCost::Measured, Ran{WAKE_COST_NS - 1, 1'000, 1e9}, 1'000'000, 255) == 0, "W1");
static_assert(helpers_wanted(ItemCost::Measured, Ran{WAKE_COST_NS, 10, 0}, 41, 255) == 2, "W2: 20.5 us left, 2C each");
static_assert(helpers_wanted(ItemCost::Measured, Ran{1e12, 1, 0}, 2, 255) == 1, "W2: no more than the items left");
static_assert(helpers_wanted(ItemCost::Coarse, Ran{}, 5, 3) == 3 && helpers_wanted(ItemCost::Coarse, Ran{}, 5, 9) == 4);
static_assert(spin_ns() <= WAKE_COST_NS, "W3");
static_assert(claim_items(Ran{}, 0, 1, 1) == 1 && claim_items(Ran{}, 100, 1, 7) == 7, "C1: never empty, min_range");
static_assert(claim_items(Ran{0, 0, 1e-3}, 1'000, 4, 1) == 125, "C1: half an even share");
static_assert(after_range(Ran{}, 10, 4).item_ns == TIMED_NS / 4, "a short range overstates its cost");
static_assert(!share_at_once(ItemCost::Measured, 0, 7) && !share_at_once(ItemCost::Coarse, 1, 0));

}  // namespace policy

// -- Checked and counted builds -----------------------------------------------

// Checked builds (TRIMKIT_CHECKED: every sanitized build, and the fuzzers)
// check what the pool and its callers promise each other at each step, and a
// broken promise ends the process saying which. Free elsewhere
#if defined(TRIMKIT_CHECKED)
constexpr bool CHECKED = true;
#else
constexpr bool CHECKED = false;
#endif

// Stop the process, naming `promise`, unless it was `kept`; nothing outside a
// checked build
inline void checked(bool kept, const char* promise) {
    if (CHECKED && !kept) {
        platform::fail_fast(promise);
    }
}

// Counted builds (TRIMKIT_PARALLEL_COUNTERS: the benchmark and the fuzzer)
// count what the pool did, for W1 on real calls and what each call cost
#if defined(TRIMKIT_PARALLEL_COUNTERS)
constexpr bool COUNTED = true;
#else
constexpr bool COUNTED = false;
#endif

struct PoolCounters {
    std::atomic<std::uint64_t> publishes{0};  // jobs shared in a slot
    std::atomic<std::uint64_t> wakes{0};      // calls that woke sleeping workers
    std::atomic<std::uint64_t> sleeps{0};     // sleeps on a word, by callers and workers
    std::atomic<std::uint64_t> spawns{0};     // worker threads started
    std::atomic<std::uint64_t> woken{0};      // sleeping workers woken, as the system counts them
    std::atomic<std::uint64_t> spin_ns{0};    // time idle workers spent spinning before they slept or saw a wake
};
inline PoolCounters counters;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// One more of `counter`, in a counted build
inline void count(std::atomic<std::uint64_t>& counter, std::uint64_t by = 1) {
    if constexpr (COUNTED) {
        counter.fetch_add(by, std::memory_order_relaxed);
    }
}

// In checked builds, what this thread is inside of: a body's ranges (any
// job's), and Exclusively
inline thread_local std::uint32_t bodies_running = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local bool holding_exclusively = false;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// -- What a call fails with when the pool, not its body, ends it --------------

struct StoppedAtExit : std::exception {
    const char* what() const noexcept override { return "the interpreter is exiting: this call was stopped"; }
};
struct ForkedDuringCall : std::exception {
    const char* what() const noexcept override {
        return "the process forked during this call, so it has no answer in the child: call it again";
    }
};
struct ThreadEndedDuringCall : std::exception {
    const char* what() const noexcept override { return "a thread running this call's items ended"; }
};
struct TooManyItems : std::exception {
    const char* what() const noexcept override { return "too many items for one call"; }
};

// -- The state: a job, its slot word, the slots, the pool ---------------------

// One call: its items and body, and what the threads sharing it agree on. It
// lives on the caller's stack, and only threads counted in its slot touch it
// (T1)
struct RangeJob {
    std::size_t count = 0;
    void (*run)(const void* body, std::size_t begin, std::size_t end) = nullptr;
    const void* body = nullptr;
    ItemCost item_cost = ItemCost::Measured;
    std::size_t min_range = 1;         // no claim is smaller, but the last
    Stop stop_by;                      // polled on the caller only
    std::atomic<std::size_t> next{0};  // the first item not yet handed out
    // what the caller measured an item to cost, in sixteenths of a ns (0: not
    // yet), and how many threads it means to share the job: a worker that
    // joins starts from both, then measures its own ranges
    std::atomic<std::uint64_t> item_cost_16{0};
    std::atomic<std::uint32_t> sharing{1};
    std::atomic<bool> stopped{false};  // a failure, a poll, a fork or the exit: no more ranges
    std::atomic<bool> failure_claimed{false};
    platform::Failure failure;  // the first, read by the caller once the job is closed
    // set in a forked child: a worker was inside when the process forked, so
    // the items it held never run here, and the job fails (child_after_fork)
    std::atomic<bool> forked{false};
};

// One slot's word (FOR REVIEWERS, at the top). Each transition is one
// read-modify-write by a delta below (but the fork's, on the child's one
// thread): the functions the model checks, from a table this generates
struct SlotWord {
    static constexpr std::uint64_t OPEN = std::uint64_t{1} << 31;
    static constexpr std::uint64_t ACTIVE = OPEN - 1;
    static constexpr std::uint64_t EPOCH = std::uint64_t{1} << 32;
    static constexpr std::uint64_t RESERVE = EPOCH + 1;  // a free slot: the next epoch, its caller counted
    static constexpr std::uint64_t JOIN = 1;             // one more worker
    static constexpr std::uint64_t LEAVE = 1;            // one fewer
    static constexpr std::uint64_t CLOSE = OPEN | 1;     // no more joins, and the caller out

    std::uint64_t bits = 0;

    constexpr std::uint32_t epoch() const { return static_cast<std::uint32_t>(bits >> 32); }
    constexpr bool open() const { return (bits & OPEN) != 0; }
    constexpr std::uint64_t active() const { return bits & ACTIVE; }
    constexpr bool free() const { return (bits & (OPEN | ACTIVE)) == 0; }

    constexpr SlotWord reserved() const { return SlotWord{bits + RESERVE}; }
    constexpr SlotWord opened() const { return SlotWord{bits | OPEN}; }
    constexpr SlotWord joined() const { return SlotWord{bits + JOIN}; }
    constexpr SlotWord left() const { return SlotWord{bits - LEAVE}; }
    constexpr SlotWord closed() const { return SlotWord{bits - CLOSE}; }
    // in a forked child: free, under the next epoch
    constexpr SlotWord freed_by_fork() const { return SlotWord{(bits & ~(OPEN | ACTIVE)) + EPOCH}; }
    // in a forked child, a job its one thread is still calling: open, only it counted
    constexpr SlotWord reopened_by_fork() const { return SlotWord{freed_by_fork().bits | OPEN | 1}; }
    // in a forked child, a job its one thread is inside as a worker: closed, only it counted
    constexpr SlotWord kept_by_fork() const { return SlotWord{freed_by_fork().bits + 1}; }
};

static_assert(SlotWord{}.free() && SlotWord{}.reserved().active() == 1 && !SlotWord{}.reserved().open());
static_assert(SlotWord{}.reserved().epoch() == 1 && SlotWord{}.reserved().opened().open());
static_assert(SlotWord{}.reserved().opened().joined().active() == 2);
static_assert(SlotWord{}.reserved().opened().joined().left().active() == 1);
static_assert(SlotWord{}.reserved().opened().closed().free());
static_assert(SlotWord{}.reserved().opened().joined().closed().active() == 1);
static_assert(!SlotWord{}.reserved().opened().joined().closed().open());
static_assert(SlotWord{}.reserved().opened().joined().closed().left().free());
static_assert(SlotWord{}.reserved().opened().joined().freed_by_fork().free());
static_assert(SlotWord{}.reserved().opened().freed_by_fork().epoch() == 2);
static_assert(SlotWord{}.reserved().opened().reopened_by_fork().closed().free());
static_assert(SlotWord{}.reserved().opened().joined().kept_by_fork().left().free());

// A place one job is shared from
struct alignas(64) JobSlot {
    std::atomic<std::uint64_t> word{0};  // a SlotWord's bits
    std::atomic<RangeJob*> job{nullptr};
    // bumped by the last worker out of a closed job: its caller sleeps on it
    std::atomic<std::uint32_t> left{0};
};

// The one pool: atomics only, shared by every caller
struct WorkerPool {
    JobSlot slots[SLOTS];
    alignas(64) std::atomic<std::uint32_t> wake{0};  // bumped to tell idle workers to look: they sleep on it
    std::atomic<std::uint32_t> spinning{0};          // workers spinning on `wake`, who will see a bump
    std::atomic<std::uint32_t> live{0};              // workers started and not yet exited
    std::atomic<std::uint32_t> wanted{0};            // how many workers the last wake wanted
    std::atomic<std::uint32_t> cap{0};               // the most workers: CPUs this process may use, less 1
    std::atomic<std::int64_t> cap_read_at{0};        // when `cap` was last read from the system (0: never)
};
inline WorkerPool pool;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// Sleep while `word` holds `expected`, for at most `timeout_ms`, counted. It
// may return early, so the caller reads the word again
inline void park(std::atomic<std::uint32_t>& word, std::uint32_t expected, std::uint32_t timeout_ms) {
    count(counters.sleeps);
    platform::wait_while_equal(word, expected, timeout_ms);
}

// -- Traced builds: each step of the protocol, logged for the model to replay ---

// Traced builds (TRIMKIT_TRACE: check/verify/tla/trace.cpp alone) log each step
// below as its model action, with the shared state after it. A step holds one
// lock from start to end, so the log is one order the steps took, each atomic,
// which check/verify/tla/SchedulerTrace.tla replays. Anywhere else a step is nothing

// One step as logged: its thread, its action, what it took (a range, a count),
// and every slot word and `left`, the wake, spinning and live counts after it
struct TraceRecord {
    std::uint32_t thread = 0;
    const char* action = nullptr;
    std::uint64_t arg[2] = {};
    std::uint64_t word[SLOTS] = {};
    std::uint32_t left[SLOTS] = {};
    std::uint32_t wake = 0;
    std::uint32_t spinning = 0;
    std::uint32_t live = 0;
};

// The log, which check/verify/tla/trace.cpp provides, and the lock each step holds
struct TraceLog {
    TraceRecord* records = nullptr;
    std::size_t capacity = 0;
    std::size_t size = 0;
    std::uint32_t callers = 0;  // callers number themselves 1, 2, ...; workers come after
    std::uint64_t workers = 0;  // the worker numbers held, a bit each: an exited one's is reused, as the model's are
    std::atomic<bool> locked{false};
};
inline TraceLog trace_log;                           // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local std::uint32_t trace_thread = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// One step of the protocol, the model's action `action`: in a traced build it
// holds the trace lock from here to the end of its scope, then is logged with
// what `took` says it took, unless `skip` said it was no step. Anywhere else
// it is empty, and its calls are nothing
#if defined(TRIMKIT_TRACE)
class TraceStep {
public:
    explicit TraceStep(const char* action) : action_(action) {
        while (trace_log.locked.exchange(true, std::memory_order_acquire)) {
            platform::cpu_relax();
        }
    }
    ~TraceStep() {
        if (!skipped_ && trace_log.size < trace_log.capacity) {
            TraceRecord& record = trace_log.records[trace_log.size++];
            if (trace_thread == 0) {  // a worker's first step: the lowest number free
                std::uint32_t free = 0;
                while ((trace_log.workers >> free & 1U) != 0) {
                    free++;
                }
                trace_log.workers |= std::uint64_t{1} << free;
                trace_thread = trace_log.callers + 1 + free;
            }
            record.thread = trace_thread;
            record.action = action_;
            record.arg[0] = took_[0];
            record.arg[1] = took_[1];
            for (std::size_t s = 0; s < SLOTS; s++) {
                record.word[s] = pool.slots[s].word.load(std::memory_order_relaxed);
                record.left[s] = pool.slots[s].left.load(std::memory_order_relaxed);
            }
            record.wake = pool.wake.load(std::memory_order_relaxed);
            record.spinning = pool.spinning.load(std::memory_order_relaxed);
            record.live = pool.live.load(std::memory_order_relaxed);
            if (action_ == std::string_view("WorkerExitRecheck")) {  // its last step: its number is free
                trace_log.workers &= ~(std::uint64_t{1} << (trace_thread - trace_log.callers - 1));
            }
        }
        trace_log.locked.store(false, std::memory_order_release);
    }
    TraceStep(const TraceStep&) = delete;
    TraceStep& operator=(const TraceStep&) = delete;

    void took(std::uint64_t first, std::uint64_t second = 0) const {
        took_[0] = first;
        took_[1] = second;
    }
    void skip() const { skipped_ = true; }

private:
    const char* action_;
    mutable std::uint64_t took_[2] = {};
    mutable bool skipped_ = false;
};
#else
class TraceStep {
public:
    explicit TraceStep(const char* /*action*/) {}
    void took(std::uint64_t /*first*/, std::uint64_t /*second*/ = 0) const {}
    void skip() const {}
};
#endif

// -- The exit gate: no call is left part way when the interpreter exits -------

// Calls that might still be running as the interpreter exits count themselves
// in (gil.hpp's GilReleased: every call that lets Python run beside it). At
// the exit the gate closes: every job stops at the next claim that reads it
// closed (one that read it open still takes its range), and run_job
// fails one that didn't run every item with StoppedAtExit; then whoever
// closed it waits for every call inside to leave. A call counts itself in
// before it reads whether the gate is closed, and the closer closes it before
// it reads the count, both sequentially consistent, so either the call sees
// the gate closed or the closer sees the call (check/verify/genmc, exit_gate)
struct ExitGate {
    std::atomic<bool> closed{false};
    std::atomic<std::uint32_t> inside{0};
};
inline ExitGate exit_gate;                          // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local std::uint32_t inside_here = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// Count this thread's call in: false if the gate has closed, when the call
// must not let go of whatever it would have (the GIL)
inline bool enter_exit_gate() {
    exit_gate.inside.fetch_add(1, std::memory_order_seq_cst);  // pairs with close_exit_gate's store
    inside_here++;
    return !exit_gate.closed.load(std::memory_order_seq_cst);
}

// Count this thread's call out
inline void leave_exit_gate() {
    inside_here--;
    // release: what the call did happens before the exit goes on (pairs with
    // wait_for_exit_gate's load); the pairing that must be sequentially
    // consistent is enter_exit_gate's with close_exit_gate's
    exit_gate.inside.fetch_sub(1, std::memory_order_release);
}

// Close the gate: every job stops at the next claim that reads it closed
inline void close_exit_gate() { exit_gate.closed.store(true, std::memory_order_seq_cst); }

// Whether the gate has closed
inline bool exit_gate_closed() { return exit_gate.closed.load(std::memory_order_relaxed); }

// Wait, a millisecond at a time, for every call inside to leave: as long as
// that takes, since a call left part way is the crash the gate exists to
// prevent. No call wakes the waiter, which would cost every call a syscall
// for the one exit that waits
inline void wait_for_exit_gate() {
    for (;;) {
        const std::uint32_t inside = exit_gate.inside.load(std::memory_order_seq_cst);
        if (inside == 0) {
            return;
        }
        platform::wait_while_equal(exit_gate.inside, inside, 1);
    }
}

// -- Thread indexes: per-thread scratch for a library that keeps it (xatlas) ---

// Every thread running the items of any one job has its own index below
// thread_slots(): a caller's is 0 (a worker that calls for_each_range from a
// job keeps its own), and each worker holds one of 1, 2, ... from when it is
// counted live until it isn't, never two at once, and the next to take an
// index sees what the last one wrote. thread_slots() is the CPUs the machine
// has online (platform::cpus_online), which affinity and a quota only
// narrow, so it is fixed while worker_cap() follows them up and down
inline std::atomic<std::uint64_t>
    held_indexes[MAX_THREAD_SLOTS / 64];                  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local std::uint32_t this_thread_index = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// How many thread indexes there are
inline std::uint32_t thread_slots() { return least(platform::cpus_online(), MAX_THREAD_SLOTS); }

// This thread's index: 0 unless it is a worker
inline std::uint32_t thread_index() { return this_thread_index; }

// The lowest free index, taken for this worker; false if none is, which a
// live worker never sees (the cap leaves room for every one)
inline bool take_thread_index() {
    for (std::uint32_t index = 1; index < thread_slots(); index++) {
        const std::uint64_t bit = std::uint64_t{1} << (index % 64);
        // acquire: pairs with give_thread_index's release, so the scratch the
        // last holder wrote is seen
        if ((held_indexes[index / 64].fetch_or(bit, std::memory_order_acquire) & bit) == 0) {
            this_thread_index = index;
            return true;
        }
    }
    return false;
}

// This worker's index given back
inline void give_thread_index() {
    const std::uint32_t index = this_thread_index;
    // release: pairs with take_thread_index's acquire
    held_indexes[index / 64].fetch_and(~(std::uint64_t{1} << (index % 64)), std::memory_order_release);
    this_thread_index = 0;
}

// The most workers: the CPUs this process may use, less the caller, and room
// for every worker's index. Read from the system at most every CAP_REFRESH_NS,
// as it costs a syscall and a cgroup file each level up, where affinity and
// quotas rarely move. So a change takes effect within that, and a worker
// started inside it keeps the affinity of the thread that started it (as
// every new thread does) until it exits idle
inline std::uint32_t worker_cap() {
    const std::int64_t now = platform::monotonic_ns();
    const std::int64_t read_at = pool.cap_read_at.load(std::memory_order_relaxed);
    if (read_at != 0 && now - read_at < CAP_REFRESH_NS) {
        return pool.cap.load(std::memory_order_relaxed);
    }
    const std::uint32_t cap = least(platform::cpus_allowed(), thread_slots()) - 1;
    pool.cap.store(cap, std::memory_order_relaxed);
    pool.cap_read_at.store(now, std::memory_order_relaxed);
    return cap;
}

// -- Running items, and how a job stops ---------------------------------------

// The items not yet handed out (an estimate while others claim)
inline std::size_t items_left(const RangeJob& job) {
    return job.count - least(job.count, job.next.load(std::memory_order_relaxed));
}

// The next range, sized by the policy for a thread that has run `ran` of the
// job (C1), into [*begin, *end), as the model's step `action`: false once none
// are left or the job has stopped. Never empty, whatever the policy says, so
// every claim moves `next` on and the job ends (T2, T3). `next` passes `count`
// only by the takes in flight, one a thread, so with `count` at most MAX_ITEMS
// it never wraps
inline bool claim_range(RangeJob& job, const policy::Ran& ran, const char* action, std::size_t* begin,
                        std::size_t* end) {
    const std::size_t take = most<std::size_t>(
        1, policy::claim_items(ran, items_left(job), job.sharing.load(std::memory_order_relaxed), job.min_range));
    const TraceStep step(action);
    if (job.stopped.load(std::memory_order_relaxed) || exit_gate_closed() ||
        job.next.load(std::memory_order_relaxed) >= job.count) {
        return false;
    }
    *begin = job.next.fetch_add(take, std::memory_order_relaxed);  // atomicity alone: each item claimed once (T2)
    if (*begin >= job.count) {
        return false;
    }
    *end = *begin + least(take, job.count - *begin);
    step.took(*begin + 1, *end + 1);  // the model numbers items from 1
    return true;
}

// The job's first failure is kept, and it stops; later ones are dropped (T4)
inline void fail_job(RangeJob& job, platform::Failure failure) {
    if (!job.failure_claimed.exchange(true, std::memory_order_relaxed)) {
        job.failure = std::move(failure);
    }
    job.stopped.store(true, std::memory_order_relaxed);
}

// Items [begin, end) run on this thread, timed: what it has run after them.
// What they throw fails the job; a thread ending inside them fails it too, as
// their items never run
inline policy::Ran run_range(RangeJob& job, const policy::Ran& ran, std::size_t begin, std::size_t end) {
    struct Running {
        Running() {
            if constexpr (CHECKED) {
                bodies_running++;
            }
        }
        ~Running() {
            if constexpr (CHECKED) {
                bodies_running--;
            }
        }
        Running(const Running&) = delete;
        Running& operator=(const Running&) = delete;
    } const running;
    const std::int64_t start = platform::monotonic_ns();
    platform::Failure failure = platform::run_catching(
        [&] { job.run(job.body, begin, end); }, [&] { fail_job(job, platform::failure<ThreadEndedDuringCall>()); });
    const auto took = static_cast<double>(platform::monotonic_ns() - start);
    if (failure) {
        fail_job(job, std::move(failure));
    }
    return policy::after_range(ran, took, end - begin);
}

// The caller only: a poll that says stop stops the job (T4). Once it has
// stopped there is nothing to poll for, and a poll may take the GIL
inline void poll_stop(RangeJob& job) {
    if (job.stop_by.poll != nullptr && !job.stopped.load(std::memory_order_relaxed) &&
        job.stop_by.poll(job.stop_by.context)) {
        job.stopped.store(true, std::memory_order_relaxed);
    }
}

// -- Fork: what the child keeps of the pool -----------------------------------

// A slot this thread published and hasn't let go of, innermost last: what a
// fork from inside a call (a poll running Python that forks) leaves the
// child's one thread holding (child_after_fork). Kept by SharedJob
struct HeldSlot {
    JobSlot* slot = nullptr;
    RangeJob* job = nullptr;
    bool closed = false;
    std::uint32_t epoch = 0;  // what it was published as
    HeldSlot* outer = nullptr;
};
inline thread_local HeldSlot* held_slots = nullptr;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// the slot and job this thread is inside as a worker, if it is one: what a
// fork from a body on a worker leaves its child's one thread inside
inline thread_local JobSlot* joined_slot = nullptr;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local RangeJob* joined_job = nullptr;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// In a forked child, whose one thread is the one that forked: every worker,
// and every other caller, belonged to the parent. Every slot is freed under a
// new epoch, but
//   - one the forking thread published and hasn't closed (it is inside that
//     call, which goes on) is opened again with it alone counted, so closing
//     it works;
//   - one it is inside of as a worker is closed with it alone counted, so its
//     leave frees it, and that job, whose caller is gone, stops.
// A job a worker of the parent was inside of lost that worker's items: it
// stops, and fails (T5). The new epoch ends any wait the forking thread was in
inline void child_after_fork() {
    std::uint64_t before[SLOTS];
    for (std::size_t s = 0; s < SLOTS; s++) {
        before[s] = pool.slots[s].word.load(std::memory_order_relaxed);
        pool.slots[s].word.store(SlotWord{before[s]}.freed_by_fork().bits, std::memory_order_relaxed);
        pool.slots[s].job.store(nullptr, std::memory_order_relaxed);
    }
    for (HeldSlot* held = held_slots; held != nullptr; held = held->outer) {
        const SlotWord word{before[held->slot - pool.slots]};
        // workers of this very call inside: the caller is counted until it
        // closes, and a closed slot may hold another caller's job since
        if (word.epoch() == held->epoch && word.active() > (held->closed ? 0 : 1)) {
            held->job->forked.store(true, std::memory_order_relaxed);
            held->job->stopped.store(true, std::memory_order_relaxed);
        }
        if (!held->closed) {
            held->slot->word.store(word.reopened_by_fork().bits, std::memory_order_relaxed);
            held->slot->job.store(held->job, std::memory_order_relaxed);
            held->epoch = word.reopened_by_fork().epoch();  // what a second fork compares with
        }
    }
    if (joined_slot != nullptr) {
        joined_job->stopped.store(true, std::memory_order_relaxed);
        joined_slot->word.store(SlotWord{before[joined_slot - pool.slots]}.kept_by_fork().bits,
                                std::memory_order_relaxed);
    }
    pool.live.store(this_thread_index != 0 ? 1 : 0, std::memory_order_relaxed);
    pool.wanted.store(0, std::memory_order_relaxed);
    pool.spinning.store(0, std::memory_order_relaxed);
    pool.cap_read_at.store(0, std::memory_order_relaxed);
    for (std::atomic<std::uint64_t>& word : held_indexes) {
        word.store(0, std::memory_order_relaxed);
    }
    if (this_thread_index != 0) {
        held_indexes[this_thread_index / 64].store(std::uint64_t{1} << (this_thread_index % 64),
                                                   std::memory_order_relaxed);
    }
    // the child's interpreter isn't exiting, and only the forking thread's own
    // calls are inside the gate
    exit_gate.closed.store(false, std::memory_order_relaxed);
    exit_gate.inside.store(inside_here, std::memory_order_relaxed);
}

// Registered once, by the module's init (src/trimkit.cpp, through common.hpp)
inline void install_fork_handler() { platform::on_fork_child(child_after_fork); }

// -- Slot transitions: each one read-modify-write of a slot word --------------

// A free slot holding `job`, published, with its caller counted; or none when
// every slot is busy, and the caller runs alone
inline JobSlot* try_publish(RangeJob& job) {
    JobSlot* reserved = nullptr;
    {
        const TraceStep step("CallerPublish");
        for (std::size_t s = 0; s < SLOTS && reserved == nullptr; s++) {
            std::uint64_t word = pool.slots[s].word.load(std::memory_order_relaxed);
            // reserved, with the caller counted: no other caller can take it,
            // and no worker can join until it opens. Acquire: pairs with the
            // release of the last leave_job of the slot's previous job, so
            // whatever that worker read of the slot (its job pointer) happens
            // before the store below
            if (SlotWord{word}.free() &&
                pool.slots[s].word.compare_exchange_strong(word, SlotWord{word}.reserved().bits,
                                                           std::memory_order_acquire, std::memory_order_relaxed)) {
                reserved = &pool.slots[s];
                step.took(s + 1);
            }
        }
    }
    if (reserved == nullptr) {
        return nullptr;
    }
    {
        const TraceStep step("CallerPublishStore");
        reserved->job.store(&job, std::memory_order_relaxed);
    }
    const TraceStep step("CallerPublishOpen");
    // release: pairs with join_job's acquire, publishing the job and its pointer
    const SlotWord opened{reserved->word.fetch_or(SlotWord::OPEN, std::memory_order_release)};
    checked(!opened.open() && opened.active() == 1, "a slot opens once, reserved by its caller alone");
    count(counters.publishes);
    return reserved;
}

// One more worker inside the job published as `seen`, unless it has closed or
// the slot holds another job since
inline bool join_job(JobSlot& slot, std::uint64_t seen) {
    const TraceStep step("WorkerJoin");
    std::uint64_t word = seen;
    while (SlotWord{word}.open() && SlotWord{word}.epoch() == SlotWord{seen}.epoch()) {
        // acquire: pairs with try_publish's release of OPEN
        if (slot.word.compare_exchange_strong(word, SlotWord{word}.joined().bits, std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
            checked(SlotWord{word}.active() >= 1 && SlotWord{word}.active() < SlotWord::ACTIVE,
                    "a worker joins only a job its caller is in");
            return true;
        }
    }
    return false;
}

// This worker out of the slot's job, which it never touches again; the last
// out of a closed job wakes its caller
inline void leave_job(JobSlot& slot) {
    SlotWord before;
    {
        const TraceStep step("WorkerLeave");
        // release: pairs with close_job's and workers_gone's acquire, so what
        // this worker wrote happens before its caller returns (T1)
        before = SlotWord{slot.word.fetch_sub(SlotWord::LEAVE, std::memory_order_release)};
    }
    checked(before.active() >= (before.open() ? 2 : 1), "a worker leaves only a job it joined");
    if (!before.open() && before.active() == 1) {
        const TraceStep step("WorkerLeaveBump");
        // release: pairs with wait_for_workers' acquire, so a caller that sees
        // this bump sees the leave before it, and doesn't sleep on the bump
        slot.left.fetch_add(1, std::memory_order_release);
        platform::wake_all(slot.left);
    }
}

// The caller's job closed to new workers, and the caller out: the word as it
// was, so workers may still be inside unless it was active 1
inline SlotWord close_job(JobSlot& slot) {
    const TraceStep step("CallerClose");
    // acquire: pairs with every earlier leave_job's release (T1). Nothing the
    // caller wrote is read through the slot after this, so no release
    const SlotWord before{slot.word.fetch_sub(SlotWord::CLOSE, std::memory_order_acquire)};
    checked(before.open() && before.active() >= 1, "a caller closes the job it opened, once");
    return before;
}

// No worker is left in the job published as `epoch`: none active, or the slot
// published again since, which it can only be once free. Without the epoch a
// caller that looked late would wait on the next caller's job
// (check/verify/tla, mutation wait_ignores_epoch)
inline bool workers_gone(const JobSlot& slot, std::uint32_t epoch) {
    // acquire: pairs with leave_job's release (T1)
    const SlotWord word{slot.word.load(std::memory_order_acquire)};
    return word.active() == 0 || word.epoch() != epoch;
}

// Wait until no worker is inside the job published as `epoch`, polling for a
// stop while it does when `poll` (never while unwinding: see SharedJob). It
// spins first (policy::close_spin_ns), then sleeps
inline void wait_for_workers(JobSlot& slot, RangeJob& job, std::uint32_t epoch, bool poll) {
    const std::int64_t spin_until = platform::monotonic_ns() + static_cast<std::int64_t>(policy::close_spin_ns());
    for (;;) {
        const bool spinning = platform::monotonic_ns() < spin_until;
        {
            const TraceStep step("CallerCloseCheck");
            if (workers_gone(slot, epoch)) {
                return;
            }
            if (spinning) {
                step.skip();  // a look that spins on is no step: the model's look is the last
                platform::cpu_relax();
                continue;
            }
        }
        // read before the count, so a worker leaving after this bumps `left`
        // and the sleep returns at once. Acquire: pairs with leave_job's
        // release, so a bump seen here means the check below sees its leave
        std::uint32_t left = 0;
        {
            const TraceStep step("CallerCloseLoadLeft");
            left = slot.left.load(std::memory_order_acquire);
        }
        {
            const TraceStep step("CallerCloseRecheck");
            if (workers_gone(slot, epoch)) {
                return;
            }
        }
        bool moved = false;
        {
            const TraceStep step("CallerClosePark");  // the sleep's compare, which the kernel makes again
            moved = slot.left.load(std::memory_order_acquire) != left;
        }
        if (!moved) {
            park(slot.left, left, POLL_MS);
        }
        if (poll) {
            poll_stop(job);
        }
    }
}

// The caller's hold on its published slot, let go however run_job leaves it:
// by returning, or unwinding (a thread exiting inside `poll`, as CPython 3.12
// and 3.13 end a thread taking the GIL while the interpreter finalizes). The
// job is stopped, the slot closed, and the caller waits until no worker is
// left inside the job, which lives on its stack (T1). While unwinding it
// never polls, as a second exception out of a destructor is std::terminate
class SharedJob {
public:
    explicit SharedJob(RangeJob& job) : job_(job) {}
    SharedJob(const SharedJob&) = delete;
    SharedJob& operator=(const SharedJob&) = delete;
    ~SharedJob() {
        if (slot_ != nullptr) {
            job_.stopped.store(true, std::memory_order_relaxed);
        }
        release(/*poll=*/false);
    }

    // published in a free slot, if there is one
    bool publish() {
        slot_ = try_publish(job_);
        if (slot_ != nullptr) {
            // field by field: GenMC can't run a copy of the padding after `closed`
            held_.slot = slot_;
            held_.job = &job_;
            held_.epoch = SlotWord{slot_->word.load(std::memory_order_relaxed)}.epoch();
            held_.outer = held_slots;
            held_slots = &held_;
        }
        return slot_ != nullptr;
    }

    bool published() const { return slot_ != nullptr; }

    // closed, then no worker left inside. A poll that throws while it waits
    // leaves the slot closed, and the destructor waits out the rest
    void release(bool poll) {
        if (slot_ == nullptr) {
            return;
        }
        if (!held_.closed) {
            const SlotWord before = close_job(*slot_);
            held_.closed = true;
            epoch_ = before.epoch();
            if (before.active() == 1) {
                let_go();
                return;
            }
        }
        wait_for_workers(*slot_, job_, epoch_, poll);
        let_go();
    }

private:
    void let_go() {
        held_slots = held_.outer;
        slot_ = nullptr;
    }

    RangeJob& job_;
    JobSlot* slot_ = nullptr;
    HeldSlot held_;
    std::uint32_t epoch_ = 0;
};

// -- Waking workers -----------------------------------------------------------

inline void* worker_main(void* unused);

// One more worker thread, unless as many as the cap are live or the system
// won't start one: the threads already running then take its ranges (T3)
inline bool spawn_worker() {
    std::uint32_t live = pool.live.load(std::memory_order_relaxed);
    for (;;) {
        if (live >= pool.cap.load(std::memory_order_relaxed)) {
            return false;
        }
        if (pool.live.compare_exchange_strong(live, live + 1, std::memory_order_relaxed)) {
            break;
        }
    }
    if (!platform::start_detached_thread(worker_main, WORKER_STACK)) {
        pool.live.fetch_sub(1, std::memory_order_relaxed);
        return false;
    }
    count(counters.spawns);
    return true;
}

// `more` more helpers for a job that wants `want` in all. The wake is bumped
// first, so a worker scanning or spinning sees the job; then only as many
// sleepers are woken as spinning workers won't cover. Both counts are read
// after the bump, sequentially consistent, and each worker leaves its count
// before it reads the wake (park_worker, worker_main): so a worker counted
// here as spinning sees the bump, and one counted live never exits without
// starting its replacement. Too few live, and one is started here, which
// starts the next: the caller pays for one thread start at most
inline void wake_helpers(std::uint32_t want, std::uint32_t more) {
    {
        const TraceStep step("CallerWake");
        pool.wake.fetch_add(1, std::memory_order_seq_cst);  // pairs with park_worker's, Holding's decrements
    }
    std::uint32_t spinning = 0;
    {
        const TraceStep step("CallerWakeSpinning");
        spinning = pool.spinning.load(std::memory_order_seq_cst);
    }
    {
        const TraceStep step("CallerWakeSpawn");
        if (pool.live.load(std::memory_order_seq_cst) < want) {
            pool.wanted.store(want, std::memory_order_relaxed);  // how far the spawn chain goes (worker_main)
            spawn_worker();
        }
    }
    const TraceStep step("CallerWakeSome");
    if (more > spinning) {
        count(counters.wakes);
        const std::uint32_t woken = platform::wake_some(pool.wake, more - spinning);
        count(counters.woken, woken);
        step.took(woken);
    }
}

// -- The worker ---------------------------------------------------------------

// Wait for the wake to move past `observed`: false once the worker has slept
// IDLE_EXIT_MS with nothing to do. It spins first (policy::spin_ns, W3),
// counted in `spinning` while it does, and out of it before it sleeps (pairs
// with wake_helpers)
inline bool park_worker(std::uint32_t observed) {
    const std::int64_t start = platform::monotonic_ns();
    const auto spin_until = start + static_cast<std::int64_t>(policy::spin_ns());
    {
        const TraceStep step("WorkerPark");
        pool.spinning.fetch_add(1, std::memory_order_seq_cst);
    }
    bool saw = false;
    while (!saw && platform::monotonic_ns() < spin_until) {
        const TraceStep step("WorkerParkSpin");
        // acquire: pairs with wake_helpers' bump, after the job it published
        saw = pool.wake.load(std::memory_order_acquire) != observed;
        if (!saw) {
            step.skip();  // a look that spins on is no step: the model's look is the last
            platform::cpu_relax();
        }
    }
    if (!saw) {
        const TraceStep step("WorkerParkSpin");  // out of budget, the model's look that gives up
    }
    {
        const TraceStep step(saw ? "WorkerParkSpinSaw" : "WorkerParkSpinDone");
        pool.spinning.fetch_sub(1, std::memory_order_seq_cst);  // out of the count before the wake is compared
    }
    if constexpr (COUNTED) {
        count(counters.spin_ns, static_cast<std::uint64_t>(platform::monotonic_ns() - start));
    }
    if (saw) {
        return true;
    }
    const std::int64_t idle_until = platform::monotonic_ns() + (std::int64_t{IDLE_EXIT_MS} * 1'000'000);
    for (;;) {
        const std::int64_t now = platform::monotonic_ns();
        if (now >= idle_until) {
            return false;
        }
        bool moved = false;
        {
            const TraceStep step("WorkerParkCompare");  // the sleep's compare, which the kernel makes again
            moved = pool.wake.load(std::memory_order_seq_cst) != observed;
        }
        if (moved) {
            return true;
        }
        park(pool.wake, observed, static_cast<std::uint32_t>(((idle_until - now) / 1'000'000) + 1));
    }
}

// The steps of a worker's life, each case of worker_main's switch, and each a
// group of actions of the model (check/verify/tla/Scheduler.tla: Worker<Step>...)
enum class WorkerStep : std::uint8_t {
    Start,  // started: the next of a spawn chain started, if the wake wants more
    Scan,   // read the wake, then look at each slot for an open job not already drained
    Join,   // counted into that job
    Claim,  // the next range of it, sized by this worker's own measure
    Run,    // that range, timed
    Leave,  // out of the job, which this worker never touches again
    Park,   // nothing joined: spin, then sleep until the wake moves
    Exit,   // idle long enough: its index and its count given back
};

// A worker thread: joins any open job, takes ranges until none are left,
// leaves, and waits for the wake to move when no job has work (T3: it joins
// only from Scan, holding nothing). However it ends, idle or unwinding in a
// body, it leaves the job it is in, gives back its index, leaves the live
// count, and, if a wake came after it last looked, starts its replacement
inline void* worker_main(void* /*unused*/) {
    if (!take_thread_index()) {
        checked(false, "a live worker finds a thread index free");
        pool.live.fetch_sub(1, std::memory_order_seq_cst);
        return nullptr;
    }
    // whatever this worker holds, given back however it ends
    struct Holding {
        std::uint32_t observed = 0;  // the wake as it last read it
        JobSlot* slot = nullptr;     // the slot it is inside, if any
        void leave() {
            joined_slot = nullptr;
            joined_job = nullptr;
            leave_job(*slot);
            slot = nullptr;
        }
        ~Holding() {
            const bool ending = slot != nullptr;  // unwinding inside a job, not idle
            if (ending) {
                leave();
            }
            {
                const TraceStep step(ending ? "WorkerExitEnding" : "WorkerExit");
                give_thread_index();
                // out of the count before it reads the wake (pairs with wake_helpers)
                pool.live.fetch_sub(1, std::memory_order_seq_cst);
            }
            const TraceStep step("WorkerExitRecheck");
            if (pool.wake.load(std::memory_order_seq_cst) != observed) {
                spawn_worker();
            }
        }
    } holding;
    // the epoch each slot last held when this worker found nothing left in
    // it: a drained job whose caller hasn't closed it yet isn't joined again
    std::uint32_t drained[SLOTS] = {};
    std::size_t at = 0;   // the slot it is looking at
    bool joined = false;  // whether it joined any job since it last read the wake
    std::uint64_t seen = 0;
    RangeJob* job = nullptr;
    policy::Ran ran;
    std::size_t begin = 0;
    std::size_t end = 0;
    WorkerStep step = WorkerStep::Start;
    for (;;) {
        switch (step) {
            case WorkerStep::Start: {
                const TraceStep traced("WorkerStart");
                // the spawn chain: a wake wanting more workers than are live started
                // this one, which starts the next
                if (pool.live.load(std::memory_order_relaxed) < pool.wanted.load(std::memory_order_relaxed)) {
                    spawn_worker();
                }
                step = WorkerStep::Scan;
                break;
            }
            case WorkerStep::Scan: {
                if (at == 0) {
                    const TraceStep traced("WorkerScan");
                    // before looking: a job published after this bumps the wake
                    holding.observed = pool.wake.load(std::memory_order_acquire);
                    joined = false;
                }
                const TraceStep traced("WorkerScanSlot");
                if (at == SLOTS) {
                    at = 0;
                    step = joined ? WorkerStep::Scan : WorkerStep::Park;
                    break;
                }
                // relaxed: only which slot to try; join_job's acquire is what pairs
                // with try_publish's release, before the job is read
                seen = pool.slots[at].word.load(std::memory_order_relaxed);
                if (!SlotWord{seen}.open() || SlotWord{seen}.epoch() == drained[at]) {
                    at++;  // nothing to join here
                    break;
                }
                step = WorkerStep::Join;
                break;
            }
            case WorkerStep::Join: {
                JobSlot& slot = pool.slots[at];
                if (!join_job(slot, seen)) {
                    at++;  // closed, or published again since
                    step = WorkerStep::Scan;
                    break;
                }
                // counted in `active` from here, so the slot can't be published
                // again until this worker leaves: the pointer is this epoch's
                {
                    const TraceStep traced("WorkerJoinLoad");
                    job = slot.job.load(std::memory_order_relaxed);
                }
                checked(job != nullptr, "an open slot holds its job");
                holding.slot = &slot;
                joined_slot = &slot;
                joined_job = job;
                joined = true;
                ran = policy::Ran{0, 0, static_cast<double>(job->item_cost_16.load(std::memory_order_relaxed)) / 16.0};
                step = WorkerStep::Claim;
                break;
            }
            case WorkerStep::Claim:
                step = claim_range(*job, ran, "WorkerClaim", &begin, &end) ? WorkerStep::Run : WorkerStep::Leave;
                break;
            case WorkerStep::Run:
                ran = run_range(*job, ran, begin, end);
                step = WorkerStep::Claim;
                break;
            case WorkerStep::Leave: {
                drained[at] = SlotWord{seen}.epoch();
                holding.leave();
                job = nullptr;
                at++;
                step = WorkerStep::Scan;
                break;
            }
            case WorkerStep::Park:
                step = park_worker(holding.observed) ? WorkerStep::Scan : WorkerStep::Exit;
                break;
            case WorkerStep::Exit:
                return nullptr;  // `holding` gives back its index and its count
        }
    }
}

// -- The caller ---------------------------------------------------------------

// The steps of a call, each case of run_job's switch, and each a group of
// actions of the model (check/verify/tla/Scheduler.tla: Caller<Step>...)
enum class CallerStep : std::uint8_t {
    Claim,    // the next range, sized by the policy (C1)
    Run,      // that range, timed
    Decide,   // the stop's poll, then how many helpers the work wants (W1, W2)
    Publish,  // the job into a free slot, if it isn't yet
    Wake,     // that many helpers, fewer if some are spinning
    Close,    // no more claims: the slot closed, and every worker inside waited for
};

// Every item of `job`, on this thread and any workers it wakes. Returns once
// no thread is inside the job; `job.failure` is then the caller's to report
inline void run_job(RangeJob& job) {
    // so `next` never wraps (claim_range): more items than this is no real job
    if (job.count > MAX_ITEMS) {
        fail_job(job, platform::failure<TooManyItems>());
        return;
    }
    SharedJob shared(job);
    const std::uint32_t cap = job.count > 1 ? worker_cap() : 0;
    policy::Ran mine;  // what this thread has run of the job
    std::uint32_t helpers = 0;
    std::uint32_t want = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
    CallerStep step = CallerStep::Claim;
    if (policy::share_at_once(job.item_cost, pool.spinning.load(std::memory_order_relaxed), cap)) {
        want = policy::helpers_wanted(job.item_cost, mine, job.count, cap);  // Coarse: all at once; else none (W1)
        step = CallerStep::Publish;
    }
    for (;;) {
        switch (step) {
            case CallerStep::Claim:
                step = claim_range(job, mine, "CallerClaim", &begin, &end) ? CallerStep::Run : CallerStep::Close;
                break;
            case CallerStep::Run:
                mine = run_range(job, mine, begin, end);
                step = CallerStep::Decide;
                break;
            case CallerStep::Decide: {
                poll_stop(job);  // outside the step's trace lock: a poll may take the GIL
                const TraceStep traced("CallerDecide");
                want = policy::helpers_wanted(job.item_cost, mine, items_left(job), cap);
                checked(want == 0 || job.item_cost == ItemCost::Coarse || mine.ns >= policy::WAKE_COST_NS,
                        "W1: a call wanted a helper before its caller ran items for C");
                traced.took(want);
                step = want > helpers ? CallerStep::Publish : CallerStep::Claim;
                break;
            }
            case CallerStep::Publish:
                step = CallerStep::Wake;
                if (!shared.published() && !shared.publish()) {
                    step = CallerStep::Claim;  // every slot busy: the next range tries again
                }
                break;
            case CallerStep::Wake:
                // want may be none: a warm publish, which only bumps the wake
                job.item_cost_16.store(static_cast<std::uint64_t>(mine.item_ns * 16.0), std::memory_order_relaxed);
                job.sharing.store(want + 1, std::memory_order_relaxed);
                wake_helpers(want, want - helpers);
                helpers = want;
                step = CallerStep::Claim;
                break;
            case CallerStep::Close: {
                shared.release(/*poll=*/true);
                const TraceStep traced("CallerCloseReturn");
                if (job.forked.load(std::memory_order_relaxed) && !job.failure) {
                    job.failure = platform::failure<ForkedDuringCall>();  // T5
                }
                // stopped at exit before every item was claimed: there is no answer
                if (!job.failure && job.next.load(std::memory_order_relaxed) < job.count && exit_gate_closed()) {
                    job.failure = platform::failure<StoppedAtExit>();
                }
                return;
            }
        }
    }
}

// `body(begin, end)` over every index in [0, count), on this thread and any
// workers the work is worth: see the top of this file for what it promises.
// Returns the first failure, if anything failed (not a Stop's stop: T4)
template <typename Body>
platform::Failure for_each_range(std::size_t count, const Body& body, ItemCost cost, std::size_t min_range, Stop stop) {
    RangeJob job;
    job.count = count;
    job.body = &body;
    job.run = [](const void* context, std::size_t begin, std::size_t end) {
        (*static_cast<const Body*>(context))(begin, end);
    };
    job.item_cost = cost;
    job.min_range = least(count, most<std::size_t>(1, min_range));  // no claim is ever past `count`
    job.stop_by = stop;
    if (count > 0) {
        run_job(job);
    }
    return std::move(job.failure);
}

// -- Two more things the extension shares between threads ---------------------

// One holder at a time of `word`, for what takes exactly one (a device's
// queue, src/vulkan/scene.hpp, which states the rule its users keep): a
// waiter sleeps on `word` rather than spinning. A holder takes no other lock
// while it holds this, the GIL included, and no body takes it (its holder may
// be waiting for that body's job), so there is nothing to deadlock on; never
// nested. Checked builds check the last two
class Exclusively {
public:
    explicit Exclusively(std::atomic<std::uint32_t>& word) : word_(word) {
        checked(!holding_exclusively, "Exclusively is never nested");
        checked(bodies_running == 0, "no body takes Exclusively");
        // acquire: pairs with the last holder's release. A compare-exchange
        // from 0, so a waiter only reads a held word, never writes it
        for (std::uint32_t free = 0;
             !word_.compare_exchange_strong(free, 1, std::memory_order_acquire, std::memory_order_relaxed); free = 0) {
            platform::wait_while_equal(word_, 1, 1000);
        }
        if constexpr (CHECKED) {
            holding_exclusively = true;
        }
    }
    ~Exclusively() {
        if constexpr (CHECKED) {
            holding_exclusively = false;
        }
        word_.store(0, std::memory_order_release);  // pairs with the next holder's acquire
        platform::wake_all(word_);
    }
    Exclusively(const Exclusively&) = delete;
    Exclusively& operator=(const Exclusively&) = delete;

private:
    std::atomic<std::uint32_t>& word_;
};

// A list threads push onto while a job runs, and whose caller takes it whole
// once the job returns: every node pushed is in it exactly once, and what a
// pusher wrote into its node before pushing is seen by whoever takes it.
// `Node` has a `Node* next`
template <typename Node>
class ChunkList {
public:
    // `node` onto the list, from any thread
    void push(Node* node) {
        node->next = head_.load(std::memory_order_relaxed);
        // release: pairs with take_all's acquire, publishing the node's contents
        while (!head_.compare_exchange_strong(node->next, node, std::memory_order_release, std::memory_order_relaxed)) {
        }
    }

    // every node pushed, newest first, from one thread once no one pushes
    Node* take_all() { return head_.exchange(nullptr, std::memory_order_acquire); }

    // the list as it is, newest first, without taking it
    Node* peek() const { return head_.load(std::memory_order_acquire); }

private:
    std::atomic<Node*> head_{nullptr};
};

}  // namespace trimkit::parallel
