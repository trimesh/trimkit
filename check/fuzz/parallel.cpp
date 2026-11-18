// Property: src/parallel.hpp's run_job, on any number of items of any
// uneven cost, from up to four threads at once, with items that throw, call
// run_job themselves, yield or sleep, and a poll that stops the job: runs
// every item exactly once unless something stopped it, and then each at most
// once; keeps the first exception; and returns only once no worker can touch
// the job again, which lives on the caller's stack, so AddressSanitizer
// reports any later touch as a use after return (check/verify/tla checks the same
// machine in every interleaving, and check/verify/genmc this header in every
// execution, both at small sizes; this runs the real one at real sizes).
// the pool's counters, so a check can say what it did (src/parallel.hpp)
#define TRIMKIT_PARALLEL_COUNTERS 1
#include "../../src/parallel.hpp"

#include <fuzzer/FuzzedDataProvider.h>
#include <pthread.h>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

extern "C" const char* __asan_default_options() { return "detect_stack_use_after_return=1"; }

namespace {

struct Plan {
    std::size_t count = 0;
    std::uint32_t spin_ns = 0;  // an item's cost
    std::uint32_t uneven = 0;   // every `uneven`-th item costs 100 times more
    std::size_t throw_at = SIZE_MAX;
    std::size_t nest_at = SIZE_MAX;
    std::size_t pause_at = SIZE_MAX;     // an item that sleeps, so a range stalls
    std::uint32_t stop_after_polls = 0;  // 0: never stop
    // 0: never; else the poll throws at this poll, as a thread exiting
    // inside it (CPython's take_gil during finalization) unwinds out of it
    std::uint32_t throw_at_poll = 0;
    trimkit::parallel::ItemCost cost = trimkit::parallel::ItemCost::Measured;
    std::size_t min_range = 1;
};

struct PollThrew {};

struct Run {
    const Plan* plan = nullptr;
    std::vector<std::atomic<std::uint32_t>> runs;
    std::atomic<std::uint32_t> polls{0};
    explicit Run(const Plan& p) : plan(&p), runs(p.count) {}
};

void spin(std::uint32_t ns) {
    const std::int64_t until = trimkit::platform::monotonic_ns() + ns;
    while (trimkit::platform::monotonic_ns() < until) {
    }
}

void check(Run& run, const trimkit::parallel::RangeJob& job, bool nested_ok);

void run_items(const void* context, std::size_t begin, std::size_t end) {
    Run& run = *static_cast<Run*>(
        const_cast<void*>(context));  // NOLINT(cppcoreguidelines-pro-type-const-cast): the run counts what ran
    const Plan& plan = *run.plan;
    for (std::size_t i = begin; i < end; i++) {
        run.runs[i].fetch_add(1, std::memory_order_relaxed);
        spin(plan.uneven != 0 && i % plan.uneven == 0 ? plan.spin_ns * 100 : plan.spin_ns);
        if (i == plan.pause_at) {
            const timespec pause{0, 200'000};
            nanosleep(&pause, nullptr);
        }
        if (i % 7 == 0) {
            sched_yield();
        }
        if (i == plan.nest_at) {
            Plan inner;
            inner.count = 300;
            inner.spin_ns = 1'000;
            Run nested(inner);
            trimkit::parallel::RangeJob job;
            job.count = inner.count;
            job.run = run_items;
            job.body = &nested;
            trimkit::parallel::run_job(job);
            check(nested, job, false);
        }
        if (i == plan.throw_at) {
            throw std::runtime_error("item");
        }
    }
}

bool poll(void* context) {
    Run& run = *static_cast<Run*>(context);
    const std::uint32_t polls = run.polls.fetch_add(1) + 1;
    if (run.plan->throw_at_poll != 0 && polls == run.plan->throw_at_poll) {
        throw PollThrew{};
    }
    return run.plan->stop_after_polls != 0 && polls >= run.plan->stop_after_polls;
}

void check(Run& run, const trimkit::parallel::RangeJob& job, bool stoppable) {
    const bool stopped = job.stopped.load();
    if (stopped && !stoppable && run.plan->throw_at == SIZE_MAX) {
        __builtin_trap();
    }
    for (std::size_t i = 0; i < run.plan->count; i++) {
        const std::uint32_t times = run.runs[i].load();
        if (times > 1 || (times == 0 && !stopped)) {
            __builtin_trap();
        }
    }
    const bool threw = run.plan->throw_at < run.plan->count && run.runs[run.plan->throw_at].load() == 1;
    if (threw != static_cast<bool>(job.failure)) {
        __builtin_trap();
    }
}

// the job in its own frame, which a throwing poll unwinds: nothing may touch
// it after, and AddressSanitizer reports a worker that does
[[gnu::noinline]] void run_in_frame(Run& run) {
    trimkit::parallel::RangeJob job;
    job.count = run.plan->count;
    job.run = run_items;
    job.body = &run;
    job.stop_by.poll = poll;
    job.stop_by.context = &run;
    job.item_cost = run.plan->cost;
    job.min_range = std::max<std::size_t>(1, std::min(run.plan->count, run.plan->min_range));  // as for_each_range
    trimkit::parallel::run_job(job);
    check(run, job, true);
}

void call(const Plan& plan) {
    Run run(plan);
    try {
        run_in_frame(run);
    } catch (const PollThrew&) {
        // unwound: every item ran at most once, and every range has ended
        for (std::size_t i = 0; i < plan.count; i++) {
            if (run.runs[i].load() > 1) {
                __builtin_trap();
            }
        }
    }
}

// with no call running, every slot is free: a call that returned or
// unwound left none open or counted
void check_slots_free() {
    for (const trimkit::parallel::JobSlot& slot : trimkit::parallel::pool.slots) {
        const std::uint64_t word = slot.word.load();
        if ((word & (trimkit::parallel::SlotWord::OPEN | trimkit::parallel::SlotWord::ACTIVE)) != 0) {
            __builtin_trap();
        }
    }
}

// -- fixed checks, run once before any fuzzing --------------------------------

void expect(bool kept, const char* what) {
    if (!kept) {
        std::fprintf(stderr, "check/fuzz/parallel.cpp: %s\n", what);
        __builtin_trap();
    }
}

// a job of `count` items of `ns` each, `body` called first on each item
template <typename Each>
struct Spun {
    std::size_t count;
    std::uint32_t ns;
    Each each;
    std::vector<std::atomic<std::uint32_t>> runs;
    Spun(std::size_t c, std::uint32_t n, Each e) : count(c), ns(n), each(e), runs(c) {}
    std::size_t ran() const {
        std::size_t total = 0;
        for (const auto& r : runs) {
            total += r.load();
        }
        return total;
    }
    void fill(trimkit::parallel::RangeJob& job) {
        job.count = count;
        job.body = this;
        job.run = [](const void* body, std::size_t begin, std::size_t end) {
            auto& spun = *static_cast<Spun*>(const_cast<void*>(body));  // NOLINT(cppcoreguidelines-pro-type-const-cast)
            for (std::size_t i = begin; i < end; i++) {
                spun.each(i);
                spun.runs[i].fetch_add(1);
                spin(spun.ns);
            }
        };
    }
};

// B3: a poll that throws stops the job, so the unwind waits only for the
// ranges already running, not for every item left
void check_throwing_poll_stops() {
    auto none = [](std::size_t) {};
    Spun<decltype(none)> spun(20'000, 10'000, none);
    std::uint32_t polls = 0;
    trimkit::parallel::RangeJob job;
    spun.fill(job);
    job.stop_by.poll = [](void* context) -> bool {
        if (++*static_cast<std::uint32_t*>(context) == 30) {  // workers are inside by then
            throw PollThrew{};
        }
        return false;
    };
    job.stop_by.context = &polls;
    try {
        trimkit::parallel::run_job(job);
    } catch (const PollThrew&) {
    }
    expect(spun.ran() < spun.count / 2, "a poll that threw left the job running to the end");
}

// B5b: a thread that ends mid-range (pthread_exit unwinds it on glibc) is
// never swallowed, which glibc aborts on: on the calling thread the call
// unwinds; on a worker the call fails, as that worker's items never ran;
// either way the pool goes on
void check_thread_ending_in_a_body() {
    std::thread caller([] {
        auto end_here = [](std::size_t i) {
            if (i == 5'000 && trimkit::parallel::thread_index() == 0) {
                pthread_exit(nullptr);
            }
        };
        Spun<decltype(end_here)> spun(20'000, 2'000, end_here);
        trimkit::parallel::RangeJob job;
        spun.fill(job);
        trimkit::parallel::run_job(job);
    });
    caller.join();
    auto end_a_worker = [](std::size_t i) {
        if (i >= 2'000 && trimkit::parallel::thread_index() != 0) {
            pthread_exit(nullptr);
        }
    };
    Spun<decltype(end_a_worker)> spun(20'000, 2'000, end_a_worker);
    trimkit::parallel::RangeJob job;
    spun.fill(job);
    trimkit::parallel::run_job(job);
    expect(spun.ran() == spun.count || job.failure != nullptr,
           "a worker ended mid-range and the call reported success");
    // and the pool still answers in full
    auto none = [](std::size_t) {};
    Spun<decltype(none)> after(20'000, 2'000, none);
    trimkit::parallel::RangeJob again;
    after.fill(again);
    trimkit::parallel::run_job(again);
    expect(after.ran() == after.count && again.failure == nullptr, "the pool doesn't answer after a thread ended");
}

// B5a: at the interpreter's exit every job stops at its next claim, and one
// that didn't run every item fails as StoppedAtExit, never a partial answer
void check_exit_stops_every_job() {
    auto none = [](std::size_t) {};
    Spun<decltype(none)> spun(20'000, 1'000, none);
    trimkit::parallel::RangeJob job;
    spun.fill(job);
    trimkit::parallel::exit_gate.closed.store(true);
    trimkit::parallel::run_job(job);
    trimkit::parallel::exit_gate.closed.store(false);
    bool stopped = false;
    try {
        if (job.failure) {
            std::rethrow_exception(job.failure);
        }
    } catch (const trimkit::parallel::StoppedAtExit&) {
        stopped = true;
    }
    expect(stopped ? spun.ran() < spun.count : spun.ran() == spun.count,
           "a job stopped at exit returned without saying so");
    expect(stopped, "a job started at exit ran anyway");
}

// B7: a fork marks a held, closed slot's call as failed only if workers of
// that very call are inside; one another caller published since isn't it.
// Run before any job, so there are no workers for the handler to forget
void check_fork_keeps_epochs() {
    constexpr std::uint64_t EPOCH = trimkit::parallel::SlotWord::EPOCH;
    constexpr std::uint64_t OPEN = trimkit::parallel::SlotWord::OPEN;
    trimkit::parallel::JobSlot& slot = trimkit::parallel::pool.slots[0];
    const std::uint64_t epoch = 7;
    trimkit::parallel::RangeJob republished;
    // closed at epoch 7; another caller has epoch 8 open, a worker inside
    slot.word.store(((epoch + 1) * EPOCH) | OPEN | 2);
    trimkit::parallel::HeldSlot held{&slot, &republished, true, epoch, nullptr};
    trimkit::parallel::held_slots = &held;
    trimkit::parallel::child_after_fork();
    expect(!republished.forked.load(), "a fork failed a call whose slot another call had published since");
    trimkit::parallel::RangeJob inside;
    // closed at epoch 7, one of its own workers still inside
    slot.word.store(epoch * EPOCH | 1);
    held = trimkit::parallel::HeldSlot{&slot, &inside, true, epoch, nullptr};
    trimkit::parallel::child_after_fork();
    trimkit::parallel::held_slots = nullptr;
    expect(inside.forked.load(), "a fork left a call that lost a worker's items to answer");
    slot.word.store(0);
}

// B6: a body forks on a worker: the child's one thread is that worker, inside
// a job whose caller is gone. The child counts it live, lets it leave its
// job (the slot closed with it alone counted), stops that job, and runs a
// call in full; it exits 0 only if all of that held
int check_child_of_a_worker(trimkit::parallel::RangeJob* left) {
    constexpr std::uint64_t ACTIVE = trimkit::parallel::SlotWord::ACTIVE;
    constexpr std::uint64_t OPEN = trimkit::parallel::SlotWord::OPEN;
    if (trimkit::parallel::pool.live.load() != 1) {
        return 1;
    }
    std::uint32_t closed_with_one = 0;
    for (const trimkit::parallel::JobSlot& slot : trimkit::parallel::pool.slots) {
        const std::uint64_t word = slot.word.load();
        closed_with_one += (word & (OPEN | ACTIVE)) == 1 ? 1 : 0;
    }
    if (closed_with_one != 1 || !left->stopped.load()) {
        return 2;
    }
    auto none = [](std::size_t) {};
    Spun<decltype(none)> spun(10'000, 1'000, none);
    trimkit::parallel::RangeJob job;
    spun.fill(job);
    trimkit::parallel::run_job(job);
    return spun.ran() == spun.count && job.failure == nullptr ? 0 : 3;
}

void check_fork_from_a_worker() {
    struct Forked {
        std::atomic<bool> once{false};
        int status = -1;
        trimkit::parallel::RangeJob* job = nullptr;
    } forked;
    auto fork_on_a_worker = [&forked](std::size_t) {
        if (trimkit::parallel::thread_index() == 0 || forked.once.exchange(true)) {
            return;
        }
        const pid_t pid = fork();
        if (pid == 0) {
            _exit(check_child_of_a_worker(forked.job));
        }
        waitpid(pid, &forked.status, 0);
    };
    Spun<decltype(fork_on_a_worker)> spun(20'000, 2'000, fork_on_a_worker);
    trimkit::parallel::RangeJob job;
    spun.fill(job);
    forked.job = &job;
    trimkit::parallel::run_job(job);
    expect(forked.once.load(), "no worker ran an item, so nothing forked");
    expect(WIFEXITED(forked.status) && WEXITSTATUS(forked.status) == 0,
           "a child forked from a worker found the pool broken");
}

// The scheduling policy (src/parallel.hpp, namespace policy): pure functions,
// checked over grids of inputs, W1 and W2 exactly; and a simulation of claim
// sizing on clocks of 1, 41 and 100 ns ticks, with no stealing, so no claim
// may run much past its aim
void check_policy() {
    namespace policy = trimkit::parallel::policy;
    using policy::Ran;
    using policy::WAKE_COST_NS;
    using trimkit::parallel::ItemCost;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (const std::uint32_t cap : {0U, 1U, 7U, 31U, 255U}) {
        for (const std::size_t left : {0UL, 1UL, 2UL, 3UL, 1'000UL, 1'000'000UL}) {
            const auto room = static_cast<std::uint32_t>(std::min<std::size_t>(cap, left > 0 ? left - 1 : 0));
            expect(policy::helpers_wanted(ItemCost::Coarse, Ran{}, left, cap) == room, "Coarse wants all it can use");
            for (const std::size_t items : {1UL, 3UL, 100UL, 1'000'000UL}) {
                std::uint32_t before = 0;
                for (int k = 0; k <= 300; k++) {
                    for (const double nudge : {-1.0, 0.0, 1.0}) {
                        const double ns = (k * WAKE_COST_NS) + nudge;
                        const std::uint32_t want =
                            policy::helpers_wanted(ItemCost::Measured, Ran{ns, items, 0}, left, cap);
                        expect(ns >= WAKE_COST_NS || want == 0, "W1: helpers wanted before the caller ran items for C");
                        const double work_left = ns / static_cast<double>(items) * static_cast<double>(left);
                        const double formula =
                            std::min(static_cast<double>(room), std::floor(work_left / (2 * WAKE_COST_NS)));
                        expect(ns < WAKE_COST_NS || static_cast<double>(want) == formula, "W2: not one helper per 2C");
                        expect(static_cast<double>(want) * 2 * WAKE_COST_NS <= std::max(0.0, work_left),
                               "W2: helpers cost more than the work they share");
                        if (nudge == 0.0) {
                            expect(want >= before, "more time run wants fewer helpers");
                            before = want;
                        }
                    }
                }
            }
            for (const double odd : {-1e9, -0.0, nan, inf, -inf}) {
                expect(
                    policy::helpers_wanted(ItemCost::Measured, Ran{odd, 10, 0}, left, cap) == (odd == inf ? room : 0),
                    "a clock gone wrong wanted helpers");
            }
            expect(policy::helpers_wanted(ItemCost::Measured, Ran{1e9, 0, 0}, left, cap) == 0,
                   "helpers wanted for a caller that ran no items");
        }
    }
    expect(policy::spin_ns() <= WAKE_COST_NS, "W3: idle workers spin longer than C");
    expect(!policy::share_at_once(ItemCost::Measured, 0, 7) && policy::share_at_once(ItemCost::Measured, 1, 7) &&
               policy::share_at_once(ItemCost::Coarse, 0, 7) && !policy::share_at_once(ItemCost::Coarse, 3, 0),
           "a call shares at once only Coarse or with workers spinning, and with a cap");
    for (const double item_ns : {0.0, 1e-3, 1.0, 50.0, 1e4, 1e9, nan, inf}) {
        for (const double ns : {0.0, 1e3, 1e5, 1e9}) {
            for (const std::size_t left : {0UL, 1UL, 7UL, 1'000UL, 1'000'000UL}) {
                for (const std::uint32_t sharing : {0U, 1U, 2U, 8U}) {
                    for (const std::size_t min_range : {1UL, 5UL}) {
                        const std::size_t take = policy::claim_items(Ran{ns, 1, item_ns}, left, sharing, min_range);
                        expect(take >= min_range, "C1: a claim smaller than min_range");
                        expect(take <= std::max<std::size_t>(min_range, left / (2 * std::max(1U, sharing))),
                               "C1: a claim of more than half an even share of what is left");
                        if (item_ns > 0 && item_ns < inf) {
                            expect(static_cast<double>(take) <=
                                       std::max(static_cast<double>(min_range), policy::MAX_CLAIM_NS / item_ns + 1),
                                   "C1: a claim aimed past MAX_CLAIM_NS");
                        }
                    }
                }
            }
        }
    }
    for (const double took : {nan, -1e9, -1.0}) {
        const Ran after = policy::after_range(Ran{1e3, 4, 5.0}, took, 4);
        expect(after.ns == 1e3 && after.items == 8 && after.item_ns == policy::TIMED_NS / 4,
               "a range timed wrong changed the time run, or claimed no floor");
    }
    // the simulation: one thread, no stealing, every range timed on a clock of
    // `tick` ns, so a short one reads as nothing. Each claim's true time is at
    // most 1.1 times its aim or is one item; claims reach half the aim within
    // 40 ranges; and after the items turn 20 times dearer, the next claim but
    // one is back within its aim
    for (const double cost : {0.5, 5.0, 50.0, 500.0, 5e3, 5e5}) {
        for (const double tick : {1.0, 41.0, 100.0}) {
            Ran ran;
            double each = cost;
            bool reached = false;
            for (int range = 0; range < 80; range++) {
                if (range == 60) {
                    each = cost * 20;
                }
                const double aim = std::min(policy::MAX_CLAIM_NS, std::max(policy::MIN_CLAIM_NS, ran.ns));
                const std::size_t take = policy::claim_items(ran, 1'000'000'000, 1, 1);
                const double truth = static_cast<double>(take) * each;
                if (range != 60) {  // the claim sized before the change is the one that may run long
                    expect(take == 1 || truth <= 1.1 * aim, "a claim ran past its aim");
                }
                reached = reached || truth >= aim / 2;
                if (range == 40) {
                    expect(reached || take == 1, "claims never grew toward their aim");
                }
                ran = policy::after_range(ran, std::floor(truth / tick) * tick, take);
            }
        }
    }
}

// W1 on real calls: a Measured call whose caller spent less than C on it
// wakes no sleeping worker and starts none, read from the pool's counters. Timed around
// run_job, which times what it spent from inside, so a call timed under C
// here spent under C there: no margin. Workers may be spinning from an
// earlier call and help, which W1 allows, as they cost no wake. And at rest
// no more workers are live than the cap allows
void check_small_calls_wake_no_one() {
    auto none = [](std::size_t) {};
    int counted = 0;
    for (int round = 0; round < 2'000 && counted < 200; round++) {
        Spun<decltype(none)> spun(16, round % 2 == 0 ? 100 : 250, none);
        trimkit::parallel::RangeJob job;
        spun.fill(job);
        const std::uint64_t roused =
            trimkit::parallel::counters.wakes.load() + trimkit::parallel::counters.spawns.load();
        const std::int64_t start = trimkit::platform::monotonic_ns();
        trimkit::parallel::run_job(job);
        const auto took = static_cast<double>(trimkit::platform::monotonic_ns() - start);
        expect(spun.ran() == spun.count, "a small call didn't run every item");
        if (took < trimkit::parallel::policy::WAKE_COST_NS) {
            counted++;
            expect(trimkit::parallel::counters.wakes.load() + trimkit::parallel::counters.spawns.load() == roused,
                   "W1: a call that spent less than C woke or started a worker");
        }
    }
    expect(counted >= 200, "too few calls were small enough to check");
    expect(trimkit::parallel::pool.live.load() <= trimkit::parallel::worker_cap() + 1,
           "more workers live than the cap allows");
}

}  // namespace

// A min_range far past the count, Coarse so the job is shared at once: no
// claim may wrap `next` round to run an item again
void check_huge_min_range() {
    for (int round = 0; round < 200; round++) {
        std::atomic<int> ran[3] = {};
        const auto failure = trimkit::parallel::for_each_range(
            3,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; i++) {
                    ran[i].fetch_add(1, std::memory_order_relaxed);
                }
            },
            trimkit::parallel::ItemCost::Coarse, std::size_t{1} << 63);
        expect(!failure, "a call with a huge min_range failed");
        for (auto& count : ran) {
            expect(count.load() == 1, "a huge min_range ran an item twice, or not at all");
        }
    }
}

// More than MAX_ITEMS fails at once with TooManyItems, running nothing: so
// `next` can't wrap (claim_range). Where size_t is 32 bits no count is more,
// so every count runs: the largest stops at its first range's throw
void check_too_many_items() {
    std::atomic<int> ran{0};
    const bool narrow = sizeof(std::size_t) < sizeof(std::uint64_t);
    const std::size_t count = narrow ? SIZE_MAX : static_cast<std::size_t>(trimkit::parallel::MAX_ITEMS + 1);
    const auto failure = trimkit::parallel::for_each_range(count, [&](std::size_t, std::size_t) {
        ran.fetch_add(1);
        throw std::runtime_error("first range");
    });
    bool too_many = false;
    try {
        std::rethrow_exception(failure);
    } catch (const trimkit::parallel::TooManyItems&) {
        too_many = true;
    } catch (...) {
    }
    if (narrow) {
        expect(!too_many && ran.load() > 0, "a 32-bit call of SIZE_MAX items was refused");
    } else {
        expect(too_many && ran.load() == 0, "a call of more than MAX_ITEMS ran, or failed some other way");
    }
}

extern "C" int LLVMFuzzerInitialize(int* /*argc*/, char*** /*argv*/) {
    check_fork_keeps_epochs();                  // first: before any worker exists
    trimkit::parallel::install_fork_handler();  // as the module's init does
    check_policy();
    check_throwing_poll_stops();
    check_thread_ending_in_a_body();
    check_exit_stops_every_job();
    check_fork_from_a_worker();
    check_small_calls_wake_no_one();
    check_huge_min_range();
    check_too_many_items();
    check_slots_free();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider input(data, size);
    const auto callers = input.ConsumeIntegralInRange<std::size_t>(1, 4);
    std::vector<Plan> plans(callers);
    for (Plan& plan : plans) {
        plan.count = input.ConsumeIntegralInRange<std::size_t>(0, 4'000);
        plan.spin_ns = input.ConsumeIntegralInRange<std::uint32_t>(0, 1'000);
        plan.uneven = input.ConsumeIntegralInRange<std::uint32_t>(0, 50);
        if (plan.count > 0 && input.ConsumeBool()) {
            plan.throw_at = input.ConsumeIntegralInRange<std::size_t>(0, plan.count - 1);
        }
        if (plan.count > 0 && input.ConsumeBool()) {
            plan.nest_at = input.ConsumeIntegralInRange<std::size_t>(0, plan.count - 1);
        }
        if (plan.count > 0 && input.ConsumeBool()) {
            plan.pause_at = input.ConsumeIntegralInRange<std::size_t>(0, plan.count - 1);
        }
        if (input.ConsumeBool()) {
            plan.stop_after_polls = input.ConsumeIntegralInRange<std::uint32_t>(1, 64);
        }
        if (input.ConsumeBool()) {
            plan.throw_at_poll = input.ConsumeIntegralInRange<std::uint32_t>(1, 64);
        }
        if (input.ConsumeBool()) {
            plan.cost = trimkit::parallel::ItemCost::Coarse;
        }
        plan.min_range = input.ConsumeIntegralInRange<std::size_t>(1, 64);
    }
    std::vector<std::thread> threads;
    for (std::size_t c = 1; c < callers; c++) {
        threads.emplace_back(call, plans[c]);
    }
    call(plans[0]);
    for (std::thread& thread : threads) {
        thread.join();
    }
    check_slots_free();
    return 0;
}
