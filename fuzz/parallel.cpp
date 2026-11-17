// Property: src/parallel.hpp's run_job, on any number of items of any
// uneven cost, from up to four threads at once, with items that throw, call
// run_job themselves, yield or sleep, and a poll that stops the job: runs
// every item exactly once unless something stopped it, and then each at most
// once; keeps the first exception; and returns only once no worker can touch
// the job again, which lives on the caller's stack, so AddressSanitizer
// reports any later touch as a use after return (verify/tla checks the same
// machine in every interleaving, and verify/genmc this header in every
// execution, both at small sizes; this runs the real one at real sizes).
// the pool's counters, so a check can say what it did (src/parallel.hpp)
#define TRIMKIT_PARALLEL_COUNTERS 1
#include "../src/parallel.hpp"

#include <fuzzer/FuzzedDataProvider.h>
#include <pthread.h>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
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
        std::fprintf(stderr, "fuzz/parallel.cpp: %s\n", what);
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

// The scheduling policy's rules, as the header states them (helpers_wanted,
// claim_items, RangeCost): pure functions, checked over a grid of inputs
void check_policy() {
    using trimkit::parallel::claim_items;
    using trimkit::parallel::GRAIN_NS;
    using trimkit::parallel::helpers_wanted;
    using trimkit::parallel::MIN_CLAIM_NS;
    for (const double wake : {5'000.0, 20'000.0, 200'000.0}) {
        for (const std::uint32_t cap : {0U, 1U, 7U, 31U}) {
            std::uint32_t before = 0;
            for (double left = 0; left < 100 * wake; left += wake / 4) {
                const std::uint32_t want = helpers_wanted(left, wake, cap);
                expect(left >= 2 * wake || want == 0, "helpers are wanted for less than two wakes' worth of work");
                expect(want <= cap, "more helpers wanted than the cap");
                expect(want >= before, "more work wants fewer helpers");
                before = want;
            }
        }
    }
    for (const double item : {0.0, 1.0, 50.0, 1'000.0, 1e6}) {
        for (const std::size_t left : {1UL, 2UL, 7UL, 1'000UL, 1'000'000UL}) {
            for (const std::uint32_t sharing : {1U, 2U, 8U}) {
                for (const double claim : {MIN_CLAIM_NS, GRAIN_NS}) {
                    const std::size_t take = claim_items(claim, item, left, sharing);
                    expect(take >= 1, "a claim of no items");
                    expect(take <= std::max<std::size_t>(1, left / (2 * sharing)),
                           "a claim of more than half an even share of what is left");
                    expect(item > 0 || take == 1, "more than one item claimed before any is measured");
                }
            }
        }
    }
    trimkit::parallel::RangeCost cost;
    cost.measured(1'000, 1);
    expect(cost.item_ns == 1'000, "the first range's cost isn't the estimate");
    cost.measured(100'000, 1);
    expect(cost.item_ns == 100'000, "a range 100 times dearer didn't replace the estimate");
    for (int i = 0; i < 20; i++) {
        cost.measured(cost.item_ns, 1);
    }
    expect(cost.claim_ns == GRAIN_NS, "claims don't grow to GRAIN_NS");
}

// A call worth less than two wakes wakes no one: the bound the header states
// for small calls, read from the pool's counters. Worth as measured: each
// round first runs the same items on this thread, and counts only when they
// took between two probes' time (so the probe ends and the pool decides,
// not merely runs out of items first) and half of W (a margin of four for
// the estimate). And at rest no more workers are live than the cap allows
void check_small_calls_wake_no_one() {
    auto none = [](std::size_t) {};
    int counted = 0;
    for (int round = 0; round < 200 && counted < 50; round++) {
        Spun<decltype(none)> spun(16, 300, none);
        trimkit::parallel::RangeJob alone;
        spun.fill(alone);
        const std::int64_t start = trimkit::platform::monotonic_ns();
        alone.run(alone.body, 0, alone.count);
        const auto took = static_cast<double>(trimkit::platform::monotonic_ns() - start);
        Spun<decltype(none)> again(16, 300, none);
        trimkit::parallel::RangeJob job;
        again.fill(job);
        const std::uint64_t woken = trimkit::parallel::counters.wakes.load();
        trimkit::parallel::run_job(job);
        expect(again.ran() == again.count, "a small call didn't run every item");
        if (took >= 2 * trimkit::parallel::PROBE_NS && took < trimkit::parallel::wake_ns() / 2) {
            counted++;
            expect(trimkit::parallel::counters.wakes.load() == woken, "a call worth less than two wakes woke a worker");
        }
    }
    expect(counted >= 50, "too few calls were small enough to check");
    expect(trimkit::parallel::pool.live.load() <= trimkit::parallel::worker_cap() + 1,
           "more workers live than the cap allows");
}

}  // namespace

// A min_range far past the count, with workers spinning so the job is shared
// at once: no claim may wrap `next` round to run an item again (it once did)
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
