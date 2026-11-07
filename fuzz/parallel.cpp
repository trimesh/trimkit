// Property: src/parallel.hpp's run_job, on any number of items of any
// uneven cost, from up to four threads at once, with items that throw, call
// run_job themselves, yield or sleep, and a poll that stops the job: runs
// every item exactly once unless something stopped it, and then each at most
// once; keeps the first exception; and returns only once no worker can touch
// the job again, which lives on the caller's stack, so AddressSanitizer
// reports any later touch as a use after return (tests/test_scheduler.py
// checks the same machine in every interleaving; this runs the real one).
#include "../src/parallel.hpp"

#include <fuzzer/FuzzedDataProvider.h>
#include <sched.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
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
};

struct Run {
    const Plan* plan = nullptr;
    std::vector<std::atomic<std::uint32_t>> runs;
    std::atomic<std::uint32_t> polls{0};
    explicit Run(const Plan& p) : plan(&p), runs(p.count) {}
};

void spin(std::uint32_t ns) {
    const std::int64_t until = trimkit::parallel::now_ns() + ns;
    while (trimkit::parallel::now_ns() < until) {
    }
}

void check(Run& run, const trimkit::parallel::Job& job, bool nested_ok);

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
            trimkit::parallel::Job job;
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
    return run.plan->stop_after_polls != 0 && run.polls.fetch_add(1) + 1 >= run.plan->stop_after_polls;
}

void check(Run& run, const trimkit::parallel::Job& job, bool stoppable) {
    const bool stopped = job.stop.load();
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
    if (threw != static_cast<bool>(job.error)) {
        __builtin_trap();
    }
}

void call(const Plan& plan) {
    Run run(plan);
    trimkit::parallel::Job job;
    job.count = plan.count;
    job.run = run_items;
    job.body = &run;
    job.poll = poll;
    job.poll_context = &run;
    trimkit::parallel::run_job(job);
    check(run, job, true);
}

}  // namespace

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
    }
    std::vector<std::thread> threads;
    for (std::size_t c = 1; c < callers; c++) {
        threads.emplace_back(call, plans[c]);
    }
    call(plans[0]);
    for (std::thread& thread : threads) {
        thread.join();
    }
    return 0;
}
