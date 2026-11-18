// The pool as it runs, logged for check/verify/tla/SchedulerTrace.tla to replay
// against the model. Built with TRIMKIT_TRACE (`uv run scripts model`):
// `trace <callers> <calls> <items> <round> <TraceLog.tla>` runs, on each of
// `callers` threads, `calls` calls of `items` items each on the real pool,
// round after round a mix of Coarse calls, Measured calls long enough to be
// shared and ones too short to be (round 0 then waits until its workers exit
// idle, and calls once more), then writes every step the pool took as
// TraceLog.tla, and prints the constants the model needs to replay it
#define TRIMKIT_TRACE 1
#include "parallel.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace parallel = trimkit::parallel;

namespace {

// about `ns` of work, so workers have something to wake for
void work(std::int64_t ns) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
    while (std::chrono::steady_clock::now() < until) {
    }
}

void call(std::size_t items, int kind) {
    std::vector<std::atomic<int>> ran(items);
    const std::int64_t each = kind == 2 ? 100 : 100'000;  // the last kind is too short to share
    const auto failure = parallel::for_each_range(
        items,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; i++) {
                work(each);
                ran[i].fetch_add(1, std::memory_order_relaxed);
            }
        },
        kind == 0 ? parallel::ItemCost::Coarse : parallel::ItemCost::Measured);
    for (auto& count : ran) {
        if (failure || count.load() != 1) {
            std::fprintf(stderr, "a call didn't run every item once\n");
            std::exit(2);
        }
    }
}

void print_log(std::FILE* out) {
    std::fprintf(out, "---- MODULE TraceLog ----\n\\* Written by check/verify/tla/trace.cpp from a traced run.\nLog == <<\n");
    for (std::size_t i = 0; i < parallel::trace_log.size; i++) {
        const parallel::TraceRecord& r = parallel::trace_log.records[i];
        std::fprintf(out, "%s[thread |-> %u, action |-> \"%s\", arg |-> <<%llu, %llu>>, word |-> <<", i == 0 ? "  " : ", ",
                     r.thread, r.action, static_cast<unsigned long long>(r.arg[0]),
                     static_cast<unsigned long long>(r.arg[1]));
        for (std::size_t s = 0; s < parallel::SLOTS; s++) {
            const parallel::SlotWord w{r.word[s]};
            std::fprintf(out, "%s<<%u, %s, %llu>>", s == 0 ? "" : ", ", w.epoch(), w.open() ? "TRUE" : "FALSE",
                         static_cast<unsigned long long>(w.active()));
        }
        std::fprintf(out, ">>, left |-> <<");
        for (std::size_t s = 0; s < parallel::SLOTS; s++) {
            std::fprintf(out, "%s%u", s == 0 ? "" : ", ", r.left[s]);
        }
        std::fprintf(out, ">>, wake |-> %u, spinning |-> %u, live |-> %u]\n", r.wake, r.spinning, r.live);
    }
    std::fprintf(out, ">>\n====\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: trace <callers> <calls> <items> <round> <TraceLog.tla>\n");
        return 1;
    }
    const auto callers = static_cast<std::uint32_t>(std::atoi(argv[1]));
    const int calls = std::atoi(argv[2]);
    const auto items = static_cast<std::size_t>(std::atoi(argv[3]));
    const int round = std::atoi(argv[4]);
    std::vector<parallel::TraceRecord> records(std::size_t{1} << 20);
    parallel::trace_log.records = records.data();
    parallel::trace_log.capacity = records.size();
    parallel::trace_log.callers = callers;  // workers number themselves after the callers
    std::vector<std::thread> threads;
    for (std::uint32_t c = 0; c < callers; c++) {
        threads.emplace_back([=] {
            parallel::trace_thread = c + 1;
            for (int k = 0; k < calls; k++) {
                call(items, static_cast<int>((c + static_cast<std::uint32_t>(k + round)) % 3));
            }
            // round 0 waits past IDLE_EXIT_MS, so its workers exit idle, and
            // calls again, so their replacements start
            if (round == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(parallel::IDLE_EXIT_MS + 500));
                call(items, 0);
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    // the workers may still be sleeping: no step of theirs lands while this
    // writes, as each takes the lock this now holds for good
    while (parallel::trace_log.locked.exchange(true)) {
    }
    if (parallel::trace_log.size == parallel::trace_log.capacity) {
        std::fprintf(stderr, "the log filled\n");
        return 2;
    }
    std::FILE* out = std::fopen(argv[5], "w");
    if (out == nullptr) {
        return 1;
    }
    print_log(out);
    std::fclose(out);
    std::printf("Callers = %u\nCalls = %d\nItems = %zu\nCap = %u\nNSlots = %zu\nrecords = %zu\n", callers, calls + (round == 0 ? 1 : 0), items,
                parallel::worker_cap(), parallel::SLOTS, parallel::trace_log.size);
    return 0;
}
