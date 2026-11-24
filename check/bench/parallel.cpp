// src/parallel.hpp against what people write by hand, on the same work.
//
//   serial      the plain loop
//   pool        parallel.hpp's run_job
//   spawn       a std::thread per core started every call, an even split, joined
//   cv_static   a persistent pool (mutex and condition_variable), an even split
//   cv_chunks   the same pool, threads taking fixed chunks (count / (8 * threads))
//               from one atomic counter
//
// Every item spins for a calibrated cost, and each cell's costs are one of:
// even; scattered, heavy items at pseudo-random places in pseudo-random
// clusters and of pseudo-random weight from 2 to 128 times, small ones far
// likelier (no period or size a branch predictor, a prefetcher or a policy
// tuned by measuring could learn); or the first eighth 20 times (a
// front-loaded mesh). `callers` threads make the same call at once, each its
// own job; `noise` threads of another process spin the whole time, as other
// work on a busy machine would. The ways take turns within each of 7 rounds,
// so a machine whose load drifts slows them alike. Each cell prints one JSON
// line: each way's best wall time and the CPU time that run took, in
// microseconds (this process's alone, read once its threads have settled, so
// a pool's spinning after the work counts against it), how many threads ran
// the pool's items and what the pool did for them (publishes, wakes, sleeps,
// spawns, workers woken, time spent spinning and wakes lost: built with
// TRIMKIT_PARALLEL_COUNTERS), and the costliest single item, which no dynamic
// schedule can promise to beat at the end of a job, and C, what the pool's
// policy takes a wake-up to cost (src/parallel.hpp, policy::WAKE_COST_NS),
// which `bench` holds against what a wake-up cost here.
// Every item is checked to run exactly once.
//
//   check/bench/parallel [threads] [callers] [noise] [most serial ns] [only: items,item_ns,skew]
//
// Cells whose serial work would pass `most serial ns` (2 s by default) are skipped.
//
// `uv run scripts bench` builds it and gates on it (scripts/__main__.py).
#ifndef TRIMKIT_PARALLEL_COUNTERS
#define TRIMKIT_PARALLEL_COUNTERS 1
#endif
#include "../../src/parallel.hpp"

#include <signal.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

double spins_per_ns = 1.0;

[[gnu::noinline]] std::uint64_t spin(std::uint64_t iterations) {
    std::uint64_t x = iterations | 1;
    for (std::uint64_t i = 0; i < iterations; i++) {
        x = (x * 6364136223846793005ULL) + 1442695040888963407ULL;
    }
    return x;
}

void calibrate() {
    constexpr std::uint64_t ITERATIONS = 50'000'000;
    const auto start = std::chrono::steady_clock::now();
    volatile std::uint64_t sink = spin(ITERATIONS);
    (void)sink;
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    spins_per_ns = static_cast<double>(ITERATIONS) / ns;
}

enum class Skew : std::uint8_t { Even, Scattered, FrontLoaded };
constexpr const char* SKEW_NAMES[] = {"even", "random 2-128x", "first 1/8 20x"};

// splitmix64, whose multiplies wrap by design (the overflow builtins say so)
std::uint64_t splitmix(std::uint64_t z) {
    std::uint64_t product = 0;
    __builtin_add_overflow(z, std::uint64_t{0x9e3779b97f4a7c15ULL}, &z);
    __builtin_mul_overflow(z ^ (z >> 30U), std::uint64_t{0xbf58476d1ce4e5b9ULL}, &product);
    __builtin_mul_overflow(product ^ (product >> 27U), std::uint64_t{0x94d049bb133111ebULL}, &product);
    return product ^ (product >> 31U);
}

// What each item costs, in spins, decided once per cell, so the timed loop is
// the same lookup whatever the skew. Scattered: each block of 256 items has a
// density, none, 1/256, 1/64 or 1/16 alike (so heavy items come in clusters
// and gaps, as a mesh's dense regions do), and each heavy item a weight of
// 2**k for k from 1 to 7 alike (2 times as often as 128): the same items,
// pseudo-random by index, for every way and round
std::vector<std::uint64_t> item_spins(std::size_t count, double item_ns, Skew skew) {
    constexpr std::uint64_t DENSITY_PER_4096[] = {0, 16, 64, 256};
    std::vector<std::uint64_t> spins(count);
    for (std::size_t i = 0; i < count; i++) {
        double weight = 1.0;
        if (skew == Skew::Scattered) {
            const std::uint64_t block = splitmix(i / 256);
            const std::uint64_t item = splitmix(~std::uint64_t{i});
            if (item % 4096 < DENSITY_PER_4096[block % 4]) {
                weight = std::ldexp(1.0, 1 + static_cast<int>((item >> 12U) % 7));
            }
        }
        if (skew == Skew::FrontLoaded && i < count / 8) {
            weight = 20.0;
        }
        spins[i] = static_cast<std::uint64_t>(weight * item_ns * spins_per_ns);
    }
    return spins;
}

struct Work {
    std::size_t count = 0;
    double item_ns = 0;
    Skew skew = Skew::Even;
    const std::vector<std::uint64_t>* spins = nullptr;  // each item's cost (item_spins)
    std::vector<std::atomic<std::uint32_t>>* ran = nullptr;

    double costliest() const {
        return spins->empty() ? 0.0 : static_cast<double>(*std::max_element(spins->begin(), spins->end())) / spins_per_ns;
    }

    // the pool's thread indexes that ran an item, a bit each (index 63 and up share one)
    std::atomic<std::uint64_t>* indexes = nullptr;

    void run(std::size_t begin, std::size_t end) const {
        std::uint64_t sink = 0;
        for (std::size_t i = begin; i < end; i++) {
            sink += spin((*spins)[i]);
            (*ran)[i].fetch_add(1, std::memory_order_relaxed);
        }
        // which of the pool's threads ran items: written once a thread, so the
        // pool pays no contended write a range that the hand-written ways don't
        if (indexes != nullptr) {
            const std::uint64_t bit = std::uint64_t{1} << std::min(63U, trimkit::parallel::thread_index());
            if ((indexes->load(std::memory_order_relaxed) & bit) == 0) {
                indexes->fetch_or(bit, std::memory_order_relaxed);
            }
        }
        if (sink == 42) {
            std::puts("");
        }
    }
};

unsigned threads = 8;

void serial(const Work& work) { work.run(0, work.count); }

void pool(const Work& work) {
    (void)trimkit::parallel::for_each_range(work.count, [&](std::size_t begin, std::size_t end) { work.run(begin, end); });
}

void spawn(const Work& work) {
    std::vector<std::thread> started;
    const std::size_t share = (work.count + threads - 1) / threads;
    for (unsigned k = 1; k < threads; k++) {
        const std::size_t begin = std::min(work.count, k * share);
        const std::size_t end = std::min(work.count, (k + 1) * share);
        started.emplace_back([&work, begin, end] { work.run(begin, end); });
    }
    work.run(0, std::min(work.count, share));
    for (std::thread& thread : started) {
        thread.join();
    }
}

// The persistent pool a careful person writes: workers wait on a condition
// variable for the next generation, each run its share, and count down
class ConditionPool {
public:
    explicit ConditionPool(unsigned count) {
        for (unsigned k = 1; k < count; k++) {
            workers_.emplace_back([this, k] { serve(k); });
        }
    }
    ConditionPool(const ConditionPool&) = delete;
    ConditionPool& operator=(const ConditionPool&) = delete;
    ~ConditionPool() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            quit_ = true;
        }
        start_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    // one task at a time: callers take turns, as such a pool makes them
    void run(const std::function<void(unsigned)>& task) {
        const std::lock_guard<std::mutex> turn(turn_);
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            task_ = task;
            remaining_ = static_cast<unsigned>(workers_.size());
            generation_++;
        }
        start_.notify_all();
        task(0);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
    }

private:
    void serve(unsigned k) {
        std::uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            start_.wait(lock, [&] { return quit_ || generation_ != seen; });
            if (quit_) {
                return;
            }
            seen = generation_;
            const std::function<void(unsigned)> task = task_;
            lock.unlock();
            task(k);
            lock.lock();
            if (--remaining_ == 0) {
                done_.notify_one();
            }
        }
    }

    std::mutex turn_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    std::uint64_t generation_ = 0;
    unsigned remaining_ = 0;
    bool quit_ = false;
    std::function<void(unsigned)> task_;
    std::vector<std::thread> workers_;
};

ConditionPool* condition_pool = nullptr;

void cv_static(const Work& work) {
    const std::size_t share = (work.count + threads - 1) / threads;
    condition_pool->run([&](unsigned k) {
        work.run(std::min(work.count, k * share), std::min(work.count, (k + 1) * share));
    });
}

void cv_chunks(const Work& work) {
    std::atomic<std::size_t> next{0};
    const std::size_t chunk = std::max<std::size_t>(1, work.count / (8 * threads));
    condition_pool->run([&](unsigned) {
        for (;;) {
            const std::size_t begin = next.fetch_add(chunk);
            if (begin >= work.count) {
                return;
            }
            work.run(begin, std::min(work.count, begin + chunk));
        }
    });
}

// every thread of this process, the running ones included, exactly: what
// getrusage reports only as of each thread's last tick
double cpu_seconds() {
    timespec now{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
    return static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_nsec) * 1e-9);
}

// long enough for a pool's workers to stop spinning and sleep, so the CPU a
// run took includes its spinning, and the next run starts from quiet
void settle() { std::this_thread::sleep_for(std::chrono::microseconds(500)); }

using Way = void (*)(const Work&);
constexpr Way WAYS[] = {serial, pool, spawn, cv_static, cv_chunks};
constexpr std::size_t WAY_COUNT = sizeof(WAYS) / sizeof(WAYS[0]);
constexpr std::size_t POOL = 1;

struct Cell {
    double wall_us[WAY_COUNT];
    double cpu_us[WAY_COUNT];
    int pool_threads = 0;
    std::uint64_t publishes = 0, wakes = 0, sleeps = 0, spawns = 0, woken = 0, spin_ns = 0;
};

// `rounds` rounds, every way once a round, in turn, with `callers` threads
// each running its own copy of the work at once: the best of each, every item
// checked to have run exactly once
Cell measure(const Work& shape, unsigned callers, int rounds) {
    std::vector<std::vector<std::atomic<std::uint32_t>>> ran(callers);
    std::vector<Work> works(callers, shape);
    std::atomic<std::uint64_t> indexes{0};
    for (unsigned c = 0; c < callers; c++) {
        ran[c] = std::vector<std::atomic<std::uint32_t>>(shape.count);
        works[c].ran = &ran[c];
    }
    Cell cell{};
    for (double& wall : cell.wall_us) {
        wall = 1e300;
    }
    for (int round = 0; round < rounds; round++) {
        for (std::size_t way = 0; way < WAY_COUNT; way++) {
            if (way == 0 && round >= 3) {
                continue;  // the serial loop's time barely moves
            }
            for (auto& each : ran) {
                for (auto& item : each) {
                    item.store(0, std::memory_order_relaxed);
                }
            }
            indexes.store(0);
            for (Work& work : works) {
                work.indexes = way == POOL ? &indexes : nullptr;
            }
            settle();
            const std::uint64_t published = trimkit::parallel::counters.publishes.load();
            const std::uint64_t woken = trimkit::parallel::counters.wakes.load();
            const std::uint64_t slept = trimkit::parallel::counters.sleeps.load();
            const std::uint64_t spawned = trimkit::parallel::counters.spawns.load();
            const std::uint64_t roused = trimkit::parallel::counters.woken.load();
            const std::uint64_t spun = trimkit::parallel::counters.spin_ns.load();
            const double cpu = cpu_seconds();
            const auto start = std::chrono::steady_clock::now();
            std::vector<std::thread> others;
            for (unsigned c = 1; c < callers; c++) {
                others.emplace_back([&, c, way] { WAYS[way](works[c]); });
            }
            WAYS[way](works[0]);
            for (std::thread& other : others) {
                other.join();
            }
            const double wall =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
            settle();
            const double used = (cpu_seconds() - cpu) * 1e6;
            for (auto& each : ran) {
                for (std::size_t i = 0; i < shape.count; i++) {
                    if (each[i].load() != 1) {
                        std::fprintf(stderr, "item %zu ran %u times\n", i, each[i].load());
                        std::abort();
                    }
                }
            }
            if (wall < cell.wall_us[way]) {
                cell.wall_us[way] = wall;
                cell.cpu_us[way] = used;
                if (way == POOL) {
                    cell.pool_threads = __builtin_popcountll(indexes.load());
                    cell.publishes = trimkit::parallel::counters.publishes.load() - published;
                    cell.wakes = trimkit::parallel::counters.wakes.load() - woken;
                    cell.sleeps = trimkit::parallel::counters.sleeps.load() - slept;
                    cell.spawns = trimkit::parallel::counters.spawns.load() - spawned;
                    cell.woken = trimkit::parallel::counters.woken.load() - roused;
                    cell.spin_ns = trimkit::parallel::counters.spin_ns.load() - spun;
                }
            }
        }
    }
    return cell;
}

// `count` threads spinning in a process of their own, as other work on the
// machine: so this process's CPU time is still the pool's alone
pid_t start_neighbour(unsigned count) {
    const pid_t pid = fork();
    if (pid == 0) {
        std::vector<std::thread> spinning;
        for (unsigned n = 0; n < count; n++) {
            spinning.emplace_back([] {
                for (;;) {
                    spin(1'000'000);
                }
            });
        }
        for (std::thread& thread : spinning) {
            thread.join();
        }
        _exit(0);
    }
    return pid;
}

}  // namespace

int main(int argc, char** argv) {
    threads = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : std::thread::hardware_concurrency();
    const unsigned callers = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 1;
    const unsigned noise = argc > 3 ? static_cast<unsigned>(std::atoi(argv[3])) : 0;
    const double most_serial_ns = argc > 4 ? std::atof(argv[4]) : 2e9;  // NOLINT(cert-err34-c)
    // one cell alone, as "items,item_ns,skew": what a cell that missed is measured again by
    std::size_t only_items = 0;
    double only_ns = 0;
    int only_skew = -1;
    if (argc > 5) {
        std::sscanf(argv[5], "%zu,%lf,%d", &only_items, &only_ns, &only_skew);  // NOLINT(cert-err34-c)
    }
    calibrate();
    ConditionPool shared(threads);
    condition_pool = &shared;
    const pid_t neighbour = noise > 0 ? start_neighbour(noise) : 0;
    for (const double item_ns : {50.0, 1'000.0, 20'000.0}) {
        for (const std::size_t count : {100UL, 1'000UL, 10'000UL, 100'000UL, 1'000'000UL}) {
            if (static_cast<double>(count) * item_ns > most_serial_ns) {
                continue;
            }
            for (int skew = 0; skew < 3; skew++) {
                if (skew != 0 && count < 1'000) {
                    continue;
                }
                if (only_skew >= 0 && (count != only_items || item_ns != only_ns || skew != only_skew)) {
                    continue;
                }
                const std::vector<std::uint64_t> spins = item_spins(count, item_ns, static_cast<Skew>(skew));
                const Work shape{count, item_ns, static_cast<Skew>(skew), &spins, nullptr};
                const Cell cell = measure(shape, callers, 7);
                std::printf(
                    "{\"items\": %zu, \"item_ns\": %.0f, \"skew\": \"%s\", \"skew_index\": %d, \"threads\": %u, "
                    "\"callers\": %u, \"noise\": %u, \"costliest_us\": %.3f, \"serial\": %.1f, \"pool\": %.1f, "
                    "\"spawn\": %.1f, \"cv_static\": %.1f, \"cv_chunks\": %.1f, \"serial_cpu\": %.1f, "
                    "\"pool_cpu\": %.1f, \"spawn_cpu\": %.1f, \"cv_static_cpu\": %.1f, \"cv_chunks_cpu\": %.1f, "
                    "\"pool_threads\": %d, \"publishes\": %llu, \"wakes\": %llu, \"sleeps\": %llu, "
                    "\"spawns\": %llu, \"woken\": %llu, \"spin_us\": %.1f, \"wake_cost_us\": %.1f, "
                    "\"rechecks\": %llu}\n",
                    count, item_ns, SKEW_NAMES[skew], skew, threads, callers, noise, shape.costliest() / 1000,
                    cell.wall_us[0], cell.wall_us[POOL], cell.wall_us[2], cell.wall_us[3], cell.wall_us[4],
                    cell.cpu_us[0], cell.cpu_us[POOL], cell.cpu_us[2], cell.cpu_us[3], cell.cpu_us[4],
                    cell.pool_threads, static_cast<unsigned long long>(cell.publishes),
                    static_cast<unsigned long long>(cell.wakes), static_cast<unsigned long long>(cell.sleeps),
                    static_cast<unsigned long long>(cell.spawns), static_cast<unsigned long long>(cell.woken),
                    static_cast<double>(cell.spin_ns) / 1000, trimkit::parallel::policy::WAKE_COST_NS / 1000,
                    static_cast<unsigned long long>(trimkit::parallel::counters.rechecks.load()));  // so far
                std::fflush(stdout);
            }
        }
    }
    if (neighbour > 0) {
        kill(neighbour, SIGKILL);
        waitpid(neighbour, nullptr, 0);
    }
}
