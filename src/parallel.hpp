// Every core on one loop over [0, n): a lazy, capped pool of worker threads,
// lock-free, behind one call, `run_job`. Python-free, so a fuzzer drives it
// directly (fuzz/parallel.cpp); common.hpp wraps it with the GIL and Ctrl-C.
//
// How a call runs: the caller times ranges of 1, 2, 4, ... items until one
// takes PROBE_NS, so the cost of an item is measured, never guessed. A job
// with more than ALONE_NS left is published in a slot and workers are woken,
// one per HELPER_NS left; everyone then takes ranges of about GRAIN_NS of
// items from one atomic counter. The caller always takes ranges too, until
// none are left, so a call never depends on a worker running at all: a
// worker only makes it faster. Then the caller closes the slot and waits for
// the workers inside a range, and only those.
//
// The state, and the transitions tests/test_scheduler.py checks in every
// interleaving (each a function of the same name below):
//   Slot.word = epoch (32 bits) | OPEN | active (31 bits): active counts the
//   caller and the workers that joined. A slot is free when OPEN and active
//   are both clear.
//   try_publish   free slot: epoch + 1, active 1, store the job, then OPEN
//   join_job      CAS from the exact word seen (OPEN, that epoch) to active + 1
//   leave_job     active - 1; the last out of a closed job wakes the caller
//   close_job     clear OPEN, active - 1; then wait for active == 0
// A worker touches a job only between a successful join and its leave, a
// join succeeds only while that job's epoch is open, and a caller returns
// only after closing and seeing active reach 0: so no worker ever touches a
// job, which lives on its caller's stack, after the caller has returned.
//
// Threads sleep by waiting on an address (futex, WaitOnAddress, __ulock),
// which compares the word and sleeps in one step, so no wakeup is lost, and
// there is no mutex to be left locked in a fork()ed child, which resets the
// pool's plain integers and starts its own workers when it needs them.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>

#if defined(_WIN32)
#include <process.h>
#pragma comment(lib, "synchronization")
// the three calls we make, rather than windows.h, whose macros (FAR, near, far
// and hundreds more) would reach every source through this header
extern "C" {
__declspec(dllimport) int __stdcall WaitOnAddress(volatile void* address, void* expected, std::size_t size,
                                                  unsigned long timeout_ms);
__declspec(dllimport) void __stdcall WakeByAddressAll(void* address);
__declspec(dllimport) int __stdcall CloseHandle(void* handle);
}
#else
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__APPLE__)
// libc++'s std::atomic::wait is built on these, since macOS 10.12
extern "C" int __ulock_wait(std::uint32_t operation, void* address, std::uint64_t value, std::uint32_t timeout_us);
extern "C" int __ulock_wake(std::uint32_t operation, void* address, std::uint64_t value);
#endif
#endif
#include <thread>

namespace trimkit::parallel {

// what each range of items aims to take, as measured on the caller
constexpr double PROBE_NS = 20'000;              // probing ranges double until one takes this
constexpr double GRAIN_NS = 50'000;              // then each range takes about this
constexpr double ALONE_NS = 300'000;             // less left than this: the caller finishes alone
constexpr double HELPER_NS = 100'000;            // one worker woken per this much left
constexpr std::int64_t CALLER_SPIN_NS = 20'000;  // the caller spins this long for workers, then sleeps
constexpr std::int64_t WORKER_SPIN_NS = 50'000;  // a worker spins this long for more work, then sleeps
// A sleeping worker costs no CPU and only the stack pages it touched, so a
// long wait keeps the pool warm across interactive use. The cost: Python
// warns on os.fork() while other threads are alive, which they then are
constexpr std::uint32_t IDLE_EXIT_MS = 10'000;
constexpr std::uint32_t POLL_MS = 50;  // a waiting caller polls for Ctrl-C this often
constexpr std::size_t SLOTS = 4;       // jobs published at once; more callers run alone
// musl's default thread stack is 128 KB, which tinybvh has overflowed
constexpr std::size_t WORKER_STACK = std::size_t{8} << 20;

constexpr std::uint64_t OPEN = std::uint64_t{1} << 31;
constexpr std::uint64_t ACTIVE = OPEN - 1;
constexpr std::uint64_t EPOCH = std::uint64_t{1} << 32;

// One call's work, on the caller's stack
struct Job {
    std::size_t count = 0;
    void (*run)(const void* body, std::size_t begin, std::size_t end) = nullptr;
    const void* body = nullptr;
    // called by the caller only, between its ranges: true stops the job
    bool (*poll)(void* context) = nullptr;
    void* poll_context = nullptr;
    std::atomic<std::size_t> next{0};   // the first item not yet handed out
    std::atomic<std::size_t> grain{1};  // items per range, set by the caller
    std::atomic<bool> stop{false};      // Ctrl-C or an exception: no more ranges
    std::atomic<bool> error_claimed{false};
    std::exception_ptr error;  // the first exception, read once the job is closed
};

struct alignas(64) Slot {
    std::atomic<std::uint64_t> word{0};
    std::atomic<Job*> job{nullptr};
    // bumped by the last worker out of a closed job: its caller sleeps on it
    std::atomic<std::uint32_t> left{0};
};

struct Pool {
    Slot slots[SLOTS];
    alignas(64) std::atomic<std::uint32_t> wake{0};  // bumped per publish: idle workers sleep on it
    std::atomic<std::uint32_t> live{0};              // workers started and not yet exited
    std::atomic<std::uint32_t> wanted{0};            // how many workers the last publish wanted
    std::atomic<std::uint32_t> cap{0};               // the most workers: CPUs this process may use, less 1
    std::atomic<bool> fork_handled{false};
};

// the extension's one pool: atomics only, shared by every caller
inline Pool pool;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// -- the platform: time, sleeping on a word, threads, CPUs -------------------

inline std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void relax() {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    __builtin_arm_yield();
#endif
}

static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t) &&
              std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

// sleep while `word` holds `expected`, for at most `timeout_ms`; it may also
// return early, so callers re-read the word
inline void park(std::atomic<std::uint32_t>& word, std::uint32_t expected, std::uint32_t timeout_ms) {
    void* address = static_cast<void*>(&word);
#if defined(_WIN32)
    WaitOnAddress(address, &expected, sizeof(expected), timeout_ms);
#elif defined(__linux__)
    timespec timeout{static_cast<time_t>(timeout_ms / 1000), static_cast<long>(timeout_ms % 1000) * 1'000'000};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): futex has no wrapper
    syscall(SYS_futex, address, FUTEX_WAIT_PRIVATE, expected, &timeout, nullptr, 0);
#elif defined(__APPLE__)
    constexpr std::uint32_t COMPARE_AND_WAIT = 1;
    constexpr std::uint32_t NO_ERRNO = 0x01000000;
    __ulock_wait(COMPARE_AND_WAIT | NO_ERRNO, address, expected, timeout_ms * 1000);
#else
    // no way to wait on a word here: nap and let the caller re-read it, never spin
    (void)address;
    (void)expected;
    std::this_thread::sleep_for(std::chrono::milliseconds(std::min<std::uint32_t>(timeout_ms, 1)));
#endif
}

// wake every thread sleeping on `word`
inline void unpark(std::atomic<std::uint32_t>& word) {
    void* address = static_cast<void*>(&word);
#if defined(_WIN32)
    WakeByAddressAll(address);
#elif defined(__linux__)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): futex has no wrapper
    syscall(SYS_futex, address, FUTEX_WAKE_PRIVATE, 0x7fffffff, nullptr, nullptr, 0);
#elif defined(__APPLE__)
    constexpr std::uint32_t COMPARE_AND_WAIT = 1;
    constexpr std::uint32_t WAKE_ALL = 0x00000100;
    constexpr std::uint32_t NO_ERRNO = 0x01000000;
    __ulock_wake(COMPARE_AND_WAIT | WAKE_ALL | NO_ERRNO, address, 0);
#else
    (void)address;
#endif
}

// the CPUs this process may run on: its affinity, where there is one
inline std::uint32_t cpu_count() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        return static_cast<std::uint32_t>(std::max(1, CPU_COUNT(&set)));
    }
#endif
    return std::max(1U, std::thread::hardware_concurrency());
}

inline void* worker_main(void* unused);

// a detached thread running worker_main, with a stack as big as a main
// thread's and every signal blocked, so signals reach Python's threads
inline bool start_worker_thread() {
#if defined(_WIN32)
    auto entry = [](void* argument) -> unsigned {
        worker_main(argument);
        return 0;
    };
    constexpr unsigned STACK_SIZE_PARAM_IS_A_RESERVATION = 0x00010000;
    const std::uintptr_t handle = _beginthreadex(nullptr, static_cast<unsigned>(WORKER_STACK), entry, nullptr,
                                                 STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (handle == 0) {
        return false;
    }
    CloseHandle(
        reinterpret_cast<void*>(handle));  // NOLINT(performance-no-int-to-ptr): the API returns a handle as an integer
    return true;
#else
    pthread_attr_t attributes;
    if (pthread_attr_init(&attributes) != 0) {
        return false;
    }
    pthread_attr_setstacksize(&attributes, WORKER_STACK);
    pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
    sigset_t every_signal;
    sigset_t previous;
    sigfillset(&every_signal);
    pthread_sigmask(SIG_SETMASK, &every_signal, &previous);
    pthread_t thread{};
    const bool started = pthread_create(&thread, &attributes, worker_main, nullptr) == 0;
    pthread_sigmask(SIG_SETMASK, &previous, nullptr);
    pthread_attr_destroy(&attributes);
    return started;
#endif
}

// -- the transitions -----------------------------------------------------------

// the next range of up to `grain` items, if any are left and the job goes on
inline bool claim_range(Job& job, std::size_t grain, std::size_t* begin, std::size_t* end) {
    if (job.stop.load(std::memory_order_relaxed)) {
        return false;
    }
    *begin = job.next.fetch_add(grain, std::memory_order_relaxed);
    if (*begin >= job.count) {
        return false;
    }
    *end = std::min(job.count, *begin + grain);
    return true;
}

// the first exception is kept and stops the job; later ones are dropped
inline void publish_error(Job& job) {
    if (!job.error_claimed.exchange(true, std::memory_order_relaxed)) {
        job.error = std::current_exception();
    }
    job.stop.store(true, std::memory_order_relaxed);
}

inline void run_range(Job& job, std::size_t begin, std::size_t end) {
    try {
        job.run(job.body, begin, end);
    } catch (...) {
        publish_error(job);
    }
}

// the caller only: Ctrl-C stops the job
inline void poll_interrupt(Job& job) {
    if (job.poll != nullptr && job.poll(job.poll_context)) {
        job.stop.store(true, std::memory_order_relaxed);
    }
}

inline void child_after_fork() {
    // only the thread that forked exists in the child: every job and worker
    // belonged to the parent, so close every slot under a new epoch
    for (Slot& slot : pool.slots) {
        const std::uint64_t word = slot.word.load(std::memory_order_relaxed);
        slot.word.store((word & ~(OPEN | ACTIVE)) + EPOCH, std::memory_order_relaxed);
        slot.job.store(nullptr, std::memory_order_relaxed);
    }
    pool.live.store(0, std::memory_order_relaxed);
    pool.wanted.store(0, std::memory_order_relaxed);
}

inline void handle_forks() {
#if !defined(_WIN32)
    if (!pool.fork_handled.exchange(true, std::memory_order_relaxed)) {
        pthread_atfork(nullptr, nullptr, child_after_fork);
    }
#endif
}

// a free slot holding `job`, or none when every slot is busy: then the
// caller runs alone
inline Slot* try_publish(Job& job) {
    for (Slot& slot : pool.slots) {
        std::uint64_t word = slot.word.load(std::memory_order_relaxed);
        if ((word & (OPEN | ACTIVE)) != 0) {
            continue;
        }
        // reserved, with the caller counted: no other caller can take it, and
        // no worker can join until OPEN
        if (!slot.word.compare_exchange_strong(word, word + EPOCH + 1, std::memory_order_relaxed)) {
            continue;
        }
        handle_forks();
        slot.job.store(&job, std::memory_order_relaxed);
        // publishes the job pointer and the job to any worker seeing OPEN
        slot.word.fetch_or(OPEN, std::memory_order_release);
        return &slot;
    }
    return nullptr;
}

inline bool spawn_worker() {
    std::uint32_t live = pool.live.load(std::memory_order_relaxed);
    for (;;) {
        if (live >= pool.cap.load(std::memory_order_relaxed)) {
            return false;
        }
        if (pool.live.compare_exchange_weak(live, live + 1, std::memory_order_relaxed)) {
            break;
        }
    }
    if (!start_worker_thread()) {
        // out of threads: the ones running take its ranges
        pool.live.fetch_sub(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

// at least `want` workers awake: one thread started here, and each new
// worker starts the next, so the caller pays for one start at most
inline void wake_workers(std::uint32_t want) {
    pool.wanted.store(want, std::memory_order_relaxed);
    if (pool.live.load(std::memory_order_relaxed) < want) {
        spawn_worker();
    }
    pool.wake.fetch_add(1, std::memory_order_release);
    unpark(pool.wake);
}

// one more worker inside the job published as `seen`, unless it has closed
// or the slot holds another job since
inline bool join_job(Slot& slot, std::uint64_t seen) {
    std::uint64_t word = seen;
    while ((word & OPEN) != 0 && (word >> 32) == (seen >> 32)) {
        if (slot.word.compare_exchange_weak(word, word + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

// the job is never touched again by this worker
inline void leave_job(Slot& slot) {
    // release: what this worker wrote, before its caller reads it
    const std::uint64_t before = slot.word.fetch_sub(1, std::memory_order_release);
    if ((before & (OPEN | ACTIVE)) == 1) {
        // closed, and this was the last worker in it
        slot.left.fetch_add(1, std::memory_order_release);
        unpark(slot.left);
    }
}

// the word as it was: workers may still be inside unless active was 1
inline std::uint64_t close_job(Slot& slot) { return slot.word.fetch_sub(OPEN | 1, std::memory_order_acq_rel); }

// No worker is left in the job published as `epoch`: none active, or the
// slot published again since, which it can only be once free. Without the
// epoch a caller that looked late would wait on the next caller's job
// (tests/test_scheduler.py, mutation wait_ignores_epoch)
inline bool workers_gone(const Slot& slot, std::uint32_t epoch) {
    const std::uint64_t word = slot.word.load(std::memory_order_acquire);
    return (word & ACTIVE) == 0 || static_cast<std::uint32_t>(word >> 32) != epoch;
}

inline void wait_for_workers(Slot& slot, Job& job, std::uint32_t epoch) {
    const std::int64_t spin_until = now_ns() + CALLER_SPIN_NS;
    while (!workers_gone(slot, epoch)) {
        if (now_ns() < spin_until) {
            relax();
            continue;
        }
        // read before the count, so a worker leaving after it bumps `left`
        // and the sleep returns at once
        const std::uint32_t left = slot.left.load(std::memory_order_acquire);
        if (workers_gone(slot, epoch)) {
            break;
        }
        park(slot.left, left, POLL_MS);
        poll_interrupt(job);
    }
}

// returns false once the worker has slept IDLE_EXIT_MS with nothing to do
inline bool park_worker(std::uint32_t observed) {
    const std::int64_t spin_until = now_ns() + WORKER_SPIN_NS;
    while (now_ns() < spin_until) {
        if (pool.wake.load(std::memory_order_acquire) != observed) {
            return true;
        }
        relax();
    }
    const std::int64_t idle_until = now_ns() + (std::int64_t{IDLE_EXIT_MS} * 1'000'000);
    for (;;) {
        const std::int64_t now = now_ns();
        if (now >= idle_until) {
            return false;
        }
        park(pool.wake, observed, static_cast<std::uint32_t>(((idle_until - now) / 1'000'000) + 1));
        if (pool.wake.load(std::memory_order_acquire) != observed) {
            return true;
        }
    }
}

inline void* worker_main(void* /*unused*/) {
    if (pool.live.load(std::memory_order_relaxed) < pool.wanted.load(std::memory_order_relaxed)) {
        spawn_worker();
    }
    // the epoch each slot last held when this worker found nothing left in
    // it: a drained job its caller hasn't closed yet isn't joined again
    std::uint32_t drained[SLOTS] = {};
    for (;;) {
        // before scanning: a job published after this bumps `wake`
        const std::uint32_t observed = pool.wake.load(std::memory_order_acquire);
        bool joined = false;
        for (std::size_t s = 0; s < SLOTS; s++) {
            Slot& slot = pool.slots[s];
            const std::uint64_t word = slot.word.load(std::memory_order_acquire);
            const auto epoch = static_cast<std::uint32_t>(word >> 32);
            if ((word & OPEN) == 0 || epoch == drained[s]) {
                continue;
            }
            Job* job = slot.job.load(std::memory_order_relaxed);
            if (!join_job(slot, word)) {
                continue;  // closed or published again: `job` is not this epoch's
            }
            std::size_t begin = 0;
            std::size_t end = 0;
            while (claim_range(*job, job->grain.load(std::memory_order_relaxed), &begin, &end)) {
                run_range(*job, begin, end);
            }
            drained[s] = epoch;
            leave_job(slot);
            joined = true;
        }
        if (!joined && !park_worker(observed)) {
            break;
        }
    }
    pool.live.fetch_sub(1, std::memory_order_relaxed);
    return nullptr;
}

// -- the caller ----------------------------------------------------------------

inline std::size_t items_left(const Job& job) {
    return job.count - std::min(job.count, job.next.load(std::memory_order_relaxed));
}

// Every item of `job`, on this thread and any workers it wakes. Returns
// once every range handed out has finished; `job.error` and whatever stopped
// it are then the caller's to report
inline void run_job(Job& job) {
    Slot* slot = nullptr;
    bool tried = false;
    std::uint32_t helpers = 0;
    double item_ns = 0.0;  // measured cost of an item; 0 while probing
    std::size_t probe = 1;
    for (;;) {
        std::size_t grain = probe;
        if (item_ns > 0.0) {
            grain = std::max<std::size_t>(1, static_cast<std::size_t>(GRAIN_NS / item_ns));
            if (slot != nullptr) {
                // the last ranges shrink, so no thread is left with a long tail
                grain = std::min(grain, std::max<std::size_t>(1, items_left(job) / (2 * (std::size_t{helpers} + 1))));
            }
        }
        job.grain.store(grain, std::memory_order_relaxed);
        std::size_t begin = 0;
        std::size_t end = 0;
        if (!claim_range(job, grain, &begin, &end)) {
            break;
        }
        const std::int64_t start = now_ns();
        run_range(job, begin, end);
        const auto took = static_cast<double>(now_ns() - start);
        const double range_item_ns = std::max(1.0, took / static_cast<double>(end - begin));
        if (item_ns == 0.0) {
            if (took >= PROBE_NS) {
                item_ns = range_item_ns;
            } else {
                probe *= 2;
            }
        } else {
            // a slow range shrinks the next at once; otherwise drift
            item_ns = took > 4.0 * GRAIN_NS ? range_item_ns : ((3.0 * item_ns) + range_item_ns) / 4.0;
        }
        poll_interrupt(job);
        if (item_ns == 0.0) {
            continue;
        }
        const double left_ns = static_cast<double>(items_left(job)) * item_ns;
        if (!tried && left_ns >= ALONE_NS) {
            // once: with one CPU, or every slot busy, the caller runs alone
            tried = true;
            const std::uint32_t cap = cpu_count() - 1;
            if (cap > 0) {
                pool.cap.store(cap, std::memory_order_relaxed);
                slot = try_publish(job);
            }
        }
        if (slot != nullptr) {
            const auto want = static_cast<std::uint32_t>(
                std::min<double>(pool.cap.load(std::memory_order_relaxed), left_ns / HELPER_NS));
            if (want > helpers) {
                wake_workers(want);
                helpers = want;
            }
        }
    }
    // the slot's job pointer is left as it is: once closed, the slot may
    // already hold the next caller's, and a stale one is never joined
    if (slot != nullptr) {
        const std::uint64_t before = close_job(*slot);
        if ((before & ACTIVE) != 1) {
            wait_for_workers(*slot, job, static_cast<std::uint32_t>(before >> 32));
        }
    }
}

}  // namespace trimkit::parallel
