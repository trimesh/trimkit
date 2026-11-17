// platform.hpp: everything the pool needs from the operating system, and the
// only place in trimkit that asks for it by name.
//
// The pool (parallel.hpp) needs a clock, a way to sleep on a word until it
// changes and to wake whoever sleeps on it, a way to start a thread, how many
// CPUs it may use, what happens in a child after fork(), and how to run a
// body and catch whatever it throws. Each is one function here, written once
// for each kind of machine:
//
//   Linux     futex, sched_getaffinity and cgroup quotas, pthreads
//   Windows   WaitOnAddress, _beginthreadex
//   macOS     __ulock (what libc++'s own std::atomic::wait is built on)
//   other     a millisecond nap in place of a sleep on a word
//   GenMC     TRIMKIT_GENMC: the model checker GenMC runs the pool's own code
//             (verify/genmc), so here a sleep is GenMC's blocking assume, the
//             clock jumps, there are two CPUs, a body fails by a flag rather
//             than a C++ exception (GenMC runs none), and idle workers exit at
//             once, so every thread of a checked execution ends and a caller
//             that would wait forever shows as one that never finishes
//
// Nothing here knows about jobs, slots or Python; gil.hpp and the Vulkan code
// use the fork hooks too.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>

#if defined(TRIMKIT_GENMC)
#include <genmc.h>
#include <pthread.h>

#include <cassert>
#else
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#if defined(__GLIBCXX__)
#include <cxxabi.h>  // abi::__forced_unwind
#endif
#if defined(_WIN32)
#include "windows.hpp"

#include <process.h>
#pragma comment(lib, "synchronization")
#else
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#if defined(__linux__)
#include "cgroup.hpp"

#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__APPLE__)
// what libc++'s std::atomic::wait is built on, since macOS 10.12
extern "C" int __ulock_wait(std::uint32_t operation, void* address, std::uint64_t value, std::uint32_t timeout_us);
extern "C" int __ulock_wake(std::uint32_t operation, void* address, std::uint64_t value);
#endif
#endif
#endif

namespace trimkit::platform {

static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t) &&
              std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

#if defined(TRIMKIT_GENMC)
constexpr bool MODEL_CHECKING = true;
#else
constexpr bool MODEL_CHECKING = false;
#endif

// -- time ----------------------------------------------------------------------

#if defined(TRIMKIT_GENMC)
// Under GenMC, the time is this thread's own, and moves on a millisecond for
// each range of items it runs (run_catching) and at no other step: so the
// pool's probe still ends, and a loop that waits for a word, reading the clock
// but writing nothing, is a spin loop GenMC explores until the word changes
inline thread_local std::int64_t genmc_now = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
#endif

// Nanoseconds on a clock that never goes back
inline std::int64_t monotonic_ns() {
#if defined(TRIMKIT_GENMC)
    return genmc_now;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}

// A hint to the CPU, inside a spin loop, that this thread is waiting;
// nothing under GenMC, which runs no CPU
inline void cpu_relax() {
#if defined(TRIMKIT_GENMC)
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    __builtin_arm_yield();
#endif
}

// -- sleeping on a word ----------------------------------------------------------

// Sleep while `word` holds `expected`, for at most `timeout_ms`. The system
// compares the word and sleeps in one step, so a wake after the word changes
// is never lost. It may also return early, so callers read the word again.
// Under GenMC it returns at once, which is that: every loop around it is then
// a plain spin loop, which GenMC explores until the word changes, and reports
// one that never can (--check-liveness) as the deadlock it would be
inline void wait_while_equal(std::atomic<std::uint32_t>& word, std::uint32_t expected, std::uint32_t timeout_ms) {
#if defined(TRIMKIT_GENMC)
    (void)word;
    (void)expected;
    (void)timeout_ms;
#else
    void* address = static_cast<void*>(&word);
#if defined(_WIN32)
    WaitOnAddress(address, &expected, sizeof(expected), timeout_ms);
#elif defined(__linux__) && defined(SYS_futex)
    // What SYS_futex reads: the kernel's own timespec, two of its words. Not
    // the C library's `timespec` on a 32-bit build, whose time_t may have been
    // made 64 bits; but on x32 (the 64-bit kernel's 32-bit ABI) the kernel's
    // words are 64 bits, and there the C library's matches it
#if defined(__x86_64__) && defined(__ILP32__)
    using FutexWord = long long;
#else
    using FutexWord = long;
#endif
    struct FutexTimeout {
        FutexWord seconds;
        FutexWord nanoseconds;
    };
    static_assert(sizeof(FutexTimeout) == 2 * sizeof(FutexWord));
    FutexTimeout timeout{static_cast<FutexWord>(timeout_ms / 1000),
                         static_cast<FutexWord>(timeout_ms % 1000) * 1'000'000};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): futex has no wrapper
    syscall(SYS_futex, address, FUTEX_WAIT_PRIVATE, expected, &timeout, nullptr, 0);
#elif defined(__APPLE__)
    constexpr std::uint32_t COMPARE_AND_WAIT = 1;
    constexpr std::uint32_t NO_ERRNO = 0x01000000;
    // a timeout of 0 means forever to __ulock_wait: never pass one
    __ulock_wait(COMPARE_AND_WAIT | NO_ERRNO, address, expected, (timeout_ms == 0 ? 1 : timeout_ms) * 1000);
#else
    // no way to sleep on a word here: nap, and let the caller read it again
    (void)address;
    (void)expected;
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms < 1 ? timeout_ms : 1));
#endif
#endif
}

// Wake every thread sleeping on `word`
inline void wake_all(std::atomic<std::uint32_t>& word) {
#if defined(TRIMKIT_GENMC)
    (void)word;
#else
    void* address = static_cast<void*>(&word);
#if defined(_WIN32)
    WakeByAddressAll(address);
#elif defined(__linux__) && defined(SYS_futex)
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
#endif
}

// Wake up to `count` threads sleeping on `word`, and say how many it woke
// where the system says, else `count`. Linux wakes them in one call;
// elsewhere there is no such call, so it is one at a time
inline std::uint32_t wake_some(std::atomic<std::uint32_t>& word, std::uint32_t count) {
#if defined(TRIMKIT_GENMC)
    (void)word;
    return count;
#else
    void* address = static_cast<void*>(&word);
#if defined(_WIN32)
    for (std::uint32_t k = 0; k < count; k++) {
        WakeByAddressSingle(address);
    }
    return count;
#elif defined(__linux__) && defined(SYS_futex)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): futex has no wrapper
    const long woken = syscall(SYS_futex, address, FUTEX_WAKE_PRIVATE, static_cast<int>(count), nullptr, nullptr, 0);
    return woken < 0 ? count : static_cast<std::uint32_t>(woken);
#elif defined(__APPLE__)
    constexpr std::uint32_t COMPARE_AND_WAIT = 1;
    constexpr std::uint32_t NO_ERRNO = 0x01000000;
    for (std::uint32_t k = 0; k < count; k++) {
        __ulock_wake(COMPARE_AND_WAIT | NO_ERRNO, address, 0);
    }
    return count;
#else
    (void)address;
    return count;
#endif
#endif
}

// -- threads and CPUs ------------------------------------------------------------

// Start a thread running `entry`, detached, with a stack of `stack_bytes` (as
// big as a main thread's: musl's default 128 KB has overflowed) and every
// signal blocked, so signals go to Python's own threads. False if the system
// won't start one
inline bool start_detached_thread(void* (*entry)(void*), std::size_t stack_bytes) {
#if defined(TRIMKIT_GENMC)
    (void)stack_bytes;
    pthread_t thread{};
    return pthread_create(&thread, nullptr, entry, nullptr) == 0;
#elif defined(_WIN32)
    struct Start {
        static unsigned __stdcall run(void* argument) {
            reinterpret_cast<void* (*)(void*)>(argument)(
                nullptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            return 0;
        }
    };
    constexpr unsigned STACK_SIZE_PARAM_IS_A_RESERVATION = 0x00010000;
    const std::uintptr_t handle =
        _beginthreadex(nullptr, static_cast<unsigned>(stack_bytes), Start::run,
                       reinterpret_cast<void*>(entry),  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
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
    pthread_attr_setstacksize(&attributes, stack_bytes);
    pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
    // every signal blocked, so each goes to a thread that takes signals (the
    // main one: Ctrl-C is Python's), but the ones a fault raises on the very
    // thread that faults: blocked there, Linux resets the handler and kills
    // the process, so a crash in a body would run no faulthandler, no
    // sanitizer report, and leave the fuzzer no input to replay
    sigset_t every_signal;
    sigset_t previous;
    sigfillset(&every_signal);
    for (const int fault : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGTRAP, SIGSYS}) {
        sigdelset(&every_signal, fault);
    }
    pthread_sigmask(SIG_SETMASK, &every_signal, &previous);
    pthread_t thread{};
    const bool started = pthread_create(&thread, &attributes, entry, nullptr) == 0;
    pthread_sigmask(SIG_SETMASK, &previous, nullptr);
    pthread_attr_destroy(&attributes);
    return started;
#endif
}

// The CPUs the machine has online, which affinity and quotas only narrow:
// fixed while the process runs, read once. Not hardware_concurrency on
// POSIX, which musl answers from the affinity at the time
inline std::uint32_t cpus_online() {
#if defined(TRIMKIT_GENMC)
    return 2;
#elif defined(__unix__) || defined(__APPLE__)
    static const long ONLINE = sysconf(_SC_NPROCESSORS_ONLN);
    return ONLINE < 1 ? 1 : static_cast<std::uint32_t>(ONLINE);
#else
    static const std::uint32_t ONLINE =
        std::thread::hardware_concurrency() < 1 ? 1 : std::thread::hardware_concurrency();
    return ONLINE;
#endif
}

#if defined(__linux__) && !defined(TRIMKIT_GENMC)
// the CPUs one cgroup's quota allows, from its `cpu.max`: "max 100000" for no
// quota, or "<quota> <period>" in microseconds
inline std::uint64_t cpu_quota(const char* directory) {
    char path[cgroup::PATH_CAPACITY];
    char text[64];
    cgroup::join(path, sizeof(path), directory, "cpu.max");
    cgroup::read_text(path, text, sizeof(text));
    char* end = nullptr;
    const unsigned long long quota = std::strtoull(text, &end, 10);
    const unsigned long long period = end == text ? 0 : std::strtoull(end, nullptr, 10);
    if (period == 0) {
        return cgroup::NO_LIMIT;
    }
    const unsigned long long cpus = (quota + period - 1) / period;
    return cpus < 1 ? 1 : cpus;
}
#endif

// The CPUs this process may run on now: its affinity, where there is one, and
// no more than a cgroup's CPU quota allows (`cpu.max` at its cgroup and each
// one above), which affinity doesn't see: a container limited to 2 CPUs on a
// 64-core host would otherwise wake 63 workers for them. A syscall and a file
// read a cgroup level, so the pool reads it at most every 100 ms
inline std::uint32_t cpus_allowed() {
    std::uint32_t count = cpus_online();
#if defined(__linux__) && !defined(TRIMKIT_GENMC)
    // as many CPUs as the kernel's mask may name: a fixed cpu_set_t holds 1024,
    // and on a bigger machine sched_getaffinity refuses it, so the set grows
    // until it doesn't
    for (int size = 1024; size <= (1 << 20); size *= 2) {
        cpu_set_t* set = CPU_ALLOC(size);
        if (set == nullptr) {
            break;
        }
        const std::size_t bytes = CPU_ALLOC_SIZE(size);
        CPU_ZERO_S(bytes, set);
        const bool read = sched_getaffinity(0, bytes, set) == 0;
        if (read) {
            const int allowed = CPU_COUNT_S(bytes, set);
            count = static_cast<std::uint32_t>(allowed < 1 ? 1 : allowed);
        }
        CPU_FREE(set);
        if (read || errno != EINVAL) {
            break;
        }
    }
    const std::uint64_t quota = cgroup::least_over_chain(cpu_quota);
    count = quota < count ? static_cast<std::uint32_t>(quota) : count;
#endif
    return count;
}

// -- fork ------------------------------------------------------------------------

// Up to this many functions run in a child after fork(), in the order they
// were registered, before anything else runs there: what each module must
// forget of its parent (the pool's threads, the GIL's counts, a device)
constexpr std::size_t FORK_HOOKS = 8;

struct ForkHooks {
    void (*hooks[FORK_HOOKS])() = {};
    std::atomic<std::size_t> count{0};
    std::atomic<std::uint32_t> generation{0};
};
inline ForkHooks fork_hooks;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// How many forks this process is from the one that loaded the extension:
// whatever a parent opened (a Vulkan device) is its own only while this holds
// the value it read then
inline std::uint32_t fork_generation() { return fork_hooks.generation.load(std::memory_order_relaxed); }

inline void run_fork_hooks() {
    fork_hooks.generation.fetch_add(1, std::memory_order_relaxed);
    const std::size_t count = fork_hooks.count.load(std::memory_order_relaxed);
    for (std::size_t k = 0; k < count; k++) {
        fork_hooks.hooks[k]();
    }
}

// `hook` to run in every child after fork(). Called by the module's init, on
// one thread, before any call can fork; false once FORK_HOOKS are taken.
// Registered with the system as one explicit call, not a header variable's
// initializer, which the standard lets wait until the variable is first used
inline bool on_fork_child(void (*hook)()) {
    const std::size_t index = fork_hooks.count.load(std::memory_order_relaxed);
    if (index >= FORK_HOOKS) {
        return false;
    }
    fork_hooks.hooks[index] = hook;
    fork_hooks.count.store(index + 1, std::memory_order_release);
#if !defined(_WIN32) && !defined(TRIMKIT_GENMC)
    if (index == 0) {
        pthread_atfork(nullptr, nullptr, run_fork_hooks);
    }
#endif
    return true;
}

// -- running a body ---------------------------------------------------------------

// What running a body can end in: nothing, or the exception it threw. Under
// GenMC, which runs no C++ exception, a flag a harness's body raises
// (`genmc_body_failed`) stands in for one
#if defined(TRIMKIT_GENMC)
struct Failure {
    bool failed = false;
    explicit operator bool() const { return failed; }
};
inline thread_local bool genmc_body_failed = false;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// GenMC can't run a class with a vtable, so the error is named, never made
template <typename Error>
Failure failure() {
    return Failure{true};
}

template <typename Run, typename Ending>
Failure run_catching(const Run& run, const Ending& /*ending*/) {
    run();
    genmc_now += 1'000'000;
    const bool failed = genmc_body_failed;
    genmc_body_failed = false;
    return Failure{failed};
}

// a broken promise: an assertion GenMC reports as the error, never an
// execution it would quietly set aside
[[noreturn]] inline void fail_fast(const char* reason) {
    assert(reason == nullptr);
    __builtin_trap();
}
#else
using Failure = std::exception_ptr;

template <typename Error>
Failure failure() {
    return std::make_exception_ptr(Error{});
}

// `run()`, and whatever it threw. A thread ending inside it (pthread_exit, or
// CPython ending a daemon thread that takes the GIL as it finalizes: glibc
// unwinds it) calls `ending()` and goes on ending, as glibc aborts an unwind
// that is caught and not thrown again
template <typename Run, typename Ending>
Failure run_catching(const Run& run, const Ending& ending) {
    try {
        run();
        return nullptr;
    }
#if defined(__GLIBCXX__)
    catch (abi::__forced_unwind&) {
        ending();
        throw;
    }
#endif
    catch (...) {
        return std::current_exception();
    }
}

// End the process now, saying why: a promise the code makes was broken
[[noreturn]] inline void fail_fast(const char* reason) {
    // the process ends either way: what writing the reason returns is moot
    (void)std::fputs("trimkit: broken promise: ", stderr);
    (void)std::fputs(reason, stderr);
    (void)std::fputc('\n', stderr);
    std::abort();
}
#endif

}  // namespace trimkit::platform
