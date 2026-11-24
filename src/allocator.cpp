// The extension's operator new and delete. Python-free, as every part's own
// code is (trimkit.hpp).
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#if defined(__linux__)
#include "cgroup.hpp"

#include <malloc.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include "windows.hpp"

#include <malloc.h>
#endif

// The extension's one door for allocation: every `new` in it, ours and vendored,
// every std::vector and std::string, comes through here, and a request past the
// memory available is std::bad_alloc, which is Python's MemoryError.
//
// Without this, an input that asks for too much ends in the kernel's SIGKILL on
// Linux: an allocation succeeds whether or not the memory is there, within a
// container's limit too, and the process is killed once the pages are touched,
// with nothing left to catch. So no input path has to remember to check.
//
// It isn't a size limit: anything that fits is allocated. It takes no locks of
// its own and allocates nothing (plain system calls on a stack buffer, no
// stdio, atomics), so it can't deadlock or recurse, and what it can't read
// counts as no limit.
//
// What the system counts as used is only what has been written, and much of
// what the door hands out isn't written at once: an output left unwritten
// until a later pass fills it, a vector's spare capacity. So the door keeps
// one ledger for the process of what it has handed out, less what came back,
// since it last looked at what the system has: a request must fit both in
// what the system has now and in what it had then, less all of that. Two
// threads asking at once, or one asking twice before writing either, can't
// each be given the same free memory. Memory handed out and never written
// counts until it comes back, so near the limit the door may refuse what
// would, as it turned out, have fit: a MemoryError, where the kernel's
// SIGKILL was the alternative.

// NOLINTBEGIN(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory): the door is the one place that calls malloc
constexpr std::uint64_t NO_LIMIT = std::numeric_limits<std::uint64_t>::max();
// how much may be handed out between looks, each a few system calls: small
// allocations can't slip past forever, and one this big always looks
constexpr std::int64_t LOOK_EVERY = std::int64_t{16} << 20;
// a thread's small allocations and frees go to the ledger this much at a
// time: one shared counter touched by every allocation would be contended
constexpr std::int64_t PASS_EVERY = std::int64_t{1} << 20;

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables): the ledger is the process's
// what the system had at the last look
static std::atomic<std::uint64_t> room_at_look{NO_LIMIT};
// what was handed out since, less what came back: bytes the allocator holds
static std::atomic<std::int64_t> granted{0};
// handed out since the last look, for when to look again
static std::atomic<std::int64_t> since_look{0};
// how many times the ledger has started afresh: of threads looking at once,
// the one that moves it on is the one that starts it, so nothing is taken
// back twice
static std::atomic<std::uint64_t> restarts{0};
// this thread's allocations and frees not yet in `granted`
static thread_local std::int64_t unpassed = 0;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

#if defined(__linux__)
// what a cgroup's limit leaves: what it holds, less the file cache the kernel
// gives back before it kills anyone
static std::uint64_t cgroup_room(const char* directory, const char* limit_file, const char* usage_file,
                                 const char* cache_name) {
    char path[trimkit::cgroup::PATH_CAPACITY];
    trimkit::cgroup::join(path, sizeof(path), directory, limit_file);
    const std::uint64_t limit = trimkit::cgroup::read_value(path, "");
    trimkit::cgroup::join(path, sizeof(path), directory, usage_file);
    const std::uint64_t usage = trimkit::cgroup::read_value(path, "");
    if (limit == NO_LIMIT || usage == NO_LIMIT) {
        return NO_LIMIT;
    }
    trimkit::cgroup::join(path, sizeof(path), directory, "memory.stat");
    const std::uint64_t cache = trimkit::cgroup::read_value(path, cache_name);
    const std::uint64_t held = cache == NO_LIMIT ? usage : usage - std::min(usage, cache);
    return limit - std::min(limit, held);
}

static std::uint64_t cgroup2_room(const char* directory) {
    return cgroup_room(directory, "memory.max", "memory.current", "inactive_file ");
}

// the version 1 memory cgroup, and this process's version 2 cgroup and each
// one above it: whichever leaves least
static std::uint64_t cgroups_room() {
    const std::uint64_t room =
        cgroup_room("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file ");
    return std::min(room, trimkit::cgroup::least_over_chain(cgroup2_room));
}
#endif

static std::uint64_t available() {
    std::uint64_t bytes = NO_LIMIT;
#if defined(__linux__)
    const std::uint64_t kib = trimkit::cgroup::read_value("/proc/meminfo", "MemAvailable:");
    if (kib != NO_LIMIT) {
        bytes = kib * 1024;
    }
    bytes = std::min(bytes, cgroups_room());
#elif defined(__APPLE__)
    std::uint64_t total = 0;
    std::size_t size = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &size, nullptr, 0) == 0) {
        bytes = total;
    }
#elif defined(_WIN32)
    // what can still be committed, within the address space
    TrimkitWin32MemoryStatus status = {};
    status.length = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        bytes = std::min<std::uint64_t>(status.available_commit, status.available_virtual);
    }
#endif
    return std::min<std::uint64_t>(bytes, std::numeric_limits<std::size_t>::max());
}

// the bytes the allocator holds for a block, counted the same way when it is
// handed out and when it comes back; 0 where there is no way to ask, which
// leaves each request checked on its own
static std::int64_t held(void* block, std::size_t alignment) {
#if defined(__linux__)
    (void)alignment;
    return static_cast<std::int64_t>(malloc_usable_size(block));
#elif defined(__APPLE__)
    (void)alignment;
    return static_cast<std::int64_t>(malloc_size(block));
#elif defined(_WIN32)
    return static_cast<std::int64_t>(alignment == 0 ? _msize(block) : _aligned_msize(block, alignment, 0));
#else
    (void)block;
    (void)alignment;
    return 0;
#endif
}

// `bytes` more held, or fewer when negative: a thread's own count until it
// reaches PASS_EVERY either way
static void tally(std::int64_t bytes) {
    unpassed += bytes;
    if (unpassed > -PASS_EVERY && unpassed < PASS_EVERY) {
        return;
    }
    granted.fetch_add(unpassed, std::memory_order_relaxed);
    if (unpassed > 0) {
        since_look.fetch_add(unpassed, std::memory_order_relaxed);
    }
    unpassed = 0;
}

// whether `request`, already in the ledger, fits: in what the system has now,
// and in what it had at the last look less everything handed out since
static bool fits(std::int64_t request) {
    std::uint64_t restart = restarts.load(std::memory_order_acquire);
    const std::uint64_t now = available();
    const std::uint64_t last = room_at_look.load(std::memory_order_relaxed);
    const std::int64_t total = granted.load(std::memory_order_relaxed);
    const std::int64_t before = total - request;
    since_look.store(0, std::memory_order_relaxed);
    if (before <= 0 || (last >= now && last - now >= static_cast<std::uint64_t>(before))) {
        // all handed out before this request already shows as used: what the
        // system has now is the whole story, so the ledger starts again from it
        if (restarts.compare_exchange_strong(restart, restart + 1, std::memory_order_acq_rel)) {
            room_at_look.store(now, std::memory_order_relaxed);
            granted.fetch_sub(before, std::memory_order_relaxed);
        }
        return static_cast<std::uint64_t>(request) <= now;
    }
    return static_cast<std::uint64_t>(request) <= now && static_cast<std::uint64_t>(total) <= last;
}

// The door: called with every request, before anything is allocated. A
// thread's small requests go by and are counted once held; a larger one, or
// the one that brings a thread's count to PASS_EVERY, is entered in the
// ledger first, where every thread sees it, and checked at least every
// LOOK_EVERY. What glibc holds after a free still counts against a limit, so
// it's handed back before refusing anything. Returns the bytes entered for
// the request, for the allocation to settle against what it holds
static std::int64_t admit(std::size_t bytes) {
    // in 64 bits whatever size_t is, kept far from overflowing the ledger
    constexpr std::uint64_t LARGEST = std::numeric_limits<std::uint64_t>::max() / 8;
    const auto request = static_cast<std::int64_t>(std::min<std::uint64_t>(bytes, LARGEST));
    if (unpassed + request < PASS_EVERY) {
        return 0;
    }
    const std::int64_t entered = unpassed + request;
    unpassed = 0;
    granted.fetch_add(entered, std::memory_order_relaxed);
    const std::int64_t unseen = since_look.fetch_add(entered, std::memory_order_relaxed) + entered;
    if (request < LOOK_EVERY && unseen < LOOK_EVERY) {
        return request;
    }
    if (fits(request)) {
        return request;
    }
#if defined(__GLIBC__)
    (void)malloc_trim(0);
    if (fits(request)) {
        return request;
    }
#endif
    // the request isn't handed out; the thread's smaller ones before it were
    granted.fetch_sub(request, std::memory_order_relaxed);
    throw std::bad_alloc();
}

// what an allocation settles in the ledger: what it holds, less what `admit`
// already entered for it; or, when it failed, the entry taken back
static void* settle(void* block, std::int64_t entered, std::size_t alignment) {
    if (block == nullptr) {
        granted.fetch_sub(entered, std::memory_order_relaxed);
        throw std::bad_alloc();
    }
    tally(held(block, alignment) - entered);
    return block;
}

static void* allocate(std::size_t bytes) {
    const std::int64_t entered = admit(bytes);
    return settle(std::malloc(std::max<std::size_t>(bytes, 1)), entered, 0);
}

static void* allocate_aligned(std::size_t bytes, std::align_val_t alignment) {
    const std::int64_t entered = admit(bytes);
    const auto align = static_cast<std::size_t>(alignment);
#if defined(_WIN32)
    void* block = _aligned_malloc(std::max<std::size_t>(bytes, 1), align);
#else
    void* block = nullptr;
    if (posix_memalign(&block, std::max(align, sizeof(void*)), std::max<std::size_t>(bytes, 1)) != 0) {
        block = nullptr;
    }
#endif
    return settle(block, entered, align);
}

static void release(void* block) {
    if (block != nullptr) {
        tally(-held(block, 0));
        std::free(block);
    }
}

static void release_aligned(void* block, std::align_val_t alignment) {
    if (block != nullptr) {
        tally(-held(block, static_cast<std::size_t>(alignment)));
#if defined(_WIN32)
        _aligned_free(block);
#else
        std::free(block);
#endif
    }
}

// NOLINTEND(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)

// Replaced for this extension alone: nothing but its init function leaves it
// (src/extension.map). The forms the standard library would forward to these
// are here too, as a system libstdc++'s own copies wouldn't reach ours.
// NOLINTBEGIN(misc-new-delete-overloads, cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)
void* operator new(std::size_t bytes) { return allocate(bytes); }
void* operator new[](std::size_t bytes) { return allocate(bytes); }
void* operator new(std::size_t bytes, const std::nothrow_t& /*unused*/) noexcept {
    try {
        return allocate(bytes);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, const std::nothrow_t& /*unused*/) noexcept {
    return operator new(bytes, std::nothrow);
}
void operator delete(void* block) noexcept { release(block); }
void operator delete[](void* block) noexcept { release(block); }
void operator delete(void* block, std::size_t /*unused*/) noexcept { release(block); }
void operator delete[](void* block, std::size_t /*unused*/) noexcept { release(block); }
void operator delete(void* block, const std::nothrow_t& /*unused*/) noexcept { release(block); }
void operator delete[](void* block, const std::nothrow_t& /*unused*/) noexcept { release(block); }
void* operator new(std::size_t bytes, std::align_val_t alignment) { return allocate_aligned(bytes, alignment); }
void* operator new[](std::size_t bytes, std::align_val_t alignment) { return allocate_aligned(bytes, alignment); }
void operator delete(void* block, std::align_val_t alignment) noexcept { release_aligned(block, alignment); }
void operator delete[](void* block, std::align_val_t alignment) noexcept { release_aligned(block, alignment); }
void operator delete(void* block, std::size_t /*unused*/, std::align_val_t alignment) noexcept {
    release_aligned(block, alignment);
}
void operator delete[](void* block, std::size_t /*unused*/, std::align_val_t alignment) noexcept {
    release_aligned(block, alignment);
}
void* operator new(std::size_t bytes, std::align_val_t alignment, const std::nothrow_t& /*unused*/) noexcept {
    try {
        return allocate_aligned(bytes, alignment);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, std::align_val_t alignment, const std::nothrow_t& /*unused*/) noexcept {
    return operator new(bytes, alignment, std::nothrow);
}
void operator delete(void* block, std::align_val_t alignment, const std::nothrow_t& /*unused*/) noexcept {
    release_aligned(block, alignment);
}
void operator delete[](void* block, std::align_val_t alignment, const std::nothrow_t& /*unused*/) noexcept {
    release_aligned(block, alignment);
}
// NOLINTEND(misc-new-delete-overloads, cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)
