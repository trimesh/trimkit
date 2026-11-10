// The one extension: every part's functions and classes in one module,
// `import trimkit`, built as the package itself (trimkit/__init__ with the
// extension suffix), so there is no Python in it at all. The top level asks
// geometry questions whatever library answers them; its one submodule,
// `trimkit.exchange`, reads and writes specific formats.
#include "common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#if defined(__linux__)
#include "cgroup.hpp"
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <malloc.h>
extern "C" __declspec(dllimport) int __stdcall GlobalMemoryStatusEx(void* status);
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
// stdio), so it can't deadlock or recurse, and what it can't read counts as
// no limit, so it can only refuse what wouldn't have fit.

// NOLINTBEGIN(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory): the door is the one place that calls malloc
constexpr std::uint64_t NO_LIMIT = std::numeric_limits<std::uint64_t>::max();
// how much a thread may ask for between looks, each a few system calls: small
// allocations can't slip past forever, and one this big always looks
constexpr std::size_t LOOK_EVERY = std::size_t{16} << 20;

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

// this process's cgroup and each one above it, whichever leaves least
static std::uint64_t cgroups_room() {
    std::uint64_t room =
        cgroup_room("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file ");
    char directory[trimkit::cgroup::DIRECTORY_CAPACITY];
    trimkit::cgroup::own_directory(directory, sizeof(directory));
    for (bool more = true; more; more = trimkit::cgroup::ascend(directory)) {
        room = std::min(room, cgroup_room(directory, "memory.max", "memory.current", "inactive_file "));
    }
    return room;
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
    // MEMORYSTATUSEX: what can still be committed, within the address space
    struct {
        std::uint32_t length;
        std::uint32_t load;
        std::uint64_t total_physical, available_physical;
        std::uint64_t total_commit, available_commit;
        std::uint64_t total_virtual, available_virtual;
        std::uint64_t available_extended;
    } status = {};
    status.length = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        bytes = std::min(status.available_commit, status.available_virtual);
    }
#endif
    return std::min<std::uint64_t>(bytes, std::numeric_limits<std::size_t>::max());
}

// The door: called with every request, before anything is allocated. What
// glibc holds after a free still counts against a limit, so it's handed back
// before refusing anything, under malloc's own locks
static void admit(std::size_t bytes) {
    static thread_local std::uint64_t asked = 0;
    asked += bytes;
    if (asked < LOOK_EVERY) {
        return;
    }
    const std::uint64_t request = asked;
    asked = 0;
    if (request <= available()) {
        return;
    }
#if defined(__GLIBC__)
    (void)malloc_trim(0);
    if (request <= available()) {
        return;
    }
#endif
    throw std::bad_alloc();
}

static void* allocate(std::size_t bytes) {
    admit(bytes);
    void* block = std::malloc(std::max<std::size_t>(bytes, 1));
    if (block == nullptr) {
        throw std::bad_alloc();
    }
    return block;
}

static void* allocate_aligned(std::size_t bytes, std::align_val_t alignment) {
    admit(bytes);
    const auto align = static_cast<std::size_t>(alignment);
#if defined(_WIN32)
    void* block = _aligned_malloc(std::max<std::size_t>(bytes, 1), align);
#else
    void* block = nullptr;
    if (posix_memalign(&block, std::max(align, sizeof(void*)), std::max<std::size_t>(bytes, 1)) != 0) {
        block = nullptr;
    }
#endif
    if (block == nullptr) {
        throw std::bad_alloc();
    }
    return block;
}

static void release_aligned(void* block) {
#if defined(_WIN32)
    _aligned_free(block);
#else
    std::free(block);
#endif
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
void operator delete(void* block) noexcept { std::free(block); }
void operator delete[](void* block) noexcept { std::free(block); }
void operator delete(void* block, std::size_t /*unused*/) noexcept { std::free(block); }
void operator delete[](void* block, std::size_t /*unused*/) noexcept { std::free(block); }
void operator delete(void* block, const std::nothrow_t& /*unused*/) noexcept { std::free(block); }
void operator delete[](void* block, const std::nothrow_t& /*unused*/) noexcept { std::free(block); }
void* operator new(std::size_t bytes, std::align_val_t alignment) { return allocate_aligned(bytes, alignment); }
void* operator new[](std::size_t bytes, std::align_val_t alignment) { return allocate_aligned(bytes, alignment); }
void operator delete(void* block, std::align_val_t /*unused*/) noexcept { release_aligned(block); }
void operator delete[](void* block, std::align_val_t /*unused*/) noexcept { release_aligned(block); }
void operator delete(void* block, std::size_t /*unused*/, std::align_val_t /*unused*/) noexcept {
    release_aligned(block);
}
void operator delete[](void* block, std::size_t /*unused*/, std::align_val_t /*unused*/) noexcept {
    release_aligned(block);
}
// NOLINTEND(misc-new-delete-overloads, cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory)

NB_MODULE(trimkit, m) {  // NOLINT(performance-unnecessary-value-param): the macro passes it by value
    trimkit::install_fork_handler();
    // the last paragraph names what this build compiled in, each upstream's
    // repository and commit, or `(system)` for a distribution's copy
    m.doc() = R"(Thin, stable-ABI numpy bindings to pinned C and C++ geometry libraries.

They depend only on `numpy`, read arrays of exactly the dtype shown (float64
ones finite and within +-1e30) in place when C-contiguous and copy any other
layout once, convert a dtype numpy casts safely (or a list) once with a
UserWarning, return fresh arrays that own their memory, and release the GIL
during compute.

Attributes
----------
__version__ : str
  This package's version.
Mesh : TypeAlias
  A triangle mesh as the scenes take it and `decompose` returns it: a
  (vertices, faces) pair of (n, 3) float64 vertices and (m, 3) int64 triangle
  indices into them, given as a tuple, list or any sequence of two.
exchange : module
  Reading and writing data formats: draco and SVG paths.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> isinstance(trimkit.__version__, str)
True
>>> mesh: trimkit.Mesh = (np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64))
>>> len(mesh)
2

Notes
-----
Built from: )" TRIMKIT_UPSTREAM ".";
    m.attr("__version__") = TRIMKIT_VERSION;
    // a triangle mesh as scenes take them: (n, 3) float64 vertices and
    // (m, 3) int64 faces, the shapes every docstring states
    const nb::object ndarray = nb::module_::import_("numpy.typing").attr("NDArray");
    const nb::module_ numpy = nb::module_::import_("numpy");
    m.attr("Mesh") = nb::module_::import_("builtins")
                         .attr("tuple")[nb::make_tuple(ndarray[numpy.attr("float64")], ndarray[numpy.attr("int64")])];
    // a module of the extension, importable as `trimkit.exchange` too
    nb::module_ exchange = m.def_submodule("exchange", R"(Reading and writing data formats: draco and SVG paths.

Attributes
----------
DracoKind : enum.IntEnum
  What a draco attribute holds, as `draco_encode` takes it.
SvgKind : enum.IntEnum
  What drew an SvgSegment, its `kind` field.
SvgSegment : numpy.dtype
  One drawing command of an SVG path, as `svg_parse` returns them and
  `svg_string` takes them.

Examples
--------
>>> import trimkit
>>> trimkit.exchange.svg_parse("M 0 0 L 1 1")[1].dtype == trimkit.exchange.SvgSegment
True)");
    nb::module_::import_("sys").attr("modules")["trimkit.exchange"] = exchange;
    bind_bvh(m);
    bind_earcut(m);
    bind_vhacd(m);
    bind_xatlas(m);
    bind_draco(exchange);
    bind_svg(exchange);
    bind_meshoptimizer(m);
    bind_collision(m);
}
