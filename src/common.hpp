// What every binding shares besides arrays (those go through numpy.hpp, the
// one door): the GIL is released around compute, big batches run across
// every core (and a vendored library's or a device driver's fan-outs on the
// same pool, Python-free), long loops poll for Ctrl-C, rows found in
// parallel gather once, and positions convert to float32 one way.
#pragma once

#include "numpy.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Each part of the one extension, src/PART.cpp, adds its functions and
// classes to the module in bind_PART, which src/trimkit.cpp calls: one per
// [part.PART] in trimkit.toml, in its order, as scripts/builder.py checks
void bind_bvh(nb::module_& m);
void bind_earcut(nb::module_& m);
void bind_vhacd(nb::module_& m);
void bind_xatlas(nb::module_& m);
void bind_draco(nb::module_& m);
void bind_svg(nb::module_& m);
void bind_meshoptimizer(nb::module_& m);
void bind_collision(nb::module_& m);

namespace trimkit {

// A `Boxes` (src/bvh.cpp, whose type no other part sees), shown as `Boxes`
struct BoxesObject {
    nb::object object;
};

// The boxes a scene culls by, as a `Boxes`: one per instance, 6 values each,
// minimum then maximum corner. Always finite, as products of two inputs
// within COORDINATE_LIMIT, though a box may reach past it: it holds what
// the scene holds
BoxesObject scene_boxes(const std::vector<double>& boxes);

// The box around `count` 3D points (xyz rows): the least and greatest value
// on each axis, or +inf then -inf for no points, the box around nothing
struct PointBounds {
    double lowest[3];
    double highest[3];
};

inline PointBounds bounds_of(const double* points, std::size_t count) {
    constexpr double INF = std::numeric_limits<double>::infinity();
    PointBounds bounds = {{INF, INF, INF}, {-INF, -INF, -INF}};
    for (std::size_t i = 0; i < count; i++) {
        for (std::size_t axis = 0; axis < 3; axis++) {
            bounds.lowest[axis] = std::min(bounds.lowest[axis], points[(3 * i) + axis]);
            bounds.highest[axis] = std::max(bounds.highest[axis], points[(3 * i) + axis]);
        }
    }
    return bounds;
}

// The determinant of a row-major 4x4 transform's 3x3 linear part
inline double determinant_of(const double* m) {
    return (m[0] * ((m[5] * m[10]) - (m[6] * m[9]))) - (m[1] * ((m[4] * m[10]) - (m[6] * m[8]))) +
           (m[2] * ((m[4] * m[9]) - (m[5] * m[8])));
}

// An enumeration from the one table its values come from: member i is
// table[i].name, with table[i].doc as its docstring, and the class docstring
// is `summary` then a numpy Attributes section naming every member
template <typename Enum, typename Entry, std::size_t COUNT>
void bind_enum(const nb::module_& scope, const char* name, const char* summary, const Entry (&table)[COUNT]) {
    // one per enumeration, alive as long as the module that shows it
    static const std::string DOC = [&] {
        std::string text = std::string(summary) + "\n\nAttributes\n----------\n";
        for (const Entry& entry : table) {
            text += std::string(entry.name) + "\n  " + entry.doc + "\n";
        }
        return text;
    }();
    nb::enum_<Enum> bound(scope, name, nb::is_arithmetic(), DOC.c_str());
    for (std::size_t i = 0; i < COUNT; i++) {
        bound.value(table[i].name, static_cast<Enum>(i), table[i].doc);
    }
}

// The one conversion several libraries need: (n, 3) float64 positions as
// float32, centered on their bounding box and scaled by a power of two to a
// half-extent just under 1, with that scale in `*scale`. Centering keeps float32's
// precision far from the origin, scaling keeps tiny meshes clear of
// absolute tolerances and huge ones (past float32's range) finite, and a
// power of two is exact, so lengths convert back without rounding.
inline std::vector<float> float32_positions(const double* values, std::size_t count, double* scale) {
    std::vector<float> positions(3 * count);
    *scale = 1.0;
    if (count == 0) {
        return positions;
    }
    const PointBounds bounds = bounds_of(values, count);
    double center[3] = {0.0, 0.0, 0.0};
    double half_extent = 0.0;
    for (std::size_t axis = 0; axis < 3; axis++) {
        // halved before subtracting, so extents up to DBL_MAX don't overflow
        center[axis] = (bounds.lowest[axis] / 2.0) + (bounds.highest[axis] / 2.0);
        half_extent = std::max(half_extent, (bounds.highest[axis] / 2.0) - (bounds.lowest[axis] / 2.0));
    }
    int exponent = 0;
    std::frexp(half_extent, &exponent);
    *scale = half_extent > 0.0 ? std::ldexp(1.0, -exponent) : 1.0;
    for (std::size_t i = 0; i < 3 * count; i++) {
        positions[i] = static_cast<float>((values[i] - center[i % 3]) * *scale);
    }
    return positions;
}

// Poll for Ctrl-C from a library's progress callback while the GIL is
// released. Only the calling thread receives signals, so others return
// immediately, and polls are rate limited. Once interrupted the error stays
// set on the calling thread: raise it with `nb::python_error()` on return.
class Interrupt {
public:
    static constexpr std::chrono::milliseconds POLL_PERIOD{50};

    bool interrupted() const { return interrupted_; }

    // returns whether the computation should stop
    bool poll() {
        const auto now = std::chrono::steady_clock::now();
        if (interrupted_ || std::this_thread::get_id() != caller_ || now - last_ < POLL_PERIOD) {
            return interrupted_;
        }
        last_ = now;
        const nb::gil_scoped_acquire acquire;
        interrupted_ = PyErr_CheckSignals() != 0;
        return interrupted_;
    }

private:
    std::thread::id caller_ = std::this_thread::get_id();
    std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
    // read from the library's worker threads, written only by the caller
    std::atomic<bool> interrupted_{false};
};

// The shape of every batch query: `body(begin, end)` over [0, count) in
// ranges, on this thread and the pool's workers (parallel.hpp), with the GIL
// released; then, with it held again, a Ctrl-C raised as KeyboardInterrupt
// or the first exception a range threw, rethrown. Each item writes its own
// output, so the result doesn't depend on how the ranges fell.
template <typename Body>
void in_parallel(std::size_t count, const Body& body) {
    if (count == 0) {
        return;
    }
    Interrupt interrupt;
    parallel::Job job;
    job.count = count;
    job.run = [](const void* context, std::size_t begin, std::size_t end) {
        (*static_cast<const Body*>(context))(begin, end);
    };
    job.body = &body;
    job.poll = [](void* context) { return static_cast<Interrupt*>(context)->poll(); };
    job.poll_context = &interrupt;
    {
        const nb::gil_scoped_release release;
        parallel::run_job(job);
    }
    if (interrupt.interrupted()) {
        throw nb::python_error();
    }
    if (job.error) {
        std::rethrow_exception(job.error);
    }
}

// The pool's fork handler, installed once by the module's init
inline void install_fork_handler() { parallel::install_fork_handler(); }

// A vendored library's own fan-out, on the same pool and Python-free:
// `body(context, begin, end)` over [0, count) on this thread and the workers,
// called by the library while the binding already holds the GIL released,
// with no Ctrl-C poll (the library's progress callback polls), and the first
// exception a range threw rethrown here, on the library's thread, for it to
// unwind through
inline void run_ranges(std::size_t count, void (*body)(void* context, std::size_t begin, std::size_t end),
                       void* context) {
    if (count == 0) {
        return;
    }
    struct Call {
        void (*body)(void* context, std::size_t begin, std::size_t end);
        void* context;
    };
    const Call call{body, context};
    parallel::Job job;
    job.count = count;
    job.eager = true;
    job.run = [](const void* called, std::size_t begin, std::size_t end) {
        const Call* c = static_cast<const Call*>(called);
        c->body(c->context, begin, end);
    };
    job.body = &call;
    parallel::run_job(job);
    if (job.error) {
        std::rethrow_exception(job.error);
    }
}

// `body(from, to)` over [0, count) for code already past the GIL (a device's
// answers widened into the caller's columns, between the chunks it reads),
// the cores sharing it a block of items at a time: for a few bytes copied
// per item, which one at a time is too little for the pool to measure. The
// same pool, measuring its first blocks like any query, so a copy the caller
// finishes in a fraction of a millisecond wakes nobody. No Ctrl-C poll: it
// never touches Python
template <typename Body>
void in_blocks_without_gil(std::size_t count, const Body& body) {
    constexpr std::size_t BLOCK = 4096;
    if (count == 0) {
        return;
    }
    const auto blocks = [&](std::size_t first, std::size_t last) {
        body(first * BLOCK, std::min(count, last * BLOCK));
    };
    parallel::Job job;
    job.count = (count + BLOCK - 1) / BLOCK;
    job.run = [](const void* context, std::size_t begin, std::size_t end) {
        (*static_cast<const decltype(blocks)*>(context))(begin, end);
    };
    job.body = &blocks;
    parallel::run_job(job);
    if (job.error) {
        std::rethrow_exception(job.error);
    }
}

// One holder at a time, for what takes exactly one (a device's queue,
// src/vulkan/scene.hpp, which states the rule its users keep): a waiter sleeps
// on the scheduler's parking rather than spinning. A holder takes no other
// lock while it holds this, the GIL included, so there is nothing to deadlock
// on; never nested
class Exclusive {
public:
    void acquire() {
        while (taken_.exchange(1, std::memory_order_acquire) != 0) {
            parallel::park(taken_, 1, 1000);
        }
    }
    void release() {
        taken_.store(0, std::memory_order_release);
        parallel::unpark(taken_);
    }

private:
    std::atomic<std::uint32_t> taken_{0};
};

class Exclusively {
public:
    explicit Exclusively(Exclusive& lock) : lock_(lock) { lock_.acquire(); }
    ~Exclusively() { lock_.release(); }
    Exclusively(const Exclusively&) = delete;
    Exclusively& operator=(const Exclusively&) = delete;

private:
    Exclusive& lock_;
};

// For a query that finds its rows: `body(begin, end, out)` appends the rows
// of items [begin, end) in order to `out`, and they come back in item order
// however the ranges fell across threads, as chunks: `size()` rows in all,
// each chunk's `offset` its first row's place. `write(offset, rows, count)`
// is then called once per chunk, from many threads, to put them where they
// go: straight into the output arrays, with no copy in between
template <typename T>
class Gathered {
public:
    struct Chunk {
        std::size_t begin = 0;   // the first item the chunk's rows came from
        std::size_t offset = 0;  // the first row's place in the whole
        std::vector<T> rows;
        Chunk* next = nullptr;
    };

    Gathered() = default;
    Gathered(const Gathered&) = delete;
    Gathered& operator=(const Gathered&) = delete;
    ~Gathered() {
        for (Chunk* chunk = head_.load(std::memory_order_acquire); chunk != nullptr;) {
            Chunk* next = chunk->next;
            delete chunk;  // NOLINT(cppcoreguidelines-owning-memory): the list owns its chunks
            chunk = next;
        }
    }

    // pushed by every thread while the query runs, never popped until it is done
    void push(std::unique_ptr<Chunk> chunk) {
        chunk->next = head_.load(std::memory_order_relaxed);
        while (!head_.compare_exchange_weak(chunk->next, chunk.get(), std::memory_order_release,
                                            std::memory_order_relaxed)) {
        }
        static_cast<void>(chunk.release());
    }

    // once the query is done: the chunks in item order, with their offsets
    void settle() {
        for (Chunk* chunk = head_.load(std::memory_order_acquire); chunk != nullptr; chunk = chunk->next) {
            chunks_.push_back(chunk);
        }
        std::sort(chunks_.begin(), chunks_.end(), [](const Chunk* a, const Chunk* b) { return a->begin < b->begin; });
        total_ = 0;
        for (Chunk* chunk : chunks_) {
            chunk->offset = total_;
            total_ += chunk->rows.size();
        }
    }

    std::size_t size() const { return total_; }

    template <typename Write>
    void write(const Write& write) const {
        in_parallel(chunks_.size(), [&](std::size_t begin, std::size_t end) {
            for (std::size_t c = begin; c < end; c++) {
                const Chunk& chunk = *chunks_[c];
                write(chunk.offset, chunk.rows.data(), chunk.rows.size());
            }
        });
    }

private:
    std::atomic<Chunk*> head_{nullptr};
    std::vector<Chunk*> chunks_;
    std::size_t total_ = 0;
};

template <typename T, typename Body>
void gather_in_parallel(std::size_t count, Gathered<T>* gathered, const Body& body) {
    in_parallel(count, [&](std::size_t begin, std::size_t end) {
        auto chunk = std::make_unique<typename Gathered<T>::Chunk>();
        chunk->begin = begin;
        body(begin, end, chunk->rows);
        if (!chunk->rows.empty()) {
            gathered->push(std::move(chunk));
        }
    });
    gathered->settle();
}

// the gathered rows as one array, written in parallel chunk by chunk
template <typename T>
std::vector<Unset<T>> gathered_rows(const Gathered<T>& gathered) {
    std::vector<Unset<T>> rows(gathered.size());
    T* out = elements(rows);
    gathered.write(
        [&](std::size_t offset, const T* chunk, std::size_t count) { std::copy_n(chunk, count, out + offset); });
    return rows;
}

// Instances, as a scene or a query names them: which mesh each one places
// (`instance_index`, by default each mesh once) and its 4x4 transform
// (`instance_matrices`, by default the identity), checked for one row per
// instance, meshes in range and a last row of [0, 0, 0, 1]. Each module then
// holds the transforms to its own rule: invertible, or rigid. The index is
// copied and the copy checked, as faces are: whatever becomes an index into
// our memory is never read from the caller's again.
struct Instances {
    std::size_t count = 0;
    // a copy: an index is checked once and then trusted, so it must be ours
    std::vector<std::int64_t> index;
    In<double, -1, 4, 4> matrices;
    bool given_index = false;
    bool given_matrices = false;

    std::size_t mesh(std::size_t i) const { return given_index ? static_cast<std::size_t>(index[i]) : i; }
    const double* matrix(std::size_t i) const {
        static constexpr double IDENTITY[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        return given_matrices ? matrices.data() + (16 * i) : IDENTITY;
    }
};

inline Instances read_instances(const Array<std::int64_t, -1>& index_argument,
                                const Array<double, -1, 4, 4>& matrices_argument, std::size_t mesh_count) {
    Instances instances;
    instances.given_index = !index_argument.is_none();
    instances.given_matrices = !matrices_argument.is_none();
    instances.count = mesh_count;
    if (instances.given_index) {
        const In<std::int64_t, -1> given = read(index_argument, "instance_index");
        instances.index.assign(given.data(), given.data() + given.shape(0));
        instances.count = instances.index.size();
    }
    if (instances.given_matrices) {
        instances.matrices = read(matrices_argument, "instance_matrices");
        if (!instances.given_index) {
            instances.count = instances.matrices.shape(0);
        }
    }
    if (instances.given_matrices && instances.matrices.shape(0) != instances.count) {
        value_error("instance_index and instance_matrices must have one row per instance");
    }
    if (!instances.given_index && instances.count != mesh_count) {
        value_error("without instance_index there is one instance per mesh");
    }
    for (std::size_t i = 0; i < instances.count; i++) {
        if (instances.given_index &&
            (instances.index[i] < 0 || static_cast<std::size_t>(instances.index[i]) >= mesh_count)) {
            value_error("instance_index names a mesh out of range at instance " + std::to_string(i));
        }
        const double* m = instances.matrix(i);
        if (m[12] != 0.0 || m[13] != 0.0 || m[14] != 0.0 || m[15] != 1.0) {
            value_error("instance_matrices must be affine: the last row of instance " + std::to_string(i) +
                        " is not [0, 0, 0, 1]");
        }
    }
    return instances;
}

}  // namespace trimkit

// `BoxesObject` returns its object, shown as `Boxes`
namespace nanobind::detail {
template <>
struct type_caster<trimkit::BoxesObject> {
    // cppcheck-suppress unknownMacro ; nanobind's, and cppcheck doesn't read nanobind's headers
    NB_TYPE_CASTER(trimkit::BoxesObject, const_name("Boxes"))
    static bool from_python(handle /*source*/, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept { return false; }
    static handle from_cpp(const trimkit::BoxesObject& boxes, rv_policy /*policy*/,
                           cleanup_list* /*cleanup*/) noexcept {
        return boxes.object.inc_ref();
    }
};
}  // namespace nanobind::detail
