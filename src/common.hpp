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
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
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

// The center and half extent of the box from `lowest` to `highest`, as six
// values: halved before subtracting, so extents up to DBL_MAX don't overflow,
// and never negative
inline void center_half(const double* lowest, const double* highest, double* box) {
    for (std::size_t axis = 0; axis < 3; axis++) {
        box[axis] = (lowest[axis] / 2.0) + (highest[axis] / 2.0);
        box[3 + axis] = std::max(0.0, (highest[axis] / 2.0) - (lowest[axis] / 2.0));
    }
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
    double box[6];
    center_half(bounds.lowest, bounds.highest, box);
    const double* center = box;
    const double half_extent = std::max({box[3], box[4], box[5]});
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

// The one way onto the pool (parallel.hpp): `body(begin, end)` over
// [0, count) in ranges, on this thread and the pool's workers, from a thread
// that holds no GIL. `eager` wakes every worker before the first item, for a
// fan-out that is coarse by construction; `interrupt`, when there is one,
// is polled by this thread between its ranges and stops the job. The first
// exception a range threw comes back, for the caller to rethrow where it may
template <typename Body>
std::exception_ptr on_pool(std::size_t count, const Body& body, bool eager, Interrupt* interrupt) {
    if (count == 0) {
        return nullptr;
    }
    parallel::Job job;
    job.count = count;
    job.eager = eager;
    job.run = [](const void* context, std::size_t begin, std::size_t end) {
        (*static_cast<const Body*>(context))(begin, end);
    };
    job.body = &body;
    if (interrupt != nullptr) {
        job.poll = [](void* context) { return static_cast<Interrupt*>(context)->poll(); };
        job.poll_context = interrupt;
    }
    parallel::run_job(job);
    return job.error;
}

// The shape of every batch query: `body(begin, end)` over [0, count) with the
// GIL released; then, with it held again, a Ctrl-C raised as
// KeyboardInterrupt or the first exception a range threw, rethrown. Each item
// writes its own output, so the result doesn't depend on how the ranges fell
template <typename Body>
void in_parallel(std::size_t count, const Body& body) {
    Interrupt interrupt;
    std::exception_ptr error;
    {
        const nb::gil_scoped_release release;
        error = on_pool(count, body, /*eager=*/false, &interrupt);
    }
    if (interrupt.interrupted()) {
        throw nb::python_error();
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

// The pool's fork handler, installed once by the module's init
inline void install_fork_handler() { parallel::install_fork_handler(); }

// A vendored library's own fan-out, Python-free: `body(context, begin, end)`
// over [0, count), called by the library while the binding already holds the
// GIL released, with no Ctrl-C poll (the library's progress callback polls),
// and the first exception a range threw rethrown here, on the library's
// thread, for it to unwind through
inline void run_ranges(std::size_t count, void (*body)(void* context, std::size_t begin, std::size_t end),
                       void* context) {
    const std::exception_ptr error =
        on_pool(count, [&](std::size_t begin, std::size_t end) { body(context, begin, end); }, /*eager=*/true, nullptr);
    if (error) {
        std::rethrow_exception(error);
    }
}

// `body(from, to)` over [0, count) for code already past the GIL (a device's
// rays copied into a chunk, its answers widened into the caller's columns),
// the cores sharing it a block of items at a time: for a few bytes copied per
// item, which one at a time is too little for the pool to measure. A copy the
// caller finishes in a fraction of a millisecond wakes nobody. No Ctrl-C
// poll: it never touches Python
template <typename Body>
void in_blocks_without_gil(std::size_t count, const Body& body) {
    constexpr std::size_t BLOCK = 4096;
    const std::exception_ptr error =
        on_pool((count + BLOCK - 1) / BLOCK,
                [&](std::size_t first, std::size_t last) { body(first * BLOCK, std::min(count, last * BLOCK)); },
                /*eager=*/false, nullptr);
    if (error) {
        std::rethrow_exception(error);
    }
}

// One holder at a time of `word`, for what takes exactly one (a device's
// queue, src/vulkan/scene.hpp, which states the rule its users keep): a
// waiter sleeps on the scheduler's parking rather than spinning. A holder
// takes no other lock while it holds this, the GIL included, so there is
// nothing to deadlock on; never nested
class Exclusively {
public:
    explicit Exclusively(std::atomic<std::uint32_t>& word) : word_(word) {
        while (word_.exchange(1, std::memory_order_acquire) != 0) {
            parallel::park(word_, 1, 1000);
        }
    }
    ~Exclusively() {
        word_.store(0, std::memory_order_release);
        parallel::unpark(word_);
    }
    Exclusively(const Exclusively&) = delete;
    Exclusively& operator=(const Exclusively&) = delete;

private:
    std::atomic<std::uint32_t>& word_;
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

// How far a rigid transform's rotation may be from orthonormal: matrices
// assembled in double precision are within ~1e-15, and float32 ones ~1e-7
constexpr double ORTHONORMAL_TOLERANCE = 1e-6;

// whether a row-major 4x4 transform's linear part is a rotation: orthonormal
// rows, and no reflection
inline bool is_rigid(const double* m) {
    bool rigid = true;
    for (std::size_t r = 0; r < 3; r++) {
        for (std::size_t c = 0; c < 3; c++) {
            const double* a = m + (4 * r);
            const double* b = m + (4 * c);
            const double product = (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
            rigid = rigid && std::abs(product - (r == c ? 1.0 : 0.0)) <= ORTHONORMAL_TOLERANCE;
        }
    }
    return rigid && determinant_of(m) > 0.0;
}

// the two arguments every scene query names its instances by
using InstanceMatrices = Array<double, -1, 4, 4>;
using InstanceIndex = Array<std::int64_t, -1>;

// The arrays a query names its instances by, viewed once: their dtype, shape
// and layout as `read` checks them, their values not yet looked at. An array
// not given is None
struct InstanceViews {
    In<std::int64_t, -1> index;
    In<double, -1, 4, 4> matrices;
    bool given_index = false;
    bool given_matrices = false;
};

inline InstanceViews view_instances(const InstanceIndex& index_argument, const InstanceMatrices& matrices_argument) {
    InstanceViews views;
    views.given_index = !index_argument.is_none();
    views.given_matrices = !matrices_argument.is_none();
    if (views.given_index) {
        views.index = view(index_argument, "instance_index");
    }
    if (views.given_matrices) {
        views.matrices = view(matrices_argument, "instance_matrices");
    }
    return views;
}

// Instances, as a scene or a query names them: which mesh each one places
// (`instance_index`, by default each mesh once) and its 4x4 transform
// (`instance_matrices`, by default the identity). Copies, checked: nothing of
// the caller's is read again once this is made
struct Instances {
    std::size_t count = 0;
    bool given_index = false;
    bool given_matrices = false;
    std::vector<std::int64_t> index;  // empty unless given
    std::vector<double> matrices;     // 16 a row, empty unless given

    std::size_t mesh(std::size_t i) const { return given_index ? static_cast<std::size_t>(index[i]) : i; }
    const double* matrix(std::size_t i) const {
        static constexpr double IDENTITY[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        return given_matrices ? matrices.data() + (16 * i) : IDENTITY;
    }
};

// whether a query's arrays are exactly the ones `instances` was read from:
// each None or not, and the same bytes, which were checked when it was read
inline bool same_instances(const InstanceViews& views, const Instances& instances) {
    if (views.given_index != instances.given_index || views.given_matrices != instances.given_matrices) {
        return false;
    }
    const bool same_index = !views.given_index || (views.index.size() == instances.index.size() &&
                                                   std::memcmp(views.index.data(), instances.index.data(),
                                                               instances.index.size() * sizeof(std::int64_t)) == 0);
    return same_index && (!views.given_matrices || (views.matrices.size() == instances.matrices.size() &&
                                                    std::memcmp(views.matrices.data(), instances.matrices.data(),
                                                                instances.matrices.size() * sizeof(double)) == 0));
}

// A query's instances read: each array copied once and the copy checked, the
// one place any of this is checked. Values finite and bounded, one row per
// instance, meshes in range, a last row of [0, 0, 0, 1], and for a scene
// that takes only rigid transforms (`rigid`), a rotation. A scene whose
// transforms need only be invertible checks that where it inverts them
inline Instances read_instances(const InstanceViews& views, std::size_t mesh_count, bool rigid) {
    Instances instances;
    instances.given_index = views.given_index;
    instances.given_matrices = views.given_matrices;
    instances.count = mesh_count;
    if (views.given_index) {
        instances.index.assign(views.index.data(), views.index.data() + views.index.shape(0));
        instances.count = instances.index.size();
    }
    if (views.given_matrices) {
        const std::size_t rows = views.matrices.shape(0);
        instances.matrices.resize(16 * rows);
        if (count_beyond(views.matrices.data(), 16 * rows, COORDINATE_LIMIT, instances.matrices.data()) != 0) {
            value_error(std::string("instance_matrices must be finite and within ") + COORDINATE_LIMIT_TEXT);
        }
        if (views.given_index && rows != instances.count) {
            value_error("instance_index and instance_matrices must have one row per instance");
        }
        instances.count = rows;
    }
    if (!views.given_index && instances.count != mesh_count) {
        value_error("without instance_index there is one instance per mesh");
    }
    for (std::size_t i = 0; i < instances.count; i++) {
        if (views.given_index &&
            (instances.index[i] < 0 || static_cast<std::size_t>(instances.index[i]) >= mesh_count)) {
            value_error("instance_index names a mesh out of range at instance " + std::to_string(i));
        }
        const double* m = instances.matrix(i);
        if (m[12] != 0.0 || m[13] != 0.0 || m[14] != 0.0 || m[15] != 1.0) {
            value_error("instance_matrices must be affine: the last row of instance " + std::to_string(i) +
                        " is not [0, 0, 0, 1]");
        }
        if (rigid && !is_rigid(m)) {
            value_error("instance_matrices must be rigid: instance " + std::to_string(i) +
                        " rotates by a matrix that isn't a rotation");
        }
    }
    return instances;
}

// A bound class whose objects pickle, the only way the package binds one.
// `T` says what it pickles as with `nb::dict constructor_arguments() const`:
// its constructor's arguments, by their names, from which a load builds it
// again (a compile error without it). Never an acceleration structure, which
// is rebuilt: a pickle holds only arrays and Python values, named, so it loads
// into any build whose constructor takes the same names.
// tests/test_invariants.py round-trips every class the package binds
template <typename T>
nb::class_<T> picklable_class(nb::module_& m, const char* name, const char* doc) {
    static_assert(std::is_same_v<decltype(std::declval<const T&>().constructor_arguments()), nb::dict>,
                  "a bound class must say what it pickles as: nb::dict constructor_arguments() const");
    nb::class_<T> bound(m, name, doc);
    bound.def(
        "__reduce__",
        [](nb::handle self) {
            const nb::dict arguments = nb::cast<const T&>(self).constructor_arguments();
            const nb::object partial = nb::module_::import_("functools").attr("partial");
            return nb::make_tuple(partial(nb::borrow(self.type()), **arguments), nb::make_tuple());
        },
        R"(Pickle as its constructor's arguments, by name: anything built from them is built again on load.

Returns
-------
reduced : tuple
  Its constructor with those arguments bound, as pickle and copy call it.)");
    return bound;
}

}  // namespace trimkit

// `BoxesObject` returns its object, shown as `Boxes`
namespace nanobind::detail {
struct boxes_shown {
    // NOLINTNEXTLINE(readability-identifier-naming): the name nanobind reads
    static constexpr auto Name = const_name("Boxes");
};
template <>
struct type_caster<trimkit::BoxesObject> : passed_through<trimkit::BoxesObject, boxes_shown> {};
}  // namespace nanobind::detail
