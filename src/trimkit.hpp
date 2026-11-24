// trimkit.hpp: what every part's algorithm shares, Python-free. The parts'
// own code (src/NAME.cpp) includes this and nothing of Python's, so it builds
// without Python, and the fuzz harnesses link the very objects the extension
// does. src/trimkit.cpp, the one file that is Python, reads arguments through
// numpy.hpp and calls what is declared at the end of this file.
#pragma once

#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace trimkit {

// Every float64 input lies within +-1e30, so no arithmetic downstream can
// overflow: the deepest products in any module are degree 6 (an instance
// transform times a coordinate, cubed by the watertight ray test, or squared
// twice by tinybvh's surface areas), and 1e180 is far from DBL_MAX. Unbounded,
// rays silently missed everything past ~5.6e102 and tinybvh's builder cast
// NaN to int near 1e300. No real geometry is near it: the observable universe
// is ~8.8e29 mm across.
constexpr double COORDINATE_LIMIT = 1e30;
// how messages name it: the UTF-8 bytes of the plus-minus sign, as bytes so
// the source stays ASCII, split so `1e3` isn't read as more hex digits
constexpr const char* COORDINATE_LIMIT_TEXT =
    "\xc2\xb1"
    "1e30";
static_assert(COORDINATE_LIMIT == 1e30, "COORDINATE_LIMIT_TEXT names the limit");

// An input refused, with the reason: std::invalid_argument, which nanobind
// raises as ValueError, the one way a part refuses what it was given
[[noreturn]] inline void refuse(const std::string& message) { throw std::invalid_argument(message); }

// int64 arguments narrowed in one place, compared before they are cast, so a
// 32-bit size_t never wraps one into range (2**32 read as 0 on win32)
//
// whether `value` is in [0, bound)
constexpr bool indexes(std::int64_t value, std::uint64_t bound) {
    return value >= 0 && static_cast<std::uint64_t>(value) < bound;
}
// `value` as a size_t, for an argument that bounds how much is wanted: any
// more than a size_t holds is as much as there can be
constexpr std::size_t at_most(std::int64_t value) {
    if (value <= 0) {
        return 0;
    }
    if (static_cast<std::uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
        return std::numeric_limits<std::size_t>::max();
    }
    return static_cast<std::size_t>(value);
}
// every compile, on every target, checks the narrowing a 32-bit build needs
static_assert(!indexes(std::int64_t{1} << 32, std::uint64_t{1} << 31) && !indexes(-1, 10) &&
              !indexes((std::int64_t{1} << 32) + 1, 2) && indexes(1, 2) &&
              !indexes(std::numeric_limits<std::int64_t>::max(), std::uint64_t{1} << 32));
static_assert(at_most(-1) == 0 && at_most(3) == 3 &&
              std::uint64_t{at_most(std::int64_t{1} << 32)} ==
                  (sizeof(std::size_t) >= 8 ? std::uint64_t{1} << 32
                                            : std::uint64_t{std::numeric_limits<std::size_t>::max()}));

// `rows` times `per_row`, a count of elements to allocate: std::bad_alloc,
// Python's MemoryError, when a size_t can't count that many, as no memory
// could hold them (a 32-bit size_t would wrap it to a small count instead)
inline std::size_t element_count(std::size_t rows, std::size_t per_row) {
    std::size_t count = 0;
    if (__builtin_mul_overflow(rows, per_row, &count)) {
        throw std::bad_alloc();
    }
    return count;
}

// The one scan of values there is. How many of `count` values are not within
// `limit`: beyond it, an infinity or not a number, one comparison that is
// false for all three, counted without branching so it vectorizes. Each value
// is written to `copy` as it is read, when there is one
template <typename T>
std::size_t count_beyond(const T* values, std::size_t count, double limit, T* copy) {
    std::size_t beyond = 0;
    for (std::size_t i = 0; i < count; i++) {
        const T value = values[i];
        if (copy != nullptr) {
            copy[i] = value;
        }
        beyond += !(static_cast<double>(std::abs(value)) <= limit) ? 1U : 0U;
    }
    return beyond;
}

// An output element left unwritten when allocated: for an output whose every
// element the binding writes, often from many threads, so it isn't filled
// once first. `to_numpy<T>` hands a vector of these to numpy as T, and
// `elements` writes them as T
template <typename T>
struct Unset {
    T value;
    // NOLINTNEXTLINE(modernize-use-equals-default,cppcoreguidelines-pro-type-member-init): left unwritten on purpose
    Unset() {}  // cppcheck-suppress uninitMemberVar ; left unwritten on purpose
};

template <typename T>
T* elements(std::vector<Unset<T>>& values) {
    static_assert(sizeof(Unset<T>) == sizeof(T) && std::is_standard_layout_v<Unset<T>>);
    return reinterpret_cast<T*>(values.data());  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same layout
}

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

// An instance's world box from `lowest` to `highest`, held to the bound every
// input is: a placement that carries its mesh past it is refused, naming the
// instance, as its boxes would be ones no input takes back (finite: finite
// inputs within the bound, by matrices within it, stay far from overflow)
inline void refuse_placed_beyond(const double* lowest, const double* highest, std::size_t instance) {
    for (std::size_t axis = 0; axis < 3; axis++) {
        // negated, so not a number is refused too
        if (!(std::abs(lowest[axis]) <= COORDINATE_LIMIT && std::abs(highest[axis]) <= COORDINATE_LIMIT)) {
            refuse("instance " + std::to_string(instance) + " is placed beyond " + COORDINATE_LIMIT_TEXT);
        }
    }
}

// How much of its farthest coordinate a box computed in double may miss its
// contents by: a transform, a center and half extent, and a gap each round
// on the order of an ulp of the coordinates, so a box padded by this never
// misses what it bounds. Every cull pads by it, as a cull may only ever keep
// too much, never drop what the exact test after it would find (bvh.cpp's
// nodes, collision.cpp's broadphase)
constexpr double BOX_SLACK = 1e-12;

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

// a row-major 4x4 transform that moves nothing, every instance's without matrices
constexpr double IDENTITY_TRANSFORM[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

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
        return given_matrices ? matrices.data() + (16 * i) : IDENTITY_TRANSFORM;
    }
};

// Rays' origins or directions, as a query gives them: `count` rows of three
// at `data`, or one row that is every ray's (`one`, with a count of 1),
// float64 or, for a float32 array taken as it is, float32
struct RayVectors {
    const void* data = nullptr;
    std::size_t count = 0;
    bool is_float32 = false;
    bool one = false;
};

// row `i` of rays as float64: a float32 widens to exactly itself
inline void ray_row(const RayVectors& rays, std::size_t i, double* row) {
    const std::size_t first = rays.one ? 0 : 3 * i;
    for (std::size_t k = 0; k < 3; k++) {
        row[k] = rays.is_float32 ? static_cast<double>(static_cast<const float*>(rays.data)[first + k])
                                 : static_cast<const double*>(rays.data)[first + k];
    }
}

// -- The pool door: every part's threads ---------------------------------------

// a part's loop that stopped at its Stop (Ctrl-C, or the interpreter's exit):
// its answer is incomplete, and src/trimkit.cpp raises instead (gil.hpp)
struct Stopped {};

// whether `stop` says to stop, the one way a part asks: once it says stop it
// keeps saying so, though a later look may say stop where an earlier one
// didn't. A Stop with no poll (a fuzz harness's) never stops
inline bool stopped(const parallel::Stop& stop) { return stop.poll != nullptr && stop.poll(stop.context); }

// The shape of every batch: `body(begin, end)` over [0, count) on the pool,
// `stop` polled between ranges, each item writing its own output, so the
// result doesn't depend on how the ranges fell. The first exception a range
// threw is rethrown, and Stopped if `stop` says to stop at one more look
// after the loop (once it said so in a range, it still does)
template <typename Body>
void in_parallel(std::size_t count, const Body& body, const parallel::Stop& stop) {
    const std::exception_ptr error = parallel::for_each_range(count, body, parallel::ItemCost::Measured, 1, stop);
    if (error) {
        std::rethrow_exception(error);
    }
    if (stopped(stop)) {
        throw Stopped{};
    }
}

// A vendored library's own fan-out: `body(context, begin, end)`
// over [0, count), called by the library while the binding already holds the
// GIL released, with no Ctrl-C poll (the library's progress callback polls),
// and the first exception a range threw rethrown here, on the library's
// thread, for it to unwind through
inline void run_ranges(std::size_t count, void (*body)(void* context, std::size_t begin, std::size_t end),
                       void* context) {
    const std::exception_ptr error = parallel::for_each_range(
        count, [&](std::size_t begin, std::size_t end) { body(context, begin, end); }, parallel::ItemCost::Coarse);
    if (error) {
        std::rethrow_exception(error);
    }
}

// `body(from, to)` over [0, count) for a copy with no Stop to poll (a
// device's rays copied into a chunk, its answers widened into the caller's
// columns), in ranges of at least 4096 items: a few bytes copied an item,
// which one at a time is too little to be worth a claim. A copy the caller
// finishes in a fraction of a millisecond wakes nobody
template <typename Body>
void in_blocks(std::size_t count, const Body& body) {
    constexpr std::size_t BLOCK = 4096;
    const std::exception_ptr error = parallel::for_each_range(count, body, parallel::ItemCost::Measured, BLOCK);
    if (error) {
        std::rethrow_exception(error);
    }
}

// the one lock there is (parallel.hpp)
using parallel::Exclusively;

// For a library that keeps scratch per thread (xatlas): every thread running
// the items of any one call on the pool has its own index below thread_slots()
inline std::uint32_t thread_slots() { return parallel::thread_slots(); }
inline std::uint32_t thread_index() { return parallel::thread_index(); }

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
        for (Chunk* chunk = list_.take_all(); chunk != nullptr;) {
            Chunk* next = chunk->next;
            delete chunk;  // NOLINT(cppcoreguidelines-owning-memory): the list owns its chunks
            chunk = next;
        }
    }

    // pushed by every thread while the query runs, never taken until it is
    // done (parallel.hpp, ChunkList)
    void push(std::unique_ptr<Chunk> chunk) { list_.push(chunk.release()); }

    // once the query is done: the chunks in item order, with their offsets
    void settle() {
        for (Chunk* chunk = list_.peek(); chunk != nullptr; chunk = chunk->next) {
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

    // a copy, so no Ctrl-C poll: it is over in a fraction of the query
    template <typename Write>
    void write(const Write& write) const {
        const std::exception_ptr error =
            parallel::for_each_range(chunks_.size(), [&](std::size_t begin, std::size_t end) {
                for (std::size_t c = begin; c < end; c++) {
                    const Chunk& chunk = *chunks_[c];
                    write(chunk.offset, chunk.rows.data(), chunk.rows.size());
                }
            });
        if (error) {
            std::rethrow_exception(error);
        }
    }

private:
    parallel::ChunkList<Chunk> list_;
    std::vector<Chunk*> chunks_;
    std::size_t total_ = 0;
};

template <typename T, typename Body>
void gather_in_parallel(std::size_t count, Gathered<T>* gathered, const Body& body, const parallel::Stop& stop) {
    in_parallel(
        count,
        [&](std::size_t begin, std::size_t end) {
            auto chunk = std::make_unique<typename Gathered<T>::Chunk>();
            chunk->begin = begin;
            body(begin, end, chunk->rows);
            if (!chunk->rows.empty()) {
                gathered->push(std::move(chunk));
            }
        },
        stop);
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

// How many of `values` are not within `limit` (count_beyond), a block at a
// time on the pool: the one scan of an input's values, for numpy.hpp's
// value checks (with the GIL released) and for a part that checks its own
template <typename T>
std::size_t count_beyond_in_parallel(const T* values, std::size_t count, double limit) {
    std::atomic<std::size_t> beyond{0};
    in_blocks(count, [&](std::size_t from, std::size_t to) {
        const std::size_t found = count_beyond<T>(values + from, to - from, limit, nullptr);
        if (found != 0) {
            beyond.fetch_add(found, std::memory_order_relaxed);
        }
    });
    return beyond.load(std::memory_order_relaxed);
}

// -- What each part's unit does (src/NAME.cpp), as src/trimkit.cpp calls it --

// triangulate.cpp: the triangles of a polygon whose first ring is its shell
// and the rest its holes, `vertices` the rings stacked in order (2 a row)
// and `faces` 3 rows of `vertices` a triangle, counter-clockwise. Of exactly
// equal vertices only the first is named. Refuses rings CDT can't take
// (edges that cross), and 2**31 vertices or more (2**30 on 32-bit)
enum class TriangulateEngine : std::uint8_t { EARCUT, CDT };
struct PolygonRing {
    const double* xy;  // `count` rows of x, y
    std::size_t count;
};
void triangulate_polygon(const std::vector<PolygonRing>& rings, TriangulateEngine engine, std::vector<double>* vertices,
                         std::vector<std::uint64_t>* faces);

// meshoptimizer.cpp: edges collapsed by quadric error until at most
// `face_count` faces remain, or before the surface would move farther than
// `maximum_error` in the units of `vertices`, open borders kept when
// `lock_border`. The output vertices are exactly `vertices[vertex_sources]`,
// and `output_faces` index them, 3 a triangle. Refuses a negative face_count
// or maximum_error
void simplify_mesh(const double* vertices, std::size_t vertex_count, const std::vector<std::uint32_t>& indices,
                   std::int64_t face_count, double maximum_error, bool lock_border,
                   std::vector<double>* output_vertices, std::vector<std::int64_t>* output_faces,
                   std::vector<std::int64_t>* vertex_sources);

// xatlas.cpp: a mesh cut into charts packed into a UV atlas: each output
// vertex's input vertex, the output faces (3 a triangle), and its uv in the
// unit square (2 a vertex). Polls `stop` (Stopped once it says to stop), and
// refuses what xatlas can't take
void parametrize_mesh(const double* vertices, std::size_t vertex_count, const std::vector<std::uint32_t>& indices,
                      const parallel::Stop& stop, std::vector<std::int64_t>* vertex_sources,
                      std::vector<std::int64_t>* faces, std::vector<double>* uv);

// vhacd.cpp: a closed mesh decomposed into approximately convex hulls, each
// its own vertices (3 a row) and faces indexing them (3 a triangle): none for
// input with no volume. Polls `stop` (Stopped once it says to stop), and
// refuses parameters V-HACD would misbehave on
struct ConvexHullMesh {
    std::vector<double> vertices;
    std::vector<std::int64_t> faces;
};
std::vector<ConvexHullMesh> decompose_mesh(const double* vertices, std::size_t vertex_count,
                                           const std::vector<std::uint32_t>& triangles, std::int64_t maximum_hulls,
                                           std::int64_t maximum_vertices, std::int64_t resolution, double maximum_error,
                                           const parallel::Stop& stop);

// draco.cpp: a glTF draco buffer, encoded sequentially so draco keeps the
// vertex order, and decoded once. A component is a glTF accessor's type, and
// a kind a DracoKind value, each in this order. Refuses what draco can't
// encode, and a buffer whose faces or attributes name values it doesn't hold
enum class DracoComponent : std::uint8_t { INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32 };
constexpr std::size_t DRACO_COMPONENTS = 7;
// the bytes a value of each DracoComponent takes, in its order
constexpr std::size_t DRACO_COMPONENT_BYTES[DRACO_COMPONENTS] = {1, 1, 2, 2, 4, 4, 4};
constexpr std::size_t DRACO_KINDS = 5;  // POSITION, NORMAL, TEXCOORD, COLOR, GENERIC
struct DracoAttributeInput {
    const void* data;  // `count` rows of `components` values, tightly packed
    DracoComponent component;
    std::size_t components;
};
struct DracoAttributeOutput {
    std::vector<std::uint8_t> values;  // one row of `components` a point, tightly packed
    DracoComponent component = DracoComponent::UINT8;
    std::size_t components = 0;
};
// `bits` none for each kind's default quantization
std::vector<char> draco_encode_mesh(const std::vector<std::uint32_t>& triangles, std::size_t count,
                                    const std::vector<DracoAttributeInput>& attributes,
                                    const std::vector<std::int64_t>& kinds, std::optional<std::int64_t> bits);
void draco_decode_mesh(const std::uint8_t* buffer, std::size_t size, const std::vector<std::int64_t>& ids,
                       std::vector<std::int64_t>* faces, std::vector<DracoAttributeOutput>* attributes);

// collision.cpp: meshes built once into coal's hierarchies (and a convex
// shape for each declared convex), then placed rigidly by each query's own
// `Instances` (read rigid, against size()) and tested or measured in pairs
// on the pool. A query's `pairs` are 2 a row, or none for every pair; each
// polls `stop` between ranges (Stopped once it says to stop)
struct CollisionMeshInput {
    const double* vertices;  // `vertex_count` rows of x, y, z
    std::size_t vertex_count;
    const std::uint32_t* faces;  // `face_count` rows of 3, indexing `vertices`
    std::size_t face_count;
    bool convex;  // measured as the solid it bounds, the caller's promise
};
// One row of an answer: a pair of instances, or one contact of a pair, with
// the face of each instance's mesh it is at (-1 for a convex solid), its
// signed distance, the witness point on each instance, and the unit
// direction from the first toward the second
struct CollisionRow {
    std::int64_t pair[2];
    std::int64_t faces[2];
    double distance;
    double points[6];
    double normal[3];
};
class CollisionMesh;  // one mesh's hierarchy and convex shape, collision.cpp's own
class CollisionMeshes {
public:
    // each mesh in order: refuses one declared convex without faces
    explicit CollisionMeshes(const std::vector<CollisionMeshInput>& meshes);
    ~CollisionMeshes();
    CollisionMeshes(const CollisionMeshes&) = delete;
    CollisionMeshes& operator=(const CollisionMeshes&) = delete;
    CollisionMeshes(CollisionMeshes&&) = delete;
    CollisionMeshes& operator=(CollisionMeshes&&) = delete;

    std::size_t size() const { return meshes_.size(); }
    bool convex(std::size_t mesh) const;
    // a mesh as it was given, from the model's own copy: 3 a row each
    void as_given(std::size_t mesh, std::vector<double>* vertices, std::vector<std::int64_t>* faces) const;
    // the box each instance is culled by, minimum then maximum corner
    std::vector<double> boxes(const Instances& instances) const;
    // the pairs that touch or come within `margin`, the given ones in order
    // or by default each i < j in ascending order, into `touching` (2 a
    // row); or with `contacts`, up to `maximum_contacts` rows a pair into it
    // instead. Refuses a margin past COORDINATE_LIMIT or negative, a
    // maximum_contacts below 1, and pairs out of range or naming one twice
    void collides(const Instances& instances, const std::optional<std::vector<std::int64_t>>& pairs, double margin,
                  std::int64_t maximum_contacts, const parallel::Stop& stop, std::vector<Unset<std::int64_t>>* touching,
                  std::vector<Unset<CollisionRow>>* contacts) const;
    // one row per pair, by default every i < j in numpy.triu_indices order:
    // past `upper_bound` infinitely far, with -1 faces and NaN points and
    // normal. Refuses an upper_bound not positive, and pairs as `collides`
    std::vector<Unset<CollisionRow>> distances(const Instances& instances,
                                               const std::optional<std::vector<std::int64_t>>& pairs,
                                               double upper_bound, const parallel::Stop& stop) const;

private:
    std::vector<std::unique_ptr<CollisionMesh>> meshes_;
};

// bvh.cpp: rays cast against meshes built once into tinybvh's double
// precision BVHs, each query's instances placed under a top level, traced on
// the CPU's pool or, for a scene that asked, on a Vulkan device
// (src/vulkan/scene.hpp) to the same bytes. A query's `origins` and
// `directions` are `count_rays` rays; each refuses values of them past
// COORDINATE_LIMIT, polls `stop` between ranges on the CPU (Stopped once it
// says to stop) and never on the device, which is over in milliseconds, and
// returns why a device fell back to the CPU, or nothing
struct RayMeshInput {
    const double* vertices = nullptr;  // `vertex_count` rows of x, y, z
    std::size_t vertex_count = 0;
    std::vector<std::uint64_t> faces;  // 3 a triangle, indexing `vertices`: the mesh takes them
};
// The four columns `intersect` answers, one row per hit, each written once
struct RayCrossings {
    std::vector<Unset<std::int64_t>> rays;
    std::vector<Unset<std::int64_t>> instances;
    std::vector<Unset<std::int64_t>> faces;
    std::vector<Unset<double>> distances;
    void resize(std::size_t total) {
        rays.resize(total);
        instances.resize(total);
        faces.resize(total);
        distances.resize(total);
    }
};
// how many rays: a row of each per ray, where neither is one for every ray.
// Refuses rows of each that differ in number
std::size_t count_rays(const RayVectors& origins, const RayVectors& directions);
struct RayMeshTrees;  // every mesh's BVH and the geometry it indexes, and their copy on a device: bvh.cpp's own
struct RayTopLevel;   // one query's instances placed under a top level, never changed once built: bvh.cpp's own
class RayMeshes {
public:
    // each mesh built once from copies (tinybvh keeps pointers into its
    // geometry), and with `use_gpu` put on the process's device, or
    // `device_failure` says why it can't be and the CPU answers
    RayMeshes(std::vector<RayMeshInput> meshes, bool use_gpu);
    ~RayMeshes();
    RayMeshes(const RayMeshes&) = delete;
    RayMeshes& operator=(const RayMeshes&) = delete;
    RayMeshes(RayMeshes&&) = delete;
    RayMeshes& operator=(RayMeshes&&) = delete;

    std::size_t size() const;
    const std::string& device_failure() const;
    // a mesh as it was given, from the scene's own copy: 3 a row each
    void as_given(std::size_t mesh, std::vector<double>* vertices, std::vector<std::uint64_t>* faces) const;
    // `instances` (read against size(), not rigid) placed under a top level
    // for any number of queries to share, and put on the device beside the
    // meshes when they are there. Refuses a transform not invertible with an
    // inverse within COORDINATE_LIMIT
    std::shared_ptr<const RayTopLevel> place(Instances instances) const;
    // the instances a top level was placed from, as they were read
    static const Instances& placed_from(const RayTopLevel& top);
    // the box each instance is culled by, minimum then maximum corner, placed
    // as `place` places it without building a top level. Refuses as it does
    std::vector<double> boxes(Instances instances) const;
    // whether each ray hits anything under `top`, into `hits` (0 or 1 each)
    std::string intersect_any(const RayTopLevel& top, const RayVectors& origins, const RayVectors& directions,
                              std::size_t count, const parallel::Stop& stop, std::vector<std::uint8_t>* hits) const;
    // where each ray crosses the surface under `top`, nearest first and up
    // to `most` a ray, into `columns` sorted by ray and then distance
    std::string intersect(const RayTopLevel& top, const RayVectors& origins, const RayVectors& directions,
                          std::size_t count, std::size_t most, const parallel::Stop& stop, RayCrossings* columns) const;

private:
    std::unique_ptr<RayMeshTrees> trees_;
};

// bvh.cpp: axis-aligned boxes in 2D or 3D built once into a BVH over their
// bounds, a static bulk-loaded R-tree. `count` boxes or queries are each a
// minimum then a maximum corner of `dimension` float64, already finite and
// within COORDINATE_LIMIT, and one query (`one`) answers its boxes where many
// answer [query, box] rows. Each query polls `stop` between ranges (Stopped
// once it says to stop)
struct BoxBvh;  // the boxes as 3D corners and tinybvh's BVH over them, bvh.cpp's own
class BoxTree {
public:
    // copies of the boxes: refuses one with a minimum above its maximum
    BoxTree(const double* boxes, std::size_t count, std::size_t dimension);
    ~BoxTree();
    BoxTree(BoxTree&& other) noexcept;
    BoxTree& operator=(BoxTree&& other) noexcept;
    BoxTree(const BoxTree&) = delete;
    BoxTree& operator=(const BoxTree&) = delete;

    std::size_t size() const;
    std::size_t dimension() const;
    // the box around every box, (2, d): +inf then -inf with none
    std::vector<double> bounds() const;
    // every box as given, (n, 2, d), exactly
    std::vector<double> boxes() const;
    // every box overlapping each query, touching included like rtree: each
    // query's ascending. Refuses a query with a minimum above its maximum
    std::vector<Unset<std::int64_t>> intersection(const double* queries, std::size_t count, bool one,
                                                  const parallel::Stop& stop) const;
    // the `maximum_hits` nearest boxes to each query, as many as there are
    // up to that, by distance and then box, with each row's distance: every
    // query has exactly that many. Refuses as `intersection` does
    void nearest(const double* queries, std::size_t count, bool one, std::size_t maximum_hits,
                 const parallel::Stop& stop, std::vector<Unset<std::int64_t>>* hits,
                 std::vector<Unset<double>>* distances) const;

private:
    std::unique_ptr<BoxBvh> tree_;
};

}  // namespace trimkit
