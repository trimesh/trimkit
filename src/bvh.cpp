// Ray queries against an instanced scene of triangle meshes, and box overlap
// queries against a set of boxes, with jbikker/tinybvh. Every ray query is
// tinybvh's own nearest-hit or any-hit traversal: all the hits along a ray are
// its nearest hit asked again from just past each one, so no traversal or
// triangle test of tinybvh's is reimplemented here. Box overlap, which tinybvh
// has no query for, walks its nodes.
//
// A `RayScene` builds each of its meshes once, in double precision (with
// spatial splits when it is big enough to gain from them), from copies of
// its inputs: tinybvh keeps pointers into its geometry, so the scene owns
// it. Every query names its instances (which mesh, placed where), placed
// under one top-level BVH without copying their triangles: one identity
// instance skips the top level entirely. A top level is built from exactly
// the instance arrays given, so the scene remembers the last one it built
// with those arrays' bytes, and a query naming the same bytes again reuses it
// rather than rebuilding: its answers are the same either way, only sooner.
// Building and every batch of queries run with the GIL released, large
// batches across every core, and queries are const, so threads may share one
// scene; the remembered top level is swapped whole, and a query keeps the
// one it started with.
//
// A scene built with `use_gpu` holds its meshes on a Vulkan device too
// (src/vulkan/vulkan.hpp) and traces there: the same trees, and a shader
// (src/vulkan/trace.comp) that makes every decision tinybvh's traversal makes,
// most in f32 behind a proven bound and the rest with tinybvh's own f64
// operations, so the same bytes come back; the crossing loop below
// (`settle`) runs in the shader too, written the same way. A device that
// can't be used, now or later, is a warning once, and the CPU answers.

#include "common.hpp"
#include "vulkan/vulkan.hpp"

#include <tiny_bvh.h>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Vec = tinybvh::bvhdbl3;
using Transforms = trimkit::Array<double, -1, 4, 4>;
using Numbers = trimkit::Array<std::int64_t, -1>;
using Points = trimkit::In<double, -1, 3>;
// rays' origins or directions: one each, or one for every ray
using RaysArg = trimkit::RowsArgument<3>;
using Rays = trimkit::Rows<3>;
using FacesArg = trimkit::Array<std::int64_t, -1, 3>;
using BoundsArg = trimkit::Array<double, -1, 2, -1>;
using Bounds = trimkit::In<double, -1, 2, -1>;

// A BVH as tinybvh builds it, with its test of a ray against one known
// triangle callable: tinybvh keeps that protected, and every hit of
// `intersect` is measured with it
struct MeshBvh : tinybvh::BVH_Double {
    using tinybvh::BVH_Double::IntersectTri;
};

// tinybvh's nodes come through the extension's one door for allocation
// (src/trimkit.cpp), rather than the aligned malloc it would otherwise call
void* allocate_aligned(std::size_t size, void* /*unused*/) { return ::operator new(size, std::align_val_t{64}); }

void free_aligned(void* block, void* /*unused*/) { ::operator delete(block, std::align_val_t{64}); }

std::unique_ptr<MeshBvh> new_bvh() {
    std::unique_ptr<MeshBvh> bvh = std::make_unique<MeshBvh>();
    bvh->context.malloc = allocate_aligned;
    bvh->context.free = free_aligned;
    return bvh;
}

// A watertight triangle test needs conservative traversal: the slab test
// rounds on the order of an ulp of the coordinates, so without padding a ray
// through a vertex or edge can skip the nodes holding it (Ize 2013). Padding
// by this much of the largest coordinate covers ray origins up to thousands
// of times farther out than the geometry
void pad(tinybvh::BVH_Double& bvh, double extent) {
    const Vec margin(1e-12 * extent);
    for (std::uint64_t i = 0; i < bvh.usedNodes; i++) {
        tinybvh::BVH_Double::BVHNode& node = bvh.bvhNode[i];
        node.aabbMin = node.aabbMin - margin;
        node.aabbMax = node.aabbMax + margin;
    }
}

// whether the boxes [a_lo, a_hi] and [b_lo, b_hi] overlap, touching included like rtree
bool overlaps(const Vec& a_lo, const Vec& a_hi, const Vec& b_lo, const Vec& b_hi) {
    for (std::int32_t axis = 0; axis < 3; axis++) {
        if (b_lo[axis] > a_hi[axis] || b_hi[axis] < a_lo[axis]) {
            return false;
        }
    }
    return true;
}

// A traversal stack of node indices for the box overlap walk: one per chunk
// of queries, reused for every query in it, so it allocates once and grows
// only on a pathological tree (deeply nested boxes) rather than overflowing
using Stack = std::vector<std::uint64_t>;

Stack make_stack() {
    Stack stack;
    stack.reserve(std::size_t{2} * TINYBVH_STACK_SIZE);
    return stack;
}

std::uint64_t pop(Stack& stack) {
    const std::uint64_t node = stack.back();
    stack.pop_back();
    return node;
}

tinybvh::RayEx make_tinybvh_ray(const double* o, const double* d) {
    return {Vec(o[0], o[1], o[2]), Vec(d[0], d[1], d[2])};
}

// a ray without a direction goes nowhere and hits nothing
bool still(const double* d) { return d[0] == 0.0 && d[1] == 0.0 && d[2] == 0.0; }

// The length of a direction, scaled by its largest component so tiny and
// huge directions neither underflow nor overflow: libstdc++'s three-argument
// hypot written out, so it is the same IEEE operations (and the same bits) on
// every platform and on the device (src/vulkan/trace.comp), where libm's hypot is not
double length(const double* d) {
    const double x = std::abs(d[0]);
    const double y = std::abs(d[1]);
    const double z = std::abs(d[2]);
    const double largest = std::max({x, y, z});
    if (largest == 0.0) {
        return 0.0;
    }
    const double u = x / largest;
    const double v = y / largest;
    const double w = z / largest;
    return largest * std::sqrt(((u * u) + (v * v)) + (w * w));
}

// how many rays: a row of each per ray, where neither is one for every ray
std::size_t count_rays(const Rays& origins, const Rays& directions) {
    if (!origins.one && !directions.one && directions.count != origins.count) {
        trimkit::value_error("origins and directions must have the same number of rays");
    }
    return origins.one ? directions.count : origins.count;
}

// The f32 image of one triangle for the device's filter (src/vulkan/trace.comp,
// docs/trace.md 4.1, 5.2): the corners, a radius never below any coordinate's
// magnitude, the f64 normal and `nu`, a bound on that normal's distance from
// the exact one: the edges' roundings, the products' and difference's, the
// result's and the f32 conversion's, inflated by 2^-20 for this arithmetic's own
trimkit::vulkan::Tri32 triangle32(const std::vector<Vec>& vertices, const std::vector<std::uint64_t>& faces,
                                  std::uint64_t face) {
    constexpr double U64 = 1.1102230246251565e-16;         // 2^-53
    constexpr double U32 = 5.960464477539063e-08;          // 2^-24
    constexpr double INFLATE = 1.0 + 9.5367431640625e-07;  // 1 + 2^-20
    const Vec& v0 = vertices[faces[3 * face]];
    const Vec& v1 = vertices[faces[(3 * face) + 1]];
    const Vec& v2 = vertices[faces[(3 * face) + 2]];
    trimkit::vulkan::Tri32 image{};
    double radius = 0.0;
    for (std::int32_t axis = 0; axis < 3; axis++) {
        image.p0[axis] = static_cast<float>(v0[axis]);
        image.p1[axis] = static_cast<float>(v1[axis]);
        image.p2[axis] = static_cast<float>(v2[axis]);
        radius = std::max({radius, std::abs(v0[axis]), std::abs(v1[axis]), std::abs(v2[axis])});
    }
    image.radius = std::nextafter(static_cast<float>(radius), std::numeric_limits<float>::infinity());
    image.prim = static_cast<std::uint32_t>(face);
    const Vec e1 = v1 - v0;
    const Vec e2 = v2 - v0;
    const double n[3] = {(e1[1] * e2[2]) - (e1[2] * e2[1]), (e1[2] * e2[0]) - (e1[0] * e2[2]),
                         (e1[0] * e2[1]) - (e1[1] * e2[0])};
    const double a1[3] = {std::abs(e1[0]), std::abs(e1[1]), std::abs(e1[2])};
    const double a2[3] = {std::abs(e2[0]), std::abs(e2[1]), std::abs(e2[2])};
    const double products[3] = {(a1[1] * a2[2]) + (a1[2] * a2[1]), (a1[2] * a2[0]) + (a1[0] * a2[2]),
                                (a1[0] * a2[1]) + (a1[1] * a2[0])};
    double nu = 0.0;
    for (std::int32_t axis = 0; axis < 3; axis++) {
        image.n[axis] = static_cast<float>(n[axis]);
        nu = std::max(nu, (4 * U64 * products[axis]) + (2 * U64 * std::abs(n[axis])) + (U32 * std::abs(n[axis])));
    }
    image.nu = std::nextafter(static_cast<float>(nu * INFLATE), std::numeric_limits<float>::infinity());
    return image;
}

// a tree's nodes copied as the device reads them
std::vector<trimkit::vulkan::Node64> nodes_of(const tinybvh::BVH_Double& bvh) {
    static_assert(sizeof(tinybvh::BVH_Double::BVHNode) == sizeof(trimkit::vulkan::Node64));
    std::vector<trimkit::vulkan::Node64> nodes(bvh.usedNodes);
    std::memcpy(nodes.data(), bvh.bvhNode, nodes.size() * sizeof(trimkit::vulkan::Node64));
    return nodes;
}

// A scene's meshes, or one top level, on the device: uploaded once from the
// trees the CPU traces, freed with the scene or top level they mirror
struct OnDevice {
    trimkit::vulkan::Device* device = nullptr;
    trimkit::vulkan::Arrays arrays;
    OnDevice() = default;
    OnDevice(const OnDevice&) = delete;
    OnDevice& operator=(const OnDevice&) = delete;
    ~OnDevice() { trimkit::vulkan::destroy(*device, &arrays); }
};

// `body(from, to)` over [0, count), the cores sharing it a block at a time: for
// a few bytes copied per item, which one at a time is too little for the pool
// to measure (and the first of them takes a page fault)
template <typename Body>
void in_blocks(std::size_t count, const Body& body) {
    constexpr std::size_t BLOCK = 4096;
    trimkit::in_parallel_without_gil((count + BLOCK - 1) / BLOCK, [&](std::size_t first, std::size_t last) {
        body(first * BLOCK, std::min(count, last * BLOCK));
    });
}

// a vector's bytes, to upload
template <typename T>
trimkit::vulkan::Span bytes_of(const std::vector<T>& values) {
    return {values.data(), values.size() * sizeof(T)};
}

// a device takes one submitter at a time
trimkit::Exclusive& device_lock() {
    static trimkit::Exclusive lock;
    return lock;
}

// One hit as `intersect` reports it
struct Hit {
    std::int64_t ray;
    std::int64_t instance;
    std::int64_t face;
    double t;
};

// The four columns `intersect` answers, each written once; the device fills
// them a chunk at a time, into room sized by its first chunk
struct Columns {
    std::vector<trimkit::Unset<std::int64_t>> rays;
    std::vector<trimkit::Unset<std::int64_t>> instances;
    std::vector<trimkit::Unset<std::int64_t>> faces;
    std::vector<trimkit::Unset<double>> distances;
    void resize(std::size_t total) {
        rays.resize(total);
        instances.resize(total);
        faces.resize(total);
        distances.resize(total);
    }
};

// A ray's progress through its crossings, see RayScene::settle
struct Crossing {
    double along = 0.0;  // how far the origin has moved along the ray
    double step = 0.0;   // the move past the last hit
    double last_t = 0.0;
    std::int64_t last_instance = -1;
    std::int64_t last_face = -1;
    std::size_t count = 0;
};

// What one traversal found for a ray: nothing, or a face of an instance (a
// row of the top level's instances) at a distance in the direction's units
struct Found {
    bool hit = false;
    std::uint64_t instance = 0;
    std::uint64_t face = 0;
    double t = 0.0;
};

// One mesh of a RayScene, built once however often the scene places it
class RayMesh {
public:
    explicit RayMesh(trimkit::Mesh<std::uint64_t> mesh) : faces_(std::move(mesh.faces)) {
        const Points& vertices = mesh.vertices;
        // copying reads every value, so it runs without the GIL too
        const nb::gil_scoped_release release;
        const double* v = vertices.data();
        vertices_.resize(vertices.shape(0));
        for (std::size_t i = 0; i < vertices.shape(0); i++) {
            vertices_[i] = Vec(v[3 * i], v[(3 * i) + 1], v[(3 * i) + 2]);
        }
        // the largest coordinate's magnitude
        const trimkit::PointBounds bounds = trimkit::bounds_of(v, vertices.shape(0));
        double extent = 0.0;
        for (std::size_t axis = 0; axis < 3 && !vertices_.empty(); axis++) {
            extent = std::max({extent, std::abs(bounds.lowest[axis]), std::abs(bounds.highest[axis])});
        }
        if (!faces_.empty()) {
            bvh_ = new_bvh();
            // spatial splits trace 20-27% faster from about 300k triangles, for
            // 40% more build time, but gain nothing below 50k at 4x the build
            const std::size_t count = faces_.size() / 3;
            if (count >= 100000) {
                bvh_->BuildHQ(vertices_.data(), faces_.data(), count);
            } else {
                bvh_->Build(vertices_.data(), faces_.data(), count);
            }
            pad(*bvh_, extent);
        }
    }

    // the BVH scenes share, or null for a mesh without faces
    MeshBvh* bvh() const { return bvh_.get(); }
    std::size_t size() const { return faces_.size() / 3; }
    // the geometry the BVH indexes, as the device gets it too
    const std::vector<Vec>& vertices() const { return vertices_; }
    const std::vector<std::uint64_t>& faces() const { return faces_; }

private:
    std::vector<Vec> vertices_;
    std::vector<std::uint64_t> faces_;
    // declared last, so it is destroyed before the geometry it points into
    std::unique_ptr<MeshBvh> bvh_;
};

using Hits = std::tuple<trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>,
                        trimkit::Out<double>>;
using MaximumHits = std::optional<trimkit::Number<std::int64_t>>;

// Instances as a scene places them: each checked transform with its
// inverse, kept only for meshes with faces (`which` holds each kept one's
// row), and every row's world box: its mesh's root box carried by the
// transform, as tinybvh's top level bounds it (the root as tinybvh keeps it,
// unpadded, as `pad` widens only the nodes; the top level's own padding
// covers the rounding), or for a mesh without faces, a point at its
// translation, as 6 values, minimum then maximum
struct Placement {
    std::vector<tinybvh::BLASInstanceEx> instances;
    std::vector<std::int64_t> which;
    std::vector<double> boxes;
    double extent = 0.0;
    bool all_identity = true;
};

// A placement built into a top level: the instance arrays it was built from,
// byte for byte (each None or not), each kept instance (a mesh with faces)
// with its caller's row, and the BVH over them, or the one identity
// instance's mesh, traced directly. Never changed once built
struct TopLevel {
    bool given_index = false;
    bool given_matrices = false;
    std::vector<std::int64_t> index;
    std::vector<double> matrices;
    std::vector<tinybvh::BLASInstanceEx> instances;
    std::vector<std::int64_t> which;
    // the largest coordinate magnitude any placed box reaches: the scene's size
    double extent = 0.0;
    const RayMesh* direct = nullptr;
    // the top level on the device when the scene traces there, or why not
    std::unique_ptr<OnDevice> on_device;
    std::string device_reason;
    // declared last, so it is destroyed before the instances it points into
    std::unique_ptr<MeshBvh> tlas;

    // what a first-hit or any-hit ray traces, or null with nothing to hit
    const tinybvh::BVH_Double* bvh() const { return direct != nullptr ? direct->bvh() : tlas.get(); }
};

// Whether a query names exactly the instance arrays `top` was built from:
// both None, or the same bytes, compared without rechecking their values
bool names(const TopLevel& top, const Transforms& matrices_argument, const Numbers& index_argument) {
    if (top.given_index == index_argument.is_none() || top.given_matrices == matrices_argument.is_none()) {
        return false;
    }
    return (!top.given_index || trimkit::same_bytes(index_argument, top.index, "instance_index")) &&
           (!top.given_matrices || trimkit::same_bytes(matrices_argument, top.matrices, "instance_matrices"));
}

class RayScene {
public:
    // it owns its meshes (std::vector of std::unique_ptr claims to be
    // copyable to nanobind's traits, so say it isn't)
    RayScene(const RayScene&) = delete;
    RayScene& operator=(const RayScene&) = delete;

    RayScene(const trimkit::Meshes& meshes, trimkit::Flag use_gpu) {
        for (const nb::handle item : meshes) {
            const std::string name = "meshes[" + std::to_string(meshes_.size()) + "]";
            meshes_.push_back(std::make_unique<RayMesh>(trimkit::read_mesh<std::uint64_t>(item, name)));
        }
        blas_.resize(meshes_.size());
        for (std::size_t m = 0; m < meshes_.size(); m++) {
            blas_[m] = meshes_[m]->bvh();
        }
        if (use_gpu.value) {
            std::string reason;
            if (!open_device(&reason)) {
                trimkit::warn("use_gpu: " + reason + "; this scene answers on the CPU");
            }
        }
        // the default instances, each mesh once where it is, built now
        last_ = build(Transforms{nb::none()}, Numbers{nb::none()});
    }

    // the boxes a query with these instances culls rays by, without building
    // a top level: the same placement a query makes
    trimkit::BoxesObject boxes(const Transforms& matrices_argument, const Numbers& index_argument) const {
        const trimkit::Instances instances = trimkit::read_instances(index_argument, matrices_argument, meshes_.size());
        return trimkit::scene_boxes(place(instances).boxes);
    }

    std::size_t size() const { return meshes_.size(); }

    trimkit::Out<bool> intersect_any(const RaysArg& origins_argument, const RaysArg& directions_argument,
                                     const Transforms& matrices_argument, const Numbers& index_argument) const {
        const Rays origins = trimkit::read_rows(origins_argument, "origins");
        const Rays directions = trimkit::read_rows(directions_argument, "directions");
        const std::size_t n = count_rays(origins, directions);
        const std::shared_ptr<const TopLevel> top = top_level(matrices_argument, index_argument);
        std::vector<std::uint8_t> hits(n, 0);
        const tinybvh::BVH_Double* bvh = top->bvh();
        std::string fallback;
        if (bvh == nullptr || !device_occlusion(*top, origins, directions, n, &hits, &fallback)) {
            // the device checks each ray's values as it copies them: here, now
            trimkit::check_rows(origins, "origins");
            trimkit::check_rows(directions, "directions");
            trimkit::in_parallel(bvh != nullptr ? n : 0, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; i++) {
                    double o[3];
                    double d[3];
                    origins.read(i, o);
                    directions.read(i, d);
                    if (!still(d)) {
                        const tinybvh::RayEx ray = make_tinybvh_ray(o, d);
                        hits[i] = bvh->IsOccluded(ray) ? 1 : 0;
                    }
                }
            });
        }
        warn_fallback(fallback);
        return trimkit::to_numpy<bool>(std::move(hits), {n});
    }

    Hits intersect(const RaysArg& origins_argument, const RaysArg& directions_argument,
                   const Transforms& matrices_argument, const Numbers& index_argument,
                   const MaximumHits& maximum_hits) const {
        if (maximum_hits && *maximum_hits < 1) {
            trimkit::value_error("maximum_hits must be at least 1");
        }
        const std::size_t most = maximum_hits ? static_cast<std::size_t>(std::int64_t{*maximum_hits})
                                              : std::numeric_limits<std::size_t>::max();
        const Rays origins = trimkit::read_rows(origins_argument, "origins");
        const Rays directions = trimkit::read_rows(directions_argument, "directions");
        const std::size_t n = count_rays(origins, directions);
        const std::shared_ptr<const TopLevel> top = top_level(matrices_argument, index_argument);
        const TopLevel& level = *top;
        // the device writes the columns itself; the CPU gathers hits a chunk of
        // rays at a time and writes them after
        Columns columns;
        std::string fallback;
        if (level.bvh() == nullptr || !device_crossings(level, origins, directions, n, most, &columns, &fallback)) {
            // the device checks each ray's values as it copies them: here, now
            trimkit::check_rows(origins, "origins");
            trimkit::check_rows(directions, "directions");
            trimkit::Gathered<Hit> found;
            if (level.bvh() != nullptr) {
                trimkit::gather_in_parallel<Hit>(
                    n, &found, [&](std::size_t begin, std::size_t end, std::vector<Hit>& hits) {
                        // room for the usual: a hit per ray with a cap of one, else
                        // two, a closed surface entered and left
                        hits.reserve((end - begin) * std::min<std::size_t>(most, 2));
                        for (std::size_t i = begin; i < end; i++) {
                            double o[3];
                            double d[3];
                            origins.read(i, o);
                            directions.read(i, d);
                            if (still(d)) {
                                continue;
                            }
                            const std::int64_t ray = static_cast<std::int64_t>(i);
                            // distances in the scene's units rather than the direction's
                            const double scale = length(d);
                            crossings(level, o, d, scale, most,
                                      [&](std::int64_t instance, std::int64_t face, double t) {
                                          hits.push_back({ray, instance, face, t * scale});
                                      });
                        }
                    });
            }
            columns.resize(found.size());
            std::int64_t* rays_out = trimkit::elements(columns.rays);
            std::int64_t* instances_out = trimkit::elements(columns.instances);
            std::int64_t* faces_out = trimkit::elements(columns.faces);
            double* distances_out = trimkit::elements(columns.distances);
            found.write([&](std::size_t offset, const Hit* hits, std::size_t count) {
                for (std::size_t h = 0; h < count; h++) {
                    rays_out[offset + h] = hits[h].ray;
                    instances_out[offset + h] = hits[h].instance;
                    faces_out[offset + h] = hits[h].face;
                    distances_out[offset + h] = hits[h].t;
                }
            });
        }
        warn_fallback(fallback);
        const std::size_t total = columns.rays.size();
        return {trimkit::to_numpy<std::int64_t>(std::move(columns.rays), {total}),
                trimkit::to_numpy<std::int64_t>(std::move(columns.instances), {total}),
                trimkit::to_numpy<std::int64_t>(std::move(columns.faces), {total}),
                trimkit::to_numpy<double>(std::move(columns.distances), {total})};
    }

private:
    // The top level for these instances: the remembered one if they are
    // exactly the arrays it was built from, else a new one, remembered
    std::shared_ptr<const TopLevel> top_level(const Transforms& matrices_argument,
                                              const Numbers& index_argument) const {
        std::shared_ptr<const TopLevel> last = remembered();
        if (names(*last, matrices_argument, index_argument)) {
            return last;
        }
        std::shared_ptr<const TopLevel> built = build(matrices_argument, index_argument);
        {
            const nb::ft_lock_guard guard(last_lock_);
            last_ = built;
        }
        return built;
    }

    std::shared_ptr<const TopLevel> remembered() const {
        const nb::ft_lock_guard guard(last_lock_);
        return last_;
    }

    // A top level for these instances, checked, from a copy of their arrays
    std::shared_ptr<const TopLevel> build(const Transforms& matrices_argument, const Numbers& index_argument) const {
        const trimkit::Instances instances = trimkit::read_instances(index_argument, matrices_argument, meshes_.size());
        Placement placement = place(instances);
        std::shared_ptr<TopLevel> top = std::make_shared<TopLevel>();
        top->given_index = instances.given_index;
        top->given_matrices = instances.given_matrices;
        if (instances.given_index) {
            top->index.assign(instances.index.data(), instances.index.data() + instances.count);
        }
        if (instances.given_matrices) {
            top->matrices.assign(instances.matrices.data(), instances.matrices.data() + (16 * instances.count));
        }
        top->instances = std::move(placement.instances);
        top->which = std::move(placement.which);
        top->extent = placement.extent;
        // one identity instance traces its mesh directly: the same answers as
        // through a top level, without the extra level per ray (19% of a query)
        if (top->instances.size() == 1 && placement.all_identity) {
            top->direct = meshes_[static_cast<std::size_t>(top->instances[0].blasIdx)].get();
        } else if (!top->instances.empty()) {
            const nb::gil_scoped_release release;
            top->tlas = new_bvh();
            // tinybvh takes the meshes' array non-const, though it only reads it
            tinybvh::BVH_Double** meshes =
                const_cast<tinybvh::BVH_Double**>(blas_.data());  // NOLINT(cppcoreguidelines-pro-type-const-cast)
            top->tlas->Build(top->instances.data(), top->instances.size(), meshes, blas_.size());
            pad(*top->tlas, placement.extent);
        }
        if (on_device_ && top->tlas) {
            const nb::gil_scoped_release release;
            upload_top(top.get());
        }
        return top;
    }

    Placement place(const trimkit::Instances& instances) const {
        constexpr std::array<double, 16> IDENTITY{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        Placement placement;
        placement.boxes.resize(6 * instances.count);
        // read and checked: the rest touches no Python
        const nb::gil_scoped_release release;
        for (std::size_t i = 0; i < instances.count; i++) {
            double* box = &placement.boxes[6 * i];
            const std::size_t mesh_index = instances.mesh(i);
            const double* transform = instances.matrix(i);
            tinybvh::BLASInstanceEx instance(static_cast<std::uint64_t>(mesh_index));
            for (std::int32_t entry = 0; entry < 16; entry++) {
                instance.transform[entry] = transform[entry];
            }
            // every instance's transform is checked, its mesh empty or not.
            // tinybvh leaves a singular transform's adjugate as its inverse, so
            // check the determinant here. The inverse carries rays into the
            // mesh's frame, so it is held to the same bound as every input: a
            // nearly singular transform's huge inverse would overflow there
            instance.InvertTransform();
            bool invertible = trimkit::determinant_of(transform) != 0.0;
            for (std::int32_t entry = 0; entry < 16; entry++) {
                invertible = invertible && std::abs(instance.invTransform[entry]) <= trimkit::COORDINATE_LIMIT;
            }
            if (!invertible) {
                trimkit::value_error("the transform of instance " + std::to_string(i) +
                                     " is not invertible with an inverse within " + trimkit::COORDINATE_LIMIT_TEXT);
            }
            tinybvh::BVH_Double* blas = meshes_[mesh_index]->bvh();
            if (blas == nullptr) {
                // a mesh without faces is never hit: a point where it is placed
                for (std::size_t axis = 0; axis < 3; axis++) {
                    box[axis] = transform[(4 * axis) + 3];
                    box[3 + axis] = transform[(4 * axis) + 3];
                }
                continue;
            }
            instance.Update(blas);
            placement.all_identity = placement.all_identity && std::equal(transform, transform + 16, IDENTITY.begin());
            for (std::int32_t axis = 0; axis < 3; axis++) {
                placement.extent =
                    std::max({placement.extent, std::abs(instance.aabbMin[axis]), std::abs(instance.aabbMax[axis])});
            }
            for (std::int32_t axis = 0; axis < 3; axis++) {
                box[axis] = instance.aabbMin[axis];
                box[3 + axis] = instance.aabbMax[axis];
            }
            placement.instances.push_back(instance);
            placement.which.push_back(static_cast<std::int64_t>(i));
        }
        return placement;
    }

    // The crossings of the scene's surface along a ray, nearest first, up to
    // `most`: tinybvh's nearest hit, then its nearest hit again from just past
    // that one, until nothing is ahead. The triangles meeting at a shared edge
    // or vertex a ray passes through all lie at one distance, and that
    // crossing is one hit, the triangle the traversal found first: so a ray's
    // count of hits is its count of crossings, which tells inside from
    // outside. The moved origin is rounded, so the triangle just crossed can
    // come back at about zero distance: then the step past it doubles, from
    // about 1e-12 of the scene (what the nodes are padded by), and two
    // surfaces closer than that step count once. Each hit's distance is then
    // measured again from the ray's own origin, by tinybvh's triangle test on
    // the triangle found: exactly what a cap of one hit reports for it,
    // whatever else the ray crossed first.
    //
    // `settle` takes what one traversal found and decides: emit a crossing,
    // move past a repeat, or stop. `crossings` runs that for one ray on the
    // CPU; `device_crossings` for every ray at once, tracing on the device
    template <typename Emit>
    bool settle(const TopLevel& level, const Vec& origin, const Vec& direction, double scene, std::size_t most,
                const Found& found, Crossing* state, const Emit& emit) const {
        if (!found.hit) {
            return false;
        }
        const std::int64_t instance =
            level.direct != nullptr ? level.which[0] : level.which[static_cast<std::size_t>(found.instance)];
        const std::int64_t face = static_cast<std::int64_t>(found.face);
        if (instance == state->last_instance && face == state->last_face) {
            // found again from the rounding: farther past it, and past the
            // whole scene is nothing left to find
            state->step *= 2.0;
            if (state->step > scene) {
                return false;
            }
            state->along = state->last_t + state->step;
            return true;
        }
        // the distance from the ray's own origin: what the traversal measured
        // when it started there, else measured again by tinybvh's triangle
        // test from there, in the mesh's frame as the top level carries rays
        // there (a grazing hit the moved origin found and the original misses
        // keeps the moved origin's distance)
        double t = found.t;
        if (state->along != 0.0) {
            const MeshBvh* mesh = level.direct != nullptr ? level.direct->bvh() : nullptr;
            tinybvh::RayEx exact(origin, direction);
            if (level.direct == nullptr) {
                const tinybvh::BLASInstanceEx& placed = level.instances[static_cast<std::size_t>(found.instance)];
                mesh = meshes_[static_cast<std::size_t>(placed.blasIdx)]->bvh();
                exact = tinybvh::RayEx(tinybvh::tinybvh_transform_point(origin, placed.invTransform),
                                       tinybvh::tinybvh_transform_vector(direction, placed.invTransform));
            }
            const std::uint64_t* corners = mesh->vertIdx + (3 * found.face);
            mesh->IntersectTri(exact, found.face, mesh->verts, corners[0], corners[1], corners[2]);
            t = exact.hit.t < BVH_DBL_FAR ? exact.hit.t : state->along + found.t;
        }
        emit(instance, face, t);
        state->count++;
        state->last_instance = instance;
        state->last_face = face;
        state->last_t = t;
        state->step = RELATIVE_STEP * std::max(scene, std::abs(t));
        state->along = t + state->step;
        return state->count < most;
    }

    template <typename Emit>
    void crossings(const TopLevel& level, const double* o, const double* d, double length_of_d, std::size_t most,
                   const Emit& emit) const {
        const tinybvh::BVH_Double* bvh = level.bvh();
        const Vec origin(o[0], o[1], o[2]);
        const Vec direction(d[0], d[1], d[2]);
        // in the direction's units: the scene's size
        const double scene = level.extent / length_of_d;
        Crossing state;
        bool more = true;
        while (more) {
            tinybvh::RayEx ray(origin + (direction * state.along), direction);
            bvh->Intersect(ray);
            Found found;
            found.hit = ray.hit.t < BVH_DBL_FAR;
            found.instance = ray.hit.inst;
            found.face = ray.hit.prim;
            found.t = ray.hit.t;
            more = settle(level, origin, direction, scene, most, found, &state, emit);
        }
    }

    // `count` of a query's origins or directions from `begin`, written for
    // the device as a chunk's, the cores sharing the pass, and how it wrote
    // them (trimkit::vulkan::STEP_*): one row for every ray; float32 rows
    // while `*narrow` holds, which the first value that isn't exactly a
    // float32 ends for the rest of the query; else float64 rows. `*taken` is
    // cleared by a value `read` refuses: the one look these values get on
    // their way to a device
    static std::uint32_t fill_rows(const Rays& rays, void* chunk, std::size_t begin, std::size_t count, bool* narrow,
                                   bool* taken) {
        double* wide = static_cast<double*>(chunk);
        if (rays.one) {
            double row[3];
            rays.read(0, row);
            *taken = trimkit::copy_coordinates(row, 3, wide) == 0 && *taken;
            return trimkit::vulkan::STEP_ONE;
        }
        if (*narrow) {
            std::atomic<bool> kept{true};
            float* out = static_cast<float*>(chunk);
            in_blocks(count, [&](std::size_t from, std::size_t to) {
                const std::size_t first = 3 * (begin + from);
                const bool same =
                    rays.wide != nullptr
                        ? trimkit::narrow_coordinates(rays.wide + first, 3 * (to - from), out + (3 * from))
                        : trimkit::narrow_coordinates(rays.narrow + first, 3 * (to - from), out + (3 * from));
                if (!same) {
                    kept.store(false, std::memory_order_relaxed);
                }
            });
            if (kept.load()) {
                return trimkit::vulkan::STEP_NARROW;
            }
            *narrow = false;
        }
        std::atomic<std::size_t> refused{0};
        in_blocks(count, [&](std::size_t from, std::size_t to) {
            const std::size_t first = 3 * (begin + from);
            const std::size_t bad =
                rays.wide != nullptr
                    ? trimkit::copy_coordinates(rays.wide + first, 3 * (to - from), wide + (3 * from))
                    : trimkit::copy_coordinates(rays.narrow + first, 3 * (to - from), wide + (3 * from));
            refused.fetch_add(bad, std::memory_order_relaxed);
        });
        *taken = refused.load() == 0 && *taken;
        return trimkit::vulkan::STEP_WIDE;
    }

    // One query of `level` traced on the device over `count` rays, `consume`
    // reading each chunk's answers: false when the scene has no device (no
    // reason: that is no fallback to warn about), when there are no rays,
    // when this top level isn't on it or the device failed (then with the
    // reason), or when a ray's value is one `read` refuses (no reason: the
    // caller's `check_rows` raises it), and the CPU answers
    template <typename Consume>
    bool on_device(const TopLevel& level, const Rays& origins, const Rays& directions, std::size_t count,
                   bool occlusion, std::size_t most, const Consume& consume, std::string* reason) const {
        if (!on_device_ || count == 0) {
            return false;
        }
        if (level.direct == nullptr && !level.on_device) {
            *reason = level.device_reason;
            return false;
        }
        const nb::gil_scoped_release release;
        trimkit::vulkan::Query query;
        query.scene = &on_device_->arrays;
        query.top = level.direct != nullptr ? nullptr : &level.on_device->arrays;
        query.mesh = level.direct != nullptr ? static_cast<std::uint32_t>(level.instances[0].blasIdx) : 0;
        query.occlusion = occlusion;
        query.most = most < trimkit::vulkan::UNBOUNDED ? static_cast<std::uint32_t>(most) : trimkit::vulkan::UNBOUNDED;
        query.extent = level.extent;
        // each array goes as float32 until a value of it isn't one
        bool narrow[2] = {true, true};
        const trimkit::Exclusively holding(device_lock());
        return trimkit::vulkan::run_query(
            *on_device_->device, query, count,
            [&](void* chunk_origins, void* chunk_directions, std::size_t begin, std::size_t rays,
                std::uint32_t* steps) {
                bool taken = true;
                steps[0] = fill_rows(origins, chunk_origins, begin, rays, &narrow[0], &taken);
                steps[1] = fill_rows(directions, chunk_directions, begin, rays, &narrow[1], &taken);
                return taken;
            },
            consume, reason);
    }

    // Every ray's crossings traced on the device, into the columns: the device
    // counts each ray's crossings (its traversal, and the moves past each hit,
    // exactly as `crossings` does them on the CPU), scans the counts into
    // offsets, writes every crossing at its offset and hands them back
    // compacted, a chunk at a time; each is widened into place with its ray
    // made the caller's and its instance mapped through `which`
    bool device_crossings(const TopLevel& level, const Rays& origins, const Rays& directions, std::size_t count,
                          std::size_t most, Columns* columns, std::string* reason) const {
        // room for the usual, as the CPU path reserves: a hit per ray with a
        // cap of one, else two, a closed surface entered and left
        const std::size_t usual = std::min<std::size_t>(most, 2);
        std::size_t total = 0;
        const bool answered = on_device(
            level, origins, directions, count, false, most,
            [&](const trimkit::vulkan::Answers& answers) {
                // this chunk's crossings and the usual for every ray after it:
                // the first chunk sizes the columns for the whole query, and a
                // later one grows them only when its rays crossed more
                const std::size_t after = count - (answers.begin + answers.count);
                const std::size_t room = total + answers.total + (after * usual);
                if (room > columns->rays.size()) {
                    columns->resize(room);
                }
                std::int64_t* rays_out = trimkit::elements(columns->rays) + total;
                std::int64_t* instances_out = trimkit::elements(columns->instances) + total;
                std::int64_t* faces_out = trimkit::elements(columns->faces) + total;
                double* distances_out = trimkit::elements(columns->distances) + total;
                const std::int64_t base = static_cast<std::int64_t>(answers.begin);
                in_blocks(answers.total, [&](std::size_t from, std::size_t to) {
                    for (std::size_t h = from; h < to; h++) {
                        rays_out[h] = base + answers.rays[h];
                        instances_out[h] = level.which[answers.words[2 * h]];
                        faces_out[h] = static_cast<std::int64_t>(answers.words[(2 * h) + 1]);
                    }
                    std::memcpy(distances_out + from, answers.distances + from, (to - from) * sizeof(double));
                });
                total += answers.total;
            },
            reason);
        columns->resize(total);
        return answered;
    }

    // Every ray's occlusion traced on the device, into `hits`
    bool device_occlusion(const TopLevel& level, const Rays& origins, const Rays& directions, std::size_t count,
                          std::vector<std::uint8_t>* hits, std::string* reason) const {
        return on_device(
            level, origins, directions, count, true, 1,
            [&](const trimkit::vulkan::Answers& answers) {
                std::uint8_t* out = hits->data() + answers.begin;
                for (std::size_t k = 0; k < answers.count; k++) {
                    out[k] = answers.occlusion[k] != 0 ? 1 : 0;
                }
            },
            reason);
    }

    // a device fell back to the CPU: said once per scene, with the GIL held
    void warn_fallback(const std::string& reason) const {
        if (!reason.empty() && !warned_.exchange(true)) {
            trimkit::warn("use_gpu: " + reason + "; the CPU answered");
        }
    }

    // Open the process's device and put every mesh on it, as src/vulkan/trace.comp
    // reads them: false with the reason when it can't
    bool open_device(std::string* reason) {
        trimkit::vulkan::Device* device = trimkit::vulkan::shared_device(reason);
        if (device == nullptr) {
            return false;
        }
        const nb::gil_scoped_release release;
        static_assert(sizeof(tinybvh::BVH_Double::BVHNode) == 64);
        static_assert(sizeof(Vec) == 3 * sizeof(double));
        // every mesh's nodes, leaf slots, corners and vertices, each in one
        // array: `table` holds where each mesh's begin, `totals` their sizes
        std::vector<std::uint32_t> table(4 * meshes_.size(), 0);
        std::size_t totals[4] = {0, 0, 0, 0};
        for (std::size_t m = 0; m < meshes_.size(); m++) {
            const MeshBvh* bvh = meshes_[m]->bvh();
            if (bvh == nullptr) {
                continue;
            }
            const std::size_t sizes[4] = {bvh->usedNodes, bvh->idxCount, meshes_[m]->faces().size(),
                                          meshes_[m]->vertices().size()};
            for (std::size_t k = 0; k < 4; k++) {
                table[(4 * m) + k] = static_cast<std::uint32_t>(totals[k]);
                totals[k] += sizes[k];
            }
        }
        // the shader indexes with 32-bit words
        if (*std::max_element(totals, totals + 4) >= std::size_t{1} << 32U) {
            *reason = "the meshes have more than 2^32 vertices, faces or nodes together";
            return false;
        }
        std::vector<trimkit::vulkan::Node64> nodes(totals[0]);
        std::vector<std::uint64_t> corners(totals[2]);
        std::vector<Vec> vertices(totals[3]);
        std::vector<trimkit::vulkan::Node32> node32(totals[0]);
        // a triangle per leaf slot: a slot no leaf reaches stays zero, unread
        std::vector<trimkit::vulkan::Tri32> tri32(totals[1]);
        for (std::size_t m = 0; m < meshes_.size(); m++) {
            const MeshBvh* bvh = meshes_[m]->bvh();
            if (bvh == nullptr) {
                continue;
            }
            const std::uint32_t* first = table.data() + (4 * m);
            const std::vector<std::uint64_t>& faces = meshes_[m]->faces();
            const std::vector<Vec>& points = meshes_[m]->vertices();
            // the nodes as the bytes tinybvh reads them as
            std::memcpy(nodes.data() + first[0], bvh->bvhNode, bvh->usedNodes * sizeof(trimkit::vulkan::Node64));
            std::copy(faces.begin(), faces.end(), corners.data() + first[2]);
            std::copy(points.begin(), points.end(), vertices.data() + first[3]);
            trimkit::vulkan::Tree tree{};
            tree.nodes = nodes.data() + first[0];
            tree.node_count = bvh->usedNodes;
            tree.slots = bvh->primIdx;
            tree.slot_count = bvh->idxCount;
            tree.limit = faces.size() / 3;
            tree.depth = trimkit::vulkan::MESH_DEPTH;
            trimkit::vulkan::Tri32* triangles = tri32.data() + first[1];
            const bool sound = trimkit::vulkan::walk(
                tree, node32.data() + first[0],
                [&](std::size_t slot, std::uint64_t face) { triangles[slot] = triangle32(points, faces, face); },
                reason);
            if (!sound) {
                return false;
            }
        }
        const trimkit::vulkan::Span spans[trimkit::vulkan::SCENE_BUFFERS] = {
            bytes_of(nodes), bytes_of(corners), bytes_of(vertices), bytes_of(table), bytes_of(node32), bytes_of(tri32),
        };
        std::unique_ptr<OnDevice> on_device = std::make_unique<OnDevice>();
        on_device->device = device;
        const trimkit::Exclusively holding(device_lock());
        if (!trimkit::vulkan::upload(*device, spans, trimkit::vulkan::SCENE_BUFFERS, &on_device->arrays, reason)) {
            return false;
        }
        on_device_ = std::move(on_device);
        return true;
    }

    // Put a top level on the device beside its scene, or record why it isn't
    void upload_top(TopLevel* top) const {
        const MeshBvh& tlas = *top->tlas;
        const std::size_t count = top->instances.size();
        std::vector<double> inverses(16 * count);
        std::vector<std::uint32_t> instance_meshes(count);
        for (std::size_t i = 0; i < count; i++) {
            for (std::int32_t entry = 0; entry < 16; entry++) {
                inverses[(16 * i) + static_cast<std::size_t>(entry)] = top->instances[i].invTransform[entry];
            }
            instance_meshes[i] = static_cast<std::uint32_t>(top->instances[i].blasIdx);
        }
        const std::vector<trimkit::vulkan::Node64> nodes = nodes_of(tlas);
        std::vector<trimkit::vulkan::Node32> node32(nodes.size());
        trimkit::vulkan::Tree tree{};
        tree.nodes = nodes.data();
        tree.node_count = nodes.size();
        tree.slots = tlas.primIdx;
        tree.slot_count = tlas.idxCount;
        tree.limit = count;
        tree.depth = trimkit::vulkan::TOP_DEPTH;
        if (!trimkit::vulkan::walk(
                tree, node32.data(), [](std::size_t /*slot*/, std::uint64_t /*instance*/) {}, &top->device_reason)) {
            return;
        }
        const trimkit::vulkan::Span spans[trimkit::vulkan::TOP_BUFFERS] = {
            bytes_of(nodes),    {tlas.primIdx, tlas.idxCount * sizeof(std::uint64_t)},
            bytes_of(inverses), bytes_of(instance_meshes),
            bytes_of(node32),
        };
        std::unique_ptr<OnDevice> on_device = std::make_unique<OnDevice>();
        on_device->device = on_device_->device;
        const trimkit::Exclusively holding(device_lock());
        if (trimkit::vulkan::upload(*on_device->device, spans, trimkit::vulkan::TOP_BUFFERS, &on_device->arrays,
                                    &top->device_reason)) {
            top->on_device = std::move(on_device);
        }
    }

    // the step past a hit, in units of the scene's size or the distance reached
    static constexpr double RELATIVE_STEP = 9.094947017729282e-13;  // 2^-40

    std::vector<std::unique_ptr<RayMesh>> meshes_;
    // each mesh's BVH, or null without faces, as every top level indexes them
    std::vector<tinybvh::BVH_Double*> blas_;
    // the top level last built, swapped whole: a query keeps the one it read.
    // Every read and swap happens with the GIL held, which serializes them, so
    // the lock is nothing on GIL builds; on free-threaded ones it is CPython's
    // own PyMutex, held for one shared_ptr copy
    mutable nb::ft_mutex last_lock_;
    mutable std::shared_ptr<const TopLevel> last_;
    // the meshes on the device, for a scene built with `use_gpu` that could
    std::unique_ptr<OnDevice> on_device_;
    // whether this scene has said that the CPU answered for the device
    mutable std::atomic<bool> warned_{false};
};

// Axis-aligned boxes in 2D or 3D, built once into a BVH over their bounds
// (tinybvh's `BuildAABB`) for overlap queries: a static, bulk-loaded R-tree.
// tinybvh has no box query, so the traversal below is ours, reading its
// nodes; 2D boxes are stored with a zero z extent.
class Boxes {
public:
    explicit Boxes(const BoundsArg& boxes_argument) : Boxes(trimkit::read(boxes_argument, "boxes")) {}

    explicit Boxes(const Bounds& boxes) : dimension_(dimension_of(boxes)) {
        if (dimension_ != 2 && dimension_ != 3) {
            trimkit::value_error("boxes must be (n, 2, 2) or (n, 2, 3), not (n, 2, " + std::to_string(dimension_) +
                                 ")");
        }
        const nb::gil_scoped_release release;
        build(corners_of(boxes, "boxes"));
    }

    // 3D boxes as (minimum, maximum) corner pairs, already checked
    explicit Boxes(std::vector<Vec>&& corners) : dimension_(3) {
        const nb::gil_scoped_release release;
        build(std::move(corners));
    }

    std::size_t size() const { return boxes_.size() / 2; }

    // the box around every box: (2, d), or +inf then -inf with none
    trimkit::Out<double> bounds() const {
        std::vector<double> bounds(2 * dimension_);
        for (std::size_t axis = 0; axis < dimension_; axis++) {
            bounds[axis] = lowest_[axis];
            bounds[dimension_ + axis] = highest_[axis];
        }
        return trimkit::to_numpy(std::move(bounds), {2, dimension_});
    }

    // every box as given: (n, 2, d), exactly
    trimkit::Out<double> boxes() const {
        const std::size_t n = size();
        std::vector<double> boxes(2 * dimension_ * n);
        {
            const nb::gil_scoped_release release;
            for (std::size_t corner = 0; corner < 2 * n; corner++) {
                double* out = &boxes[dimension_ * corner];
                out[0] = boxes_[corner][0];
                out[1] = boxes_[corner][1];
                if (dimension_ == 3) {
                    out[2] = boxes_[corner][2];
                }
            }
        }
        return trimkit::to_numpy(std::move(boxes), {n, 2, dimension_});
    }

    trimkit::Out<std::int64_t> intersect(const BoundsArg& queries_argument) const {
        const Bounds queries = trimkit::read(queries_argument, "queries");
        if (dimension_of(queries) != dimension_) {
            trimkit::value_error("queries must have the boxes' dimension, " + std::to_string(dimension_));
        }
        const std::size_t n = queries.shape(0);
        const std::vector<Vec> boxes = corners_of(queries, "queries");
        // [query, box] rows, in query order
        trimkit::Gathered<std::int64_t> pairs;
        trimkit::gather_in_parallel<std::int64_t>(
            n, &pairs, [&](std::size_t begin, std::size_t end, std::vector<std::int64_t>& rows) {
                Stack stack = make_stack();
                std::vector<std::int64_t> boxes_hit;
                for (std::size_t q = begin; q < end; q++) {
                    boxes_hit.clear();
                    overlapping(boxes[2 * q], boxes[(2 * q) + 1], stack,
                                [&](std::uint64_t box) { boxes_hit.push_back(static_cast<std::int64_t>(box)); });
                    // ascending boxes, so the same query always reads the same
                    std::sort(boxes_hit.begin(), boxes_hit.end());
                    for (const std::int64_t box : boxes_hit) {
                        rows.push_back(static_cast<std::int64_t>(q));
                        rows.push_back(box);
                    }
                }
            });
        const std::size_t count = pairs.size() / 2;
        return trimkit::to_numpy<std::int64_t>(trimkit::gathered_rows(pairs), {count, 2});
    }

private:
    void build(std::vector<Vec>&& corners) {
        boxes_ = std::move(corners);
        for (std::size_t corner = 0; corner < boxes_.size(); corner += 2) {
            for (std::size_t axis = 0; axis < 3; axis++) {
                lowest_[axis] = std::min(lowest_[axis], boxes_[corner][static_cast<std::int32_t>(axis)]);
                highest_[axis] = std::max(highest_[axis], boxes_[corner + 1][static_cast<std::int32_t>(axis)]);
            }
        }
        // an empty build is an error to tinybvh: no boxes, no BVH
        if (!boxes_.empty()) {
            bvh_ = new_bvh();
            bvh_->BuildAABB(boxes_.data(), boxes_.size() / 2);
        }
    }

    static std::size_t dimension_of(const trimkit::In<double, -1, 2, -1>& bounds) {
        return bounds.ndim() == 3 ? bounds.shape(2) : 0;
    }

    // (min, max) corner pairs as 3D points, each pair checked to be ordered
    std::vector<Vec> corners_of(const trimkit::In<double, -1, 2, -1>& bounds, const char* name) const {
        const std::size_t n = bounds.shape(0);
        const std::size_t d = dimension_;
        const double* b = bounds.data();
        std::vector<Vec> corners(2 * n);
        for (std::size_t i = 0; i < n; i++) {
            const double* lo = b + (2 * d * i);
            const double* hi = lo + d;
            for (std::size_t a = 0; a < d; a++) {
                if (lo[a] > hi[a]) {
                    trimkit::value_error(std::string(name) + " has a minimum above its maximum at row " +
                                         std::to_string(i));
                }
            }
            corners[2 * i] = Vec(lo[0], lo[1], d == 3 ? lo[2] : 0.0);
            corners[(2 * i) + 1] = Vec(hi[0], hi[1], d == 3 ? hi[2] : 0.0);
        }
        return corners;
    }

    // every box overlapping [lo, hi], touching included like rtree
    template <typename Emit>
    void overlapping(const Vec& lo, const Vec& hi, Stack& stack, const Emit& emit) const {
        if (!bvh_) {
            return;
        }
        stack.push_back(0);
        while (!stack.empty()) {
            const tinybvh::BVH_Double::BVHNode& node = bvh_->bvhNode[pop(stack)];
            if (!overlaps(lo, hi, node.aabbMin, node.aabbMax)) {
                continue;
            }
            if (!node.isLeaf()) {
                stack.push_back(node.leftFirst);
                stack.push_back(node.leftFirst + 1);
                continue;
            }
            for (std::uint64_t j = node.leftFirst; j < node.leftFirst + node.triCount; j++) {
                const std::size_t box = static_cast<std::size_t>(bvh_->primIdx[j]);
                if (overlaps(lo, hi, boxes_[2 * box], boxes_[(2 * box) + 1])) {
                    emit(box);
                }
            }
        }
    }

    std::size_t dimension_ = 0;
    std::vector<Vec> boxes_;
    // the box around every box, minimum then maximum: empty, inverted
    static constexpr double FAR = std::numeric_limits<double>::infinity();
    double lowest_[3] = {FAR, FAR, FAR};
    double highest_[3] = {-FAR, -FAR, -FAR};
    // declared last, so it is destroyed before the boxes it was built from
    std::unique_ptr<MeshBvh> bvh_;
};

// tinybvh reports a fatal error (a broken invariant: it validates nothing it
// is given) by calling this, which throws, rather than let it exit the
// process: on any thread, as the pool carries a worker's exception back
[[noreturn]] void throw_tinybvh_error(const char* message) { throw std::runtime_error(message); }

}  // namespace

trimkit::BoxesObject trimkit::scene_boxes(const std::vector<double>& boxes) {
    const std::size_t count = boxes.size() / 6;
    std::vector<Vec> corners(2 * count);
    for (std::size_t i = 0; i < count; i++) {
        const double* box = &boxes[6 * i];
        corners[2 * i] = Vec(box[0], box[1], box[2]);
        corners[(2 * i) + 1] = Vec(box[3], box[4], box[5]);
    }
    return {nb::cast(Boxes(std::move(corners)))};
}

void bind_bvh(nb::module_& m) {
    tinybvh::tinybvh_set_fatal_error_handler(throw_tinybvh_error);

    nb::class_<RayScene>(m, "RayScene",
                         R"(Meshes, each built once into a double precision BVH, and the only
thing rays are cast against.

Every query names its instances (which mesh, placed where), as a
CollisionScene's do, so a bare query casts at each mesh once, where it
is. `intersect` answers where each ray crosses the surface, nearest first
and as many as asked; `intersect_any` only whether it does. Instances are placed under a top-level BVH built from exactly the
arrays a query gives; the scene remembers the last one, so a query naming
the same instances again (the same bytes) costs only comparing them, and
moving anything costs that top level, never a mesh. Answers never depend
on what was asked before.

Parameters
----------
meshes : sequence of Mesh
  The meshes to place, each a (vertices, faces) pair of (n, 3) float64
  vertices and (m, 3) int64 triangle indices into them. Copied once
  into tinybvh's padded float layout, which the scene owns, so a mesh
  can never change under the threads sharing the scene.
use_gpu : bool, default False
  Trace on a Vulkan device (the first GPU that computes in double
  precision) instead of the CPU: the same trees, the same traversal, so
  the same answers to the byte, sooner for large batches of rays. When
  no such device can be used, now or later, the scene warns once (a
  UserWarning naming why) and the CPU answers.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> scene = trimkit.RayScene([(vertices, faces)])
>>> len(scene)
1)")
        .def(nb::init<const trimkit::Meshes&, trimkit::Flag>(), nb::arg("meshes"), nb::kw_only(),
             nb::arg("use_gpu") = false)
        .def("__len__", &RayScene::size, R"(How many meshes the scene was built from.

Returns
-------
count : int
  The number of meshes the scene was built from.)")
        .def("boxes", &RayScene::boxes, nb::arg("instance_matrices").none() = nb::none(),
             nb::arg("instance_index").none() = nb::none(),
             R"(The box each instance is culled by, as a query with these instances
places it: what the scene thinks it holds.

Parameters
----------
instance_matrices : (k, 4, 4) float64, optional
  Each instance's affine transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.

Returns
-------
boxes : Boxes
  One 3D box per instance, in its order, holding every face it places
  (an instance of a mesh without faces: a point at its translation).

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> boxes = trimkit.RayScene([(vertices, faces)]).boxes()
>>> boxes.bounds.tolist()
[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])")
        .def("intersect", &RayScene::intersect, nb::arg("origins"), nb::arg("directions"),
             nb::arg("instance_matrices").none() = nb::none(), nb::arg("instance_index").none() = nb::none(),
             nb::kw_only(), nb::arg("maximum_hits").none() = nb::none(),
             R"(Where each ray crosses the scene's surface, nearest first, from either side.

Parameters
----------
origins : (r, 3) or (3,) float64 or float32
  Where each ray starts, or where every ray starts.
directions : (r, 3) or (3,) float64 or float32
  Each ray's direction, or every ray's, which need not be unit length.
instance_matrices : (k, 4, 4) float64, optional
  Each instance's affine transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.
maximum_hits : int, optional
  The most hits to report per ray, nearest first, which bounds the work
  at that many traversals per ray: 1 is the nearest hit of each ray.
  Defaults to every hit.

Returns
-------
rays : (h,) int64
  Ray of each hit, one row per hit, sorted by ray and then distance.
  A ray through a shared edge or vertex crosses the surface once there
  and gets one hit, so a closed mesh is crossed an odd number of times
  from inside and an even number from outside, whatever the ray's
  direction.
instances : (h,) int64
  Instance of each hit.
faces : (h,) int64
  Face of each hit in its instance's mesh.
distances : (h,) float64
  Distance from the ray's origin to each hit.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> scene = trimkit.RayScene([(vertices, faces)])
>>> origins, directions = np.array([[0.2, 0.2, 1.0]]), np.array([[0.0, 0.0, -1.0]])
>>> rays, instances, faces, distances = scene.intersect(origins, directions)
>>> rays.tolist(), instances.tolist(), faces.tolist(), distances.tolist()
([0], [0], [0], [1.0]))")
        .def("intersect_any", &RayScene::intersect_any, nb::arg("origins"), nb::arg("directions"),
             nb::arg("instance_matrices").none() = nb::none(), nb::arg("instance_index").none() = nb::none(),
             R"(Whether each ray hits anything, stopping at the first triangle found.

Parameters
----------
origins : (r, 3) or (3,) float64 or float32
  Where each ray starts, or where every ray starts.
directions : (r, 3) or (3,) float64 or float32
  Each ray's direction, or every ray's, which need not be unit length.
instance_matrices : (k, 4, 4) float64, optional
  Each instance's affine transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.

Returns
-------
hits : (r,) bool
  True for every ray that hits a triangle.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> scene = trimkit.RayScene([(vertices, faces)])
>>> origins, directions = np.array([[0.2, 0.2, 1.0]]), np.array([[0.0, 0.0, -1.0]])
>>> hits = scene.intersect_any(origins, directions)
>>> hits.tolist()
[True])");

    nb::class_<Boxes>(m, "Boxes",
                      R"(Axis-aligned boxes built into a BVH for overlap queries: a static,
numpy-native replacement for a bulk-loaded R-tree.

Parameters
----------
boxes : (n, 2, d) float64
  Each box's minimum and maximum corner, in 2 or 3 dimensions. Copied
  into the tree the object owns, so the boxes can't change under it.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> len(tree)
2)")
        .def(nb::init<const BoundsArg&>(), nb::arg("boxes"))
        .def("__len__", &Boxes::size, R"(How many boxes it holds.

Returns
-------
count : int
  The number of boxes it holds.)")
        .def_prop_ro("bounds", &Boxes::bounds, nb::rv_policy::move,
                     R"(The box around every box.

Returns
-------
bounds : (2, d) float64
  The minimum and maximum corner over every box, or +inf then -inf
  with no boxes.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> bounds = tree.bounds
>>> bounds.tolist()
[[0.0, 0.0], [3.0, 3.0]])")
        .def("boxes", &Boxes::boxes,
             R"(Every box, exactly as given.

Returns
-------
boxes : (n, 2, d) float64
  Each box's minimum and maximum corner, a fresh copy.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> boxes = tree.boxes()
>>> boxes.shape
(2, 2, 2))")
        .def("intersect", &Boxes::intersect, nb::arg("queries"),
             R"(Every box overlapping each query box, touching included.

Parameters
----------
queries : (m, 2, d) float64
  Each query's minimum and maximum corner, in the boxes' dimension.

Returns
-------
pairs : (k, 2) int64
  Rows of [query, box] for every overlap, sorted by query
  and then box.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> pairs = tree.intersect(np.array([[[0.5, 0.5], [2.5, 2.5]]]))
>>> pairs.tolist()
[[0, 0], [0, 1]])");
}
