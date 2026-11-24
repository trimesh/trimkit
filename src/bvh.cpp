// Ray queries against an instanced scene of triangle meshes, and box overlap
// queries against a set of boxes, with jbikker/tinybvh. Python-free
// (trimkit.hpp). Every ray query is tinybvh's own nearest-hit or any-hit
// traversal: all the hits along a ray are its nearest hit asked again from
// just past each one, so no traversal or triangle test of tinybvh's is
// reimplemented here. Box overlap, which tinybvh has no query for, walks its
// nodes.
//
// A `RayMeshes` builds each of its meshes once, in double precision (with
// spatial splits when it is big enough to gain from them), from copies of
// its inputs: tinybvh keeps pointers into its geometry, so the scene owns
// it. Every query names its instances (which mesh, placed where), placed
// under one top-level BVH without copying their triangles: one identity
// instance skips the top level entirely. A top level is built from exactly
// the instances given and never changed, so a caller may keep one for every
// query naming the same instances (src/trimkit.cpp remembers the last), and
// its answers are the same either way, only sooner. Queries are const, so
// threads may share one scene, and every batch runs across every core.
//
// A scene built with `use_gpu` holds its meshes on a Vulkan device too
// (src/vulkan/scene.hpp, the only Vulkan this file sees) and traces there: the same trees, and a shader
// (src/vulkan/trace.comp) that makes every decision tinybvh's traversal makes,
// most in f32 behind a proven bound and the rest with tinybvh's own f64
// operations, so the same bytes come back; the crossing loop below
// (`crossings`) runs in the shader too, written the same way. A device that
// can't be used, now or later, is a reason returned, and the CPU answers.

#include "trimkit.hpp"
#include "vulkan/scene.hpp"

#include <tiny_bvh.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Vec = tinybvh::bvhdbl3;

// A BVH as tinybvh builds it, with its test of a ray against one known
// triangle callable: tinybvh keeps that protected, and every hit of
// `intersect` is measured with it
struct MeshBvh : tinybvh::BVH_Double {
    using tinybvh::BVH_Double::IntersectTri;
};

// tinybvh's nodes come through the extension's one door for allocation
// (src/allocator.cpp), rather than the aligned malloc it would otherwise call
void* allocate_aligned(std::size_t size, void* /*unused*/) { return ::operator new(size, std::align_val_t{64}); }

void free_aligned(void* block, void* /*unused*/) { ::operator delete(block, std::align_val_t{64}); }

// tinybvh reports a fatal error (a broken invariant: it validates nothing it
// is given) by calling this, which throws, rather than let it exit the
// process: on any thread, as the pool carries a worker's exception back
[[noreturn]] void throw_tinybvh_error(const char* message) { throw std::runtime_error(message); }

// installed once, before the first BVH is built (every traversal is of one):
// out of `built_bvh`, whose every instantiation would have a static of its own
void install_tinybvh_error_handler() {
    static const bool INSTALLED = [] {
        tinybvh::tinybvh_set_fatal_error_handler(throw_tinybvh_error);
        return true;
    }();
    static_cast<void>(INSTALLED);
}

// The one way a BVH is made: tinybvh's allocation through the extension's
// door, `build(bvh)` run on it, then the build's scratch tinybvh keeps after
// it given back. That is `fragment`, a box per primitive (64 bytes each, a
// third of what a box query's tree holds): nothing here reads it again, as
// no BVH is rebuilt or refit, and tinybvh allocates it afresh for a rebuild,
// as it does after Load(), which keeps none
template <typename Build>
std::unique_ptr<MeshBvh> built_bvh(const Build& build) {
    install_tinybvh_error_handler();
    std::unique_ptr<MeshBvh> bvh = std::make_unique<MeshBvh>();
    bvh->context.malloc = allocate_aligned;
    bvh->context.free = free_aligned;
    build(*bvh);
    bvh->context.free(bvh->fragment, bvh->context.userdata);
    bvh->fragment = nullptr;
    return bvh;
}

// A watertight triangle test needs conservative traversal: the slab test
// rounds on the order of an ulp of the coordinates, so without padding a ray
// through a vertex or edge can skip the nodes holding it (Ize 2013). Padding
// by BOX_SLACK of the largest coordinate covers ray origins up to thousands
// of times farther out than the geometry
void pad(tinybvh::BVH_Double& bvh, double extent) {
    const Vec margin(trimkit::BOX_SLACK * extent);
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

// the squared distance between the boxes [a_lo, a_hi] and [b_lo, b_hi], zero
// where they overlap: each axis's gap squared, summed in axis order, so its
// sqrt is numpy's to the bit (the build never contracts it into an fma)
double gap_squared(const Vec& a_lo, const Vec& a_hi, const Vec& b_lo, const Vec& b_hi) {
    double total = 0.0;
    for (std::int32_t axis = 0; axis < 3; axis++) {
        const double gap = std::max({b_lo[axis] - a_hi[axis], a_lo[axis] - b_hi[axis], 0.0});
        total += gap * gap;
    }
    return total;
}

// One box as `nearest` keeps it: ordered by distance and then box, so the
// lower index wins a tie
struct NearestBox {
    double distance;
    std::uint64_t box;
    bool operator<(const NearestBox& other) const {
        return distance < other.distance || (distance == other.distance && box < other.box);
    }
};

// A node `nearest` has yet to visit, with its squared gap from the query
struct NearestNode {
    std::uint64_t node;
    double gap;
};

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

// A direction scaled by an exact power of two so its largest component is in
// [1, 2). tinybvh measures t in the direction's units, against 1e300 for no
// hit and with reciprocals capped at 2^1000, so a direction as short as
// 1e-280 missed what (0, 0, -1) hits; and a power of two only moves the
// exponent, so every other ray's distances (t times the length) are the same
// bits as unscaled. The factor is built from the exponent's bits: the same IEEE
// operations as src/vulkan/trace.comp's `normalize_exponent`
void normalize_exponent(double* d) {
    double largest = std::max({std::abs(d[0]), std::abs(d[1]), std::abs(d[2])});
    if (largest == 0.0) {
        return;
    }
    // times 2^k, k within [-1022, 1023]: exact unless a component goes subnormal
    const auto times_power_of_two = [d, &largest](int k) {
        const std::uint64_t bits = static_cast<std::uint64_t>(k + 1023) << 52U;
        double factor = 0.0;
        std::memcpy(&factor, &bits, sizeof(factor));
        for (std::size_t axis = 0; axis < 3; axis++) {
            d[axis] *= factor;
        }
        largest *= factor;
    };
    if (largest < std::numeric_limits<double>::min()) {
        times_power_of_two(600);  // a subnormal's exponent bits are 0: normal first
    }
    std::uint64_t bits = 0;
    std::memcpy(&bits, &largest, sizeof(bits));
    const int exponent = static_cast<int>((bits >> 52U) & 0x7FFU) - 1023;
    if (exponent != 0) {
        times_power_of_two(-exponent);
    }
}

// The values of rays refused as every input's are, naming them: a block at
// a time on the pool, float32 ones held to the same bound as float64 ones.
// Only the CPU needs it: the device looks at each value as it copies it
void refuse_rays_beyond(const trimkit::RayVectors& rays, const char* name) {
    const std::size_t values = 3 * rays.count;
    const std::size_t beyond =
        rays.is_float32
            ? trimkit::count_beyond_in_parallel(static_cast<const float*>(rays.data), values, trimkit::COORDINATE_LIMIT)
            : trimkit::count_beyond_in_parallel(static_cast<const double*>(rays.data), values,
                                                trimkit::COORDINATE_LIMIT);
    if (beyond != 0) {
        trimkit::refuse(std::string(name) + " must be finite and within " + trimkit::COORDINATE_LIMIT_TEXT);
    }
}

// One hit as `intersect` reports it
struct Hit {
    std::int64_t ray;
    std::int64_t instance;
    std::int64_t face;
    double t;
};

// One mesh of a RayMeshes, built once however often the scene places it
class RayMesh {
public:
    explicit RayMesh(trimkit::RayMeshInput& mesh) : faces_(std::move(mesh.faces)) {
        const double* v = mesh.vertices;
        vertices_.resize(mesh.vertex_count);
        for (std::size_t i = 0; i < mesh.vertex_count; i++) {
            vertices_[i] = Vec(v[3 * i], v[(3 * i) + 1], v[(3 * i) + 2]);
        }
        // the largest coordinate's magnitude
        const trimkit::PointBounds bounds = trimkit::bounds_of(v, mesh.vertex_count);
        double extent = 0.0;
        for (std::size_t axis = 0; axis < 3 && !vertices_.empty(); axis++) {
            extent = std::max({extent, std::abs(bounds.lowest[axis]), std::abs(bounds.highest[axis])});
        }
        if (!faces_.empty()) {
            // spatial splits trace 20-27% faster from about 300k triangles, for
            // 40% more build time, but gain nothing below 50k at 4x the build
            const std::size_t count = faces_.size() / 3;
            bvh_ = built_bvh([&](MeshBvh& bvh) {
                if (count >= 100000) {
                    bvh.BuildHQ(vertices_.data(), faces_.data(), count);
                } else {
                    bvh.Build(vertices_.data(), faces_.data(), count);
                }
            });
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

// the step past a hit: 2^-40 of the scene's size or the distance reached,
// the distance only up to NEAR_HITS scenes away, and at least 2^-50 of the
// farthest of the ray's origin and the hit, a few ulps of either: about the
// rounding the moved origin carries
constexpr double RELATIVE_STEP = 9.094947017729282e-13;
constexpr double NEAR_HITS = 1024.0;
constexpr double ROUNDING_STEP = 8.881784197001252e-16;

}  // namespace

struct trimkit::RayMeshTrees {
    std::vector<std::unique_ptr<RayMesh>> meshes;
    // each mesh's BVH, or null without faces, as every top level indexes them
    std::vector<tinybvh::BVH_Double*> blas;
    // the meshes on the device, for a scene built with `use_gpu` that could
    std::unique_ptr<vulkan::DeviceArrays> device_meshes;
    // why a scene built with `use_gpu` couldn't be
    std::string device_failure;
};

// A query's instances built into a top level, never changed once built: the
// instances as they were read, which a later query's arrays are compared
// with; each kept instance (a mesh with faces) with its checked transform,
// its inverse and its caller's row in `which`; every row's world box, 6
// values, minimum then maximum (its mesh's root box carried by the transform,
// as tinybvh's top level bounds it: the root as tinybvh keeps it, unpadded,
// as `pad` widens only the nodes and the top level's own padding covers the
// rounding; or for a mesh without faces, a point at its translation); and
// the BVH over the kept instances, or the one identity instance's mesh,
// traced directly
struct trimkit::RayTopLevel {
    Instances given;
    std::vector<tinybvh::BLASInstanceEx> instances;
    std::vector<std::int64_t> which;
    std::vector<double> boxes;
    // the largest coordinate magnitude any placed box reaches: the scene's size
    double extent = 0.0;
    const RayMesh* direct = nullptr;
    // the top level on the device when the scene traces there, or why not
    std::unique_ptr<vulkan::DeviceArrays> device_top;
    std::string device_reason;
    // declared last, so it is destroyed before the instances it points into
    std::unique_ptr<MeshBvh> tlas;

    // what a first-hit or any-hit ray traces, or null with nothing to hit
    const tinybvh::BVH_Double* bvh() const { return direct != nullptr ? direct->bvh() : tlas.get(); }
};

// Axis-aligned boxes in 2D or 3D, built once into a BVH over their bounds
// (tinybvh's `BuildAABB`) for overlap queries: a static, bulk-loaded R-tree.
// tinybvh has no box query, so the traversal below is ours, reading its
// nodes; 2D boxes are stored with a zero z extent.
struct trimkit::BoxBvh {
    // boxes as given, or a query's, as 3D (minimum, maximum) corner pairs,
    // each pair checked to be ordered
    static std::vector<Vec> corners_of(const double* boxes, std::size_t count, std::size_t d, const char* name) {
        std::vector<Vec> corners(2 * count);
        for (std::size_t i = 0; i < count; i++) {
            const double* lo = boxes + (2 * d * i);
            const double* hi = lo + d;
            for (std::size_t a = 0; a < d; a++) {
                if (lo[a] > hi[a]) {
                    refuse(std::string(name) + " has a minimum above its maximum at row " + std::to_string(i));
                }
            }
            corners[2 * i] = Vec(lo[0], lo[1], d == 3 ? lo[2] : 0.0);
            corners[(2 * i) + 1] = Vec(hi[0], hi[1], d == 3 ? hi[2] : 0.0);
        }
        return corners;
    }

    BoxBvh(std::vector<Vec>&& corners, std::size_t d) : dimension(d), boxes(std::move(corners)) {
        for (std::size_t corner = 0; corner < boxes.size(); corner += 2) {
            for (std::size_t axis = 0; axis < 3; axis++) {
                lowest[axis] = std::min(lowest[axis], boxes[corner][static_cast<std::int32_t>(axis)]);
                highest[axis] = std::max(highest[axis], boxes[corner + 1][static_cast<std::int32_t>(axis)]);
            }
        }
        // an empty build is an error to tinybvh: no boxes, no BVH
        if (!boxes.empty()) {
            bvh = built_bvh([&](MeshBvh& tree) { tree.BuildAABB(boxes.data(), boxes.size() / 2); });
        }
    }

    // every box overlapping [lo, hi], touching included like rtree
    template <typename Emit>
    void overlapping(const Vec& lo, const Vec& hi, Stack& stack, const Emit& emit) const {
        if (!bvh) {
            return;
        }
        stack.push_back(0);
        while (!stack.empty()) {
            const tinybvh::BVH_Double::BVHNode& node = bvh->bvhNode[pop(stack)];
            if (!overlaps(lo, hi, node.aabbMin, node.aabbMax)) {
                continue;
            }
            if (!node.isLeaf()) {
                stack.push_back(node.leftFirst);
                stack.push_back(node.leftFirst + 1);
                continue;
            }
            for (std::uint64_t j = node.leftFirst; j < node.leftFirst + node.triCount; j++) {
                const std::size_t box = static_cast<std::size_t>(bvh->primIdx[j]);
                if (overlaps(lo, hi, boxes[2 * box], boxes[(2 * box) + 1])) {
                    emit(box);
                }
            }
        }
    }

    // The `most` nearest boxes to [lo, hi] into `best`, sorted, with `most` at
    // least 1 and at most the boxes there are. `best` is a max-heap of the
    // nearest so far, and once it is full a node or box is skipped only when
    // its distance is past the worst kept: one at that distance is looked at,
    // as its index may be lower. The walk steps into the nearer child and
    // stacks the farther, which is 10-25% faster than stacking both
    void nearest_of(const Vec& lo, const Vec& hi, std::size_t most, std::vector<NearestNode>& stack,
                    std::vector<NearestBox>& best) const {
        const tinybvh::BVH_Double::BVHNode* nodes = bvh->bvhNode;
        best.clear();
        double worst = INF;
        // true only when sqrt(gap) > worst: the first test only saves the sqrt
        const auto too_far = [&](double gap) { return gap > worst * worst && std::sqrt(gap) > worst; };
        stack.push_back({0, 0.0});
        while (!stack.empty()) {
            NearestNode next = stack.back();
            stack.pop_back();
            while (!too_far(next.gap)) {
                const tinybvh::BVH_Double::BVHNode& node = nodes[next.node];
                if (node.isLeaf()) {
                    for (std::uint64_t j = node.leftFirst; j < node.leftFirst + node.triCount; j++) {
                        const auto box = static_cast<std::size_t>(bvh->primIdx[j]);
                        const double gap = gap_squared(lo, hi, boxes[2 * box], boxes[(2 * box) + 1]);
                        if (too_far(gap)) {
                            continue;
                        }
                        const NearestBox found{std::sqrt(gap), box};
                        if (best.size() == most) {
                            if (!(found < best.front())) {
                                continue;
                            }
                            std::pop_heap(best.begin(), best.end());
                            best.pop_back();
                        }
                        best.push_back(found);
                        std::push_heap(best.begin(), best.end());
                        if (best.size() == most) {
                            worst = best.front().distance;
                        }
                    }
                    break;
                }
                const std::uint64_t left = node.leftFirst;
                NearestNode nearer{left, gap_squared(lo, hi, nodes[left].aabbMin, nodes[left].aabbMax)};
                NearestNode farther{left + 1, gap_squared(lo, hi, nodes[left + 1].aabbMin, nodes[left + 1].aabbMax)};
                if (farther.gap < nearer.gap) {
                    std::swap(nearer, farther);
                }
                stack.push_back(farther);
                next = nearer;
            }
        }
        std::sort_heap(best.begin(), best.end());
    }

    static constexpr double INF = std::numeric_limits<double>::infinity();
    std::size_t dimension = 0;
    std::vector<Vec> boxes;
    // the box around every box, minimum then maximum: empty, inverted
    double lowest[3] = {INF, INF, INF};
    double highest[3] = {-INF, -INF, -INF};
    // declared last, so it is destroyed before the boxes it was built from
    std::unique_ptr<MeshBvh> bvh;
};

namespace {

using trimkit::RayMeshTrees;
using trimkit::RayTopLevel;

// Place `top->given`: each instance of a mesh with faces kept, in its
// order, with its world box; whether every kept transform is the identity
bool place_instances(const RayMeshTrees& trees, RayTopLevel* top) {
    const trimkit::Instances& instances = top->given;
    bool all_identity = true;
    // as many as there could be, written by index, then cut to those kept
    std::size_t kept = 0;
    top->instances.resize(instances.count, tinybvh::BLASInstanceEx(0));
    top->which.resize(instances.count);
    top->boxes.resize(6 * instances.count);
    for (std::size_t i = 0; i < instances.count; i++) {
        double* box = &top->boxes[6 * i];
        const std::size_t mesh_index = instances.mesh(i);
        const double* transform = instances.matrix(i);
        tinybvh::BLASInstanceEx instance(static_cast<std::uint64_t>(mesh_index));
        for (std::int32_t entry = 0; entry < 16; entry++) {
            instance.transform[entry] = transform[entry];
        }
        // The one check made here, as only here is the inverse: tinybvh
        // leaves a singular transform's adjugate as its inverse, so the
        // determinant is checked, and the inverse carries rays into the
        // mesh's frame, so it is held to the same bound as every input (a
        // nearly singular transform's huge inverse would overflow there).
        // Every instance's, its mesh empty or not
        instance.InvertTransform();
        bool invertible = trimkit::determinant_of(transform) != 0.0;
        for (std::int32_t entry = 0; entry < 16; entry++) {
            invertible = invertible && std::abs(instance.invTransform[entry]) <= trimkit::COORDINATE_LIMIT;
        }
        if (!invertible) {
            trimkit::refuse("the transform of instance " + std::to_string(i) +
                            " is not invertible with an inverse within " + trimkit::COORDINATE_LIMIT_TEXT);
        }
        tinybvh::BVH_Double* blas = trees.meshes[mesh_index]->bvh();
        if (blas == nullptr) {
            // a mesh without faces is never hit: a point where it is placed
            for (std::size_t axis = 0; axis < 3; axis++) {
                box[axis] = transform[(4 * axis) + 3];
                box[3 + axis] = transform[(4 * axis) + 3];
            }
            continue;
        }
        instance.Update(blas);
        all_identity = all_identity && std::equal(transform, transform + 16, trimkit::IDENTITY_TRANSFORM);
        for (std::int32_t axis = 0; axis < 3; axis++) {
            top->extent = std::max({top->extent, std::abs(instance.aabbMin[axis]), std::abs(instance.aabbMax[axis])});
            box[axis] = instance.aabbMin[axis];
            box[3 + axis] = instance.aabbMax[axis];
        }
        trimkit::refuse_placed_beyond(box, box + 3, i);
        top->instances[kept] = instance;
        top->which[kept] = static_cast<std::int64_t>(i);
        kept++;
    }
    top->instances.resize(kept, tinybvh::BLASInstanceEx(0));
    top->which.resize(kept);
    return all_identity;
}

// The crossings of the scene's surface along a ray, nearest first, up to
// `most`: tinybvh's nearest hit, then its nearest hit again from just past
// that one, until nothing is ahead. The triangles meeting at a shared edge
// or vertex a ray passes through all lie at one distance, and that
// crossing is one hit, the triangle the traversal found first: so a ray's
// count of hits is its count of crossings, which tells inside from
// outside. The moved origin is rounded, so the triangle just crossed can
// come back at about zero distance: then the step past it doubles, from
// about 1e-12 of the scene (what the nodes are padded by), or a few ulps of
// a far hit, and two surfaces closer than that step count once. Each hit's distance is then
// measured again from the ray's own origin, by tinybvh's triangle test on
// the triangle found: exactly what a cap of one hit reports for it,
// whatever else the ray crossed first.
//
// `crossings` does that for one ray on the CPU, calling `emit(instance,
// face, t)` for each; src/vulkan/trace.comp does the same for every ray
// of a chunk on the device, operation for operation
template <typename Emit>
void crossings(const RayMeshTrees& trees, const RayTopLevel& level, const double* o, const double* d,
               double length_of_d, std::size_t most, const Emit& emit) {
    const tinybvh::BVH_Double* bvh = level.bvh();
    const Vec origin(o[0], o[1], o[2]);
    const Vec direction(d[0], d[1], d[2]);
    // in the direction's units: the scene's size
    const double scene = level.extent / length_of_d;
    double along = 0.0;  // how far the origin has moved along the ray
    double step = 0.0;   // the move past the last hit
    double last_t = 0.0;
    std::int64_t last_instance = -1;
    std::int64_t last_face = -1;
    std::size_t count = 0;
    while (count < most) {
        tinybvh::RayEx ray(origin + (direction * along), direction);
        bvh->Intersect(ray);
        if (!(ray.hit.t < BVH_DBL_FAR)) {
            break;
        }
        const std::size_t placed_at = static_cast<std::size_t>(ray.hit.inst);
        const std::int64_t instance = level.direct != nullptr ? level.which[0] : level.which[placed_at];
        const std::int64_t face = static_cast<std::int64_t>(ray.hit.prim);
        if (instance == last_instance && face == last_face) {
            // found again from the rounding: farther past it, and past the
            // whole scene is nothing left to find
            step *= 2.0;
            if (step > scene) {
                break;
            }
            along = last_t + step;
            continue;
        }
        // the distance from the ray's own origin: what the traversal measured
        // when it started there, else measured again by tinybvh's triangle
        // test from there, in the mesh's frame as the top level carries rays
        // there (a grazing hit the moved origin found and the original misses
        // keeps the moved origin's distance)
        double t = ray.hit.t;
        if (along != 0.0) {
            const MeshBvh* mesh = level.direct != nullptr ? level.direct->bvh() : nullptr;
            tinybvh::RayEx exact(origin, direction);
            if (level.direct == nullptr) {
                const tinybvh::BLASInstanceEx& placed = level.instances[placed_at];
                mesh = trees.meshes[static_cast<std::size_t>(placed.blasIdx)]->bvh();
                exact = tinybvh::RayEx(tinybvh::tinybvh_transform_point(origin, placed.invTransform),
                                       tinybvh::tinybvh_transform_vector(direction, placed.invTransform));
            }
            const std::uint64_t* corners = mesh->vertIdx + (3 * ray.hit.prim);
            mesh->IntersectTri(exact, ray.hit.prim, mesh->verts, corners[0], corners[1], corners[2]);
            t = exact.hit.t < BVH_DBL_FAR ? exact.hit.t : along + ray.hit.t;
        }
        emit(instance, face, t);
        count++;
        last_instance = instance;
        last_face = face;
        last_t = t;
        // past the hit by 2^-40 of the scene or of the distance reached, the
        // distance held to NEAR_HITS scenes, or by the rounding of a far
        // hit's moved origin when that is more (refinding the hit doubles
        // it). 2^-40 of a far hit's own distance (about 9 at 1e13) jumped
        // surfaces that doubles still tell apart; within NEAR_HITS scenes
        // the rounding term is always the smaller, so every step there is
        // the bits it was
        const double reach = std::max({std::abs(o[0]), std::abs(o[1]), std::abs(o[2]), std::abs(t)});
        step =
            std::max(RELATIVE_STEP * std::max(scene, std::min(std::abs(t), NEAR_HITS * scene)), ROUNDING_STEP * reach);
        along = t + step;
    }
}

// One query of `level` traced on the device over `count` rays, `consume`
// reading each chunk's answers: false when the scene has no device (no
// reason: that is no fallback to warn about), when there are no rays,
// when this top level isn't on it or the device failed (then with the
// reason), or when a ray's value is one the CPU refuses (no reason: its
// check refuses it), and the CPU answers. No stop is polled: that would
// take the GIL while holding the device, and a device query is over in
// milliseconds
template <typename Consume>
bool trace_on_device(const RayMeshTrees& trees, const RayTopLevel& level, const trimkit::RayVectors& origins,
                     const trimkit::RayVectors& directions, std::size_t count, bool occlusion, std::size_t most,
                     const Consume& consume, std::string* reason) {
    if (!trees.device_meshes || count == 0) {
        return false;
    }
    if (level.direct == nullptr && !level.device_top) {
        *reason = level.device_reason;
        return false;
    }
    // one identity instance traces its mesh directly, as on the CPU
    const std::uint32_t mesh = level.direct != nullptr ? static_cast<std::uint32_t>(level.instances[0].blasIdx) : 0;
    return trimkit::vulkan::trace(*trees.device_meshes, level.device_top.get(), mesh, level.extent, occlusion, most,
                                  origins, directions, count, consume, reason);
}

// Every ray's crossings traced on the device, into the columns: the device
// runs each ray's loop as `crossings` does on the CPU and hands back a
// chunk's crossings in ray order; each is widened into place with its ray
// made the caller's and its instance mapped through `which`
bool device_crossings(const RayMeshTrees& trees, const RayTopLevel& level, const trimkit::RayVectors& origins,
                      const trimkit::RayVectors& directions, std::size_t count, std::size_t most,
                      trimkit::RayCrossings* columns, std::string* reason) {
    std::size_t total = 0;
    // an instance the device names indexes `which`: one it doesn't hold is
    // a device that can't be believed, and the CPU answers
    std::atomic<bool> believed{true};
    const std::size_t placed = level.which.size();
    const bool answered = trace_on_device(
        trees, level, origins, directions, count, false, most,
        [&](const trimkit::vulkan::Answers& answers) {
            // the columns grow as a vector does, to twice what they held
            const std::size_t needed = total + answers.total;
            if (needed > columns->rays.size()) {
                columns->resize(std::max(needed, 2 * columns->rays.size()));
            }
            std::int64_t* rays_out = trimkit::elements(columns->rays) + total;
            std::int64_t* instances_out = trimkit::elements(columns->instances) + total;
            std::int64_t* faces_out = trimkit::elements(columns->faces) + total;
            double* distances_out = trimkit::elements(columns->distances) + total;
            const std::int64_t base = static_cast<std::int64_t>(answers.begin);
            trimkit::in_blocks(answers.total, [&](std::size_t from, std::size_t to) {
                bool held = true;
                for (std::size_t h = from; h < to; h++) {
                    const std::size_t instance = answers.words[2 * h];
                    held = held && instance < placed;
                    rays_out[h] = base + answers.rays[h];
                    instances_out[h] = level.which[instance < placed ? instance : 0];
                    faces_out[h] = static_cast<std::int64_t>(answers.words[(2 * h) + 1]);
                }
                if (!held) {
                    believed.store(false, std::memory_order_relaxed);
                }
                std::memcpy(distances_out + from, answers.distances + from, (to - from) * sizeof(double));
            });
            total += answers.total;
        },
        reason);
    columns->resize(total);
    if (answered && !believed.load()) {
        *reason = "the device answered an instance the scene doesn't hold";
        return false;
    }
    return answered;
}

// Every ray's occlusion traced on the device, into `hits`
bool device_occlusion(const RayMeshTrees& trees, const RayTopLevel& level, const trimkit::RayVectors& origins,
                      const trimkit::RayVectors& directions, std::size_t count, std::vector<std::uint8_t>* hits,
                      std::string* reason) {
    return trace_on_device(
        trees, level, origins, directions, count, true, 1,
        [&](const trimkit::vulkan::Answers& answers) {
            std::uint8_t* out = hits->data() + answers.begin;
            for (std::size_t k = 0; k < answers.count; k++) {
                out[k] = answers.occlusion[k] != 0 ? 1 : 0;
            }
        },
        reason);
}

// Open the process's device and put every mesh on it: false with the
// reason when it can't
bool open_device(RayMeshTrees* trees, std::string* reason) {
    trimkit::vulkan::Device* device = trimkit::vulkan::shared_device(reason);
    if (device == nullptr) {
        return false;
    }
    static_assert(sizeof(tinybvh::BVH_Double::BVHNode) == sizeof(trimkit::vulkan::Node64));
    static_assert(sizeof(Vec) == 3 * sizeof(double));
    const std::vector<std::unique_ptr<RayMesh>>& meshes = trees->meshes;
    std::vector<trimkit::vulkan::MeshTree> device_trees(meshes.size());
    for (std::size_t m = 0; m < meshes.size(); m++) {
        const MeshBvh* bvh = meshes[m]->bvh();
        if (bvh == nullptr) {
            continue;  // a mesh without faces: no tree
        }
        device_trees[m].nodes = bvh->bvhNode;
        device_trees[m].node_count = static_cast<std::size_t>(bvh->usedNodes);
        device_trees[m].slots = bvh->primIdx;
        device_trees[m].slot_count = static_cast<std::size_t>(bvh->idxCount);
        device_trees[m].corners = meshes[m]->faces().data();
        device_trees[m].corner_count = meshes[m]->faces().size();
        device_trees[m].vertices = meshes[m]->vertices().data();
        device_trees[m].vertex_count = meshes[m]->vertices().size();
    }
    trees->device_meshes = trimkit::vulkan::put_scene(device, device_trees.data(), device_trees.size(), reason);
    return trees->device_meshes != nullptr;
}

// Put a top level on the device beside its scene, or record why it isn't
void upload_top(const RayMeshTrees& trees, RayTopLevel* top) {
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
    top->device_top = trimkit::vulkan::put_top(
        trees.device_meshes->device, tlas.bvhNode, static_cast<std::size_t>(tlas.usedNodes), tlas.primIdx,
        static_cast<std::size_t>(tlas.idxCount), inverses, instance_meshes, &top->device_reason);
}

}  // namespace

std::size_t trimkit::count_rays(const RayVectors& origins, const RayVectors& directions) {
    if (!origins.one && !directions.one && directions.count != origins.count) {
        refuse("origins and directions must have the same number of rays");
    }
    return origins.one ? directions.count : origins.count;
}

trimkit::RayMeshes::RayMeshes(std::vector<RayMeshInput> meshes, bool use_gpu)
    : trees_(std::make_unique<RayMeshTrees>()) {
    trees_->meshes.resize(meshes.size());
    trees_->blas.resize(meshes.size());
    for (std::size_t m = 0; m < meshes.size(); m++) {
        trees_->meshes[m] = std::make_unique<RayMesh>(meshes[m]);
        trees_->blas[m] = trees_->meshes[m]->bvh();
    }
    if (use_gpu) {
        std::string reason;
        if (!open_device(trees_.get(), &reason)) {
            trees_->device_failure = reason;
        }
    }
}

trimkit::RayMeshes::~RayMeshes() = default;

std::size_t trimkit::RayMeshes::size() const { return trees_->meshes.size(); }

const std::string& trimkit::RayMeshes::device_failure() const { return trees_->device_failure; }

void trimkit::RayMeshes::as_given(std::size_t mesh, std::vector<double>* vertices,
                                  std::vector<std::uint64_t>* faces) const {
    const std::vector<Vec>& points = trees_->meshes[mesh]->vertices();
    vertices->resize(3 * points.size());
    for (std::size_t i = 0; i < points.size(); i++) {
        for (std::size_t axis = 0; axis < 3; axis++) {
            (*vertices)[(3 * i) + axis] = points[i][static_cast<std::int32_t>(axis)];
        }
    }
    *faces = trees_->meshes[mesh]->faces();
}

std::shared_ptr<const trimkit::RayTopLevel> trimkit::RayMeshes::place(Instances instances) const {
    std::shared_ptr<RayTopLevel> top = std::make_shared<RayTopLevel>();
    top->given = std::move(instances);
    const bool all_identity = place_instances(*trees_, top.get());
    // one identity instance traces its mesh directly: the same answers as
    // through a top level, without the extra level per ray (19% of a query)
    if (top->instances.size() == 1 && all_identity) {
        top->direct = trees_->meshes[static_cast<std::size_t>(top->instances[0].blasIdx)].get();
    } else if (!top->instances.empty()) {
        tinybvh::BVH_Double** meshes = trees_->blas.data();
        top->tlas = built_bvh([&](MeshBvh& bvh) {
            bvh.Build(top->instances.data(), top->instances.size(), meshes, trees_->blas.size());
        });
        pad(*top->tlas, top->extent);
    }
    if (trees_->device_meshes && top->tlas) {
        upload_top(*trees_, top.get());
    }
    return top;
}

const trimkit::Instances& trimkit::RayMeshes::placed_from(const RayTopLevel& top) { return top.given; }

std::vector<double> trimkit::RayMeshes::boxes(Instances instances) const {
    RayTopLevel placed;
    placed.given = std::move(instances);
    place_instances(*trees_, &placed);
    return std::move(placed.boxes);
}

std::string trimkit::RayMeshes::intersect_any(const RayTopLevel& top, const RayVectors& origins,
                                              const RayVectors& directions, std::size_t count,
                                              const parallel::Stop& stop, std::vector<std::uint8_t>* hits) const {
    hits->assign(count, 0);
    const tinybvh::BVH_Double* bvh = top.bvh();
    std::string fallback;
    if (bvh == nullptr || !device_occlusion(*trees_, top, origins, directions, count, hits, &fallback)) {
        // the device checks each ray's values as it copies them: here, now
        refuse_rays_beyond(origins, "origins");
        refuse_rays_beyond(directions, "directions");
        std::uint8_t* out = hits->data();
        in_parallel(
            bvh != nullptr ? count : 0,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; i++) {
                    double o[3];
                    double d[3];
                    ray_row(origins, i, o);
                    ray_row(directions, i, d);
                    normalize_exponent(d);
                    // every ray written, whatever a device left there
                    const tinybvh::RayEx ray = make_tinybvh_ray(o, d);
                    out[i] = !still(d) && bvh->IsOccluded(ray) ? 1 : 0;
                }
            },
            stop);
    }
    return fallback;
}

std::string trimkit::RayMeshes::intersect(const RayTopLevel& top, const RayVectors& origins,
                                          const RayVectors& directions, std::size_t count, std::size_t most,
                                          const parallel::Stop& stop, RayCrossings* columns) const {
    // the device writes the columns itself; the CPU gathers hits a chunk of
    // rays at a time and writes them after
    std::string fallback;
    if (top.bvh() == nullptr || !device_crossings(*trees_, top, origins, directions, count, most, columns, &fallback)) {
        // the device checks each ray's values as it copies them: here, now
        refuse_rays_beyond(origins, "origins");
        refuse_rays_beyond(directions, "directions");
        Gathered<Hit> found;
        if (top.bvh() != nullptr) {
            gather_in_parallel<Hit>(
                count, &found,
                [&](std::size_t begin, std::size_t end, std::vector<Hit>& hits) {
                    // room for the usual: a hit per ray with a cap of one, else
                    // two, a closed surface entered and left
                    hits.reserve((end - begin) * std::min<std::size_t>(most, 2));
                    for (std::size_t i = begin; i < end; i++) {
                        double o[3];
                        double d[3];
                        ray_row(origins, i, o);
                        ray_row(directions, i, d);
                        if (still(d)) {
                            continue;
                        }
                        normalize_exponent(d);
                        const std::int64_t ray = static_cast<std::int64_t>(i);
                        // distances in the scene's units rather than the direction's
                        const double scale = length(d);
                        crossings(*trees_, top, o, d, scale, most,
                                  [&](std::int64_t instance, std::int64_t face, double t) {
                                      hits.push_back({ray, instance, face, t * scale});
                                  });
                    }
                },
                stop);
        }
        columns->resize(found.size());
        std::int64_t* rays_out = elements(columns->rays);
        std::int64_t* instances_out = elements(columns->instances);
        std::int64_t* faces_out = elements(columns->faces);
        double* distances_out = elements(columns->distances);
        found.write([&](std::size_t offset, const Hit* hits, std::size_t found_count) {
            for (std::size_t h = 0; h < found_count; h++) {
                rays_out[offset + h] = hits[h].ray;
                instances_out[offset + h] = hits[h].instance;
                faces_out[offset + h] = hits[h].face;
                distances_out[offset + h] = hits[h].t;
            }
        });
    }
    return fallback;
}

trimkit::BoxTree::BoxTree(const double* boxes, std::size_t count, std::size_t dimension)
    : tree_(std::make_unique<BoxBvh>(BoxBvh::corners_of(boxes, count, dimension, "boxes"), dimension)) {}

trimkit::BoxTree::~BoxTree() = default;
trimkit::BoxTree::BoxTree(BoxTree&& other) noexcept = default;
trimkit::BoxTree& trimkit::BoxTree::operator=(BoxTree&& other) noexcept = default;

std::size_t trimkit::BoxTree::size() const { return tree_->boxes.size() / 2; }

std::size_t trimkit::BoxTree::dimension() const { return tree_->dimension; }

std::vector<double> trimkit::BoxTree::bounds() const {
    const std::size_t d = tree_->dimension;
    std::vector<double> bounds(2 * d);
    for (std::size_t axis = 0; axis < d; axis++) {
        bounds[axis] = tree_->lowest[axis];
        bounds[d + axis] = tree_->highest[axis];
    }
    return bounds;
}

std::vector<double> trimkit::BoxTree::boxes() const {
    const std::size_t d = tree_->dimension;
    const std::vector<Vec>& corners = tree_->boxes;
    std::vector<double> boxes(d * corners.size());
    for (std::size_t corner = 0; corner < corners.size(); corner++) {
        double* out = &boxes[d * corner];
        out[0] = corners[corner][0];
        out[1] = corners[corner][1];
        if (d == 3) {
            out[2] = corners[corner][2];
        }
    }
    return boxes;
}

std::vector<trimkit::Unset<std::int64_t>> trimkit::BoxTree::intersection(const double* queries, std::size_t count,
                                                                         bool one, const parallel::Stop& stop) const {
    const BoxBvh& tree = *tree_;
    const std::vector<Vec> corners = BoxBvh::corners_of(queries, count, tree.dimension, "queries");
    Gathered<std::int64_t> gathered;
    gather_in_parallel<std::int64_t>(
        count, &gathered,
        [&](std::size_t begin, std::size_t end, std::vector<std::int64_t>& hits) {
            Stack stack = make_stack();
            std::vector<std::int64_t> boxes_hit;
            for (std::size_t q = begin; q < end; q++) {
                boxes_hit.clear();
                tree.overlapping(corners[2 * q], corners[(2 * q) + 1], stack,
                                 [&](std::uint64_t box) { boxes_hit.push_back(static_cast<std::int64_t>(box)); });
                // ascending boxes, so the same query always reads the same
                std::sort(boxes_hit.begin(), boxes_hit.end());
                for (const std::int64_t box : boxes_hit) {
                    if (!one) {
                        hits.push_back(static_cast<std::int64_t>(q));
                    }
                    hits.push_back(box);
                }
            }
        },
        stop);
    return gathered_rows(gathered);
}

void trimkit::BoxTree::nearest(const double* queries, std::size_t count, bool one, std::size_t maximum_hits,
                               const parallel::Stop& stop, std::vector<Unset<std::int64_t>>* hits,
                               std::vector<Unset<double>>* distances) const {
    const BoxBvh& tree = *tree_;
    const std::vector<Vec> corners = BoxBvh::corners_of(queries, count, tree.dimension, "queries");
    const std::size_t most = std::min(maximum_hits, size());
    const std::size_t rows = element_count(most, count);
    // one query's hits are its boxes, many queries' are [query, box]
    const std::size_t width = one ? 1 : 2;
    hits->resize(element_count(rows, width));
    distances->resize(rows);
    std::int64_t* hits_out = elements(*hits);
    double* distances_out = elements(*distances);
    // no rows, no walk: a tree without boxes has no root to start at
    in_parallel(
        rows == 0 ? 0 : count,
        [&](std::size_t begin, std::size_t end) {
            std::vector<NearestNode> stack;
            stack.reserve(std::size_t{2} * TINYBVH_STACK_SIZE);
            std::vector<NearestBox> best;
            best.reserve(most);
            for (std::size_t q = begin; q < end; q++) {
                tree.nearest_of(corners[2 * q], corners[(2 * q) + 1], most, stack, best);
                std::int64_t* hit = hits_out + (width * most * q);
                double* distance = distances_out + (most * q);
                for (const NearestBox& found : best) {
                    if (!one) {
                        *hit++ = static_cast<std::int64_t>(q);
                    }
                    *hit++ = static_cast<std::int64_t>(found.box);
                    *distance++ = found.distance;
                }
            }
        },
        stop);
}
