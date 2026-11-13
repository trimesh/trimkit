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
// (src/vulkan/scene.hpp, the only Vulkan this file sees) and traces there: the same trees, and a shader
// (src/vulkan/trace.comp) that makes every decision tinybvh's traversal makes,
// most in f32 behind a proven bound and the rest with tinybvh's own f64
// operations, so the same bytes come back; the crossing loop below
// (`crossings`) runs in the shader too, written the same way. A device that
// can't be used, now or later, is a warning once, and the CPU answers.

#include "common.hpp"
#include "vulkan/scene.hpp"

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
using Points = trimkit::In<double, -1, 3>;
// rays' origins or directions: one each, or one for every ray
using RaysArg = trimkit::RaysArgument;
using Rays = trimkit::Rays;
using FacesArg = trimkit::Array<std::int64_t, -1, 3>;
using BoundsArg = trimkit::Array<double, -1, 2, -1>;
using Bounds = trimkit::In<double, -1, 2, -1>;

using BoxQueriesArgument = trimkit::BoxQueriesArgument;

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

// One hit as `intersect` reports it
struct Hit {
    std::int64_t ray;
    std::int64_t instance;
    std::int64_t face;
    double t;
};

// The four columns `intersect` answers, each written once; the device fills
// them a chunk at a time, growing them as a vector grows
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
struct TopLevel {
    trimkit::Instances given;
    std::vector<tinybvh::BLASInstanceEx> instances;
    std::vector<std::int64_t> which;
    std::vector<double> boxes;
    // the largest coordinate magnitude any placed box reaches: the scene's size
    double extent = 0.0;
    const RayMesh* direct = nullptr;
    // the top level on the device when the scene traces there, or why not
    std::unique_ptr<trimkit::vulkan::DeviceArrays> device_top;
    std::string device_reason;
    // declared last, so it is destroyed before the instances it points into
    std::unique_ptr<MeshBvh> tlas;

    // what a first-hit or any-hit ray traces, or null with nothing to hit
    const tinybvh::BVH_Double* bvh() const { return direct != nullptr ? direct->bvh() : tlas.get(); }
};

class RayScene {
public:
    // it owns its meshes (std::vector of std::unique_ptr claims to be
    // copyable to nanobind's traits, so say it isn't)
    RayScene(const RayScene&) = delete;
    RayScene& operator=(const RayScene&) = delete;

    RayScene(const trimkit::Meshes& meshes, trimkit::Flag use_gpu) {
        std::vector<trimkit::Mesh<std::uint64_t>> read = trimkit::read_meshes<std::uint64_t>(meshes);
        meshes_.resize(read.size());
        blas_.resize(read.size());
        for (std::size_t m = 0; m < read.size(); m++) {
            meshes_[m] = std::make_unique<RayMesh>(std::move(read[m]));
            blas_[m] = meshes_[m]->bvh();
        }
        if (use_gpu.value) {
            std::string reason;
            if (!open_device(&reason)) {
                trimkit::warn("use_gpu: " + reason + "; this scene answers on the CPU");
            }
        }
        // the default instances, each mesh once where it is, built now
        last_ = build(trimkit::read_instances(trimkit::InstanceViews{}, meshes_.size(), /*rigid=*/false));
    }

    // the boxes a query with these instances culls rays by, without building
    // a top level: the same placement a query makes
    trimkit::BoxesObject boxes(const trimkit::InstanceMatrices& matrices_argument,
                               const trimkit::InstanceIndex& index_argument) const {
        TopLevel placed;
        placed.given = trimkit::read_instances(trimkit::view_instances(index_argument, matrices_argument),
                                               meshes_.size(), /*rigid=*/false);
        place(&placed);
        return trimkit::scene_boxes(placed.boxes);
    }

    std::size_t size() const { return meshes_.size(); }

    trimkit::Out<bool> intersect_any(const RaysArg& origins_argument, const RaysArg& directions_argument,
                                     const trimkit::InstanceMatrices& matrices_argument,
                                     const trimkit::InstanceIndex& index_argument) const {
        const Rays origins = trimkit::read_rays(origins_argument, "origins");
        const Rays directions = trimkit::read_rays(directions_argument, "directions");
        const std::size_t n = count_rays(origins, directions);
        const std::shared_ptr<const TopLevel> top = top_level(matrices_argument, index_argument);
        std::vector<std::uint8_t> hits(n, 0);
        const tinybvh::BVH_Double* bvh = top->bvh();
        std::string fallback;
        if (bvh == nullptr || !device_occlusion(*top, origins, directions, n, &hits, &fallback)) {
            // the device checks each ray's values as it copies them: here, now
            trimkit::check_rays(origins, "origins");
            trimkit::check_rays(directions, "directions");
            trimkit::in_parallel(bvh != nullptr ? n : 0, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; i++) {
                    double o[3];
                    double d[3];
                    trimkit::ray_row(origins, i, o);
                    trimkit::ray_row(directions, i, d);
                    // every ray written, whatever a device left there
                    const tinybvh::RayEx ray = make_tinybvh_ray(o, d);
                    hits[i] = !still(d) && bvh->IsOccluded(ray) ? 1 : 0;
                }
            });
        }
        warn_fallback(fallback);
        return trimkit::to_numpy<bool>(std::move(hits), {n});
    }

    Hits intersect(const RaysArg& origins_argument, const RaysArg& directions_argument,
                   const trimkit::InstanceMatrices& matrices_argument, const trimkit::InstanceIndex& index_argument,
                   const MaximumHits& maximum_hits) const {
        if (maximum_hits && *maximum_hits < 1) {
            trimkit::value_error("maximum_hits must be at least 1");
        }
        const std::size_t most = maximum_hits ? static_cast<std::size_t>(std::int64_t{*maximum_hits})
                                              : std::numeric_limits<std::size_t>::max();
        const Rays origins = trimkit::read_rays(origins_argument, "origins");
        const Rays directions = trimkit::read_rays(directions_argument, "directions");
        const std::size_t n = count_rays(origins, directions);
        const std::shared_ptr<const TopLevel> top = top_level(matrices_argument, index_argument);
        const TopLevel& level = *top;
        // the device writes the columns itself; the CPU gathers hits a chunk of
        // rays at a time and writes them after
        Columns columns;
        std::string fallback;
        if (level.bvh() == nullptr || !device_crossings(level, origins, directions, n, most, &columns, &fallback)) {
            // the device checks each ray's values as it copies them: here, now
            trimkit::check_rays(origins, "origins");
            trimkit::check_rays(directions, "directions");
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
                            trimkit::ray_row(origins, i, o);
                            trimkit::ray_row(directions, i, d);
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
    // exactly the arrays it was built from, else a new one, remembered. The
    // arrays are viewed once, for the comparison and for the read after it
    std::shared_ptr<const TopLevel> top_level(const trimkit::InstanceMatrices& matrices_argument,
                                              const trimkit::InstanceIndex& index_argument) const {
        const trimkit::InstanceViews views = trimkit::view_instances(index_argument, matrices_argument);
        std::shared_ptr<const TopLevel> last = remembered();
        if (trimkit::same_instances(views, last->given)) {
            return last;
        }
        std::shared_ptr<const TopLevel> built = build(trimkit::read_instances(views, meshes_.size(), /*rigid=*/false));
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

    // a top level for instances already read
    std::shared_ptr<const TopLevel> build(trimkit::Instances instances) const {
        std::shared_ptr<TopLevel> top = std::make_shared<TopLevel>();
        top->given = std::move(instances);
        const bool all_identity = place(top.get());
        // one identity instance traces its mesh directly: the same answers as
        // through a top level, without the extra level per ray (19% of a query)
        if (top->instances.size() == 1 && all_identity) {
            top->direct = meshes_[static_cast<std::size_t>(top->instances[0].blasIdx)].get();
        } else if (!top->instances.empty()) {
            const nb::gil_scoped_release release;
            top->tlas = new_bvh();
            // tinybvh takes the meshes' array non-const, though it only reads it
            tinybvh::BVH_Double** meshes =
                const_cast<tinybvh::BVH_Double**>(blas_.data());  // NOLINT(cppcoreguidelines-pro-type-const-cast)
            top->tlas->Build(top->instances.data(), top->instances.size(), meshes, blas_.size());
            pad(*top->tlas, top->extent);
        }
        if (device_meshes_ && top->tlas) {
            const nb::gil_scoped_release release;
            upload_top(top.get());
        }
        return top;
    }

    // Place `top->given`: each instance of a mesh with faces kept, in its
    // order, with its world box; whether every kept transform is the identity
    bool place(TopLevel* top) const {
        constexpr std::array<double, 16> IDENTITY{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const trimkit::Instances& instances = top->given;
        bool all_identity = true;
        // as many as there could be, written by index, then cut to those kept
        std::size_t kept = 0;
        top->instances.resize(instances.count, tinybvh::BLASInstanceEx(0));
        top->which.resize(instances.count);
        top->boxes.resize(6 * instances.count);
        // read and checked: the rest touches no Python
        const nb::gil_scoped_release release;
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
            all_identity = all_identity && std::equal(transform, transform + 16, IDENTITY.begin());
            for (std::int32_t axis = 0; axis < 3; axis++) {
                top->extent =
                    std::max({top->extent, std::abs(instance.aabbMin[axis]), std::abs(instance.aabbMax[axis])});
                box[axis] = instance.aabbMin[axis];
                box[3 + axis] = instance.aabbMax[axis];
            }
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
    // about 1e-12 of the scene (what the nodes are padded by), and two
    // surfaces closer than that step count once. Each hit's distance is then
    // measured again from the ray's own origin, by tinybvh's triangle test on
    // the triangle found: exactly what a cap of one hit reports for it,
    // whatever else the ray crossed first.
    //
    // `crossings` does that for one ray on the CPU, calling `emit(instance,
    // face, t)` for each; src/vulkan/trace.comp does the same for every ray
    // of a chunk on the device, operation for operation
    template <typename Emit>
    void crossings(const TopLevel& level, const double* o, const double* d, double length_of_d, std::size_t most,
                   const Emit& emit) const {
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
                    mesh = meshes_[static_cast<std::size_t>(placed.blasIdx)]->bvh();
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
            step = RELATIVE_STEP * std::max(scene, std::abs(t));
            along = t + step;
        }
    }

    // One query of `level` traced on the device over `count` rays, `consume`
    // reading each chunk's answers: false when the scene has no device (no
    // reason: that is no fallback to warn about), when there are no rays,
    // when this top level isn't on it or the device failed (then with the
    // reason), or when a ray's value is one `read` refuses (no reason: the
    // caller's `check_rays` raises it), and the CPU answers. No Ctrl-C is
    // looked for: that would take the GIL while holding the device, and a
    // device query is over in milliseconds
    template <typename Consume>
    bool trace_on_device(const TopLevel& level, const Rays& origins, const Rays& directions, std::size_t count,
                         bool occlusion, std::size_t most, const Consume& consume, std::string* reason) const {
        if (!device_meshes_ || count == 0) {
            return false;
        }
        if (level.direct == nullptr && !level.device_top) {
            *reason = level.device_reason;
            return false;
        }
        // one identity instance traces its mesh directly, as on the CPU
        const std::uint32_t mesh = level.direct != nullptr ? static_cast<std::uint32_t>(level.instances[0].blasIdx) : 0;
        const nb::gil_scoped_release release;
        return trimkit::vulkan::trace(*device_meshes_, level.device_top.get(), mesh, level.extent, occlusion, most,
                                      origins, directions, count, consume, reason);
    }

    // Every ray's crossings traced on the device, into the columns: the device
    // runs each ray's loop as `crossings` does on the CPU and hands back a
    // chunk's crossings in ray order; each is widened into place with its ray
    // made the caller's and its instance mapped through `which`
    bool device_crossings(const TopLevel& level, const Rays& origins, const Rays& directions, std::size_t count,
                          std::size_t most, Columns* columns, std::string* reason) const {
        std::size_t total = 0;
        // an instance the device names indexes `which`: one it doesn't hold is
        // a device that can't be believed, and the CPU answers
        std::atomic<bool> believed{true};
        const std::size_t placed = level.which.size();
        const bool answered = trace_on_device(
            level, origins, directions, count, false, most,
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
                trimkit::in_blocks_without_gil(answers.total, [&](std::size_t from, std::size_t to) {
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
    bool device_occlusion(const TopLevel& level, const Rays& origins, const Rays& directions, std::size_t count,
                          std::vector<std::uint8_t>* hits, std::string* reason) const {
        return trace_on_device(
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

    // Open the process's device and put every mesh on it: false with the
    // reason when it can't
    bool open_device(std::string* reason) {
        trimkit::vulkan::Device* device = trimkit::vulkan::shared_device(reason);
        if (device == nullptr) {
            return false;
        }
        static_assert(sizeof(tinybvh::BVH_Double::BVHNode) == sizeof(trimkit::vulkan::Node64));
        static_assert(sizeof(Vec) == 3 * sizeof(double));
        std::vector<trimkit::vulkan::MeshTree> trees(meshes_.size());
        for (std::size_t m = 0; m < meshes_.size(); m++) {
            const MeshBvh* bvh = meshes_[m]->bvh();
            if (bvh == nullptr) {
                continue;  // a mesh without faces: no tree
            }
            trees[m].nodes = bvh->bvhNode;
            trees[m].node_count = static_cast<std::size_t>(bvh->usedNodes);
            trees[m].slots = bvh->primIdx;
            trees[m].slot_count = static_cast<std::size_t>(bvh->idxCount);
            trees[m].corners = meshes_[m]->faces().data();
            trees[m].corner_count = meshes_[m]->faces().size();
            trees[m].vertices = meshes_[m]->vertices().data();
            trees[m].vertex_count = meshes_[m]->vertices().size();
        }
        const nb::gil_scoped_release release;
        device_meshes_ = trimkit::vulkan::put_scene(device, trees.data(), trees.size(), reason);
        return device_meshes_ != nullptr;
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
        top->device_top = trimkit::vulkan::put_top(
            device_meshes_->device, tlas.bvhNode, static_cast<std::size_t>(tlas.usedNodes), tlas.primIdx,
            static_cast<std::size_t>(tlas.idxCount), inverses, instance_meshes, &top->device_reason);
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
    std::unique_ptr<trimkit::vulkan::DeviceArrays> device_meshes_;
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
        build(corners_of(boxes.data(), boxes.shape(0), "boxes"));
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
        return trimkit::to_numpy_rows(std::move(boxes), {2, dimension_});
    }

    // every box overlapping each query, touching included like rtree: one
    // query, (2d,) or (2, d), answers its sorted boxes, (n,), and many,
    // (q, 2d) or (q, 2, d), their [query, box] rows, (k, 2). Every form is
    // (q, 2, d) in memory, so they read as it and share one traversal; the
    // last extent tells the layouts apart, as 2d is never d
    trimkit::Out<std::int64_t> intersection(const BoxQueriesArgument& argument) const {
        // its shape as numpy sees it: none when it isn't array-like at all
        const nb::object seen = trimkit::detail::seen_as_array(argument.object);
        std::vector<std::size_t> extents;
        if (seen.is_valid()) {
            for (const nb::handle extent : nb::borrow<nb::tuple>(seen.attr("shape"))) {
                extents.push_back(nb::cast<std::size_t>(extent));
            }
        }
        const std::size_t d = dimension_;
        const std::size_t last = extents.empty() ? 0 : extents.back();
        if (extents.size() == 1 && last == 2 * d) {
            return answer(trimkit::read<double, -1>(argument.object, "queries").data(), 1, true);
        }
        if (extents.size() == 2 && extents[0] == 2 && last == d) {
            return answer(trimkit::read<double, -1, -1>(argument.object, "queries").data(), 1, true);
        }
        if (extents.size() == 2 && last == 2 * d) {
            return answer(trimkit::read<double, -1, -1>(argument.object, "queries").data(), extents[0], false);
        }
        if (extents.size() == 3 && extents[1] == 2 && last == d) {
            return answer(trimkit::read<double, -1, 2, -1>(argument.object, "queries").data(), extents[0], false);
        }
        // a shape that's none of them is the wrong type, as every argument's is
        trimkit::type_error("queries must be float64 of shape (" + std::to_string(2 * d) + ",) or (2, " +
                            std::to_string(d) + ") for one query, or (*, " + std::to_string(2 * d) + ") or (*, 2, " +
                            std::to_string(d) + ") for many, not " + trimkit::describe(argument.object));
    }

private:
    // `count` queries of (2, d) corners at `data`: their sorted boxes for
    // `one` query, else [query, box] rows in query order
    trimkit::Out<std::int64_t> answer(const double* data, std::size_t count, bool one) const {
        const std::vector<Vec> corners = corners_of(data, count, "queries");
        trimkit::Gathered<std::int64_t> pairs;
        trimkit::gather_in_parallel<std::int64_t>(
            count, &pairs, [&](std::size_t begin, std::size_t end, std::vector<std::int64_t>& rows) {
                Stack stack = make_stack();
                std::vector<std::int64_t> boxes_hit;
                for (std::size_t q = begin; q < end; q++) {
                    boxes_hit.clear();
                    overlapping(corners[2 * q], corners[(2 * q) + 1], stack,
                                [&](std::uint64_t box) { boxes_hit.push_back(static_cast<std::int64_t>(box)); });
                    // ascending boxes, so the same query always reads the same
                    std::sort(boxes_hit.begin(), boxes_hit.end());
                    for (const std::int64_t box : boxes_hit) {
                        rows.push_back(static_cast<std::int64_t>(q));
                        rows.push_back(box);
                    }
                }
            });
        auto rows = trimkit::gathered_rows(pairs);
        if (!one) {
            return trimkit::to_numpy_rows<std::int64_t>(std::move(rows), {2});
        }
        // one query: its column of boxes
        const std::int64_t* row = trimkit::elements(rows);
        std::vector<std::int64_t> boxes(rows.size() / 2);
        for (std::size_t i = 0; i < boxes.size(); i++) {
            boxes[i] = row[(2 * i) + 1];
        }
        return trimkit::to_numpy(std::move(boxes), {boxes.size()});
    }

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

    // `n` (min, max) corner pairs of the boxes' dimension at `b`, as 3D
    // points, each pair checked to be ordered
    std::vector<Vec> corners_of(const double* b, std::size_t n, const char* name) const {
        const std::size_t d = dimension_;
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
    static constexpr double INF = std::numeric_limits<double>::infinity();
    double lowest_[3] = {INF, INF, INF};
    double highest_[3] = {-INF, -INF, -INF};
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
        .def("intersection", &Boxes::intersection, nb::arg("queries"),
             R"(Every box overlapping one query box or each of many, touching included.

A superset of an R-tree's `intersection`: one query box, flat as
(min..., max...) or as its two corners, answers the boxes it overlaps,
and a batch of either answers every overlap at once.

Parameters
----------
queries : (2d,), (2, d), (m, 2d) or (m, 2, d) float64
  One query's or each query's minimum and maximum corner, in the
  boxes' dimension d: flat, (min..., max...), or as two corners.

Returns
-------
hits : (n,) or (k, 2) int64
  For one query, every box it overlaps, ascending. For many,
  rows of [query, box] for every overlap, sorted by query and
  then box.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> hits = tree.intersection(np.array([0.5, 0.5, 2.5, 2.5]))
>>> hits.tolist()
[0, 1]
>>> tree.intersection(np.array([[[0.5, 0.5], [2.5, 2.5]]])).tolist()
[[0, 0], [0, 1]])");
}
