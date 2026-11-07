// Ray queries against an instanced scene of triangle meshes, and box overlap
// queries against a set of boxes, with jbikker/tinybvh.
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

#include "common.hpp"

#include <tiny_bvh.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using Vec = tinybvh::bvhdbl3;
using Transforms = trimkit::Array<double, -1, 4, 4>;
using Numbers = trimkit::Array<std::int64_t, -1>;
using PointsArg = trimkit::Array<double, -1, 3>;
using Points = trimkit::In<double, -1, 3>;
using FacesArg = trimkit::Array<std::int64_t, -1, 3>;
using BoundsArg = trimkit::Array<double, -1, 2, -1>;
using Bounds = trimkit::In<double, -1, 2, -1>;

// A watertight triangle test needs conservative traversal: the slab test
// rounds on the order of an ulp of the coordinates, so without padding a ray
// through a vertex or edge can skip the nodes holding it (Ize 2013). Padding
// by this much of the largest coordinate covers ray origins up to thousands
// of times farther out than the geometry
void pad(tinybvh::BVH_Double& bvh, double extent) {
    const Vec margin(1e-12 * extent);
    for (std::uint64_t i = 0; i < bvh.usedNodes; i++) {
        auto& node = bvh.bvhNode[i];
        node.aabbMin = node.aabbMin - margin;
        node.aabbMax = node.aabbMax + margin;
    }
}

// A traversal stack of node indices: one per chunk of queries, reused for
// every query in it, so it allocates once and grows only on a pathological
// tree (deeply nested boxes) rather than overflowing
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

// A ray as tinybvh's traversals see it: origin, direction and its safe
// reciprocal, so the tests below round exactly like tinybvh's own
struct Ray {
    Vec origin;
    Vec direction;
    Vec reciprocal;
};

Ray make_ray(const Vec& origin, const Vec& direction) {
    return {origin, direction, tinybvh::tinybvh_safercp(direction)};
}

// tinybvh's two-node slab test, for one node, without shrinking the ray:
// every node it would visit for a first hit is visited here too
bool slab(const tinybvh::BVH_Double::BVHNode& node, const Ray& ray) {
    // per axis, the near and far planes in the ray's direction, each found as
    // tinybvh does: fma(plane, 1 / direction, -origin / direction)
    std::array<double, 3> near{};
    std::array<double, 3> far{};
    for (std::int32_t a = 0; a < 3; a++) {
        const double r = ray.reciprocal[a];
        const double o = ray.origin[a] * r;
        const bool positive = r >= 0;
        near[static_cast<std::size_t>(a)] = tinybvh::tinybvh_fma(positive ? node.aabbMin[a] : node.aabbMax[a], r, -o);
        far[static_cast<std::size_t>(a)] = tinybvh::tinybvh_fma(positive ? node.aabbMax[a] : node.aabbMin[a], r, -o);
    }
    // nested exactly as tinybvh's SLAB_TEST_TWO_NODES nests them, with its own
    // min and max, which drop a NaN (a plane at 0 * inf, from an axis the ray
    // doesn't move along) where std::max would keep it and miss the node
    const double tmin =
        tinybvh::tinybvh_max(tinybvh::tinybvh_max(near[0], near[1]), tinybvh::tinybvh_max(near[2], 0.0));
    const double tmax =
        tinybvh::tinybvh_min(tinybvh::tinybvh_min(far[0], far[1]), tinybvh::tinybvh_min(far[2], BVH_DBL_FAR));
    return tmax >= tmin;
}

// Woop, Benthin and Wald's watertight ray/triangle test (2013), arithmetic
// for arithmetic what tinybvh's `TriHit` does with WATERTIGHT_TRITEST, so a
// hit here has exactly the distance `intersect_first` reports for it
bool watertight(const Ray& ray, const Vec& v0, const Vec& v1, const Vec& v2, double& distance) {
    const std::uint32_t kz = tinybvh::tinybvh_maxdim(ray.direction);
    std::uint32_t kx = (1U << kz) & 3U;
    std::uint32_t ky = (1U << kx) & 3U;
    if (ray.direction[static_cast<std::int32_t>(kz)] < 0) {
        std::swap(kx, ky);
    }
    const auto x = static_cast<std::int32_t>(kx);
    const auto y = static_cast<std::int32_t>(ky);
    const auto z = static_cast<std::int32_t>(kz);
    const double sz = ray.reciprocal[z];
    const double sx = ray.direction[x] * sz;
    const double sy = ray.direction[y] * sz;
    const Vec c = v0 - ray.origin;
    const Vec a = v1 - ray.origin;
    const Vec b = v2 - ray.origin;
    const double ax = a[x] - (sx * a[z]);
    const double ay = a[y] - (sy * a[z]);
    const double bx = b[x] - (sx * b[z]);
    const double by = b[y] - (sy * b[z]);
    const double cx = c[x] - (sx * c[z]);
    const double cy = c[y] - (sy * c[z]);
    const double u = (cx * by) - (cy * bx);
    const double v = (ax * cy) - (ay * cx);
    const double w = (bx * ay) - (by * ax);
    // inside only with all three edge functions of one sign (or zero)
    const bool inside = (u >= 0 && v >= 0 && w >= 0) || (u <= 0 && v <= 0 && w <= 0);
    if (!inside) {
        return false;
    }
    const double det = u + v + w;
    if (det == 0) {
        return false;
    }
    const double t = ((u * (sz * a[z])) + (v * (sz * b[z])) + (w * (sz * c[z]))) * (1.0 / det);
    // written so a NaN distance is never ahead
    const bool ahead = t >= 0 && t < BVH_DBL_FAR;
    if (!ahead) {
        return false;
    }
    distance = t;
    return true;
}

tinybvh::RayEx make_tinybvh_ray(const double* o, const double* d) {
    return {Vec(o[0], o[1], o[2]), Vec(d[0], d[1], d[2])};
}

// a ray without a direction goes nowhere and hits nothing
bool still(const double* d) { return d[0] == 0.0 && d[1] == 0.0 && d[2] == 0.0; }

// without squaring, so tiny and huge directions neither underflow nor overflow
double length(const double* d) { return std::hypot(d[0], d[1], d[2]); }

void check_rays(const Points& origins, const Points& directions) {
    if (directions.shape(0) != origins.shape(0)) {
        trimkit::value_error("origins and directions must have the same number of rays");
    }
}

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
            bvh_ = std::make_unique<tinybvh::BVH_Double>();
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
    tinybvh::BVH_Double* bvh() const { return bvh_.get(); }
    std::size_t size() const { return faces_.size() / 3; }

    // every face of this mesh `ray` (in the mesh's frame) hits, appended
    template <typename Emit>
    void every_hit(const Ray& ray, Stack& stack, const Emit& emit) const {
        if (!bvh_) {
            return;
        }
        stack.push_back(0);
        while (!stack.empty()) {
            const auto& node = bvh_->bvhNode[pop(stack)];
            if (!slab(node, ray)) {
                continue;
            }
            if (!node.isLeaf()) {
                stack.push_back(node.leftFirst);
                stack.push_back(node.leftFirst + 1);
                continue;
            }
            for (std::uint64_t j = node.leftFirst; j < node.leftFirst + node.triCount; j++) {
                // tinybvh's indices are 64-bit everywhere; every one is a
                // checked index into our own vectors, so it fits a size_t
                const auto face = static_cast<std::size_t>(bvh_->primIdx[j]);
                const auto corner = [&](std::size_t k) -> const Vec& {
                    return vertices_[static_cast<std::size_t>(faces_[(3 * face) + k])];
                };
                double t = 0.0;
                if (watertight(ray, corner(0), corner(1), corner(2), t)) {
                    emit(face, t);
                }
            }
        }
    }

private:
    std::vector<Vec> vertices_;
    std::vector<std::uint64_t> faces_;
    // declared last, so it is destroyed before the geometry it points into
    std::unique_ptr<tinybvh::BVH_Double> bvh_;
};

using First = std::tuple<trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>, trimkit::Out<double>>;
using All = std::tuple<trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>,
                       trimkit::Out<double>>;

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
    const RayMesh* direct = nullptr;
    // declared last, so it is destroyed before the instances it points into
    std::unique_ptr<tinybvh::BVH_Double> tlas;

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

    explicit RayScene(const trimkit::Meshes& meshes) {
        for (const nb::handle item : meshes) {
            const std::string name = "meshes[" + std::to_string(meshes_.size()) + "]";
            meshes_.push_back(std::make_unique<RayMesh>(trimkit::read_mesh<std::uint64_t>(item, name)));
        }
        blas_.resize(meshes_.size());
        for (std::size_t m = 0; m < meshes_.size(); m++) {
            blas_[m] = meshes_[m]->bvh();
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

    First intersect_first(const PointsArg& origins_argument, const PointsArg& directions_argument,
                          const Transforms& matrices_argument, const Numbers& index_argument) const {
        const Points origins = trimkit::read(origins_argument, "origins");
        const Points directions = trimkit::read(directions_argument, "directions");
        check_rays(origins, directions);
        const std::shared_ptr<const TopLevel> top = top_level(matrices_argument, index_argument);
        const std::size_t n = origins.shape(0);
        std::vector<std::int64_t> instances(n, -1);
        std::vector<std::int64_t> faces(n, -1);
        std::vector<double> distances(n, std::numeric_limits<double>::infinity());
        const TopLevel& level = *top;
        const tinybvh::BVH_Double* bvh = level.bvh();
        if (bvh != nullptr) {
            trimkit::in_parallel(n, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; i++) {
                    const double* d = directions.data() + (3 * i);
                    if (still(d)) {
                        continue;
                    }
                    tinybvh::RayEx ray = make_tinybvh_ray(origins.data() + (3 * i), d);
                    bvh->Intersect(ray);
                    if (ray.hit.t < BVH_DBL_FAR) {
                        instances[i] = level.direct != nullptr ? level.which[0]
                                                               : level.which[static_cast<std::size_t>(ray.hit.inst)];
                        faces[i] = static_cast<std::int64_t>(ray.hit.prim);
                        distances[i] = ray.hit.t * length(d);
                    }
                }
            });
        }
        return {trimkit::to_numpy(std::move(instances), {n}), trimkit::to_numpy(std::move(faces), {n}),
                trimkit::to_numpy(std::move(distances), {n})};
    }

    trimkit::Out<bool> intersect_any(const PointsArg& origins_argument, const PointsArg& directions_argument,
                                     const Transforms& matrices_argument, const Numbers& index_argument) const {
        const Points origins = trimkit::read(origins_argument, "origins");
        const Points directions = trimkit::read(directions_argument, "directions");
        check_rays(origins, directions);
        const std::shared_ptr<const TopLevel> top = top_level(matrices_argument, index_argument);
        const std::size_t n = origins.shape(0);
        std::vector<std::uint8_t> hits(n, 0);
        const tinybvh::BVH_Double* bvh = top->bvh();
        if (bvh != nullptr) {
            trimkit::in_parallel(n, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; i++) {
                    const double* d = directions.data() + (3 * i);
                    if (!still(d)) {
                        const tinybvh::RayEx ray = make_tinybvh_ray(origins.data() + (3 * i), d);
                        hits[i] = bvh->IsOccluded(ray) ? 1 : 0;
                    }
                }
            });
        }
        return trimkit::to_numpy<bool>(std::move(hits), {n});
    }

    All intersect_all(const PointsArg& origins_argument, const PointsArg& directions_argument,
                      const Transforms& matrices_argument, const Numbers& index_argument) const {
        const Points origins = trimkit::read(origins_argument, "origins");
        const Points directions = trimkit::read(directions_argument, "directions");
        check_rays(origins, directions);
        const std::shared_ptr<const TopLevel> top = top_level(matrices_argument, index_argument);
        const TopLevel& level = *top;
        const std::size_t n = origins.shape(0);
        struct Hit {
            std::int64_t ray;
            std::int64_t instance;
            std::int64_t face;
            double t;
        };
        std::vector<Hit> found;
        if (level.direct != nullptr || level.tlas) {
            found = trimkit::gather_in_parallel<Hit>(n, [&](std::size_t begin, std::size_t end,
                                                            std::vector<Hit>& hits) {
                const double* o = origins.data();
                const double* d = directions.data();
                Stack top_stack = make_stack();
                Stack bottom = make_stack();
                for (std::size_t i = begin; i < end; i++) {
                    if (still(d + (3 * i))) {
                        continue;
                    }
                    const std::size_t before = hits.size();
                    const Ray world = make_ray(Vec(o[3 * i], o[(3 * i) + 1], o[(3 * i) + 2]),
                                               Vec(d[3 * i], d[(3 * i) + 1], d[(3 * i) + 2]));
                    const auto ray = static_cast<std::int64_t>(i);
                    if (level.direct != nullptr) {
                        level.direct->every_hit(world, bottom, [&](std::uint64_t face, double t) {
                            hits.push_back({ray, level.which[0], static_cast<std::int64_t>(face), t});
                        });
                    } else {
                        every_instance_hit(
                            level, world, top_stack, bottom, [&](std::size_t instance, std::uint64_t face, double t) {
                                hits.push_back({ray, level.which[instance], static_cast<std::int64_t>(face), t});
                            });
                    }
                    // nearest first, ties in a fixed order, then distances
                    // in the scene's units rather than the direction's
                    std::sort(hits.begin() + static_cast<std::ptrdiff_t>(before), hits.end(),
                              [](const Hit& a, const Hit& b) {
                                  return std::tie(a.t, a.instance, a.face) < std::tie(b.t, b.instance, b.face);
                              });
                    const double scale = length(d + (3 * i));
                    for (std::size_t h = before; h < hits.size(); h++) {
                        hits[h].t *= scale;
                    }
                }
            });
        }
        // the hits as four columns, each written once
        const std::size_t total = found.size();
        std::vector<trimkit::Unset<std::int64_t>> rays(total);
        std::vector<trimkit::Unset<std::int64_t>> instances(total);
        std::vector<trimkit::Unset<std::int64_t>> faces(total);
        std::vector<trimkit::Unset<double>> distances(total);
        for (std::size_t h = 0; h < total; h++) {
            trimkit::elements(rays)[h] = found[h].ray;
            trimkit::elements(instances)[h] = found[h].instance;
            trimkit::elements(faces)[h] = found[h].face;
            trimkit::elements(distances)[h] = found[h].t;
        }
        return {trimkit::to_numpy<std::int64_t>(std::move(rays), {total}),
                trimkit::to_numpy<std::int64_t>(std::move(instances), {total}),
                trimkit::to_numpy<std::int64_t>(std::move(faces), {total}),
                trimkit::to_numpy<double>(std::move(distances), {total})};
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
        auto top = std::make_shared<TopLevel>();
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
        // one identity instance traces its mesh directly: the same answers as
        // through a top level, without the extra level per ray (19% of a query)
        if (top->instances.size() == 1 && placement.all_identity) {
            top->direct = meshes_[static_cast<std::size_t>(top->instances[0].blasIdx)].get();
        } else if (!top->instances.empty()) {
            const nb::gil_scoped_release release;
            top->tlas = std::make_unique<tinybvh::BVH_Double>();
            // tinybvh takes the meshes' array non-const, though it only reads it
            tinybvh::BVH_Double** meshes =
                const_cast<tinybvh::BVH_Double**>(blas_.data());  // NOLINT(cppcoreguidelines-pro-type-const-cast)
            top->tlas->Build(top->instances.data(), top->instances.size(), meshes, blas_.size());
            pad(*top->tlas, placement.extent);
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

    // every hit of a world-space ray on every instance, through the top level
    template <typename Emit>
    void every_instance_hit(const TopLevel& level, const Ray& world, Stack& top, Stack& bottom,
                            const Emit& emit) const {
        top.push_back(0);
        while (!top.empty()) {
            const auto& node = level.tlas->bvhNode[pop(top)];
            if (!slab(node, world)) {
                continue;
            }
            if (!node.isLeaf()) {
                top.push_back(node.leftFirst);
                top.push_back(node.leftFirst + 1);
                continue;
            }
            for (std::uint64_t j = node.leftFirst; j < node.leftFirst + node.triCount; j++) {
                const auto index = static_cast<std::size_t>(level.tlas->primIdx[j]);
                const tinybvh::BLASInstanceEx& instance = level.instances[index];
                // into the mesh's frame exactly as tinybvh's top level does it
                const Ray local = make_ray(tinybvh::tinybvh_transform_point(world.origin, instance.invTransform),
                                           tinybvh::tinybvh_transform_vector(world.direction, instance.invTransform));
                meshes_[static_cast<std::size_t>(instance.blasIdx)]->every_hit(
                    local, bottom, [&](std::uint64_t face, double t) { emit(index, face, t); });
            }
        }
    }

    std::vector<std::unique_ptr<RayMesh>> meshes_;
    // each mesh's BVH, or null without faces, as every top level indexes them
    std::vector<tinybvh::BVH_Double*> blas_;
    // the top level last built, swapped whole: a query keeps the one it read.
    // Every read and swap happens with the GIL held, which serializes them, so
    // the lock is nothing on GIL builds; on free-threaded ones it is CPython's
    // own PyMutex, held for one shared_ptr copy
    mutable nb::ft_mutex last_lock_;
    mutable std::shared_ptr<const TopLevel> last_;
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
        std::vector<std::int64_t> pairs = trimkit::gather_in_parallel<std::int64_t>(
            n, [&](std::size_t begin, std::size_t end, std::vector<std::int64_t>& rows) {
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
        return trimkit::to_numpy(std::move(pairs), {count, 2});
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
            bvh_ = std::make_unique<tinybvh::BVH_Double>();
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
        auto overlaps = [&](const Vec& a, const Vec& b) {
            for (std::int32_t axis = 0; axis < 3; axis++) {
                if (a[axis] > hi[axis] || b[axis] < lo[axis]) {
                    return false;
                }
            }
            return true;
        };
        stack.push_back(0);
        while (!stack.empty()) {
            const auto& node = bvh_->bvhNode[pop(stack)];
            if (!overlaps(node.aabbMin, node.aabbMax)) {
                continue;
            }
            if (!node.isLeaf()) {
                stack.push_back(node.leftFirst);
                stack.push_back(node.leftFirst + 1);
                continue;
            }
            for (std::uint64_t j = node.leftFirst; j < node.leftFirst + node.triCount; j++) {
                const auto box = static_cast<std::size_t>(bvh_->primIdx[j]);
                if (overlaps(boxes_[2 * box], boxes_[(2 * box) + 1])) {
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
    std::unique_ptr<tinybvh::BVH_Double> bvh_;
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
is. Instances are placed under a top-level BVH built from exactly the
arrays a query gives; the scene remembers the last one, so a query naming
the same instances again (the same bytes) costs only comparing them, and
moving anything costs that top level, never a mesh. Answers never depend
on what was asked before.

Parameters
----------
meshes : sequence of Mesh
  The meshes to place, each a (vertices, faces) tuple of (n, 3) float64
  vertices and (m, 3) int64 triangle indices into them. Copied once
  into tinybvh's padded float layout, which the scene owns, so a mesh
  can never change under the threads sharing the scene.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> scene = trimkit.RayScene([(triangle, faces)])
>>> len(scene)
1)")
        .def(nb::init<const trimkit::Meshes&>(), nb::arg("meshes"))
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
>>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> trimkit.RayScene([(triangle, faces)]).boxes().bounds.tolist()
[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])")
        .def("intersect_first", &RayScene::intersect_first, nb::arg("origins"), nb::arg("directions"),
             nb::arg("instance_matrices").none() = nb::none(), nb::arg("instance_index").none() = nb::none(),
             R"(The nearest triangle each ray hits, from either side.

Parameters
----------
origins : (r, 3) float64
  Where each ray starts.
directions : (r, 3) float64
  Each ray's direction, which need not be unit length.
instance_matrices : (k, 4, 4) float64, optional
  Each instance's affine transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.

Returns
-------
instances : (r,) int64
  Instance of each first hit, or -1 for a miss.
faces : (r,) int64
  Face of each first hit in its instance's mesh, or -1.
distances : (r,) float64
  Distance from the origin to the hit, or infinity.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> scene = trimkit.RayScene([(triangle, faces)])
>>> origins, directions = np.array([[0.2, 0.2, 1.0]]), np.array([[0.0, 0.0, -1.0]])
>>> instances, hit, distances = scene.intersect_first(origins, directions)
>>> instances.tolist(), hit.tolist(), distances.tolist()
([0], [0], [1.0]))")
        .def("intersect_all", &RayScene::intersect_all, nb::arg("origins"), nb::arg("directions"),
             nb::arg("instance_matrices").none() = nb::none(), nb::arg("instance_index").none() = nb::none(),
             R"(Every triangle each ray hits, from either side.

Parameters
----------
origins : (r, 3) float64
  Where each ray starts.
directions : (r, 3) float64
  Each ray's direction, which need not be unit length.
instance_matrices : (k, 4, 4) float64, optional
  Each instance's affine transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.

Returns
-------
rays : (h,) int64
  Ray of each hit, one row per hit, sorted by ray and then
  distance. A ray through a shared edge or vertex hits every triangle
  there.
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
>>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> scene = trimkit.RayScene([(triangle, faces)])
>>> origins, directions = np.array([[0.2, 0.2, 1.0]]), np.array([[0.0, 0.0, -1.0]])
>>> rays, instances, hit, distances = scene.intersect_all(origins, directions)
>>> rays.tolist(), distances.tolist()
([0], [1.0]))")
        .def("intersect_any", &RayScene::intersect_any, nb::arg("origins"), nb::arg("directions"),
             nb::arg("instance_matrices").none() = nb::none(), nb::arg("instance_index").none() = nb::none(),
             R"(Whether each ray hits anything, stopping at the first triangle found.

Parameters
----------
origins : (r, 3) float64
  Where each ray starts.
directions : (r, 3) float64
  Each ray's direction, which need not be unit length.
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
>>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> scene = trimkit.RayScene([(triangle, faces)])
>>> origins, directions = np.array([[0.2, 0.2, 1.0]]), np.array([[0.0, 0.0, -1.0]])
>>> scene.intersect_any(origins, directions).tolist()
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
>>> boxes = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> len(boxes)
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
>>> boxes = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> boxes.bounds.tolist()
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
>>> boxes = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> boxes.boxes().shape
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
>>> boxes = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> boxes.intersect(np.array([[[0.5, 0.5], [2.5, 2.5]]])).tolist()
[[0, 0], [0, 1]])");
}
