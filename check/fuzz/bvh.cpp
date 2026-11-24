// Property: for any triangles and rays, the BVH's first hit agrees with a
// brute-force test of every triangle in extended precision. A hit it reports
// really is on that triangle at that distance, and no triangle is hit clearly
// nearer, nor clearly missed, where "clearly" allows for rounding in the BVH.
// Both as tinybvh's own traversal answers, built either way (`BuildHQ` is
// what a scene of 100k faces or more uses), and as `trimkit.RayScene`
// answers (src/bvh.cpp: its meshes padded, placed under a top level, and
// every crossing found nearest first), whose distances are in world units
#include "../../src/trimkit.hpp"

#include <fuzzer/FuzzedDataProvider.h>
#include <tiny_bvh.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

// the reference is long double: 11 more bits than the BVH on x86, so its own
// rounding near vertices and edges is far below the tolerance it judges by
using R = long double;

struct Hit {
    R t = INFINITY;
    R margin = -INFINITY;  // smallest barycentric coordinate: < 0 is outside
    bool ill = false;      // a sliver, or the ray all but in its plane: t is ill-conditioned
};

using V = tinybvh::bvhdbl3;
struct L {
    R x, y, z;
};

L widen(V a) { return {a.x, a.y, a.z}; }
L sub(L a, L b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
L cross(L a, L b) { return {(a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z), (a.x * b.y) - (a.y * b.x)}; }
R dot(L a, L b) { return (a.x * b.x) + (a.y * b.y) + (a.z * b.z); }

// Möller-Trumbore from either side
Hit test(V o_, V d_, V a_, V b_, V c_) {
    const L o = widen(o_);
    const L d = widen(d_);
    const L a = widen(a_);
    const L e1 = sub(widen(b_), a);
    const L e2 = sub(widen(c_), a);
    const L p = cross(d, e2);
    const R det = dot(p, e1);
    Hit h;
    // the barycentrics and t are ratios of terms the size of the whole
    // problem (the origin's offset and the edges) to this determinant, so
    // they lose as many digits as it is small next to that scale: for a ray
    // near the plane or a sliver of a triangle, this can't judge to 1e-9
    const L s = sub(o, a);
    const R scale = std::max({dot(e1, e1), dot(e2, e2), dot(s, s)});
    if (!(std::abs(det) > 1e-6 * std::sqrt(dot(d, d)) * scale)) {
        h.ill = true;
        return h;
    }
    const R u = dot(s, p) / det;
    const L q = cross(s, e1);
    const R v = dot(d, q) / det;
    h.t = dot(q, e2) / det;
    h.margin = std::min({u, v, 1.0 - u - v});
    return h;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider in(data, size);
    const auto m = in.ConsumeIntegralInRange<std::uint64_t>(1, 64);
    std::vector<V> vertices(3 * m);
    std::vector<std::uint64_t> faces(3 * m);
    for (std::uint64_t i = 0; i < 3 * m; i++) {
        vertices[i] = V(in.ConsumeFloatingPointInRange(-10.0, 10.0), in.ConsumeFloatingPointInRange(-10.0, 10.0),
                        in.ConsumeFloatingPointInRange(-10.0, 10.0));
        faces[i] = i;
    }
    tinybvh::BVH_Double bvh;
    if (in.ConsumeBool()) {
        bvh.BuildHQ(vertices.data(), faces.data(), m);
    } else {
        bvh.Build(vertices.data(), faces.data(), m);
    }
    constexpr double EPSILON = 1e-9;  // relative to coordinates of order 10
    // the same triangles as a scene of one mesh, placed as itself
    std::vector<double> flat(9 * m);
    for (std::uint64_t i = 0; i < 3 * m; i++) {
        flat[3 * i] = vertices[i].x;
        flat[(3 * i) + 1] = vertices[i].y;
        flat[(3 * i) + 2] = vertices[i].z;
    }
    std::vector<trimkit::RayMeshInput> meshes(1);
    meshes[0].vertices = flat.data();
    meshes[0].vertex_count = 3 * m;
    meshes[0].faces = faces;
    const trimkit::RayMeshes scene(std::move(meshes), false);
    trimkit::Instances itself;
    itself.count = 1;
    const std::shared_ptr<const trimkit::RayTopLevel> top = scene.place(itself);
    const auto rays = in.ConsumeIntegralInRange<int>(1, 16);
    for (int r = 0; r < rays; r++) {
        const V o(in.ConsumeFloatingPointInRange(-20.0, 20.0), in.ConsumeFloatingPointInRange(-20.0, 20.0),
                  in.ConsumeFloatingPointInRange(-20.0, 20.0));
        const V d(in.ConsumeFloatingPointInRange(-1.0, 1.0), in.ConsumeFloatingPointInRange(-1.0, 1.0),
                  in.ConsumeFloatingPointInRange(-1.0, 1.0));
        if (std::sqrt(dot(widen(d), widen(d))) < 1e-3) {
            continue;
        }
        // rounding is on the scale of the coordinates (up to about 40 apart),
        // so distances are compared in world units: in t that is this much
        const R tolerance = EPSILON * 40.0 / std::sqrt(dot(widen(d), widen(d)));
        tinybvh::RayEx ray(o, d);
        bvh.Intersect(ray);
        const bool hit = ray.hit.t < BVH_DBL_FAR;
        if (hit) {
            // the reported triangle is hit at the reported distance. Where
            // the ray is all but edge-on to it, no floating point test can
            // say where, this one included, so those are not judged
            const std::uint64_t p = ray.hit.prim;
            if (p >= m) {
                __builtin_trap();
            }
            const Hit h = test(o, d, vertices[3 * p], vertices[(3 * p) + 1], vertices[(3 * p) + 2]);
            if (!h.ill && (h.margin < -EPSILON || !(std::abs(h.t - ray.hit.t) <= tolerance))) {
                __builtin_trap();
            }
        }
        // the scene's nearest crossing, in world units, as t along `d`
        const double origin[3] = {o.x, o.y, o.z};
        const double direction[3] = {d.x, d.y, d.z};
        trimkit::RayCrossings crossings;
        scene.intersect(*top, {origin, 1, false, false}, {direction, 1, false, false}, 1, 1, trimkit::parallel::Stop{},
                        &crossings);
        const bool scene_hit = !crossings.rays.empty();
        const R scene_t =
            scene_hit ? trimkit::elements(crossings.distances)[0] / std::sqrt(dot(widen(d), widen(d))) : INFINITY;
        if (scene_hit) {
            const std::int64_t p = trimkit::elements(crossings.faces)[0];
            if (crossings.rays.size() != 1 || p < 0 || static_cast<std::uint64_t>(p) >= m) {
                __builtin_trap();
            }
            const auto f = static_cast<std::uint64_t>(p);
            const Hit h = test(o, d, vertices[3 * f], vertices[(3 * f) + 1], vertices[(3 * f) + 2]);
            if (!h.ill && (h.margin < -EPSILON || !(std::abs(h.t - scene_t) <= tolerance))) {
                __builtin_trap();
            }
        }
        for (std::uint64_t f = 0; f < m; f++) {
            const Hit h = test(o, d, vertices[3 * f], vertices[(3 * f) + 1], vertices[(3 * f) + 2]);
            // a clear hit ahead of the origin is never missed or beaten
            const bool clear = h.margin > EPSILON && h.t > tolerance;
            if (clear && (!hit || ray.hit.t > h.t + tolerance)) {
                __builtin_trap();
            }
            if (clear && (!scene_hit || scene_t > h.t + tolerance)) {
                __builtin_trap();
            }
        }
    }
    return 0;
}
