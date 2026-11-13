// Property: for any triangles and rays, the BVH's first hit agrees with a
// brute-force test of every triangle in extended precision. A hit it reports
// really is on that triangle at that distance, and no triangle is hit clearly
// nearer, nor clearly missed, where "clearly" allows for rounding in the BVH.
#include <fuzzer/FuzzedDataProvider.h>
#include <tiny_bvh.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
        for (std::uint64_t f = 0; f < m; f++) {
            const Hit h = test(o, d, vertices[3 * f], vertices[(3 * f) + 1], vertices[(3 * f) + 2]);
            // a clear hit ahead of the origin is never missed or beaten
            const bool clear = h.margin > EPSILON && h.t > tolerance;
            if (clear && (!hit || ray.hit.t > h.t + tolerance)) {
                __builtin_trap();
            }
        }
    }
    return 0;
}
