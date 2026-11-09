// Polygon triangulation with mapbox/earcut.hpp.
//
// Rings arrive the way shapely holds them, an exterior and a sequence of
// interiors. earcut reads each numpy buffer in place through the view types
// below, so the only copies are the stacked vertices returned beside the
// faces, and its uint64 indices become the int64 output without a copy.

#include "common.hpp"

#include <earcut.hpp>

#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

// a 2D point as a view into an (n, 2) float64 buffer
struct Point {
    const double* xy;
};

}  // namespace

// how earcut reads a coordinate out of our point view
namespace mapbox::util {
template <>
struct nth<0, Point> {
    static double get(const Point& p) { return p.xy[0]; }
};
template <>
struct nth<1, Point> {
    static double get(const Point& p) { return p.xy[1]; }
};
}  // namespace mapbox::util

namespace {

struct Ring {
    using value_type = Point;
    const double* xy;
    std::size_t count;
    std::size_t size() const { return count; }
    Point operator[](std::size_t i) const { return {xy + 2 * i}; }
};

struct Polygon {
    using value_type = Ring;
    std::vector<Ring> rings;
    std::size_t size() const { return rings.size(); }
    bool empty() const { return rings.empty(); }
    const Ring& operator[](std::size_t i) const { return rings[i]; }
};

using Ring2 = trimkit::In<double, -1, 2>;
using RingArg = trimkit::Array<double, -1, 2>;
using Interiors = nb::typed<nb::sequence, RingArg>;

std::tuple<trimkit::Out<double>, trimkit::Out<std::int64_t>> triangulate(const RingArg& exterior_argument,
                                                                         const Interiors& interiors) {
    const Ring2 exterior = trimkit::read(exterior_argument, "exterior");
    // every interior viewed in place, held so the views below stay valid
    std::vector<Ring2> holes;
    for (const nb::handle item : interiors) {
        holes.push_back(trimkit::read<double, -1, 2>(item, "interiors[" + std::to_string(holes.size()) + "]"));
    }
    std::vector<double> vertices;
    std::vector<std::uint64_t> faces;
    {
        const nb::gil_scoped_release release;
        // the first ring is the shell and the rest are holes, indexed as stacked
        Polygon polygon;
        polygon.rings.push_back({exterior.data(), exterior.shape(0)});
        std::size_t count = exterior.shape(0);
        for (const Ring2& hole : holes) {
            polygon.rings.push_back({hole.data(), hole.shape(0)});
            count += hole.shape(0);
        }
        vertices.reserve(2 * count);
        for (const Ring& ring : polygon.rings) {
            vertices.insert(vertices.end(), ring.xy, ring.xy + (2 * ring.count));
        }
        // a polygon with n vertices and h holes has n + 2h - 2 triangles:
        // reserve a bound on that, earcut's reset keeps the capacity, and the
        // result moves to numpy, so the output is one allocation filled once
        mapbox::detail::Earcut<std::uint64_t> earcut;
        earcut.indices.reserve(3 * (count + (2 * holes.size())));
        earcut(polygon);
        faces = std::move(earcut.indices);
    }
    const std::size_t vertex_count = vertices.size() / 2;
    const std::size_t face_count = faces.size() / 3;
    return {trimkit::to_numpy(std::move(vertices), {vertex_count, 2}),
            trimkit::to_numpy<std::int64_t>(std::move(faces), {face_count, 3})};
}

}  // namespace

void bind_earcut(nb::module_& m) {
    m.def("triangulate", &triangulate, nb::arg("exterior"), nb::arg("interiors") = nb::tuple(),
          R"(Triangulate a polygon with holes.

Parameters
----------
exterior : (n, 2) float64
  The polygon's outer ring, as shapely's `exterior.coords`. A repeated
  closing vertex is fine.
interiors : sequence of (n_i, 2) float64
  Each hole's ring, as shapely's `interiors[i].coords`.

Returns
-------
vertices : (N, 2) float64
  Every ring stacked in order, the exterior then each interior.
faces : (t, 3) int64
  Triangle indices into `vertices`.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> exterior = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
>>> vertices, faces = trimkit.triangulate(exterior)
>>> faces.tolist()
[[2, 3, 0], [2, 0, 1]])");
}
