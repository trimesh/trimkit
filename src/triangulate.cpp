// Polygon triangulation: mapbox/earcut.hpp's ear clipping, or CDT's
// constrained Delaunay triangulation. Python-free (trimkit.hpp).
//
// Both engines go through one front end: the rings stacked into the
// vertices returned beside the faces, and exactly equal vertices (shapely's
// repeated closing vertex, a hole touching the shell) found once, so
// whichever engine ran, faces index the first of them. earcut reads the
// rings in place through the view types below. CDT takes the distinct
// vertices, by row, and every ring's edges as constraints.

#include "trimkit.hpp"

#include <CDT.h>
#include <earcut.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
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

}  // namespace

void trimkit::triangulate_polygon(const std::vector<PolygonRing>& rings, TriangulateEngine engine,
                                  std::vector<double>* vertices_out, std::vector<std::uint64_t>* faces_out) {
    // rows are uint32 in the table below, which has up to four slots a row
    // that a size_t must count, and CDT numbers its vertices in 32 bits past
    // three of its own: under 2**31 in all (2**30 on 32-bit), counted ring by
    // ring so a 32-bit size_t can't wrap
    constexpr std::size_t MOST =
        std::min<std::uint64_t>(std::uint64_t{1} << 31U, (std::numeric_limits<std::size_t>::max() / 4) + 1);
    std::size_t count = 0;
    for (const PolygonRing& ring : rings) {
        if (ring.count >= MOST - count) {
            refuse("this polygon is too large to index with uint32");
        }
        count += ring.count;
    }
    std::vector<double>& vertices = *vertices_out;
    std::vector<std::uint64_t>& faces = *faces_out;
    vertices.clear();
    faces.clear();
    // the first ring is the shell and the rest are holes, indexed as stacked
    Polygon polygon;
    polygon.rings.reserve(rings.size());
    for (const PolygonRing& ring : rings) {
        polygon.rings.push_back({ring.xy, ring.count});
    }
    const std::size_t holes = rings.empty() ? 0 : rings.size() - 1;
    vertices.reserve(2 * count);
    for (const Ring& ring : polygon.rings) {
        vertices.insert(vertices.end(), ring.xy, ring.xy + (2 * ring.count));
    }
    // first[row]: the first row exactly equal to `row`, found through a
    // flat table of rows (row + 1, 0 when empty) at most half full,
    // probed from a hash of the coordinates' bits (-0 as 0, which it
    // equals). 7 ns a vertex at 10k vertices, 13 ns at 1M as the table
    // outgrows the cache: under 1% of earcut's time on the same rings.
    // earcut drops a repeated neighbour from its own list but names each
    // row as given, so without this a hole touching the shell leaves two
    // rows at one point, and faces not joined there. Not CDT's
    // FindDuplicates, whose hash(x) ^ hash(y) is 0 for every point on y = x
    std::vector<std::uint32_t> first(count);
    {
        std::size_t slots = 1;
        while (slots <= 2 * count) {
            slots <<= 1;
        }
        std::vector<std::uint32_t> table(slots, 0);
        // splitmix64's finalizer: every input bit moves every output bit. Its
        // multiplies wrap by design, which the overflow builtins say (our
        // code is checked for unsigned overflow, everywhere else a bug)
        const auto mix = [](std::uint64_t z) {
            std::uint64_t product = 0;
            __builtin_mul_overflow(z ^ (z >> 30U), std::uint64_t{0xbf58476d1ce4e5b9ULL}, &product);
            __builtin_mul_overflow(product ^ (product >> 27U), std::uint64_t{0x94d049bb133111ebULL}, &product);
            return product ^ (product >> 31U);
        };
        const auto bits = [](double value) {
            const double positive = value == 0.0 ? 0.0 : value;
            std::uint64_t word = 0;
            std::memcpy(&word, &positive, sizeof(word));
            return word;
        };
        for (std::size_t row = 0; row < count; row++) {
            const double x = vertices[2 * row];
            const double y = vertices[(2 * row) + 1];
            for (std::size_t slot = mix(bits(x) ^ mix(bits(y))) & (slots - 1);; slot = (slot + 1) & (slots - 1)) {
                const std::uint32_t held = table[slot];
                if (held == 0) {
                    table[slot] = static_cast<std::uint32_t>(row + 1);
                    first[row] = static_cast<std::uint32_t>(row);
                    break;
                }
                const std::size_t at = 2 * (std::size_t{held} - 1);
                if (vertices[at] == x && vertices[at + 1] == y) {
                    first[row] = held - 1;
                    break;
                }
            }
        }
    }
    if (engine == TriangulateEngine::CDT) {
        // CDT numbers the distinct vertices in row order: rows[v] is
        // vertex v's row, and vertex[row] the vertex of `row` and of
        // every row equal to it
        std::vector<std::uint32_t> rows;
        std::vector<CDT::VertInd> vertex(count);
        for (std::size_t row = 0; row < count; row++) {
            if (first[row] == row) {
                vertex[row] = static_cast<CDT::VertInd>(rows.size());
                rows.push_back(static_cast<std::uint32_t>(row));
            } else {
                vertex[row] = vertex[first[row]];
            }
        }
        if (rows.size() >= 3) {
            // every ring's edges, closing it: CDT skips one whose ends are
            // the same vertex, so a repeated closing vertex needs nothing
            std::vector<CDT::Edge> edges;
            edges.reserve(count);
            std::size_t begin = 0;
            for (const Ring& ring : polygon.rings) {
                for (std::size_t i = 0; i < ring.count; i++) {
                    edges.emplace_back(vertex[begin + i], vertex[begin + ((i + 1) % ring.count)]);
                }
                begin += ring.count;
            }
            // crossing edges are refused rather than split: a split adds
            // a vertex, and `vertices` is exactly the rings
            CDT::Triangulation<double> cdt(CDT::VertexInsertionOrder::Auto,
                                           CDT::IntersectingConstraintEdges::NotAllowed, 0.0);
            try {
                // all at once, so CDT inserts them in its KD-tree order
                cdt.insertVertices(
                    rows.begin(), rows.end(), [&](std::uint32_t row) { return vertices[2 * std::size_t{row}]; },
                    [&](std::uint32_t row) { return vertices[(2 * std::size_t{row}) + 1]; });
                cdt.insertEdges(edges);
                // what is outside the shell or inside a hole, by how many
                // ring edges away from outside it is: winding doesn't matter
                cdt.eraseOuterTrianglesAndHoles();
            } catch (const CDT::IntersectingConstraintsError& error) {
                // named as rows of `vertices`, not as CDT numbers them
                const auto named = [&](const CDT::Edge& edge) {
                    return "(" + std::to_string(rows[edge.v1()]) + ", " + std::to_string(rows[edge.v2()]) + ")";
                };
                refuse("ring edges cross: the edge between vertices " + named(error.e1()) + " crosses " +
                       named(error.e2()));
            } catch (const CDT::Error& error) {
                // description(), as what() adds CDT's source file and line
                refuse(error.description());
            }
            faces.reserve(3 * cdt.triangles.size());
            for (const CDT::Triangle& triangle : cdt.triangles) {
                std::transform(triangle.vertices.begin(), triangle.vertices.end(), std::back_inserter(faces),
                               [&](CDT::VertInd v) { return std::uint64_t{rows[v]}; });
            }
        }
    } else {
        // a polygon with n vertices and h holes has n + 2h - 2 triangles:
        // reserve a bound on that, earcut's reset keeps the capacity, and the
        // result moves to numpy, so the output is one allocation filled once
        mapbox::detail::Earcut<std::uint64_t> earcut;
        earcut.indices.reserve(3 * (count + (2 * holes)));
        earcut(polygon);
        faces = std::move(earcut.indices);
        std::transform(faces.begin(), faces.end(), faces.begin(), [&](std::uint64_t index) { return first[index]; });
    }
}
