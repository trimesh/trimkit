// Properties of `parse` on any bytes:
// - every kind is an SvgKind, a point index is -1 exactly where its kind has
//   no such point, and the padding and unused kind_data are zero
// - vertices are used in the order they were added, each first by the
//   segment that added it: every point is either the vertex the pen is on,
//   a closing line's return to its subpath's first vertex, or the next new
//   vertex, and no vertex goes unused
// - every coordinate is within the bound, and the output is at most the 43
//   bytes per byte of input `parse` documents
// - every arc and conic is a piece of at most MAX_ARC that `format` writes
//   as one `A` command: an arc's three points on one circle, the middle
//   halfway along it, its center finite and within the bound; a conic's
//   weight between 1/2 and 1
// - `format` writes it all at 17 digits, and that parses completely. Every
//   segment's end comes back bit for bit, in order, and without arcs or
//   conics everything does. An arc's halfway point and center, and a
//   conic's control and weight, are recomputed from the radii written, so
//   an arc whose radius is near the coordinates' rounding (radius 5 at
//   y = 2e16, where doubles are 4 apart) can come back as a line or split
//   again; the tests check that ordinary coordinates round trip exactly,
//   with those within rounding
#include "../../src/svgpath.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    svgpath::SvgPath parsed;
    static_cast<void>(svgpath::parse(reinterpret_cast<const char*>(data), size, &parsed));
    const std::size_t bytes =
        (parsed.vertices.size() * sizeof(double)) + (parsed.segments.size() * sizeof(svgpath::SvgSegment));
    if (bytes > (43 * size) + 16) {
        __builtin_trap();  // past the documented bound
    }
    for (const double value : parsed.vertices) {
        if (!svgpath::within(value)) {
            __builtin_trap();
        }
    }
    std::int64_t next_vertex = 0;  // the vertex the next new point must be
    std::int64_t pen = -1;         // the vertex the last segment ended on
    std::int64_t start = -1;       // the first vertex of the current subpath
    bool arcs = false;
    for (const svgpath::SvgSegment& segment : parsed.segments) {
        const std::uint8_t kind = segment.kind;
        if (kind > svgpath::CONIC) {
            __builtin_trap();
        }
        for (const std::uint8_t byte : segment.padding) {
            if (byte != 0) {
                __builtin_trap();
            }
        }
        // a new subpath starts on the next new vertex, anything else on the pen
        if (segment.points[0] == next_vertex) {
            start = next_vertex;
            next_vertex++;
        } else if (segment.points[0] != pen) {
            __builtin_trap();
        }
        const bool uses[3] = {kind != svgpath::LINE, kind == svgpath::CUBIC, true};
        for (int slot = 1; slot < 4; slot++) {
            const std::int64_t vertex = segment.points[slot];
            if (!uses[slot - 1]) {
                if (vertex != -1) {
                    __builtin_trap();
                }
            } else if (vertex == next_vertex) {
                next_vertex++;
            } else if (!(slot == 3 && kind == svgpath::LINE && vertex == start)) {
                __builtin_trap();  // only a close returns to an old vertex
            }
        }
        pen = segment.points[3];
        if (kind != svgpath::ARC && kind != svgpath::CONIC) {
            if (segment.kind_data[0] != 0.0 || segment.kind_data[1] != 0.0) {
                __builtin_trap();
            }
            continue;
        }
        arcs = true;
        double coordinates[6];
        double largest = 0.0;
        const int slots[3] = {0, 1, 3};
        for (int which = 0; which < 3; which++) {
            for (int axis = 0; axis < 2; axis++) {
                coordinates[(2 * which) + axis] =
                    parsed.vertices[(2 * static_cast<std::size_t>(segment.points[slots[which]])) + axis];
                largest = std::fmax(largest, std::abs(coordinates[(2 * which) + axis]));
            }
        }
        if (kind == svgpath::ARC) {
            if (!svgpath::within(segment.kind_data[0]) || !svgpath::within(segment.kind_data[1])) {
                __builtin_trap();  // its center
            }
            // the middle is as far from either end, to within rounding of
            // the coordinates, which far from the origin is coarse
            const double before = std::hypot(coordinates[2] - coordinates[0], coordinates[3] - coordinates[1]);
            const double after = std::hypot(coordinates[4] - coordinates[2], coordinates[5] - coordinates[3]);
            if (std::abs(before - after) > (1e-6 * (before + after)) + (1e-14 * largest)) {
                __builtin_trap();
            }
        } else if (!(segment.kind_data[0] >= svgpath::MAX_ARC_HALF_COSINE && segment.kind_data[0] < 1.0) ||
                   segment.kind_data[1] != 0.0) {
            __builtin_trap();  // its weight
        }
        svgpath::SvgArcCommands commands;
        if (svgpath::arc_commands(kind, &coordinates[0], &coordinates[2], &coordinates[4], segment.kind_data[0],
                                  &commands) != nullptr) {
            __builtin_trap();  // a piece `format` would refuse
        }
        // and one it writes as a single `A`, unless its radius is so near the
        // coordinates' rounding that its sweep, recomputed, can pass MAX_ARC
        if (commands.count != 1 && std::fmin(commands.radius_x, commands.radius_y) > 1e-8 * largest) {
            __builtin_trap();
        }
    }
    if (static_cast<std::size_t>(next_vertex) != parsed.vertices.size() / 2) {
        __builtin_trap();  // a vertex no segment uses
    }

    // the round trip
    std::string written;
    std::size_t refused = 0;
    if (svgpath::format(&parsed, 17, &written, &refused) != nullptr) {
        __builtin_trap();  // anything `parse` produced formats: by construction, kept as a tripwire
    }
    svgpath::SvgPath again;
    if (svgpath::parse(written.data(), written.size(), &again) != written.size()) {
        __builtin_trap();
    }
    // each segment's end, in order: the original's among the new one's
    const auto ends = [](const svgpath::SvgPath* path) {
        std::vector<double> coordinates;
        for (const svgpath::SvgSegment& segment : path->segments) {
            const std::size_t vertex = static_cast<std::size_t>(segment.points[3]);
            coordinates.push_back(path->vertices[2 * vertex]);
            coordinates.push_back(path->vertices[(2 * vertex) + 1]);
        }
        return coordinates;
    };
    const std::vector<double> before = ends(&parsed);
    const std::vector<double> after = ends(&again);
    std::size_t found = 0;
    for (std::size_t index = 0; index < after.size() && found < before.size(); index += 2) {
        if (std::memcmp(&after[index], &before[found], 2 * sizeof(double)) == 0) {
            found += 2;
        }
    }
    if (found != before.size()) {
        __builtin_trap();
    }
    const bool same_segments =
        again.segments.size() == parsed.segments.size() &&
        (parsed.segments.empty() || std::memcmp(again.segments.data(), parsed.segments.data(),
                                                parsed.segments.size() * sizeof(svgpath::SvgSegment)) == 0);
    if (!arcs && (!same_segments || again.vertices != parsed.vertices)) {
        __builtin_trap();
    }
    return 0;
}
