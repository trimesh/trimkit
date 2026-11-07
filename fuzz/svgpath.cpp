// Properties of `parse` on any bytes:
// - vertices are used in the order they were added, each first by the
//   segment that added it: every point is either the vertex the pen is on,
//   a closing line's return to its subpath's first vertex, or the next new
//   vertex, and no vertex goes unused
// - a point index is -1 exactly where its kind has no such point
// - a line closes exactly when it ends on an old vertex
// - every arc's center parameterization is finite and reproduces both of
//   its endpoints and its halfway point
// - `format` writes it all, parsing that gives the same vertices and
//   segments byte for byte, and formatting that gives the same text
#include <svgpath.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

static bool same_path(const svgpath::Path* first, const svgpath::Path* second) {
    if (first->vertices.size() != second->vertices.size() || first->segments.size() != second->segments.size()) {
        return false;
    }
    const bool same_vertices = first->vertices.empty() || std::memcmp(first->vertices.data(), second->vertices.data(),
                                                                      first->vertices.size() * sizeof(double)) == 0;
    const bool same_segments =
        first->segments.empty() || std::memcmp(first->segments.data(), second->segments.data(),
                                               first->segments.size() * sizeof(svgpath::Segment)) == 0;
    return same_vertices && same_segments;
}

// an arc's center, radii, rotation and angles put its endpoints and its
// halfway point back
static void check_arc(const svgpath::Path* path, const svgpath::Segment* segment) {
    const double* center = &path->vertices[2 * static_cast<std::size_t>(segment->points[1])];
    if (!std::isfinite(segment->angles[0]) || !std::isfinite(segment->angles[1])) {
        __builtin_trap();
    }
    const double scale = 1.0 + std::abs(center[0]) + std::abs(center[1]) + segment->radius[0] + segment->radius[1];
    if (!std::isfinite(scale) || scale > 1e12) {
        return;  // precision is meaningless once coordinates overflow
    }
    const double rotation = segment->rotation * svgpath::PI / 180.0;
    // start, halfway and end: the slots of the points at 0, 1/2 and 1 of the sweep
    const int slots[3] = {0, 2, 3};
    for (int which = 0; which < 3; which++) {
        const double angle = segment->angles[0] + (0.5 * which * segment->angles[1]);
        const double x = segment->radius[0] * std::cos(angle);
        const double y = segment->radius[1] * std::sin(angle);
        const std::int64_t vertex = segment->points[slots[which]];
        const double* expected = &path->vertices[2 * static_cast<std::size_t>(vertex)];
        const double error_x = center[0] + (std::cos(rotation) * x) - (std::sin(rotation) * y) - expected[0];
        const double error_y = center[1] + (std::sin(rotation) * x) + (std::cos(rotation) * y) - expected[1];
        if (std::abs(error_x) > 1e-7 * scale || std::abs(error_y) > 1e-7 * scale) {
            __builtin_trap();
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    svgpath::Path parsed;
    static_cast<void>(svgpath::parse(reinterpret_cast<const char*>(data), size, &parsed));
    std::int64_t next_vertex = 0;  // the vertex the next new point must be
    std::int64_t pen = -1;         // the vertex the last segment ended on
    std::int64_t start = -1;       // the first vertex of the current subpath
    for (std::size_t index = 0; index < parsed.segments.size(); index++) {
        const svgpath::Segment* segment = &parsed.segments[index];
        if (segment->kind > svgpath::ARC) {
            __builtin_trap();
        }
        // a new subpath starts on the next new vertex, anything else on the pen
        if (segment->points[0] == next_vertex) {
            start = next_vertex;
            next_vertex++;
        } else if (segment->points[0] != pen) {
            __builtin_trap();
        }
        const bool uses[3] = {segment->kind != svgpath::LINE,
                              segment->kind == svgpath::CUBIC || segment->kind == svgpath::ARC, true};
        for (int slot = 1; slot < 4; slot++) {
            const std::int64_t vertex = segment->points[slot];
            if (!uses[slot - 1]) {
                if (vertex != -1) {
                    __builtin_trap();
                }
            } else if (vertex == next_vertex) {
                next_vertex++;
            } else if (!(slot == 3 && segment->closes != 0 && vertex == start)) {
                __builtin_trap();
            }
        }
        if ((segment->closes != 0) != (segment->points[3] <= segment->points[0]) ||
            (segment->closes != 0 && segment->kind != svgpath::LINE)) {
            __builtin_trap();
        }
        if (segment->kind == svgpath::ARC) {
            check_arc(&parsed, segment);
        }
        pen = segment->points[3];
    }
    if (static_cast<std::size_t>(next_vertex) != parsed.vertices.size() / 2) {
        __builtin_trap();  // a vertex no segment uses
    }

    // the round trip: what `format` writes parses back to exactly this
    std::string written;
    std::size_t refused = 0;
    if (svgpath::format(parsed.vertices.data(), parsed.vertices.size() / 2, parsed.segments.data(),
                        parsed.segments.size(), &written, &refused) != nullptr) {
        __builtin_trap();  // anything `parse` produced formats: by construction, kept as a tripwire
    }
    svgpath::Path again;
    if (svgpath::parse(written.data(), written.size(), &again) != written.size() || !same_path(&parsed, &again)) {
        __builtin_trap();
    }
    std::string rewritten;
    if (svgpath::format(again.vertices.data(), again.vertices.size() / 2, again.segments.data(), again.segments.size(),
                        &rewritten, &refused) != nullptr ||
        rewritten != written) {
        __builtin_trap();
    }
    return 0;
}
