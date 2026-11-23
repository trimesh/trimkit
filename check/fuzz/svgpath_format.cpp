// `format` on raw, hand-built segments rather than on what `parse` produced:
// two bytes pick how many vertices follow and how many digits to write, then
// the vertices as doubles, any bit pattern, then segments of 21 bytes each:
// the kind as a byte, any value, four indices as bytes near the vertex range
// (some past it, some negative), and kind_data as two doubles, any bits.
//
// Properties: every finite double written with 17 digits reads back as
// itself, `format` reads only what it validated (the sanitizers check that),
// a refusal names a segment that exists or the arguments as a whole, and
// whatever it accepts is path data that parses completely, and formats
// again. At 17 digits and without arcs or conics, which are recomputed from
// the radii written and split past MAX_ARC, the parse gives back exactly the
// input's kinds and the coordinates of every point they name (with -0 read
// as 0, since an absolute coordinate is 0 + the number written).
#include "../../src/svgpath.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 2) {
        return 0;
    }
    const std::size_t vertex_count = data[0] % 17;
    const int digits = data[1] % 19;  // 0 and 18 are refused
    const std::size_t vertex_bytes = vertex_count * 2 * sizeof(double);
    if (size < 2 + vertex_bytes) {
        return 0;
    }
    const std::uint8_t* cursor = data + 2;
    svgpath::SvgPath path;
    path.vertices.resize(2 * vertex_count);
    if (vertex_count > 0) {
        std::memcpy(path.vertices.data(), cursor, vertex_bytes);
    }
    cursor += vertex_bytes;
    constexpr std::size_t SEGMENT_BYTES = 1 + 4 + (2 * sizeof(double));
    for (std::size_t left = size - 2 - vertex_bytes; left >= SEGMENT_BYTES; left -= SEGMENT_BYTES) {
        svgpath::SvgSegment segment = {cursor[0], {}, {}, {}};
        for (int slot = 0; slot < 4; slot++) {
            segment.points[slot] = static_cast<std::int64_t>(cursor[1 + slot] % 20) - 2;
        }
        std::memcpy(segment.kind_data, cursor + 5, sizeof(segment.kind_data));
        path.segments.push_back(segment);
        cursor += SEGMENT_BYTES;
    }

    // every finite double, written with 17 digits, reads back as itself
    for (const double value : path.vertices) {
        if (!svgpath::within(value)) {
            continue;
        }
        char number[svgpath::MOST_NUMBER_CHARACTERS];
        const char* end = svgpath::write_number(number, value, 17);
        double read = 0.0;
        svgpath::PathDataReader reader = {number, static_cast<std::size_t>(end - number), 0};
        if (!svgpath::read_number(&reader, &read) || reader.offset != reader.size ||
            std::memcmp(&read, &value, sizeof(double)) != 0) {
            __builtin_trap();
        }
    }

    std::string written;
    std::size_t refused = 0;
    if (svgpath::format(&path, digits, &written, &refused) != nullptr) {
        if (refused > path.segments.size()) {
            __builtin_trap();  // a refusal names a segment that exists, or all of them
        }
        return 0;
    }
    svgpath::SvgPath parsed;
    if (svgpath::parse(written.data(), written.size(), &parsed) != written.size()) {
        __builtin_trap();  // everything `format` writes is path data
    }
    std::string rewritten;
    if (svgpath::format(&parsed, digits, &rewritten, &refused) != nullptr) {
        __builtin_trap();
    }

    for (const svgpath::SvgSegment& segment : path.segments) {
        if (segment.kind == svgpath::ARC || segment.kind == svgpath::CONIC) {
            return 0;
        }
    }
    if (digits != 17 || parsed.segments.size() != path.segments.size()) {
        if (digits == 17) {
            __builtin_trap();
        }
        return 0;
    }
    for (std::size_t index = 0; index < path.segments.size(); index++) {
        const svgpath::SvgSegment* given = &path.segments[index];
        const svgpath::SvgSegment* read = &parsed.segments[index];
        if (given->kind != read->kind) {
            __builtin_trap();
        }
        for (int slot = 0; slot < 4; slot++) {
            if (read->points[slot] < 0) {
                continue;  // a slot this kind doesn't use
            }
            const double* expected = &path.vertices[2 * static_cast<std::size_t>(given->points[slot])];
            const double* actual = &parsed.vertices[2 * static_cast<std::size_t>(read->points[slot])];
            if (expected[0] != actual[0] || expected[1] != actual[1]) {
                __builtin_trap();
            }
        }
    }
    return 0;
}
