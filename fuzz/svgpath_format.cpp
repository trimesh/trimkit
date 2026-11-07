// `format` on raw, hand-built rows rather than on what `parse` produced: the
// first byte picks how many vertices follow as doubles, and the rest are
// Segment rows, any bit pattern at all.
//
// Properties: every finite double is written as std::to_chars writes it,
// `format` reads only what it validated (the sanitizers check that), and
// whatever it accepts is path data that parses completely and
// then round trips byte for byte. Without arcs, which `parse` may legally
// rewrite (radii scaled up, a zero-length arc dropped), the parse gives back
// exactly the input's kinds, closes flags and coordinates (with -0 read as
// 0, since an absolute coordinate is 0 + the number written).
#include <svgpath.hpp>

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 1) {
        return 0;
    }
    const std::size_t vertex_count = data[0] % 17;
    const std::size_t vertex_bytes = vertex_count * 2 * sizeof(double);
    if (size - 1 < vertex_bytes) {
        return 0;
    }
    // copied out to aligned storage, as numpy's arrays are
    std::vector<double> vertices(2 * vertex_count);
    if (vertex_bytes > 0) {
        std::memcpy(vertices.data(), data + 1, vertex_bytes);
    }
    const std::size_t segment_count = (size - 1 - vertex_bytes) / sizeof(svgpath::Segment);
    std::vector<svgpath::Segment> segments(segment_count);
    if (segment_count > 0) {
        std::memcpy(segments.data(), data + 1 + vertex_bytes, segment_count * sizeof(svgpath::Segment));
    }

    // svgpath writes numbers exactly as std::to_chars does, character for
    // character, for every finite double the fuzzer makes
    for (const double value : vertices) {
        if (!std::isfinite(value)) {
            continue;
        }
        char expected[64];
        const std::to_chars_result result = std::to_chars(expected, expected + sizeof(expected), value);
        std::string number;
        svgpath::write_number(&number, value);
        if (number != std::string(expected, result.ptr)) {
            __builtin_trap();
        }
    }

    std::string written;
    std::size_t refused = 0;
    if (svgpath::format(vertices.data(), vertex_count, segments.data(), segment_count, &written, &refused) != nullptr) {
        if (refused >= segment_count) {
            __builtin_trap();  // a refusal names a segment that exists
        }
        return 0;
    }
    svgpath::Path parsed;
    if (svgpath::parse(written.data(), written.size(), &parsed) != written.size()) {
        __builtin_trap();  // everything `format` writes is path data
    }
    std::string rewritten;
    if (svgpath::format(parsed.vertices.data(), parsed.vertices.size() / 2, parsed.segments.data(),
                        parsed.segments.size(), &rewritten, &refused) != nullptr) {
        __builtin_trap();
    }
    svgpath::Path again;
    static_cast<void>(svgpath::parse(rewritten.data(), rewritten.size(), &again));
    if (again.vertices.size() != parsed.vertices.size() || again.segments.size() != parsed.segments.size() ||
        (!parsed.vertices.empty() &&
         std::memcmp(again.vertices.data(), parsed.vertices.data(), parsed.vertices.size() * sizeof(double)) != 0) ||
        (!parsed.segments.empty() && std::memcmp(again.segments.data(), parsed.segments.data(),
                                                 parsed.segments.size() * sizeof(svgpath::Segment)) != 0)) {
        __builtin_trap();
    }

    for (std::size_t index = 0; index < segment_count; index++) {
        if (segments[index].kind == svgpath::ARC) {
            return 0;
        }
    }
    if (parsed.segments.size() != segment_count) {
        __builtin_trap();
    }
    for (std::size_t index = 0; index < segment_count; index++) {
        const svgpath::Segment* given = &segments[index];
        const svgpath::Segment* read = &parsed.segments[index];
        if (given->kind != read->kind || (given->closes != 0) != (read->closes != 0)) {
            __builtin_trap();
        }
        for (int slot = 0; slot < 4; slot++) {
            if (read->points[slot] < 0) {
                continue;  // a slot this kind doesn't use
            }
            const double* expected = &vertices[2 * static_cast<std::size_t>(given->points[slot])];
            const double* actual = &parsed.vertices[2 * static_cast<std::size_t>(read->points[slot])];
            if (expected[0] != actual[0] || expected[1] != actual[1]) {
                __builtin_trap();
            }
        }
    }
    return 0;
}
