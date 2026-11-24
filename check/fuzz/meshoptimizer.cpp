// Property: any small mesh simplifies as `trimkit.simplify` does
// (src/meshoptimizer.cpp): refused exactly when face_count or maximum_error
// is out of range, else into triangles that index the output vertices, never
// more of them than went in, each output vertex exactly the input vertex it
// names, and the same answer twice. Positions span any scale, as the unit
// centers and scales them itself.
#include "../../src/trimkit.hpp"

#include <fuzzer/FuzzedDataProvider.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// any int64, drawn as a uint64's bits: libFuzzer's own range arithmetic
// wraps for a signed range this wide, which UBSan reports
std::int64_t any_int64(FuzzedDataProvider& provider) {
    const std::uint64_t bits = provider.ConsumeIntegral<std::uint64_t>();
    std::int64_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider input(data, size);
    const std::uint32_t vertex_count = input.ConsumeIntegralInRange<std::uint32_t>(3, 96);
    const std::uint32_t face_count = input.ConsumeIntegralInRange<std::uint32_t>(1, 192);
    // any scale within the coordinate bound, and any offset: the unit
    // centers and scales, so a mesh 1e-20 across and one at 1e25 are the same
    const double scale = std::ldexp(1.0, input.ConsumeIntegralInRange<int>(0, 140) - 60);
    const double offset = input.ConsumeFloatingPointInRange<double>(-1e20, 1e20);
    std::vector<double> vertices(3 * std::size_t{vertex_count});
    for (double& value : vertices) {
        value = offset + (scale * input.ConsumeFloatingPointInRange<double>(-1.0, 1.0));
    }
    std::vector<std::uint32_t> indices;
    if (input.ConsumeBool() && vertex_count >= 9) {
        // a grid's connectivity with fuzzed positions: random faces rarely
        // form a surface, and a surface is what has edges to collapse
        const std::uint32_t side = 3 + ((vertex_count - 9) / 16);
        for (std::uint32_t row = 0; row + 1 < side; row++) {
            for (std::uint32_t column = 0; column + 1 < side; column++) {
                const std::uint32_t corner = (row * side) + column;
                indices.insert(indices.end(),
                               {corner, corner + side, corner + 1, corner + 1, corner + side, corner + side + 1});
            }
        }
    } else {
        indices.resize(3 * std::size_t{face_count});
        for (std::uint32_t& index : indices) {
            index = input.ConsumeIntegralInRange<std::uint32_t>(0, vertex_count - 1);
        }
    }
    // as Python can pass them: any int64, and any double
    const std::int64_t wanted =
        input.ConsumeBool() ? any_int64(input) : input.ConsumeIntegralInRange<std::int64_t>(0, 200);
    const double maximum_error = input.ConsumeBool()   ? input.ConsumeFloatingPoint<double>()
                                 : input.ConsumeBool() ? std::numeric_limits<double>::infinity()
                                                       : scale * input.ConsumeFloatingPointInRange<double>(0.0, 2.0);
    const bool lock_border = input.ConsumeBool();
    const bool refused = wanted < 0 || !(maximum_error >= 0.0);

    std::vector<double> output_vertices;
    std::vector<std::int64_t> output_faces;
    std::vector<std::int64_t> sources;
    try {
        trimkit::simplify_mesh(vertices.data(), vertex_count, indices, wanted, maximum_error, lock_border,
                               &output_vertices, &output_faces, &sources);
    } catch (const std::invalid_argument&) {
        if (!refused) {
            __builtin_trap();
        }
        return 0;
    }
    if (refused || output_faces.size() % 3 != 0 || output_faces.size() > indices.size() ||
        output_vertices.size() != 3 * sources.size() || sources.size() > vertex_count) {
        __builtin_trap();
    }
    for (const std::int64_t index : output_faces) {
        if (index < 0 || static_cast<std::size_t>(index) >= sources.size()) {
            __builtin_trap();
        }
    }
    for (std::size_t i = 0; i < sources.size(); i++) {
        for (std::size_t k = 0; k < 3; k++) {
            if (output_vertices[(3 * i) + k] != vertices[(3 * static_cast<std::size_t>(sources[i])) + k]) {
                __builtin_trap();
            }
        }
    }
    std::vector<double> vertices_again;
    std::vector<std::int64_t> faces_again;
    std::vector<std::int64_t> sources_again;
    trimkit::simplify_mesh(vertices.data(), vertex_count, indices, wanted, maximum_error, lock_border, &vertices_again,
                           &faces_again, &sources_again);
    if (faces_again != output_faces || sources_again != sources) {
        __builtin_trap();
    }
    return 0;
}
