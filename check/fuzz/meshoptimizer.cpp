// Property: any small mesh, with or without per-vertex attributes, simplifies
// with the options src/meshoptimizer.cpp uses into triangles that only
// reference input vertices, never more of them than went in, with a finite
// error, and the same answer twice.
#include <fuzzer/FuzzedDataProvider.h>
#include <meshoptimizer.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

struct Attributes {
    std::vector<float> values;
    std::vector<float> weights;
};

static std::size_t simplify(const std::vector<std::uint32_t>& indices, const std::vector<float>& positions,
                            const Attributes& attributes, std::size_t target, float target_error, unsigned int options,
                            std::vector<std::uint32_t>* kept, float* error) {
    const std::size_t attribute_count = attributes.weights.size();
    kept->assign(indices.size(), 0);
    const std::size_t count = meshopt_simplifyWithAttributes(
        kept->data(), indices.data(), indices.size(), positions.data(), positions.size() / 3, 3 * sizeof(float),
        attribute_count > 0 ? attributes.values.data() : nullptr, attribute_count * sizeof(float),
        attribute_count > 0 ? attributes.weights.data() : nullptr, attribute_count, nullptr, target, target_error,
        options, error);
    kept->resize(count);
    return count;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider input(data, size);
    const std::uint32_t vertex_count = input.ConsumeIntegralInRange<std::uint32_t>(3, 96);
    const std::uint32_t face_count = input.ConsumeIntegralInRange<std::uint32_t>(1, 192);
    std::vector<float> positions(3 * std::size_t{vertex_count});
    for (float& position : positions) {
        // within [-1, 1]: what the binding's centering and scaling produce
        position = input.ConsumeFloatingPointInRange<float>(-1.0F, 1.0F);
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
    // as the binding passes them: unit weights, values as given
    Attributes attributes;
    const std::size_t attribute_count = input.ConsumeIntegralInRange<std::size_t>(0, 4);
    attributes.weights.assign(attribute_count, 1.0F);
    attributes.values.resize(attribute_count * vertex_count);
    for (float& value : attributes.values) {
        value = input.ConsumeFloatingPointInRange<float>(-4.0F, 4.0F);
    }
    const std::size_t target = 3 * input.ConsumeIntegralInRange<std::size_t>(0, indices.size() / 3);
    const float target_error = input.ConsumeBool() ? 1e30F : input.ConsumeFloatingPointInRange<float>(0.0F, 2.0F);
    unsigned int options = meshopt_SimplifyErrorAbsolute | meshopt_SimplifyRegularizeLight;
    if (input.ConsumeBool()) {
        options |= meshopt_SimplifyLockBorder;
    }

    std::vector<std::uint32_t> kept;
    float error = 0.0F;
    const std::size_t count = simplify(indices, positions, attributes, target, target_error, options, &kept, &error);
    if (count % 3 != 0 || count > indices.size() || !std::isfinite(error) || error < 0.0F) {
        __builtin_trap();
    }
    for (const std::uint32_t index : kept) {
        if (index >= vertex_count) {
            __builtin_trap();
        }
    }
    std::vector<std::uint32_t> again;
    float error_again = 0.0F;
    simplify(indices, positions, attributes, target, target_error, options, &again, &error_again);
    if (again != kept || error_again != error) {
        __builtin_trap();
    }
    return 0;
}
