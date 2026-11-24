// Property: any small mesh parameterizes as `trimkit.parametrize` does
// (src/xatlas.cpp), on the real pool, within libFuzzer's timeout: valid
// indices are never refused, every output vertex maps back to a real input
// vertex, every face indexes the output, and every uv is in the unit square.
// Positions span any scale, as the unit centers and scales them itself.
#include "../../src/trimkit.hpp"

#include <fuzzer/FuzzedDataProvider.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider in(data, size);
    const auto n = in.ConsumeIntegralInRange<std::uint32_t>(3, 64);
    const auto m = in.ConsumeIntegralInRange<std::uint32_t>(1, 128);
    const double scale = std::ldexp(1.0, in.ConsumeIntegralInRange<int>(0, 140) - 60);
    const double offset = in.ConsumeFloatingPointInRange<double>(-1e20, 1e20);
    std::vector<double> points(3 * std::size_t{n});
    for (double& p : points) {
        p = offset + (scale * in.ConsumeFloatingPointInRange<double>(-1.0, 1.0));
    }
    std::vector<std::uint32_t> indices(3 * std::size_t{m});
    if (in.ConsumeBool() && n >= 9) {
        // a grid's connectivity with fuzzed positions: random faces almost never
        // form a surface, and a surface is what reaches charting and packing
        const std::uint32_t side = 3 + (n - 9) / 16;
        indices.clear();
        for (std::uint32_t i = 0; i + 1 < side; i++) {
            for (std::uint32_t j = 0; j + 1 < side; j++) {
                const std::uint32_t a = (i * side) + j;
                indices.insert(indices.end(), {a, a + side, a + 1, a + 1, a + side, a + side + 1});
            }
        }
    } else {
        for (auto& i : indices) {
            i = in.ConsumeIntegralInRange<std::uint32_t>(0, n - 1);
        }
    }
    std::vector<std::int64_t> sources;
    std::vector<std::int64_t> faces;
    std::vector<double> uv;
    try {
        trimkit::parametrize_mesh(points.data(), n, indices, trimkit::parallel::Stop{}, &sources, &faces, &uv);
    } catch (const std::invalid_argument&) {
        __builtin_trap();  // every index is in range: nothing to refuse
    }
    if (uv.size() != 2 * sources.size() || faces.size() % 3 != 0) {
        __builtin_trap();
    }
    for (const std::int64_t source : sources) {
        if (source < 0 || source >= n) {
            __builtin_trap();
        }
    }
    for (const std::int64_t index : faces) {
        if (index < 0 || static_cast<std::size_t>(index) >= sources.size()) {
            __builtin_trap();
        }
    }
    for (const double value : uv) {
        if (!(value >= 0.0 && value <= 1.0)) {
            __builtin_trap();
        }
    }
    return 0;
}
