// Property: any small mesh decomposes as `trimkit.decompose` does
// (src/vhacd.cpp), on the real pool, within libFuzzer's timeout: refused
// exactly when a parameter is out of range, else into hulls whose vertices
// are finite and whose faces index them. Mostly hunting hangs and crashes.
#include "../../src/trimkit.hpp"

#include <fuzzer/FuzzedDataProvider.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
    FuzzedDataProvider in(data, size);
    const auto n = in.ConsumeIntegralInRange<std::uint32_t>(4, 48);
    const auto m = in.ConsumeIntegralInRange<std::uint32_t>(1, 96);
    std::vector<double> points(3 * std::size_t{n});
    for (double& p : points) {
        p = in.ConsumeFloatingPointInRange<double>(-10.0, 10.0);
    }
    std::vector<std::uint32_t> triangles(3 * std::size_t{m});
    for (auto& t : triangles) {
        t = in.ConsumeIntegralInRange<std::uint32_t>(0, n - 1);
    }
    // in range, sized for the fuzzer's time, or now and then any int64
    const bool any = in.ConsumeIntegralInRange<int>(0, 15) == 0;
    const std::int64_t hulls = any ? any_int64(in) : in.ConsumeIntegralInRange<std::int64_t>(1, 8);
    const std::int64_t vertices = any ? any_int64(in) : in.ConsumeIntegralInRange<std::int64_t>(4, 64);
    const std::int64_t resolution = any ? any_int64(in) : in.ConsumeIntegralInRange<std::int64_t>(1000, 20000);
    const double error = in.ConsumeFloatingPointInRange<double>(-1.0, 101.0);
    constexpr std::int64_t MOST = 4294967295;
    const bool refused = hulls < 1 || hulls > MOST || vertices < 4 || vertices > MOST || resolution < 1 ||
                         resolution > 64000000 || error < 0.0 || error > 100.0;
    // a refusal comes first, so an in-range but huge resolution never runs
    if (!refused && resolution > 20000) {
        return 0;
    }
    std::vector<trimkit::ConvexHullMesh> decomposed;
    try {
        decomposed = trimkit::decompose_mesh(points.data(), n, triangles, hulls, vertices, resolution, error,
                                             trimkit::parallel::Stop{});
    } catch (const std::invalid_argument&) {
        if (!refused) {
            __builtin_trap();
        }
        return 0;
    }
    if (refused) {
        __builtin_trap();
    }
    for (const trimkit::ConvexHullMesh& hull : decomposed) {
        for (const double v : hull.vertices) {
            if (!std::isfinite(v)) {
                __builtin_trap();
            }
        }
        for (const std::int64_t index : hull.faces) {
            if (index < 0 || static_cast<std::size_t>(index) >= hull.vertices.size() / 3) {
                __builtin_trap();
            }
        }
    }
    return 0;
}
