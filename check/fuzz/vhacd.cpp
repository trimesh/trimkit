// Property: any small mesh decomposes, within libFuzzer's timeout, into hulls
// whose vertices are all finite. This is mostly hunting hangs and crashes.
#define ENABLE_VHACD_IMPLEMENTATION 1
#include <VHACD.h>
#include <fuzzer/FuzzedDataProvider.h>

#include <cmath>
#include <cstdint>
#include <vector>

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
    VHACD::IVHACD::Parameters params;
    params.m_resolution = in.ConsumeIntegralInRange<std::uint32_t>(1000, 20000);
    params.m_maxConvexHulls = in.ConsumeIntegralInRange<std::uint32_t>(1, 8);
    params.m_asyncACD = false;
    VHACD::IVHACD* vhacd = VHACD::CreateVHACD();
    vhacd->Compute(points.data(), n, triangles.data(), m, params);
    VHACD::IVHACD::ConvexHull hull;
    for (std::uint32_t h = 0; h < vhacd->GetNConvexHulls(); h++) {
        vhacd->GetConvexHull(h, hull);
        for (const auto& v : hull.m_points) {
            if (!std::isfinite(v.mX) || !std::isfinite(v.mY) || !std::isfinite(v.mZ)) {
                __builtin_trap();
            }
        }
    }
    vhacd->Release();
    return 0;
}
