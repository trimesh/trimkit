// Property: any simple polygon, built from the fuzz input as a star around
// the origin with up to 144 disjoint star-shaped holes strictly inside it,
// triangulates within bounds (every index in range, at most n + 2h - 2
// triangles, all wound one way, clean under the sanitizers), and with at
// most one hole tiles exactly the polygon's area. With more holes earcut
// 3.2.4 itself sometimes leaves out or overlaps area (an upstream bug pinned
// by test_hole_tiling_upstream_bug in tests/test_earcut.py), so exact tiling
// is upstream's to fix there. Many holes exercise the block index that
// bridges them (vendor/patches/earcut-hole-bridge-blocks.patch).
#include <earcut.hpp>
#include <fuzzer/FuzzedDataProvider.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <set>
#include <vector>

using Ring = std::vector<std::array<double, 2>>;

// a ring with strictly increasing angles around the origin and no gap of
// half a turn or more, so the origin is in its kernel and it is simple; the
// largest gap bounds the disk around the origin it is guaranteed to contain
static Ring star(FuzzedDataProvider& in, double lo, double hi, std::size_t most_vertices, double center_x,
                 double center_y, double& inscribed) {
    std::set<std::uint16_t> angles;
    const auto n = in.ConsumeIntegralInRange<std::size_t>(3, most_vertices);
    while (angles.size() < n && in.remaining_bytes() > 0) {
        angles.insert(in.ConsumeIntegral<std::uint16_t>());
    }
    if (angles.size() < 3) {
        inscribed = 0.0;
        return {};
    }
    int gap = 65536 - *angles.rbegin() + *angles.begin();
    for (auto a = angles.begin(), b = std::next(a); b != angles.end(); ++a, ++b) {
        gap = std::max(gap, *b - *a);
    }
    if (gap >= 32768) {
        inscribed = 0.0;
        return {};
    }
    inscribed = lo * std::cos(M_PI * gap / 65536.0);
    Ring ring;
    for (const std::uint16_t a : angles) {
        const double t = 2.0 * M_PI * a / 65536.0;
        const double r = lo + ((hi - lo) * in.ConsumeProbability<double>());
        ring.push_back({center_x + (r * std::cos(t)), center_y + (r * std::sin(t))});
    }
    return ring;
}

static double shoelace(const Ring& r) {
    double sum = 0.0;
    for (std::size_t i = 0; i < r.size(); i++) {
        const auto& p = r[i];
        const auto& q = r[(i + 1) % r.size()];
        sum += (p[0] * q[1]) - (q[0] * p[1]);
    }
    return std::abs(sum) / 2.0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider in(data, size);
    double inscribed = 0.0;
    std::vector<Ring> polygon{star(in, 0.5, 1.0, 2000, 0.0, 0.0, inscribed)};
    if (polygon[0].size() < 3) {
        return 0;
    }
    // holes on a grid of cells inside the square inscribed in the disk the
    // outer ring is guaranteed to contain, each within its own cell: so they
    // are disjoint and strictly inside, however the fuzzer shapes them
    const auto side = in.ConsumeIntegralInRange<std::size_t>(0, 12);
    const double half = 0.7 * inscribed;  // half * sqrt(2) < inscribed
    const double cell = side > 0 ? 2.0 * half / static_cast<double>(side) : 0.0;
    for (std::size_t row = 0; row < side; row++) {
        for (std::size_t column = 0; column < side; column++) {
            if (!in.ConsumeBool()) {
                continue;
            }
            double unused = 0.0;
            const double center_x = -half + ((static_cast<double>(column) + 0.5) * cell);
            const double center_y = -half + ((static_cast<double>(row) + 0.5) * cell);
            Ring hole = star(in, 0.1 * cell, 0.45 * cell, 12, center_x, center_y, unused);
            if (hole.size() >= 3) {
                polygon.push_back(hole);
            }
        }
    }
    std::vector<std::array<double, 2>> points;
    for (const Ring& r : polygon) {
        points.insert(points.end(), r.begin(), r.end());
    }
    const std::vector<std::uint64_t> index = mapbox::earcut<std::uint64_t>(polygon);
    // at most n + 2h - 2 triangles, n the vertices and h the holes
    if (index.size() % 3 != 0 || index.size() / 3 > points.size() + (2 * (polygon.size() - 1)) - 2) {
        __builtin_trap();
    }

    double total = 0.0;
    double largest = 0.0;
    std::vector<double> areas;
    for (std::size_t i = 0; i + 2 < index.size(); i += 3) {
        const auto& a = points.at(index[i]);
        const auto& b = points.at(index[i + 1]);
        const auto& c = points.at(index[i + 2]);
        areas.push_back(((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0])));
        largest = std::max(largest, std::abs(areas.back()));
        total += std::abs(areas.back()) / 2.0;
    }
    // all one winding, allowing slivers over collinear vertices to round either way
    const double noise = 1e-9 * largest;
    bool positive = true, negative = true;
    for (const double a : areas) {
        positive = positive && a >= -noise;
        negative = negative && a <= noise;
    }
    double truth = shoelace(polygon[0]);
    for (std::size_t hole = 1; hole < polygon.size(); hole++) {
        truth -= shoelace(polygon[hole]);
    }
    // every coordinate is inside the unit disk, so area errors are absolute
    const bool tiles = std::abs(total - truth) <= 1e-9;
    if (!(positive || negative) || (polygon.size() <= 2 && !tiles)) {
        __builtin_trap();
    }
    return 0;
}
