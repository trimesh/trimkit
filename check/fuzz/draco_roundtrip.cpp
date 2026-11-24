// Property: any mesh built from the fuzz input encodes as
// `trimkit.exchange.draco_encode` does (src/draco.cpp, sequential, so the
// order is kept) and decodes back with exactly the same faces, positions
// within one quantization step, and integers of any range draco can hold
// exactly. A range it can't hold is refused, never written wrong.
#include "../../src/trimkit.hpp"

#include <fuzzer/FuzzedDataProvider.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider in(data, size);
    const auto points = in.ConsumeIntegralInRange<std::uint32_t>(3, 300);
    const auto face_count = in.ConsumeIntegralInRange<std::uint32_t>(1, 600);
    const int bits = in.ConsumeIntegralInRange<int>(8, 24);
    std::vector<std::uint32_t> triangles(3 * std::size_t{face_count});
    for (std::uint32_t& v : triangles) {
        v = in.ConsumeIntegralInRange<std::uint32_t>(0, points - 1);
    }
    std::vector<float> position(3 * std::size_t{points});
    for (float& p : position) {
        p = static_cast<float>(in.ConsumeFloatingPointInRange<double>(-1000.0, 1000.0));
    }
    // the widest range draco can read back, at either end of int32, and its
    // ends too, which raw bytes would rarely reach; one more and it refuses
    const bool too_wide = in.ConsumeIntegralInRange<int>(0, 7) == 0;
    // (a range one wider starts at int32's lowest, so its highest still fits)
    const std::int32_t lowest = too_wide || in.ConsumeBool() ? std::numeric_limits<std::int32_t>::min() : 1;
    const std::int32_t highest = lowest + (std::numeric_limits<std::int32_t>::max() - (too_wide ? 0 : 1));
    const std::int32_t ends[] = {lowest, highest};
    std::vector<std::int32_t> integer(points);
    for (std::int32_t& value : integer) {
        value = in.ConsumeBool() ? in.PickValueInArray(ends) : in.ConsumeIntegralInRange(lowest, highest);
    }
    integer[0] = lowest;
    integer[1] = highest;
    const std::vector<trimkit::DracoAttributeInput> attributes = {
        {position.data(), trimkit::DracoComponent::FLOAT32, 3}, {integer.data(), trimkit::DracoComponent::INT32, 1}};
    const std::vector<std::int64_t> kinds = {0, 4};  // POSITION, GENERIC
    std::vector<char> encoded;
    try {
        encoded = trimkit::draco_encode_mesh(triangles, points, attributes, kinds, bits);
    } catch (const std::invalid_argument&) {
        if (!too_wide) {
            __builtin_trap();
        }
        return 0;
    }
    if (too_wide) {
        __builtin_trap();  // a range draco can't hold must be refused
    }
    std::vector<std::int64_t> faces;
    std::vector<trimkit::DracoAttributeOutput> decoded;
    try {
        trimkit::draco_decode_mesh(reinterpret_cast<const std::uint8_t*>(encoded.data()), encoded.size(), {0, 1},
                                   &faces, &decoded);
    } catch (const std::invalid_argument&) {
        __builtin_trap();  // draco must read back what it just wrote
    }
    if (faces.size() != triangles.size() || decoded.size() != 2 ||
        decoded[0].values.size() != position.size() * sizeof(float) ||
        decoded[1].values.size() != integer.size() * sizeof(std::int32_t)) {
        __builtin_trap();
    }
    for (std::size_t i = 0; i < faces.size(); i++) {
        if (faces[i] != triangles[i]) {
            __builtin_trap();
        }
    }
    const auto [lo, hi] = std::minmax_element(position.begin(), position.end());
    const double step = (double{*hi} - *lo) / ((1 << bits) - 1);
    for (std::size_t i = 0; i < position.size(); i++) {
        float value = 0.0F;
        std::memcpy(&value, decoded[0].values.data() + (i * sizeof(float)), sizeof(float));
        if (std::abs(double{value} - position[i]) > step + 1e-3) {
            __builtin_trap();
        }
    }
    for (std::size_t i = 0; i < integer.size(); i++) {
        std::int32_t value = 0;
        std::memcpy(&value, decoded[1].values.data() + (i * sizeof(std::int32_t)), sizeof(std::int32_t));
        if (value != integer[i]) {
            __builtin_trap();
        }
    }
    return 0;
}
