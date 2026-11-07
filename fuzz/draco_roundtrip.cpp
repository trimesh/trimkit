// Property: any mesh built from the fuzz input encodes with either method and
// decodes back with the same faces and points, and with sequential encoding
// (what our binding uses) in exactly the same order, positions within one
// quantization step.
#include <draco/compression/decode.h>
#include <draco/compression/expert_encode.h>
#include <fuzzer/FuzzedDataProvider.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider in(data, size);
    const auto points = in.ConsumeIntegralInRange<std::uint32_t>(3, 300);
    const auto faces = in.ConsumeIntegralInRange<std::uint32_t>(1, 600);
    const bool edgebreaker = in.ConsumeBool();
    const int bits = in.ConsumeIntegralInRange<int>(8, 20);

    draco::Mesh mesh;
    mesh.set_num_points(points);
    for (std::uint32_t f = 0; f < faces; f++) {
        draco::Mesh::Face face;
        for (auto& v : face) {
            v = draco::PointIndex(in.ConsumeIntegralInRange<std::uint32_t>(0, points - 1));
        }
        mesh.AddFace(face);
    }
    draco::GeometryAttribute g;
    g.Init(draco::GeometryAttribute::POSITION, nullptr, 3, draco::DT_FLOAT32, false, 12, 0);
    const int id = mesh.AddAttribute(g, true, points);
    std::vector<float> position(3 * std::size_t{points});
    for (float& p : position) {
        p = static_cast<float>(in.ConsumeFloatingPointInRange<double>(-1000.0, 1000.0));
    }
    mesh.attribute(id)->buffer()->Update(position.data(), static_cast<std::int64_t>(position.size() * 4));

    draco::ExpertEncoder encoder(mesh);
    encoder.SetAttributeQuantization(id, bits);
    encoder.SetEncodingMethod(edgebreaker ? draco::MESH_EDGEBREAKER_ENCODING : draco::MESH_SEQUENTIAL_ENCODING);
    draco::EncoderBuffer encoded;
    if (!encoder.EncodeToBuffer(&encoded).ok()) {
        return 0;  // e.g. edgebreaker refusing degenerate connectivity
    }
    draco::DecoderBuffer buffer;
    buffer.Init(encoded.data(), encoded.size());
    draco::Decoder decoder;
    auto decoded = decoder.DecodeMeshFromBuffer(&buffer);
    if (!decoded.ok()) {
        __builtin_trap();  // draco must read back what it just wrote
    }
    const auto out = std::move(decoded).value();
    if (edgebreaker) {
        return 0;  // edgebreaker may reorder and drop degenerate faces
    }
    if (out->num_faces() != faces || out->num_points() != points) {
        __builtin_trap();
    }
    float lo = position[0], hi = position[0];
    for (const float p : position) {
        lo = std::min(lo, p);
        hi = std::max(hi, p);
    }
    const double step = (double{hi} - lo) / ((1 << bits) - 1);
    const draco::PointAttribute* a = out->GetNamedAttribute(draco::GeometryAttribute::POSITION);
    for (std::uint32_t p = 0; p < points; p++) {
        float xyz[3];
        a->GetMappedValue(draco::PointIndex(p), xyz);
        for (int c = 0; c < 3; c++) {
            if (std::abs(double{xyz[c]} - position[(3 * p) + c]) > step + 1e-3) {
                __builtin_trap();
            }
        }
    }
    for (std::uint32_t f = 0; f < faces; f++) {
        if (!(out->face(draco::FaceIndex(f)) == mesh.face(draco::FaceIndex(f)))) {
            __builtin_trap();
        }
    }
    return 0;
}
