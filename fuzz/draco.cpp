// Any bytes either fail to decode or decode to a mesh whose faces index real
// points and whose attributes can be read for every point, the same reads
// `draco_decode` makes. Draco asking for absurd amounts of memory is a
// decompression bomb, not a bug (a valid buffer of a few bytes can decode to
// millions of faces): it refuses what doesn't fit, by throwing, and
// `uv run scripts fuzz` ignores out-of-memory reports for what does.
#include <draco/compression/decode.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    draco::DecoderBuffer buffer;
    buffer.Init(reinterpret_cast<const char*>(data), size);
    draco::Decoder decoder;
    std::unique_ptr<draco::Mesh> mesh;
    try {
        auto decoded = decoder.DecodeMeshFromBuffer(&buffer);
        if (!decoded.ok()) {
            return 0;
        }
        mesh = std::move(decoded).value();
    } catch (const std::bad_alloc&) {
        return 0;
    }
    const std::uint32_t points = mesh->num_points();
    for (std::uint32_t f = 0; f < mesh->num_faces(); f++) {
        for (const auto& p : mesh->face(draco::FaceIndex(f))) {
            if (p.value() >= points) {
                __builtin_trap();
            }
        }
    }
    std::vector<std::uint8_t> value;
    for (std::int32_t a = 0; a < mesh->num_attributes(); a++) {
        const draco::PointAttribute* attribute = mesh->attribute(a);
        value.resize(static_cast<std::size_t>(attribute->byte_stride()));
        for (std::uint32_t p = 0; p < points; p++) {
            std::memcpy(value.data(), attribute->GetAddress(attribute->mapped_index(draco::PointIndex(p))),
                        value.size());
        }
    }
    return 0;
}
