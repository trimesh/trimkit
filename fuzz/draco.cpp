// Any bytes either fail to decode or decode to a mesh whose faces index real
// points and whose attributes can be read for every point they hold a value
// for, the same checked reads `draco_decode` makes. Draco asking for absurd amounts of memory is a
// decompression bomb, not a bug (a valid buffer of a few bytes can decode to
// millions of faces, and draco_decode has no size limit, as its docstring
// says), so `uv run scripts fuzz` runs this with out-of-memory reports ignored.
#include <draco/compression/decode.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    draco::DecoderBuffer buffer;
    buffer.Init(reinterpret_cast<const char*>(data), size);
    draco::Decoder decoder;
    auto decoded = decoder.DecodeMeshFromBuffer(&buffer);
    if (!decoded.ok()) {
        return 0;
    }
    const auto mesh = std::move(decoded).value();
    const std::uint32_t points = mesh->num_points();
    for (std::uint32_t f = 0; f < mesh->num_faces(); f++) {
        for (const auto& p : mesh->face(draco::FaceIndex(f))) {
            if (p.value() >= points) {
                __builtin_trap();
            }
        }
    }
    // the reads `draco_decode` makes (src/draco.cpp): a point's value only once
    // it is one the attribute holds, within its buffer's bytes. Draco does
    // decode buffers whose points name values past that; the binding refuses
    // them, and reading one here is the crash it would have been
    std::vector<std::uint8_t> value;
    for (std::int32_t a = 0; a < mesh->num_attributes(); a++) {
        const draco::PointAttribute* attribute = mesh->attribute(a);
        const auto stride = static_cast<std::size_t>(attribute->byte_stride());
        const auto offset = static_cast<std::size_t>(attribute->byte_offset());
        const std::size_t bytes = attribute->buffer() == nullptr ? 0 : attribute->buffer()->data_size();
        const std::size_t held =
            stride == 0 || offset > bytes ? 0 : std::min(attribute->size(), (bytes - offset) / stride);
        value.resize(stride);
        for (std::uint32_t p = 0; p < points; p++) {
            const draco::AttributeValueIndex at = attribute->mapped_index(draco::PointIndex(p));
            if (at.value() >= held) {
                break;
            }
            std::memcpy(value.data(), attribute->GetAddress(at), value.size());
        }
    }
    return 0;
}
