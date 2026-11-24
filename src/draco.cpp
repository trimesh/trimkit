// glTF `KHR_draco_mesh_compression` with google/draco, and nothing else.
// Python-free (trimkit.hpp).
//
// Both directions speak the extension's own vocabulary: attributes are passed
// in order with a kind for each, and identified in the buffer by the draco
// unique id stored in the extension JSON. Only glTF accessor component types
// are taken.
//
// Decoding has no size limit: draco sizes its allocations from counts the
// buffer declares, and a valid buffer of a few bytes can really decode to
// millions of points (rANS codes a near-certain symbol in almost no bits). What
// doesn't fit in the memory available is std::bad_alloc (src/allocator.cpp),
// not the kernel's SIGKILL. draco copies raw bytes in its own little-endian
// format, so on a big-endian machine src/trimkit.cpp refuses before calling.

#include "trimkit.hpp"

#include <draco/compression/decode.h>
#include <draco/compression/expert_encode.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

// draco's data type for each DracoComponent, in its order
constexpr draco::DataType DATA_TYPES[] = {draco::DT_INT8,  draco::DT_UINT8,  draco::DT_INT16,  draco::DT_UINT16,
                                          draco::DT_INT32, draco::DT_UINT32, draco::DT_FLOAT32};
static_assert(std::size(DATA_TYPES) == trimkit::DRACO_COMPONENTS);

// draco's attribute type for each kind, in DracoKind's order
constexpr draco::GeometryAttribute::Type KIND_TYPES[] = {
    draco::GeometryAttribute::POSITION, draco::GeometryAttribute::NORMAL, draco::GeometryAttribute::TEX_COORD,
    draco::GeometryAttribute::COLOR, draco::GeometryAttribute::GENERIC};
static_assert(std::size(KIND_TYPES) == trimkit::DRACO_KINDS);

// how hard draco tries, which trades export time for size and is not lossy
constexpr int COMPRESSION_LEVEL = 6;

// bits float attributes are quantized onto by default, what blender emits: a position
// moves at most half a step, i.e. `extents.max() * 2**-15`
int quantization_bits(draco::GeometryAttribute::Type type) {
    switch (type) {
        case draco::GeometryAttribute::POSITION:
            return 14;
        case draco::GeometryAttribute::NORMAL:
            return 10;
        case draco::GeometryAttribute::TEX_COORD:
            return 12;
        case draco::GeometryAttribute::COLOR:
            return 10;
        default:
            return 0;
    }
}

// a step here is float32's own precision at the top of an extent: draco takes
// up to 30, which keeps nothing more and costs about what lossless does
constexpr std::int64_t MOST_QUANTIZATION_BITS = 24;

// draco wraps each integer attribute's values within their range, in int32:
// for a range past this its encoder ignores that it failed, and writes a buffer
// its own decoder refuses
constexpr std::int64_t MOST_INTEGER_RANGE = std::numeric_limits<std::int32_t>::max() - 1;

template <typename T>
bool range_fits(const void* data, std::size_t size) {
    const auto* values = static_cast<const T*>(data);
    const auto [lowest, highest] = std::minmax_element(values, values + size);
    return std::int64_t{*highest} - std::int64_t{*lowest} <= MOST_INTEGER_RANGE;
}

}  // namespace

std::vector<char> trimkit::draco_encode_mesh(const std::vector<std::uint32_t>& triangles, std::size_t count,
                                             const std::vector<DracoAttributeInput>& attributes,
                                             const std::vector<std::int64_t>& kinds, std::optional<std::int64_t> bits) {
    if (bits && (*bits < 0 || *bits > MOST_QUANTIZATION_BITS)) {
        refuse("quantization_bits must be from 0, lossless, to " + std::to_string(MOST_QUANTIZATION_BITS));
    }
    if (kinds.size() != attributes.size()) {
        refuse("kinds must have one entry per attribute");
    }
    for (std::size_t i = 0; i < kinds.size(); i++) {
        if (!indexes(kinds[i], DRACO_KINDS)) {
            refuse("kinds[" + std::to_string(i) + "] is not a DracoKind");
        }
    }
    const std::size_t face_count = triangles.size() / 3;
    draco::Mesh mesh;
    mesh.set_num_points(static_cast<std::uint32_t>(count));
    mesh.SetNumFaces(face_count);
    for (std::size_t i = 0; i < face_count; i++) {
        const draco::Mesh::Face face{draco::PointIndex(triangles[3 * i]), draco::PointIndex(triangles[(3 * i) + 1]),
                                     draco::PointIndex(triangles[(3 * i) + 2])};
        mesh.SetFace(draco::FaceIndex(static_cast<std::uint32_t>(i)), face);
    }
    // one value per point in point order, so draco keeps our vertex order
    draco::ExpertEncoder encoder(mesh);
    for (std::size_t i = 0; i < attributes.size(); i++) {
        const DracoAttributeInput& a = attributes[i];
        const draco::DataType type = DATA_TYPES[static_cast<std::size_t>(a.component)];
        const draco::GeometryAttribute::Type kind = KIND_TYPES[static_cast<std::size_t>(kinds[i])];
        const std::size_t values = count * a.components;
        if ((type == draco::DT_INT32 && !range_fits<std::int32_t>(a.data, values)) ||
            (type == draco::DT_UINT32 && !range_fits<std::uint32_t>(a.data, values))) {
            refuse("draco failed to encode: attributes[" + std::to_string(i) +
                   "] has a range of integers over 2**31 - 2");
        }
        const auto stride = static_cast<std::int64_t>(
            a.components * trimkit::DRACO_COMPONENT_BYTES[static_cast<std::size_t>(a.component)]);
        draco::GeometryAttribute g;
        g.Init(kind, nullptr, static_cast<std::uint8_t>(a.components), type,
               type != draco::DT_FLOAT32 && kind != draco::GeometryAttribute::GENERIC, stride, 0);
        const int id = mesh.AddAttribute(g, true, static_cast<std::uint32_t>(count));
        // the contract callers rely on: draco attribute i is attributes[i]
        if (id < 0 || mesh.attribute(id)->unique_id() != i) {
            refuse("draco failed to encode: could not add attribute " + std::to_string(i));
        }
        mesh.attribute(id)->buffer()->Update(a.data, stride * static_cast<std::int64_t>(count));
        const int quantized = bits ? static_cast<int>(*bits) : quantization_bits(kind);
        if (type == draco::DT_FLOAT32 && quantized > 0) {
            encoder.SetAttributeQuantization(id, quantized);
        }
    }
    encoder.SetSpeedOptions(10 - COMPRESSION_LEVEL, 10 - COMPRESSION_LEVEL);
    encoder.SetEncodingMethod(draco::MESH_SEQUENTIAL_ENCODING);
    draco::EncoderBuffer encoded;
    const draco::Status status = encoder.EncodeToBuffer(&encoded);
    if (!status.ok()) {
        refuse("draco failed to encode: " + status.error_msg_string());
    }
    return {encoded.data(), encoded.data() + encoded.size()};
}

void trimkit::draco_decode_mesh(const std::uint8_t* buffer, std::size_t size, const std::vector<std::int64_t>& ids,
                                std::vector<std::int64_t>* faces, std::vector<DracoAttributeOutput>* attributes) {
    std::vector<std::uint32_t> wanted(ids.size());
    for (std::size_t i = 0; i < ids.size(); i++) {
        if (!indexes(ids[i], std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 1)) {
            refuse("draco_attributes[" + std::to_string(i) + "] is not a draco attribute id");
        }
        wanted[i] = static_cast<std::uint32_t>(ids[i]);
    }
    draco::DecoderBuffer input;
    input.Init(reinterpret_cast<const char*>(buffer), size);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    draco::Decoder decoder;
    draco::StatusOr<std::unique_ptr<draco::Mesh>> decoded = decoder.DecodeMeshFromBuffer(&input);
    if (!decoded.ok()) {
        refuse("draco failed to decode: " + decoded.status().error_msg_string());
    }
    const std::unique_ptr<draco::Mesh> mesh = std::move(decoded).value();
    // a face naming a point the mesh doesn't have, or a point naming a value
    // its attribute doesn't hold: a buffer draco decoded but that isn't a mesh
    bool malformed = false;
    const std::size_t points = mesh->num_points();
    faces->resize(3 * std::size_t{mesh->num_faces()});
    for (std::uint32_t i = 0; i < mesh->num_faces(); i++) {
        const auto& face = mesh->face(draco::FaceIndex(i));
        for (std::size_t j = 0; j < 3; j++) {
            (*faces)[(3 * std::size_t{i}) + j] = face[j].value();
            malformed = malformed || face[j].value() >= points;
        }
    }
    attributes->resize(wanted.size());
    std::vector<const draco::PointAttribute*> found(wanted.size(), nullptr);
    // one copy when each point has its own value, else gather: either way no
    // value is read that the attribute doesn't hold
    for (std::size_t w = 0; w < wanted.size() && !malformed; w++) {
        const draco::PointAttribute* a = mesh->GetAttributeByUniqueId(wanted[w]);
        found[w] = a;
        if (a == nullptr) {
            continue;
        }
        // the values the attribute holds, and no more than its bytes do, the
        // buffer's stride and offset compared before they are narrowed
        const std::uint64_t bytes = a->buffer() == nullptr ? 0 : a->buffer()->data_size();
        const std::int64_t stride_given = a->byte_stride();
        const std::int64_t offset_given = a->byte_offset();
        if (!indexes(stride_given, bytes + 1) || stride_given == 0 || !indexes(offset_given, bytes + 1) ||
            std::uint64_t{points} * static_cast<std::uint64_t>(stride_given) >
                std::numeric_limits<std::size_t>::max()) {
            malformed = malformed || points > 0;
            continue;
        }
        const auto stride = static_cast<std::size_t>(stride_given);
        const auto offset = static_cast<std::size_t>(offset_given);
        const std::size_t held = std::min<std::size_t>(a->size(), (bytes - offset) / stride);
        std::vector<std::uint8_t>& values = (*attributes)[w].values;
        values.resize(points * stride);
        if (a->is_mapping_identity()) {
            malformed = malformed || held < points;
            std::copy_n(a->GetAddress(draco::AttributeValueIndex(0)), std::min(held, points) * stride, values.data());
            continue;
        }
        for (std::uint32_t p = 0; p < points; p++) {
            const draco::AttributeValueIndex at = a->mapped_index(draco::PointIndex(p));
            if (at.value() >= held) {
                malformed = true;
                break;
            }
            std::copy_n(a->GetAddress(at), stride, values.data() + (p * stride));
        }
    }
    if (malformed) {
        refuse("draco failed to decode: the buffer's faces or attributes name values it doesn't hold");
    }
    for (std::size_t w = 0; w < wanted.size(); w++) {
        const draco::PointAttribute* a = found[w];
        if (a == nullptr) {
            refuse("the buffer has no draco attribute " + std::to_string(wanted[w]));
        }
        const draco::DataType* type = std::find(std::begin(DATA_TYPES), std::end(DATA_TYPES), a->data_type());
        if (type == std::end(DATA_TYPES)) {
            refuse("draco attribute " + std::to_string(wanted[w]) + " is not a glTF component type");
        }
        DracoAttributeOutput& out = (*attributes)[w];
        out.component = static_cast<DracoComponent>(type - std::begin(DATA_TYPES));
        out.components = a->num_components();
        // the output is viewed as (points, c) of the type: only a tightly
        // packed attribute has exactly that many bytes, so refuse any other
        if (out.values.size() !=
            points * out.components * trimkit::DRACO_COMPONENT_BYTES[static_cast<std::size_t>(out.component)]) {
            refuse("draco attribute " + std::to_string(wanted[w]) + " is not tightly packed");
        }
    }
}
