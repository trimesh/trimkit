// glTF `KHR_draco_mesh_compression` with google/draco, and nothing else.
//
// Both functions speak the extension's own vocabulary: attributes are passed
// in order with a `DracoKind` for each, and identified in the buffer by the draco
// unique id stored in the extension JSON. Only glTF accessor component types
// are accepted.
//
// Decoding has no size limit: draco sizes its allocations from counts the
// buffer declares, and a valid buffer of a few bytes can really decode to
// millions of points (rANS codes a near-certain symbol in almost no bits). What
// doesn't fit in the memory available is a MemoryError (src/trimkit.cpp), not the
// kernel's SIGKILL.

#include "common.hpp"

#include <draco/compression/decode.h>
#include <draco/compression/expert_encode.h>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

// the component types a glTF accessor may have
struct ComponentType {
    draco::DataType draco = draco::DT_INVALID;
    nb::dlpack::dtype numpy;
};

constexpr ComponentType COMPONENT_TYPES[] = {
    {draco::DT_INT8, nb::dtype<std::int8_t>()},   {draco::DT_UINT8, nb::dtype<std::uint8_t>()},
    {draco::DT_INT16, nb::dtype<std::int16_t>()}, {draco::DT_UINT16, nb::dtype<std::uint16_t>()},
    {draco::DT_INT32, nb::dtype<std::int32_t>()}, {draco::DT_UINT32, nb::dtype<std::uint32_t>()},
    {draco::DT_FLOAT32, nb::dtype<float>()},
};

// the glTF component type matching a numpy dtype or a draco data type
template <typename Match>
const ComponentType* component(Match match) {
    for (const ComponentType& c : COMPONENT_TYPES) {
        if (match(c)) {
            return &c;
        }
    }
    return nullptr;
}

// how hard draco tries, which trades export time for size and is not lossy
constexpr int COMPRESSION_LEVEL = 6;

// The attribute kinds `draco_encode` accepts: each one's index here is its
// `DracoKind` value, named here, the one table both come from
enum class DracoKind : std::uint8_t {};
struct NamedKind {
    const char* name;
    draco::GeometryAttribute::Type type;
    const char* doc;
};
constexpr NamedKind KINDS[] = {
    {"POSITION", draco::GeometryAttribute::POSITION, "Vertex positions: float ones quantized to 14 bits by default."},
    {"NORMAL", draco::GeometryAttribute::NORMAL, "Vertex normals: float ones quantized to 10 bits by default."},
    {"TEXCOORD", draco::GeometryAttribute::TEX_COORD,
     "Texture coordinates: float ones quantized to 12 bits by default."},
    {"COLOR", draco::GeometryAttribute::COLOR, "Vertex colors: float ones quantized to 10 bits by default."},
    {"GENERIC", draco::GeometryAttribute::GENERIC, "Anything else, stored losslessly by default."}};

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

struct Attribute {
    trimkit::AnyIn array;
    draco::GeometryAttribute::Type kind = draco::GeometryAttribute::GENERIC;
    draco::DataType type = draco::DT_INVALID;
    std::size_t components = 1;
};

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

// draco reads and writes its little-endian format by copying raw bytes, with
// no byte order of its own: on a big-endian machine every value would come
// out swapped, so it refuses there before reading anything
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
constexpr bool BIG_ENDIAN_MACHINE = true;
#else
constexpr bool BIG_ENDIAN_MACHINE = false;
#endif

void refuse_big_endian() {
    if (BIG_ENDIAN_MACHINE) {
        trimkit::not_implemented_error(
            "draco reads and writes only little-endian data, and this machine is big-endian");
    }
}

nb::bytes draco_encode(const trimkit::Array<std::int64_t, -1, 3>& faces_argument,
                       const trimkit::AnyArguments& attributes, const trimkit::Array<std::int64_t, -1>& kinds_argument,
                       const std::optional<trimkit::Number<std::int64_t>>& bits_argument) {
    refuse_big_endian();
    if (bits_argument && (*bits_argument < 0 || *bits_argument > MOST_QUANTIZATION_BITS)) {
        trimkit::value_error("quantization_bits must be from 0, lossless, to " +
                             std::to_string(MOST_QUANTIZATION_BITS));
    }
    const trimkit::In<std::int64_t, -1> kind_numbers = trimkit::read(kinds_argument, "kinds");
    // validate every attribute while we hold the GIL
    std::vector<Attribute> inputs(nb::len(attributes));
    for (std::size_t i = 0; i < inputs.size(); i++) {
        const std::string name = "attributes[" + std::to_string(i) + "]";
        Attribute& a = inputs[i];
        a.array = trimkit::read_any(attributes[i], name);
        if (const auto* c = component([&](const ComponentType& t) { return t.numpy == a.array.dtype(); })) {
            a.type = c->draco;
        } else {
            trimkit::type_error(name + " is not a glTF component type");
        }
        if (a.array.ndim() == 2) {
            a.components = a.array.shape(1);
        }
        if (a.array.ndim() < 1 || a.array.ndim() > 2 || a.components < 1 || a.components > 16 ||
            a.array.shape(0) != inputs[0].array.shape(0)) {
            trimkit::value_error(name + " must be (n,) or (n, c<=16) like the rest");
        }
    }
    if (kind_numbers.shape(0) != inputs.size()) {
        trimkit::value_error("kinds must have one entry per attribute");
    }
    for (std::size_t i = 0; i < inputs.size(); i++) {
        const std::int64_t k = kind_numbers(i);
        if (k < 0 || static_cast<std::size_t>(k) >= std::size(KINDS)) {
            trimkit::value_error("kinds[" + std::to_string(i) + "] is not a DracoKind");
        }
        inputs[i].kind = KINDS[static_cast<std::size_t>(k)].type;
    }
    if (inputs.empty()) {
        trimkit::value_error("at least one attribute is required");
    }
    const std::size_t count = inputs[0].array.shape(0);
    if (count == 0) {
        trimkit::value_error("draco can't encode a primitive without vertices");
    }
    const std::vector<std::uint32_t> triangles = trimkit::read_indices<std::uint32_t>(faces_argument, "faces", count);
    const std::size_t face_count = triangles.size() / 3;

    draco::EncoderBuffer encoded;
    draco::Status status;
    {
        const nb::gil_scoped_release release;
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
        for (std::size_t i = 0; i < inputs.size() && status.ok(); i++) {
            const Attribute& a = inputs[i];
            const std::size_t values = count * a.components;
            if ((a.type == draco::DT_INT32 && !range_fits<std::int32_t>(a.array.data(), values)) ||
                (a.type == draco::DT_UINT32 && !range_fits<std::uint32_t>(a.array.data(), values))) {
                status = draco::Status(draco::Status::DRACO_ERROR,
                                       "attributes[" + std::to_string(i) + "] has a range of integers over 2**31 - 2");
                break;
            }
            const auto stride = static_cast<std::int64_t>(a.components * a.array.itemsize());
            draco::GeometryAttribute g;
            g.Init(a.kind, nullptr, static_cast<std::uint8_t>(a.components), a.type,
                   a.type != draco::DT_FLOAT32 && a.kind != draco::GeometryAttribute::GENERIC, stride, 0);
            const int id = mesh.AddAttribute(g, true, static_cast<std::uint32_t>(count));
            // the contract callers rely on: draco attribute i is attributes[i]
            if (id < 0 || mesh.attribute(id)->unique_id() != i) {
                status = draco::Status(draco::Status::DRACO_ERROR, "could not add attribute " + std::to_string(i));
                break;
            }
            mesh.attribute(id)->buffer()->Update(a.array.data(), stride * static_cast<std::int64_t>(count));
            const int bits = bits_argument ? static_cast<int>(*bits_argument) : quantization_bits(a.kind);
            if (a.type == draco::DT_FLOAT32 && bits > 0) {
                encoder.SetAttributeQuantization(id, bits);
            }
        }
        encoder.SetSpeedOptions(10 - COMPRESSION_LEVEL, 10 - COMPRESSION_LEVEL);
        encoder.SetEncodingMethod(draco::MESH_SEQUENTIAL_ENCODING);
        if (status.ok()) {
            status = encoder.EncodeToBuffer(&encoded);
        }
    }
    if (!status.ok()) {
        trimkit::value_error("draco failed to encode: " + status.error_msg_string());
    }
    return nb::bytes(encoded.data(), encoded.size());
}

std::tuple<trimkit::Out<std::int64_t>, trimkit::AnyOuts> draco_decode(
    const trimkit::Array<std::uint8_t, -1>& buffer_argument,
    const trimkit::Array<std::int64_t, -1>& attributes_argument) {
    refuse_big_endian();
    // bytes and memoryviews are uint8 buffers already, viewed without a copy
    const trimkit::In<std::uint8_t, -1> buffer = trimkit::read(buffer_argument, "buffer");
    const trimkit::In<std::int64_t, -1> draco_attributes = trimkit::read(attributes_argument, "draco_attributes");
    std::vector<std::uint32_t> wanted(draco_attributes.shape(0));
    for (std::size_t i = 0; i < wanted.size(); i++) {
        const std::int64_t id = draco_attributes(i);
        if (id < 0 || id > std::numeric_limits<std::uint32_t>::max()) {
            trimkit::value_error("draco_attributes[" + std::to_string(i) + "] is not a draco attribute id");
        }
        wanted[i] = static_cast<std::uint32_t>(id);
    }

    std::unique_ptr<draco::Mesh> mesh;
    draco::Status status;
    // a face naming a point the mesh doesn't have, or a point naming a value
    // its attribute doesn't hold: a buffer draco decoded but that isn't a mesh
    bool malformed = false;
    std::vector<std::int64_t> faces;
    std::vector<std::vector<std::uint8_t>> values(wanted.size());
    std::vector<const draco::PointAttribute*> found(wanted.size(), nullptr);
    {
        const nb::gil_scoped_release release;
        draco::DecoderBuffer input;
        input.Init(reinterpret_cast<const char*>(buffer.data()), buffer.size());
        draco::Decoder decoder;
        draco::StatusOr<std::unique_ptr<draco::Mesh>> decoded = decoder.DecodeMeshFromBuffer(&input);
        status = decoded.status();
        if (status.ok()) {
            mesh = std::move(decoded).value();
            const std::size_t points = mesh->num_points();
            faces.resize(3 * std::size_t{mesh->num_faces()});
            for (std::uint32_t i = 0; i < mesh->num_faces(); i++) {
                const auto& face = mesh->face(draco::FaceIndex(i));
                for (std::size_t j = 0; j < 3; j++) {
                    faces[(3 * std::size_t{i}) + j] = face[j].value();
                    malformed = malformed || face[j].value() >= points;
                }
            }
            // one copy when each point has its own value, else gather: either
            // way no value is read that the attribute doesn't hold
            for (std::size_t w = 0; w < wanted.size(); w++) {
                const draco::PointAttribute* a = mesh->GetAttributeByUniqueId(wanted[w]);
                found[w] = a;
                if (a == nullptr) {
                    continue;
                }
                const auto stride = static_cast<std::size_t>(a->byte_stride());
                // the values the attribute holds, and no more than its bytes do
                const auto offset = static_cast<std::size_t>(a->byte_offset());
                const std::size_t bytes = a->buffer() == nullptr ? 0 : a->buffer()->data_size();
                const std::size_t held =
                    stride == 0 || offset > bytes ? 0 : std::min(a->size(), (bytes - offset) / stride);
                values[w].resize(points * stride);
                if (a->is_mapping_identity()) {
                    malformed = malformed || held < points;
                    std::copy_n(a->GetAddress(draco::AttributeValueIndex(0)), std::min(held, points) * stride,
                                values[w].data());
                    continue;
                }
                for (std::uint32_t p = 0; p < points; p++) {
                    const draco::AttributeValueIndex at = a->mapped_index(draco::PointIndex(p));
                    if (at.value() >= held) {
                        malformed = true;
                        break;
                    }
                    std::copy_n(a->GetAddress(at), stride, values[w].data() + (p * stride));
                }
            }
        }
    }
    if (!status.ok()) {
        trimkit::value_error("draco failed to decode: " + status.error_msg_string());
    }
    if (malformed) {
        trimkit::value_error("draco failed to decode: the buffer's faces or attributes name values it doesn't hold");
    }

    const std::size_t points = mesh->num_points();
    nb::list arrays;
    for (std::size_t w = 0; w < wanted.size(); w++) {
        const draco::PointAttribute* a = found[w];
        if (a == nullptr) {
            trimkit::value_error("the buffer has no draco attribute " + std::to_string(wanted[w]));
        }
        const auto* type = component([&](const ComponentType& t) { return t.draco == a->data_type(); });
        if (type == nullptr) {
            trimkit::value_error("draco attribute " + std::to_string(wanted[w]) + " is not a glTF component type");
        }
        const std::size_t c = a->num_components();
        // the output is viewed as (points, c) of the type: only a tightly
        // packed attribute has exactly that many bytes, so refuse any other
        if (values[w].size() != points * c * type->numpy.bits / 8) {
            trimkit::value_error("draco attribute " + std::to_string(wanted[w]) + " is not tightly packed");
        }
        arrays.append(c == 1 ? trimkit::to_numpy(std::move(values[w]), {points}, type->numpy)
                             : trimkit::to_numpy(std::move(values[w]), {points, c}, type->numpy));
    }
    return {trimkit::to_numpy_rows(std::move(faces), {3}), nb::borrow<trimkit::AnyOuts>(nb::tuple(arrays))};
}

}  // namespace

void bind_draco(nb::module_& m) {
    trimkit::bind_enum<DracoKind>(m, "DracoKind", "What a draco attribute holds, as `draco_encode` takes it.", KINDS);
    m.def("draco_encode", &draco_encode, nb::arg("faces"), nb::arg("attributes"), nb::arg("kinds"), nb::kw_only(),
          nb::arg("quantization_bits").none() = nb::none(),
          R"(Compress a glTF triangle primitive, keeping vertex order.

Parameters
----------
faces : (f, 3) int64
  Triangles indexing the attribute rows.
attributes : sequence of (n,) or (n, c) arrays
  One array per glTF attribute, each a glTF component type. Draco
  attribute i is attributes[i], which is what the extension's
  `attributes` JSON maps each glTF name to. Integers must fit int32, with
  `max - min` at most `2**31 - 2`.
kinds : (k,) int64
  Each attribute's `DracoKind`: POSITION, NORMAL, TEXCOORD, COLOR or
  GENERIC for anything else.
quantization_bits : int, optional
  What every float attribute is rounded onto, across its extent: a position
  moves at most `extent * 2**-(bits + 1)`, so this is how much precision
  survives. 0 stores them losslessly, up to 24, where a step is float32's
  own precision at the top of the extent. By default each kind's own:
  POSITION 14, NORMAL 10, TEXCOORD 12 and COLOR 10 bits, GENERIC lossless.
  Other attributes are never rounded.

Returns
-------
buffer : bytes
  The draco buffer for the extension's `bufferView`.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> kinds = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
>>> buffer = trimkit.exchange.draco_encode(faces, [vertices.astype(np.float32)], kinds)
>>> type(buffer).__name__
'bytes')");
    m.def("draco_decode", &draco_decode, nb::arg("buffer"), nb::arg("draco_attributes"),
          R"(Decompress a glTF draco buffer, decoding it once.

Parameters
----------
buffer : bytes, memoryview or (n,) uint8
  The contents of the extension's `bufferView`.
draco_attributes : (k,) int64
  The draco attribute numbers to return, from the extension's
  `attributes` JSON, in the order wanted.

Returns
-------
faces : (f, 3) int64
  Triangles indexing the attribute rows.
attributes : tuple of (n,) or (n, c) arrays
  One per requested draco attribute, in the requested order
  and stored component type.

Notes
-----
It has no size limit: a buffer of a few bytes can really hold
millions of points. What it declares beyond the memory available
raises MemoryError.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> kinds = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
>>> buffer = np.frombuffer(trimkit.exchange.draco_encode(faces, [vertices.astype(np.float32)], kinds), np.uint8)
>>> faces, attributes = trimkit.exchange.draco_decode(buffer, np.array([0], dtype=np.int64))
>>> faces.tolist(), attributes[0].tolist()
([[0, 1, 2]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))");
}
