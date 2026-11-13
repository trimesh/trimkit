// Write a cube as draco buffers in both encodings, with and without extra
// attributes, as seeds for fuzz/draco.cpp: our binding only writes sequential
// buffers, while glTF files from Blender and most exporters are edgebreaker.
#include <draco/compression/expert_encode.h>

#include <cstdint>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        return 1;
    }
    const float corner[8][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    const std::uint32_t faces[12][3] = {{0, 1, 3}, {0, 3, 2}, {4, 6, 7}, {4, 7, 5}, {0, 4, 5}, {0, 5, 1},
                                        {2, 3, 7}, {2, 7, 6}, {0, 2, 6}, {0, 6, 4}, {1, 5, 7}, {1, 7, 3}};
    int written = 0;
    for (const auto method : {draco::MESH_SEQUENTIAL_ENCODING, draco::MESH_EDGEBREAKER_ENCODING}) {
        for (const int attributes : {1, 3}) {
            draco::Mesh mesh;
            mesh.set_num_points(8);
            for (const auto& f : faces) {
                mesh.AddFace({draco::PointIndex(f[0]), draco::PointIndex(f[1]), draco::PointIndex(f[2])});
            }
            // the corners double as normals and as colors, which only needs to be valid
            const draco::GeometryAttribute::Type types[3] = {
                draco::GeometryAttribute::POSITION, draco::GeometryAttribute::NORMAL, draco::GeometryAttribute::COLOR};
            for (int a = 0; a < attributes; a++) {
                draco::GeometryAttribute g;
                g.Init(types[a], nullptr, 3, draco::DT_FLOAT32, false, 12, 0);
                mesh.attribute(mesh.AddAttribute(g, true, 8))->buffer()->Update(corner, sizeof(corner));
            }
            draco::ExpertEncoder encoder(mesh);
            encoder.SetEncodingMethod(method);
            draco::EncoderBuffer out;
            if (encoder.EncodeToBuffer(&out).ok()) {
                std::ofstream(std::string(argv[1]) + "/cube" + std::to_string(written++), std::ios::binary)
                    .write(out.data(), static_cast<std::streamsize>(out.size()));
            }
        }
    }
    return written == 4 ? 0 : 1;
}
