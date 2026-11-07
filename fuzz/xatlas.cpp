// Property: any small mesh parameterizes, within libFuzzer's timeout, into an
// atlas whose vertices map back to real input vertices and whose texture
// coordinates fall inside the atlas.
#include <fuzzer/FuzzedDataProvider.h>
#include <xatlas.h>

#include <cstdint>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider in(data, size);
    const auto n = in.ConsumeIntegralInRange<std::uint32_t>(3, 64);
    const auto m = in.ConsumeIntegralInRange<std::uint32_t>(1, 128);
    std::vector<float> points(3 * std::size_t{n});
    for (float& p : points) {
        p = in.ConsumeFloatingPointInRange<float>(-10.0F, 10.0F);
    }
    std::vector<std::uint32_t> indices(3 * std::size_t{m});
    if (in.ConsumeBool() && n >= 9) {
        // a grid's connectivity with fuzzed positions: random faces almost never
        // form a surface, and a surface is what reaches charting and packing
        const std::uint32_t side = 3 + (n - 9) / 16;
        indices.clear();
        for (std::uint32_t i = 0; i + 1 < side; i++) {
            for (std::uint32_t j = 0; j + 1 < side; j++) {
                const std::uint32_t a = (i * side) + j;
                indices.insert(indices.end(), {a, a + side, a + 1, a + 1, a + side, a + side + 1});
            }
        }
    } else {
        for (auto& i : indices) {
            i = in.ConsumeIntegralInRange<std::uint32_t>(0, n - 1);
        }
    }
    xatlas::MeshDecl mesh;
    mesh.vertexPositionData = points.data();
    mesh.vertexPositionStride = 12;
    mesh.vertexCount = n;
    mesh.indexData = indices.data();
    mesh.indexCount = static_cast<std::uint32_t>(indices.size());
    mesh.indexFormat = xatlas::IndexFormat::UInt32;
    xatlas::Atlas* atlas = xatlas::Create();
    if (xatlas::AddMesh(atlas, mesh) == xatlas::AddMeshError::Success) {
        // the options src/xatlas.cpp runs with; positions here are already
        // near the origin, which is all its centering ensures
        xatlas::PackOptions pack;
        pack.maxChartSize = 4096;
        xatlas::Generate(atlas, xatlas::ChartOptions(), pack);
        for (std::uint32_t k = 0; k < atlas->meshCount; k++) {
            const xatlas::Mesh& out = atlas->meshes[k];
            for (std::uint32_t v = 0; v < out.vertexCount; v++) {
                const auto& x = out.vertexArray[v];
                if (x.xref >= n || x.uv[0] < 0 || x.uv[1] < 0 || x.uv[0] > static_cast<float>(atlas->width) ||
                    x.uv[1] > static_cast<float>(atlas->height)) {
                    __builtin_trap();
                }
            }
        }
    }
    xatlas::Destroy(atlas);
    return 0;
}
