// UV parameterization with jpcy/xatlas.
//
// xatlas only reads float32 positions and uint32 indices, so the input is
// converted first, then charting and packing run with the GIL released.
// The progress callback polls for Ctrl-C and cancels on one.

#include "common.hpp"

#include <xatlas.h>

#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace {

// xatlas reports progress from its worker threads and stops when we say so
bool progress(xatlas::ProgressCategory /*category*/, int /*percent*/, void* interrupt) {
    return !static_cast<trimkit::Interrupt*>(interrupt)->poll();
}

using Result = std::tuple<trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>, trimkit::Out<double>>;

Result parametrize(const trimkit::Array<double, -1, 3>& vertices_argument,
                   const trimkit::Array<std::int64_t, -1, 3>& faces_argument) {
    const trimkit::In<double, -1, 3> vertices = trimkit::read(vertices_argument, "vertices");
    const std::size_t n_vertices = vertices.shape(0);
    const std::vector<std::uint32_t> indices =
        trimkit::read_indices<std::uint32_t>(faces_argument, "faces", n_vertices);

    std::vector<std::int64_t> vertex_sources;
    std::vector<std::int64_t> out_faces;
    std::vector<double> uv;
    trimkit::Interrupt interrupt;
    xatlas::AddMeshError error = xatlas::AddMeshError::Success;
    // with no faces there is nothing to parametrize, and given no indices
    // xatlas would read the vertices as an unindexed triangle list instead
    if (!indices.empty()) {
        const nb::gil_scoped_release release;
        // the one unavoidable conversion: xatlas only reads float32 positions,
        // and its tolerances are absolute, so centered and scaled to an extent
        // near 1 (a 1 cm grid got every uv at one point, a sphere offset by 1e8
        // every uv at zero, and a 1e20 mesh hung). The uv come out the same, as
        // xatlas normalizes them to the atlas and the scale is a power of two
        double scale = 1.0;
        const std::vector<float> positions = trimkit::float32_positions(vertices.data(), n_vertices, &scale);
        xatlas::MeshDecl mesh;
        mesh.vertexPositionData = positions.data();
        mesh.vertexPositionStride = 3 * sizeof(float);
        mesh.vertexCount = static_cast<std::uint32_t>(n_vertices);
        mesh.indexData = indices.data();
        mesh.indexCount = static_cast<std::uint32_t>(indices.size());
        mesh.indexFormat = xatlas::IndexFormat::UInt32;

        xatlas::Atlas* atlas = xatlas::Create();
        // freed however this block ends, a std::bad_alloc included
        const std::unique_ptr<xatlas::Atlas, void (*)(xatlas::Atlas*)> owner(atlas, &xatlas::Destroy);
        xatlas::SetProgressCallback(atlas, &progress, &interrupt);
        error = xatlas::AddMesh(atlas, mesh);
        if (error == xatlas::AddMeshError::Success) {
            // cap charts at 4096 texels: a needle triangle (10^7 : 1) otherwise
            // overflows xatlas's packing bitmap, and ordinary charts are smaller
            xatlas::PackOptions pack;
            pack.maxChartSize = 4096;
            xatlas::Generate(atlas, xatlas::ChartOptions(), pack);
        }
        if (error == xatlas::AddMeshError::Success && !interrupt.interrupted() && atlas->meshCount == 1) {
            const xatlas::Mesh& out = atlas->meshes[0];
            // xatlas reports texels: normalize into the unit square as float64.
            // Not by sides rounded up to powers of two: xatlas sizes its atlas
            // to land near 1024, usually just over, so that left about two
            // thirds of the texture empty (36% used on average, measured)
            const double width = atlas->width > 0 ? atlas->width : 1.0;
            const double height = atlas->height > 0 ? atlas->height : 1.0;
            vertex_sources.resize(out.vertexCount);
            uv.resize(2 * std::size_t{out.vertexCount});
            for (std::size_t i = 0; i < out.vertexCount; i++) {
                vertex_sources[i] = out.vertexArray[i].xref;
                uv[2 * i] = static_cast<double>(out.vertexArray[i].uv[0]) / width;
                uv[(2 * i) + 1] = static_cast<double>(out.vertexArray[i].uv[1]) / height;
            }
            out_faces.assign(out.indexArray, out.indexArray + out.indexCount);
        }
    }
    if (interrupt.interrupted()) {
        throw nb::python_error();
    }
    if (error != xatlas::AddMeshError::Success) {
        trimkit::value_error(xatlas::StringForEnum(error));
    }

    const std::size_t vertex_count = vertex_sources.size();
    const std::size_t face_count = out_faces.size() / 3;
    return {trimkit::to_numpy(std::move(vertex_sources), {vertex_count}),
            trimkit::to_numpy(std::move(out_faces), {face_count, 3}),
            trimkit::to_numpy(std::move(uv), {vertex_count, 2})};
}

}  // namespace

void bind_xatlas(nb::module_& m) {
    m.def("parametrize", &parametrize, nb::arg("vertices"), nb::arg("faces"),
          R"(Cut a mesh into charts and pack them into a UV atlas.

Parameters
----------
vertices : (n, 3) float64
  Mesh vertices.
faces : (m, 3) int64
  Triangle indices into `vertices`.

Returns
-------
vertex_sources : (k,) int64
  The input vertex each output vertex came from.
faces : (m, 3) int64
  Triangles indexing the output vertices.
uv : (k, 2) float64
  Texture coordinates in the unit square.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> vertex_sources, faces, uv = trimkit.parametrize(triangle, faces)
>>> vertex_sources.tolist(), uv.shape
([0, 1, 2], (3, 2)))");
}
