// Approximate convex decomposition with kmammou/v-hacd (VHACD.h).
//
// V-HACD runs synchronously on the calling thread with the GIL released, using
// its own thread pool, whose answers match running alone. Its progress callback is where
// we poll for Ctrl-C, cancelling the decomposition cleanly if one arrives.

#include "common.hpp"

#define ENABLE_VHACD_IMPLEMENTATION 1
#include <VHACD.h>

#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace {

// cancel the decomposition from V-HACD's progress updates on Ctrl-C
class Progress final : public VHACD::IVHACD::IUserCallback {
public:
    explicit Progress(VHACD::IVHACD* vhacd) : vhacd_(vhacd) {}
    trimkit::Interrupt interrupt;

    void Update(double /*overall*/, double /*stage*/, const char* /*stage_name*/, const char* /*operation*/) override {
        if (interrupt.poll()) {
            vhacd_->Cancel();
        }
    }

private:
    VHACD::IVHACD* vhacd_;
};

// refuse what V-HACD would misbehave on rather than check
void check_parameters(std::uint32_t maximum_hulls, std::uint32_t maximum_vertices, std::uint32_t resolution,
                      double maximum_error) {
    if (maximum_hulls < 1) {
        trimkit::value_error("maximum_hulls must be at least 1");
    }
    if (maximum_vertices < 4) {
        trimkit::value_error("maximum_vertices must be at least 4, a tetrahedron");
    }
    // its voxel grid multiplies the dimensions in uint32 and packs each axis
    // into 10 bits: past its documented maximum that overflows and crashes
    if (resolution < 1 || resolution > 64000000) {
        trimkit::value_error("resolution must be from 1 to 64000000");
    }
    if (std::isnan(maximum_error) || maximum_error < 0.0 || maximum_error > 100.0) {
        trimkit::value_error("maximum_error must be from 0 to 100");
    }
}

// releases a V-HACD instance however the decomposition ends
struct Release {
    void operator()(VHACD::IVHACD* vhacd) const { vhacd->Release(); }
};

using Hulls = std::tuple<trimkit::Out<double>, trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>,
                         trimkit::Out<std::int64_t>>;

Hulls decompose(const trimkit::Array<double, -1, 3>& vertices_argument,
                const trimkit::Array<std::int64_t, -1, 3>& faces_argument, trimkit::Number<std::uint32_t> maximum_hulls,
                trimkit::Number<std::uint32_t> maximum_vertices, trimkit::Number<std::uint32_t> resolution,
                trimkit::Number<double> maximum_error) {
    const trimkit::In<double, -1, 3> vertices = trimkit::read(vertices_argument, "vertices");
    check_parameters(maximum_hulls, maximum_vertices, resolution, maximum_error);
    // everything else stays V-HACD's own default, its thread pool included
    VHACD::IVHACD::Parameters parameters;
    parameters.m_maxConvexHulls = maximum_hulls;
    parameters.m_maxNumVerticesPerCH = maximum_vertices;
    parameters.m_resolution = resolution;
    parameters.m_minimumVolumePercentErrorAllowed = maximum_error;

    const std::size_t vertex_count = vertices.shape(0);
    const std::vector<std::uint32_t> triangles =
        trimkit::read_indices<std::uint32_t>(faces_argument, "faces", vertex_count);

    // the hulls, then the outputs at exactly their size: every hull's
    // vertices one after another, its faces indexing them per hull
    std::vector<VHACD::IVHACD::ConvexHull> hulls;
    bool interrupted = false;
    // a decomposition needs a volume: skip input V-HACD can't handle
    if (!triangles.empty() && vertex_count > 3) {
        const nb::gil_scoped_release release;
        const std::unique_ptr<VHACD::IVHACD, Release> vhacd(VHACD::CreateVHACD());
        Progress progress(vhacd.get());
        parameters.m_callback = &progress;
        vhacd->Compute(vertices.data(), static_cast<std::uint32_t>(vertex_count), triangles.data(),
                       static_cast<std::uint32_t>(triangles.size() / 3), parameters);
        interrupted = progress.interrupt.interrupted();
        hulls.resize(interrupted ? 0 : vhacd->GetNConvexHulls());
        for (std::uint32_t h = 0; h < hulls.size(); h++) {
            vhacd->GetConvexHull(h, hulls[h]);
        }
    }
    if (interrupted) {
        throw nb::python_error();
    }
    std::vector<std::int64_t> vertex_offsets(hulls.size() + 1, 0);
    std::vector<std::int64_t> face_offsets(hulls.size() + 1, 0);
    for (std::size_t h = 0; h < hulls.size(); h++) {
        vertex_offsets[h + 1] = vertex_offsets[h] + static_cast<std::int64_t>(hulls[h].m_points.size());
        face_offsets[h + 1] = face_offsets[h] + static_cast<std::int64_t>(hulls[h].m_triangles.size());
    }
    const auto point_count = static_cast<std::size_t>(vertex_offsets.back());
    const auto triangle_count = static_cast<std::size_t>(face_offsets.back());
    std::vector<double> points(3 * point_count);
    std::vector<std::int64_t> indices(3 * triangle_count);
    for (std::size_t h = 0; h < hulls.size(); h++) {
        double* point = points.data() + (3 * vertex_offsets[h]);
        for (const VHACD::Vertex& vertex : hulls[h].m_points) {
            point[0] = vertex.mX;
            point[1] = vertex.mY;
            point[2] = vertex.mZ;
            point += 3;
        }
        std::int64_t* face = indices.data() + (3 * face_offsets[h]);
        for (const VHACD::Triangle& triangle : hulls[h].m_triangles) {
            face[0] = triangle.mI0;
            face[1] = triangle.mI1;
            face[2] = triangle.mI2;
            face += 3;
        }
    }
    const std::size_t offset_count = vertex_offsets.size();
    return {trimkit::to_numpy(std::move(points), {point_count, 3}),
            trimkit::to_numpy(std::move(indices), {triangle_count, 3}),
            trimkit::to_numpy(std::move(vertex_offsets), {offset_count}),
            trimkit::to_numpy(std::move(face_offsets), {offset_count})};
}

}  // namespace

void bind_vhacd(nb::module_& m) {
    m.def("decompose", &decompose, nb::arg("vertices"), nb::arg("faces"), nb::kw_only(), nb::arg("maximum_hulls") = 64,
          nb::arg("maximum_vertices") = 64, nb::arg("resolution") = 400000, nb::arg("maximum_error") = 1.0,
          R"(Decompose a closed triangle mesh into approximately convex hulls.

Parameters
----------
vertices : (n, 3) float64
  Mesh vertices.
faces : (m, 3) int64
  Triangle indices into `vertices`, a closed surface: one with holes
  comes out hollow.
maximum_hulls : int
  The most hulls to return.
maximum_vertices : int
  The most vertices in each hull.
resolution : int
  How finely to sample the mesh, in voxels: raise it for thin features.
maximum_error : float
  Stop splitting a hull once its volume is within this percent of what it
  covers, from 0 to 100.

Returns
-------
vertices : (N, 3) float64
  Every hull's vertices, stacked.
faces : (F, 3) int64
  Every hull's triangles, indexing that hull's own vertices.
vertex_offsets : (h + 1,) int64
  Where each hull's vertices start in `vertices`.
face_offsets : (h + 1,) int64
  Where each hull's faces start in `faces`.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> cube = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
>>> faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
...                   [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int64)
>>> vertices, faces, vertex_offsets, face_offsets = trimkit.decompose(cube, faces, maximum_hulls=1)
>>> vertex_offsets.tolist(), face_offsets.tolist()
([0, 8], [0, 12]))");
}
