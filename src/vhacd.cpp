// Approximate convex decomposition with kmammou/v-hacd (VHACD.h).
//
// V-HACD runs synchronously on the calling thread with the GIL released. Its
// fan-outs (splitting hulls, costing merges) run on the extension's one pool
// through the hook vendor/patches/vhacd-parallel-for.patch adds, in place of
// the 8-thread pool it would start per call, and answer the same however the
// ranges fall. Its progress callback is where we poll for Ctrl-C, cancelling
// the decomposition cleanly if one arrives.

#include "common.hpp"

#define ENABLE_VHACD_IMPLEMENTATION 1
#include <VHACD.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
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

// V-HACD's fan-outs on the one pool: a body over [0, count), every item once,
// an exception a worker's range threw rethrown here, on V-HACD's thread
void parallel_for(void* /*unused*/, std::size_t count, void (*body)(void* context, std::size_t begin, std::size_t end),
                  void* context) {
    trimkit::run_ranges(count, body, context);
}

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

// one hull at exactly its size, its faces indexing its own vertices
struct Hull {
    std::vector<double> vertices;
    std::vector<std::int64_t> faces;
};

trimkit::MeshOuts decompose(const trimkit::Array<double, -1, 3>& vertices_argument,
                            const trimkit::Array<std::int64_t, -1, 3>& faces_argument,
                            trimkit::Number<std::uint32_t> maximum_hulls,
                            trimkit::Number<std::uint32_t> maximum_vertices, trimkit::Number<std::uint32_t> resolution,
                            trimkit::Number<double> maximum_error) {
    const trimkit::In<double, -1, 3> vertices = trimkit::read(vertices_argument, "vertices");
    check_parameters(maximum_hulls, maximum_vertices, resolution, maximum_error);
    // everything else stays V-HACD's own default
    VHACD::IVHACD::Parameters parameters;
    parameters.m_parallelFor = &parallel_for;
    parameters.m_maxConvexHulls = maximum_hulls;
    parameters.m_maxNumVerticesPerCH = maximum_vertices;
    parameters.m_resolution = resolution;
    parameters.m_minimumVolumePercentErrorAllowed = maximum_error;

    const std::size_t vertex_count = vertices.shape(0);
    const std::vector<std::uint32_t> triangles =
        trimkit::read_indices<std::uint32_t>(faces_argument, "faces", vertex_count);

    std::vector<Hull> hulls;
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
        VHACD::IVHACD::ConvexHull computed;
        for (std::uint32_t h = 0; h < hulls.size(); h++) {
            vhacd->GetConvexHull(h, computed);
            hulls[h].vertices.resize(3 * computed.m_points.size());
            for (std::size_t i = 0; i < computed.m_points.size(); i++) {
                hulls[h].vertices[(3 * i) + 0] = computed.m_points[i].mX;
                hulls[h].vertices[(3 * i) + 1] = computed.m_points[i].mY;
                hulls[h].vertices[(3 * i) + 2] = computed.m_points[i].mZ;
            }
            hulls[h].faces.resize(3 * computed.m_triangles.size());
            for (std::size_t i = 0; i < computed.m_triangles.size(); i++) {
                hulls[h].faces[(3 * i) + 0] = computed.m_triangles[i].mI0;
                hulls[h].faces[(3 * i) + 1] = computed.m_triangles[i].mI1;
                hulls[h].faces[(3 * i) + 2] = computed.m_triangles[i].mI2;
            }
        }
    }
    if (interrupted) {
        throw nb::python_error();
    }
    nb::list meshes;
    for (Hull& hull : hulls) {
        const std::size_t vertex_rows = hull.vertices.size() / 3;
        const std::size_t face_rows = hull.faces.size() / 3;
        meshes.append(nb::make_tuple(trimkit::to_numpy(std::move(hull.vertices), {vertex_rows, 3}),
                                     trimkit::to_numpy(std::move(hull.faces), {face_rows, 3})));
    }
    return meshes;
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
hulls : list of Mesh
  One closed convex (vertices, faces) per hull, as `RayScene` and
  `CollisionScene` take them: empty for input with no volume.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
>>> faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
...                   [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int64)
>>> hulls = trimkit.decompose(vertices, faces, maximum_hulls=1)
>>> vertices, faces = hulls[0]
>>> len(hulls), len(vertices), len(faces)
(1, 8, 12))");
}
