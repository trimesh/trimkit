// Approximate convex decomposition with kmammou/v-hacd (VHACD.h).
// Python-free (trimkit.hpp).
//
// V-HACD runs synchronously on the calling thread. Its fan-outs (splitting
// hulls, costing merges) run on the extension's one pool through the hook
// vendor/patches/vhacd-parallel-for.patch adds, in place of the 8-thread pool
// it would start per call, and answer the same however the ranges fall. Its
// progress callback polls the Stop, cancelling the decomposition cleanly once
// it says to stop.

#include "trimkit.hpp"

#define ENABLE_VHACD_IMPLEMENTATION 1
#include <VHACD.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace {

// cancel the decomposition from V-HACD's progress updates once `stop` says to
class Progress final : public VHACD::IVHACD::IUserCallback {
public:
    Progress(VHACD::IVHACD* vhacd, const trimkit::parallel::Stop& stop) : vhacd_(vhacd), stop_(stop) {}

    void Update(double /*overall*/, double /*stage*/, const char* /*stage_name*/, const char* /*operation*/) override {
        if (trimkit::stopped(stop_)) {
            vhacd_->Cancel();
        }
    }

private:
    VHACD::IVHACD* vhacd_;
    trimkit::parallel::Stop stop_;
};

// V-HACD's fan-outs on the one pool: a body over [0, count), every item once,
// an exception a worker's range threw rethrown here, on V-HACD's thread
void parallel_for(void* /*unused*/, std::size_t count, void (*body)(void* context, std::size_t begin, std::size_t end),
                  void* context) {
    trimkit::run_ranges(count, body, context);
}

// releases a V-HACD instance however the decomposition ends
struct Release {
    void operator()(VHACD::IVHACD* vhacd) const { vhacd->Release(); }
};

}  // namespace

std::vector<trimkit::ConvexHullMesh> trimkit::decompose_mesh(const double* vertices, std::size_t vertex_count,
                                                             const std::vector<std::uint32_t>& triangles,
                                                             std::int64_t maximum_hulls, std::int64_t maximum_vertices,
                                                             std::int64_t resolution, double maximum_error,
                                                             const parallel::Stop& stop) {
    // refuse what V-HACD would misbehave on rather than check: each is a
    // uint32 to it, and its voxel grid multiplies the dimensions in uint32
    // and packs each axis into 10 bits, so past its documented maximum
    // resolution that overflows and crashes
    constexpr std::int64_t MOST = std::numeric_limits<std::uint32_t>::max();
    if (maximum_hulls < 1 || maximum_hulls > MOST) {
        refuse("maximum_hulls must be from 1 to 4294967295");
    }
    if (maximum_vertices < 4 || maximum_vertices > MOST) {
        refuse("maximum_vertices must be from 4, a tetrahedron, to 4294967295");
    }
    if (resolution < 1 || resolution > 64000000) {
        refuse("resolution must be from 1 to 64000000");
    }
    if (std::isnan(maximum_error) || maximum_error < 0.0 || maximum_error > 100.0) {
        refuse("maximum_error must be from 0 to 100");
    }
    std::vector<ConvexHullMesh> hulls;
    // a decomposition needs a volume: skip input V-HACD can't handle
    if (triangles.empty() || vertex_count <= 3) {
        return hulls;
    }
    // everything else stays V-HACD's own default
    VHACD::IVHACD::Parameters parameters;
    parameters.m_parallelFor = &parallel_for;
    parameters.m_maxConvexHulls = static_cast<std::uint32_t>(maximum_hulls);
    // its hull builder reads this limit as an int, negative from 2^31 up, which
    // it then takes for a tetrahedron: no hull has 2^31 vertices, so cap it there
    parameters.m_maxNumVerticesPerCH =
        static_cast<std::uint32_t>(std::min<std::int64_t>(maximum_vertices, std::numeric_limits<std::int32_t>::max()));
    parameters.m_resolution = static_cast<std::uint32_t>(resolution);
    parameters.m_minimumVolumePercentErrorAllowed = maximum_error;
    const std::unique_ptr<VHACD::IVHACD, Release> vhacd(VHACD::CreateVHACD());
    Progress progress(vhacd.get(), stop);
    parameters.m_callback = &progress;
    vhacd->Compute(vertices, static_cast<std::uint32_t>(vertex_count), triangles.data(),
                   static_cast<std::uint32_t>(triangles.size() / 3), parameters);
    if (stopped(stop)) {
        throw Stopped{};
    }
    hulls.resize(vhacd->GetNConvexHulls());
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
    return hulls;
}
