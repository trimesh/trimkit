// UV parameterization with jpcy/xatlas. Python-free (trimkit.hpp).
//
// xatlas only reads float32 positions and uint32 indices, so the input is
// converted first, then charting and packing run on the pool. The progress
// callback polls the Stop and cancels on it.

#include "trimkit.hpp"

#include <xatlas.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

// xatlas's task groups, on the one pool (vendor/patches/xatlas-one-pool.patch):
// its tasks are coarse (a mesh, a group of charts, a chart), so every worker
// is woken at once, and each thread running them has its own scratch index
std::uint32_t xatlas::TrimkitThreadSlots() { return trimkit::thread_slots(); }
std::uint32_t xatlas::TrimkitThreadIndex() { return trimkit::thread_index(); }
void xatlas::TrimkitRunTasks(std::uint32_t count, void (*run)(void* context, std::uint32_t begin, std::uint32_t end),
                             void* context) {
    struct Tasks {
        void (*run)(void*, std::uint32_t, std::uint32_t);
        void* context;
    } tasks{run, context};
    trimkit::run_ranges(
        count,
        [](void* each, std::size_t begin, std::size_t end) {
            const Tasks& group = *static_cast<const Tasks*>(each);
            group.run(group.context, static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(end));
        },
        &tasks);
}

namespace {

// xatlas reports progress from whichever thread runs a task, and stops when
// the Stop says so (only the calling thread's look can see Ctrl-C)
// cppcheck-suppress constParameterCallback ; xatlas's callback type, which takes a void*
bool progress(xatlas::ProgressCategory /*category*/, int /*percent*/, void* context) {
    const auto& stop = *static_cast<const trimkit::parallel::Stop*>(context);
    return !trimkit::stopped(stop);
}

}  // namespace

void trimkit::parametrize_mesh(const double* vertices, std::size_t vertex_count,
                               const std::vector<std::uint32_t>& indices, const parallel::Stop& stop,
                               std::vector<std::int64_t>* vertex_sources, std::vector<std::int64_t>* faces,
                               std::vector<double>* uv) {
    // with no faces there is nothing to parametrize, and given no indices
    // xatlas would read the vertices as an unindexed triangle list instead
    if (indices.empty()) {
        return;
    }
    // the one unavoidable conversion: xatlas only reads float32 positions,
    // and its tolerances are absolute, so centered and scaled to an extent
    // near 1 (a 1 cm grid got every uv at one point, a sphere offset by 1e8
    // every uv at zero, and a 1e20 mesh hung). The uv come out the same, as
    // xatlas normalizes them to the atlas and the scale is a power of two
    double scale = 1.0;
    const std::vector<float> positions = float32_positions(vertices, vertex_count, &scale);
    xatlas::MeshDecl mesh;
    mesh.vertexPositionData = positions.data();
    mesh.vertexPositionStride = 3 * sizeof(float);
    mesh.vertexCount = static_cast<std::uint32_t>(vertex_count);
    mesh.indexData = indices.data();
    mesh.indexCount = static_cast<std::uint32_t>(indices.size());
    mesh.indexFormat = xatlas::IndexFormat::UInt32;

    // xatlas prints its warnings (charts it couldn't parameterize, a run cut
    // short by a stop) to stdout through one global hook, and a library never
    // writes to its caller's terminal: the hook cleared, once, before any atlas
    static const bool QUIET = [] {
        xatlas::SetPrint(nullptr, false);
        return true;
    }();
    (void)QUIET;
    xatlas::Atlas* atlas = xatlas::Create();
    // freed however this block ends, a std::bad_alloc included
    const std::unique_ptr<xatlas::Atlas, void (*)(xatlas::Atlas*)> owner(atlas, &xatlas::Destroy);
    // xatlas takes a context it may write through: a copy of the Stop
    parallel::Stop polled = stop;
    xatlas::SetProgressCallback(atlas, &progress, &polled);
    const xatlas::AddMeshError error = xatlas::AddMesh(atlas, mesh);
    if (error == xatlas::AddMeshError::Success) {
        // cap charts at 4096 texels: a needle triangle (10^7 : 1) otherwise
        // overflows xatlas's packing bitmap, and ordinary charts are smaller
        xatlas::PackOptions pack;
        pack.maxChartSize = 4096;
        xatlas::Generate(atlas, xatlas::ChartOptions(), pack);
    }
    if (stopped(stop)) {
        throw Stopped{};
    }
    if (error != xatlas::AddMeshError::Success) {
        refuse(xatlas::StringForEnum(error));
    }
    // one mesh went in, so one comes out: anything else is xatlas failing,
    // not the input being wrong
    if (atlas->meshCount != 1) {
        throw std::runtime_error("xatlas produced no atlas for the mesh");
    }
    const xatlas::Mesh& out = atlas->meshes[0];
    // xatlas reports texels: normalize into the unit square as float64.
    // Not by sides rounded up to powers of two: xatlas sizes its atlas
    // to land near 1024, usually just over, so that left about two
    // thirds of the texture empty (36% used on average, measured)
    const double width = atlas->width > 0 ? atlas->width : 1.0;
    const double height = atlas->height > 0 ? atlas->height : 1.0;
    vertex_sources->resize(out.vertexCount);
    uv->resize(2 * std::size_t{out.vertexCount});
    for (std::size_t i = 0; i < out.vertexCount; i++) {
        (*vertex_sources)[i] = out.vertexArray[i].xref;
        (*uv)[2 * i] = static_cast<double>(out.vertexArray[i].uv[0]) / width;
        (*uv)[(2 * i) + 1] = static_cast<double>(out.vertexArray[i].uv[1]) / height;
    }
    faces->assign(out.indexArray, out.indexArray + out.indexCount);
}
