// Quadric edge-collapse simplification with zeux/meshoptimizer.
//
// meshoptimizer reads float32 positions and uint32 indices, so the input is
// converted first (see `float32_positions`). Its simplifier only removes
// vertices and never moves them, so each output vertex is the caller's exact
// float64 value, with any per-vertex attribute carried over by the same
// index. It keeps no global state, so concurrent calls are
// safe with the GIL released for the whole call; it has no progress
// callback, so a call can't be interrupted (a million faces take ~1 s).

#include "common.hpp"

#include <meshoptimizer.h>

#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

using Simplified = std::tuple<trimkit::Out<double>, trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>>;

// Renumber the vertices the kept triangles use, in order of first use, so the
// output is compact and the same for the same input.
static void compact(const std::vector<std::uint32_t>& kept, std::size_t vertex_count,
                    std::vector<std::int64_t>* vertex_sources, std::vector<std::int64_t>* faces) {
    std::vector<std::int64_t> renumbered(vertex_count, -1);
    faces->resize(kept.size());
    for (std::size_t i = 0; i < kept.size(); i++) {
        const std::uint32_t source = kept[i];
        if (renumbered[source] < 0) {
            renumbered[source] = static_cast<std::int64_t>(vertex_sources->size());
            vertex_sources->push_back(source);
        }
        (*faces)[i] = renumbered[source];
    }
}

static Simplified simplify(const trimkit::Array<double, -1, 3>& vertices_argument,
                           const trimkit::Array<std::int64_t, -1, 3>& faces_argument,
                           trimkit::Number<std::int64_t> face_count, trimkit::Number<double> maximum_error,
                           trimkit::Flag lock_border) {
    const trimkit::In<double, -1, 3> vertices = trimkit::read(vertices_argument, "vertices");
    const std::size_t vertex_count = vertices.shape(0);
    const std::vector<std::uint32_t> indices =
        trimkit::read_indices<std::uint32_t>(faces_argument, "faces", vertex_count);
    if (face_count < 0) {
        trimkit::value_error("face_count must be at least 0");
    }
    if (!(maximum_error >= 0.0)) {
        trimkit::value_error("maximum_error must be at least 0");
    }
    std::vector<std::int64_t> vertex_sources;
    std::vector<std::int64_t> output_faces;
    std::vector<double> output_vertices;
    {
        // converting reads every value, so it runs without the GIL too
        const nb::gil_scoped_release release;
        double scale = 1.0;
        const std::vector<float> positions = trimkit::float32_positions(vertices.data(), vertex_count, &scale);
        // never more than the input, clamped before multiplying so it can't overflow
        const std::size_t target = 3 * std::min(static_cast<std::size_t>(face_count), indices.size() / 3);
        // in the scaled units meshoptimizer sees, exactly, as the scale is a power of two
        const double scaled_error = maximum_error * scale;
        const float target_error =
            scaled_error < static_cast<double>(FLT_MAX) ? static_cast<float>(scaled_error) : FLT_MAX;
        // absolute error, so `maximum_error` is a length rather than a fraction
        // of the extent; RegularizeLight measured as removing 90-95% of slivers
        // and every inverted face for the same error and time
        unsigned int options = meshopt_SimplifyErrorAbsolute | meshopt_SimplifyRegularizeLight;
        if (lock_border) {
            options |= meshopt_SimplifyLockBorder;
        }
        // the destination must hold the whole input, whatever the target
        std::vector<std::uint32_t> kept(indices.size());
        std::size_t kept_count = 0;
        if (!indices.empty()) {
            kept_count = meshopt_simplify(kept.data(), indices.data(), indices.size(), positions.data(), vertex_count,
                                          3 * sizeof(float), target, target_error, options, nullptr);
        }
        kept.resize(kept_count);
        compact(kept, vertex_count, &vertex_sources, &output_faces);
        // it only removes vertices, so each output vertex is exactly its source
        output_vertices.resize(3 * vertex_sources.size());
        for (std::size_t i = 0; i < vertex_sources.size(); i++) {
            std::copy_n(vertices.data() + (3 * vertex_sources[i]), 3, output_vertices.data() + (3 * i));
        }
    }
    const std::size_t source_count = vertex_sources.size();
    const std::size_t triangle_count = output_faces.size() / 3;
    return {trimkit::to_numpy(std::move(output_vertices), {source_count, 3}),
            trimkit::to_numpy(std::move(output_faces), {triangle_count, 3}),
            trimkit::to_numpy(std::move(vertex_sources), {source_count})};
}

void bind_meshoptimizer(nb::module_& m) {
    m.def("simplify", &simplify, nb::arg("vertices"), nb::arg("faces"), nb::arg("face_count"), nb::kw_only(),
          nb::arg("maximum_error") = std::numeric_limits<double>::infinity(), nb::arg("lock_border") = false,
          R"(Collapse edges by quadric error until at most `face_count` faces remain.

Parameters
----------
vertices : (n, 3) float64
  Mesh vertices, merged: edges only collapse across shared vertices, so an
  unmerged triangle soup comes back unchanged.
faces : (m, 3) int64
  Triangle indices into `vertices`.
face_count : int
  Stop once at most this many faces remain. Topology can stop it sooner,
  leaving more faces than asked for.
maximum_error : float
  Stop before the surface moves farther than this, in the units of
  `vertices`. The default stops only on `face_count`.
lock_border : bool
  Keep every vertex on an open boundary edge.

Returns
-------
vertices : (k, 3) float64
  The output vertices.
faces : (t, 3) int64
  Triangles indexing the output vertices.
vertex_sources : (k,) int64
  The input vertex each output vertex is, or -1 for one that is new, so
  per-vertex attributes carry over by index. Today none is new: the
  output vertices are exactly `vertices[vertex_sources]`.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> vertices, faces, vertex_sources = trimkit.simplify(vertices, faces, 4)
>>> len(faces), sorted(vertex_sources.tolist())
(4, [0, 1, 2, 3]))");
}
