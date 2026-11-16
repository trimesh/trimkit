// The surface of voxels with src/contour.hpp, which is ours.
//
// Rows are copied into the header's own records and each copy checked, so
// nothing the caller does to the array during the call can reach the sweep;
// a volume is read in place, a column at a time, and each value checked as it
// is read. It keeps no global state and runs through `trimkit::with_pool`:
// without the GIL, reading the voxels and every pass over the columns of
// cells on the pool, where Ctrl-C stops them between ranges.

#include "common.hpp"

#include "contour.hpp"

#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Contoured = std::tuple<trimkit::Out<double>, trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>>;

static Contoured contour_voxels(const trimkit::VoxelsArgument& voxels_argument,
                                const trimkit::Array<double, -1>& weights_argument,
                                trimkit::Number<double> threshold_argument) {
    // what each view holds stays alive through the call
    const trimkit::Voxels voxels = trimkit::read_voxels(voxels_argument, "voxels");
    contour::VoxelInput input;
    // checked by the header, with the rest of the input
    input.threshold = threshold_argument;
    trimkit::In<double, -1> weights;
    if (voxels.is_volume) {
        if (!weights_argument.is_none()) {
            trimkit::value_error("weights are for rows of voxels: a volume holds its own");
        }
        input.form = voxels.is_occupancy ? contour::VoxelForm::OCCUPANCY_VOLUME : contour::VoxelForm::WEIGHT_VOLUME;
        input.occupancy = voxels.occupancy.data();
        input.volume_weights = voxels.weights.data();
        for (std::size_t k = 0; k < 3; k++) {
            input.volume_shape[k] = voxels.is_occupancy ? voxels.occupancy.shape(k) : voxels.weights.shape(k);
        }
    } else {
        const std::size_t columns = voxels.rows.shape(1);
        if (columns != 3 && columns != 4) {
            trimkit::value_error("voxels must be (n, 3) or (n, 4) rows, not (n, " + std::to_string(columns) + ")");
        }
        input.form = columns == 3 ? contour::VoxelForm::VOXEL_ROWS : contour::VoxelForm::RUN_ROWS;
        input.rows = voxels.rows.data();
        input.row_count = voxels.rows.shape(0);
        if (!weights_argument.is_none()) {
            weights = trimkit::read(weights_argument, "weights");
            if (weights.shape(0) != input.row_count) {
                trimkit::value_error("weights must be one per row: " + std::to_string(weights.shape(0)) + " for " +
                                     std::to_string(input.row_count) + " rows");
            }
            input.row_weights = weights.data();
        }
    }

    contour::SurfaceMesh mesh;
    const char* problem =
        trimkit::with_pool([&](const auto& parallel) { return contour::surface(input, parallel, &mesh); });
    if (problem != nullptr) {
        trimkit::value_error(problem);
    }
    const std::size_t vertex_count = mesh.vertex_sources.size();
    return {trimkit::to_numpy_rows<double>(std::move(mesh.vertices), {3}),
            trimkit::to_numpy_rows<std::int64_t>(std::move(mesh.faces), {3}),
            trimkit::to_numpy<std::int64_t>(std::move(mesh.vertex_sources), {vertex_count})};
}

void bind_contour(nb::module_& m) {
    m.def("contour", &contour_voxels, nb::arg("voxels"), nb::kw_only(), nb::arg("weights").none() = nb::none(),
          nb::arg("threshold") = 0.5,
          R"(The closed surface of a set of voxels, as rows or as a volume.

A voxel is an integer index and is centered on it. It is inside when it
weighs more than `threshold`, and space no voxel covers weighs 0, so the
surface is closed whatever the input: every edge is in exactly two faces.
The surface passes between each inside voxel and each outside voxel beside
it, where the weight interpolated between them is `threshold`. Inside
voxels that share only an edge or a corner are kept apart.

Parameters
----------
voxels : (n, 3) or (n, 4) int64, or (X, Y, Z) bool or float64
  Rows of a voxel index `x, y, z`, with four columns followed by a length
  of at least 1: that many voxels extending along +Z. Rows can be in any
  order, and no two may hold the same voxel. Or a volume, whose element
  `[x, y, z]` is voxel `x, y, z`: a bool volume weighs 1 where it is true,
  and a float64 volume holds each voxel's weight.
weights : (n,) float64, optional
  For rows only: the weight of every voxel in each row. The default is 1,
  which with the default `threshold` puts each crossing halfway.
threshold : float
  A voxel is inside when it weighs more than this, at least 0 so that
  empty space is outside.

Returns
-------
vertices : (k, 3) float64
  One vertex per crossing, in index space.
faces : (m, 3) int64
  Triangles indexing `vertices`, wound counter-clockwise seen from outside.
vertex_sources : (k,) int64
  The input element holding the inside voxel of each vertex's crossing:
  its row, or its index into the flattened volume, so per-voxel attributes
  carry over by index.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> runs = np.array([[0, 0, 0, 1]])
>>> vertices, faces, vertex_sources = trimkit.contour(runs)
>>> len(vertices), len(faces), vertex_sources.tolist()
(6, 8, [0, 0, 0, 0, 0, 0])
>>> volume = np.zeros((3, 3, 3), dtype=bool)
>>> volume[1, 1, 1] = True
>>> vertices, faces, vertex_sources = trimkit.contour(volume)
>>> len(faces), set(vertex_sources.tolist())
(8, {13}))");
}
