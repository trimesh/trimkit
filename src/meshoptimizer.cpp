// Quadric edge-collapse simplification with zeux/meshoptimizer.
// Python-free (trimkit.hpp).
//
// meshoptimizer reads float32 positions and uint32 indices, so the input is
// converted first (see `float32_positions`). Its simplifier only removes
// vertices and never moves them, so each output vertex is the caller's exact
// float64 value, with any per-vertex attribute carried over by the same
// index. It keeps no global state, so concurrent calls are safe; it has no
// progress callback, so a call can't be interrupted (a million faces take ~1 s).

#include "trimkit.hpp"

#include <meshoptimizer.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <vector>

void trimkit::simplify_mesh(const double* vertices, std::size_t vertex_count, const std::vector<std::uint32_t>& indices,
                            std::int64_t face_count, double maximum_error, bool lock_border,
                            std::vector<double>* output_vertices, std::vector<std::int64_t>* output_faces,
                            std::vector<std::int64_t>* vertex_sources) {
    if (face_count < 0) {
        refuse("face_count must be at least 0");
    }
    if (!(maximum_error >= 0.0)) {
        refuse("maximum_error must be at least 0");
    }
    double scale = 1.0;
    const std::vector<float> positions = float32_positions(vertices, vertex_count, &scale);
    // never more than the input, clamped before multiplying so it can't overflow
    const std::size_t target = 3 * std::min(at_most(face_count), indices.size() / 3);
    // in the scaled units meshoptimizer sees, exactly, as the scale is a power of two
    const double scaled_error = maximum_error * scale;
    const float target_error = scaled_error < static_cast<double>(FLT_MAX) ? static_cast<float>(scaled_error) : FLT_MAX;
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
    // the vertices the kept triangles use, renumbered in order of first use,
    // so the output is compact and the same for the same input
    std::vector<std::int64_t> renumbered(vertex_count, -1);
    output_faces->resize(kept.size());
    for (std::size_t i = 0; i < kept.size(); i++) {
        const std::uint32_t source = kept[i];
        if (renumbered[source] < 0) {
            renumbered[source] = static_cast<std::int64_t>(vertex_sources->size());
            vertex_sources->push_back(source);
        }
        (*output_faces)[i] = renumbered[source];
    }
    // it only removes vertices, so each output vertex is exactly its source
    output_vertices->resize(3 * vertex_sources->size());
    for (std::size_t i = 0; i < vertex_sources->size(); i++) {
        std::copy_n(vertices + (3 * (*vertex_sources)[i]), 3, output_vertices->data() + (3 * i));
    }
}
