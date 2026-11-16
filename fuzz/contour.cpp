// Property: any voxels, as rows or a volume, weighted or not, at any
// threshold, are refused by `surface` or give a surface where every edge is
// in exactly two faces, wound opposite ways, with a positive volume when any
// voxel is inside and nothing when none is, every vertex from an input
// element that is inside, and the same mesh however the items are split into
// ranges. Rows laid end to end with finite weights at a valid threshold are
// never refused, and a volume is the same surface as its non-zero voxels
// given as rows.
#include "../src/contour.hpp"

#include <fuzzer/FuzzedDataProvider.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <utility>
#include <vector>

// the surface with the items taken `step` at a time, as threads would
static const char* surface_in_steps(const contour::VoxelInput& input, std::size_t step, contour::SurfaceMesh* mesh) {
    return contour::surface(
        input,
        [step](std::size_t count, const auto& body) {
            for (std::size_t begin = 0; begin < count; begin += step) {
                body(begin, begin + step < count ? begin + step : count);
            }
        },
        mesh);
}

template <typename T>
static bool same(const std::vector<contour::UnwrittenValue<T>>& a, const std::vector<contour::UnwrittenValue<T>>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// closed, wound outward when anything is inside, every source in range and
// inside by `inside_source`: traps otherwise
template <typename InsideSource>
static void check_mesh(const contour::SurfaceMesh& mesh, bool inside, bool exact, std::size_t sources,
                       const InsideSource& inside_source) {
    if (mesh.faces.size() % 3 != 0 || mesh.vertices.size() != 3 * mesh.vertex_sources.size()) {
        __builtin_trap();
    }
    if (mesh.faces.empty() == inside) {
        __builtin_trap();
    }
    for (const contour::UnwrittenValue<std::int64_t> raw : mesh.vertex_sources) {
        const std::int64_t source = raw.value;
        if (source < 0 || static_cast<std::size_t>(source) >= sources ||
            !inside_source(static_cast<std::size_t>(source))) {
            __builtin_trap();
        }
    }
    // each directed edge once, and its reverse once
    std::map<std::pair<std::int64_t, std::int64_t>, int> edges;
    double volume = 0.0;
    for (std::size_t f = 0; f < mesh.faces.size(); f += 3) {
        const double* corner[3];
        for (std::size_t k = 0; k < 3; k++) {
            const std::int64_t vertex = mesh.faces[f + k].value;
            if (vertex < 0 || static_cast<std::size_t>(vertex) >= mesh.vertex_sources.size()) {
                __builtin_trap();
            }
            corner[k] = &mesh.vertices[3 * static_cast<std::size_t>(vertex)].value;
            edges[{vertex, mesh.faces[f + ((k + 1) % 3)].value}]++;
        }
        volume += corner[0][0] * ((corner[1][1] * corner[2][2]) - (corner[1][2] * corner[2][1]));
        volume -= corner[0][1] * ((corner[1][0] * corner[2][2]) - (corner[1][2] * corner[2][0]));
        volume += corner[0][2] * ((corner[1][0] * corner[2][1]) - (corner[1][1] * corner[2][0]));
    }
    for (const auto& [edge, times] : edges) {
        const auto reverse = edges.find({edge.second, edge.first});
        if (times != 1 || reverse == edges.end()) {
            __builtin_trap();
        }
    }
    // the sum cancels badly out at the limit, and a crossing may sit on a
    // voxel's center, flattening it to nothing: only the topology is checked then
    if (inside && exact && !(volume > 0.0)) {
        __builtin_trap();
    }
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    FuzzedDataProvider provider(data, size);
    const auto form = static_cast<contour::VoxelForm>(provider.ConsumeIntegralInRange<int>(0, 3));
    const bool weighted = provider.ConsumeBool();
    // mostly a threshold a voxel can be on either side of; sometimes one
    // that must be refused
    double threshold = provider.ConsumeIntegralInRange<int>(0, 4) * 0.25;
    bool valid = true;
    if (provider.ConsumeIntegralInRange<int>(0, 15) == 0) {
        const double odd[] = {-0.25, -0.0, std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::infinity(), 2e30};
        threshold = odd[provider.ConsumeIntegralInRange<int>(0, 4)];
        valid = threshold >= 0.0 && threshold <= contour::WEIGHT_LIMIT;
    }
    // weights mostly on a grid of quarters, so some sit on the threshold;
    // sometimes past what is accepted
    bool odd_weight = false;
    const auto weight = [&]() {
        if (provider.ConsumeIntegralInRange<int>(0, 31) == 0) {
            odd_weight = true;
            const double odd[] = {std::numeric_limits<double>::quiet_NaN(), -std::numeric_limits<double>::infinity(),
                                  2e30, -1e30};
            const double chosen = odd[provider.ConsumeIntegralInRange<int>(0, 3)];
            valid = valid && std::abs(chosen) <= contour::WEIGHT_LIMIT;
            return chosen;
        }
        // from 0, as the provider's arithmetic wraps on a negative bound
        return (provider.ConsumeIntegralInRange<int>(0, 8) - 2) * 0.25;
    };
    contour::VoxelInput input;
    input.form = form;
    input.threshold = threshold;
    std::vector<std::int64_t> rows;
    std::vector<double> weights;
    std::vector<bool> source_inside;
    bool inside = false;
    bool extreme = false;
    bool anywhere = false;
    std::vector<std::uint8_t> occupancy;
    if (form == contour::VoxelForm::VOXEL_ROWS || form == contour::VoxelForm::RUN_ROWS) {
        const std::size_t columns = form == contour::VoxelForm::VOXEL_ROWS ? 3 : 4;
        // mostly runs end to end in a small block, where voxels meet in every
        // way; sometimes any rows of that block, which overlap; and sometimes
        // values out at the limit, which must be refused or survive the arithmetic
        const int kind = provider.ConsumeIntegralInRange<int>(0, 15);
        extreme = kind == 0;
        anywhere = kind == 1;
        const std::size_t count = provider.ConsumeIntegralInRange<std::size_t>(0, 48);
        // the first voxel free in each column of a 4 by 4 block
        std::int64_t top[16] = {};
        for (std::size_t i = 0; i < count; i++) {
            // short, so a wall of cells stays small; ranges start at 0, as the
            // provider's arithmetic wraps on a negative bound
            std::int64_t length = 1;
            if (columns == 4) {
                length = extreme ? provider.ConsumeIntegralInRange<std::int64_t>(0, 8) - 2
                                 : provider.ConsumeIntegralInRange<std::int64_t>(1, 5);
            }
            if (extreme || anywhere) {
                for (std::size_t k = 0; k < 3; k++) {
                    rows.push_back(extreme ? static_cast<std::int64_t>(provider.ConsumeIntegral<std::uint64_t>())
                                           : provider.ConsumeIntegralInRange<std::int64_t>(0, 6) - 3);
                }
            } else {
                const auto column = provider.ConsumeIntegralInRange<std::size_t>(0, 15);
                top[column] += provider.ConsumeIntegralInRange<std::int64_t>(0, 2);
                rows.insert(rows.end(), {static_cast<std::int64_t>(column % 4), static_cast<std::int64_t>(column / 4),
                                         top[column] - 8});
                top[column] += length;
            }
            if (columns == 4) {
                rows.push_back(length);
            }
            const double row_weight = weighted ? weight() : 1.0;
            if (weighted) {
                weights.push_back(row_weight);
            }
            source_inside.push_back(row_weight > threshold);
            inside = inside || row_weight > threshold;
        }
        input.rows = rows.data();
        input.row_count = count;
        input.row_weights = weighted ? weights.data() : nullptr;
    } else {
        for (std::size_t& extent : input.volume_shape) {
            extent = provider.ConsumeIntegralInRange<std::size_t>(0, 5);
        }
        const std::size_t count = input.volume_shape[0] * input.volume_shape[1] * input.volume_shape[2];
        for (std::size_t i = 0; i < count; i++) {
            if (form == contour::VoxelForm::OCCUPANCY_VOLUME) {
                occupancy.push_back(provider.ConsumeBool() ? 1 : 0);
                source_inside.push_back(occupancy.back() != 0 && 1.0 > threshold);
            } else {
                weights.push_back(weight());
                source_inside.push_back(weights.back() > threshold);
            }
            inside = inside || source_inside.back();
        }
        static_assert(sizeof(bool) == sizeof(std::uint8_t));
        input.occupancy = reinterpret_cast<const bool*>(occupancy.data());
        input.volume_weights = weights.data();
    }

    contour::SurfaceMesh mesh;
    const bool refused = surface_in_steps(input, 3, &mesh) != nullptr;
    if (refused != !valid && !extreme && !anywhere) {
        __builtin_trap();
    }
    if (refused) {
        return 0;
    }
    // the volume is exact unless out at the limit, or a weight on the
    // threshold or one of 1e30 puts a crossing on a voxel's center
    bool flattened = odd_weight;
    for (const double each : weights) {
        flattened = flattened || each == threshold;
    }
    check_mesh(mesh, inside, !extreme && !flattened, source_inside.size(),
               [&](std::size_t source) { return static_cast<bool>(source_inside[source]); });

    contour::SurfaceMesh again;
    if (surface_in_steps(input, 1 + (source_inside.size() % 7), &again) != nullptr ||
        !same(again.vertices, mesh.vertices) || !same(again.faces, mesh.faces) ||
        !same(again.vertex_sources, mesh.vertex_sources)) {
        __builtin_trap();
    }

    // a volume is its non-zero voxels as rows, in the same order
    if (form == contour::VoxelForm::OCCUPANCY_VOLUME || form == contour::VoxelForm::WEIGHT_VOLUME) {
        const std::size_t size_y = input.volume_shape[1];
        const std::size_t size_z = input.volume_shape[2];
        std::vector<std::int64_t> voxel_rows;
        std::vector<double> voxel_weights;
        std::vector<std::int64_t> flat;
        for (std::size_t i = 0; i < source_inside.size(); i++) {
            const double each = form == contour::VoxelForm::OCCUPANCY_VOLUME ? occupancy[i] : weights[i];
            if (each == 0.0) {
                continue;
            }
            voxel_rows.insert(voxel_rows.end(), {static_cast<std::int64_t>(i / (size_y * size_z)),
                                                 static_cast<std::int64_t>((i / size_z) % size_y),
                                                 static_cast<std::int64_t>(i % size_z)});
            voxel_weights.push_back(each);
            flat.push_back(static_cast<std::int64_t>(i));
        }
        contour::VoxelInput as_rows;
        as_rows.form = contour::VoxelForm::VOXEL_ROWS;
        as_rows.rows = voxel_rows.data();
        as_rows.row_count = voxel_weights.size();
        as_rows.row_weights = voxel_weights.data();
        as_rows.threshold = threshold;
        contour::SurfaceMesh from_rows;
        if (surface_in_steps(as_rows, 2, &from_rows) != nullptr || !same(from_rows.vertices, mesh.vertices) ||
            !same(from_rows.faces, mesh.faces) || from_rows.vertex_sources.size() != mesh.vertex_sources.size()) {
            __builtin_trap();
        }
        for (std::size_t i = 0; i < mesh.vertex_sources.size(); i++) {
            if (flat[static_cast<std::size_t>(from_rows.vertex_sources[i].value)] != mesh.vertex_sources[i].value) {
                __builtin_trap();
            }
        }
    }
    return 0;
}
