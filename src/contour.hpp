// contour: the closed surface of a set of voxels, by marching cubes over
// run-length voxels.
//
// Header-only, written as plain C-style C++17: structs, functions and loops,
// depending only on the standard library. One door, `surface`: everything
// else is in `detail`.
//
// - A voxel is an integer index (x, y, z) and is centered on it. Each has a
//   weight, space no voxel covers weighs 0, and a voxel is inside when it
//   weighs more than the threshold. The threshold is at least 0, so empty
//   space is always outside and the surface is always closed.
// - Voxels arrive in one of the forms `VoxelForm` names and are read into
//   runs: a voxel and a length of voxels extending along +Z, one weight for
//   all of them, sorted by column and then along Z, never overlapping.
// - A cell is the cube between eight neighboring voxel centers. Wherever an
//   edge of a cell joins an inside voxel to an outside one the surface
//   crosses it, at the point where the weight interpolated along the edge
//   is the threshold: halfway for weights of 0 and 1 and a threshold of a half.
// - How the crossings of one cell join into triangles depends only on which
//   of its eight corners are inside, so it is a table of 256 cases. The
//   table is computed here, at compile time, rather than copied: on each
//   face of the cube every group of adjacent inside corners is cut off by
//   one line between two crossings, the lines chain into loops, and each
//   loop is filled with a fan. A face's lines depend only on that face's
//   four corners, which the two cells sharing it agree on, so the surface is
//   closed whatever the input: every edge is in exactly two triangles.
//   Two inside voxels that share only an edge or a corner are separated.
// - Cells whose corners are all inside or all outside are skipped a whole
//   run at a time, so the work follows the surface, not the volume. A weight
//   volume's voxels whose six neighbors are all on their own side of the
//   threshold touch no crossing, so their weights are never interpolated:
//   they are read as nothing when outside and as one weight when inside,
//   and a field costs its shell too.
// - The columns of cells are independent items, mapped over by whatever runs
//   ranges in parallel for the caller, and so is reading the input.
// - The work is O(input + output), plus a sort of rows not given in order:
//   every loop is bounded by one of them. Any input returns: what can't be
//   meshed is refused with the reason, and a surface too large to hold is
//   std::bad_alloc.
//
// Every function is `static`: each file that includes this header gets its
// own private copy, so two different versions can never collide.
//
// Copyright (c) 2026 Michael Dawson-Haggerty, MIT License.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace contour {

// voxel indices stay inside this: a half-integer below it is exact as a float64
constexpr std::int64_t INDEX_LIMIT = std::int64_t{1} << (std::numeric_limits<double>::digits - 1);
// weights and the threshold stay inside this, so no difference of two overflows
constexpr double WEIGHT_LIMIT = 1e30;

// How the voxels are given: rows of int64, or a dense volume in C order whose
// element (x, y, z) is voxel (x, y, z)
enum class VoxelForm : std::uint8_t {
    VOXEL_ROWS,        // (n, 3): one voxel x, y, z per row
    RUN_ROWS,          // (n, 4): x, y, z and a length of at least 1 along +Z
    OCCUPANCY_VOLUME,  // (X, Y, Z) bool: true weighs 1 and false 0
    WEIGHT_VOLUME,     // (X, Y, Z) float64: each voxel's weight
};

// The voxels to mesh, as `form` says: `rows` and `row_weights` (one per row,
// or nullptr for all ones) for the row forms, or `volume_shape` and
// `occupancy` or `volume_weights` for the volume forms. A form is named, not
// told from null pointers, as an empty array may have no data at all
struct VoxelInput {
    VoxelForm form = VoxelForm::RUN_ROWS;
    const std::int64_t* rows = nullptr;
    std::size_t row_count = 0;
    const double* row_weights = nullptr;
    const std::uint8_t* occupancy = nullptr;  // numpy bool bytes, each tested != 0, never read as bool
    const double* volume_weights = nullptr;
    std::size_t volume_shape[3] = {0, 0, 0};
    double threshold = 0.5;
};

// A value left unwritten when a vector of them is sized: every output is
// written exactly once, from many threads, so it isn't filled once first
template <typename T>
struct UnwrittenValue {
    T value;
    // NOLINTNEXTLINE(modernize-use-equals-default,cppcoreguidelines-pro-type-member-init): unwritten on purpose
    UnwrittenValue() {}
};

// flat rows: vertices are x, y, z and faces are three vertex indices wound
// counter-clockwise seen from outside; `vertex_sources` names the input
// element holding the inside voxel of each vertex's crossing: its row for
// the row forms, its flat index for the volume forms
struct SurfaceMesh {
    std::vector<UnwrittenValue<double>> vertices;
    std::vector<UnwrittenValue<std::int64_t>> faces;
    std::vector<UnwrittenValue<std::int64_t>> vertex_sources;
};

namespace detail {

// Voxels [begin, end) along Z at (x, y), all weighing `weight`. The source
// of its voxel at height h is `source_base + stride * h`, with one stride for
// every run: 0 for rows, where the base is the row, and 1 for a volume,
// where the base is the flat index of (x, y, 0)
struct VoxelRun {
    std::int64_t x;
    std::int64_t y;
    std::int64_t begin;
    std::int64_t end;
    double weight;
    std::int64_t source_base;
    // unwritten when a vector of them is sized: each is written by the pool
    // NOLINTNEXTLINE(modernize-use-equals-default,cppcoreguidelines-pro-type-member-init): unwritten on purpose
    VoxelRun() {}
    VoxelRun(std::int64_t run_x, std::int64_t run_y, std::int64_t run_begin, std::int64_t run_end, double run_weight,
             std::int64_t run_source_base)
        : x(run_x), y(run_y), begin(run_begin), end(run_end), weight(run_weight), source_base(run_source_base) {}
};

// The triangles of each of the 256 cases. A corner of the cube is numbered
// x + 2y + 4z and is bit `corner` of the case. An edge is numbered
// 4 * axis + u + 2v, for the axis it runs along and where it sits on the next
// two axes in cyclic order. `edges` lists `edge_count` edges, three per
// triangle.
struct CubeCaseTable {
    std::uint8_t edge_count[256];
    // no case has more than five triangles: one more would write past the
    // row, which a compile-time evaluation refuses to do
    std::uint8_t edges[256][15];
};

static constexpr CubeCaseTable build_cube_case_table() {
    // the edge between two corners, which differ in one bit: the axis
    const auto edge_between = [](int a, int b) {
        int axis = 0;
        while (((a ^ b) >> axis) != 1) {
            axis++;
        }
        const int low = a & b;
        return (4 * axis) + ((low >> ((axis + 1) % 3)) & 1) + (2 * ((low >> ((axis + 2) % 3)) & 1));
    };
    // whether two edges lie in one face of the cube: an edge is in the face
    // on its side of each of the two axes it doesn't run along
    const auto share_face = [](int a, int b) {
        for (int i = 1; i < 3; i++) {
            for (int j = 1; j < 3; j++) {
                const int face_a = (2 * ((a / 4 + i) % 3)) + ((a >> (i - 1)) & 1);
                const int face_b = (2 * ((b / 4 + j) % 3)) + ((b >> (j - 1)) & 1);
                if (face_a == face_b) {
                    return true;
                }
            }
        }
        return false;
    };
    CubeCaseTable table{};
    for (int inside = 0; inside < 256; inside++) {
        // the crossing each crossing's line leads to, keeping inside on the left
        int next[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        for (int axis = 0; axis < 3; axis++) {
            for (int side = 0; side < 2; side++) {
                const int u = (axis + 1) % 3;
                const int v = (axis + 2) % 3;
                // the face's corners, counter-clockwise seen from outside the cube
                const int around[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
                int face[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4; i++) {
                    const int along_u = around[i][side == 1 ? 0 : 1];
                    const int along_v = around[i][side == 1 ? 1 : 0];
                    face[i] = (side << axis) | (along_u << u) | (along_v << v);
                }
                for (int i = 0; i < 4; i++) {
                    const bool here = ((inside >> face[i]) & 1) != 0;
                    const bool after = ((inside >> face[(i + 1) % 4]) & 1) != 0;
                    if (here || !after) {
                        continue;
                    }
                    // a group of inside corners starts after corner i: one
                    // line from where it starts to where it ends
                    int last = i + 1;
                    while (((inside >> face[(last + 1) % 4]) & 1) != 0) {
                        last++;
                    }
                    next[edge_between(face[i], face[(i + 1) % 4])] = edge_between(face[last % 4], face[(last + 1) % 4]);
                }
            }
        }
        // each loop of crossings as a fan of triangles
        bool used[12] = {};
        int count = 0;
        for (int first = 0; first < 12; first++) {
            if (next[first] < 0 || used[first]) {
                continue;
            }
            int loop[12] = {};
            int length = 0;
            for (int edge = first; !used[edge]; edge = next[edge]) {
                used[edge] = true;
                loop[length] = edge;
                length++;
            }
            // A fan's inner lines must not lie in a face of the cube, where
            // the cell across the face could draw the same line and leave an
            // edge in four triangles: fan around the first crossing whose
            // lines all pass through the inside of the cube. Every loop has
            // one, as check/tests/test_contour.py checks through every pair of cells.
            int apex = 0;
            for (int candidate = length - 1; candidate >= 0; candidate--) {
                bool through = true;
                for (int step = 2; step + 1 < length; step++) {
                    through = through && !share_face(loop[candidate], loop[(candidate + step) % length]);
                }
                if (through) {
                    apex = candidate;
                }
            }
            for (int step = 1; step + 1 < length; step++) {
                table.edges[inside][count] = static_cast<std::uint8_t>(loop[apex]);
                table.edges[inside][count + 1] = static_cast<std::uint8_t>(loop[(apex + step) % length]);
                table.edges[inside][count + 2] = static_cast<std::uint8_t>(loop[(apex + step + 1) % length]);
                count += 3;
            }
        }
        table.edge_count[inside] = static_cast<std::uint8_t>(count);
    }
    return table;
}

static constexpr CubeCaseTable CUBE_CASES = build_cube_case_table();

// A column of cells, named by the voxel column at its lowest corner, and the
// runs of that voxel column: [begin, end), in order along Z, wherever they
// were read into
struct CellColumn {
    std::int64_t x;
    std::int64_t y;
    const VoxelRun* begin;
    const VoxelRun* end;
};

// what one corner of a cell holds: nothing weighs 0 and has no source
struct CellCorner {
    double weight;
    std::int64_t source_base;
};

// how many vertices and face indices come before a column of cells
struct OutputOffsets {
    std::size_t vertices;
    std::size_t indices;
};

// One column's place in the output as its sweep goes up: the next vertex and
// face index it writes, and for each of the four columns at its corners the
// first of that column's vertices not yet below the sweep, and the end of them
struct ColumnOutputCursor {
    OutputOffsets next;
    std::size_t owned[4];
    std::size_t owned_end[4];
};

// rows per item when rows are read on the pool, and voxel columns per item
// when a volume is
constexpr std::size_t ROW_BLOCK = 4096;
constexpr std::size_t COLUMN_BLOCK = 64;

// more vertices or face indices than this are more than memory holds
constexpr std::size_t TOO_MANY = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 64;

// Read `input` into runs, held in `run_blocks`, and the voxel columns
// holding any into `voxel_columns`, in order, each naming its runs in order
// along Z: rows into one block, sorted; a volume into a block per item, as
// read, never copied again. Every value is checked as it is read: nullptr,
// or what is wrong with them. `source_stride` is the stride of every run's
// sources (see VoxelRun). The items of each map write only their own block's
// output, joined in block order, so what is read is the same however the
// ranges fall.
template <typename Parallel>
static const char* read_voxels(const VoxelInput& input, const Parallel& parallel,
                               std::vector<std::vector<VoxelRun>>* run_blocks, std::vector<CellColumn>* voxel_columns,
                               std::int64_t* source_stride) {
    run_blocks->clear();
    voxel_columns->clear();
    // each block's voxel columns, joined in order into `voxel_columns`
    const auto join_columns = [&](const std::vector<std::vector<CellColumn>>& block_columns) {
        std::size_t total = 0;
        for (const std::vector<CellColumn>& columns : block_columns) {
            total += columns.size();
        }
        voxel_columns->reserve(total);
        for (const std::vector<CellColumn>& columns : block_columns) {
            voxel_columns->insert(voxel_columns->end(), columns.begin(), columns.end());
        }
    };
    if (input.form == VoxelForm::VOXEL_ROWS || input.form == VoxelForm::RUN_ROWS) {
        // rows are copied, so nothing the caller does to them can reach the sweep
        *source_stride = 0;
        const std::size_t width = input.form == VoxelForm::VOXEL_ROWS ? 3 : 4;
        const std::size_t count = input.row_count;
        const std::size_t blocks = (count + ROW_BLOCK - 1) / ROW_BLOCK;
        std::vector<VoxelRun>* runs = &run_blocks->emplace_back();
        runs->resize(count);
        // each block's first bad row, so the lowest bad row of all is reported
        std::vector<const char*> block_problems(blocks, nullptr);
        parallel(blocks, [&](std::size_t first_block, std::size_t last_block) {
            for (std::size_t block = first_block; block < last_block; block++) {
                for (std::size_t i = block * ROW_BLOCK; i < std::min(count, (block + 1) * ROW_BLOCK); i++) {
                    // read once, with the GIL released: what is checked is what is kept
                    const std::int64_t* row = input.rows + (width * i);
                    const std::int64_t x = row[0];
                    const std::int64_t y = row[1];
                    const std::int64_t z = row[2];
                    const std::int64_t length = width == 4 ? row[3] : 1;
                    const double weight = input.row_weights != nullptr ? input.row_weights[i] : 1.0;
                    const char* problem = nullptr;
                    if (length < 1) {
                        problem = "runs must have a length of at least 1";
                    } else if (x <= -INDEX_LIMIT || x >= INDEX_LIMIT || y <= -INDEX_LIMIT || y >= INDEX_LIMIT ||
                               z <= -INDEX_LIMIT || z >= INDEX_LIMIT || length > INDEX_LIMIT ||
                               z > INDEX_LIMIT - length) {
                        problem = "voxels must be within 2^52 of zero";
                    } else if (!(std::abs(weight) <= WEIGHT_LIMIT)) {
                        problem = "weights must be finite and within 1e30 of zero";
                    }
                    if (problem != nullptr) {
                        block_problems[block] = problem;
                        break;
                    }
                    (*runs)[i] = VoxelRun(x, y, z, z + length, weight, static_cast<std::int64_t>(i));
                }
            }
        });
        for (const char* problem : block_problems) {
            if (problem != nullptr) {
                return problem;
            }
        }
        const auto before = [](const VoxelRun& a, const VoxelRun& b) {
            if (a.x != b.x) {
                return a.x < b.x;
            }
            return a.y != b.y ? a.y < b.y : a.begin < b.begin;
        };
        // per block: 1 when a run is before the one ahead of it, 2 when one
        // overlaps the one ahead of it in its column; and the runs that
        // start a voxel column, each a column ending where the next begins
        std::vector<char> block_order(blocks, 0);
        std::vector<std::vector<CellColumn>> block_columns(blocks);
        const auto check_order = [&]() {
            parallel(blocks, [&](std::size_t first_block, std::size_t last_block) {
                for (std::size_t block = first_block; block < last_block; block++) {
                    block_order[block] = 0;
                    block_columns[block].clear();
                    for (std::size_t i = block * ROW_BLOCK; i < std::min(count, (block + 1) * ROW_BLOCK); i++) {
                        const VoxelRun& above = (*runs)[i];
                        if (i > 0) {
                            const VoxelRun& below = (*runs)[i - 1];
                            if (before(above, below)) {
                                block_order[block] = 1;
                                break;
                            }
                            if (below.x == above.x && below.y == above.y) {
                                if (above.begin < below.end) {
                                    block_order[block] = 2;
                                }
                                continue;
                            }
                        }
                        block_columns[block].push_back({above.x, above.y, &above, &above});
                    }
                }
            });
        };
        check_order();
        if (std::find(block_order.begin(), block_order.end(), 1) != block_order.end()) {
            std::sort(runs->begin(), runs->end(), before);
            check_order();
        }
        if (std::find(block_order.begin(), block_order.end(), 2) != block_order.end()) {
            return "voxels must not overlap";
        }
        join_columns(block_columns);
        // each column ends where the next begins
        for (std::size_t k = 0; k < voxel_columns->size(); k++) {
            (*voxel_columns)[k].end =
                k + 1 < voxel_columns->size() ? (*voxel_columns)[k + 1].begin : runs->data() + count;
        }
        return nullptr;
    }

    // A volume is read in place, each voxel once, into the runs of its own
    // block: a caller changing it during the call changes the answer, never
    // what the sweep can rely on
    *source_stride = 1;
    const std::size_t size_x = input.volume_shape[0];
    const std::size_t size_y = input.volume_shape[1];
    const std::size_t size_z = input.volume_shape[2];
    for (const std::size_t extent : input.volume_shape) {
        // compared as 64 bits: a 32-bit size_t would cut the limit to nothing
        if (static_cast<std::uint64_t>(extent) >= static_cast<std::uint64_t>(INDEX_LIMIT)) {
            return "voxels must be within 2^52 of zero";
        }
    }
    if (size_x == 0 || size_y == 0 || size_z == 0) {
        return nullptr;
    }
    if (size_x > std::numeric_limits<std::size_t>::max() / size_y / size_z) {
        return "a volume must hold no more voxels than memory can";
    }
    const bool occupancy = input.form == VoxelForm::OCCUPANCY_VOLUME;
    const double threshold = input.threshold;
    const std::size_t plane = size_y * size_z;
    // Append the runs of voxel column (x, y) to `out`: stretches of one
    // weight as read, nothing weighing 0. Returns false at a weight it can't
    // read. A weight volume's voxel on its own side of all six neighbors,
    // where beyond the volume is outside, is never interpolated: it is read
    // as nothing outside and as the most there is inside
    const auto scan_column = [&](std::size_t x, std::size_t y, std::vector<VoxelRun>* out) {
        const std::size_t column = (x * size_y) + y;
        const auto emit = [&](std::size_t begin, std::size_t end, double weight) {
            out->emplace_back(static_cast<std::int64_t>(x), static_cast<std::int64_t>(y),
                              static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end), weight,
                              static_cast<std::int64_t>(column * size_z));
        };
        const std::uint8_t* occupied = occupancy ? input.occupancy + (column * size_z) : nullptr;
        const double* self = occupancy ? nullptr : input.volume_weights + (column * size_z);
        // the four columns beside, or nothing beyond the volume's edge
        const double* const neighbors[4] = {
            !occupancy && x > 0 ? self - plane : nullptr, !occupancy && x + 1 < size_x ? self + plane : nullptr,
            !occupancy && y > 0 ? self - size_z : nullptr, !occupancy && y + 1 < size_y ? self + size_z : nullptr};
        double current = 0.0;
        std::size_t begin = 0;
        bool below = false;
        for (std::size_t z = 0; z < size_z; z++) {
            double weight = 0.0;
            if (occupancy) {
                weight = occupied[z] != 0 ? 1.0 : 0.0;
            } else {
                weight = self[z];
                if (!(std::abs(weight) <= WEIGHT_LIMIT)) {
                    return false;
                }
                const bool inside = weight > threshold;
                bool shell = inside != below || inside != (z + 1 < size_z && self[z + 1] > threshold);
                for (const double* neighbor : neighbors) {
                    shell = shell || inside != (neighbor != nullptr && neighbor[z] > threshold);
                }
                if (!shell) {
                    weight = inside ? std::numeric_limits<double>::max() : 0.0;
                }
                below = inside;
            }
            if (weight != current) {
                if (current != 0.0) {
                    emit(begin, z, current);
                }
                current = weight;
                begin = z;
            }
        }
        if (current != 0.0) {
            emit(begin, size_z, current);
        }
        return true;
    };
    const std::size_t column_count = size_x * size_y;
    const std::size_t blocks = (column_count + COLUMN_BLOCK - 1) / COLUMN_BLOCK;
    run_blocks->resize(blocks);
    std::vector<std::vector<CellColumn>> block_columns(blocks);
    std::vector<char> block_unreadable(blocks, 0);
    parallel(blocks, [&](std::size_t first_block, std::size_t last_block) {
        for (std::size_t block = first_block; block < last_block; block++) {
            std::vector<VoxelRun>& out = (*run_blocks)[block];
            // where each column's runs end in `out`, which may move as it
            // grows: named by address once the block is read
            std::vector<std::size_t> ends;
            for (std::size_t column = block * COLUMN_BLOCK; column < std::min(column_count, (block + 1) * COLUMN_BLOCK);
                 column++) {
                const std::size_t first = out.size();
                if (!scan_column(column / size_y, column % size_y, &out)) {
                    block_unreadable[block] = 1;
                    break;
                }
                if (out.size() > first) {
                    ends.push_back(out.size());
                }
            }
            const VoxelRun* begin = out.data();
            for (const std::size_t end : ends) {
                block_columns[block].push_back({begin->x, begin->y, begin, out.data() + end});
                begin = out.data() + end;
            }
        }
    });
    if (std::find(block_unreadable.begin(), block_unreadable.end(), 1) != block_unreadable.end()) {
        return "weights must be finite and within 1e30 of zero";
    }
    join_columns(block_columns);
    return nullptr;
}

// Call `visit(z, repeat, corners, inside)` for the cells the surface passes
// through in one column of cells, upward along Z. `beside` is the four voxel
// columns at the column's corners, ordered dx + 2 * dy. A visit is `repeat`
// cells alike, one above the next, the first with its lowest corner at z:
// their eight corners are ordered dx + 2 * dy + 4 * dz, and bit k of
// `inside` is set when corner k weighs more than `threshold`. Cells that are
// all in or all out are never visited, so a solid costs its surface. Each of
// the four voxel columns' runs are in order and apart, so their starts and
// ends are a rising sequence, merged here four ways.
template <typename Visit>
static void sweep_cell_column(const std::vector<CellColumn>& columns, const std::array<std::size_t, 4>& beside,
                              double threshold, const Visit& visit) {
    constexpr std::int64_t NONE = std::numeric_limits<std::int64_t>::max();
    const VoxelRun* cursor[4];
    const VoxelRun* end[4];
    // the next height at which voxel column k changes: where its run starts,
    // or where it ends once the sweep is inside it
    std::int64_t pending[4];
    for (std::size_t k = 0; k < 4; k++) {
        cursor[k] = columns[beside[k]].begin;
        end[k] = columns[beside[k]].end;
        pending[k] = cursor[k] < end[k] ? cursor[k]->begin : NONE;
    }
    // corners[0..4) are the voxels below a change and corners[4..8) above it
    CellCorner corners[8] = {{0.0, -1}, {0.0, -1}, {0.0, -1}, {0.0, -1}, {0.0, -1}, {0.0, -1}, {0.0, -1}, {0.0, -1}};
    int below = 0;
    std::int64_t z = std::min({pending[0], pending[1], pending[2], pending[3]});
    while (z != NONE) {
        int above = 0;
        for (std::size_t k = 0; k < 4; k++) {
            if (pending[k] == z) {
                if (cursor[k]->begin == z) {
                    pending[k] = cursor[k]->end;
                } else {
                    // past the end of a run: the next, which may start right here
                    cursor[k]++;
                    pending[k] = cursor[k] < end[k] ? cursor[k]->begin : NONE;
                    if (pending[k] == z) {
                        pending[k] = cursor[k]->end;
                    }
                }
            }
            corners[4 + k] = {0.0, -1};
            if (cursor[k] < end[k] && cursor[k]->begin <= z) {
                corners[4 + k] = {cursor[k]->weight, cursor[k]->source_base};
            }
            if (corners[4 + k].weight > threshold) {
                above |= 1 << k;
            }
        }
        const std::int64_t next = std::min({pending[0], pending[1], pending[2], pending[3]});
        // the cell across the change, then the cells within the stretch above
        const int across = below | (above << 4);
        if (across != 0 && across != 255) {
            visit(z - 1, std::int64_t{1}, corners, across);
        }
        std::copy(corners + 4, corners + 8, corners);
        below = above;
        if (above != 0 && above != 15 && next != NONE && next - z > 1) {
            visit(z, next - z - 1, corners, above | (above << 4));
        }
        z = next;
    }
}

}  // namespace detail

// The surface of `input`: nullptr, or what is wrong with the input.
//
// `parallel(count, body)` calls `body(begin, end)` over ranges that cover
// [0, count), on as many threads as it likes, and throws to give up. Each
// column of cells is one item, and each map over them writes only its own
// item's output, so the mesh is the same however the ranges fall:
//   count  how many vertices and face indices each column of cells makes
//   place  each crossing's vertex, written by the one column that owns it
//   join   each cell's triangles, naming vertices their owners placed
// A crossing lies on a grid edge, which up to four columns of cells share:
// the column whose own voxel column the edge starts from owns it, and holds
// its vertices in order of `4 * z + axis`. A cell going up its column finds
// a neighbor's vertex a step or two on from the last one it found there, so
// no table is shared between threads.
// Throws std::bad_alloc for a surface too large to hold, before making it,
// and std::logic_error should a crossing ever lack its vertex.
template <typename Parallel>
static const char* surface(const VoxelInput& input, const Parallel& parallel, SurfaceMesh* mesh) {
    using detail::CellColumn;
    using detail::CellCorner;
    using detail::ColumnOutputCursor;
    using detail::CUBE_CASES;
    using detail::OutputOffsets;
    using detail::TOO_MANY;
    *mesh = SurfaceMesh();
    const double threshold = input.threshold;
    if (std::isnan(threshold) || threshold < 0.0 || threshold > WEIGHT_LIMIT) {
        return "threshold must be at least 0 and within 1e30";
    }
    // the runs, which the columns name: kept as read until the mesh is made
    std::vector<std::vector<detail::VoxelRun>> run_blocks;
    std::vector<CellColumn> voxel_columns;
    std::int64_t source_stride = 0;
    const char* problem = detail::read_voxels(input, parallel, &run_blocks, &voxel_columns, &source_stride);
    if (problem != nullptr) {
        return problem;
    }

    // Every column of cells the surface can pass through, in order: each
    // voxel column, and the columns a step back from it along y, x and both,
    // which hold no runs unless they are voxel columns themselves. Stream k
    // is the voxel columns stepped back k & 1 along y and k >> 1 along x,
    // each in order, merged: of columns alike, the voxel column comes first
    // and is kept
    const std::size_t voxel_count = voxel_columns.size();
    std::vector<CellColumn> columns;
    columns.reserve((4 * voxel_count) + 1);
    std::size_t heads[4] = {0, 0, 0, 0};
    for (;;) {
        std::size_t first = 4;
        std::int64_t first_x = 0;
        std::int64_t first_y = 0;
        for (std::size_t k = 0; k < 4; k++) {
            if (heads[k] == voxel_count) {
                continue;
            }
            const std::int64_t x = voxel_columns[heads[k]].x - static_cast<std::int64_t>(k >> 1U);
            const std::int64_t y = voxel_columns[heads[k]].y - static_cast<std::int64_t>(k & 1U);
            if (first == 4 || x < first_x || (x == first_x && y < first_y)) {
                first = k;
                first_x = x;
                first_y = y;
            }
        }
        if (first == 4) {
            break;
        }
        const CellColumn& voxel_column = voxel_columns[heads[first]++];
        if (columns.empty() || columns.back().x != first_x || columns.back().y != first_y) {
            columns.push_back(first == 0 ? voxel_column : CellColumn{first_x, first_y, nullptr, nullptr});
        }
    }
    const std::size_t count = columns.size();
    // what stands in for a column that isn't there: no runs, and no crossings
    columns.push_back({0, 0, nullptr, nullptr});

    // The columns at each column's four corners: itself, then +x, +y, both.
    // The one a step along y is next if it is anywhere, and `ahead`, the
    // first column not before a step along x, only ever moves forward
    std::vector<std::array<std::size_t, 4>> beside(count);
    const auto found = [&](std::size_t c, std::int64_t x, std::int64_t y) {
        return c < count && columns[c].x == x && columns[c].y == y ? c : count;
    };
    std::size_t ahead = 0;
    for (std::size_t c = 0; c < count; c++) {
        const std::int64_t x = columns[c].x;
        const std::int64_t y = columns[c].y;
        while (ahead < count && (columns[ahead].x < x + 1 || (columns[ahead].x == x + 1 && columns[ahead].y < y))) {
            ahead++;
        }
        const std::size_t along_x = found(ahead, x + 1, y);
        beside[c] = {c, along_x, found(c + 1, x, y + 1), found(along_x == count ? ahead : ahead + 1, x + 1, y + 1)};
    }

    // `visit(c, cursor, z, repeat, corners, inside)` for the cells of every column c
    std::vector<OutputOffsets> starts(count + 2, OutputOffsets{0, 0});
    const auto map_cells = [&](const auto& visit) {
        parallel(count, [&](std::size_t begin, std::size_t end) {
            for (std::size_t c = begin; c < end; c++) {
                ColumnOutputCursor cursor = {starts[c], {}, {}};
                for (std::size_t k = 0; k < 4; k++) {
                    cursor.owned[k] = starts[beside[c][k]].vertices;
                    cursor.owned_end[k] = starts[beside[c][k] + 1].vertices;
                }
                detail::sweep_cell_column(columns, beside[c], threshold,
                                          [&](std::int64_t z, std::int64_t repeat, const CellCorner* corners,
                                              int inside) { visit(c, &cursor, z, repeat, corners, inside); });
            }
        });
    };

    // count: each column's extent, from the crossings on the three edges
    // that start at a cell's lowest corner, then summed into where each
    // starts. Every sum stops at TOO_MANY, so none can overflow
    const auto add_cells = [](std::size_t total, int each, std::int64_t repeat) {
        const auto cells = static_cast<std::size_t>(std::min(repeat, static_cast<std::int64_t>(TOO_MANY / 16)));
        return std::min(TOO_MANY, total + (static_cast<std::size_t>(each) * cells));
    };
    map_cells(
        [&](std::size_t c, ColumnOutputCursor*, std::int64_t, std::int64_t repeat, const CellCorner*, int inside) {
            const int crossings =
                ((inside ^ (inside >> 1)) & 1) + ((inside ^ (inside >> 2)) & 1) + ((inside ^ (inside >> 4)) & 1);
            starts[c + 1].vertices = add_cells(starts[c + 1].vertices, crossings, repeat);
            starts[c + 1].indices = add_cells(starts[c + 1].indices, CUBE_CASES.edge_count[inside], repeat);
        });
    for (std::size_t c = 0; c <= count; c++) {
        starts[c + 1].vertices = std::min(TOO_MANY, starts[c + 1].vertices + starts[c].vertices);
        starts[c + 1].indices = std::min(TOO_MANY, starts[c + 1].indices + starts[c].indices);
    }
    const OutputOffsets total = starts[count];
    if (total.vertices == TOO_MANY || total.indices == TOO_MANY) {
        throw std::bad_alloc();
    }
    mesh->vertices.resize(3 * total.vertices);
    mesh->vertex_sources.resize(total.vertices);
    mesh->faces.resize(total.indices);
    std::vector<UnwrittenValue<std::int64_t>> keys(total.vertices);

    // place: where the weight along each crossing edge passes the threshold
    map_cells([&](std::size_t c, ColumnOutputCursor* cursor, std::int64_t z, std::int64_t repeat,
                  const CellCorner* corners, int inside) {
        for (std::int64_t cell = z; cell < z + repeat; cell++) {
            for (int axis = 0; axis < 3; axis++) {
                const int high = 1 << axis;
                if (((inside ^ (inside >> high)) & 1) == 0) {
                    continue;
                }
                const std::size_t vertex = cursor->next.vertices++;
                double position[3] = {static_cast<double>(columns[c].x), static_cast<double>(columns[c].y),
                                      static_cast<double>(cell)};
                position[axis] += (threshold - corners[0].weight) / (corners[high].weight - corners[0].weight);
                for (std::size_t k = 0; k < 3; k++) {
                    mesh->vertices[(3 * vertex) + k].value = position[k];
                }
                // the inside voxel: the low corner, or the high one, a step up along Z for a Z edge
                const bool low_inside = (inside & 1) != 0;
                const std::int64_t height = cell + (!low_inside && axis == 2 ? 1 : 0);
                const std::int64_t base = low_inside ? corners[0].source_base : corners[high].source_base;
                mesh->vertex_sources[vertex].value = base + (source_stride * height);
                keys[vertex].value = (4 * cell) + axis;
            }
        }
    });

    // join: a triangle's corners are crossings on the cell's twelve edges,
    // each owned by the column at the corner of the cell its edge starts from
    map_cells([&](std::size_t, ColumnOutputCursor* cursor, std::int64_t z, std::int64_t repeat, const CellCorner*,
                  int inside) {
        for (std::int64_t cell = z; cell < z + repeat; cell++) {
            for (int t = 0; t < CUBE_CASES.edge_count[inside]; t++) {
                const int edge = CUBE_CASES.edges[inside][t];
                const int axis = edge / 4;
                const int low = ((edge & 1) << ((axis + 1) % 3)) | (((edge >> 1) & 1) << ((axis + 2) % 3));
                // the owner's vertices from this cell's height on, then the one
                const std::size_t owner = static_cast<std::size_t>(low) & 3U;
                const std::size_t owned_end = cursor->owned_end[owner];
                std::size_t& owned = cursor->owned[owner];
                while (owned < owned_end && keys[owned].value < 4 * cell) {
                    owned++;
                }
                const std::int64_t key = (4 * (cell + (low >> 2))) + axis;
                std::size_t vertex = owned;
                while (vertex < owned_end && keys[vertex].value < key) {
                    vertex++;
                }
                if (vertex == owned_end || keys[vertex].value != key) {
                    throw std::logic_error("contour: a crossing has no vertex");
                }
                mesh->faces[cursor->next.indices++].value = static_cast<std::int64_t>(vertex);
            }
        }
    });
    return nullptr;
}

}  // namespace contour
