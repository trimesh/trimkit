// The one extension: every part's functions and classes in one module,
// `import trimkit`, built as the package itself (trimkit/__init__ with the
// extension suffix), so there is no Python in it at all. The top level asks
// geometry questions whatever library answers them; its one submodule,
// `trimkit.exchange`, reads and writes specific formats.
#include "numpy.hpp"

#include "contour.hpp"
#include "svgpath.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// -- What the bindings below share -------------------------------------------

// one member of a bound enumeration: its name and docstring, in a table
// whose order is the values' (bind_enum)
struct EnumMember {
    const char* name;
    const char* doc;
};

namespace trimkit {

// An enumeration from the one table its values come from: member i is
// table[i].name, with table[i].doc as its docstring, and the class docstring
// is `summary` then a numpy Attributes section naming every member
template <typename Enum, typename Entry, std::size_t COUNT>
static void bind_enum(const nb::module_& scope, const char* name, const char* summary, const Entry (&table)[COUNT]) {
    // one per enumeration, alive as long as the module that shows it
    static const std::string DOC = [&] {
        std::string text = std::string(summary) + "\n\nAttributes\n----------\n";
        for (const Entry& entry : table) {
            text += std::string(entry.name) + "\n  " + entry.doc + "\n";
        }
        return text;
    }();
    nb::enum_<Enum> bound(scope, name, nb::is_arithmetic(), DOC.c_str());
    for (std::size_t i = 0; i < COUNT; i++) {
        bound.value(table[i].name, static_cast<Enum>(i), table[i].doc);
    }
}

// A bound class whose objects pickle, the only way the package binds one.
// `T` says what it pickles as with `nb::dict constructor_arguments() const`:
// its constructor's arguments, by their names, from which a load builds it
// again (a compile error without it). Never an acceleration structure, which
// is rebuilt: a pickle holds only arrays and Python values, named, so it loads
// into any build whose constructor takes the same names.
// check/tests/test_invariants.py round-trips every class the package binds
template <typename T>
static nb::class_<T> picklable_class(nb::module_& m, const char* name, const char* doc) {
    static_assert(std::is_same_v<decltype(std::declval<const T&>().constructor_arguments()), nb::dict>,
                  "a bound class must say what it pickles as: nb::dict constructor_arguments() const");
    nb::class_<T> bound(m, name, doc);
    bound.def(
        "__reduce__",
        [](nb::handle self) {
            const nb::dict arguments = nb::cast<const T&>(self).constructor_arguments();
            const nb::object partial = nb::module_::import_("functools").attr("partial");
            return nb::make_tuple(partial(nb::borrow(self.type()), **arguments), nb::make_tuple());
        },
        R"(Pickle as its constructor's arguments, by name: anything built from them is built again on load.

Returns
-------
reduced : tuple
  Its constructor with those arguments bound, as pickle and copy call it.)");
    return bound;
}

}  // namespace trimkit

// -- contour -------------------------------------------------------------------
//
// The surface of voxels with src/contour.hpp, which is ours.
//
// Rows are copied into the header's own records and each copy checked, so
// nothing the caller does to the array during the call can reach the sweep;
// a volume is read in place, a column at a time, and each value checked as it
// is read. It keeps no global state and runs through `trimkit::with_pool`:
// without the GIL, reading the voxels and every pass over the columns of
// cells on the pool, where Ctrl-C stops them between ranges.

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
            trimkit::refuse("weights are for rows of voxels: a volume holds its own");
        }
        input.form = voxels.is_occupancy ? contour::VoxelForm::OCCUPANCY_VOLUME : contour::VoxelForm::WEIGHT_VOLUME;
        input.occupancy = trimkit::mask_bytes(voxels.occupancy);
        input.volume_weights = voxels.weights.data();
        for (std::size_t k = 0; k < 3; k++) {
            input.volume_shape[k] = voxels.is_occupancy ? voxels.occupancy.shape(k) : voxels.weights.shape(k);
        }
    } else {
        const std::size_t columns = voxels.rows.shape(1);
        if (columns != 3 && columns != 4) {
            trimkit::refuse("voxels must be (n, 3) or (n, 4) rows, not (n, " + std::to_string(columns) + ")");
        }
        input.form = columns == 3 ? contour::VoxelForm::VOXEL_ROWS : contour::VoxelForm::RUN_ROWS;
        input.rows = voxels.rows.data();
        input.row_count = voxels.rows.shape(0);
        if (!weights_argument.is_none()) {
            weights = trimkit::read(weights_argument, "weights");
            if (weights.shape(0) != input.row_count) {
                trimkit::refuse("weights must be one per row: " + std::to_string(weights.shape(0)) + " for " +
                                std::to_string(input.row_count) + " rows");
            }
            input.row_weights = weights.data();
        }
    }

    contour::SurfaceMesh mesh;
    const char* problem = trimkit::with_pool([&](const trimkit::parallel::Stop& stop) {
        const auto parallel = [&](std::size_t count, const auto& body) { trimkit::in_parallel(count, body, stop); };
        return contour::surface(input, parallel, &mesh);
    });
    if (problem != nullptr) {
        trimkit::refuse(problem);
    }
    const std::size_t vertex_count = mesh.vertex_sources.size();
    return {trimkit::to_numpy_rows<double>(std::move(mesh.vertices), {3}),
            trimkit::to_numpy_rows<std::int64_t>(std::move(mesh.faces), {3}),
            trimkit::to_numpy<std::int64_t>(std::move(mesh.vertex_sources), {vertex_count})};
}

static void bind_contour(nb::module_& m) {
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
>>> runs = np.array([[0, 0, 0, 1]], dtype=np.int64)
>>> vertices, faces, vertex_sources = trimkit.contour(runs)
>>> len(vertices), len(faces), vertex_sources.tolist()
(6, 8, [0, 0, 0, 0, 0, 0])
>>> volume = np.zeros((3, 3, 3), dtype=bool)
>>> volume[1, 1, 1] = True
>>> vertices, faces, vertex_sources = trimkit.contour(volume)
>>> len(faces), set(vertex_sources.tolist())
(8, {13}))");
}

// -- svg -----------------------------------------------------------------------
//
// SVG path data to and from vertices and one record per segment, with our
// header-only `svgpath`.
//
// The parser fills a vertex array and a std::vector<svgpath::SvgSegment>,
// both handed to numpy without a copy; the segments are viewed as the
// `SvgSegment` structured dtype, built from the struct's own offsets so the
// two can never disagree. The writer copies both into vectors of its own
// before it lets go of the GIL: no other thread can change what it checks
// while it writes, on a free-threaded build either.

// what svgpath parses is exactly what the door lets `svg_string` read back
static_assert(svgpath::COORDINATE_LIMIT == trimkit::COORDINATE_LIMIT, "svgpath and numpy.hpp share one bound");
static_assert(contour::WEIGHT_LIMIT == trimkit::COORDINATE_LIMIT, "contour and numpy.hpp share one bound");

static nb::object make_segment_dtype() {
    nb::dict specification;
    specification["names"] = nb::make_tuple("kind", "points", "kind_data");
    specification["formats"] = nb::make_tuple("u1", "(4,)i8", "(2,)f8");
    specification["offsets"] =
        nb::make_tuple(offsetof(svgpath::SvgSegment, kind), offsetof(svgpath::SvgSegment, points),
                       offsetof(svgpath::SvgSegment, kind_data));
    specification["itemsize"] = sizeof(svgpath::SvgSegment);
    return nb::module_::import_("numpy").attr("dtype")(specification);
}

static nb::tuple svg_parse(std::string_view path, const nb::object& segment_dtype) {
    // how many segments a path holds isn't known until it is parsed, so its
    // vectors grow as it goes
    svgpath::SvgPath parsed;
    {
        const trimkit::GilReleased release;
        // like an SVG renderer, malformed data keeps everything before the
        // error rather than failing the whole path, so how far it got is unused
        static_cast<void>(svgpath::parse(path.data(), path.size(), &parsed));
    }
    return nb::make_tuple(trimkit::to_numpy_rows(std::move(parsed.vertices), {2}),
                          trimkit::to_numpy_records(std::move(parsed.segments), segment_dtype));
}

static std::string svg_string(const trimkit::Array<double, -1, 2>& vertices_argument,
                              const nb::handle& segments_argument, const trimkit::Number<std::int64_t>& digits_argument,
                              const nb::object& segment_dtype) {
    if (digits_argument < 1 || digits_argument > 17) {
        trimkit::refuse("digits must be from 1 to 17");
    }
    const int digits = static_cast<int>(digits_argument);
    // copied once, under the GIL, then only the copies are read
    const trimkit::In<double, -1, 2> vertices = trimkit::read(vertices_argument, "vertices");
    const trimkit::Records<svgpath::SvgSegment> segments =
        trimkit::read_records<svgpath::SvgSegment>(segments_argument, segment_dtype, "SvgSegment", "segments");
    svgpath::SvgPath path;
    path.vertices.assign(vertices.data(), vertices.data() + vertices.size());
    path.segments.assign(segments.rows(), segments.rows() + segments.size());
    std::string written;
    const char* refusal = nullptr;
    std::size_t refused = 0;
    {
        const trimkit::GilReleased release;
        refusal = svgpath::format(&path, digits, &written, &refused);
    }
    if (refusal != nullptr) {
        trimkit::refuse(refused < path.segments.size() ? "segment " + std::to_string(refused) + ": " + refusal
                                                       : std::string(refusal));
    }
    return written;
}

// svgpath's segment kinds, named, in its order: each one's index here is its value
static constexpr EnumMember SVG_KINDS[] = {
    {"LINE", "A straight line: points start, -1, -1, end. The L, H, V and Z commands."},
    {"QUADRATIC", "A quadratic Bezier curve: points start, control, -1, end. Q and T."},
    {"CUBIC", "A cubic Bezier curve: points start, control, control, end. C and S."},
    {"ARC",
     "A circular arc of at most 120 degrees: points start, the point halfway along it, -1, end, and "
     "kind_data its center. A with equal radii."},
    {"CONIC",
     "An elliptical arc of at most 120 degrees, an exact rational quadratic Bezier: points start, "
     "control, -1, end, and kind_data its weight, then 0. A with unequal radii."}};
static_assert(svgpath::LINE == 0 && svgpath::QUADRATIC == 1 && svgpath::CUBIC == 2 && svgpath::ARC == 3 &&
                  svgpath::CONIC == 4,
              "SVG_KINDS lists svgpath's kinds in its order");

static void bind_svg(nb::module_& m) {
    trimkit::bind_enum<svgpath::SvgKind>(m, "SvgKind", "What drew an SvgSegment, its `kind` field.", SVG_KINDS);

    const nb::object segment_dtype = make_segment_dtype();
    m.attr("SvgSegment") = segment_dtype;

    m.def(
        "svg_parse", [segment_dtype](std::string_view path) { return svg_parse(path, segment_dtype); }, nb::arg("path"),
        nb::sig("def svg_parse(path: str) -> tuple[numpy.typing.NDArray[numpy.float64], "
                "numpy.typing.NDArray[numpy.void]]"),
        R"(Parse SVG path data into vertices and one record per segment.

Parameters
----------
path : str
  The `d` attribute of an SVG `<path>`.

Returns
-------
vertices : (n, 2) float64
  Every point a segment names, once, in path order and absolute
  coordinates: on-curve points, and Bezier and conic control points,
  never an arc's center.
segments : (m,) SvgSegment
  `kind` is an `SvgKind`. `points` are vertex indices: start, then a
  control (quadratic, conic), a cubic's first control or an arc's
  halfway point, then a cubic's second control, then end; -1 where
  the kind has none. `kind_data` is what the kind needs beyond its
  points: an arc's center, the same for every piece of one SVG arc; a
  conic's weight, then 0; zeros for any other kind. A segment starts on
  the vertex the last one ended on, unless it starts a new subpath. A
  line that ends on its subpath's first vertex, without starting there,
  closes it (`Z`).

Notes
-----
Arcs are split into pieces of at most 120 degrees. Equal radii make
`ARC` pieces: start, the point halfway along, end, the three-point form
of a circular arc. Unequal radii make `CONIC` pieces: start, control
(where the tangents meet), end and a weight w, the rational quadratic
Bezier (S (1-t)^2 + 2 w C t (1-t) + E t^2) / ((1-t)^2 + 2 w t (1-t) +
t^2). That is exactly the ellipse, and an affine transform of the
vertices transforms it exactly. Its middle is (S + 2wC + E) / (2 + 2w).

Accepts the SVG 2 path grammar and three things it doesn't: commas
anywhere whitespace may be, signed arc radii and rotation, and numbers
ending in a point ("1."). Malformed data returns everything before the
first bad token, as SVG renders it: that includes a command whose
coordinates overflow, from a relative step or a reflected control
point. Arcs with a zero radius are lines, and so are arcs that can't
be described in normal doubles within +-1e30, center included. Every
number and coordinate is within +-1e30, like every trimkit input: a
number past it ends the path. The output is at most 43 bytes per byte
of path data.

Examples
--------
>>> import trimkit
>>> vertices, segments = trimkit.exchange.svg_parse("M 0 0 L 1 1")
>>> vertices.tolist(), segments["kind"].tolist(), segments["points"].tolist()
([[0.0, 0.0], [1.0, 1.0]], [0], [[0, -1, -1, 1]]))");

    m.def(
        "svg_string",
        [segment_dtype](const trimkit::Array<double, -1, 2>& vertices, const nb::handle& segments,
                        const trimkit::Number<std::int64_t>& digits) {
            return svg_string(vertices, segments, digits, segment_dtype);
        },
        nb::arg("vertices"), nb::arg("segments"), nb::kw_only(), nb::arg("digits") = 8,
        nb::sig("def svg_string(vertices: numpy.typing.NDArray[numpy.float64], "
                "segments: numpy.typing.NDArray[numpy.void], *, digits: int = 8) -> str"),
        R"(Write vertices and segments back as SVG path data, the inverse of `svg_parse`.

Parameters
----------
vertices : (n, 2) float64
  Points the segments index.
segments : (m,) SvgSegment
  Segments as `svg_parse` returns them, or built by hand. Only the
  points and `kind_data` a kind uses are read: an arc's center is not,
  as its three points say where its circle is.
digits : int
  The most significant digits a number is written with, 1 to 17: 8
  keeps a coordinate near 1000 to within 5e-5. Each number is its shortest
  decimal that reads back as the same double, rounded half away from
  zero, so 17 round trips every double exactly.

Returns
-------
path : str
  Absolute path data, independent of locale. Arcs and conics are `A`
  commands of at most 120 degrees: a longer one, built by hand, is
  split. At 17 digits, parsing it gives back every line and Bezier, and
  every segment's start and end, bit for bit, for anything `svg_parse`
  returned; an arc's halfway point and center, and a conic's control and
  weight, within rounding.

Examples
--------
>>> import trimkit
>>> path = trimkit.exchange.svg_string(*trimkit.exchange.svg_parse("M 0 0 L 1 1 Z"))
>>> path
'M0 0 L1 1 Z')");
}

// -- bvh -------------------------------------------------------------------------
//
// A `RayScene` holds its meshes as the unit (bvh.cpp) built them, once, with
// the GIL released, and remembers the last top level it placed: a query
// naming the same instance arrays (the same bytes) again reuses it rather
// than placing them again, and a query keeps the one it started with. Each
// query reads its rays through the door, then runs on the pool with the GIL
// released, where Ctrl-C stops the CPU between ranges. A `Boxes` holds the
// unit's `BoxTree` the same way.

namespace {

using MaximumRayHits = std::optional<trimkit::Number<std::int64_t>>;
using RayHitColumns = std::tuple<trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>,
                                 trimkit::Out<double>>;
using NearestBoxHits = std::tuple<trimkit::Out<std::int64_t>, trimkit::Out<double>>;

// Axis-aligned boxes in 2D or 3D, as the unit's `BoxTree` holds them
class Boxes {
public:
    explicit Boxes(const trimkit::BoxesArgument& argument)
        : tree_([&] {
              // read before the GIL is released, so it is let go of after it is held again
              const trimkit::BoxQueries boxes = trimkit::read_box_queries(argument.object, 0, "boxes");
              const trimkit::GilReleased release;
              return trimkit::BoxTree(boxes.data, boxes.count, boxes.dimension);
          }()) {}

    // `count` 3D boxes as a scene culls by them, 6 values each, minimum then
    // maximum corner: always within COORDINATE_LIMIT, as a scene refuses an
    // instance placed past it (refuse_placed_beyond)
    Boxes(const double* boxes, std::size_t count)
        : tree_([&] {
              const trimkit::GilReleased release;
              return trimkit::BoxTree(boxes, count, 3);
          }()) {}

    std::size_t size() const { return tree_.size(); }

    // pickled as the boxes it was built from
    nb::dict constructor_arguments() const {
        const nb::dict arguments;
        arguments["boxes"] = boxes();
        return arguments;
    }

    // the box around every box: (2, d), or +inf then -inf with none
    trimkit::Out<double> bounds() const { return trimkit::to_numpy(tree_.bounds(), {2, tree_.dimension()}); }

    // every box as given: (n, 2, d), exactly
    trimkit::Out<double> boxes() const {
        std::vector<double> boxes;
        {
            const trimkit::GilReleased release;
            boxes = tree_.boxes();
        }
        return trimkit::to_numpy_rows(std::move(boxes), {2, tree_.dimension()});
    }

    // every box overlapping each query, touching included like rtree: one
    // query, (2d,) or (2, d), answers its sorted boxes, (n,), and many,
    // (q, 2d) or (q, 2, d), their [query, box] rows, (k, 2). Every form is
    // (q, 2, d) in memory, so they read as it and share one traversal
    trimkit::Out<std::int64_t> intersection(const trimkit::BoxQueriesArgument& argument) const {
        const trimkit::BoxQueries queries = trimkit::read_box_queries(argument.object, tree_.dimension(), "queries");
        std::vector<trimkit::Unset<std::int64_t>> hits = trimkit::with_pool([&](const trimkit::parallel::Stop& stop) {
            return tree_.intersection(queries.data, queries.count, queries.one, stop);
        });
        if (queries.one) {
            return trimkit::to_numpy<std::int64_t>(std::move(hits), {hits.size()});
        }
        return trimkit::to_numpy_rows<std::int64_t>(std::move(hits), {2});
    }

    // the `maximum_hits` nearest boxes to each query, as many as there are
    // up to that, by distance and then box: one query answers its boxes,
    // (n,), and many their [query, box] rows, (k, 2), with each row's
    // distance
    NearestBoxHits nearest(const trimkit::BoxQueriesArgument& argument,
                           const trimkit::Number<std::int64_t>& maximum_hits) const {
        if (maximum_hits < 1) {
            trimkit::refuse("maximum_hits must be at least 1");
        }
        const trimkit::BoxQueries queries = trimkit::read_box_queries(argument.object, tree_.dimension(), "queries");
        std::vector<trimkit::Unset<std::int64_t>> hits;
        std::vector<trimkit::Unset<double>> distances;
        trimkit::with_pool([&](const trimkit::parallel::Stop& stop) {
            tree_.nearest(queries.data, queries.count, queries.one, trimkit::at_most(maximum_hits), stop, &hits,
                          &distances);
        });
        const std::size_t rows = distances.size();
        if (queries.one) {
            return {trimkit::to_numpy<std::int64_t>(std::move(hits), {rows}),
                    trimkit::to_numpy<double>(std::move(distances), {rows})};
        }
        return {trimkit::to_numpy_rows<std::int64_t>(std::move(hits), {2}),
                trimkit::to_numpy<double>(std::move(distances), {rows})};
    }

private:
    trimkit::BoxTree tree_;
};

class RayScene {
public:
    // every mesh read first, then built without the GIL
    RayScene(const trimkit::Meshes& meshes, trimkit::Flag use_gpu)
        : use_gpu_(use_gpu.value), scene_([&] {
              std::vector<trimkit::Mesh<std::uint64_t>> read = trimkit::read_meshes<std::uint64_t>(meshes);
              std::vector<trimkit::RayMeshInput> inputs(read.size());
              for (std::size_t m = 0; m < read.size(); m++) {
                  inputs[m] = {read[m].vertices.data(), read[m].vertices.shape(0), std::move(read[m].faces)};
              }
              // copying reads every value, so it runs without the GIL too
              const trimkit::GilReleased release;
              return trimkit::RayMeshes(std::move(inputs), use_gpu.value);
          }()) {
        if (!scene_.device_failure().empty()) {
            trimkit::warn("use_gpu: " + scene_.device_failure() + "; this scene answers on the CPU");
        }
        // the default instances, each mesh once where it is, placed now
        trimkit::Instances instances =
            trimkit::read_instances(trimkit::InstanceViews{}, scene_.size(), /*rigid=*/false);
        const trimkit::GilReleased release;
        last_ = scene_.place(std::move(instances));
    }

    // the boxes a query with these instances culls rays by, without building
    // a top level: the same placement a query makes
    Boxes boxes(const trimkit::InstanceMatrices& matrices_argument,
                const trimkit::InstanceIndex& index_argument) const {
        trimkit::Instances instances = trimkit::read_instances(
            trimkit::view_instances(index_argument, matrices_argument), scene_.size(), /*rigid=*/false);
        std::vector<double> boxes;
        {
            const trimkit::GilReleased release;
            boxes = scene_.boxes(std::move(instances));
        }
        return {boxes.data(), boxes.size() / 6};
    }

    std::size_t size() const { return scene_.size(); }

    // pickled as its meshes, exactly as it holds them, and `use_gpu` as asked:
    // a load without a device answers on the CPU, as a build without one does
    nb::dict constructor_arguments() const {
        nb::list meshes;
        for (std::size_t m = 0; m < scene_.size(); m++) {
            std::vector<double> vertices;
            std::vector<std::uint64_t> faces;
            scene_.as_given(m, &vertices, &faces);
            meshes.append(nb::make_tuple(trimkit::to_numpy_rows(std::move(vertices), {3}),
                                         trimkit::to_numpy_rows<std::int64_t>(std::move(faces), {3})));
        }
        const nb::dict arguments;
        arguments["meshes"] = meshes;
        arguments["use_gpu"] = use_gpu_;
        return arguments;
    }

    trimkit::Out<bool> intersect_any(const trimkit::RaysArgument& origins_argument,
                                     const trimkit::RaysArgument& directions_argument,
                                     const trimkit::InstanceMatrices& matrices_argument,
                                     const trimkit::InstanceIndex& index_argument) const {
        const trimkit::Rays origins = trimkit::read_rays(origins_argument, "origins");
        const trimkit::Rays directions = trimkit::read_rays(directions_argument, "directions");
        const std::size_t n = trimkit::count_rays(origins.vectors, directions.vectors);
        const std::shared_ptr<const trimkit::RayTopLevel> top = top_level(matrices_argument, index_argument);
        std::vector<std::uint8_t> hits;
        warn_fallback(trimkit::with_pool([&](const trimkit::parallel::Stop& stop) {
            return scene_.intersect_any(*top, origins.vectors, directions.vectors, n, stop, &hits);
        }));
        return trimkit::to_numpy<bool>(std::move(hits), {n});
    }

    RayHitColumns intersect(const trimkit::RaysArgument& origins_argument,
                            const trimkit::RaysArgument& directions_argument,
                            const trimkit::InstanceMatrices& matrices_argument,
                            const trimkit::InstanceIndex& index_argument, const MaximumRayHits& maximum_hits) const {
        if (maximum_hits && *maximum_hits < 1) {
            trimkit::refuse("maximum_hits must be at least 1");
        }
        const std::size_t most =
            maximum_hits ? trimkit::at_most(*maximum_hits) : std::numeric_limits<std::size_t>::max();
        const trimkit::Rays origins = trimkit::read_rays(origins_argument, "origins");
        const trimkit::Rays directions = trimkit::read_rays(directions_argument, "directions");
        const std::size_t n = trimkit::count_rays(origins.vectors, directions.vectors);
        const std::shared_ptr<const trimkit::RayTopLevel> top = top_level(matrices_argument, index_argument);
        trimkit::RayCrossings columns;
        warn_fallback(trimkit::with_pool([&](const trimkit::parallel::Stop& stop) {
            return scene_.intersect(*top, origins.vectors, directions.vectors, n, most, stop, &columns);
        }));
        const std::size_t total = columns.rays.size();
        return {trimkit::to_numpy<std::int64_t>(std::move(columns.rays), {total}),
                trimkit::to_numpy<std::int64_t>(std::move(columns.instances), {total}),
                trimkit::to_numpy<std::int64_t>(std::move(columns.faces), {total}),
                trimkit::to_numpy<double>(std::move(columns.distances), {total})};
    }

private:
    // The top level for these instances: the remembered one if they are
    // exactly the arrays it was built from, else a new one, remembered. The
    // arrays are viewed once, for the comparison and for the read after it
    std::shared_ptr<const trimkit::RayTopLevel> top_level(const trimkit::InstanceMatrices& matrices_argument,
                                                          const trimkit::InstanceIndex& index_argument) const {
        const trimkit::InstanceViews views = trimkit::view_instances(index_argument, matrices_argument);
        std::shared_ptr<const trimkit::RayTopLevel> last;
        {
            const nb::ft_lock_guard guard(last_lock_);
            last = last_;
        }
        if (trimkit::same_instances(views, trimkit::RayMeshes::placed_from(*last))) {
            return last;
        }
        trimkit::Instances instances = trimkit::read_instances(views, scene_.size(), /*rigid=*/false);
        std::shared_ptr<const trimkit::RayTopLevel> built;
        {
            const trimkit::GilReleased release;
            built = scene_.place(std::move(instances));
        }
        {
            const nb::ft_lock_guard guard(last_lock_);
            last_ = built;
        }
        return built;
    }

    // a device fell back to the CPU: said once per scene, with the GIL held
    void warn_fallback(const std::string& reason) const {
        if (!reason.empty() && !warned_.exchange(true)) {
            trimkit::warn("use_gpu: " + reason + "; the CPU answered");
        }
    }

    // whether it was asked to trace on a device, which a pickle asks again
    bool use_gpu_ = false;
    trimkit::RayMeshes scene_;
    // the top level last placed, swapped whole: a query keeps the one it read.
    // Every read and swap happens with the GIL held, which serializes them, so
    // the lock is nothing on GIL builds; on free-threaded ones it is CPython's
    // own PyMutex, held for one shared_ptr copy
    mutable nb::ft_mutex last_lock_;
    mutable std::shared_ptr<const trimkit::RayTopLevel> last_;
    // whether this scene has said that the CPU answered for the device
    mutable std::atomic<bool> warned_{false};
};

}  // namespace

static void bind_bvh(nb::module_& m) {
    trimkit::picklable_class<RayScene>(m, "RayScene",
                                       R"(Meshes, each built once into a double precision BVH, and the only
thing rays are cast against.

Every query names its instances (which mesh, placed where), as a
CollisionScene's do, so a bare query casts at each mesh once, where it
is. `intersect` answers where each ray crosses the surface, nearest first
and as many as asked; `intersect_any` only whether it does. Instances
are placed under a top-level BVH built from exactly the arrays a query
gives; the scene remembers the last one, so a query naming
the same instances again (the same bytes) costs only comparing them, and
moving anything costs that top level, never a mesh. Answers never depend
on what was asked before.

Parameters
----------
meshes : sequence of Mesh
  The meshes to place, each a (vertices, faces) pair of (n, 3) float64
  vertices and (m, 3) int64 triangle indices into them. Copied once
  into tinybvh's float64 layout, which the scene owns, so a mesh
  can never change under the threads sharing the scene.
use_gpu : bool, default False
  Trace on a Vulkan device (the first GPU that computes in double
  precision) instead of the CPU: the same trees, the same traversal, so
  the same answers to the byte, sooner for large batches of rays. When
  no such device can be used, now or later, the scene warns once (a
  UserWarning naming why) and the CPU answers.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> scene = trimkit.RayScene([(vertices, faces)])
>>> len(scene)
1)")
        .def(nb::init<const trimkit::Meshes&, trimkit::Flag>(), nb::arg("meshes"), nb::kw_only(),
             nb::arg("use_gpu") = false)
        .def("__len__", &RayScene::size, R"(How many meshes the scene was built from.

Returns
-------
count : int
  The number of meshes the scene was built from.)")
        .def("boxes", &RayScene::boxes, nb::arg("instance_matrices").none() = nb::none(),
             nb::arg("instance_index").none() = nb::none(),
             R"(The box each instance is culled by, as a query with these instances
places it: what the scene thinks it holds.

Parameters
----------
instance_matrices : (k, 4, 4) float64, optional
  Each instance's affine transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.

Returns
-------
boxes : Boxes
  One 3D box per instance, in its order, holding every face it places
  (an instance of a mesh without faces: a point at its translation).

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> boxes = trimkit.RayScene([(vertices, faces)]).boxes()
>>> boxes.bounds.tolist()
[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])")
        .def("intersect", &RayScene::intersect, nb::arg("origins"), nb::arg("directions"),
             nb::arg("instance_matrices").none() = nb::none(), nb::arg("instance_index").none() = nb::none(),
             nb::kw_only(), nb::arg("maximum_hits").none() = nb::none(),
             R"(Where each ray crosses the scene's surface, nearest first, from either side.

Parameters
----------
origins : (r, 3) or (3,) float64 or float32
  Where each ray starts, or where every ray starts.
directions : (r, 3) or (3,) float64 or float32
  Each ray's direction, or every ray's, which need not be unit length.
instance_matrices : (k, 4, 4) float64, optional
  Each instance's affine transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.
maximum_hits : int, optional
  The most hits to report per ray, nearest first, which bounds the work
  at that many traversals per ray: 1 is the nearest hit of each ray.
  Defaults to every hit.

Returns
-------
rays : (h,) int64
  Ray of each hit, one row per hit, sorted by ray and then distance.
  A ray through a shared edge or vertex crosses the surface once there
  and gets one hit, so a closed mesh is crossed an odd number of times
  from inside and an even number from outside, whatever the ray's
  direction. Two surfaces closer together than about 1e-12 of the hit's
  distance or of the scene's farthest coordinate, whichever is more, count
  once; past a thousand times the scene, closer than a few ulps of the
  ray's origin or the hit.
instances : (h,) int64
  Instance of each hit.
faces : (h,) int64
  Face of each hit in its instance's mesh.
distances : (h,) float64
  Distance from the ray's origin to each hit.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> scene = trimkit.RayScene([(vertices, faces)])
>>> origins, directions = np.array([[0.2, 0.2, 1.0]]), np.array([[0.0, 0.0, -1.0]])
>>> rays, instances, faces, distances = scene.intersect(origins, directions)
>>> rays.tolist(), instances.tolist(), faces.tolist(), distances.tolist()
([0], [0], [0], [1.0]))")
        .def("intersect_any", &RayScene::intersect_any, nb::arg("origins"), nb::arg("directions"),
             nb::arg("instance_matrices").none() = nb::none(), nb::arg("instance_index").none() = nb::none(),
             R"(Whether each ray hits anything, stopping at the first triangle found.

Parameters
----------
origins : (r, 3) or (3,) float64 or float32
  Where each ray starts, or where every ray starts.
directions : (r, 3) or (3,) float64 or float32
  Each ray's direction, or every ray's, which need not be unit length.
instance_matrices : (k, 4, 4) float64, optional
  Each instance's affine transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.

Returns
-------
hits : (r,) bool
  True for every ray that hits a triangle.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> scene = trimkit.RayScene([(vertices, faces)])
>>> origins, directions = np.array([[0.2, 0.2, 1.0]]), np.array([[0.0, 0.0, -1.0]])
>>> hits = scene.intersect_any(origins, directions)
>>> hits.tolist()
[True])");

    trimkit::picklable_class<Boxes>(m, "Boxes",
                                    R"(Axis-aligned boxes built into a BVH for overlap and nearest queries: a
static, numpy-native replacement for a bulk-loaded R-tree.

Parameters
----------
boxes : (n, 2d) or (n, 2, d) float64
  Each box's minimum and maximum corner, in 2 or 3 dimensions: flat,
  (min..., max...), or as two corners. Copied into the tree the object
  owns, so the boxes can't change under it.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> len(tree)
2)")
        .def(nb::init<const trimkit::BoxesArgument&>(), nb::arg("boxes"))
        .def("__len__", &Boxes::size, R"(How many boxes it holds.

Returns
-------
count : int
  The number of boxes it holds.)")
        .def_prop_ro("bounds", &Boxes::bounds, nb::rv_policy::move,
                     R"(The box around every box.

Returns
-------
bounds : (2, d) float64
  The minimum and maximum corner over every box, or +inf then -inf
  with no boxes.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> bounds = tree.bounds
>>> bounds.tolist()
[[0.0, 0.0], [3.0, 3.0]])")
        .def("boxes", &Boxes::boxes,
             R"(Every box's two corners, with exactly the values given.

Returns
-------
boxes : (n, 2, d) float64
  Each box's minimum and maximum corner, a fresh copy.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> boxes = tree.boxes()
>>> boxes.shape
(2, 2, 2))")
        .def("intersection", &Boxes::intersection, nb::arg("queries"),
             R"(Every box overlapping one query box or each of many, touching included.

A superset of an R-tree's `intersection`: one query box, flat as
(min..., max...) or as its two corners, answers the boxes it overlaps,
and a batch of either answers every overlap at once.

Parameters
----------
queries : (2d,), (2, d), (m, 2d) or (m, 2, d) float64
  One query's or each query's minimum and maximum corner, in the
  boxes' dimension d: flat, (min..., max...), or as two corners.

Returns
-------
hits : (n,) or (k, 2) int64
  For one query, every box it overlaps, ascending. For many,
  rows of [query, box] for every overlap, sorted by query and
  then box.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> hits = tree.intersection(np.array([0.5, 0.5, 2.5, 2.5]))
>>> hits.tolist()
[0, 1]
>>> tree.intersection(np.array([[[0.5, 0.5], [2.5, 2.5]]])).tolist()
[[0, 0], [0, 1]])")
        .def("nearest", &Boxes::nearest, nb::arg("queries"), nb::kw_only(), nb::arg("maximum_hits") = 1,
             R"(The nearest boxes to one query box or each of many, nearest first.

A point is a query box with equal corners, and a box overlapping or
touching a query is at distance 0. Each query gets exactly
`maximum_hits` boxes, or every box when there are fewer. Of boxes at
the same distance, the lowest index comes first.

Parameters
----------
queries : (2d,), (2, d), (m, 2d) or (m, 2, d) float64
  One query's or each query's minimum and maximum corner, in the
  boxes' dimension d: flat, (min..., max...), or as two corners.
maximum_hits : int, default 1
  How many of the nearest boxes to answer per query, at least 1.

Returns
-------
hits : (n,) or (k, 2) int64
  For one query, its nearest boxes. For many, rows of [query, box],
  sorted by query, then distance, then box.
distances : (n,) or (k,) float64
  The distance from each hit's query to its box. On each axis the gap
  is how far apart they are, or 0 where they overlap. The distance is
  the square root of the squared gaps summed in axis order, the same
  float64 numpy computes. A gap under about 1e-162 squares to 0, so
  boxes that close but apart are at distance 0 too.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
>>> hits, distances = tree.nearest(np.array([4.0, 3.0, 4.0, 3.0]), maximum_hits=2)
>>> hits.tolist(), distances.tolist()
([1, 0], [1.0, 3.605551275463989]))");
}

// -- triangulate --------------------------------------------------------------
//
// Rings arrive the way shapely holds them, an exterior and a sequence of
// interiors, each read in place and handed to the unit (triangulate.cpp) as
// it is, with the GIL released.

// The engines `triangulate` takes (trimkit.hpp): each one's index in TRIANGULATE_ENGINES is its value
constexpr EnumMember TRIANGULATE_ENGINES[] = {
    {"EARCUT", "mapbox/earcut.hpp's ear clipping, the default: fastest on convex rings, and never refuses a ring."},
    {"CDT",
     "CDT's constrained Delaunay triangulation: tiles exactly with any holes, is faster on long concave "
     "rings, and refuses rings whose edges cross."}};
static_assert(std::size(TRIANGULATE_ENGINES) == 2 && static_cast<int>(trimkit::TriangulateEngine::EARCUT) == 0 &&
              static_cast<int>(trimkit::TriangulateEngine::CDT) == 1);

using RingArg = trimkit::Array<double, -1, 2>;
using Interiors = nb::typed<nb::sequence, RingArg>;

static std::tuple<trimkit::Out<double>, trimkit::Out<std::int64_t>> triangulate(
    const RingArg& exterior_argument, const Interiors& interiors, std::optional<trimkit::TriangulateEngine> engine) {
    const trimkit::In<double, -1, 2> exterior = trimkit::read(exterior_argument, "exterior");
    // every interior viewed in place, held so the rings below stay valid
    std::vector<trimkit::In<double, -1, 2>> holes(nb::len(interiors));
    for (std::size_t h = 0; h < holes.size(); h++) {
        holes[h] = trimkit::read<double, -1, 2>(interiors[h], "interiors[" + std::to_string(h) + "]");
    }
    std::vector<trimkit::PolygonRing> rings{{exterior.data(), exterior.shape(0)}};
    for (const trimkit::In<double, -1, 2>& hole : holes) {
        rings.push_back({hole.data(), hole.shape(0)});
    }
    std::vector<double> vertices;
    std::vector<std::uint64_t> faces;
    {
        const trimkit::GilReleased release;
        trimkit::triangulate_polygon(rings, engine.value_or(trimkit::TriangulateEngine::EARCUT), &vertices, &faces);
    }
    return {trimkit::to_numpy_rows(std::move(vertices), {2}),
            trimkit::to_numpy_rows<std::int64_t>(std::move(faces), {3})};
}

static void bind_triangulate(nb::module_& m) {
    trimkit::bind_enum<trimkit::TriangulateEngine>(m, "TriangulateEngine", "How `triangulate` fills a polygon.",
                                                   TRIANGULATE_ENGINES);
    m.def("triangulate", &triangulate, nb::arg("exterior"), nb::arg("interiors") = nb::tuple(), nb::kw_only(),
          nb::arg("engine").none() = nb::none(),
          R"(Triangulate a polygon with holes.

Parameters
----------
exterior : (n, 2) float64
  The polygon's outer ring, as shapely's `exterior.coords`. A repeated
  closing vertex is fine.
interiors : sequence of (n_i, 2) float64
  Each hole's ring, as shapely's `interiors[i].coords`.
engine : None or TriangulateEngine
  EARCUT, the default, is fastest on convex rings and never refuses a
  ring, but can leave out or overlap area with many holes. CDT is a
  constrained Delaunay triangulation: it tiles the polygon exactly with
  any holes, in either winding, and raises ValueError where ring edges
  cross.

Returns
-------
vertices : (N, 2) float64
  Every ring stacked in order, the exterior then each interior.
faces : (t, 3) int64
  Counter-clockwise triangles indexing `vertices`. Of exactly equal
  vertices, such as a repeated closing vertex, only the first is used.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> exterior = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
>>> vertices, faces = trimkit.triangulate(exterior)
>>> faces.tolist()
[[2, 3, 0], [2, 0, 1]])");
}

// -- meshoptimizer --------------------------------------------------------------
//
// The mesh read through the door as float64 vertices and uint32 faces, and
// simplified (meshoptimizer.cpp) with the GIL released for the whole call.

static std::tuple<trimkit::Out<double>, trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>> simplify(
    const trimkit::Array<double, -1, 3>& vertices_argument, const trimkit::Array<std::int64_t, -1, 3>& faces_argument,
    trimkit::Number<std::int64_t> face_count, trimkit::Number<double> maximum_error, trimkit::Flag lock_border) {
    const trimkit::Mesh<std::uint32_t> mesh = trimkit::read_mesh<std::uint32_t>(vertices_argument, faces_argument);
    std::vector<double> output_vertices;
    std::vector<std::int64_t> output_faces;
    std::vector<std::int64_t> vertex_sources;
    {
        // converting reads every value, so it runs without the GIL too
        const trimkit::GilReleased release;
        trimkit::simplify_mesh(mesh.vertices.data(), mesh.vertices.shape(0), mesh.faces, face_count, maximum_error,
                               lock_border, &output_vertices, &output_faces, &vertex_sources);
    }
    const std::size_t source_count = vertex_sources.size();
    return {trimkit::to_numpy_rows(std::move(output_vertices), {3}),
            trimkit::to_numpy_rows(std::move(output_faces), {3}),
            trimkit::to_numpy(std::move(vertex_sources), {source_count})};
}

static void bind_meshoptimizer(nb::module_& m) {
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

// -- xatlas ---------------------------------------------------------------------
//
// The mesh read through the door, parametrized (xatlas.cpp) on the pool with
// the GIL released, where Ctrl-C stops it at xatlas's next progress report.

static std::tuple<trimkit::Out<std::int64_t>, trimkit::Out<std::int64_t>, trimkit::Out<double>> parametrize(
    const trimkit::Array<double, -1, 3>& vertices_argument, const trimkit::Array<std::int64_t, -1, 3>& faces_argument) {
    const trimkit::Mesh<std::uint32_t> mesh = trimkit::read_mesh<std::uint32_t>(vertices_argument, faces_argument);
    std::vector<std::int64_t> vertex_sources;
    std::vector<std::int64_t> faces;
    std::vector<double> uv;
    trimkit::with_pool([&](const trimkit::parallel::Stop& stop) {
        trimkit::parametrize_mesh(mesh.vertices.data(), mesh.vertices.shape(0), mesh.faces, stop, &vertex_sources,
                                  &faces, &uv);
    });
    const std::size_t vertex_count = vertex_sources.size();
    return {trimkit::to_numpy(std::move(vertex_sources), {vertex_count}), trimkit::to_numpy_rows(std::move(faces), {3}),
            trimkit::to_numpy_rows(std::move(uv), {2})};
}

static void bind_xatlas(nb::module_& m) {
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
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> vertex_sources, faces, uv = trimkit.parametrize(vertices, faces)
>>> vertex_sources.tolist(), uv.shape
([0, 1, 2], (3, 2)))");
}

// -- vhacd ----------------------------------------------------------------------
//
// The mesh read through the door and decomposed (vhacd.cpp) on the pool with
// the GIL released, where Ctrl-C stops it at V-HACD's next progress report.

static trimkit::MeshOuts decompose(const trimkit::Array<double, -1, 3>& vertices_argument,
                                   const trimkit::Array<std::int64_t, -1, 3>& faces_argument,
                                   trimkit::Number<std::int64_t> maximum_hulls,
                                   trimkit::Number<std::int64_t> maximum_vertices,
                                   trimkit::Number<std::int64_t> resolution, trimkit::Number<double> maximum_error) {
    const trimkit::In<double, -1, 3> vertices = trimkit::read(vertices_argument, "vertices");
    const std::vector<std::uint32_t> triangles =
        trimkit::read_indices<std::uint32_t>(faces_argument, "faces", vertices.shape(0));
    std::vector<trimkit::ConvexHullMesh> hulls = trimkit::with_pool([&](const trimkit::parallel::Stop& stop) {
        return trimkit::decompose_mesh(vertices.data(), vertices.shape(0), triangles, maximum_hulls, maximum_vertices,
                                       resolution, maximum_error, stop);
    });
    nb::list meshes;
    for (trimkit::ConvexHullMesh& hull : hulls) {
        meshes.append(nb::make_tuple(trimkit::to_numpy_rows(std::move(hull.vertices), {3}),
                                     trimkit::to_numpy_rows(std::move(hull.faces), {3})));
    }
    return meshes;
}

static void bind_vhacd(nb::module_& m) {
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

// -- draco ----------------------------------------------------------------------
//
// glTF `KHR_draco_mesh_compression` (draco.cpp). Attributes arrive as arrays
// of any glTF component type, each read through the door as it is, and come
// back as arrays of the type the buffer holds.

namespace {

// numpy's dtype for each DracoComponent, in its order
constexpr nb::dlpack::dtype COMPONENT_DTYPES[] = {
    nb::dtype<std::int8_t>(),  nb::dtype<std::uint8_t>(),  nb::dtype<std::int16_t>(), nb::dtype<std::uint16_t>(),
    nb::dtype<std::int32_t>(), nb::dtype<std::uint32_t>(), nb::dtype<float>()};
static_assert(std::size(COMPONENT_DTYPES) == trimkit::DRACO_COMPONENTS);
// whether each DracoComponent's dtype is as wide as trimkit.hpp says
constexpr bool component_widths_agree() {
    for (std::size_t i = 0; i < trimkit::DRACO_COMPONENTS; i++) {
        if (COMPONENT_DTYPES[i].bits / 8 != trimkit::DRACO_COMPONENT_BYTES[i]) {
            return false;
        }
    }
    return true;
}
static_assert(component_widths_agree());

// The attribute kinds `draco_encode` takes: each one's index here is its
// `DracoKind` value, in the unit's order (trimkit.hpp)
enum class DracoKind : std::uint8_t {};
constexpr EnumMember DRACO_KIND_NAMES[] = {
    {"POSITION", "Vertex positions: float ones quantized to 14 bits by default."},
    {"NORMAL", "Vertex normals: float ones quantized to 10 bits by default."},
    {"TEXCOORD", "Texture coordinates: float ones quantized to 12 bits by default."},
    {"COLOR", "Vertex colors: float ones quantized to 10 bits by default."},
    {"GENERIC", "Anything else, stored losslessly by default."}};
static_assert(std::size(DRACO_KIND_NAMES) == trimkit::DRACO_KINDS);

// draco reads and writes its little-endian format by copying raw bytes, with
// no byte order of its own: on a big-endian machine every value would come
// out swapped, so it refuses there before reading anything
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
constexpr bool BIG_ENDIAN_MACHINE = true;
#else
constexpr bool BIG_ENDIAN_MACHINE = false;
#endif

void refuse_big_endian() {
    if (BIG_ENDIAN_MACHINE) {
        trimkit::not_implemented_error(
            "draco reads and writes only little-endian data, and this machine is big-endian");
    }
}

nb::bytes draco_encode(const trimkit::Array<std::int64_t, -1, 3>& faces_argument,
                       const trimkit::AnyArguments& attributes, const trimkit::Array<std::int64_t, -1>& kinds_argument,
                       const std::optional<trimkit::Number<std::int64_t>>& bits_argument) {
    refuse_big_endian();
    const trimkit::In<std::int64_t, -1> kind_numbers = trimkit::read(kinds_argument, "kinds");
    // every attribute read and checked while we hold the GIL, held so its
    // view stays valid
    std::vector<trimkit::AnyIn> arrays(nb::len(attributes));
    std::vector<trimkit::DracoAttributeInput> inputs(arrays.size());
    for (std::size_t i = 0; i < arrays.size(); i++) {
        const std::string name = "attributes[" + std::to_string(i) + "]";
        arrays[i] = trimkit::read_any(attributes[i], name);
        const nb::dlpack::dtype* type =
            std::find(std::begin(COMPONENT_DTYPES), std::end(COMPONENT_DTYPES), arrays[i].dtype());
        if (type == std::end(COMPONENT_DTYPES)) {
            trimkit::type_error(name + " is not a glTF component type");
        }
        inputs[i].data = arrays[i].data();
        inputs[i].component = static_cast<trimkit::DracoComponent>(type - std::begin(COMPONENT_DTYPES));
        inputs[i].components = arrays[i].ndim() == 2 ? arrays[i].shape(1) : 1;
        if (arrays[i].ndim() < 1 || arrays[i].ndim() > 2 || inputs[i].components < 1 || inputs[i].components > 16 ||
            arrays[i].shape(0) != arrays[0].shape(0)) {
            trimkit::refuse(name + " must be (n,) or (n, c<=16) like the rest");
        }
    }
    if (arrays.empty()) {
        trimkit::refuse("at least one attribute is required");
    }
    const std::size_t count = arrays[0].shape(0);
    if (count == 0) {
        trimkit::refuse("draco can't encode a primitive without vertices");
    }
    const std::vector<std::uint32_t> triangles = trimkit::read_indices<std::uint32_t>(faces_argument, "faces", count);
    const std::vector<std::int64_t> kinds(kind_numbers.data(), kind_numbers.data() + kind_numbers.shape(0));
    std::vector<char> encoded;
    {
        const trimkit::GilReleased release;
        encoded =
            trimkit::draco_encode_mesh(triangles, count, inputs, kinds,
                                       bits_argument ? std::optional<std::int64_t>(*bits_argument) : std::nullopt);
    }
    return nb::bytes(encoded.data(), encoded.size());
}

std::tuple<trimkit::Out<std::int64_t>, trimkit::AnyOuts> draco_decode(
    const trimkit::Array<std::uint8_t, -1>& buffer_argument,
    const trimkit::Array<std::int64_t, -1>& attributes_argument) {
    refuse_big_endian();
    // bytes and memoryviews are uint8 buffers already, viewed without a copy
    const trimkit::In<std::uint8_t, -1> buffer = trimkit::read(buffer_argument, "buffer");
    const trimkit::In<std::int64_t, -1> ids_given = trimkit::read(attributes_argument, "draco_attributes");
    const std::vector<std::int64_t> ids(ids_given.data(), ids_given.data() + ids_given.shape(0));
    std::vector<std::int64_t> faces;
    std::vector<trimkit::DracoAttributeOutput> decoded;
    {
        const trimkit::GilReleased release;
        trimkit::draco_decode_mesh(buffer.data(), buffer.size(), ids, &faces, &decoded);
    }
    nb::list arrays;
    for (trimkit::DracoAttributeOutput& a : decoded) {
        const nb::dlpack::dtype type = COMPONENT_DTYPES[static_cast<std::size_t>(a.component)];
        const std::size_t points =
            a.values.size() / (a.components * trimkit::DRACO_COMPONENT_BYTES[static_cast<std::size_t>(a.component)]);
        arrays.append(a.components == 1 ? trimkit::to_numpy(std::move(a.values), {points}, type)
                                        : trimkit::to_numpy(std::move(a.values), {points, a.components}, type));
    }
    return {trimkit::to_numpy_rows(std::move(faces), {3}), nb::borrow<trimkit::AnyOuts>(nb::tuple(arrays))};
}

}  // namespace

static void bind_draco(nb::module_& m) {
    trimkit::bind_enum<DracoKind>(m, "DracoKind", "What a draco attribute holds, as `draco_encode` takes it.",
                                  DRACO_KIND_NAMES);
    m.def("draco_encode", &draco_encode, nb::arg("faces"), nb::arg("attributes"), nb::arg("kinds"), nb::kw_only(),
          nb::arg("quantization_bits").none() = nb::none(),
          R"(Compress a glTF triangle primitive, keeping vertex order.

Parameters
----------
faces : (f, 3) int64
  Triangles indexing the attribute rows.
attributes : sequence of (n,) or (n, c) arrays
  One array per glTF attribute, each a glTF component type. Draco
  attribute i is attributes[i], which is what the extension's
  `attributes` JSON maps each glTF name to. Integers must fit int32, with
  `max - min` at most `2**31 - 2`.
kinds : (k,) int64
  Each attribute's `DracoKind`: POSITION, NORMAL, TEXCOORD, COLOR or
  GENERIC for anything else.
quantization_bits : int, optional
  What every float attribute is rounded onto, across its extent: a position
  moves at most `extent * 2**-(bits + 1)`, so this is how much precision
  survives. 0 stores them losslessly, up to 24, where a step is float32's
  own precision at the top of the extent. By default each kind's own:
  POSITION 14, NORMAL 10, TEXCOORD 12 and COLOR 10 bits, GENERIC lossless.
  Other attributes are never rounded.

Returns
-------
buffer : bytes
  The draco buffer for the extension's `bufferView`.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> kinds = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
>>> buffer = trimkit.exchange.draco_encode(faces, [vertices.astype(np.float32)], kinds)
>>> type(buffer).__name__
'bytes')");
    m.def("draco_decode", &draco_decode, nb::arg("buffer"), nb::arg("draco_attributes"),
          R"(Decompress a glTF draco buffer, decoding it once.

Parameters
----------
buffer : bytes, memoryview or (n,) uint8
  The contents of the extension's `bufferView`.
draco_attributes : (k,) int64
  The draco attribute numbers to return, from the extension's
  `attributes` JSON, in the order wanted.

Returns
-------
faces : (f, 3) int64
  Triangles indexing the attribute rows.
attributes : tuple of (n,) or (n, c) arrays
  One per requested draco attribute, in the requested order
  and stored component type.

Notes
-----
It has no size limit: a buffer of a few bytes can really hold
millions of points. What it declares beyond the memory available
raises MemoryError.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
>>> faces = np.array([[0, 1, 2]], dtype=np.int64)
>>> kinds = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
>>> buffer = np.frombuffer(trimkit.exchange.draco_encode(faces, [vertices.astype(np.float32)], kinds), np.uint8)
>>> faces, attributes = trimkit.exchange.draco_decode(buffer, np.array([0], dtype=np.int64))
>>> faces.tolist(), attributes[0].tolist()
([[0, 1, 2]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))");
}

// -- collision -------------------------------------------------------------------
//
// A `CollisionScene` holds its meshes as the unit (collision.cpp) built them,
// once, with the GIL released. Each query reads its instances and pairs
// through the door, then runs on the pool with the GIL released, where Ctrl-C
// stops it between ranges. A `CollisionResult` holds what one found as numpy
// arrays.

namespace {

// every mesh convex or none, or a mask of which
using ConvexArg = std::variant<trimkit::Flag, trimkit::Array<bool, -1>>;
using PairsArg = trimkit::Array<std::int64_t, -1, 2>;
using FacesArg = trimkit::Array<std::int64_t, -1, 2>;
using DistancesArg = trimkit::Array<double, -1>;
using PointsArg = trimkit::Array<double, -1, 2, 3>;
using NormalsArg = trimkit::Array<double, -1, 3>;

// an array argument copied, read once (a conversion warns once): a row
// count other than `rows`, when given, is refused
template <typename T, nb::ssize_t... Shape>
trimkit::Out<T> copied_rows(const trimkit::Array<T, Shape...>& argument, const char* name,
                            std::optional<std::size_t> expected = std::nullopt) {
    const trimkit::In<T, Shape...> given = trimkit::view(argument, name);
    const std::size_t rows = expected.value_or(given.shape(0));
    if (given.shape(0) != rows) {
        trimkit::refuse(std::string(name) + " must have one row per pair");
    }
    std::vector<T> values(given.data(), given.data() + given.size());
    if constexpr (sizeof...(Shape) == 1) {
        return trimkit::to_numpy(std::move(values), {rows});
    } else if constexpr (sizeof...(Shape) == 2) {
        return trimkit::to_numpy(std::move(values), {rows, static_cast<std::size_t>(given.shape(1))});
    } else {
        return trimkit::to_numpy(std::move(values), {rows, static_cast<std::size_t>(given.shape(1)),
                                                     static_cast<std::size_t>(given.shape(2))});
    }
}

// the same for an optional one: None stays None
template <typename T, nb::ssize_t... Shape>
std::optional<trimkit::Out<T>> copied(const trimkit::Array<T, Shape...>& argument, const char* name, std::size_t rows) {
    if (argument.is_none()) {
        return std::nullopt;
    }
    return copied_rows(argument, name, rows);
}

// What a query found, one row per pair or per contact: which instances, and,
// when the query measured them, the faces, signed distances, witness points
// and normals, each None when it didn't
class CollisionResult {
public:
    // from Python, as a pickle loads one: checked and copied like any input
    CollisionResult(const PairsArg& pairs_argument, const FacesArg& faces_argument,
                    const DistancesArg& distances_argument, const PointsArg& points_argument,
                    const NormalsArg& normals_argument)
        : pairs_(copied_rows(pairs_argument, "pairs")) {
        const std::size_t rows = size();
        faces_ = copied(faces_argument, "faces", rows);
        distances_ = copied(distances_argument, "distances", rows);
        points_ = copied(points_argument, "points", rows);
        normals_ = copied(normals_argument, "normals", rows);
    }

    // from a query that found only pairs
    explicit CollisionResult(trimkit::Out<std::int64_t> pairs) : pairs_(std::move(pairs)) {}

    // from a query's rows, every field measured
    explicit CollisionResult(const std::vector<trimkit::Unset<trimkit::CollisionRow>>& rows) {
        const std::size_t count = rows.size();
        std::vector<std::int64_t> pairs(2 * count);
        std::vector<std::int64_t> faces(2 * count);
        std::vector<double> distances(count);
        std::vector<double> points(6 * count);
        std::vector<double> normals(3 * count);
        for (std::size_t r = 0; r < count; r++) {
            const trimkit::CollisionRow& found = rows[r].value;
            std::copy_n(found.pair, 2, &pairs[2 * r]);
            std::copy_n(found.faces, 2, &faces[2 * r]);
            distances[r] = found.distance;
            std::copy_n(found.points, 6, &points[6 * r]);
            std::copy_n(found.normal, 3, &normals[3 * r]);
        }
        pairs_ = trimkit::to_numpy_rows(std::move(pairs), {2});
        faces_ = trimkit::to_numpy_rows(std::move(faces), {2});
        distances_ = trimkit::to_numpy(std::move(distances), {count});
        points_ = trimkit::to_numpy_rows(std::move(points), {2, 3});
        normals_ = trimkit::to_numpy_rows(std::move(normals), {3});
    }

    std::size_t size() const { return pairs_.shape(0); }
    trimkit::Out<std::int64_t> pairs() const { return pairs_; }
    std::optional<trimkit::Out<std::int64_t>> faces() const { return faces_; }
    std::optional<trimkit::Out<double>> distances() const { return distances_; }
    std::optional<trimkit::Out<double>> points() const { return points_; }
    std::optional<trimkit::Out<double>> normals() const { return normals_; }

    // pickled as its arrays, None where the query measured nothing
    nb::dict constructor_arguments() const {
        const nb::dict arguments;
        arguments["pairs"] = pairs_;
        arguments["faces"] = faces_;
        arguments["distances"] = distances_;
        arguments["points"] = points_;
        arguments["normals"] = normals_;
        return arguments;
    }

private:
    trimkit::Out<std::int64_t> pairs_;
    std::optional<trimkit::Out<std::int64_t>> faces_;
    std::optional<trimkit::Out<double>> distances_;
    std::optional<trimkit::Out<double>> points_;
    std::optional<trimkit::Out<double>> normals_;
};

class CollisionScene {
public:
    // every mesh and the mask read first, then built without the GIL
    CollisionScene(const trimkit::Meshes& meshes, const ConvexArg& convex_argument)
        : meshes_([&] {
              const std::vector<trimkit::Mesh<std::uint32_t>> read = trimkit::read_meshes<std::uint32_t>(meshes);
              const trimkit::Flag* every = std::get_if<trimkit::Flag>(&convex_argument);
              trimkit::In<bool, -1> mask;
              if (every == nullptr) {
                  mask = trimkit::read(std::get<1>(convex_argument), "convex");
                  if (mask.shape(0) != read.size()) {
                      trimkit::refuse("convex must have one row per mesh");
                  }
              }
              std::vector<trimkit::CollisionMeshInput> inputs(read.size());
              for (std::size_t i = 0; i < read.size(); i++) {
                  inputs[i] = {read[i].vertices.data(), read[i].vertices.shape(0), read[i].faces.data(),
                               read[i].faces.size() / 3,
                               every != nullptr ? every->value : trimkit::mask_bytes(mask)[i] != 0};
              }
              const trimkit::GilReleased release;
              return trimkit::CollisionMeshes(inputs);
          }()) {}

    std::size_t size() const { return meshes_.size(); }

    // pickled as its meshes as given and which of them are convex
    nb::dict constructor_arguments() const {
        nb::list meshes;
        std::vector<std::uint8_t> convex(meshes_.size());
        for (std::size_t i = 0; i < meshes_.size(); i++) {
            std::vector<double> vertices;
            std::vector<std::int64_t> faces;
            meshes_.as_given(i, &vertices, &faces);
            meshes.append(nb::make_tuple(trimkit::to_numpy_rows(std::move(vertices), {3}),
                                         trimkit::to_numpy_rows(std::move(faces), {3})));
            convex[i] = meshes_.convex(i) ? 1 : 0;
        }
        const std::size_t count = convex.size();
        const nb::dict arguments;
        arguments["meshes"] = meshes;
        arguments["convex"] = trimkit::to_numpy<bool>(std::move(convex), {count});
        return arguments;
    }

    CollisionResult collides(const trimkit::InstanceMatrices& matrices_argument,
                             const trimkit::InstanceIndex& index_argument, const PairsArg& pairs_argument,
                             trimkit::Number<double> margin, trimkit::Flag return_distance,
                             trimkit::Number<std::int64_t> maximum_contacts) const {
        const trimkit::Instances instances = rigid_instances(matrices_argument, index_argument);
        const std::optional<std::vector<std::int64_t>> pairs = read_pairs(pairs_argument);
        std::vector<trimkit::Unset<std::int64_t>> touching;
        std::vector<trimkit::Unset<trimkit::CollisionRow>> contacts;
        trimkit::with_pool([&](const trimkit::parallel::Stop& stop) {
            meshes_.collides(instances, pairs, margin, maximum_contacts, stop, &touching,
                             return_distance ? &contacts : nullptr);
        });
        if (return_distance) {
            return CollisionResult(contacts);
        }
        return CollisionResult(trimkit::to_numpy_rows<std::int64_t>(std::move(touching), {2}));
    }

    CollisionResult distances(const trimkit::InstanceMatrices& matrices_argument,
                              const trimkit::InstanceIndex& index_argument, const PairsArg& pairs_argument,
                              trimkit::Number<double> upper_bound) const {
        const trimkit::Instances instances = rigid_instances(matrices_argument, index_argument);
        const std::optional<std::vector<std::int64_t>> pairs = read_pairs(pairs_argument);
        return CollisionResult(trimkit::with_pool([&](const trimkit::parallel::Stop& stop) {
            return meshes_.distances(instances, pairs, upper_bound, stop);
        }));
    }

    // the box each instance is culled by, as these instances place it
    Boxes boxes(const trimkit::InstanceMatrices& matrices_argument,
                const trimkit::InstanceIndex& index_argument) const {
        const trimkit::Instances instances = rigid_instances(matrices_argument, index_argument);
        std::vector<double> boxes;
        {
            const trimkit::GilReleased release;
            boxes = meshes_.boxes(instances);
        }
        return {boxes.data(), boxes.size() / 6};
    }

private:
    // Which mesh each instance places and where, read and checked, rigid
    // transforms only: `instance_index` and `instance_matrices` may each be
    // None, for one instance per mesh and the identity
    trimkit::Instances rigid_instances(const trimkit::InstanceMatrices& matrices_argument,
                                       const trimkit::InstanceIndex& index_argument) const {
        return trimkit::read_instances(trimkit::view_instances(index_argument, matrices_argument), meshes_.size(),
                                       /*rigid=*/true);
    }

    // the instance pairs a query names, or none for every pair, copied for
    // the unit to check the copy, as faces are: what indexes our memory is
    // never read from the caller's again
    static std::optional<std::vector<std::int64_t>> read_pairs(const PairsArg& pairs_argument) {
        if (pairs_argument.is_none()) {
            return std::nullopt;
        }
        const trimkit::In<std::int64_t, -1, 2> given = trimkit::read(pairs_argument, "pairs");
        return std::vector<std::int64_t>(given.data(), given.data() + (2 * given.shape(0)));
    }

    trimkit::CollisionMeshes meshes_;
};

}  // namespace

static void bind_collision(nb::module_& m) {
    trimkit::picklable_class<CollisionResult>(m, "CollisionResult",
                                              R"(What a collision or distance query found: one row per pair of
instances, or per contact of a pair, as `CollisionScene.collides` and
`CollisionScene.distances` return it.

Every row names its pair of instances. A query that measured them also
fills each row's faces, signed distance, witness points and normal, and
each of those is None when it didn't, as `collides` without
`return_distance`: an answer not computed is never a placeholder.

Parameters
----------
pairs : (c, 2) int64
  The instances of each row.
faces : (c, 2) int64, optional
  The face of each instance's mesh the row is at, or -1 for a convex
  solid, which has no face to name.
distances : (c,) float64, optional
  The signed distance between the pair at the row, negative by how deep
  they overlap.
points : (c, 2, 3) float64, optional
  The witness point on each instance, in the scene: closest when apart,
  deepest when overlapping.
normals : (c, 3) float64, optional
  The unit direction from the first instance toward the second along
  which the distance is measured.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> result = trimkit.CollisionResult(np.array([[0, 1]], dtype=np.int64))
>>> len(result), result.distances is None
(1, True))")
        .def(nb::init<const PairsArg&, const FacesArg&, const DistancesArg&, const PointsArg&, const NormalsArg&>(),
             nb::arg("pairs"), nb::arg("faces").none() = nb::none(), nb::arg("distances").none() = nb::none(),
             nb::arg("points").none() = nb::none(), nb::arg("normals").none() = nb::none())
        .def("__len__", &CollisionResult::size, R"(How many rows it holds.

Returns
-------
count : int
  The number of rows it holds: pairs, or contacts.)")
        .def_prop_ro("pairs", &CollisionResult::pairs, nb::rv_policy::move,
                     R"(The instances of each row.

Returns
-------
pairs : (c, 2) int64
  Each row's pair of instances, first then second.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(vertices, triangles)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> placed[1, 0, 3] = 0.5
>>> result = scene.collides(placed, np.zeros(2, dtype=np.int64), return_distance=True)
>>> pairs = result.pairs
>>> pairs.tolist()
[[0, 1]])")
        .def_prop_ro("faces", &CollisionResult::faces, nb::rv_policy::move,
                     R"(The face of each instance's mesh each row is at, or None.

Returns
-------
faces : (c, 2) int64 or None
  Each row's face of each instance's mesh, -1 for a convex solid; None
  when the query measured nothing.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(vertices, triangles)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> placed[1, 0, 3] = 0.5
>>> result = scene.collides(placed, np.zeros(2, dtype=np.int64), return_distance=True)
>>> faces = result.faces
>>> faces.tolist()
[[-1, -1]])")
        .def_prop_ro("distances", &CollisionResult::distances, nb::rv_policy::move,
                     R"(The signed distance of each row, or None.

Returns
-------
distances : (c,) float64 or None
  Each row's signed distance, negative by how deep the pair overlaps;
  None when the query measured nothing.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(vertices, triangles)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> placed[1, 0, 3] = 0.5
>>> result = scene.collides(placed, np.zeros(2, dtype=np.int64), return_distance=True)
>>> distances = result.distances
>>> bool(distances[0] < 0.0)
True)")
        .def_prop_ro("points", &CollisionResult::points, nb::rv_policy::move,
                     R"(The witness point on each instance of each row, or None.

Returns
-------
points : (c, 2, 3) float64 or None
  Each row's witness point on each instance, in the scene; None when
  the query measured nothing.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(vertices, triangles)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> placed[1, 0, 3] = 0.5
>>> result = scene.collides(placed, np.zeros(2, dtype=np.int64), return_distance=True)
>>> points = result.points
>>> points.shape
(1, 2, 3))")
        .def_prop_ro("normals", &CollisionResult::normals, nb::rv_policy::move,
                     R"(The unit normal of each row, or None.

Returns
-------
normals : (c, 3) float64 or None
  Each row's unit direction from the first instance toward the second;
  None when the query measured nothing.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(vertices, triangles)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> placed[1, 0, 3] = 0.5
>>> result = scene.collides(placed, np.zeros(2, dtype=np.int64), return_distance=True)
>>> normals = result.normals
>>> bool(np.isclose(np.linalg.norm(normals[0]), 1.0))
True)");

    trimkit::picklable_class<CollisionScene>(m, "CollisionScene",
                                             R"(Meshes to place and query against each other. A scene never changes:
each query names its own instances, so threads may share one.

Every query takes its instances (which mesh, placed where) and the pairs
of them to test, all optional, so a bare `scene.collides()` asks about the
scene as built: each mesh once, where it is, every pair. A robot passes its
links' transforms and the pairs of links not joined to each other.

`pairs` names exactly the pairs to test, which is how anything is left
out: a robot's ignored links, obstacles against each other. A path is one
query: N poses of L links are N * L instances, each pose's pairs named
(`pairs=None` would pair a link with itself at another pose), and the
pairs barely change from call to call, so build them once. Distances over
every pair are O(k**2) in memory. For gradients, use convex meshes: two
surfaces are 0 apart with no normal once they touch.

A convex mesh is tested as a solid and any other mesh as a surface, in
every pair: a mesh inside a convex one touches it, and a distance is
negative only when a convex solid is penetrated.

Parameters
----------
meshes : sequence of Mesh
  The meshes to place, each a (vertices, faces) pair of (n, 3) float64
  vertices and (m, 3) int64 triangle indices into them. Copied once
  into coal's own storage, which the scene owns, so a mesh can never
  change under the threads sharing the scene.
convex : bool or (n,) bool, optional
  Which meshes to measure as convex solids, such as convex hulls:
  all of them or none, or a mask of which. Not checked: a mesh that
  isn't convex gives wrong distances, never a crash. Defaults to
  False.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(vertices, faces)], convex=True)
>>> len(scene)
1)")
        .def(nb::init<const trimkit::Meshes&, const ConvexArg&>(), nb::arg("meshes"),
             nb::arg("convex") = trimkit::Flag{})
        .def("__len__", &CollisionScene::size, R"(How many meshes the scene was built from.

Returns
-------
count : int
  The number of meshes the scene was built from.)")
        .def("boxes", &CollisionScene::boxes, nb::arg("instance_matrices").none() = nb::none(),
             nb::arg("instance_index").none() = nb::none(),
             R"(The box each instance is culled by, as a query with these instances
places it: what the scene thinks it holds.

Parameters
----------
instance_matrices : (k, 4, 4) float64, optional
  Each instance's rigid transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.

Returns
-------
boxes : Boxes
  One 3D box per instance, in its order, holding every face it places
  (an instance of a mesh without faces: a point at its translation).

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(vertices, faces)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> boxes = scene.boxes()
>>> boxes.bounds.tolist()
[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])")
        .def("collides", &CollisionScene::collides, nb::arg("instance_matrices").none() = nb::none(),
             nb::arg("instance_index").none() = nb::none(), nb::kw_only(), nb::arg("pairs").none() = nb::none(),
             nb::arg("margin") = 0.0, nb::arg("return_distance") = trimkit::Flag{}, nb::arg("maximum_contacts") = 1,
             R"(The pairs of instances that touch, and optionally where.

Parameters
----------
instance_matrices : (k, 4, 4) float64, optional
  Each instance's rigid transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.
pairs : (p, 2) int64, optional
  The pairs of instances to test. Defaults to every pair, found by
  their bounding boxes.
margin : float
  Also count pairs closer than this.
return_distance : bool
  Also measure where each touching pair touches: one row per contact,
  with its faces, signed distance, witness points and normal. Without
  it only the pairs are found, as fast as the test allows.
maximum_contacts : int
  With `return_distance`, the most contacts to report per pair, which
  bounds the work: 1 is the deepest a convex pair has, and a mesh's
  first found. At least 1.

Returns
-------
result : CollisionResult
  Each pair of instances that touches or overlaps: rows of `pairs` in
  its order, or by default each touching i < j in ascending order, and
  with `return_distance` one row per contact of each. Its other fields
  are None without `return_distance`. Empty when nothing touches.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(vertices, faces)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> placed[1, 0, 3] = 0.5
>>> result = scene.collides(placed, np.zeros(2, dtype=np.int64))
>>> result.pairs.tolist(), result.distances is None
([[0, 1]], True))")
        .def("distances", &CollisionScene::distances, nb::arg("instance_matrices").none() = nb::none(),
             nb::arg("instance_index").none() = nb::none(), nb::kw_only(), nb::arg("pairs").none() = nb::none(),
             nb::arg("upper_bound") = std::numeric_limits<double>::infinity(),
             R"(The distance between each pair of instances, and where it is.

Parameters
----------
instance_matrices : (k, 4, 4) float64, optional
  Each instance's rigid transform from its mesh into the scene.
  Defaults to the identity.
instance_index : (k,) int64, optional
  Which mesh each instance places. Defaults to each mesh once.
pairs : (p, 2) int64, optional
  The pairs of instances to measure. Defaults to every pair, in the
  order of `numpy.triu_indices(k, 1)`.
upper_bound : float
  Pairs farther apart than this are infinitely far, with no points or
  normal, and are measured only as far as it takes to tell.

Returns
-------
result : CollisionResult
  One row per pair, every field filled. A pair with a convex mesh has a
  signed distance, negative by how deep it reaches into the solid; two
  surfaces are 0 once their triangles touch, with a NaN normal. A pair
  farther than `upper_bound` is infinitely far, with -1 faces and NaN
  points and normal.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
>>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
>>> scene = trimkit.CollisionScene([(vertices, faces)], convex=True)
>>> placed = np.tile(np.eye(4), (2, 1, 1))
>>> placed[1, 0, 3] = 2.0
>>> result = scene.distances(placed, np.zeros(2, dtype=np.int64))
>>> result.distances.tolist(), result.normals.tolist()
([1.0], [[1.0, 0.0, 0.0]]))");
}

NB_MODULE(trimkit, m) {  // NOLINT(performance-unnecessary-value-param): the macro passes it by value
    trimkit::init_numpy();
    trimkit::install_fork_handler();
    // before the interpreter finalizes, every call that released the GIL has
    // taken it back, so none loses its thread mid-computation (gil.hpp)
    nb::module_::import_("atexit").attr("register")(nb::cpp_function(&trimkit::exit_at_interpreter_exit));
    // the last paragraph names what this build compiled in, each upstream's
    // repository and commit, or `(system)` for a distribution's copy
    m.doc() = R"(Thin, stable-ABI numpy bindings to pinned C and C++ geometry libraries.

They depend only on `numpy`, read arrays of exactly the dtype shown (float64
ones finite and within +-1e30) in place when C-contiguous and copy any other
layout once, convert a dtype numpy casts safely (or a list) once with a
UserWarning, return fresh arrays that own their memory, and release the GIL
during compute.

Attributes
----------
__version__ : str
  This package's version.
Mesh : TypeAlias
  A triangle mesh as the scenes take it and `decompose` returns it: a
  (vertices, faces) pair of (n, 3) float64 vertices and (m, 3) int64 triangle
  indices into them, given as a tuple, list or any sequence of two.
exchange : module
  Reading and writing data formats: draco and SVG paths.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> isinstance(trimkit.__version__, str)
True
>>> mesh: trimkit.Mesh = (np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64))
>>> len(mesh)
2

Notes
-----
Built from: )" TRIMKIT_UPSTREAM ".";
    m.attr("__version__") = TRIMKIT_VERSION;
    // a triangle mesh as scenes take them: (n, 3) float64 vertices and
    // (m, 3) int64 faces, the shapes every docstring states
    const nb::object ndarray = nb::module_::import_("numpy.typing").attr("NDArray");
    const nb::module_ numpy = nb::module_::import_("numpy");
    m.attr("Mesh") = nb::module_::import_("builtins")
                         .attr("tuple")[nb::make_tuple(ndarray[numpy.attr("float64")], ndarray[numpy.attr("int64")])];
    // a module of the extension, importable as `trimkit.exchange` too
    nb::module_ exchange = m.def_submodule("exchange", R"(Reading and writing data formats: draco and SVG paths.

Attributes
----------
DracoKind : enum.IntEnum
  What a draco attribute holds, as `draco_encode` takes it.
SvgKind : enum.IntEnum
  What drew an SvgSegment, its `kind` field.
SvgSegment : numpy.dtype
  One segment of an SVG path, as `svg_parse` returns them and
  `svg_string` takes them.

Examples
--------
>>> import trimkit
>>> trimkit.exchange.svg_parse("M 0 0 L 1 1")[1].dtype == trimkit.exchange.SvgSegment
True)");
    nb::module_::import_("sys").attr("modules")["trimkit.exchange"] = exchange;
    bind_bvh(m);
    bind_triangulate(m);
    bind_vhacd(m);
    bind_xatlas(m);
    bind_draco(exchange);
    bind_svg(exchange);
    bind_meshoptimizer(m);
    bind_collision(m);
    bind_contour(m);
}
