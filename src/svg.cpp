// SVG path data to and from indexed vertices and segments, with our
// header-only `svgpath`.
//
// The parser fills a vertex array and a std::vector<svgpath::Segment>, both
// handed to numpy without a copy; the segments are viewed as the `SvgSegment`
// structured dtype, built from the struct's own offsets so the two can never
// disagree. `svg_string` reads the same struct back out of a SvgSegment array.

#include "common.hpp"

#include <svgpath.hpp>

// what svgpath parses is exactly what the door lets `svg_string` read back
static_assert(svgpath::COORDINATE_LIMIT == trimkit::COORDINATE_LIMIT, "svgpath and numpy.hpp share one bound");

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

static nb::object make_segment_dtype() {
    nb::list names;
    nb::list formats;
    nb::list offsets;
    names.append("kind");
    formats.append("u1");
    offsets.append(offsetof(svgpath::Segment, kind));
    names.append("closes");
    formats.append("?");
    offsets.append(offsetof(svgpath::Segment, closes));
    names.append("points");
    formats.append("(4,)i8");
    offsets.append(offsetof(svgpath::Segment, points));
    names.append("radius");
    formats.append("(2,)f8");
    offsets.append(offsetof(svgpath::Segment, radius));
    names.append("rotation");
    formats.append("f8");
    offsets.append(offsetof(svgpath::Segment, rotation));
    names.append("angles");
    formats.append("(2,)f8");
    offsets.append(offsetof(svgpath::Segment, angles));
    nb::dict specification;
    specification["names"] = names;
    specification["formats"] = formats;
    specification["offsets"] = offsets;
    specification["itemsize"] = sizeof(svgpath::Segment);
    return nb::module_::import_("numpy").attr("dtype")(specification);
}

static nb::tuple svg_parse(std::string_view path, const nb::object& segment_dtype) {
    // how many segments a path holds isn't known until it is parsed, so its
    // vectors grow as it goes
    svgpath::Path parsed;
    {
        const nb::gil_scoped_release release;
        // like an SVG renderer, malformed data keeps everything before the
        // error rather than failing the whole path, so how far it got is unused
        static_cast<void>(svgpath::parse(path.data(), path.size(), &parsed));
    }
    const std::size_t vertex_count = parsed.vertices.size() / 2;
    const trimkit::Out<double> vertices = trimkit::to_numpy(std::move(parsed.vertices), {vertex_count, 2});
    return nb::make_tuple(vertices, trimkit::to_numpy_records(std::move(parsed.segments), segment_dtype));
}

static std::string svg_string(const trimkit::Array<double, -1, 2>& vertices_argument, const nb::handle& segments,
                              const nb::object& segment_dtype) {
    const trimkit::In<double, -1, 2> vertices = trimkit::read(vertices_argument, "vertices");
    const trimkit::Records<svgpath::Segment> records =
        trimkit::read_records<svgpath::Segment>(segments, segment_dtype, "SvgSegment", "segments");
    std::string path;
    const char* refusal = nullptr;
    std::size_t refused = 0;
    {
        const nb::gil_scoped_release release;
        refusal = svgpath::format(vertices.data(), vertices.shape(0), records.rows(), records.size(), &path, &refused);
    }
    if (refusal != nullptr) {
        trimkit::value_error("segment " + std::to_string(refused) + ": " + refusal);
    }
    return path;
}

// svgpath's segment kinds, named, in its order: each one's index here is its value
struct NamedSvgKind {
    const char* name;
    const char* doc;
};
static constexpr NamedSvgKind KINDS[] = {{"LINE", "A straight line: the L, H, V and Z commands."},
                                         {"QUADRATIC", "A quadratic Bezier curve: Q and T."},
                                         {"CUBIC", "A cubic Bezier curve: C and S."},
                                         {"ARC", "An elliptical arc: A."}};
static_assert(svgpath::LINE == 0 && svgpath::QUADRATIC == 1 && svgpath::CUBIC == 2 && svgpath::ARC == 3,
              "KINDS lists svgpath's kinds in its order");

void bind_svg(nb::module_& m) {
    trimkit::bind_enum<svgpath::Kind>(m, "SvgKind", "What drew an SvgSegment, its `kind` field.", KINDS);

    const nb::object segment_dtype = make_segment_dtype();
    m.attr("SvgSegment") = segment_dtype;

    m.def(
        "svg_parse", [segment_dtype](std::string_view path) { return svg_parse(path, segment_dtype); }, nb::arg("path"),
        nb::sig("def svg_parse(path: str) -> tuple[numpy.typing.NDArray[numpy.float64], "
                "numpy.typing.NDArray[numpy.void]]"),
        R"(Parse SVG path data into vertices and one segment per drawing command.

Parameters
----------
path : str
  The `d` attribute of an SVG `<path>`.

Returns
-------
vertices : (n, 2) float64
  Every point once in path order and absolute coordinates: a
  segment starts on the vertex the previous one ended on, and a closing
  `Z` ends on its subpath's first vertex. A segment that starts on any
  other vertex starts a new subpath.
segments : (m,) SvgSegment
  `kind` is an `SvgKind`, `points` are vertex indices
  (start, a curve's first control point or an arc's center, a cubic's
  second control point or the point halfway along an arc, end; -1 where
  unused), `closes` marks the line a `Z` drew, and `radius`, `rotation`
  and `angles` describe an arc's ellipse (NaN for other kinds). An arc is
  circular exactly when its two radii are equal, and then its start,
  halfway point and end are its three-point form. Malformed data returns everything before
  the first bad token, as SVG renders it: that includes a command whose
  coordinates overflow, from a relative step or a reflected control point.

Notes
-----
Accepts the SVG 2 path grammar and three things it doesn't: commas
anywhere whitespace may be, signed arc radii and rotation, and numbers
ending in a point ("1."). Arcs whose radii describe no finite ellipse
are lines, like arcs with a zero radius, and so are arcs whose radii are
still subnormal once scaled up to reach their endpoints: they lie within
1e-308 of that line, and couldn't be written back exactly. Every number
and coordinate is within +-1e30, like every trimkit input: a number past
it ends the path, and an arc whose ellipse reaches past it is a line.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> vertices, segments = trimkit.exchange.svg_parse("M 0 0 L 1 1")
>>> vertices.tolist(), segments["kind"].tolist()
([[0.0, 0.0], [1.0, 1.0]], [0]))");

    m.def(
        "svg_string",
        [segment_dtype](const trimkit::Array<double, -1, 2>& vertices, const nb::handle& segments) {
            return svg_string(vertices, segments, segment_dtype);
        },
        nb::arg("vertices"), nb::arg("segments"),
        nb::sig("def svg_string(vertices: numpy.typing.NDArray[numpy.float64], "
                "segments: numpy.typing.NDArray[numpy.void]) -> str"),
        R"(Write vertices and segments back as SVG path data, the inverse of `svg_parse`.

Parameters
----------
vertices : (n, 2) float64
  Points the segments index.
segments : (m,) SvgSegment
  Segments as `svg_parse` returns them, or built by hand.

Returns
-------
path : str
  Absolute path data. Parsing it reproduces `vertices` and `segments`
  exactly, for anything `svg_parse` returned. Numbers are written in the
  shortest form that reads back as the same double, independent of locale.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> trimkit.exchange.svg_string(*trimkit.exchange.svg_parse("M 0 0 L 1 1 Z"))
'M0 0 L1 1 Z')");
}
