// SVG path data to and from vertices and one record per segment, with our
// header-only `svgpath`.
//
// The parser fills a vertex array and a std::vector<svgpath::SvgSegment>,
// both handed to numpy without a copy; the segments are viewed as the
// `SvgSegment` structured dtype, built from the struct's own offsets so the
// two can never disagree. The writer copies both into vectors of its own
// before it lets go of the GIL: no other thread can change what it checks
// while it writes, on a free-threaded build either.

#include "common.hpp"

#include "svgpath.hpp"

// what svgpath parses is exactly what the door lets `svg_string` read back
static_assert(svgpath::COORDINATE_LIMIT == trimkit::COORDINATE_LIMIT, "svgpath and numpy.hpp share one bound");

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

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
        trimkit::value_error("digits must be from 1 to 17");
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
        trimkit::value_error(refused < path.segments.size() ? "segment " + std::to_string(refused) + ": " + refusal
                                                            : std::string(refusal));
    }
    return written;
}

// svgpath's segment kinds, named, in its order: each one's index here is its value
struct NamedSvgKind {
    const char* name;
    const char* doc;
};
static constexpr NamedSvgKind KINDS[] = {
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
              "KINDS lists svgpath's kinds in its order");

void bind_svg(nb::module_& m) {
    trimkit::bind_enum<svgpath::SvgKind>(m, "SvgKind", "What drew an SvgSegment, its `kind` field.", KINDS);

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
