import contextlib
import math
import struct
import sys
import threading

import numpy as np
import pytest
import svg_reference as ref
from helpers import check_svg_arc, check_svg_segments, svg_round_trip, svg_rows

import trimkit

X = trimkit.exchange
LINE, QUADRATIC, CUBIC, ARC, CONIC = (int(k) for k in X.SvgKind)


def rows(path: str) -> np.ndarray:
    # every segment with its points resolved, to check against references
    return svg_rows(*X.svg_parse(path))


def test_commands():
    vertices, segments = X.svg_parse(
        "M10 10 h5 v5 l-5-5 Z m1 1 C1 2 3 4 5 6 s1 1 2 2 q1 1 2 2 t3 3 a5 5 0 1 0 10 0"
    )
    assert segments.dtype == X.SvgSegment and segments.dtype.itemsize == 56
    assert vertices.dtype == np.float64 and vertices.shape[1] == 2
    # shared, not repeated: each segment starts where the last ended, and the
    # close ends on the subpath's first vertex
    points = segments["points"]
    assert (points[1:4, 0] == points[0:3, 3]).all() and points[3, 3] == points[0, 0]
    s = svg_rows(vertices, segments)
    # the half circle is two arcs of 90 degrees
    assert s["kind"].tolist() == [0, 0, 0, 0, 2, 2, 1, 1, 3, 3]
    # only the `m` after the close starts on a new vertex
    assert s["moves"].tolist() == [True] + [False] * 3 + [True] + [False] * 5
    assert s["closes"].tolist() == [False, False, False, True] + [False] * 6
    # relative moves after a close are from the closed subpath's start
    assert s["start"][4].tolist() == [11, 11]
    # smooth curves reflect the previous control point about the current point
    assert s["control"][5].tolist() == [[7, 8], [6, 7]]
    assert s["control"][7][0].tolist() == [10, 11]
    # a half circle of radius 5 the large way round (any way is half), sweep
    # flag 0: through (17, 18), each arc's middle a quarter of the way along,
    # and both arcs carry its center, the chord's midpoint, exactly
    arcs = s[-2:]
    assert arcs["end"][0].tolist() == [17, 18] and arcs["end"][1].tolist() == [22, 13]
    quarter = 5 / 2**0.5
    assert np.allclose(arcs["middle"], [[17 - quarter, 13 + quarter], [17 + quarter, 13 + quarter]])
    assert arcs["center"].tolist() == [[17, 13], [17, 13]]
    # slots a kind doesn't use are -1, and kind_data it doesn't use 0
    assert (points[:4, 1:3] == -1).all() and (points[6:8, 2] == -1).all() and (points[8:, 2] == -1).all()
    assert (segments["kind_data"][:8] == 0).all()


def test_layout_is_pinned():
    # one record per segment, meant to stay readable for decades: kind, seven
    # zero bytes, four int64 vertex indices, two float64 of kind_data
    assert X.SvgSegment.itemsize == 56
    assert [(name, X.SvgSegment.fields[name][1]) for name in X.SvgSegment.names] == [
        ("kind", 0),
        ("points", 8),
        ("kind_data", 40),
    ]
    vertices, segments = X.svg_parse("M0 0 L1 0 Q1 1 2 1 C3 1 3 2 4 2 Z")
    golden = [(0, (0, -1, -1, 1)), (1, (1, 2, -1, 3)), (2, (3, 4, 5, 6)), (0, (6, -1, -1, 0))]
    corners = [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [3, 2], [4, 2]]
    assert [
        (k, tuple(p)) for k, p in zip(segments["kind"].tolist(), segments["points"].tolist(), strict=True)
    ] == golden
    assert vertices.tolist() == corners
    # and the bytes themselves, which are this machine's order
    if sys.byteorder == "little":
        assert segments.tobytes() == b"".join(struct.pack("<B7x4q2d", k, *p, 0.0, 0.0) for k, p in golden)
        assert vertices.tobytes() == np.array(corners, "<f8").tobytes()
    # an arc's kind_data is its center, a conic's its weight then 0
    _, arcs = X.svg_parse("M0 0 A1 1 0 0 1 1 1 A2 1 0 0 1 3 2")
    assert arcs["kind"].tolist() == [ARC, CONIC]
    assert np.allclose(arcs["kind_data"][0], [0, 1]) and np.allclose(arcs["kind_data"][1], [0.5**0.5, 0])


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        ("M1.5.5-1e1+2", [[1.5, 0.5, -10, 2]]),
        ("M0,0L1,1,2,2", [[0, 0, 1, 1], [1, 1, 2, 2]]),
        ("m1 1 2 2", [[1, 1, 3, 3]]),
        ("M0 0a1 1 0 001 1", None),  # packed arc flags
        ("M0 0 A0 1 0 0 0 5 5", [[0, 0, 5, 5]]),  # zero radius is a line
        ("M0 0 A1 1 0 0 0 0 0", []),  # zero length draws nothing
    ],
)
def test_grammar(d, expected):
    s = rows(d)
    if expected is not None:
        assert np.column_stack([s["start"], s["end"]]).reshape((-1, 4)).tolist() == expected
    else:
        assert s["kind"].tolist() == [ARC]


@pytest.mark.parametrize("sweep", [0, 1])
def test_chained_and_arcs_consistent(sweep):
    # a long random path of every command: the end of each segment is the
    # start of the next, and every arc's pieces lie on its ellipse in order
    rng = np.random.default_rng(sweep)
    parts = ["M 0 0"]
    for _ in range(2000):
        c = rng.choice(list("LHVCSQTAlhvcsqta"))
        if c in "Aa":
            rx, ry, rot = [*rng.uniform(0.1, 5, 2).tolist(), float(rng.uniform(0, 360))]
            if rng.integers(0, 3) == 0:
                ry = rx
            large = int(rng.integers(0, 2))
            x, y = rng.uniform(-5, 5, 2).tolist()
            parts.append(f"{c}{rx!r} {ry!r} {rot!r} {large} {sweep} {x!r} {y!r}")
        else:
            n = {"L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2}[c.upper()]
            parts.append(c + " ".join(f"{v:.6g}" for v in rng.uniform(-5, 5, n)))
    d = " ".join(parts)
    s = rows(d)
    assert np.array_equal(s["end"][:-1], s["start"][1:])
    check_svg_segments(s, ref.evaluate(ref.parse(ref.LENIENT_PARSER, d)), ref)
    conics = s["weight"][s["kind"] == CONIC]
    assert len(conics) and (conics >= 0.5).all() and (conics < 1).all()


@pytest.mark.parametrize(
    ("degrees", "pieces"),
    [(1, 1), (60, 1), (119, 1), (121, 2), (179, 2), (181, 2), (239, 2), (241, 4), (300, 4), (359, 4)],
)
@pytest.mark.parametrize("radii", ["1 1", "2 1"])
def test_arcs_split_at_120_degrees(degrees, pieces, radii):
    # every piece is at most 120 degrees: two past it, four past twice it
    angle = math.radians(degrees)
    # on the unit circle, or that circle stretched to an ellipse twice as wide
    width = 2.0 if radii == "2 1" else 1.0
    end = (width * math.cos(angle), math.sin(angle))
    s = rows(f"M{width!r} 0 A{radii} 0 {int(degrees > 180)} 1 {end[0]!r} {end[1]!r}")
    assert len(s) == pieces
    assert (s["kind"] == (ARC if width == 1 else CONIC)).all()
    check_svg_arc(s, ((0.0, 0.0), width, 1.0, (1.0, 0.0)), True, degrees > 180)


@pytest.mark.parametrize(
    ("d", "count"),
    [
        ("", 0),
        ("L 1 2", 0),  # must start with a moveto
        ("M 0 0 L 1 1 L 2 x L 3 3", 1),
        ("M 0 0 L 1 1 Z 3", 2),
        ("M 0 0 L 1 inf", 0),
        ("M 0 0 A 1 1 0 2 0 3 3", 0),
    ],
)
def test_malformed_keeps_prefix(d, count):
    # like an SVG renderer, everything before the first error is kept
    assert len(X.svg_parse(d)[1]) == count


@pytest.mark.parametrize("rx", ["1e150", "2e30"])
def test_radii_past_the_bound(rx):
    # a number past 1e30 ends the path, like one that overflows a double
    assert len(rows(f"M1 0 L2 0 A{rx} 12 0 0 1 1 14")) == 1


@pytest.mark.parametrize(
    "path",
    [
        # radii scaled up (F.6.6) past 1e30 to reach their endpoints make an
        # ellipse reaching past it: like subnormal ones, that arc is a line
        "M0 0 A1 1e-291 0 0 0 0 1",
        # found by fuzzing: a chord of 1e-300 against radii of 1e30 underflows
        # (the endpoints coincide as far as the ellipse can tell)
        "M0 0 A1e30 1e30 0 0 1 1e-300 0",
        # an arc whose middle lies closer to its chord than the coordinates'
        # precision: three points on one line, so its chord
        "M1e6 1e6 A1e30 1e30 0 0 1 1000001 1000001",
        "M1 0 A3.2e-168 12 0 0 1 1 14",
        # all the way round a circle of 1e30 reaches past the bound
        "M0 0 A1e30 1e30 0 1 1 1 0",
        # a circle bulging toward the origin, its center past the bound
        "M1e30 0 A1e30 1e30 0 0 0 1e30 1e29",
    ],
)
def test_arcs_no_double_describes_are_lines(path):
    assert rows(path)["kind"].tolist() == [LINE]


@pytest.mark.parametrize(
    ("rx", "ry"), [("1e30", "12"), ("1e-300", "1e-300"), ("3.2e-168", "3.2e-168"), ("1e7", "3e7")]
)
def test_extreme_radii(rx, ry):
    # found by fuzzing: radii like these underflowed or overflowed the center
    # conversion, so the arc no longer passed through its endpoints. Tiny
    # radii are scaled up to reach, a half circle
    s = rows(f"M1 0 A{rx} {ry} 0 0 1 1 14")
    assert set(s["kind"].tolist()) == {ARC if rx == ry else CONIC}
    ellipse = ref.ellipse((1.0, 0.0), (1.0, 14.0), float(rx), float(ry), 0.0, large=False, sweep=True)
    check_svg_arc(s, ellipse, True, False)


@pytest.mark.parametrize(
    ("path", "circular"),
    [
        ("M0 0 A5 5 0 0 1 8 4", True),
        ("M0 0 A5 5 0 1 1 6 0", True),  # the large way round
        ("M0 0 A1 1 0 0 0 10 0", True),  # radii scaled up together stay equal
        ("M0 0 A5 5 45 0 1 3 4", True),  # rotation doesn't make a circle an ellipse
        ("M0 0 A5 3 0 0 1 4 0", False),
        ("M0 0 A5 3 30 1 0 4 1", False),
    ],
)
def test_arc_three_point_form(path, circular):
    # equal radii are arcs: start, halfway point and end, the three-point arc
    # trimesh's `Arc` takes, on the circle around the center they carry. Any
    # other is a conic
    s = rows(path)
    assert (s["kind"] == (ARC if circular else CONIC)).all()
    if circular:
        radius = np.linalg.norm(s["start"] - s["center"], axis=1)
        for point in ("middle", "end"):
            assert np.allclose(np.linalg.norm(s[point] - s["center"], axis=1), radius)


def test_centers_merge_circles():
    # one circle drawn as two half arcs: four pieces, one center, exactly, so
    # an importer can tell it is one closed circle by comparing them
    s = rows("M0 0 A5 5 0 0 1 10 0 A5 5 0 0 1 0 0")
    assert len(s) == 4 and (s["kind"] == ARC).all()
    assert len({tuple(c) for c in s["center"].tolist()}) == 1 and s["center"][0].tolist() == [5, 0]


def test_string_ignores_arc_centers():
    # an arc's circle is its three points: its center isn't read back, and a
    # scribbled one changes nothing
    vertices, segments = X.svg_parse("M0 0 A5 5 0 0 1 8 4 A5 3 30 1 0 4 1")
    scribbled = segments.copy()
    scribbled["kind_data"][scribbled["kind"] == ARC] = np.nan
    assert X.svg_string(vertices, scribbled) == X.svg_string(vertices, segments)


def test_conics_are_affine():
    # a conic's vertices and weight are geometry, not a frame: the same arc
    # turned, scaled and moved is the same pieces, turned, scaled and moved
    a_vertices, a = X.svg_parse("M1 2 A5 3 20 1 0 4 1")
    turn = math.radians(35)
    matrix = 2.5 * np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
    start, end = (matrix @ [1, 2] + [7, -3]).tolist(), (matrix @ [4, 1] + [7, -3]).tolist()
    b_vertices, b = X.svg_parse(f"M{start[0]!r} {start[1]!r} A12.5 7.5 55 1 0 {end[0]!r} {end[1]!r}")
    assert np.array_equal(a["kind"], b["kind"]) and np.array_equal(a["points"], b["points"])
    assert np.allclose(a_vertices @ matrix.T + [7, -3], b_vertices, atol=1e-12)
    assert np.allclose(a["kind_data"], b["kind_data"], atol=1e-13)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "M0 0",
        "M10 10 h5 v5 l-5-5 Z m1 1 C1 2 3 4 5 6 s1 1 2 2 q1 1 2 2 t3 3 a5 5 0 1 0 10 0",
        "M0 0 L1 1 Z L2 2",  # drawing on after a close starts from its first vertex
        "M0 0 L1 1 Z M0 0 L2 2",  # a moveto there instead starts a new vertex
        "M0 0 L1 1 Z Z",  # a second close draws nothing new
        "M0 0 M1 1 M2 2 L3 3",  # moves that draw nothing add nothing
        "M0 0 L1 1 M1 1 L2 2",  # a moveto to the pen still starts a new vertex
        "M1 1 Z",
        "M0 0 A0.1 0.1 0 0 1 10 10",  # radii scaled up to reach, F.6.6
        "M0 0 A5 5 0 1 0 10 0 A5 5 0 0 1 20 0",  # semicircles
        "M0 0 A5 3 30 1 0 4 1 A7 2 -10 0 1 -3 2",  # ellipses
        "M1 0 A1e30 12 0 0 1 1 14",  # a radius at the bound stays there
        "M0 0 A1e30 1e30 0 0 1 1 0",  # its middle 1.25e-31 off the chord, near 0 still a double
        "M1 0 A1e-300 1e-300 0 0 1 1 14",
        # found by hypothesis: written back turned 90 degrees, whose cosine in
        # radians (6e-17) is the short radius' size, tilting it into 4 pieces
        "M0 0 A2.220446049250313e-16 2 0 0 0 0 4",
        "M0 0 L1e-300 -0.0 L1e300 5e-324",
        "M 0 0 L 1 1 L 2 x L 3 3",  # malformed: the parsed prefix
    ],
)
def test_round_trip(path):
    svg_round_trip(path)


def test_string_is_compact_absolute_and_locale_free():
    assert X.svg_string(*X.svg_parse("m1 1 l2 0.5 z")) == "M1 1 L3 1.5 Z"
    assert X.svg_string(*X.svg_parse("")) == ""
    # arcs are written at most 120 degrees each, never the large arc
    assert X.svg_string(*X.svg_parse("M0 0 A5 5 0 0 1 10 0")) == "M0 0 A5 5 0 0 1 5 -5 A5 5 0 0 1 10 0"


@pytest.mark.parametrize(
    ("number", "digits", "written"),
    [
        ("3.14159265358979", 8, "3.1415927"),
        ("123456789.123", 8, "123456790"),  # an integer padded with zeros
        ("1e-7", 8, "1e-07"),  # scientific where it is shorter
        ("0.000123456789", 4, "0.0001235"),
        ("2.675", 3, "2.68"),  # the decimal shown, rounded half away from zero
        ("9.995", 3, "10"),  # a carry into a new digit
        ("-0.125", 2, "-0.13"),
        ("0.1", 17, "0.1"),  # never more digits than the shortest
        ("1152921504606846976", 17, "1152921504606847000"),  # 2^60, its shortest digits
    ],
)
def test_digits(number, digits, written):
    vertices, segments = X.svg_parse(f"M0 0 L{number} 0")
    assert X.svg_string(vertices, segments, digits=digits) == f"M0 0 L{written} 0"


def test_digits_must_be_1_to_17():
    arrays = X.svg_parse("M0 0 L1 1")
    for digits in (0, 18, -5, 2**40):
        with pytest.raises(ValueError, match="digits must be from 1 to 17"):
            X.svg_string(*arrays, digits=digits)


def significant(text: str) -> str:
    # a written number's significant digits, without sign, point or exponent
    return text.lower().split("e")[0].lstrip("-").replace(".", "").strip("0")


def test_numbers_read_back():
    # Random bit patterns within the bound, decimals, integers past 2^53, and
    # the powers. At 17 digits every number is its shortest digits (Python's
    # repr has the same), and reads back as exactly itself; at the default
    # 8, at most 8 of them, within half a unit of the eighth
    rng = np.random.default_rng(0)
    patterns = rng.integers(0, 2**64, 40_000, dtype=np.uint64).view(np.float64)
    integers = 2.0**53 + rng.integers(-(2**40), 2**60, 4_000).astype(np.float64) * 16
    values = np.concatenate(
        [
            patterns[np.abs(patterns) <= 1e30],
            np.round(rng.uniform(-1e4, 1e4, 4_000), 3),
            integers,
            np.ldexp(1.0, np.arange(-1074, 100, dtype=np.int32)),
            10.0 ** np.arange(-300, 31, dtype=np.int64),
            [0.0, -0.0, 5e-324, 1e30, -1e30, 2.0**60, 1e22, 1e23],
        ]
    )
    values = values[: len(values) // 4 * 4]
    vertices = np.ascontiguousarray(values.reshape(-1, 2))
    # one line per pair of vertices: `M x y L x y`, so every number is written
    template_vertices, template = X.svg_parse("M0 0 L0 0 " * (len(vertices) // 2))
    assert len(template_vertices) == len(vertices)
    for digits in (17, 8):
        written = X.svg_string(vertices, template, digits=digits).replace("M", "").replace("L", "").split()
        read = np.array([float(w) for w in written])
        if digits == 17:
            assert read.tobytes() == values.tobytes()
            assert [significant(w) for w in written] == [significant(repr(float(v))) for v in values]
        else:
            assert max(len(significant(w)) for w in written) <= 8
            assert (np.abs(read - values) <= 5e-8 * np.abs(values) + 5e-324).all()


def test_string_hand_built():
    # a hand-built array: a segment that doesn't start on the vertex the last
    # one ended on gets a moveto, and what a kind doesn't use is ignored
    vertices, segments = X.svg_parse("M0 0 L1 0 M5 5 L6 5")
    assert X.svg_string(vertices, segments) == "M0 0 L1 0 M5 5 L6 5"
    unused = segments.copy()
    unused["points"][:, 1:3] = 12345
    unused["kind_data"] = np.nan
    assert X.svg_string(vertices, unused) == "M0 0 L1 0 M5 5 L6 5"
    joined = segments.copy()
    joined["points"][1, 0] = joined["points"][0, 3]
    assert X.svg_string(vertices, joined) == "M0 0 L1 0 L6 5"

    bad = segments.copy()
    bad["kind"][1] = 9
    with pytest.raises(ValueError, match="segment 1: kind is not an SvgKind"):
        X.svg_string(vertices, bad)
    bad = segments.copy()
    bad["points"][1, 3] = len(vertices)
    with pytest.raises(ValueError, match="segment 1: points names a vertex out of range"):
        X.svg_string(vertices, bad)


def segment(kind: int, points, kind_data=(0.0, 0.0)) -> np.ndarray:
    row = np.zeros(1, X.SvgSegment)
    row["kind"], row["points"], row["kind_data"] = kind, points, kind_data
    return row


def test_string_hand_built_arcs():
    vertices = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [2, 0], [0.5**0.5, 0.5**0.5]], dtype=np.float64)
    # three points on the unit circle, 270 degrees from start to end: four `A`s
    written = X.svg_string(vertices, segment(ARC, [0, 1, -1, 2]), digits=17)
    assert written.count("A") == 4
    s = rows(written)
    assert len(s) == 4 and s["end"][-1].tolist() == [0, -1]
    check_svg_arc(s, ((0.0, 0.0), 1.0, 1.0, (1.0, 0.0)), True, True)
    # and 90 degrees is one
    assert X.svg_string(vertices, segment(ARC, [0, 5, -1, 1])).count("A") == 1
    with pytest.raises(
        ValueError, match="segment 0: an arc's or conic's three points must not lie on one line"
    ):
        X.svg_string(vertices, segment(ARC, [3, 0, -1, 4]))
    # a conic of weight 0.2 sweeps 2 acos(0.2), 157 degrees: two `A`s
    s = rows(X.svg_string(vertices, segment(CONIC, [0, 4, -1, 1], (0.2, 0.0))))
    assert len(s) == 2 and s["start"][0].tolist() == [1, 0] and s["end"][1].tolist() == [0, 1]
    for weight in (0.0, 1.0, -0.5, np.nan):
        with pytest.raises(ValueError, match="weight must be between 0 and 1"):
            X.svg_string(vertices, segment(CONIC, [0, 4, -1, 1], (weight, 0.0)))


def test_string_close_is_by_vertex():
    # a close returns to its subpath's first vertex, by index: the same
    # coordinates in another vertex are not a close
    vertices, segments = X.svg_parse("M0 0 L1 0 L1 1 Z")
    assert X.svg_string(vertices, segments) == "M0 0 L1 0 L1 1 Z"
    copy = np.vstack([vertices, vertices[:1]])
    other = segments.copy()
    other["points"][2, 3] = len(copy) - 1
    assert X.svg_string(copy, other) == "M0 0 L1 0 L1 1 L0 0"


@pytest.mark.parametrize(
    ("path", "segment_count", "vertex_count"),
    [
        ("M0 0" + "h1v1h-1z" * 1000, 4000, 3001),  # every square shares its corner
        ("M0 0" + "Z" * 10000, 0, 0),  # closes of nothing draw nothing
        ("M0 0 h1" + "Z" * 10000, 2, 2),  # only the first close has something to close
        ("M1 1 " * 10000, 0, 0),  # moves that draw nothing
        ("Z" * 10000, 0, 0),  # no moveto
        pytest.param("M0 0" + " l1 2" * 2_000_000, 2_000_000, 2_000_001, id="10MB"),
        ("M0 0" + "l1 2" * 1000, 1000, 1001),  # a letter ends a number: no separators
        ("M0 0l1-2-3.5.5" * 1000, 2000, 3000),  # minified: signs and points separate
    ],
)
def test_shapes(path, segment_count, vertex_count):
    vertices, segments = X.svg_parse(path)
    assert (len(segments), len(vertices)) == (segment_count, vertex_count)
    svg_round_trip(path)


@pytest.mark.parametrize(
    "path",
    [
        "M0 0 h" + ".1" * 50_000,  # two bytes a line
        "M0 0" + "h1z" * 20_000,  # three bytes for a line and a close: the most
        "M0 0 a1 1 0 11 1 1" + " 1 1 0 11 1 1" * 5_000,  # twelve for four arcs
        "M0 0 a1 2 0 11 1 1" + " 1 2 0 11 1 1" * 5_000,  # or four conics
    ],
)
def test_output_is_linear_in_the_input(path):
    # the bound `svg_parse` documents, at the inputs that come closest to it
    assert sum(a.nbytes for a in X.svg_parse(path)) <= 43 * len(path)


def test_string_rejects_other_arrays():
    vertices, segments = X.svg_parse("M0 0 L1 1 L2 2 L3 3 L4 4")
    for wrong in (np.zeros(3), segments.reshape((1, -1)), segments.tolist()):
        with pytest.raises(TypeError, match="SvgSegment"):
            X.svg_string(vertices, wrong)


# found by check/fuzz/svgpath.cpp: `svg_string` refused what `svg_parse` produced,
# a vertex a relative step had pushed to infinity
FUZZED_OVERFLOW = (
    "M01 1 M01 1 2 2 t3 3a5 30000 5 0 1 0 1\nM0-5 000000000000000.0000000000"
    + "0800000000002000 0 1 0 1\n 055 10 1 0 10  1"
    + "0" * 302
    + "127660\n 0550 a5 5 0 1 "
    + "0" * 91
    + "10  1\n  05550\n 32164482 10 1 0 102 2 t3 3a5 5 0 1 0 1\n5 10 1 0 102 2 t"
    + "3 3 1 0 1\n 055  11 0 0 10  1\n 0 1\n 055 10 1 0 102055 10 1 0 10  1\n 5  "
    + "10 1 0 10  1\n 0550 a50800000000002000000000000800000000080000000000200"
    + "00000000008000000000800000061"
    + "0" * 140
    + "8000000000020000000000008000000000800000000002000000000000800000000080"
    + "0000061000000000000 5 0 1 0 1\n 055 10 00000000008000000000800000000002"
    + "0000000000008000000000800000061"
    + "0" * 60
    + " 5 0 1 0 1\n 055 10 1 0 102 2 t3 3a5 5 0 1 0 1\n 055  10 1 0 10  1\n 0550"
    + " a5 5 0 1 0 1\n 055 10 1 0 102 2 t3 3a5 5 0 1 0 1\n 055  11 0 0 10  1\n 0"
    + "550 a5 5 0 1 0 1\n 055 10 1 0 10  1\n 5  10 1 0 10  1\n 0550 a50800000000"
    + "00200000000000080000000008000000000020000000000008000000000800000061"
    + "0" * 140
    + "8000000000020000000000008000000000800000000002000000000000800000000080"
    + "0000061000000000000 5 0 1 0 1\n 055 10 1 0 102 2 t3 3a5 5 0 1 0 1\n5 10 "
    + "1 0 102 7 t3 3a5 5 0 1 0 1\n 055  11 0 0 10  1\n 0550 a5 5 0 1 0 1\n 055 "
    + "10 1 0 10  1\n 5  10 1 0 10  1\n 0550 a5080000000M0-5 000000000000000.00"
    + "000000000800000000002000 0 1 0 1\n 055 10 1 0 10  1"
    + "0" * 100
    + "200000000000080000000008000000"
)


@pytest.mark.parametrize(
    "path",
    [
        "M1 1 L2 2 m1e308 0 l1e308 0",  # a relative step past the largest double
        "M1 1 L2 2 C1 1 1e308 1 1e308 0 S1 1 1 1",  # a reflected control point past it
        FUZZED_OVERFLOW,
    ],
)
def test_overflowing_command_ends_the_path(path):
    # finite numbers, but a coordinate that overflows: the path ends before
    # that command, as for a number that overflows, and what remains is
    # finite and writes back exactly
    vertices, segments = X.svg_parse(path)
    assert np.isfinite(vertices).all() and len(segments) > 0
    svg_round_trip(path)


def test_copied_layout_outlives_the_call():
    # any other layout is copied once, and that copy must stay alive while it is
    # read: big enough for numpy to really free it, so ASan sees any later read
    vertices, segments = X.svg_parse("M0 0 " + "L1 1 " * 5000)
    reversed_rows = segments[::-1]
    assert X.svg_string(vertices, reversed_rows) == X.svg_string(vertices, reversed_rows.copy())


def test_string_reads_a_copy():
    # what svg_string writes is what the arrays held when it was called:
    # it copies them before letting go of the GIL, so another thread
    # changing them meanwhile can't put an unchecked value in front of it
    vertices, segments = X.svg_parse("M0 0 " + "A1 2 0 0 1 1 1 a1 2 0 0 0 -1 -1 " * 20_000)
    expected = X.svg_string(vertices, segments)
    stop = threading.Event()

    def scribble():
        rng = np.random.default_rng(0)
        while not stop.is_set():
            segments["kind_data"][rng.integers(0, len(segments)), 0] = np.nan
            vertices[rng.integers(0, len(vertices))] = np.inf

    kept = segments.copy(), vertices.copy()
    thread = threading.Thread(target=scribble)
    thread.start()
    try:
        for _ in range(5):
            # what it copied may already be scribbled on: refused, never
            # misread. Values only ever move to ones refused (a vertex to inf)
            # or never read (an arc's center), so an answer is the first one
            with contextlib.suppress(ValueError):
                assert X.svg_string(vertices, segments) == expected
    finally:
        stop.set()
        thread.join()
    segments[:], vertices[:] = kept
    assert X.svg_string(vertices, segments) == expected
