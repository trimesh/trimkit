import decimal
import math

import numpy as np
import pytest
from helpers import evaluate_arc, svg_rows

import trimkit


def rows(path: str) -> np.ndarray:
    # every segment with its points resolved, to check against references
    return svg_rows(*trimkit.exchange.svg_parse(path))


def test_commands():
    vertices, segments = trimkit.exchange.svg_parse(
        "M10 10 h5 v5 l-5-5 Z m1 1 C1 2 3 4 5 6 s1 1 2 2 q1 1 2 2 t3 3 a5 5 0 1 0 10 0"
    )
    assert segments.dtype == trimkit.exchange.SvgSegment and segments.dtype.itemsize == 80
    assert vertices.dtype == np.float64 and vertices.shape[1] == 2
    # shared, not repeated: each segment starts where the last ended, and the
    # close ends on the subpath's first vertex
    points = segments["points"]
    assert (points[1:4, 0] == points[0:3, 3]).all() and points[3, 3] == points[0, 0]
    s = svg_rows(vertices, segments)
    assert s["kind"].tolist() == [0, 0, 0, 0, 2, 2, 1, 1, 3]
    # only the `m` after the close starts on a new vertex
    assert s["moves"].tolist() == [True, False, False, False, True, False, False, False, False]
    assert s["closes"].tolist() == [False, False, False, True, False, False, False, False, False]
    # relative moves after a close are from the closed subpath's start
    assert s["start"][4].tolist() == [11, 11]
    # smooth curves reflect the previous control point about the current point
    assert s["control"][5].tolist() == [[7, 8], [6, 7]]
    assert s["control"][7][0].tolist() == [10, 11]
    # a half circle of radius 5 sweeping counter-clockwise in y-down space
    arc = s[-1]
    assert np.allclose(arc["center"], [17, 13])
    assert np.allclose(arc["angles"], [np.pi, -np.pi])
    assert np.allclose(arc["middle"], [17, 18])
    # slots a kind doesn't use are NaN
    assert np.isnan(s["control"][0]).all()
    assert np.isnan(s["control"][6][1]).all()


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
        assert s["kind"].tolist() == [trimkit.exchange.SvgKind.ARC]


def test_chained_and_arcs_consistent():
    # a long random path of every command: the end of each segment is the start
    # of the next within a subpath, and every arc's center parameterization
    # reproduces both its endpoints, which checks the whole F.6.5 conversion
    rng = np.random.default_rng(0)
    parts = ["M 0 0"]
    for _ in range(2000):
        c = rng.choice(list("LHVCSQTAlhvcsqta"))
        if c in "Aa":
            rx, ry, rot = [*rng.uniform(0.1, 5, 2).tolist(), rng.uniform(0, 360)]
            flags = rng.integers(0, 2, 2)
            x, y = rng.uniform(-5, 5, 2)
            parts.append(f"{c}{rx} {ry} {rot} {flags[0]} {flags[1]} {x} {y}")
        else:
            n = {"L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2}[c.upper()]
            parts.append(c + " ".join(f"{v:.6g}" for v in rng.uniform(-5, 5, n)))
    s = rows(" ".join(parts))
    assert len(s) == 2000  # no command was dropped
    assert np.allclose(s["end"][:-1], s["start"][1:])

    arcs = s[s["kind"] == trimkit.exchange.SvgKind.ARC]
    assert len(arcs) > 50
    start, sweep = arcs["angles"].T
    assert np.allclose(evaluate_arc(arcs, start), arcs["start"], atol=1e-9)
    assert np.allclose(evaluate_arc(arcs, start + sweep), arcs["end"], atol=1e-9)
    assert np.allclose(evaluate_arc(arcs, start + sweep / 2), arcs["middle"], atol=1e-9)
    # the sweep flag decides the direction of travel
    flags = [p for p in parts if p[0] in "Aa"]
    written = np.array([int(p.split()[4]) for p in flags])
    assert np.array_equal(sweep > 0, written[: len(sweep)] == 1)


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
    assert len(trimkit.exchange.svg_parse(d)[1]) == count


@pytest.mark.parametrize("rx", ["1e150", "2e30"])
def test_radii_past_the_bound(rx):
    # a number past 1e30 ends the path, like one that overflows a double
    assert len(rows(f"M1 0 L2 0 A{rx} 12 0 0 1 1 14")) == 1


def test_arc_reaching_past_the_bound_is_a_line():
    # radii scaled up (F.6.6) past 1e30 to reach their endpoints make an
    # ellipse reaching past it: like subnormal ones, that arc is a line
    assert rows("M0 0 A1 1e-291 0 0 0 0 1")["kind"].tolist() == [trimkit.exchange.SvgKind.LINE]


def test_arc_below_its_radii_precision_is_a_line():
    # found by fuzzing: a chord of 1e-300 against radii of 1e30 underflows the
    # center conversion to nothing (the endpoints coincide as far as the
    # ellipse can tell), so that arc is its chord
    assert rows("M0 0 A1e30 1e30 0 0 1 1e-300 0")["kind"].tolist() == [trimkit.exchange.SvgKind.LINE]


@pytest.mark.parametrize("rx", ["3.2e-168", "1e-300", "1e30"])
def test_extreme_radii(rx):
    # found by fuzzing: squaring radii like these underflowed or overflowed the
    # center conversion, so the arc no longer passed through its endpoints
    s = rows(f"M1 0 A{rx} 12 0 0 1 1 14")
    assert s["kind"].tolist() == [trimkit.exchange.SvgKind.ARC]
    assert all(np.isfinite(s[k]).all() for k in ("center", "radius", "angles"))
    # a huge radius puts the center equally far away, so error is relative to it
    scale = 1 + np.abs(s["center"]).sum() + s["radius"].sum()
    start, sweep = s["angles"].T
    assert np.abs(evaluate_arc(s, start) - s["start"]).max() < 1e-12 * scale
    assert np.abs(evaluate_arc(s, start + sweep) - s["end"]).max() < 1e-12 * scale


@pytest.mark.parametrize(
    ("path", "circular"),
    [
        ("M0 0 A5 5 0 0 1 10 0", True),  # a half circle
        ("M0 0 A5 5 0 1 1 6 0", True),  # the large way round
        ("M0 0 A1 1 0 0 0 10 0", True),  # radii scaled up together stay equal
        ("M0 0 A5 5 45 0 1 3 4", True),  # rotation doesn't make a circle an ellipse
        ("M0 0 A5 3 0 0 1 4 0", False),
        ("M0 0 A5 3 30 1 0 4 1", False),
    ],
)
def test_arc_three_point_form(path, circular):
    # an arc is circular exactly when its radii are equal, and then start,
    # halfway point and end are the three-point arc trimesh's `Arc` takes
    vertices, segments = trimkit.exchange.svg_parse(path)
    (arc,) = svg_rows(vertices, segments)
    assert (arc["radius"][0] == arc["radius"][1]) == circular
    # the halfway point is a vertex of its own, read in place
    three = vertices[segments["points"][0, [0, 2, 3]]]
    assert np.array_equal(three, [arc["start"], arc["middle"], arc["end"]])
    if circular:
        # on the circle, and as far from either end along it
        radius = arc["radius"][0]
        assert np.allclose(np.linalg.norm(three - arc["center"], axis=1), radius)
        assert np.isclose(np.linalg.norm(three[1] - three[0]), np.linalg.norm(three[1] - three[2]))


def round_trip(path: str) -> None:
    # parsing what svg_string writes reproduces the segments bit for bit
    vertices, segments = trimkit.exchange.svg_parse(path)
    again = trimkit.exchange.svg_parse(trimkit.exchange.svg_string(vertices, segments))
    assert again[0].tobytes() == vertices.tobytes() and again[1].tobytes() == segments.tobytes()


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
        "M1 0 A3.2e-168 12 0 0 1 1 14",
        "M0 0 L1e-300 -0.0 L1e300 5e-324",
        # found by fuzzing: radii scaled up to reach both endpoints but still
        # subnormal read back as not quite reaching them, and scaled again
        "M3.93406263e-316 5.432309224886e-312 A2.9460499388320934e-307 2.583497e-317 "
        "2.2444127800473476e-189 0 0 7.290231990012995e-304 4e-323",
        "M 0 0 L 1 1 L 2 x L 3 3",  # malformed: the parsed prefix
    ],
)
def test_round_trip(path):
    round_trip(path)


def test_string_is_compact_absolute_and_locale_free():
    path = trimkit.exchange.svg_string(*trimkit.exchange.svg_parse("m1 1 l2 0.5 z"))
    assert path == "M1 1 L3 1.5 Z"
    assert trimkit.exchange.svg_string(*trimkit.exchange.svg_parse("")) == ""


def to_chars(value: float) -> str:
    """
    What C++'s std::to_chars(first, last, value) writes, from Python's own
    shortest round-trip digits: fixed or scientific notation, whichever is
    shorter, fixed on a tie, and a fixed integer past its shortest digits
    written exactly.
    """
    if value == 0:
        return "-0" if math.copysign(1.0, value) < 0 else "0"
    shortest = decimal.Decimal(repr(abs(value))).normalize().as_tuple()
    digits = "".join(map(str, shortest.digits))
    point = len(digits) + shortest.exponent
    exponent = point - 1
    scientific = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    scientific += f"e{'-' if exponent < 0 else '+'}{abs(exponent):02d}"
    if point > len(digits):
        fixed = str(int(abs(value)))
    elif point > 0:
        fixed = digits[:point] + ("." + digits[point:] if point < len(digits) else "")
    else:
        fixed = "0." + "0" * -point + digits
    return ("-" if value < 0 else "") + (fixed if len(fixed) <= len(scientific) else scientific)


def test_numbers_are_written_as_to_chars():
    # svg_string writes every number as std::to_chars would (svgpath formats
    # with Dragonbox rather than link libstdc++'s formatting): random bit
    # patterns within the bound, decimals, integers past 2^53, and the powers
    rng = np.random.default_rng(0)
    patterns = rng.integers(0, 2**64, 40_000, dtype=np.uint64).view(np.float64)
    integers = 2.0**53 + rng.integers(-(2**40), 2**60, 4_000).astype(np.float64) * 16
    values = np.concatenate(
        [
            patterns[np.abs(patterns) <= 1e30],
            np.round(rng.uniform(-1e4, 1e4, 4_000), 3),
            integers,
            np.ldexp(1.0, np.arange(-1074, 100)),
            10.0 ** np.arange(-300, 31),
            [0.0, -0.0, 5e-324, 1e30, -1e30, 2.0**60, 1e22, 1e23],
        ]
    )
    values = values[: len(values) // 4 * 4]
    vertices = np.ascontiguousarray(values.reshape(-1, 2))
    # one line per pair of vertices: `M x y L x y`, so every number is written
    template_vertices, template = trimkit.exchange.svg_parse("M0 0 L0 0 " * (len(vertices) // 2))
    assert len(template_vertices) == len(vertices)
    written = trimkit.exchange.svg_string(vertices, template).replace("M", "").replace("L", "").split()
    assert written == [to_chars(float(v)) for v in values]


def test_string_hand_built():
    # a hand-built array: a segment that doesn't start on the vertex the last
    # one ended on gets a moveto, and unused slots are ignored
    vertices, segments = trimkit.exchange.svg_parse("M0 0 L1 0 M5 5 L6 5")
    assert trimkit.exchange.svg_string(vertices, segments) == "M0 0 L1 0 M5 5 L6 5"
    unused = segments.copy()
    unused["points"][:, 1:3] = 12345
    unused["radius"] = np.nan
    assert trimkit.exchange.svg_string(vertices, unused) == "M0 0 L1 0 M5 5 L6 5"
    joined = segments.copy()
    joined["points"][1, 0] = joined["points"][0, 3]
    assert trimkit.exchange.svg_string(vertices, joined) == "M0 0 L1 0 L6 5"

    bad = segments.copy()
    bad["kind"][1] = 9
    with pytest.raises(ValueError, match="SvgKind"):
        trimkit.exchange.svg_string(vertices, bad)
    with pytest.raises(ValueError, match="finite"):
        trimkit.exchange.svg_string(
            np.where(np.arange(len(vertices))[:, None] == 3, np.nan, vertices), segments
        )
    bad = segments.copy()
    bad["points"][1, 3] = len(vertices)
    with pytest.raises(ValueError, match="out of range"):
        trimkit.exchange.svg_string(vertices, bad)
    # float32 vertices cast safely: converted with a warning, the same path
    with pytest.warns(UserWarning, match="vertices: float32 converted to float64"):
        converted = trimkit.exchange.svg_string(vertices.astype(np.float32), segments)
    assert converted == trimkit.exchange.svg_string(vertices, segments)


def test_string_close_is_by_vertex():
    # a close returns to its subpath's first vertex, by index: the same
    # coordinates in another vertex are not a close
    vertices, segments = trimkit.exchange.svg_parse("M0 0 L1 0 L1 1 Z")
    assert trimkit.exchange.svg_string(vertices, segments) == "M0 0 L1 0 L1 1 Z"
    copy = np.vstack([vertices, vertices[:1]])
    bad = segments.copy()
    bad["points"][2, 3] = len(copy) - 1
    with pytest.raises(ValueError, match="first vertex"):
        trimkit.exchange.svg_string(copy, bad)
    bad = segments.copy()
    bad["closes"][1] = True
    with pytest.raises(ValueError, match="first vertex"):
        trimkit.exchange.svg_string(vertices, bad)
    vertices, segments = trimkit.exchange.svg_parse("M0 0 Q1 1 2 0 Z")
    bad = segments.copy()
    bad["closes"][0] = True
    with pytest.raises(ValueError, match="only a line"):
        trimkit.exchange.svg_string(vertices, bad)


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
    vertices, segments = trimkit.exchange.svg_parse(path)
    assert (len(segments), len(vertices)) == (segment_count, vertex_count)
    round_trip(path)


def test_string_rejects_other_arrays():
    vertices, segments = trimkit.exchange.svg_parse("M0 0 L1 1 L2 2 L3 3 L4 4")
    for wrong in (np.zeros(3), segments.reshape((1, -1)), segments.tolist()):
        with pytest.raises(TypeError, match="SvgSegment"):
            trimkit.exchange.svg_string(vertices, wrong)


# found by fuzz/svgpath.cpp: `svg_string` refused what `svg_parse` produced,
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
    vertices, segments = trimkit.exchange.svg_parse(path)
    assert np.isfinite(vertices).all() and len(segments) > 0
    round_trip(path)


def test_copied_layout_outlives_the_call():
    # any other layout is copied once, and that copy must stay alive while it is
    # read: big enough for numpy to really free it, so ASan sees any later read
    vertices, segments = trimkit.exchange.svg_parse("M0 0 " + "L1 1 " * 5000)
    reversed_rows = segments[::-1]
    assert trimkit.exchange.svg_string(vertices, reversed_rows) == trimkit.exchange.svg_string(
        vertices, reversed_rows.copy()
    )
