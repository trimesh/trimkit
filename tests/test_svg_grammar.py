"""
`trimkit.exchange.svg_parse` and `trimkit.exchange.svg_string` against the SVG 2 path grammar
and an independent reading of the spec, `svg_reference`.

- It accepts exactly the `LENIENT` language, and so all of `STRICT`.
- Every segment it emits is bit-identical to the reference's, and every arc's
  center parameterization reproduces its endpoints, flags and direction.
- Malformed data keeps exactly the segments before the error.
- It agrees with `svg.path`, the package it replaces in trimesh.
"""

import numpy as np
import pytest
import svg_reference as ref
from helpers import evaluate_arc, svg_rows
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.lark import from_lark

import trimkit

pytestmark = pytest.mark.filterwarnings("ignore::hypothesis.errors.HypothesisWarning")
# derandomized and without an example database: the same examples everywhere,
# release builds train PGO on them, and the fuzzers do the exploring
SETTINGS = settings(
    max_examples=100, deadline=None, suppress_health_check=list(HealthCheck), database=None, derandomize=True
)

# appended to find out whether everything before it parsed: its line comes
# out last exactly when the data before it was valid
SENTINEL = " M 1234.5678 -8765.4321 L 2468.1357 -1357.2468"


def accepts(d: str) -> bool:
    before, after = (
        svg_rows(*trimkit.exchange.svg_parse(d)),
        svg_rows(*trimkit.exchange.svg_parse(d + SENTINEL)),
    )
    # appending never changes what was already parsed
    assert after[: len(before)].tobytes() == before.tobytes()
    if len(after) != len(before) + 1:
        return False
    last = after[-1]
    return last["start"].tolist() == [1234.5678, -8765.4321] and last["end"].tolist() == [
        2468.1357,
        -1357.2468,
    ]


def _number(draw, strict: bool, signed: bool = True) -> str:
    # one coordinate in any lexical form the grammar allows: leading "+", no
    # leading zero, exponents, underflow and, when lenient, a trailing point
    value = draw(st.floats(-1e6, 1e6, allow_nan=False) | st.integers(-999, 999))
    if not signed:
        value = abs(value)
    forms = ["{!r}", "{:.6g}", "{:e}", "{:.3f}e-999"] + ([] if strict else ["{:.0f}."])
    text = draw(st.sampled_from(forms)).format(value)
    if draw(st.booleans()) and text.lstrip("-").startswith("0.") and text[-1] != ".":
        text = text.replace("0.", ".", 1)
    if draw(st.booleans()) and not text.startswith("-"):
        text = "+" + text if signed else text
    return text


@st.composite
def paths(draw, strict: bool = False) -> str:
    """
    Valid path data, written in every way the grammar allows: STRICT's, or
    with the LENIENT language's commas, signed arc numbers and trailing points.
    """
    out = []

    def separate(token: str, first: bool = False) -> None:
        # nothing between tokens only where they can't merge: before a sign.
        # STRICT allows a comma only between two arguments
        options = [" ", "\t", "\n "]
        if not (strict and first):
            options += [",", " , "]
        if token[0] in "+-":
            options.append("")
        out.append(draw(st.sampled_from(options)) + token)

    def command(letter: str) -> None:
        if out:
            out.append(draw(st.sampled_from(["", " ", "\n"] + ([] if strict else [","]))))
        out.append(letter)

    command(draw(st.sampled_from(["M", "m"])))
    for i in range(2):
        separate(_number(draw, strict), first=i == 0)
    for _ in range(draw(st.integers(0, 12))):
        letter = draw(st.sampled_from("MmLlHhVvCcSsQqTtAaZz"))
        command(letter)
        repeats = 1 if letter in "Zz" else draw(st.integers(1, 3))
        for r in range(repeats):
            if letter in "Aa":
                for i in range(3):
                    separate(_number(draw, strict, signed=not strict), first=r == 0 and i == 0)
                # flags are single characters, so they pack against each other
                # and the coordinate after them, but a separator must end the
                # rotation before them or the first flag reads as its digit
                for packed in (False, True):
                    gap = draw(st.sampled_from(["", " ", ","] if packed else [" ", ","]))
                    out.append(gap + draw(st.sampled_from("01")))
                for _ in range(2):
                    separate(_number(draw, strict))
            else:
                for i in range(ref.ARGUMENTS[letter.upper()]):
                    separate(_number(draw, strict), first=r == 0 and i == 0)
    return "".join(out)


TEXT = st.text(alphabet="MmLlHhVvCcSsQqTtAaZzXe0123456789.+- ,\t\n", max_size=40)


@SETTINGS
@given(st.one_of(paths(), from_lark(ref.LENIENT_PARSER), from_lark(ref.STRICT_PARSER), TEXT))
def test_accepts_exactly_the_lenient_language(d):
    accepted = accepts(d)
    assert accepted == (ref.parse(ref.LENIENT_PARSER, d) is not None)
    if ref.parse(ref.STRICT_PARSER, d) is not None:
        assert accepted


@SETTINGS
@given(paths(strict=True))
def test_strict_paths_are_strict(d):
    # the generator writes STRICT data when asked, so svg.path can read it
    assert ref.parse(ref.STRICT_PARSER, d) is not None


@SETTINGS
@given(paths())
def test_segments_match_reference(d):
    commands = ref.parse(ref.LENIENT_PARSER, d)
    assert commands is not None
    expected = ref.evaluate(commands)
    s = svg_rows(*trimkit.exchange.svg_parse(d))
    assert len(s) == len(expected)
    for row, e in zip(s, expected, strict=True):
        # the same doubles: both parses round correctly, and the arithmetic
        # on them is the spec's, in the same order
        assert row["kind"] == e["kind"] and bool(row["closes"]) == bool(e["flags"] & ref.CLOSE)
        assert row["moves"] == e["moves"]
        assert row["start"].tolist() == list(e["start"])
        assert row["end"].tolist() == list(e["end"])
        used = len(e.get("control", ()))
        assert row["control"][:used].tolist() == [list(c) for c in e.get("control", ())]
        assert np.isnan(row["control"][used:]).all()
        if e["kind"] == ref.ARC:
            assert row["rotation"] == e["rotation"]
            # the radii as written, or scaled up together to just reach both
            # endpoints, which leaves a half ellipse (F.6.6)
            k = row["radius"] / np.array(e["radius"])
            assert np.isclose(k[0], k[1], rtol=1e-12) and k.min() >= 1 - 1e-12
            if not np.isclose(abs(row["angles"][1]), np.pi, rtol=1e-9):
                assert row["radius"].tolist() == list(e["radius"])
    arc_flags = np.array([e["flags"] for e in expected if e["kind"] == ref.ARC], dtype=np.int64)
    _check_arcs(s[s["kind"] == trimkit.exchange.SvgKind.ARC], arc_flags)


def _check_arcs(arcs: np.ndarray, flags: np.ndarray) -> None:
    # every arc's center parameterization reproduces both its endpoints and
    # its halfway point, its
    # direction is the sweep flag as written (`flags`, from the reference),
    # and it is the large arc exactly when asked
    # (radii scaled up to reach both endpoints make a half ellipse, F.6.6).
    # With the radii checked against the input, that leaves one center
    start, sweep = arcs["angles"].T
    scale = 1.0 + np.abs(arcs["center"]).sum(axis=1) + arcs["radius"].sum(axis=1)
    for angle, point in (
        (start, arcs["start"]),
        (start + sweep / 2, arcs["middle"]),
        (start + sweep, arcs["end"]),
    ):
        on = evaluate_arc(arcs, angle)
        assert (np.abs(on - point).max(axis=1, initial=0) <= 1e-9 * scale).all()
    assert np.array_equal(sweep > 0, (flags & ref.SWEEP) > 0)
    half = np.isclose(np.abs(sweep), np.pi, rtol=1e-9)
    large = (flags & ref.LARGE_ARC) > 0
    assert (half | ((np.abs(sweep) > np.pi) == large)).all()


@SETTINGS
@given(paths(), st.sampled_from([" X", " L 1", " A 1 1 0 2 0 1 1", " L 1e999 0", " L - 1"]))
def test_error_keeps_the_prefix(d, bad):
    # SVG renders up to the command with the first error
    whole, prefix = trimkit.exchange.svg_parse(d + bad + SENTINEL), trimkit.exchange.svg_parse(d)
    assert whole[0].tobytes() == prefix[0].tobytes() and whole[1].tobytes() == prefix[1].tobytes()


@SETTINGS
@given(paths(strict=True))
def test_matches_svg_path(d):
    # the parser trimesh used until now reads the same segments
    from svg.path import Arc, Close, CubicBezier, Line, Move, QuadraticBezier, parse_path

    try:
        old = [seg for seg in parse_path(d) if not isinstance(seg, Move)]
    except ZeroDivisionError:
        # svg.path divides by radii like 1e-178, which we draw
        return
    # svg.path keeps arcs to their own start, which draw nothing (F.6.2)
    old = [seg for seg in old if not (isinstance(seg, Arc) and seg.start == seg.end)]
    s = svg_rows(*trimkit.exchange.svg_parse(d))
    assert len(old) == len(s)
    for seg, row in zip(old, s, strict=True):
        assert [seg.start.real, seg.start.imag] == row["start"].tolist()
        assert [seg.end.real, seg.end.imag] == row["end"].tolist()
        kind = {
            Line: trimkit.exchange.SvgKind.LINE,
            Close: trimkit.exchange.SvgKind.LINE,
            QuadraticBezier: trimkit.exchange.SvgKind.QUADRATIC,
            CubicBezier: trimkit.exchange.SvgKind.CUBIC,
        }
        # arc centers aren't compared: svg.path's are only good to about 1e-6
        # and wrong for extreme radii, while `_check_arcs` pins ours exactly.
        # svg.path draws every arc with nonzero radii, where we draw a line for
        # one with no ellipse within the coordinate bound (`_drawable`)
        if (
            isinstance(seg, Arc)
            and seg.radius.real != 0
            and seg.radius.imag != 0
            and ref._drawable(
                (seg.start.real, seg.start.imag),
                (seg.end.real, seg.end.imag),
                seg.radius.real,
                seg.radius.imag,
                seg.rotation,
            )
        ):
            assert row["kind"] == trimkit.exchange.SvgKind.ARC
        elif isinstance(seg, Arc):
            assert row["kind"] == trimkit.exchange.SvgKind.LINE
        else:
            assert row["kind"] == kind[type(seg)]
            assert bool(row["closes"]) == isinstance(seg, Close)
        if isinstance(seg, CubicBezier):
            assert row["control"].tolist() == [[c.real, c.imag] for c in (seg.control1, seg.control2)]
        if isinstance(seg, QuadraticBezier):
            assert row["control"][0].tolist() == [seg.control.real, seg.control.imag]


@SETTINGS
@given(paths())
def test_string_round_trips(d):
    # whatever the grammar allows, writing the segments back and parsing
    # them again gives the same segments, bit for bit, NaN slots included
    vertices, segments = trimkit.exchange.svg_parse(d)
    written = trimkit.exchange.svg_string(vertices, segments)
    again = trimkit.exchange.svg_parse(written)
    assert again[0].tobytes() == vertices.tobytes() and again[1].tobytes() == segments.tobytes()
    # and what is written is path data the reference grammar accepts: the
    # lenient one only for a signed arc rotation, kept as parsed (SVG 1.1).
    # Arcs are written with their F.6.6-scaled radii, the same ellipse
    assert ref.parse(ref.LENIENT_PARSER, written) is not None or written == ""
