"""
An independent reading of SVG path data, from the SVG 2 specification.

`STRICT` transcribes the path data grammar of SVG 2 (paths.html#PathDataBNF)
into Lark. Whitespace may sit between any two tokens, so it is ignored. Each
number is one greedy token, as the spec requires ("M 100-200" is 100, -200;
"M0.6.5" is 0.6, .5). LALR with a contextual lexer means an arc flag position
only ever lexes a single "0" or "1".

`LENIENT` is the language `svgpath` accepts. It differs from `STRICT` in
exactly three ways:
- commas are whitespace anywhere;
- arc radii and rotation may be signed, as real files write them and SVG 1.1
  allowed for rotation (radii then use their magnitude, F.6.6);
- a number may end in a point, as in SVG 1.1 ("1.").

`evaluate` walks a parse into absolute segments, the way SVG 2 §9.3 and
Appendix F.6 describe them.
"""

import math
import sys

from lark import Lark, Token

TEMPLATE = r"""
start: (moveto drawto*)?
?drawto: moveto | closepath | lineto | horizontal | vertical | curveto
       | smooth_curveto | quadratic | smooth_quadratic | arc
moveto: /[Mm]/ pair (SEP? pair)*
closepath: /[Zz]/
lineto: /[Ll]/ pair (SEP? pair)*
horizontal: /[Hh]/ NUMBER (SEP? NUMBER)*
vertical: /[Vv]/ NUMBER (SEP? NUMBER)*
curveto: /[Cc]/ triplet (SEP? triplet)*
smooth_curveto: /[Ss]/ double (SEP? double)*
quadratic: /[Qq]/ double (SEP? double)*
smooth_quadratic: /[Tt]/ pair (SEP? pair)*
arc: /[Aa]/ arc_argument (SEP? arc_argument)*
arc_argument: ARC_NUMBER SEP? ARC_NUMBER SEP? ARC_NUMBER SEP? FLAG SEP? FLAG SEP? arc_pair
// its own rule, so the parser state after an arc knows the next arc's
// unsigned radius may follow, rather than sharing a state with `pair`
arc_pair: NUMBER SEP? NUMBER
triplet: pair SEP? pair SEP? pair
double: pair SEP? pair
pair: NUMBER SEP? NUMBER
FLAG: /[01]/
"""

# sign? (digit* "." digit+ | digit+) exponent?
_UNSIGNED = r"([0-9]*\.[0-9]+|[0-9]+)([eE][+-]?[0-9]+)?"

STRICT = (
    TEMPLATE
    + rf"""
NUMBER: /[+-]?{_UNSIGNED}/
ARC_NUMBER: /{_UNSIGNED}/
SEP: ","
%ignore /[ \t\n\f\r]+/
"""
)

# the leniencies: a trailing point, signed arc numbers, commas as whitespace
_LENIENT_NUMBER = r"/[+-]?([0-9]*\.[0-9]+|[0-9]+\.?)([eE][+-]?[0-9]+)?/"
LENIENT = (
    TEMPLATE.replace("SEP? ", "").replace("ARC_NUMBER", "NUMBER")
    + rf"""
NUMBER: {_LENIENT_NUMBER}
%ignore /[ \t\n\f\r,]+/
"""
)

# trimkit's bound on every coordinate (src/numpy.hpp, vendor/svgpath)
COORDINATE_LIMIT = 1e30

STRICT_PARSER = Lark(STRICT, parser="lalr", lexer="contextual")
LENIENT_PARSER = Lark(LENIENT, parser="lalr", lexer="contextual")

# segment kinds and flags, as `trimkit.exchange.SvgSegment` stores them
LINE, QUADRATIC, CUBIC, ARC = 0, 1, 2, 3
LARGE_ARC, SWEEP, CLOSE = 1, 2, 4

ARGUMENTS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def parse(grammar: Lark, d: str):
    """
    The parse of `d`, or None where the grammar rejects it. A number, or a
    point it makes (relative steps, reflections), beyond +-1e30 is an error, as
    is one too large for a double: trimkit holds every coordinate to 1e30.
    """
    try:
        tree = grammar.parse(d)
    except Exception:
        return None
    commands = []
    for node in tree.children:
        tokens = [t for t in node.scan_values(lambda v: isinstance(v, Token)) if t.type != "SEP"]
        values = [float(t) for t in tokens[1:]]
        if not all(abs(v) <= COORDINATE_LIMIT for v in values):
            return None
        commands.append((str(tokens[0]), values))
    points = [p for s in evaluate(commands) for p in (s["start"], s["end"], *s.get("control", ()))]
    if not all(abs(c) <= COORDINATE_LIMIT for p in points for c in p):
        return None
    return commands


def _drawable(start, end, rx, ry, rotation) -> bool:
    # F.6.5.1 and F.6.6: the radii, scaled up when too small (beyond the
    # parser's 16 ulp band around exactly reaching), are finite and, when
    # scaled, normal: subnormal ones can't be written back precisely
    phi = math.radians(rotation)
    dx, dy = (start[0] - end[0]) / 2.0, (start[1] - end[1]) / 2.0
    x1 = math.cos(phi) * dx + math.sin(phi) * dy
    y1 = -math.sin(phi) * dx + math.cos(phi) * dy
    reach = math.hypot(x1 / abs(rx), y1 / abs(ry))
    if reach <= 1.0 + 16.0 * sys.float_info.epsilon:
        return True
    scaled = (abs(rx) * reach, abs(ry) * reach)
    # and, past the coordinate bound, would reach beyond it: a line instead
    return all(sys.float_info.min <= r <= COORDINATE_LIMIT for r in scaled)


def evaluate(commands) -> list[dict]:
    """
    The segments that a parse draws, in absolute coordinates.
    """
    out = []
    current = start = (0.0, 0.0)
    # the last command, and its last control point, for S and T reflections
    last, control = None, None
    # a moveto starts a subpath on a new vertex at the next segment drawn, and
    # closing a subpath with nothing drawn yet draws nothing
    moved = False
    drawn = False
    for letter, values in commands:
        upper = letter.upper()
        n = ARGUMENTS[upper]
        groups = [values[i : i + n] for i in range(0, len(values), n)] if n else [[]]
        for k, a in enumerate(groups):
            # coordinate pairs after a moveto's first are implicit linetos
            command = "L" if upper == "M" and k > 0 else upper
            ox, oy = current if letter.islower() else (0.0, 0.0)

            def point(x, y):
                return (ox + x, oy + y)

            def reflect(kinds):
                if last in kinds:
                    return (2.0 * current[0] - control[0], 2.0 * current[1] - control[1])
                return current

            segment = None
            if command == "M":
                current = start = point(*a)
                moved = True
                drawn = False
            elif command == "Z":
                if drawn:
                    segment = dict(kind=LINE, flags=CLOSE, end=start)
            elif command == "L":
                segment = dict(kind=LINE, end=point(*a))
            elif command == "H":
                segment = dict(kind=LINE, end=(ox + a[0], current[1]))
            elif command == "V":
                segment = dict(kind=LINE, end=(current[0], oy + a[0]))
            elif command in "CS":
                first = point(a[0], a[1]) if command == "C" else reflect("CS")
                second = point(*a[-4:-2])
                segment = dict(kind=CUBIC, control=(first, second), end=point(*a[-2:]))
                control = second
            elif command in "QT":
                first = point(a[0], a[1]) if command == "Q" else reflect("QT")
                segment = dict(kind=QUADRATIC, control=(first,), end=point(*a[-2:]))
                control = first
            elif command == "A":
                end = point(a[5], a[6])
                if end == current:
                    pass  # an arc to its own start draws nothing, F.6.2
                elif a[0] == 0.0 or a[1] == 0.0 or not _drawable(current, end, *a[:3]):
                    # a zero radius is a line, F.6.2, and so are radii that once
                    # scaled to reach both endpoints (F.6.6) overflow a double
                    segment = dict(kind=LINE, end=end)
                else:
                    flags = (LARGE_ARC if a[3] else 0) | (SWEEP if a[4] else 0)
                    radius = (abs(a[0]), abs(a[1]))
                    segment = dict(kind=ARC, flags=flags, rotation=a[2], radius=radius, end=end)
            if segment is not None:
                segment.update(start=current, moves=moved)
                moved = False
                segment.setdefault("flags", 0)
                out.append(segment)
                current = segment["end"]
                drawn = command != "Z"
            last = command
    return out
