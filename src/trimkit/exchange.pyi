"""
Reading and writing data formats: draco and SVG paths.

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
True
"""

from collections.abc import Sequence
import enum
from typing import Annotated

import numpy
import numpy.dtypes
import numpy.typing
from numpy.typing import NDArray


class DracoKind(enum.IntEnum):
    """
    What a draco attribute holds, as `draco_encode` takes it.

    Attributes
    ----------
    POSITION
      Vertex positions: float ones quantized to 14 bits by default.
    NORMAL
      Vertex normals: float ones quantized to 10 bits by default.
    TEXCOORD
      Texture coordinates: float ones quantized to 12 bits by default.
    COLOR
      Vertex colors: float ones quantized to 10 bits by default.
    GENERIC
      Anything else, stored losslessly by default.
    """

    POSITION = 0
    """Vertex positions: float ones quantized to 14 bits by default."""

    NORMAL = 1
    """Vertex normals: float ones quantized to 10 bits by default."""

    TEXCOORD = 2
    """Texture coordinates: float ones quantized to 12 bits by default."""

    COLOR = 3
    """Vertex colors: float ones quantized to 10 bits by default."""

    GENERIC = 4
    """Anything else, stored losslessly by default."""

def draco_encode(faces: Annotated[NDArray[numpy.int64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], attributes: Sequence[NDArray], kinds: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)], *, quantization_bits: int | numpy.integer | None = None) -> bytes:
    """
    Compress a glTF triangle primitive, keeping vertex order.

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
    'bytes'
    """

def draco_decode(buffer: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], draco_attributes: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> tuple[NDArray[numpy.int64], tuple[NDArray, ...]]:
    """
    Decompress a glTF draco buffer, decoding it once.

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
    ([[0, 1, 2]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    """

class SvgKind(enum.IntEnum):
    """
    What drew an SvgSegment, its `kind` field.

    Attributes
    ----------
    LINE
      A straight line: points start, -1, -1, end. The L, H, V and Z commands.
    QUADRATIC
      A quadratic Bezier curve: points start, control, -1, end. Q and T.
    CUBIC
      A cubic Bezier curve: points start, control, control, end. C and S.
    ARC
      A circular arc of at most 120 degrees: points start, the point halfway along it, -1, end, and kind_data its center. A with equal radii.
    CONIC
      An elliptical arc of at most 120 degrees, an exact rational quadratic Bezier: points start, control, -1, end, and kind_data its weight, then 0. A with unequal radii.
    """

    LINE = 0
    """
    A straight line: points start, -1, -1, end. The L, H, V and Z commands.
    """

    QUADRATIC = 1
    """A quadratic Bezier curve: points start, control, -1, end. Q and T."""

    CUBIC = 2
    """A cubic Bezier curve: points start, control, control, end. C and S."""

    ARC = 3
    """
    A circular arc of at most 120 degrees: points start, the point halfway along it, -1, end, and kind_data its center. A with equal radii.
    """

    CONIC = 4
    """
    An elliptical arc of at most 120 degrees, an exact rational quadratic Bezier: points start, control, -1, end, and kind_data its weight, then 0. A with unequal radii.
    """

SvgSegment: numpy.dtypes.VoidDType = ...

def svg_parse(path: str) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.void]]:
    """
    Parse SVG path data into vertices and one record per segment.

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
    ([[0.0, 0.0], [1.0, 1.0]], [0], [[0, -1, -1, 1]])
    """

def svg_string(vertices: numpy.typing.NDArray[numpy.float64], segments: numpy.typing.NDArray[numpy.void], *, digits: int = 8) -> str:
    """
    Write vertices and segments back as SVG path data, the inverse of `svg_parse`.

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
    'M0 0 L1 1 Z'
    """
