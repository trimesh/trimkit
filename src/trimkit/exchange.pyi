"""
Reading and writing data formats: draco and SVG paths.

Attributes
----------
DracoKind : enum.IntEnum
  What a draco attribute holds, as `draco_encode` takes it.
SvgKind : enum.IntEnum
  What drew an SvgSegment, its `kind` field.
SvgSegment : numpy.dtype
  One drawing command of an SVG path, as `svg_parse` returns them and
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
    >>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> faces = np.array([[0, 1, 2]], dtype=np.int64)
    >>> kinds = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
    >>> buffer = trimkit.exchange.draco_encode(faces, [triangle.astype(np.float32)], kinds)
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
    >>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> faces = np.array([[0, 1, 2]], dtype=np.int64)
    >>> kinds = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
    >>> buffer = np.frombuffer(trimkit.exchange.draco_encode(faces, [triangle.astype(np.float32)], kinds), np.uint8)
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
      A straight line: the L, H, V and Z commands.
    QUADRATIC
      A quadratic Bezier curve: Q and T.
    CUBIC
      A cubic Bezier curve: C and S.
    ARC
      An elliptical arc: A.
    """

    LINE = 0
    """A straight line: the L, H, V and Z commands."""

    QUADRATIC = 1
    """A quadratic Bezier curve: Q and T."""

    CUBIC = 2
    """A cubic Bezier curve: C and S."""

    ARC = 3
    """An elliptical arc: A."""

SvgSegment: numpy.dtypes.VoidDType = ...

def svg_parse(path: str) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.void]]:
    """
    Parse SVG path data into vertices and one segment per drawing command.

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
    ([[0.0, 0.0], [1.0, 1.0]], [0])
    """

def svg_string(vertices: numpy.typing.NDArray[numpy.float64], segments: numpy.typing.NDArray[numpy.void]) -> str:
    """
    Write vertices and segments back as SVG path data, the inverse of `svg_parse`.

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
    >>> path = trimkit.exchange.svg_string(*trimkit.exchange.svg_parse("M 0 0 L 1 1 Z"))
    >>> path
    'M0 0 L1 1 Z'
    """
