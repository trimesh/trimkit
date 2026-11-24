"""
Small numpy-only fixtures and predicates shared by the tests.
"""

import ast
import contextlib
import ctypes
import dataclasses
import functools
import math
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

# draco refuses big-endian machines outright (its format is little-endian and
# it copies raw bytes): test_invariants asserts that, and everything else
# skips draco there, saying so
LITTLE_ENDIAN = sys.byteorder == "little"


def run_python(
    *args: str, timeout: float = 600, check: bool = False, **options
) -> subprocess.CompletedProcess:
    """
    `python *args` with its output captured, as subprocess.run would run it.
    A child still running after `timeout` seconds gets SIGABRT first, so its
    faulthandler (on in every child: conftest.py) prints every thread's stack,
    and the test fails with that stderr: a hang says where it hung. `check`
    fails the test with the stderr too, not only the exit code.
    """
    pipes = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    with subprocess.Popen([sys.executable, *args], **pipes, **options) as child:
        try:
            out, err = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                child.send_signal(signal.SIGABRT)
            try:
                out, err = child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                out, err = child.communicate()
            pytest.fail(f"a child Python was still running after {timeout} s:\n{_tail(err)}")
    if check and child.returncode != 0:
        pytest.fail(f"a child Python exited with {child.returncode}:\n{_tail(err)}")
    return subprocess.CompletedProcess(child.args, child.returncode, out, err)


def _tail(output: str | bytes) -> str:
    return (output if isinstance(output, str) else output.decode(errors="replace"))[-4000:]


def resident() -> int | None:
    """
    Bytes of this process in memory now, as /proc counts them: what a
    device's driver maps for itself too, which no allocator's count sees.
    None where there is no /proc.
    """
    try:
        with open("/proc/self/statm") as statm:
            return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        return None


def heap_in_use() -> int | None:
    """
    Bytes the process has allocated and not freed, every allocator call
    counted (numpy's buffers, ours, the vendored libraries'), as glibc's
    mallinfo2 counts them: None where there is no glibc to ask.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        query = libc.mallinfo2
    except (OSError, AttributeError):
        return None

    class Mallinfo2(ctypes.Structure):
        _fields_ = [
            (field, ctypes.c_size_t)
            for field in (
                *("arena", "ordblks", "smblks", "hblks", "hblkhd"),
                *("usmblks", "fsmblks", "uordblks", "fordblks", "keepcost"),
            )
        ]

    query.restype = Mallinfo2
    info = query()
    return info.uordblks + info.hblkhd


def runs_here(name: str) -> None:
    """
    Skip a test of public call `name` where it refuses to run at all, with why.
    """
    if name.startswith("exchange.draco_") and not LITTLE_ENDIAN:
        pytest.skip("draco refuses big-endian machines, as test_draco_refuses_only_big_endian asserts")


def _varint(number: int) -> bytes:
    out = bytearray()
    while number > 0x7F:
        out.append(number & 0x7F | 0x80)
        number >>= 7
    out.append(number)
    return bytes(out)


def draco_points_bomb(points: int) -> bytes:
    """
    A buffer of a few KB whose one triangle declares `points` points, each with
    16 float32 components: draco's sizes come from this, not from what follows.
    """
    import trimkit

    count = 300
    faces = np.arange(count, dtype=np.int64).reshape((-1, 3))
    kind = np.array([trimkit.exchange.DracoKind.GENERIC], np.int64)
    value = np.zeros((count, 16), np.float32)
    valid = trimkit.exchange.draco_encode(faces, [value], kind, quantization_bits=0)
    # after the header and its two counts: raw connectivity, as uint16 indices
    start = 11 + len(_varint(count // 3)) + len(_varint(count))
    assert valid[start] == 1
    attributes = valid[start + 1 + count * 2 :]
    return valid[:11] + _varint(1) + _varint(points) + b"\x01" + struct.pack("<3I", 0, 1, 2) + attributes


def draco_faces_bomb() -> bytes:
    """
    An edgebreaker header alone, declaring all the faces and attributes it may.
    """
    faces = 1_400_000_000
    return (
        b"DRACO\x02\x02\x01\x01\x00\x00\x00"
        + _varint(faces // 2)
        + _varint(faces)
        + b"\xff"
        + _varint(faces)
        + _varint(0)
    )


def sphere(rows: int = 20, cols: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """
    A closed, outward-wound UV sphere with poles collapsed to single vertices.
    """
    theta = np.linspace(0, np.pi, rows)[1:-1]
    phi = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    ring = np.column_stack(
        [(np.sin(t) * np.cos(p)).ravel(), (np.sin(t) * np.sin(p)).ravel(), np.cos(t).ravel()]
    )
    vertices = np.vstack([[0, 0, 1], ring, [0, 0, -1]])

    i, j = np.meshgrid(np.arange(rows - 3, dtype=np.int64), np.arange(cols, dtype=np.int64), indexing="ij")
    a = 1 + i * cols + j
    b = 1 + i * cols + (j + 1) % cols
    c, d = a + cols, b + cols
    bottom = len(vertices) - 1
    k = np.arange(cols, dtype=np.int64)
    faces = np.vstack(
        [
            np.column_stack([np.zeros(cols, int), 1 + k, 1 + (k + 1) % cols]),
            np.column_stack([a.ravel(), c.ravel(), b.ravel()]),
            np.column_stack([b.ravel(), c.ravel(), d.ravel()]),
            np.column_stack([bottom - cols + (k + 1) % cols, bottom - cols + k, np.full(cols, bottom)]),
        ]
    )
    # int64 as every face index is, whatever numpy's default int is here
    return vertices, faces.astype(np.int64)


def box(lo=(0, 0, 0), hi=(1, 1, 1)) -> tuple[np.ndarray, np.ndarray]:
    """
    An axis-aligned box wound outward.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    vertices = lo + corners * (hi - lo)
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=np.int64,
    )  # fmt: skip
    return vertices, faces


@functools.cache
def cycloidal() -> tuple[list, np.ndarray, np.ndarray]:
    """
    A real assembly, a cycloidal drive in millimetres: 13 meshes placed 72
    times, 61,442 faces (376,520 as placed), none of them closed. From
    trimesh's models/cycloidal.3DXML, every coordinate as trimesh loads it.
    Returns (meshes, instance_matrices, instance_index).
    """
    with np.load(Path(__file__).parent / "models" / "cycloidal.npz") as data:
        # stored little-endian, read as this machine's float64
        vertices = np.split(data["vertices"].astype(np.float64), np.cumsum(data["vertex_counts"])[:-1])
        faces = np.split(data["faces"].astype(np.int64), np.cumsum(data["face_counts"])[:-1])
        meshes = list(zip(map(np.ascontiguousarray, vertices), map(np.ascontiguousarray, faces), strict=True))
        return meshes, data["instance_matrices"].astype(np.float64), data["instance_index"].astype(np.int64)


def subdivided(mesh) -> tuple[np.ndarray, np.ndarray]:
    """
    A mesh with each face split in four at its edges' midpoints, an edge two
    faces share getting one midpoint.
    """
    vertices, faces = mesh
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    a, b, c = (inverse.reshape(-1, 3) + len(vertices)).T
    v0, v1, v2 = faces.T
    split = np.column_stack([v0, a, c, a, v1, b, c, b, v2, a, b, c]).reshape(-1, 3)
    return np.vstack([vertices, vertices[unique].mean(axis=1)]), split


def exploded(meshes, matrices, index, factor: float) -> np.ndarray:
    """
    The instances' transforms with each moved away from the middle of them
    all, `factor` times as far as it was.
    """
    placed = np.array([transform(meshes[i][0], m).mean(axis=0) for i, m in zip(index, matrices, strict=True)])
    moved = matrices.copy()
    moved[:, :3, 3] += (placed - placed.mean(axis=0)) * (factor - 1.0)
    return moved


def camera(position, target, pixels: int, field_of_view: float = 40.0) -> np.ndarray:
    """
    The (pixels * pixels, 3) directions of a pinhole camera at `position`
    looking at `target` with z up, one per pixel.
    """
    forward = np.asarray(target, float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    span = np.tan(np.radians(field_of_view / 2)) * np.linspace(-1, 1, pixels)
    u, v = np.meshgrid(span, span)
    up = np.cross(right, forward)
    return np.ascontiguousarray((forward + u[..., None] * right + v[..., None] * up).reshape(-1, 3))


def instances(meshes, matrices=None, index=None) -> tuple[np.ndarray, np.ndarray]:
    """
    A scene's instances spelled out, with every scene's defaults: each mesh
    once (or one per matrix), at the identity.
    """
    if index is None:
        index = np.arange(len(meshes) if matrices is None else len(matrices), dtype=np.int64)
    if matrices is None:
        matrices = np.tile(np.eye(4), (len(index), 1, 1))
    return np.asarray(matrices), np.asarray(index)


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Points carried by a 4x4 affine transform.
    """
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def flatten(meshes, matrices=None, index=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Every instance's transformed copy concatenated into one mesh, and where
    each instance's faces start in it.
    """
    matrices, index = instances(meshes, matrices, index)
    parts = [transform(meshes[g][0], m) for g, m in zip(index, matrices, strict=True)]
    offsets = np.cumsum([0] + [len(p) for p in parts])
    vertices = np.ascontiguousarray(np.vstack(parts)) if parts else np.zeros((0, 3))
    faces = [meshes[g][1] + o for g, o in zip(index, offsets[:-1], strict=True)]
    faces = np.vstack(faces) if faces else np.zeros((0, 3), np.int64)
    starts = np.cumsum([0] + [len(meshes[g][1]) for g in index])
    return vertices, faces, starts


def check_boxes(scene, meshes, matrices=None, index=None) -> None:
    """
    That a scene of `meshes` knows where these instances of them are, by its
    `boxes(...)` (or `scene` is those Boxes already): one 3D box per
    instance, between the box of its placed faces and its mesh's whole box
    carried by its transform (Arvo), to a few ulp (and a ray scene's padding
    of 1e-12 of the mesh's extent); an instance of a mesh without faces, the
    point at its translation. Round-trips through `Boxes` too.
    """
    import trimkit

    boxes = scene
    if not isinstance(scene, trimkit.Boxes):
        assert len(scene) == len(meshes)
        boxes = scene.boxes(matrices, index)
    matrices, index = instances(meshes, matrices, index)
    assert isinstance(boxes, trimkit.Boxes) and len(boxes) == len(index)
    corners = boxes.boxes()
    assert corners.shape == (len(index), 2, 3)
    assert np.array_equal(trimkit.Boxes(corners).boxes(), corners)
    eps = np.finfo(np.float64).eps
    for (low, high), g, m in zip(corners, index, matrices, strict=True):
        vertices, faces = meshes[g]
        if len(faces) == 0:
            assert np.array_equal(low, m[:3, 3]) and np.array_equal(high, m[:3, 3])
            continue
        placed = transform(vertices[faces.ravel()], m)
        local = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        local += np.array([[-1.0], [1.0]]) * 2e-12 * np.abs(vertices).max()
        center = transform(local.mean(axis=0), m)
        half = np.abs(m[:3, :3]) @ ((local[1] - local[0]) / 2)
        slack = 16 * eps * max(1.0, np.abs(center).max() + half.max())
        assert (low <= placed.min(axis=0) + slack).all() and (high >= placed.max(axis=0) - slack).all()
        assert (low >= center - half - slack).all() and (high <= center + half + slack).all()


def first_hits(scene, origins, directions, *instances, **options):
    """
    The nearest hit of each ray as dense rows, one per ray: its instance and
    face, -1 for a miss, and its distance, inf for a miss. `intersect` with
    `maximum_hits=1`, scattered back onto the rays.
    """
    rays, instances_hit, faces, distances = scene.intersect(
        origins, directions, *instances, maximum_hits=1, **options
    )
    instance_of = np.full(len(origins), -1, dtype=np.int64)
    face_of = np.full(len(origins), -1, dtype=np.int64)
    distance_of = np.full(len(origins), np.inf)
    instance_of[rays], face_of[rays], distance_of[rays] = instances_hit, faces, distances
    return instance_of, face_of, distance_of


def device_scene(meshes, device_required):
    """
    A RayScene on the device, or a skip (with `--device-required`, a failure)
    quoting the warning the scene gives when there is none. A fallback to the
    CPU later warns again, which the suite's `filterwarnings = error` raises.
    """
    import trimkit

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scene = trimkit.RayScene(meshes, use_gpu=True)
    if caught:
        message = str(caught[0].message)
        assert message.startswith("use_gpu: ")
        if device_required:
            pytest.fail(message)
        pytest.skip(message)
    return scene


def ray_scene(meshes, matrices=None, index=None):
    """
    A RayScene of these meshes, checked to know where these instances of
    them are: its queries then name the same instances.
    """
    import trimkit

    scene = trimkit.RayScene(meshes)
    check_boxes(scene, meshes, matrices, index)
    return scene


def ball_runs(radius: int) -> np.ndarray:
    """
    The voxels within `radius` of the origin as (n, 4) runs along +Z.
    """
    x, y = (g.ravel() for g in np.mgrid[-radius : radius + 1, -radius : radius + 1])
    inside = radius**2 - x**2 - y**2
    keep = inside >= 0
    half = np.sqrt(inside[keep]).astype(np.int64)
    return np.column_stack([x[keep], y[keep], -half, 2 * half + 1]).astype(np.int64)


def circle(n: int) -> np.ndarray:
    """
    `n` points evenly around the unit circle, counter-clockwise.
    """
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angle), np.sin(angle)])


def areas2d(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Signed area of each 2D triangle, positive when counter-clockwise.
    """
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return 0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])


def leaves(result) -> list:
    """
    Every array, bytes, str, int, float or None in a result, in a fixed
    order: an object's as what it holds, a scene's the boxes of its meshes.
    """
    import trimkit

    if isinstance(result, (tuple, list)):
        return [a for r in result for a in leaves(r)]
    if isinstance(result, trimkit.CollisionResult):
        return leaves(collision_fields(result))
    if isinstance(result, trimkit.Boxes):
        return [result.boxes()]
    if isinstance(result, (trimkit.RayScene, trimkit.CollisionScene)):
        return [len(result), result.boxes().boxes()]
    assert result is None or isinstance(result, (np.ndarray, bytes, str, int, float)), type(result)
    return [result]


COLLISION_FIELDS = ("pairs", "faces", "distances", "points", "normals")


def collision_fields(result) -> list:
    """
    A CollisionResult's arrays in a fixed order, those it holds: its pairs
    always, the rest when its query measured them.
    """
    held = [getattr(result, field) for field in COLLISION_FIELDS]
    return [field for field in held if field is not None]


def public_api() -> list[str]:
    """
    Every public function, class (its constructor) and method of `trimkit` and
    `trimkit.exchange`, as `name`, `Class.method` or `exchange.name`: what the
    call tables below must each cover exactly, so nothing new goes untested (a
    property counts as a call). Constants, the enums, the `Mesh` alias and the
    dtype aren't.
    """
    import trimkit

    names = []
    for prefix, module in (("", trimkit), ("exchange.", trimkit.exchange)):
        names += _public_names(prefix, module)
    return sorted(names)


def _public_names(prefix: str, module) -> list[str]:
    import enum
    import types

    names = []
    for name in dir(module):
        value = getattr(module, name)
        # a submodule and the `Mesh` alias (a generic alias is callable)
        if (
            name.startswith("_")
            or not callable(value)
            or isinstance(value, (types.ModuleType, types.GenericAlias))
            or (isinstance(value, type) and issubclass(value, enum.Enum))
        ):
            continue
        names.append(prefix + name)
        if isinstance(value, type):
            names += [
                f"{prefix}{name}.{member}"
                for member in dir(value)
                if not member.startswith("_")
                and (callable(getattr(value, member)) or isinstance(getattr(value, member), property))
            ]
    return names


# -- the sweep: every public call, every argument through every family -------
#
# `CASES[name]` is how to call each public name, built per size; `families`
# turns each parameter's stub annotation into the variants of a valid value it
# must survive; `checked_call` makes one call and checks everything a call
# promises. test_invariants sweeps every case through them, so a new public
# name can't merge without a case, nor a new annotation without a generator.

SIZES = ("empty", "small", "big", "slow")


@dataclasses.dataclass(frozen=True)
class PublicCallCase:
    """
    How to call one public name, as keyword arguments (a method's include
    its constructor's, which `call` builds its object from), each size built
    fresh by a function, or a string saying why no such call exists. A
    class's calls share their builders: `arguments` gives each those its
    signature takes. `small` builds a list: one per form the signature
    allows, between them every array a union admits, and every family runs
    on each. `oracle` checks an answer against an independent reference.
    `ranges[parameter](arguments)` is (low, high): an integer argument, or an
    index array's first element, answers for low <= value < high and is
    refused otherwise (a float, low <= value <= high). `holds`: a container,
    keeping whatever values it is given.
    """

    name: str
    oracle: Callable[[dict, object], None]
    small: Callable[[], list[dict]]
    empty: Callable[[], dict] | str
    big: Callable[[], dict] | str
    slow: Callable[[], dict] | str
    ranges: dict = dataclasses.field(default_factory=dict)
    holds: bool = False

    def arguments(self, size: str):
        """
        The arguments of the call of this size, None where there is none:
        of what its builder gives, those this name takes (of small forms,
        those that then differ). Every argument built is some call's of its
        class, so a misspelt one can't be dropped unseen.
        """
        build = getattr(self, size)
        if isinstance(build, str):
            return None
        owner = self.name.removeprefix("exchange.").partition(".")[0]
        known, taken = set(), set()
        for name, parameters in stub_parameters().items():
            if name.removeprefix("exchange.").partition(".")[0] == owner:
                known |= {parameter for parameter, *_ in parameters}
            if name == self.name:
                taken = {parameter for parameter, *_ in parameters}
        built = build()
        forms = []
        for given in [built] if isinstance(built, dict) else built:
            assert set(given) <= known, (self.name, sorted(set(given) - known))
            form = {key: value for key, value in given.items() if key in taken}
            if isinstance(built, dict):
                return form
            if _frozen(form) not in map(_frozen, forms):
                forms.append(form)
        return forms


# a parameter without a default
REQUIRED = object()


@functools.cache
def stub_parameters() -> dict[str, list[tuple[str, ast.expr, object, bool]]]:
    """
    Every public name's parameters as the installed stubs declare them:
    (name, annotation, default or REQUIRED, whether its constructor's), a
    method's after its class's constructor's.
    """
    import trimkit

    def literal(node):
        if node is None:
            return REQUIRED
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "float":
            return float(node.args[0].value)
        return ast.literal_eval(node)

    def signature(function: ast.FunctionDef, constructor: bool) -> list:
        given = function.args
        positional = [None] * (len(given.args) - len(given.defaults)) + list(given.defaults)
        pairs = list(zip(given.args, positional, strict=True)) + list(
            zip(given.kwonlyargs, given.kw_defaults, strict=True)
        )
        return [(a.arg, a.annotation, literal(d), constructor) for a, d in pairs if a.arg != "self"]

    found = {}
    for prefix, stub in (("", "__init__.pyi"), ("exchange.", "exchange.pyi")):
        tree = ast.parse((Path(trimkit.__file__).parent / stub).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                found[prefix + node.name] = signature(node, False)
            elif isinstance(node, ast.ClassDef) and not any(
                ast.unparse(b).endswith("Enum") for b in node.bases
            ):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                built = signature(next(m for m in methods if m.name == "__init__"), True)
                found[prefix + node.name] = built
                for method in methods:
                    if not method.name.startswith("_"):
                        found[f"{prefix}{node.name}.{method.name}"] = built + signature(method, False)
    return found


def bound(name: str, arguments: dict) -> Callable[[], object]:
    """
    Public `name` ready to call with keyword `arguments`, as a call of no
    arguments: a method's or property's object built first, from the
    constructor's among them, so calling it runs only the method.
    """
    import trimkit

    module = trimkit.exchange if name.startswith("exchange.") else trimkit
    owner, _, member = name.removeprefix("exchange.").partition(".")
    target = getattr(module, owner)
    if not member:
        return lambda: target(**arguments)
    built = {p for p, _, _, constructor in stub_parameters()[name] if constructor}
    instance = target(**{k: v for k, v in arguments.items() if k in built})
    rest = {k: v for k, v in arguments.items() if k not in built}
    if callable(getattr(type(instance), member)):
        method = getattr(instance, member)
        return lambda: method(**rest)
    return lambda: getattr(instance, member)


def call(name: str, arguments: dict):
    """
    Public `name` called with keyword `arguments`: a method or property on
    an object built from the constructor's among them.
    """
    return bound(name, arguments)()


# -- families: the variants of a valid argument, from its annotation ---------

# the answer to compare with is the form's own, or the oracle judges it
AS_GIVEN, ORACLE = object(), object()


class Variant(NamedTuple):
    """
    One variant of a valid argument and what it must do: `answers` True,
    answer (as the call with `reference` instead does, AS_GIVEN the form's
    own, or as the oracle says, ORACLE); False, be refused with `refusal`
    matching `match`; None, either, never refused matching `forbidden`.
    `warns`: converted, with the one warning, which starts so.
    """

    value: object
    answers: bool | None
    reference: object = ORACLE
    warns: str = ""
    refusal: tuple = (ValueError, TypeError)
    match: str = ""
    forbidden: str = ""


class ArgumentContext(NamedTuple):
    # as messages name it: an item of it `parameter[0]`, a mesh's faces `parameter[0][1]`
    parameter: str
    # the (dtype, ndim) of every array the annotation takes as it is
    accepted: tuple
    # what ranges[parameter] gives, or None
    bounds: tuple | None
    holds: bool


# integers on both sides of every limit there is: a sign, 32 bits, 64 bits
INTEGERS = (-1, 0, 2**31, 2**32, 2**32 + 1, 2**63 - 1)
FLOATS = (math.nan, math.inf, -math.inf, 1e31, -1e31, 1e30, -1e30, -1.0, 0.0, 1.0)


def _union(annotation: ast.expr) -> list[ast.expr]:
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _union(annotation.left) + _union(annotation.right)
    return [annotation]


def _array_member(member: ast.expr) -> tuple | None:
    # (dtype or None for any, ndim or None for any) of an NDArray annotation
    text = ast.unparse(member)
    shaped = re.fullmatch(r"Annotated\[NDArray\[numpy\.(\w+)\], dict\(shape=(?:\(([^)]*)\)|(\d+)).*\]", text)
    if shaped:
        return np.dtype(shaped[1].rstrip("_")), len(
            [s for s in (shaped[2] or shaped[3]).split(",") if s.strip()]
        )
    bare = re.fullmatch(r"(?:numpy\.typing\.)?NDArray(?:\[numpy\.(\w+)\])?", text)
    if bare:
        return (np.dtype(bare[1].rstrip("_")) if bare[1] else None), None
    return None


def families(annotation: ast.expr | str) -> dict[str, Callable]:
    """
    Every family of variants of an argument with this stub annotation, by
    name: each a generator of Variants from a valid value and its
    ArgumentContext, yielding nothing for a value another member of a union
    takes. An annotation with no generator raises.
    """
    import enum

    import trimkit

    if isinstance(annotation, str):
        annotation = ast.parse(annotation, mode="eval").body
    found = {}
    members = _union(annotation)
    accepted = tuple(a for a in map(_array_member, members) if a)
    for member in members:
        text = ast.unparse(member)
        array = _array_member(member)
        named = getattr(trimkit, text, None) or getattr(trimkit.exchange, text, None)
        if array:
            found.update(_array_families(*array, accepted))
        elif text in SCALAR_FAMILIES:
            found.update(SCALAR_FAMILIES[text])
        elif text == "Mesh":
            found.update(_mesh_families())
        elif isinstance(named, type) and issubclass(named, enum.Enum):
            found.update(_enum_families(named))
        elif isinstance(member, ast.Subscript) and ast.unparse(member.value) == "Sequence":
            found.update(_sequence_families(member.slice))
        else:
            raise TypeError(f"no family generates an argument annotated {text}")
    return found


def _bounded(value, context: ArgumentContext, number) -> Variant:
    # a number within the parameter's range answers; outside, its type is
    # right and its value isn't: a ValueError, which names the parameter a
    # number was given for (an index array's may name what it indexes)
    if context.holds:
        return Variant(value, True)
    if context.bounds is None:
        return Variant(value, None)
    low, high = context.bounds[:2]
    inside = low <= number < high if isinstance(number, int) else low <= number <= high
    named = "" if isinstance(value, np.ndarray) else re.escape(context.parameter)
    return Variant(value, inside, refusal=(ValueError,), match=named)


def _relaid(value: np.ndarray):
    # exactly these values in every other layout: strided, reversed views
    # of reversed copies (negative strides, as np.flipud and np.fliplr make),
    # Fortran-ordered and misaligned
    raw = np.zeros(value.nbytes + 1, dtype=np.uint8)
    shifted = np.frombuffer(raw.data, dtype=value.dtype, count=value.size, offset=1).reshape(value.shape)
    shifted[...] = value
    # at an odd byte offset, where every read in place would be undefined
    assert shifted.flags.c_contiguous and (not shifted.flags.aligned or value.dtype.alignment == 1)
    return {
        "strided": np.repeat(value, 2, axis=0)[::2],
        "reversed": np.ascontiguousarray(value[::-1])[::-1],
        "fortran": np.asfortranarray(value) if value.ndim > 1 else None,
        "flipped": np.ascontiguousarray(value[..., ::-1])[..., ::-1] if value.ndim > 1 else None,
        "misaligned": shifted,
    }


def _layout(layout: str):
    # exactly these values in another layout: copied once, answering the same
    def relaid(value, context):
        moved = _relaid(value)[layout]
        if moved is not None:
            yield Variant(moved, True, AS_GIVEN)

    return relaid


def _read_only(value, context):
    frozen = value.copy()
    frozen.flags.writeable = False
    yield Variant(frozen, True, AS_GIVEN)


def _empty(value, context):
    yield Variant(value[:0], None)


def _repeated_rows(value, context):
    yield Variant(np.repeat(value, 2, axis=0), None)


def _listed(value, context):
    # read as numpy reads it: the same values, converted once
    yield _retyped(value, value.tolist(), context)


def _byteswapped(value, context):
    # the other byte order, as a file written on another machine reads: the
    # same values, converted once to this machine's
    if value.dtype.itemsize > 1:
        swapped = value.astype(value.dtype.newbyteorder())
        said = f"{re.escape(context.parameter)}: {swapped.dtype.str} converted to "
        yield Variant(swapped, True, AS_GIVEN, warns=said)


def _narrower(value, context):
    # a narrower dtype of the same kind, which numpy casts back safely
    yield _retyped(value, value.astype({"f8": "f4", "f4": "f2", "i8": "i4"}[value.dtype.str[1:]]), context)


def _lossy(value, context):
    # a dtype that casts back only with loss
    wrong = {"f": np.complex128, "i": np.float64, "b": np.int64, "u": np.int8}[value.dtype.kind]
    yield _retyped(value, value.astype(wrong), context)


def _not_finite(value, context):
    # every float input is refused unless finite, and float64 ones unless
    # within 1e30 too: past it, products downstream overflow
    if value.dtype.kind != "f" or not value.size:
        return
    past = math.nextafter(1e30, math.inf)
    for bad in [math.nan, math.inf, -math.inf] + (
        [1e31, -1e31, past, -past] if value.dtype == np.float64 else []
    ):
        broken = value.copy()
        broken.flat[0] = bad
        if context.holds:
            yield Variant(broken, True)
        else:
            named = re.escape(context.parameter) + ".*must be finite"
            yield Variant(broken, False, refusal=(ValueError,), match=named)


def _at_the_bound(value, context):
    # scaled so its largest magnitude is exactly 1e30, which is in range:
    # refused for something else maybe, never for that
    finite = np.isfinite(value)
    magnitude = np.where(finite, np.abs(value), 0.0)
    if magnitude.max(initial=0.0) == 0:
        return
    largest = np.argmax(magnitude)
    with np.errstate(invalid="ignore"):
        scaled = value * (1e30 / magnitude.flat[largest])
    scaled[finite] = np.clip(scaled[finite], -1e30, 1e30)
    scaled.flat[largest] = math.copysign(1e30, value.flat[largest])
    yield Variant(scaled, None, forbidden="must be finite")


def _power_of_two(value, context):
    for scale in (2.0**-20, 2.0**20):
        with np.errstate(invalid="ignore"):
            scaled = value * scale
        yield Variant(scaled, None)


def _integers_in(value, context):
    # the first element at every limit there is: an index within its range
    # answers, at its end or past it is refused
    if not value.size:
        return
    numbers = list(INTEGERS)
    if context.bounds:
        low, high = context.bounds[:2]
        numbers += [low, high - 1, high]
    for number in numbers:
        broken = value.copy()
        broken.flat[0] = number
        yield _bounded(broken, context, number)


def _two(value, context):
    # a uint8 holding 2 seen as bool: true, if anything
    if value.size:
        raw = value.astype(np.uint8)
        raw.flat[0] = 2
        truth = value.copy()
        truth.flat[0] = True
        yield Variant(raw.view(bool), None, truth)


def _buffers(value, context):
    # the buffer as bytes or a memoryview reads the same, and cut short or
    # with a byte flipped is refused or reads as something consistent
    yield Variant(value.tobytes(), True, AS_GIVEN)
    yield Variant(memoryview(value.tobytes()), True, AS_GIVEN)
    yield Variant(value[: len(value) // 2].copy(), None)
    flipped = value.copy()
    flipped[len(flipped) // 2] ^= 0xFF
    yield Variant(flipped, None)


def _fields(value, context):
    # records of any other dtype are refused; each numeric field's first
    # value at every limit it holds is refused or written
    yield Variant(np.zeros(len(value)), False, refusal=(TypeError,))
    for field in value.dtype.names if value.size else ():
        base = value.dtype[field].base
        numbers = FLOATS
        if base.kind in "iu":
            limits = np.iinfo(base)
            numbers = [n for n in (*INTEGERS, limits.max) if limits.min <= n <= limits.max]
        for number in numbers:
            broken = value.copy()
            broken[field].flat[0] = number
            yield Variant(broken, None)


# Every family of an array argument: its name, which arrays it varies (by
# dtype kind, f8 for float64 alone, * for an array of any dtype), and its
# generator. Fortran order and flipped columns only have a meaning in two
# dimensions or more
ARRAY_FAMILIES = (
    ("strided", "fiubV*", _layout("strided")),
    ("reversed", "fiubV*", _layout("reversed")),
    ("misaligned", "fiubV*", _layout("misaligned")),
    ("fortran", "fiub*", _layout("fortran")),
    ("flipped", "fiub*", _layout("flipped")),
    ("read-only", "fiubV*", _read_only),
    ("empty", "fiubV*", _empty),
    ("repeated rows", "fiubV*", _repeated_rows),
    ("list", "fib*", _listed),
    ("byteswapped", "fi*", _byteswapped),
    ("narrower", "fi", _narrower),
    ("lossy", "fibu", _lossy),
    ("not finite", "f*", _not_finite),
    ("1e30", ("f8",), _at_the_bound),
    ("power of two", ("f8",), _power_of_two),
    ("integers", "i", _integers_in),
    ("two", "b", _two),
    ("bytes", "u", _buffers),
    ("fields", "V", _fields),
)


def _array_families(dtype, ndim, accepted: tuple) -> dict:
    """
    The families of an array argument of `dtype` (None: any) and `ndim`
    (None: any), each named for it, and each yielding only for a value of
    that dtype and ndim, knowing which arrays its union takes as they are.
    """
    label = f"{'any' if dtype is None else dtype.name}[{'*' if ndim is None else ndim}]"
    codes = {"*"} if dtype is None else {dtype.kind, dtype.str[1:]}

    def guarded(make):
        def generate(value, context):
            matches = isinstance(value, np.ndarray) and (ndim is None or value.ndim == ndim)
            if matches and (dtype is None or value.dtype == dtype or dtype.kind == value.dtype.kind == "V"):
                yield from make(value, context._replace(accepted=accepted))

        return generate

    return {
        f"{label} {family}": guarded(make)
        for family, kinds, make in ARRAY_FAMILIES
        if codes & set(kinds) and not (family in ("fortran", "flipped") and ndim is not None and ndim < 2)
    }


def _retyped(value, wrong, context) -> Variant:
    """
    The same values as another dtype, or a list: taken as they are where the
    union takes that dtype, else converted with the warning to the first
    dtype it takes that numpy casts them to safely (answering as those
    values given so), else refused, naming the argument. A bare array of any
    dtype takes whatever it is given, to answer or refuse.
    """
    seen = np.asarray(wrong)
    named = re.escape(context.parameter)
    if any(dtype is None for dtype, _ in context.accepted):
        return Variant(wrong, None)
    taken = any(dtype == seen.dtype and ndim in (None, seen.ndim) for dtype, ndim in context.accepted)
    if taken and not isinstance(wrong, list):
        return Variant(wrong, True, seen.astype(value.dtype))
    for dtype, ndim in context.accepted:
        if ndim in (None, seen.ndim) and np.can_cast(seen.dtype, dtype, "safe"):
            reference = seen.astype(dtype)
            if reference.tobytes() == value.astype(dtype).tobytes() and dtype == value.dtype:
                reference = AS_GIVEN
            given = "a list" if isinstance(wrong, list) else seen.dtype.name
            said = f"{named}: {given} converted to {dtype.name} "
            return Variant(wrong, True, reference, warns=said)
    return Variant(wrong, False, refusal=(TypeError,), match=named)


def _integers(value, context):
    numbers = list(INTEGERS)
    if context.bounds is not None:
        low, high, *top = context.bounds
        numbers += [low - 1, low] + ([min([high - 1, *top]), high] if high != math.inf else [])
    for number in numbers:
        yield _bounded(number, context, number)
    # past int64: refused, never wrapped around
    yield Variant(2**63, False, refusal=(TypeError,))


def _numpy_integers(value, context):
    # no argument ever needs numpy.int64(10), or refuses it: what numpy's
    # own arithmetic returns answers exactly as Python's int; a float
    # never stands for one
    if not isinstance(value, int) or isinstance(value, bool):
        return
    for kind in (np.int64, np.int32, np.int16, np.uint8, np.uint64):
        if np.iinfo(kind).min <= value <= np.iinfo(kind).max:
            yield Variant(kind(value), True, value)
    for wrong in (float(value), np.float64(value)):
        yield Variant(wrong, False, refusal=(TypeError,))


def _floats(value, context):
    # the bounds, and the nearest float just past each: a bound compared in
    # float32, or written a little wide, takes the one past it
    edges = ()
    if context.bounds:
        low, high = context.bounds[:2]
        edges = (low, high, math.nextafter(low, -math.inf), math.nextafter(high, math.inf))
    for number in (*FLOATS, *edges):
        yield _bounded(number, context, number)


def _numpy_floats(value, context):
    # numpy's scalars answer exactly as Python's float, and an int as the
    # float it is
    if not isinstance(value, float):
        return
    yield Variant(np.float64(value), True, value)
    if float(np.float32(value)) == value:
        yield Variant(np.float32(value), True, value)
    if math.isfinite(value) and value == int(value):
        yield Variant(int(value), True, value)


def _bools(value, context):
    for truth in (True, False):
        yield Variant(truth, True)
        yield Variant(np.bool_(truth), True, truth)
    # nothing else is a bool
    for wrong in (0, 1, None, "yes", np.int64(1)):
        yield Variant(wrong, False, refusal=(TypeError,))


def _texts(value, context):
    # nothing; and a NUL or non-ASCII character ends the data where it is
    yield Variant("", True)
    for stop in ("\0", "é", "\U0001f600"):
        yield Variant(value + stop + " M 5 5 L 6 6", True)
    yield Variant(value.encode(), False, refusal=(TypeError,))


# the families of each annotation that isn't an array: a numpy scalar's are its Python twin's
_INTEGER_FAMILIES = {"integers": _integers, "numpy integers": _numpy_integers}
_FLOAT_FAMILIES = {"floats": _floats, "numpy floats": _numpy_floats}
SCALAR_FAMILIES = {
    # valid alone, though maybe not beside what else is given
    "None": {"None": lambda value, context: iter([Variant(None, None)])},
    "int": _INTEGER_FAMILIES,
    "numpy.integer": _INTEGER_FAMILIES,
    "float": _FLOAT_FAMILIES,
    "numpy.floating": _FLOAT_FAMILIES,
    "bool": {"bools": _bools},
    "numpy.bool_": {"bools": _bools},
    "str": {"text": _texts},
}


def _enum_families(members) -> dict:
    def every_member(value, context):
        # a member, or its int, which answers as the member does; nothing else is one
        for member in members:
            yield Variant(member, True)
            yield Variant(int(member), True, member)
        for wrong in (-1, len(members), "EARCUT"):
            yield Variant(wrong, False, refusal=(TypeError,))

    return {"members": every_member}


def _nested(prefix: str, index: int, inner: dict, wrap) -> dict:
    # the families of the value at `index` of a whole, each variant wrapped back into it
    def lifted(make):
        def generate(value, context):
            item = wrap(value)
            if item is None:
                return
            part, rebuild, bounds = item
            inside = context._replace(bounds=bounds, parameter=f"{context.parameter}[{index}]")
            for variant in make(part, inside):
                reference = variant.reference
                if reference is not AS_GIVEN and reference is not ORACLE:
                    reference = rebuild(reference)
                yield variant._replace(value=rebuild(variant.value), reference=reference)

        return generate

    return {prefix + name: lifted(make) for name, make in inner.items()}


def _sequence_families(item: ast.expr) -> dict:
    def whole(value, context):
        yield Variant(tuple(value), True, AS_GIVEN)
        yield Variant([], None)
        if len(value):
            yield Variant([*value, value[0]], None)

    def first(value):
        if not isinstance(value, (list, tuple)) or not len(value):
            return None
        return value[0], lambda new: [new, *value[1:]], None

    return {"sequence": whole, **_nested("item ", 0, families(item), first)}


def _mesh_families() -> dict:
    def pair(value, context):
        # any sequence of two: a list is as good as a tuple, a triple isn't
        yield Variant(list(value), True, AS_GIVEN)
        yield Variant((*value, value[1]), False, refusal=(TypeError,))

    def part(index):
        def wrap(value):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return None
            # faces index the mesh's own vertices
            bounds = None if index == 0 else (0, len(value[0]))
            return value[index], lambda new: (new, value[1]) if index == 0 else (value[0], new), bounds

        return wrap

    vertices = families("Annotated[NDArray[numpy.float64], dict(shape=(None, 3))]")
    faces = families("Annotated[NDArray[numpy.int64], dict(shape=(None, 3))]")
    return {
        "mesh": pair,
        **_nested("vertices ", 0, vertices, part(0)),
        **_nested("faces ", 1, faces, part(1)),
    }


def form_variants(name: str, form: dict):
    """
    Every variant of every argument of a small form of public `name`'s case,
    as (parameter, family, variant): each family first with None, so even
    one with no variant for this form is seen.
    """
    case = CASES[name]
    for parameter, annotation, default, _ in stub_parameters()[name]:
        bounds = case.ranges[parameter](form) if parameter in case.ranges else None
        context = ArgumentContext(parameter, (), bounds, case.holds)
        for family, generate in families(annotation).items():
            yield parameter, family, None
            for variant in generate(form.get(parameter, default), context):
                yield parameter, family, variant


# -- one checked call ----------------------------------------------------------


def pool_threads() -> set[str]:
    """
    The pool's threads alive now, by thread id: those named "trimkit"
    (platform.hpp, name_this_thread), as /proc lists them. Empty where
    there is no /proc to ask.
    """
    found = set()
    try:
        tasks = os.listdir("/proc/self/task")
    except OSError:
        return found
    for task in tasks:
        try:
            with open(f"/proc/self/task/{task}/comm") as comm:
                if comm.read().strip() == "trimkit":
                    found.add(task)
        except OSError:
            pass  # it exited as it was read
    return found


class PoolCensus:
    """
    The pool's threads, followed from call to call. One may start while a
    call runs, or just after it, finishing the chain of starts the call's
    wake began (Scheduler.tla, P3); but one starting after another exited,
    with no call since, started for nothing, as an idle exit's replacement
    once did. `between_calls` fails on that. An exit is seen by its thread's
    own /proc entry going, which no listing can miss.
    """

    def __init__(self):
        self.known = set()
        self.exited = False

    def between_calls(self) -> set[str]:
        now = pool_threads()
        self.exited = self.exited or not all(os.path.exists(f"/proc/self/task/{t}") for t in self.known)
        started = now - self.known
        assert not (self.exited and started), (
            f"{len(started)} pool threads started after one exited, no call since"
        )
        self.known |= now
        return now

    def returned(self) -> None:
        self.known = {t for t in self.known if os.path.exists(f"/proc/self/task/{t}")} | pool_threads()
        self.exited = False


def _frozen(value):
    # everything an argument is, to compare before and after a call
    if isinstance(value, np.ndarray):
        # a record's fields one by one: numpy's copy of a strided record
        # leaves the padding between them unset, so its bytes differ each time
        held = (
            tuple(_frozen(value[name]) for name in value.dtype.names)
            if value.dtype.names
            else value.tobytes()
        )
        return ("array", str(value.dtype), value.shape, value.strides, value.flags.writeable, held)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_frozen(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple((key, _frozen(v)) for key, v in value.items()))
    if isinstance(value, memoryview):
        return ("memoryview", value.tobytes())
    return (type(value).__name__, repr(value))


def input_arrays(value) -> list[np.ndarray]:
    if isinstance(value, dict):
        return input_arrays(list(value.values()))
    if isinstance(value, (list, tuple)):
        return [a for item in value for a in input_arrays(item)]
    if isinstance(value, (bytes, memoryview)):
        return [np.frombuffer(value, np.uint8)]
    return [value] if isinstance(value, np.ndarray) else []


def same_answers(got, expected) -> bool:
    """
    Two answers alike to the bit: every leaf the same dtype, shape and bytes.
    """
    got, expected = leaves(got), leaves(expected)
    return len(got) == len(expected) and all(
        type(a) is type(b)
        and (
            a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            if isinstance(a, np.ndarray)
            else a == b
        )
        for a, b in zip(got, expected, strict=True)
    )


def checked_call(
    census: PoolCensus,
    name: str,
    arguments: dict,
    *,
    variant: Variant | None = None,
    expected=None,
):
    """
    Call public `name` and check everything a call promises, returning its
    answer, or None when it refused:
    - it answers, or refuses with ValueError, TypeError or MemoryError
      (draco: NotImplementedError on a big-endian machine), as `variant`
      says it must (no variant: it answers);
    - an answer is `expected`'s to the bit when given, and passes the oracle
      otherwise;
    - it warns only what the variant says its argument converted from and
      to, or that a device scene answers on the CPU;
    - every input is exactly as it was, every output a fresh, writable,
      C-contiguous array sharing memory with no input and no other output;
    - no pool thread started between the last call `census` saw and this one.
    """
    census.between_calls()
    before = _frozen(arguments)
    refused = answer = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            answer = call(name, arguments)
        except BaseException as error:
            refused = error
    assert _frozen(arguments) == before, f"{name} changed its input"
    _check_warnings(caught, arguments, variant)
    if refused is not None:
        _check_refusal(name, refused, variant)
    else:
        assert variant is None or variant.answers is not False, f"{name} answered {variant}"
        _check_answer(name, arguments, answer, expected)
    census.returned()
    return answer


def _check_warnings(caught, arguments: dict, variant: Variant | None) -> None:
    # the conversion a variant makes, said once, and a device answering on the CPU
    messages = [str(w.message) for w in caught]
    assert all(issubclass(w.category, UserWarning) for w in caught), messages
    converted = [m for m in messages if " converted to " in m]
    device = [m for m in messages if m.startswith("use_gpu: ")]
    assert len(converted) + len(device) == len(messages), messages
    assert not device or arguments.get("use_gpu"), messages
    if variant is not None and variant.warns:
        assert len(converted) == 1 and re.match(variant.warns, converted[0]), messages
    else:
        assert not converted, messages


def _check_refusal(name: str, refused: BaseException, variant: Variant | None) -> None:
    if isinstance(refused, KeyboardInterrupt):
        raise refused
    big_endian = isinstance(refused, NotImplementedError) and name.startswith("exchange.draco_")
    contract = (ValueError, TypeError, MemoryError)
    if not isinstance(refused, contract) and not (big_endian and not LITTLE_ENDIAN):
        raise AssertionError(f"{name} raised outside the error contract") from refused
    if variant is None or variant.answers:
        raise AssertionError(f"{name} refused {variant}") from refused
    assert isinstance(refused, variant.refusal) or variant.answers is None, refused
    assert re.search(variant.match, str(refused)), refused
    assert not (variant.forbidden and re.search(variant.forbidden, str(refused))), refused


def _check_answer(name: str, arguments: dict, answer, expected) -> None:
    inputs = input_arrays(arguments)
    # an object answers what it holds, unaliased too
    outputs = [leaf for leaf in leaves(answer) if isinstance(leaf, np.ndarray)]
    for i, output in enumerate(outputs):
        assert output.flags.writeable and output.flags.c_contiguous, name
        assert not any(np.shares_memory(output, given) for given in inputs), f"{name} aliased an input"
        assert not any(np.shares_memory(output, other) for other in outputs[:i]), f"{name} aliased an output"
    if expected is not None:
        assert same_answers(answer, expected), f"{name} answered otherwise"
    else:
        CASES[name].oracle(arguments, answer)


# -- oracles: each answer against an independent reference ---------------------


def _meshes(arguments: dict) -> list:
    return [(np.asarray(v, np.float64), np.asarray(f, np.int64)) for v, f in arguments["meshes"]]


def brute_force(vertices, faces, origins, directions):
    """
    Möller-Trumbore for every ray against every triangle, from either side:
    the (r, m) parameter, in units of each ray's direction, of each triangle
    hit ahead of its origin, else inf; how far inside its triangle each ray
    crosses the plane in barycentric units (negative outside, 0 on an edge),
    NaN where it is parallel; and the parameter of that crossing.
    """
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    e1, e2 = b - a, c - a
    p = np.cross(directions[:, None], e2[None])
    det = np.einsum("rmk,mk->rm", p, e1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = origins[:, None] - a[None]
        u = np.einsum("rmk,rmk->rm", s, p) / det
        q = np.cross(s, e1[None])
        v = np.einsum("rk,rmk->rm", directions, q) / det
        t = np.einsum("rmk,mk->rm", q, e2) / det
        inside = np.minimum(np.minimum(u, v), 1 - u - v)
    hit = (det != 0) & (inside >= 0) & (t > 0)
    return np.where(hit, t, np.inf), np.where(det != 0, inside, np.nan), t


def _crossings(arguments: dict):
    """
    Every placed triangle each ray crosses, by brute force: rows of (ray,
    instance, face, parameter), sorted by ray then parameter, and which rays
    are too close to call: through an edge or vertex within 1e-9 of it,
    starting on the surface, or crossing twice closer than a scene counts
    as two (bvh.cpp, `crossings`: 2^-40 of the scene's extent).
    """
    meshes = _meshes(arguments)
    matrices, index = instances(meshes, arguments.get("instance_matrices"), arguments.get("instance_index"))
    origins, directions = (
        np.atleast_2d(np.asarray(arguments[k], np.float64)) for k in ("origins", "directions")
    )
    origins, directions = np.broadcast_arrays(origins, directions)
    rows, unclear, extent = [], np.zeros(len(origins), dtype=bool), 0.0
    for k, (g, m) in enumerate(zip(index, matrices, strict=True)):
        vertices, faces = meshes[g]
        if not len(faces):
            continue
        placed = transform(vertices, m)
        extent = max(extent, np.abs(placed[np.unique(faces)]).max())
        t, inside, plane = brute_force(placed, faces, origins, directions)
        unclear |= ((np.abs(inside) < 1e-9) & (plane > -1e-9)).any(axis=1)
        ray, face = np.nonzero(np.isfinite(t))
        rows.append(np.column_stack([ray, np.full(len(ray), k), face, t[ray, face]]))
    found = np.concatenate(rows) if rows else np.zeros((0, 4))
    found = found[np.lexsort((found[:, 3], found[:, 0]))]
    ray = found[:, 0].astype(np.int64)
    step = 4 * 2.0**-40 * np.maximum(extent / np.linalg.norm(directions[ray], axis=1), found[:, 3])
    # two crossings of a ray closer than that (a shared edge, a face twice), or one at its origin
    tied = (ray[1:] == ray[:-1]) & (found[1:, 3] - found[:-1, 3] <= step[:-1])
    unclear[ray[1:][tied]] = True
    unclear[ray[found[:, 3] <= step]] = True
    return found, unclear, directions


def _intersect_oracle(arguments: dict, answer) -> None:
    rays, hit_instances, hit_faces, distances = answer
    found, unclear, directions = _crossings(arguments)
    assert np.array_equal(np.lexsort((distances, rays)), np.arange(len(rays), dtype=np.int64))
    limit = arguments.get("maximum_hits")
    limit = math.inf if limit is None else limit
    parameters = distances / np.linalg.norm(directions[rays], axis=1)
    for ray in range(len(directions)):
        mine = rays == ray
        truth = found[found[:, 0] == ray]
        got = list(zip(hit_instances[mine].tolist(), hit_faces[mine].tolist(), strict=True))
        assert len(got) <= limit
        if unclear[ray]:
            continue
        wanted = truth[: int(min(limit, len(truth)))]
        assert got == [(int(i), int(f)) for i, f in wanted[:, 1:3]], (ray, got)
        assert np.allclose(parameters[mine], wanted[:, 3], rtol=1e-9, atol=0), ray


def _intersect_any_oracle(arguments: dict, answer) -> None:
    found, unclear, directions = _crossings(arguments)
    truth = np.zeros(len(directions), dtype=bool)
    truth[found[:, 0].astype(np.int64)] = True
    assert answer.dtype == bool and answer.shape == truth.shape
    assert np.array_equal(answer[~unclear], truth[~unclear])


def _scene_oracle(arguments: dict, answer) -> None:
    # a scene, or the boxes it gives for these instances
    matrices, index = arguments.get("instance_matrices"), arguments.get("instance_index")
    check_boxes(answer, _meshes(arguments), matrices, index)


def _corners(boxes) -> np.ndarray:
    # (n, 2, d) from either form of boxes
    given = np.asarray(boxes, np.float64)
    return given.reshape((len(given), 2, given.shape[1] // 2)) if given.ndim == 2 else given


def _queries(arguments: dict) -> tuple[np.ndarray, bool]:
    # every query as (m, 2, d), and whether it was one alone
    d = _corners(arguments["boxes"]).shape[2]
    given = np.asarray(arguments["queries"], np.float64)
    alone = given.ndim == 1 or given.shape == (2, d)
    return given.reshape((-1, 2, d)), alone


def overlapping(boxes, queries):
    """
    Brute force: [query, box] for every overlap, touching included, sorted.
    """
    lo, hi = boxes[:, 0], boxes[:, 1]
    rows = [
        [q, b]
        for q, query in enumerate(queries)
        for b in np.flatnonzero(((lo <= query[1]) & (hi >= query[0])).all(axis=1))
    ]
    return np.array(rows, dtype=np.int64).reshape((-1, 2))


def box_distance(lo, hi, query_lo, query_hi) -> float:
    """
    The contract's distance between a box and a query box, each operation
    rounded once, as IEEE arithmetic does: each axis's gap (the larger of
    the two differences, or 0), squared, summed in axis order, then the
    square root, correctly rounded. Computed exactly, by fractions and an
    integer square root, so it holds where numpy's own doesn't: on x87
    (Debian's i386), whose 80-bit registers round twice.
    """
    total = Fraction(0)
    for axis in range(len(lo)):
        below = float(Fraction(float(lo[axis])) - Fraction(float(query_hi[axis])))
        above = float(Fraction(float(query_lo[axis])) - Fraction(float(hi[axis])))
        gap = Fraction(max(below, above, 0.0))
        total = Fraction(float(total + Fraction(float(gap * gap))))
    if total == 0:
        return 0.0
    # sqrt(total) scaled near 2**300, floored: 2**-300 of it is far less than
    # the least distance from a square root of a double to a midpoint
    scale = max(0, 300 - (total.numerator.bit_length() - total.denominator.bit_length()) // 2)
    return float(Fraction(math.isqrt(total.numerator * 4**scale // total.denominator), 2**scale))


def same_box_distances(got, truth, boxes, queries, rows) -> bool:
    """
    Whether `got`, the distances of [query, box] `rows`, are numpy's brute
    force `truth` to the bit, or, wherever they differ, `box_distance`'s.
    """
    got, truth = np.asarray(got).ravel(), np.asarray(truth).ravel()
    rows = np.asarray(rows).reshape((-1, 2))
    if got.shape != truth.shape:
        return False
    differ = np.flatnonzero(got.view(np.uint64) != truth.view(np.uint64))
    return all(
        got[i] == box_distance(boxes[b, 0], boxes[b, 1], queries[q, 0], queries[q, 1])
        for i in differ
        for q, b in [rows[i]]
    )


def nearest_rows(boxes, queries, k):
    """
    Brute force: [query, box] rows of each query's k nearest boxes by
    distance and then box, and their distances, each axis's gap squared
    summed in axis order as the contract says.
    """
    lo, hi = boxes[:, 0], boxes[:, 1]
    rows, distances = [np.zeros((0, 2), np.int64)], [np.zeros(0)]
    for q, query in enumerate(queries):
        gaps = np.maximum(np.maximum(lo - query[1], query[0] - hi), 0.0)
        squared = gaps[:, 0] * gaps[:, 0]
        for axis in range(1, boxes.shape[2]):
            squared = squared + gaps[:, axis] * gaps[:, axis]
        distance = np.sqrt(squared)
        order = np.lexsort((np.arange(len(boxes), dtype=np.int64), distance))[:k]
        rows.append(np.column_stack([np.full(len(order), q, dtype=np.int64), order]))
        distances.append(distance[order])
    return np.concatenate(rows).astype(np.int64).reshape((-1, 2)), np.concatenate(distances)


def _boxes_oracle(arguments: dict, answer) -> None:
    # a Boxes, or the copy it gives of its boxes: the corners given, exactly
    corners = _corners(arguments["boxes"])
    if not isinstance(answer, np.ndarray):
        assert len(answer) == len(corners)
        answer = answer.boxes()
    assert answer.shape == corners.shape and answer.tobytes() == corners.tobytes()


def _bounds_oracle(arguments: dict, answer) -> None:
    corners = _corners(arguments["boxes"])
    low, high = corners[:, 0].min(axis=0, initial=np.inf), corners[:, 1].max(axis=0, initial=-np.inf)
    assert answer.shape == (2, corners.shape[2]) and answer.tobytes() == np.array([low, high]).tobytes()


def _intersection_oracle(arguments: dict, answer) -> None:
    queries, alone = _queries(arguments)
    truth = overlapping(_corners(arguments["boxes"]), queries)
    assert np.array_equal(answer, truth[:, 1] if alone else truth) and answer.dtype == np.int64


def _nearest_oracle(arguments: dict, answer) -> None:
    queries, alone = _queries(arguments)
    hits, distances = answer
    rows, truth = nearest_rows(_corners(arguments["boxes"]), queries, arguments.get("maximum_hits", 1))
    assert np.array_equal(hits, rows[:, 1] if alone else rows)
    assert same_box_distances(distances, truth, _corners(arguments["boxes"]), queries, rows)


def _held_oracle(field: str | None):
    # a CollisionResult holds exactly what it is given, None where nothing
    def oracle(arguments: dict, answer) -> None:
        for key in COLLISION_FIELDS if field is None else (field,):
            held = getattr(answer, key) if field is None else answer
            given = arguments.get(key)
            if given is None:
                assert held is None
                continue
            truth = np.asarray(given, np.int64 if key in ("pairs", "faces") else np.float64)
            assert held.shape == truth.shape and held.tobytes() == truth.tobytes()
        if field is None:
            assert len(answer) == len(np.asarray(arguments["pairs"]))

    return oracle


class PlacedInstance(NamedTuple):
    # one instance of a collision query, as brute force sees it
    vertices: np.ndarray  # placed: the mesh's vertices its faces use, carried by its transform
    convex: bool
    face_count: int
    box: tuple | None  # the vertices' box, None with no faces


def _placed_instances(arguments: dict) -> list[PlacedInstance]:
    meshes = _meshes(arguments)
    matrices, index = instances(meshes, arguments.get("instance_matrices"), arguments.get("instance_index"))
    convex = np.broadcast_to(np.asarray(arguments.get("convex", False), dtype=bool), (len(meshes),))
    placed = []
    for g, m in zip(index, matrices, strict=True):
        vertices, faces = meshes[g]
        used = transform(vertices[np.unique(faces)], m) if len(faces) else np.zeros((0, 3))
        box = (used.min(axis=0), used.max(axis=0)) if len(faces) else None
        placed.append(PlacedInstance(used, bool(convex[g]), len(faces), box))
    return placed


def _pair_bounds(first: PlacedInstance, second: PlacedInstance) -> tuple[float, float]:
    # the gap between two instances' boxes, and between their nearest two vertices
    if first.box is None or second.box is None:
        return math.inf, math.inf
    gaps = np.maximum(np.maximum(second.box[0] - first.box[1], first.box[0] - second.box[1]), 0)
    return np.linalg.norm(gaps), np.linalg.norm(first.vertices[:, None] - second.vertices[None], axis=2).min()


def _check_collision_row(
    answer, row: int, pair: tuple, limit: float, *, least: bool, tolerance: float
) -> None:
    """
    One measured row of a pair of PlacedInstances: its faces real or -1 for
    a solid, its distance within `limit` (the upper bound, or the margin), at
    least the boxes' gap and, when it is the `least`, at most the nearest
    vertices'; negative only into a solid, its witness points that far apart
    within their boxes and its normal along them.
    """
    distance, points, normal = answer.distances[row], answer.points[row], answer.normals[row]
    gap, nearest = _pair_bounds(*pair)
    for side, instance in enumerate(pair):
        face = answer.faces[row][side]
        assert face == -1 if (instance.convex or math.isinf(distance)) else 0 <= face < instance.face_count
    if math.isinf(distance):
        assert (distance > 0 and gap == math.inf) or nearest > limit - tolerance, (row, nearest)
        assert np.isnan(points).all() and np.isnan(normal).all()
        return
    assert distance <= limit + tolerance and (distance <= nearest + tolerance or not least), (row, distance)
    if distance < 0:
        assert pair[0].convex or pair[1].convex, "a negative distance with no solid"
        assert gap <= tolerance
    else:
        assert gap - tolerance <= distance, (row, gap, distance)
    for side, instance in enumerate(pair):
        low, high = instance.box
        assert (points[side] >= low - tolerance).all() and (points[side] <= high + tolerance).all()
    if distance > tolerance:
        assert abs(np.linalg.norm(points[1] - points[0]) - distance) <= tolerance
        assert np.allclose(normal, (points[1] - points[0]) / distance, atol=1e-6)
    if np.isfinite(normal).all():
        assert abs(np.linalg.norm(normal) - 1) <= 1e-6


def _collision_oracle(measures: bool):
    """
    A collision or distance query against what brute force can bound: a
    pair's distance is at least its boxes' gap and at most its nearest two
    vertices', its witness points that far apart within their boxes, and a
    pair a query reports is near enough by its boxes. Whether it is exact
    is test_collision's work, on shapes with exact answers.
    """

    def oracle(arguments: dict, answer) -> None:
        placed = _placed_instances(arguments)
        pairs = arguments.get("pairs")
        named = pairs is not None
        pairs = np.asarray(pairs, np.int64) if named else np.column_stack(np.triu_indices(len(placed), 1))
        tolerance = 1e-6 * (1.0 + max((np.abs(p.vertices).max(initial=0) for p in placed), default=0.0))
        found = answer.pairs.tolist()
        assert len(answer) == len(found)
        margin = arguments.get("margin", 0.0)
        if measures:
            assert np.array_equal(answer.pairs, pairs.reshape((-1, 2)))
        else:
            # rows of pairs in their order, as many as a pair has contacts
            measured = arguments.get("return_distance", False)
            contacts = arguments.get("maximum_contacts", 1) if measured else 1
            position, used = 0, 0
            for pair in map(tuple, found):
                while position < len(pairs) and (tuple(pairs[position]) != pair or used == contacts):
                    position, used = position + 1, 0
                assert position < len(pairs), f"{pair} is no pair asked, or out of order"
                assert _pair_bounds(placed[pair[0]], placed[pair[1]])[0] <= margin + tolerance, pair
                used += 1
            # and none missing: vertices lie on their surfaces, so a pair with
            # two of them within the margin is a pair within it
            reported = set(map(tuple, found))
            for a, b in pairs.reshape((-1, 2)).tolist():
                if _pair_bounds(placed[a], placed[b])[1] <= margin - tolerance:
                    assert (a, b) in reported, f"{(a, b)} is within the margin, and not reported"
            if not measured:
                assert all(getattr(answer, field) is None for field in COLLISION_FIELDS[1:])
                return
        limit = arguments.get("upper_bound", math.inf) if measures else margin
        for row, (a, b) in enumerate(found):
            _check_collision_row(
                answer, row, (placed[a], placed[b]), limit, least=measures, tolerance=tolerance
            )

    return oracle


def shoelace(ring: np.ndarray) -> float:
    x, y = np.asarray(ring, np.float64).T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _simple_polygon(rings: list) -> bool:
    """
    Whether rings are a valid polygon, by brute force: each a closed loop of
    at least three distinct vertices that no other edge crosses or touches,
    each hole inside the exterior and outside every other hole.
    """
    loops = []
    for given in rings:
        points = np.asarray(given, np.float64)
        keep = (
            np.r_[True, (points[1:] != points[:-1]).any(axis=1)] if len(points) else np.zeros(0, dtype=bool)
        )
        ring = points[keep]
        if len(ring) > 1 and (ring[0] == ring[-1]).all():
            ring = ring[:-1]
        if len(np.unique(ring, axis=0)) != len(ring) or len(ring) < 3:
            return False
        loops.append(ring)
    starts = np.concatenate(loops)
    ends = np.concatenate([np.roll(loop, -1, axis=0) for loop in loops])
    owner = np.concatenate([np.full(len(loop), i, dtype=np.int64) for i, loop in enumerate(loops)])
    position = np.concatenate([np.arange(len(loop), dtype=np.int64) for loop in loops])
    sizes = np.array([len(loop) for loop in loops], dtype=np.int64)[owner]

    def cross(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (
            b[..., 0] - o[..., 0]
        )

    p, q, r, s = starts[:, None], ends[:, None], starts[None], ends[None]
    meet = (cross(p, q, r) * cross(p, q, s) <= 0) & (cross(r, s, p) * cross(r, s, q) <= 0)
    meet &= (np.minimum(p, q) <= np.maximum(r, s)).all(axis=2) & (np.minimum(r, s) <= np.maximum(p, q)).all(
        axis=2
    )
    gap = (position[:, None] - position[None]) % sizes[:, None]
    adjacent = (owner[:, None] == owner[None]) & ((gap == 0) | (gap == 1) | (gap == sizes[:, None] - 1))
    if (meet & ~adjacent).any():
        return False

    def inside(point, loop):
        x, y = point
        a, b = loop, np.roll(loop, -1, axis=0)
        straddles = (a[:, 1] > y) != (b[:, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
        return bool((straddles & (x < at)).sum() % 2)

    holes = loops[1:]
    return all(inside(h[0], loops[0]) for h in holes) and not any(
        inside(h[0], other) for i, h in enumerate(holes) for j, other in enumerate(holes) if i != j
    )


def _triangulate_oracle(arguments: dict, answer) -> None:
    import trimkit

    vertices, faces = answer
    rings = [arguments["exterior"], *arguments.get("interiors", ())]
    assert (
        vertices.tobytes() == np.vstack([np.asarray(r, np.float64).reshape((-1, 2)) for r in rings]).tobytes()
    )
    assert faces.dtype == np.int64 and faces.shape[1:] == (3,)
    assert faces.min(initial=0) >= 0 and faces.max(initial=-1) < len(vertices)
    # of exactly equal vertices, only the first is used
    _, first, inverse = np.unique(vertices, axis=0, return_index=True, return_inverse=True)
    assert np.array_equal(first[inverse.ravel()][faces], faces)
    areas = areas2d(vertices, faces)
    assert (areas >= 0).all(), "a triangle wound clockwise"
    engine = arguments.get("engine")
    exact = engine == trimkit.TriangulateEngine.CDT or len(rings) == 1
    if exact and _simple_polygon(rings):
        truth = abs(shoelace(rings[0])) - sum(abs(shoelace(r)) for r in rings[1:])
        assert abs(areas.sum() - truth) <= 1e-9 * abs(truth), (areas.sum(), truth)


def check_hulls(mesh, hulls, closed: bool = True) -> None:
    """
    A list of (vertices, faces) tuples, each a closed convex surface wound
    outward, together covering the volume of `mesh` when it is `closed`.
    """
    assert type(hulls) is list
    total = 0.0
    for hull in hulls:
        assert type(hull) is tuple
        v, f = hull
        assert f.min() >= 0 and f.max() < len(v)
        assert is_closed(f)
        # convex and outward: every vertex is behind every face plane
        normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        distance = np.einsum("fj,vj->fv", normals, v) - np.einsum("fj,fj->f", normals, v[f[:, 0]])[:, None]
        assert distance.max() < 1e-9 * np.abs(normals).max()
        total += volume(v, f)
    # together they cover the mesh: at least its volume (more where hulls overlap)
    if closed:
        assert total >= volume(*mesh) * (1 - 1e-12)


def _decompose_oracle(arguments: dict, answer) -> None:
    vertices, faces = np.asarray(arguments["vertices"], np.float64), np.asarray(arguments["faces"], np.int64)
    # covering it, unless the cap on a hull's vertices cut one down
    most = arguments.get("maximum_vertices", 64)
    capped = any(len(v) >= most for v, _ in answer)
    closed = len(faces) > 0 and is_closed(faces) and volume(vertices, faces) > 0 and not capped
    check_hulls((vertices, faces), answer, closed)
    assert len(answer) <= arguments.get("maximum_hulls", 64)
    if not len(faces):
        assert answer == []
    for hull_vertices, _ in answer:
        assert len(hull_vertices) <= most
        # within the mesh's box, give or take the voxels V-HACD samples it with
        used = vertices[np.unique(faces)]
        extent = np.ptp(used, axis=0).max()
        slack = 2 * extent * arguments.get("resolution", 400_000) ** (-1 / 3)
        assert (hull_vertices >= used.min(axis=0) - slack).all() and (
            hull_vertices <= used.max(axis=0) + slack
        ).all()


def _parametrize_oracle(arguments: dict, answer) -> None:
    vertices, faces = np.asarray(arguments["vertices"], np.float64), np.asarray(arguments["faces"], np.int64)
    vertex_sources, new_faces, uv = answer
    # seams duplicate vertices a face uses, but every triangle is exactly where it was
    assert len(new_faces) == len(faces) and len(vertex_sources) == len(uv)
    assert np.isin(vertex_sources, faces).all()
    assert vertex_sources.min(initial=0) >= 0 and vertex_sources.max(initial=-1) < len(vertices)
    assert np.array_equal(vertices[vertex_sources][new_faces], vertices[faces])
    assert uv.min(initial=0) >= 0.0 and uv.max(initial=1) <= 1.0


def _simplify_oracle(arguments: dict, answer) -> None:
    vertices, faces = np.asarray(arguments["vertices"], np.float64), np.asarray(arguments["faces"], np.int64)
    kept, new_faces, vertex_sources = answer
    # a subset of the input vertices, unmoved and each used, and no more faces
    assert len(np.unique(vertex_sources)) == len(vertex_sources) and len(new_faces) <= len(faces)
    assert kept.tobytes() == vertices[vertex_sources].tobytes()
    assert np.array_equal(np.unique(new_faces), np.arange(len(vertex_sources), dtype=np.int64))
    if arguments.get("lock_border"):
        edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2)), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        assert set(unique[counts == 1].ravel().tolist()) <= set(vertex_sources.tolist())


def crossings(voxels, weights=None, threshold: float = 0.5):
    """
    Where a surface of voxels must have its vertices, by brute force: one
    between each inside voxel (weighing more than `threshold`) and each
    voxel beside it that isn't, where their weights interpolate to the
    threshold (space no voxel covers weighs 0), sorted, and the input row
    holding each inside voxel. `voxels` are (n, 3) or (n, 4) rows, or a
    bool or float64 volume.
    """
    voxels = np.asarray(voxels)
    if voxels.ndim == 3:
        values = voxels.astype(np.float64).ravel()
        rows = np.argwhere(np.ones(voxels.shape, dtype=bool)).astype(np.int64)
        keep = values != 0
        rows, values, sources = rows[keep], values[keep], np.flatnonzero(keep)
    else:
        lengths = voxels[:, 3] if voxels.shape[1] == 4 else np.ones(len(voxels), dtype=np.int64)
        assert lengths.sum(initial=0) < 4_000_000, "a surface too large to check"
        # numpy takes a count only as its own index type, 32 bits on a 32-bit machine
        lengths = lengths.astype(np.intp)
        sources = np.repeat(np.arange(len(voxels), dtype=np.int64), lengths)
        rows = voxels[sources, :3].copy()
        rows[:, 2] += np.arange(len(rows), dtype=np.int64) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        values = np.ones(len(rows)) if weights is None else np.asarray(weights, np.float64)[sources]
    weight = {tuple(r): w for r, w in zip(rows.tolist(), values.tolist(), strict=True)}
    points, origins = [], []
    for row, w, source in zip(rows.tolist(), values.tolist(), sources.tolist(), strict=True):
        if w <= threshold:
            continue
        for axis in range(3):
            for step in (-1, 1):
                beside = list(row)
                beside[axis] += step
                outside = weight.get(tuple(beside), 0.0)
                if outside > threshold:
                    continue
                point = [float(c) for c in row]
                point[axis] += step * (threshold - w) / (outside - w)
                points.append(point)
                origins.append(source)
    points = np.array(points, np.float64).reshape((-1, 3))
    # sorted as rounded, so rounding never reorders two answers to compare
    order = np.lexsort(points.round(9).T)
    return points[order], np.array(origins, np.int64)[order]


def _contour_oracle(arguments: dict, answer) -> None:
    vertices, faces, vertex_sources = answer
    threshold = arguments.get("threshold", 0.5)
    assert threshold >= 0, "answered with empty space inside"
    points, sources = crossings(arguments["voxels"], arguments.get("weights"), threshold)
    order = np.lexsort(vertices.round(9).T)
    assert len(vertices) == len(points) and is_closed(faces)
    assert faces.min(initial=0) >= 0 and faces.max(initial=-1) < len(vertices)
    assert np.allclose(vertices[order], points, rtol=1e-12, atol=1e-9)
    # and each vertex's source is the input element holding its inside voxel
    assert sorted(
        zip(map(tuple, vertices[order].round(9).tolist()), vertex_sources[order].tolist(), strict=True)
    ) == sorted(zip(map(tuple, points.round(9).tolist()), sources.tolist(), strict=True))


def _draco_matches(faces, decoded, source: dict) -> None:
    # decoded within what quantizing the source can move it, the rest exact
    assert np.array_equal(faces, source["faces"])
    defaults = {0: 14, 1: 10, 2: 12, 3: 10, 4: 0}
    for out, given, kind in zip(decoded, source["attributes"], source["kinds"], strict=True):
        value = np.asarray(given)
        assert out.dtype == value.dtype and out.shape == value.shape
        used = source.get("quantization_bits")
        used = defaults[int(kind)] if used is None else used
        if value.dtype.kind != "f" or used == 0:
            assert np.array_equal(out, value)
        else:
            extent = np.ptp(value.astype(np.float64), axis=0).max(initial=0)
            step = extent * 2.0 ** -(used + 1)
            assert np.abs(out.astype(np.float64) - value).max(initial=0) <= step + np.finfo(
                value.dtype
            ).eps * (1 + np.abs(value).max(initial=0))


def _draco_encode_oracle(arguments: dict, answer) -> None:
    import trimkit

    every = np.arange(len(arguments["attributes"]), dtype=np.int64)
    _draco_matches(*trimkit.exchange.draco_decode(answer, every), arguments)


def _draco_decode_oracle(arguments: dict, answer) -> None:
    import trimkit

    faces, decoded = answer
    wanted = np.asarray(arguments["draco_attributes"], np.int64)
    assert len(decoded) == len(wanted) and isinstance(decoded, tuple)
    for values in decoded:
        assert faces.min(initial=0) >= 0 and faces.max(initial=-1) < len(values)
    # the small form's buffer, whole, decodes to what it was encoded from
    source = _draco_forms()[1]
    if bytes(arguments["buffer"]) == trimkit.exchange.draco_encode(**source):
        chosen = [source["attributes"][i] for i in wanted]
        _draco_matches(faces, decoded, {**source, "attributes": chosen, "kinds": source["kinds"][wanted]})


def _svg_parse_oracle(arguments: dict, answer) -> None:
    import svg_reference as reference

    # the longest start of the path the reference grammar reads, as it reads it
    path = arguments["path"]
    starts = (reference.parse(reference.LENIENT_PARSER, path[:end]) for end in range(len(path), -1, -1))
    commands = next(read for read in starts if read is not None)
    check_svg_segments(svg_rows(*answer), reference.evaluate(commands), reference)


def _svg_string_oracle(arguments: dict, answer) -> None:
    import svg_reference as reference

    # path data the reference reads, drawing every segment it was given, in
    # order, each from its start to its end
    assert isinstance(answer, str)
    commands = reference.parse(reference.LENIENT_PARSER, answer)
    assert commands is not None
    drawn = reference.evaluate(commands)
    vertices = np.asarray(arguments["vertices"], np.float64).reshape((-1, 2))
    segments = arguments["segments"]
    if not len(segments):
        assert not drawn
        return
    tolerance = 10.0 ** (1 - arguments.get("digits", 8)) * (1 + np.abs(vertices).max(initial=0))
    # except where the digits written round a segment's ends together: an arc
    # whose ends are one point is drawn as nothing, as SVG says
    starts, ends = vertices[segments["points"][:, 0]], vertices[segments["points"][:, 3]]
    if np.abs(ends - starts).max(axis=1).min() <= 2 * tolerance:
        return
    assert len(drawn) == len(segments), (len(drawn), len(segments))
    for column, key in ((0, "start"), (3, "end")):
        written = np.array([segment[key] for segment in drawn]).reshape((-1, 2))
        given = vertices[segments["points"][:, column]]
        assert np.abs(written - given).max() <= tolerance, key


# -- the registry ---------------------------------------------------------------
#
# Every input is built inside the call that uses it, so no scene outlives a
# test (nanobind reports those as leaks), and nothing big stays cached: a
# 32-bit process ran out of memory when the slow inputs were. A builder gives
# the arguments of every call of its class, and each call takes its own
# (PublicCallCase.arguments).


def _in_a_row(offsets, rotate: bool = False) -> dict:
    # an instance of the first mesh per offset along x, every other one
    # turned 30 degrees about z
    matrices = np.tile(np.eye(4), (len(offsets), 1, 1))
    matrices[:, 0, 3] = offsets
    if rotate:
        turn = math.radians(30)
        matrices[1::2, :2, :2] = [[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]]
    return {"instance_matrices": matrices, "instance_index": np.zeros(len(offsets), np.int64)}


def _small_meshes() -> list:
    # a sphere and a box, apart
    return [sphere(6, 12), box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))]


def _no_mesh():
    return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)


def _big_mesh():
    return sphere(100, 200)


def _ray_forms() -> list[dict]:
    # rays at three instances, one stretched, as each dtype and shape the
    # signature takes: many from many, one alone, or one shared by many
    rng = np.random.default_rng(0)
    placed = _in_a_row([0.0, 2.5, 5.0], rotate=True) | {"instance_index": np.array([0, 1, 0], dtype=np.int64)}
    placed["instance_matrices"][2, :3, :3] *= 1.5
    origins = rng.uniform(-3, 8, (40, 3))
    aimed = placed["instance_matrices"][rng.integers(0, 3, 40), :3, 3]
    directions = aimed + rng.normal(size=(40, 3)) * 0.6 - origins
    one, down = np.array([0.1, 0.2, 5.0]), np.array([0.05, -0.02, -1.0])
    meshes = _small_meshes()
    return [
        {"meshes": meshes, "origins": origins, "directions": directions, **placed, "maximum_hits": 2},
        {"meshes": meshes, "origins": one, "directions": down},
        {"meshes": meshes, "origins": one.astype(np.float32), "directions": directions.astype(np.float32)},
        {"meshes": meshes, "origins": origins.astype(np.float32), "directions": down.astype(np.float32)},
    ]


def _no_rays(meshes: list) -> dict:
    return {"meshes": meshes, "origins": np.zeros((0, 3)), "directions": np.zeros((0, 3)), **_in_a_row([])}


def _no_mesh_rays() -> dict:
    return _no_rays([_no_mesh()])


def _big_rays() -> dict:
    origins, directions = np.random.default_rng(0).normal(size=(2, 20_000, 3))
    placed = _in_a_row(np.arange(5_000, dtype=np.int64) * 3.0)
    return {"meshes": [_big_mesh()], "origins": origins, "directions": directions, **placed}


def _slow_rays(instances: int) -> dict:
    origins, directions = np.random.default_rng(0).normal(size=(2, 2_000_000, 3))
    placed = _in_a_row(np.arange(instances, dtype=np.int64) * 3.0) if instances else {}
    return {"meshes": [sphere(60, 120)], "origins": origins, "directions": directions, **placed}


def _million_instances() -> dict:
    # a million instances: their boxes build into a Boxes as big as the slow boxes
    many = np.tile(np.eye(4), (1_000_000, 1, 1))
    many[:, :3, 3] = np.random.default_rng(0).uniform(-100, 100, (1_000_000, 3))
    return {"meshes": [box()], "instance_matrices": many, "instance_index": np.zeros(len(many), np.int64)}


def _box_forms() -> list[dict]:
    # boxes and queries in 3D and 2D, as corners and flat, in each shape taken
    rng = np.random.default_rng(1)
    corners = rng.uniform(-1, 1, (40, 3))
    three = np.stack([corners, corners + 0.4], axis=1)
    two = np.ascontiguousarray(three[:, :, :2])
    asked = three[:10] + 0.1
    return [
        {"boxes": three, "queries": asked, "maximum_hits": 3},
        {"boxes": two.reshape((40, 4)), "queries": two[3].ravel() + 0.05},
        {"boxes": three.reshape((40, 6)), "queries": asked.reshape((10, 6))},
        {"boxes": two, "queries": two[5] + 0.05},
    ]


def _boxes_of_themselves(low: np.ndarray, size: float) -> dict:
    # boxes asked for those each of them meets, and its nearest four
    boxes = np.stack([low, low + size], axis=1)
    return {"boxes": boxes, "queries": boxes, "maximum_hits": 4}


def _collision_forms() -> list[dict]:
    # a sphere and a box placed four times along x, near, apart and
    # overlapping, the box solid in the first form, with named pairs
    meshes = _small_meshes()
    first = {
        "meshes": meshes,
        "convex": np.array([False, True]),
        **_in_a_row([0.0, 1.6, 3.0, 4.4], rotate=True),
        "instance_index": np.array([0, 1, 0, 1], dtype=np.int64),
        "pairs": np.array([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]], dtype=np.int64),
        "margin": 0.2,
        "return_distance": True,
        "maximum_contacts": 2,
        "upper_bound": 1.0,
    }
    return [first, {"meshes": meshes}]


def _big_collision() -> dict:
    neighbors = np.arange(4_999, dtype=np.int64)
    return {
        "meshes": [_big_mesh()],
        "convex": np.array([True]),
        **_in_a_row(np.arange(5_000, dtype=np.int64) * 3.0),
        "pairs": np.column_stack([neighbors, neighbors + 1]),
        "return_distance": True,
    }


def _near_spheres(pairs: int) -> dict:
    # two big spheres a hair apart, the one pair tested many times: every test
    # descends both hierarchies to the leaves nearest the gap
    return {
        "meshes": [sphere(200, 400)],
        **_in_a_row([0.0, 2.0001]),
        "pairs": np.tile(np.array([[0, 1]], dtype=np.int64), (pairs, 1)),
        "return_distance": True,
        "maximum_contacts": 8,
    }


def _result_forms() -> list[dict]:
    return [
        {
            "pairs": np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64),
            "faces": np.array([[3, -1], [-1, -1], [0, 5]], dtype=np.int64),
            "distances": np.array([-0.25, np.inf, 0.5]),
            "points": np.array([[[1.0, 0, 0], [0.75, 0, 0]], [[np.nan] * 3] * 2, [[0, 0, 0], [0, 0.5, 0]]]),
            "normals": np.array([[-1.0, 0, 0], [np.nan] * 3, [0, 1.0, 0]]),
        },
        {"pairs": np.array([[0, 1]], dtype=np.int64)},
    ]


def _big_result() -> dict:
    # what a big query found, as a pickle gives it back
    name = "CollisionScene.distances"
    return call(name, CASES[name].arguments("big")).__reduce__()[0].keywords


def _mesh(mesh, **options) -> dict:
    vertices, faces = mesh
    return {"vertices": vertices, "faces": faces, **options}


def _among_many(mesh, **options) -> dict:
    # a small mesh among many vertices it doesn't use: decompose and
    # parametrize take seconds on a big one
    vertices, faces = mesh
    return {"vertices": np.vstack([vertices, _big_mesh()[0]]), "faces": faces, **options}


def _triangulate_forms() -> list[dict]:
    # a circle closed as shapely closes it, with a square hole wound the other way
    ring = circle(24) * 2.0
    exterior = np.vstack([ring, ring[:1]])
    hole = np.array([[-0.3, -0.3], [-0.3, 0.3], [0.3, 0.3], [0.3, -0.3]])
    return [{"exterior": exterior, "interiors": [hole]}, {"exterior": exterior}]


def _contour_forms() -> list[dict]:
    rng = np.random.default_rng(2)
    runs = ball_runs(2)
    return [
        {"voxels": runs, "weights": np.linspace(0.6, 1.0, len(runs)), "threshold": 0.5},
        {"voxels": rng.random((4, 4, 4)) < 0.5},
        {"voxels": rng.choice([0.0, 0.2, 0.7, 1.5], size=(4, 4, 4)), "threshold": 0.3},
    ]


def _draco_forms() -> list[dict]:
    vertices, faces = sphere(6, 12)
    position = vertices.astype(np.float32)
    generic = np.arange(len(vertices), dtype=np.int16)
    return [
        {
            "faces": faces,
            "attributes": [position],
            "kinds": np.array([0], dtype=np.int64),
            "quantization_bits": 12,
        },
        {"faces": faces, "attributes": [position, generic], "kinds": np.array([0, 4], dtype=np.int64)},
    ]


def _draco_primitive(count: int) -> dict:
    position = [np.random.default_rng(0).random((count, 3)).astype(np.float32)]
    faces = np.arange(3 * (count // 3), dtype=np.int64).reshape((-1, 3))
    return {"faces": faces, "attributes": position, "kinds": np.array([0], dtype=np.int64)}


def _draco_buffer(source: dict, wanted: list) -> dict:
    # what decodes the attributes `wanted` of a primitive, encoded
    buffer = call("exchange.draco_encode", source)
    return {"buffer": np.frombuffer(buffer, np.uint8), "draco_attributes": np.array(wanted, dtype=np.int64)}


SVG_PATH = "M0 0 c1 2 3 4 5 6 a3 2 30 1 0 4 4 l1 -1 a2 2 0 0 1 3 0 q1 2 3 4 Z m 2 2 h 3 v 1 z"


def _long_path(curves: int) -> str:
    return "M0 0 " + "c1 2 3 4 5 6 a 3 2 30 1 0 4 4 " * curves


def _parsed(path: str, **options) -> dict:
    vertices, segments = call("exchange.svg_parse", {"path": path})
    return {"vertices": vertices, "segments": segments, **options}


RAY_RANGES = {"instance_index": lambda a: (0, len(a["meshes"])), "maximum_hits": lambda a: (1, math.inf)}
COLLISION_RANGES = {
    "instance_index": lambda a: (0, len(a["meshes"])),
    # as many instances as it places: by its index, its matrices, or each mesh once
    "pairs": lambda a: (0, len(a.get("instance_index", a.get("instance_matrices", a["meshes"])))),
    "maximum_contacts": lambda a: (1, math.inf),
    "margin": lambda a: (0.0, 1e30),
    "upper_bound": lambda a: (5e-324, math.inf),
}
FACES_RANGE = {"faces": lambda a: (0, len(a["vertices"]))}
NO_PRIMITIVE = "draco has no empty primitive: it refuses one without vertices"
HOLDS_ONLY = "a CollisionResult holds arrays a query computed: it computes nothing itself, GIL or not"
BOUNDS_KEPT = "the box around every box is kept from the build: a few values, no work to release the GIL for"


def _million_boxes() -> dict:
    return _boxes_of_themselves(np.random.default_rng(0).uniform(-100, 100, (1_000_000, 3)), 1.0)


CASES: dict[str, PublicCallCase] = {
    case.name: case
    for case in (
        *(
            PublicCallCase(name, oracle, _ray_forms, empty, _big_rays, slow, RAY_RANGES)
            for name, oracle, empty, slow in (
                ("RayScene", _scene_oracle, lambda: _no_rays([]), lambda: {"meshes": [sphere(500, 1000)]}),
                ("RayScene.boxes", _scene_oracle, lambda: _no_rays([]), _million_instances),
                ("RayScene.intersect", _intersect_oracle, _no_mesh_rays, lambda: _slow_rays(8)),
                ("RayScene.intersect_any", _intersect_any_oracle, _no_mesh_rays, lambda: _slow_rays(0)),
            )
        ),
        *(
            PublicCallCase(
                name,
                oracle,
                _box_forms,
                lambda: _boxes_of_themselves(np.zeros((0, 3)), 0.0),
                lambda: _boxes_of_themselves(_big_mesh()[0], 1e-3),
                slow,
                {"maximum_hits": lambda a: (1, math.inf)},
            )
            for name, oracle, slow in (
                ("Boxes", _boxes_oracle, _million_boxes),
                ("Boxes.bounds", _bounds_oracle, BOUNDS_KEPT),
                ("Boxes.boxes", _boxes_oracle, _million_boxes),
                ("Boxes.intersection", _intersection_oracle, _million_boxes),
                ("Boxes.nearest", _nearest_oracle, _million_boxes),
            )
        ),
        *(
            PublicCallCase(
                f"CollisionResult.{field}" if field else "CollisionResult",
                _held_oracle(field),
                _result_forms,
                lambda: {"pairs": np.zeros((0, 2), dtype=np.int64)},
                _big_result,
                HOLDS_ONLY,
                holds=True,
            )
            for field in (None, *COLLISION_FIELDS)
        ),
        *(
            PublicCallCase(name, oracle, _collision_forms, empty, _big_collision, slow, COLLISION_RANGES)
            for name, oracle, empty, slow in (
                ("CollisionScene", _scene_oracle, lambda: {"meshes": []}, lambda: _near_spheres(0)),
                ("CollisionScene.boxes", _scene_oracle, lambda: {"meshes": []}, _million_instances),
                (
                    "CollisionScene.collides",
                    _collision_oracle(False),
                    lambda: {"meshes": [_no_mesh()] * 2},
                    lambda: _near_spheres(2_000),
                ),
                (
                    "CollisionScene.distances",
                    _collision_oracle(True),
                    lambda: {"meshes": []},
                    lambda: _near_spheres(200),
                ),
            )
        ),
        PublicCallCase(
            "triangulate",
            _triangulate_oracle,
            _triangulate_forms,
            lambda: {"exterior": np.zeros((0, 2))},
            lambda: {"exterior": circle(40_000)},
            lambda: {"exterior": circle(2_000_000)},
        ),
        PublicCallCase(
            "decompose",
            _decompose_oracle,
            lambda: [
                _mesh(box(), resolution=20_000, maximum_hulls=4, maximum_vertices=16, maximum_error=2.0)
            ],
            lambda: _mesh(_no_mesh()),
            lambda: _among_many(box(), resolution=20_000),
            lambda: _mesh(sphere(60, 120), resolution=50_000),
            {
                **FACES_RANGE,
                "maximum_hulls": lambda a: (1, 2**32),
                "maximum_vertices": lambda a: (4, 2**32),
                # its top, 64,000,000 voxels, took 4 GB and 17 s on a unit box: tried up to 100,000
                "resolution": lambda a: (1, 64_000_001, 100_000),
                "maximum_error": lambda a: (0.0, 100.0),
            },
        ),
        PublicCallCase(
            "parametrize",
            _parametrize_oracle,
            lambda: [_mesh(sphere(6, 12))],
            lambda: _mesh(_no_mesh()),
            lambda: _among_many(sphere(6, 12)),
            lambda: _mesh(sphere(60, 120)),
            FACES_RANGE,
        ),
        PublicCallCase(
            "simplify",
            _simplify_oracle,
            lambda: [_mesh(sphere(6, 12), face_count=40, maximum_error=0.5, lock_border=True)],
            lambda: _mesh(_no_mesh(), face_count=0),
            lambda: _mesh(_big_mesh(), face_count=4),
            lambda: _mesh(sphere(400, 800), face_count=1000),
            {
                **FACES_RANGE,
                "face_count": lambda a: (0, math.inf),
                "maximum_error": lambda a: (0.0, math.inf),
            },
        ),
        PublicCallCase(
            "contour",
            _contour_oracle,
            _contour_forms,
            lambda: {"voxels": np.zeros((0, 4), dtype=np.int64)},
            lambda: {"voxels": ball_runs(60), "weights": np.full(len(ball_runs(60)), 0.75)},
            # a ball of radius 700 peaked at 1.15 GB (704 MB of it the surface), which
            # late in the suite was more than the 2 GB limit had left: a MemoryError
            lambda: {"voxels": ball_runs(550)},
            {"threshold": lambda a: (0.0, 1e30)},
        ),
        PublicCallCase(
            "exchange.draco_encode",
            _draco_encode_oracle,
            _draco_forms,
            NO_PRIMITIVE,
            lambda: _draco_primitive(20_000),
            lambda: _draco_primitive(2_000_000),
            {
                "faces": lambda a: (0, len(a["attributes"][0])),
                "kinds": lambda a: (0, 5),
                "quantization_bits": lambda a: (0, 25),
            },
        ),
        PublicCallCase(
            "exchange.draco_decode",
            _draco_decode_oracle,
            lambda: [_draco_buffer(_draco_forms()[1], [1, 0])],
            NO_PRIMITIVE,
            lambda: _draco_buffer(_draco_primitive(200_000), [0]),
            lambda: _draco_buffer(_draco_primitive(2_000_000), [0]),
            {"draco_attributes": lambda a: (0, 2)},
        ),
        PublicCallCase(
            "exchange.svg_parse",
            _svg_parse_oracle,
            lambda: [{"path": SVG_PATH}],
            lambda: {"path": ""},
            lambda: {"path": "M0 0 L1 1 L2 0" * 20_000},
            lambda: {"path": _long_path(1_000_000)},
        ),
        PublicCallCase(
            "exchange.svg_string",
            _svg_string_oracle,
            lambda: [_parsed(SVG_PATH, digits=12), _parsed(SVG_PATH)],
            lambda: _parsed(""),
            lambda: _parsed("M0 0 L1 1 L2 0" * 20_000),
            lambda: _parsed(_long_path(1_000_000)),
            {"digits": lambda a: (1, 18)},
        ),
    )
}


# Bugs in src/ the sweep found that aren't fixed yet, as "name: form N,
# parameter family", and what is wrong: a variant of that family fails a check
# there, and the sweep fails once none does, so a fix deletes its line
SRC_BUGS: dict[str, str] = {}


def volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Signed volume by the divergence theorem, positive when wound outward.
    """
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def is_closed(faces: np.ndarray) -> bool:
    """
    Every directed edge appears exactly once and its reverse exactly once.
    """
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    unique = np.unique(edges, axis=0)
    flipped = np.unique(edges[:, ::-1], axis=0)
    return len(unique) == len(edges) and np.array_equal(unique, flipped)


def forked_answer(query, expected, seconds: float = 30.0) -> bytes:
    """
    b"1" if a forked child's `query` gives exactly `expected`, b"0" if not,
    b"" if it raises or hangs. One that hangs gets SIGABRT first, so its
    faulthandler (conftest.py) prints every thread's stack to the stderr the
    test shows, then SIGKILL. Python warns about forking with threads alive,
    which is the point.
    """
    import select

    read, write = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        try:
            os.write(write, b"1" if same_answers(query(), expected) else b"0")
        finally:
            os._exit(0)
    os.close(write)
    try:
        ready, _, _ = select.select([read], [], [], seconds)
        if not ready:
            os.kill(pid, signal.SIGABRT)
            time.sleep(2.0)  # for its faulthandler to write
            os.kill(pid, signal.SIGKILL)
        return os.read(read, 1) if ready else b""
    finally:
        os.close(read)
        os.waitpid(pid, 0)


def longest_gil_stall(call) -> float:
    """
    The longest a background Python thread was held off during `call`, as a
    fraction of the call: about 1 when the call holds the GIL throughout,
    however short the call, as a held GIL stops the thread outright for as
    long as it is held, and small when it releases it around its work.
    """
    # the call's start and end, then the thread's last stamp and its longest gap
    # within them, kept as it goes: a list of every stamp grew by gigabytes
    # over a call of a few seconds, which is what a 2 GiB limit killed
    window = [math.inf, math.inf]
    last = [time.perf_counter()]
    longest = [0.0]
    stop = threading.Event()

    def stamp():
        while not stop.is_set():
            now = time.perf_counter()
            longest[0] = max(longest[0], min(now, window[1]) - max(last[0], window[0]))
            last[0] = now

    worker = threading.Thread(target=stamp)
    worker.start()
    try:
        time.sleep(0.05)
        window[0] = time.perf_counter()
        call()
        window[1] = time.perf_counter()
    finally:
        stop.set()
        worker.join()
    tic, toc = window
    # with no stamp after the call, its last gap ends at the call's end
    return max(longest[0], toc - max(last[0], tic)) / max(toc - tic, 1e-9)


@contextlib.contextmanager
def raises_seen(target):
    """
    A list that gets True each time `target`, a function of the extension,
    raises from inside itself while this is open, as sys.monitoring's C_RAISE
    sees it. Tells a call that stopped part way at Ctrl-C from one that never
    looked, which finishes and returns, the interpreter then raising the
    interrupt still pending as soon as it has (a C_RETURN instead).
    """
    monitoring = sys.monitoring
    tool = monitoring.PROFILER_ID
    raised = []

    def seen(code, offset, called, argument):
        if called is target:
            raised.append(True)

    monitoring.use_tool_id(tool, "trimkit's tests")
    try:
        monitoring.register_callback(tool, monitoring.events.C_RAISE, seen)
        # C_RAISE comes with CALL, and isn't set on its own
        monitoring.set_events(tool, monitoring.events.CALL)
        yield raised
    finally:
        monitoring.set_events(tool, 0)
        monitoring.register_callback(tool, monitoring.events.C_RAISE, None)
        monitoring.free_tool_id(tool)


def cpus_allowed() -> int:
    """
    The CPUs the pool may use here, as src/platform.hpp counts them: the
    affinity, and no more than any cgroup's `cpu.max` quota, rounded up,
    from this process's cgroup to the root. `docker --cpus=1` on a big host
    leaves the affinity wide and the pool no workers.
    """
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    root = Path("/sys/fs/cgroup")
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return count
    own = next((line[3:] for line in lines if line.startswith("0::/")), "")
    directory = root / own.lstrip("/")
    while True:
        try:
            quota, period = (root / directory / "cpu.max").read_text().split()
        except (OSError, ValueError):
            quota = period = None
        if quota is not None and quota != "max" and int(period) > 0:
            count = min(count, max(1, -(-int(quota) // int(period))))
        if directory == root:
            return count
        directory = directory.parent


def long_call(seconds: float = 1.0):
    """
    A call that takes at least `seconds` here, however many cores this
    machine has and however fast, and the seconds it took: the one place a
    test gets a call long enough to land a signal, a fork or an exit
    inside. It is RayScene.intersect, 250,000 rays against one mesh placed
    many times in one place, so each ray traverses every instance for its
    nearest hit: the work is rays times instances and the memory rays plus
    instances, so the instances double until a call takes `seconds` without
    the memory growing with the machine. Every range polls for Ctrl-C
    (trimkit.hpp's in_parallel, polled through gil.hpp's with_pool), as every query's does.
    """
    import trimkit

    scene = trimkit.RayScene([sphere(100, 200)])
    origins, directions = np.random.default_rng(0).normal(size=(2, 250_000, 3)) * 0.1
    instances = 1
    while True:
        matrices = np.tile(np.eye(4), (instances, 1, 1))
        index = np.zeros(instances, dtype=np.int64)

        def call(matrices=matrices, index=index):
            return scene.intersect(origins, directions, matrices, index, maximum_hits=1)

        tic = time.perf_counter()
        call()
        took = time.perf_counter() - tic
        if took >= seconds:
            return call, took
        instances *= 2


# how many times `assert_releases_gil` measures before it fails, waiting
# twice as long before each try as the last, from GIL_FIRST_WAIT
GIL_POLLS = 6
GIL_FIRST_WAIT = 0.01


def assert_releases_gil(call) -> None:
    """
    The one test that `call` releases the GIL around its work: a background
    thread is held off for under a third of the call at once (Python's checks
    before and after the work hold it). A held GIL stalls every measurement,
    and a busy machine's scheduler only some, so it measures until one is
    under a third, backing off exponentially between tries so a busy machine
    settles, and fails only after GIL_POLLS measurements that all were over.
    """
    stalls = []
    for poll in range(GIL_POLLS):
        stalls.append(longest_gil_stall(call))
        if stalls[-1] < 1 / 3:
            return
        time.sleep(GIL_FIRST_WAIT * 2**poll)
    raise AssertionError(
        f"the GIL was held for {min(stalls):.0%} of the call at once, the least of {GIL_POLLS}"
    )


# what one parsed SVG segment means, every point resolved: lets tests compare
# `trimkit.exchange.svg_parse` against independent references field by field
SVG_ROWS = np.dtype(
    [
        ("kind", "u1"),
        ("closes", "?"),  # a line back to its subpath's first vertex: a `Z`
        ("moves", "?"),  # starts a subpath: not on the vertex the last segment ended on
        ("start", "(2,)f8"),
        ("end", "(2,)f8"),
        ("control", "(2,2)f8"),  # a Bezier's, or a conic's in the first row
        ("middle", "(2,)f8"),  # on the curve halfway along an arc, a conic or a cubic
        ("center", "(2,)f8"),  # an arc's, from its `kind_data`
        ("weight", "f8"),  # a conic's, from its `kind_data`
    ]
)


def svg_rows(vertices: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Expand parsed SVG segments into one row per segment with every point
    resolved, NaN where a kind has none.
    """
    kinds, points, data = segments["kind"], segments["points"], segments["kind_data"]
    rows = np.zeros(len(segments), SVG_ROWS)
    rows["kind"] = kinds
    start, end = points[:, 0], points[:, 3]
    rows["moves"][:1] = True
    rows["moves"][1:] = start[1:] != end[:-1]
    # each row's subpath starts on the vertex of the last row that moved
    subpath = start[np.maximum.accumulate(np.where(rows["moves"], np.arange(len(kinds), dtype=np.int64), 0))]
    rows["closes"] = (kinds == 0) & (end == subpath) & (start != subpath)
    rows["start"], rows["end"] = vertices[start], vertices[end]
    for field in ("control", "middle", "center", "weight"):
        rows[field] = np.nan
    for kind, slots in ((1, 1), (2, 2), (4, 1)):
        chosen = kinds == kind
        for slot in range(slots):
            rows["control"][chosen, slot] = vertices[points[chosen, 1 + slot]]
    arc = kinds == 3
    rows["middle"][arc] = vertices[points[arc, 1]]
    rows["center"][arc] = data[arc]
    conic = kinds == 4
    w = data[conic, :1]
    rows["weight"][conic] = w[:, 0]
    rows["middle"][conic] = (
        rows["start"][conic] + 2 * w * rows["control"][conic, 0] + rows["end"][conic]
    ) / (2 + 2 * w)
    cubic = kinds == 2
    rows["middle"][cubic] = (
        rows["start"][cubic] + 3 * rows["control"][cubic].sum(axis=1) + rows["end"][cubic]
    ) / 8
    return rows


def check_svg_arc(pieces: np.ndarray, ellipse, sweep_positive: bool, large: bool) -> None:
    """
    Every piece an SVG arc became lies on the ellipse `svg_reference.ellipse`
    gives for it, in order and in the direction of travel, each at most 120
    degrees with its middle halfway along it, and together the large arc
    exactly when asked (or a half ellipse, from radii scaled up to reach).
    An arc's center is the circle's.
    """
    center, radius_x, radius_y, (cos, sin) = ellipse[:4]
    # rounding is relative to the coordinates, so in the unit frame it is
    # each axis' extent over its radius: the tolerance follows a needle-thin
    # ellipse far from the origin, but stays tight for an upright one
    extent_x = abs(center[0]) + radius_x * abs(cos) + radius_y * abs(sin)
    extent_y = abs(center[1]) + radius_x * abs(sin) + radius_y * abs(cos)
    spread = (abs(cos) * extent_x + abs(sin) * extent_y) / radius_x
    spread += (abs(sin) * extent_x + abs(cos) * extent_y) / radius_y
    tolerance = 1e-9 * (1.0 + spread)

    def angle_and_reach(point):
        x, y = point[..., 0] - center[0], point[..., 1] - center[1]
        u, v = (cos * x + sin * y) / radius_x, (-sin * x + cos * y) / radius_y
        return np.arctan2(v, u), np.hypot(u, v)

    def turned(angle):
        return angle % (2 * math.pi) if sweep_positive else angle % (2 * math.pi) - 2 * math.pi

    arcs = pieces[pieces["kind"] == 3]
    assert np.abs(arcs["center"] - center).max(initial=0) <= tolerance * (
        abs(center[0]) + abs(center[1]) + radius_x
    )
    total = 0.0
    for row in pieces:
        assert row["kind"] in (3, 4)
        angles, reach = angle_and_reach(np.array([row["start"], row["middle"], row["end"]]))
        assert np.abs(reach - 1).max() <= tolerance
        sweep, half = turned(angles[2] - angles[0]), turned(angles[1] - angles[0])
        assert abs(sweep) <= math.radians(120) + tolerance
        assert abs(half - sweep / 2) <= tolerance
        total += sweep
    assert abs(abs(total) - math.pi) <= tolerance or (abs(total) > math.pi) == large


def svg_round_trip(path: str) -> str:
    """
    Parse `path`, write it with 17 digits, and check that parsing what was
    written gives back the same kinds and points, and the same vertices bit
    for bit, except an arc's halfway point and a conic's control, recomputed
    from the radii written, which come back within rounding, magnified by an
    ellipse's eccentricity, as do an arc's center and a conic's weight.
    Returns what was written.
    """
    import svg_reference as reference

    import trimkit

    vertices, segments = trimkit.exchange.svg_parse(path)
    written = trimkit.exchange.svg_string(vertices, segments, digits=17)
    again_vertices, again = trimkit.exchange.svg_parse(written)
    assert again["kind"].tobytes() == segments["kind"].tobytes()
    assert again["points"].tobytes() == segments["points"].tobytes()
    exact = np.ones(len(vertices), dtype=bool)
    exact[segments["points"][segments["kind"] >= 3, 1]] = False
    assert again_vertices[exact].tobytes() == vertices[exact].tobytes()
    # the most eccentric arc written, its long radius over its short: what
    # recomputing an ellipse from its radii magnifies rounding by. Read by the
    # reference only where there are arcs (kinds 3 and 4): its parse tree of
    # a path of millions of lines is gigabytes
    curved = bool((segments["kind"] >= 3).any())
    arcs = [
        s["radius"]
        for s in (reference.evaluate(reference.parse(reference.LENIENT_PARSER, written)) if curved else ())
        if "radius" in s
    ]
    eccentricity = max((max(r) / min(r) for r in arcs), default=1.0)
    magnified = 64 * np.finfo(np.float64).eps * eccentricity
    scale = 1.0 + np.abs(vertices).max(initial=0)
    assert np.abs(again_vertices[~exact] - vertices[~exact]).max(initial=0) <= (1e-11 + magnified) * scale
    arc, conic = segments["kind"] == 3, segments["kind"] == 4
    reach = 1.0 + np.abs(segments["kind_data"][arc]).max(initial=0)
    assert np.abs(again["kind_data"][arc] - segments["kind_data"][arc]).max(initial=0) <= (
        1e-11 + magnified
    ) * max(scale, reach)
    assert (
        np.abs(again["kind_data"][conic] - segments["kind_data"][conic]).max(initial=0) <= 1e-12 + magnified
    )
    rest = ~(arc | conic)
    assert (again["kind_data"][rest] == 0).all() and (segments["kind_data"][rest] == 0).all()
    return written


def check_svg_segments(rows: np.ndarray, expected: list, reference) -> None:
    """
    `svg_rows` of a parse against `svg_reference.evaluate`'s segments for the
    same data (`reference` is that module). Lines and Beziers are the same
    doubles: both parses round correctly, and the arithmetic on them is the
    spec's, in the same order. An arc is one, two or four pieces, the last
    ending exactly where it ends, all on the reference's ellipse
    (`check_svg_arc`). Or it is a line, where its middle is within rounding
    of its chord, or no double can place a center for it at all.
    """
    position = 0
    for e in expected:
        row = rows[position]
        assert row["moves"] == e["moves"] and row["start"].tolist() == list(e["start"])
        if e["kind"] != reference.ARC or row["kind"] == 0:
            assert row["kind"] == e["kind"] or e["kind"] == reference.ARC
            assert bool(row["closes"]) == bool(e["flags"] & reference.CLOSE)
            assert row["end"].tolist() == list(e["end"])
            used = len(e.get("control", ()))
            assert row["control"][:used].tolist() == [list(c) for c in e.get("control", ())]
            assert np.isnan(row["control"][used:]).all()
            if e["kind"] == reference.ARC:
                arc = reference.ellipse(
                    e["start"],
                    e["end"],
                    *e["radius"],
                    e["rotation"],
                    large=bool(e["flags"] & reference.LARGE_ARC),
                    sweep=bool(e["flags"] & reference.SWEEP),
                )
                if arc is not None:
                    middle = np.array(reference.arc_point(arc, arc[4] + arc[5] / 2))
                    chord = (np.array(e["start"]) + np.array(e["end"])) / 2
                    scale = 1.0 + np.abs(arc[0]).sum() + arc[1] + arc[2]
                    assert np.abs(middle - chord).max() <= 1e-12 * scale
            position += 1
            continue
        pieces = 1
        while rows["end"][position + pieces - 1].tolist() != list(e["end"]):
            pieces += 1
        assert pieces in (1, 2, 4)
        group = rows[position : position + pieces]
        assert not group["moves"][1:].any() and not group["closes"].any()
        # the pieces of one arc carry one center, exactly
        assert (group["center"][group["kind"] == 3] == group["center"][0]).all()
        circle = e["radius"][0] == e["radius"][1]
        assert (group["kind"] == (3 if circle else 4)).all() or (not circle and (group["kind"] == 1).all())
        if group["kind"][0] != 1:
            large, sweep = bool(e["flags"] & reference.LARGE_ARC), bool(e["flags"] & reference.SWEEP)
            arc = reference.ellipse(
                e["start"], e["end"], *e["radius"], e["rotation"], large=large, sweep=sweep
            )
            check_svg_arc(group, arc, sweep, large)
        position += pieces
    assert position == len(rows)
