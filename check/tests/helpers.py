"""
Small numpy-only fixtures and predicates shared by the tests.
"""

import ctypes
import functools
import math
import os
import signal
import struct
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

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
        vertices = np.split(data["vertices"], np.cumsum(data["vertex_counts"])[:-1])
        faces = np.split(data["faces"].astype(np.int64), np.cumsum(data["face_counts"])[:-1])
        meshes = list(zip(map(np.ascontiguousarray, vertices), map(np.ascontiguousarray, faces), strict=True))
        return meshes, data["instance_matrices"], data["instance_index"].astype(np.int64)


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


def check_boxes(boxes, meshes, matrices=None, index=None) -> None:
    """
    That a scene's `boxes(...)` hold exactly what it was given: one 3D box
    per instance, between the box of its placed faces and its mesh's whole
    box carried by its transform (Arvo), to a few ulp (and a ray scene's
    padding of 1e-12 of the mesh's extent); an instance of a mesh without
    faces, the point at its translation. Round-trips through `Boxes` too.
    """
    import trimkit

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


def same(a, b) -> None:
    """
    Two answers, exactly: each array of one is the other's, in dtype, shape
    and every byte.
    """
    a, b = (x if isinstance(x, (tuple, list)) else (x,) for x in (a, b))
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes()


def ray_scene(meshes, matrices=None, index=None):
    """
    A RayScene of these meshes, checked to know where these instances of
    them are: its queries then name the same instances.
    """
    import trimkit

    scene = trimkit.RayScene(meshes)
    check_boxes(scene.boxes(matrices, index), meshes, matrices, index)
    return scene


def collision_boxes(scene, meshes, matrices=None, index=None) -> None:
    """
    That a CollisionScene knows where these instances of its meshes are.
    """
    check_boxes(scene.boxes(matrices, index), meshes, matrices, index)


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
    Every array, bytes, str, int or float in a result, in a fixed order.
    """
    import trimkit

    if isinstance(result, (tuple, list)):
        return [a for r in result for a in leaves(r)]
    if isinstance(result, trimkit.CollisionResult):
        return leaves(collision_fields(result))
    assert isinstance(result, (np.ndarray, bytes, str, int, float)), type(result)
    return [result]


def collision_fields(result) -> list:
    """
    A CollisionResult's arrays in a fixed order, those it holds: its pairs
    always, the rest when its query measured them.
    """
    fields = (result.pairs, result.faces, result.distances, result.points, result.normals)
    return [field for field in fields if field is not None]


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


def small_calls() -> dict:
    """
    A quick call of everything in `public_api()`, by its name.
    """
    import trimkit

    vertices, faces = sphere(16, 32)
    position = [vertices.astype(np.float32)]
    buffer = (
        trimkit.exchange.draco_encode(
            faces, position, np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
        )
        if LITTLE_ENDIAN
        else b""
    )
    rays = np.random.default_rng(0).normal(size=(2, 500, 3))
    corners = np.random.default_rng(0).uniform(-1, 1, (500, 3))
    boxes = np.stack([corners, corners + 0.1], axis=1)
    placed = np.tile(np.eye(4), (3, 1, 1))
    placed[:, 0, 3] = [0.0, 3.0, 6.0]
    three = np.zeros(3, dtype=np.int64)
    path = "M0 0 c1 2 3 4 5 6 a3 2 30 1 0 4 4 z" * 200
    segments = trimkit.exchange.svg_parse(path)
    mesh = (vertices, faces)
    # a plain and a convex sphere, two instances of each, each overlapping the next
    overlapping = np.tile(np.eye(4), (4, 1, 1))
    overlapping[:, 0, 3] = [0.0, 1.5, 3.0, 4.5]
    apart = overlapping.copy()
    apart[:, 0, 3] *= 2.0
    which = np.array([0, 0, 1, 1], dtype=np.int64)

    def spheres():
        return trimkit.CollisionScene([mesh, mesh], convex=np.array([False, True]))

    return {
        # building inside the call keeps no scene alive past the tests
        "RayScene": lambda: len(trimkit.RayScene([mesh])),
        "RayScene.boxes": lambda: trimkit.RayScene([mesh]).boxes(placed, three).boxes(),
        "RayScene.intersect": lambda: trimkit.RayScene([mesh]).intersect(*rays, placed, three),
        "RayScene.intersect_any": lambda: trimkit.RayScene([mesh]).intersect_any(*rays),
        "Boxes": lambda: len(trimkit.Boxes(boxes)),
        "Boxes.bounds": lambda: trimkit.Boxes(boxes).bounds,
        "Boxes.boxes": lambda: trimkit.Boxes(boxes).boxes(),
        "Boxes.intersection": lambda: trimkit.Boxes(boxes).intersection(boxes),
        "Boxes.nearest": lambda: trimkit.Boxes(boxes).nearest(boxes, maximum_hits=4),
        "CollisionResult": lambda: len(spheres().distances(apart, which)),
        "CollisionResult.pairs": lambda: spheres().collides(overlapping, which).pairs,
        "CollisionResult.faces": lambda: spheres().collides(overlapping, which, return_distance=True).faces,
        "CollisionResult.distances": lambda: spheres().distances(apart, which).distances,
        "CollisionResult.points": lambda: spheres().distances(apart, which).points,
        "CollisionResult.normals": lambda: (
            spheres().collides(overlapping, which, return_distance=True).normals
        ),
        "CollisionScene": lambda: spheres().collides(overlapping, which),
        "CollisionScene.boxes": lambda: spheres().boxes(overlapping, which).boxes(),
        "CollisionScene.collides": lambda: spheres().collides(
            overlapping, which, margin=0.1, return_distance=True, maximum_contacts=4
        ),
        "CollisionScene.distances": lambda: spheres().distances(apart, which),
        "triangulate": lambda: trimkit.triangulate(circle(500)),
        "exchange.svg_parse": lambda: trimkit.exchange.svg_parse(path),
        "exchange.svg_string": lambda: trimkit.exchange.svg_string(*segments),
        "decompose": lambda: trimkit.decompose(*box(), resolution=20_000),
        "parametrize": lambda: trimkit.parametrize(*mesh),
        "simplify": lambda: trimkit.simplify(*mesh, len(faces) // 4),
        "contour": lambda: trimkit.contour(ball_runs(12)),
        "exchange.draco_encode": lambda: trimkit.exchange.draco_encode(
            faces, position, np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
        ),
        "exchange.draco_decode": lambda: trimkit.exchange.draco_decode(buffer, np.array([0], dtype=np.int64)),
    }


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


def longest_gil_stall(call, minimum: float = 0.2) -> float:
    """
    The longest a background Python thread was held off during `call`, as a
    fraction of the call: about 1 when the call holds the GIL throughout,
    and small when it releases it around its work, as a held GIL stops the
    thread outright for as long as it is held.
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
    elapsed = toc - tic
    assert elapsed > minimum, f"call took {elapsed:.3f}s, make the input larger"
    # with no stamp after the call, its last gap ends at the call's end
    return max(longest[0], toc - max(last[0], tic)) / elapsed


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
    center, radius_x, radius_y, phi = ellipse[:4]
    cos, sin = math.cos(phi), math.sin(phi)
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
    import trimkit

    vertices, segments = trimkit.exchange.svg_parse(path)
    written = trimkit.exchange.svg_string(vertices, segments, digits=17)
    again_vertices, again = trimkit.exchange.svg_parse(written)
    assert again["kind"].tobytes() == segments["kind"].tobytes()
    assert again["points"].tobytes() == segments["points"].tobytes()
    exact = np.ones(len(vertices), dtype=bool)
    exact[segments["points"][segments["kind"] >= 3, 1]] = False
    assert again_vertices[exact].tobytes() == vertices[exact].tobytes()
    scale = 1.0 + np.abs(vertices).max(initial=0)
    assert np.abs(again_vertices[~exact] - vertices[~exact]).max(initial=0) <= 1e-11 * scale
    arc, conic = segments["kind"] == 3, segments["kind"] == 4
    reach = 1.0 + np.abs(segments["kind_data"][arc]).max(initial=0)
    assert np.abs(again["kind_data"][arc] - segments["kind_data"][arc]).max(initial=0) <= 1e-11 * max(
        scale, reach
    )
    assert np.abs(again["kind_data"][conic] - segments["kind_data"][conic]).max(initial=0) <= 1e-12
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
