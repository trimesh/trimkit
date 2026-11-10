"""
Small numpy-only fixtures and predicates shared by the tests.
"""

import math
import struct
import sys
import threading
import time
import warnings

import numpy as np
import pytest

# draco refuses big-endian machines outright (its format is little-endian and
# it copies raw bytes): test_invariants asserts that, and everything else
# skips draco there, saying so
LITTLE_ENDIAN = sys.byteorder == "little"


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
    if isinstance(result, (tuple, list)):
        return [a for r in result for a in leaves(r)]
    assert isinstance(result, (np.ndarray, bytes, str, int, float)), type(result)
    return [result]


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
        "Boxes.intersect": lambda: trimkit.Boxes(boxes).intersect(boxes),
        "CollisionScene": lambda: spheres().collides(overlapping, which),
        "CollisionScene.boxes": lambda: spheres().boxes(overlapping, which).boxes(),
        "CollisionScene.collides": lambda: spheres().collides(overlapping, which, margin=0.1),
        "CollisionScene.distances": lambda: spheres().distances(apart, which),
        "triangulate": lambda: trimkit.triangulate(circle(500)),
        "exchange.svg_parse": lambda: trimkit.exchange.svg_parse(path),
        "exchange.svg_string": lambda: trimkit.exchange.svg_string(*segments),
        "decompose": lambda: trimkit.decompose(*box(), resolution=20_000),
        "parametrize": lambda: trimkit.parametrize(*mesh),
        "simplify": lambda: trimkit.simplify(*mesh, len(faces) // 4),
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
        ("closes", "?"),
        ("moves", "?"),  # starts a subpath: not on the vertex the last segment ended on
        ("start", "(2,)f8"),
        ("end", "(2,)f8"),
        ("control", "(2,2)f8"),
        ("radius", "(2,)f8"),
        ("rotation", "f8"),
        ("center", "(2,)f8"),
        ("middle", "(2,)f8"),  # an arc's point halfway along its sweep
        ("angles", "(2,)f8"),
    ]
)


def evaluate_arc(arcs: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    A point on each arc at `angle`, from its center parameterization.
    """
    phi = np.radians(arcs["rotation"])
    rx, ry = arcs["radius"].T
    x = rx * np.cos(angle)
    y = ry * np.sin(angle)
    return arcs["center"] + np.column_stack(
        [np.cos(phi) * x - np.sin(phi) * y, np.sin(phi) * x + np.cos(phi) * y]
    )


def svg_rows(vertices: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Expand indexed vertices and segments into one row per segment with every
    point resolved, NaN where a kind has none.
    """
    import trimkit

    rows = np.zeros(len(segments), SVG_ROWS)
    for field in ("kind", "closes", "radius", "rotation", "angles"):
        rows[field] = segments[field]
    points = segments["points"]
    rows["moves"][:1] = True
    rows["moves"][1:] = points[1:, 0] != points[:-1, 3]

    def at(k):
        out = np.full((len(points), 2), np.nan)
        used = points[:, k] >= 0
        out[used] = vertices[points[used, k]]
        return out

    arc = (segments["kind"] == trimkit.exchange.SvgKind.ARC)[:, None]
    rows["start"], rows["end"] = at(0), at(3)
    rows["control"][:, 0] = np.where(arc, np.nan, at(1))
    rows["control"][:, 1] = np.where(arc, np.nan, at(2))
    rows["center"] = np.where(arc, at(1), np.nan)
    rows["middle"] = np.where(arc, at(2), np.nan)
    return rows
