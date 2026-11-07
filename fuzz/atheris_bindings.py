"""
Fuzz the Python boundary of every function: random dtypes, shapes, strides,
read-only flags and values go in, and each call must either raise
ValueError or TypeError, or return fresh arrays with indices in range.
Run by `uv run scripts atheris` against an ASan build.
"""

import contextlib
import pathlib
import sys

import atheris

# the tests' helpers, for `public_api`
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

with atheris.instrument_imports():
    import numpy as np
    from helpers import public_api

    import trimkit
DTYPES = [np.float64, np.float32, np.int64, np.int32, np.uint32, np.uint8, np.int16, np.bool_]


def array(data: atheris.FuzzedDataProvider, columns: int | None, dtype=None) -> np.ndarray:
    rows = data.ConsumeIntInRange(0, 40)
    shape = (
        (rows,)
        if columns is None
        else (rows, data.ConsumeIntInRange(0, 4) if data.ConsumeBool() else columns)
    )
    dtype = dtype or DTYPES[data.ConsumeIntInRange(0, len(DTYPES) - 1)]
    raw = data.ConsumeBytes(int(np.prod(shape)) * np.dtype(dtype).itemsize * 2)
    flat = np.frombuffer(raw.ljust(int(np.prod(shape)) * np.dtype(dtype).itemsize * 2, b"\0"), dtype=dtype)
    # every other element, so the array may arrive strided
    out = (
        flat[::2].reshape(shape) if data.ConsumeBool() else flat[: int(np.prod(shape))].reshape(shape).copy()
    )
    if data.ConsumeBool():
        out.flags.writeable = False
    return out


def indices(data: atheris.FuzzedDataProvider, count: int) -> np.ndarray:
    rows = data.ConsumeIntInRange(0, 60)
    values = data.ConsumeIntListInRange(rows * 3, -2, max(count, 1) + 1)
    return np.array(values + [0] * (rows * 3 - len(values)), dtype=np.int64).reshape((rows, 3))


def check(result) -> None:
    # every array, however deeply nested, is fresh: writable and C-contiguous
    if isinstance(result, (tuple, list)):
        for item in result:
            check(item)
    elif isinstance(result, np.ndarray):
        assert result.flags.writeable and result.flags.c_contiguous


def case_earcut(data: atheris.FuzzedDataProvider):
    exterior = array(data, 2)
    interiors = [array(data, 2) for _ in range(data.ConsumeIntInRange(0, 3))]
    vertices, faces = trimkit.triangulate(exterior, interiors)
    assert faces.max(initial=0) < max(len(vertices), 1)
    result = (vertices, faces)
    return result


def case_svg_parse(data: atheris.FuzzedDataProvider):
    text = data.ConsumeUnicodeNoSurrogates(200)
    stop = data.ConsumeBool()
    if stop:
        # a NUL or non-ASCII character ends the path data where it is
        bad = data.PickValueInList(["\0", "\u00e9", "\u2003", "\U0001f600"])
        text = "M0 0 L1 1" + bad + text
    result = trimkit.exchange.svg_parse(text)
    vertices, segments = result
    assert not stop or len(segments) == 1
    assert segments.dtype == trimkit.exchange.SvgSegment and segments["points"].max(initial=-1) < len(
        vertices
    )
    # the round trip holds for anything parsed
    written = trimkit.exchange.svg_string(vertices, segments)
    again = trimkit.exchange.svg_parse(written)
    assert again[0].tobytes() == vertices.tobytes() and again[1].tobytes() == segments.tobytes()
    assert trimkit.exchange.svg_string(*again) == written
    return result


def case_svg_string(data: atheris.FuzzedDataProvider):
    # arbitrary vertices and bytes as segments: refused cleanly or written
    size = trimkit.exchange.SvgSegment.itemsize
    raw = data.ConsumeBytes(size * data.ConsumeIntInRange(0, 4))
    segments = np.frombuffer(raw[: len(raw) // size * size], dtype=np.uint8).view(trimkit.exchange.SvgSegment)
    result = trimkit.exchange.svg_string(array(data, 2, np.float64), segments)
    return result


def case_vhacd(data: atheris.FuzzedDataProvider):
    vertices = array(data, 3)
    result = trimkit.decompose(
        vertices,
        indices(data, len(vertices)),
        # every keyword, with values on both sides of each valid range
        resolution=data.ConsumeIntInRange(0, 20000),
        maximum_hulls=data.ConsumeIntInRange(0, 4),
        maximum_error=data.ConsumeFloat(),
        maximum_vertices=data.ConsumeIntInRange(0, 64),
    )
    for hull_vertices, hull_faces in result:
        assert hull_faces.max(initial=-1) < len(hull_vertices)
    return result


def case_xatlas(data: atheris.FuzzedDataProvider):
    vertices = array(data, 3)
    result = trimkit.parametrize(vertices, indices(data, len(vertices)))
    assert result[0].max(initial=0) < max(len(vertices), 1)
    return result


def case_draco_encode(data: atheris.FuzzedDataProvider):
    position = array(data, 3, np.float32)
    buffer = trimkit.exchange.draco_encode(
        indices(data, len(position)),
        [position],
        np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64),
        quantization_bits=data.ConsumeIntInRange(-1, 31) if data.ConsumeBool() else None,
    )
    result = trimkit.exchange.draco_decode(buffer, np.array([0], dtype=np.int64))
    assert result[0].shape[1:] == (3,)
    return result


def case_draco_decode(data: atheris.FuzzedDataProvider):
    wanted = np.array(data.ConsumeIntListInRange(data.ConsumeIntInRange(0, 3), -1, 3), dtype=np.int64)
    result = trimkit.exchange.draco_decode(data.ConsumeBytes(400), wanted)
    return result


def case_rays(data: atheris.FuzzedDataProvider):
    vertices = array(data, 3)
    scene = trimkit.RayScene([(vertices, indices(data, len(vertices)))])
    origins, directions = array(data, 3), array(data, 3)
    result = scene.intersect_first(origins, directions)
    assert result[1].max(initial=-1) < 60
    rays, _, _, _ = scene.intersect_all(origins, directions)
    assert rays.max(initial=-1) < len(origins)
    scene.intersect_any(origins, directions)
    return result


def case_instances(data: atheris.FuzzedDataProvider):
    meshes = []
    for _ in range(2):
        vertices = array(data, 3)
        meshes.append((vertices, indices(data, len(vertices))))
    scene = trimkit.RayScene(meshes)
    origins, directions = array(data, 3), array(data, 3)
    # two placements, the first asked again after the second: a remembered
    # top level answers exactly as the one built for it did
    placements = []
    for _ in range(2):
        transforms = array(data, 16).reshape((-1, 4, 4))
        placements.append((transforms, array(data, None, np.int64)))
    answers = [scene.intersect_first(origins, directions, *placed) for placed in (*placements, placements[0])]
    assert all(a.tobytes() == b.tobytes() for a, b in zip(answers[0], answers[2], strict=True))
    # the boxes of those instances, one each, ordered corners
    boxes = scene.boxes(*placements[0]).boxes()
    assert boxes.shape == (len(placements[0][1]), 2, 3) and (boxes[:, 0] <= boxes[:, 1]).all()
    return (*answers[0], boxes)


def case_meshoptimizer(data: atheris.FuzzedDataProvider):
    vertices = array(data, 3)
    result = trimkit.simplify(
        vertices,
        indices(data, len(vertices)),
        data.ConsumeIntInRange(-1, 70),
        maximum_error=data.ConsumeFloat(),
        lock_border=data.ConsumeBool(),
    )
    assert result[1].max(initial=-1) < len(result[0])
    assert result[2].max(initial=-1) < max(len(vertices), 1)
    return result


def rigid(data: atheris.FuzzedDataProvider, count: int) -> np.ndarray:
    # rigid transforms from fuzzed numbers, so queries reach the kernels and
    # not only the refusal of anything else: a rotation from a normalized
    # quaternion, in plain arithmetic, as numpy's BLAS threads crash under
    # Atheris's ASan runtime by themselves
    out = np.tile(np.eye(4), (count, 1, 1))
    for i in range(count):
        w, x, y, z = (data.ConsumeFloatInRange(-1.0, 1.0) for _ in range(4))
        norm = (w * w + x * x + y * y + z * z) ** 0.5
        w, x, y, z = (1.0, 0.0, 0.0, 0.0) if norm < 1e-6 else (w / norm, x / norm, y / norm, z / norm)
        out[i, :3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        out[i, :3, 3] = [data.ConsumeFloatInRange(-10.0, 10.0) for _ in range(3)]
    return out


def case_collision(data: atheris.FuzzedDataProvider):
    meshes = []
    for _ in range(data.ConsumeIntInRange(0, 3)):
        vertices = array(data, 3)
        meshes.append((vertices, indices(data, len(vertices))))
    convex = (
        np.array([data.ConsumeBool() for _ in meshes], dtype=bool)
        if data.ConsumeBool()
        else data.ConsumeBool()
    )
    scene = trimkit.CollisionScene(meshes, convex)
    index = array(data, None, np.int64) if data.ConsumeBool() else None
    count = len(index) if index is not None else len(meshes)
    matrices = rigid(data, count) if data.ConsumeBool() else array(data, 16).reshape((-1, 4, 4))
    pairs = array(data, 2, np.int64) if data.ConsumeBool() else None
    hits = scene.collides(matrices, index, pairs=pairs, margin=data.ConsumeFloat())
    distances, points, normals = scene.distances(
        matrices, index, pairs=pairs, upper_bound=data.ConsumeFloat()
    )
    assert hits.shape[1:] == (2,) and hits.min(initial=0) >= 0 and hits.max(initial=-1) < max(count, 1)
    assert points.shape == (len(distances), 2, 3) and normals.shape == (len(distances), 3)
    boxes = scene.boxes(matrices, index).boxes()
    assert boxes.shape == (count, 2, 3) and (boxes[:, 0] <= boxes[:, 1]).all()
    return distances, points, normals, boxes


def case_boxes(data: atheris.FuzzedDataProvider):
    d = data.ConsumeIntInRange(1, 4)
    bounds = array(data, 2 * d).reshape((-1, 2, d)) if data.ConsumeBool() else array(data, None)
    tree = trimkit.Boxes(bounds)
    result = tree.intersect(array(data, 2 * d).reshape((-1, 2, d)))
    assert result.shape[1:] == (2,) and result[:, 1].max(initial=-1) < len(tree)
    # every box back exactly, and the union of them
    copy = tree.boxes()
    assert copy.tobytes() == bounds.tobytes()
    if len(copy):
        assert np.array_equal(tree.bounds, [copy[:, 0].min(axis=0), copy[:, 1].max(axis=0)])
    return result, copy, tree.bounds


# every public name, the case that fuzzes it: checked against the compiled
# modules at startup, so nothing public goes unfuzzed
CASES = {
    "triangulate": case_earcut,
    "exchange.svg_parse": case_svg_parse,
    "exchange.svg_string": case_svg_string,
    "decompose": case_vhacd,
    "parametrize": case_xatlas,
    "exchange.draco_encode": case_draco_encode,
    "exchange.draco_decode": case_draco_decode,
    "RayScene": case_instances,
    "RayScene.boxes": case_instances,
    "RayScene.intersect_first": case_rays,
    "RayScene.intersect_all": case_rays,
    "RayScene.intersect_any": case_rays,
    "simplify": case_meshoptimizer,
    "Boxes": case_boxes,
    "Boxes.bounds": case_boxes,
    "Boxes.boxes": case_boxes,
    "Boxes.intersect": case_boxes,
    "CollisionScene": case_collision,
    "CollisionScene.boxes": case_collision,
    "CollisionScene.collides": case_collision,
    "CollisionScene.distances": case_collision,
}
NAMES = sorted(CASES)
if public_api() != NAMES:
    sys.exit(f"fuzz/atheris_bindings.py must fuzz exactly {public_api()}, not {NAMES}")


def one(data: atheris.FuzzedDataProvider) -> None:
    check(CASES[NAMES[data.ConsumeIntInRange(0, len(NAMES) - 1)]](data))


def test_one_input(raw: bytes) -> None:
    data = atheris.FuzzedDataProvider(raw)
    # refusing bad input with an exception is the correct outcome
    with contextlib.suppress(ValueError, TypeError):
        one(data)


atheris.Setup(sys.argv, test_one_input)
atheris.Fuzz()
