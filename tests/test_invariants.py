"""
The package-wide promises: numpy is the only dependency, Python is a
docstring, every array input is exactly typed and bounded, read in place when
C-contiguous and aligned and copied once otherwise, converted only when numpy
casts the dtype safely and then with a warning, every output is a fresh array
that owns its memory and aliases no input, the tables cover the whole public
API, and the GIL is released while computing.
"""

import _thread
import functools
import importlib.metadata
import os
import pathlib
import select
import signal
import subprocess
import sys
import threading
import time
import tracemalloc
import warnings

import numpy as np
import pytest
from helpers import (
    LITTLE_ENDIAN,
    box,
    circle,
    first_hits,
    leaves,
    longest_gil_stall,
    public_api,
    runs_here,
    small_calls,
    sphere,
)

import trimkit

API = public_api()


# Inputs slow enough to see a background thread run. Each is built for the one
# test that uses it and goes when that test does: cached, they held 900 MiB for
# the rest of the session, which is what a 32-bit process ran out of.
def _big_mesh():
    return sphere(60, 120)


def _big_rays():
    placed = np.tile(np.eye(4), (8, 1, 1))
    placed[:, 0, 3] = np.arange(8) * 3.0
    return _big_mesh(), np.random.default_rng(0).normal(size=(2, 2_000_000, 3)), placed


def _big_corners():
    return np.random.default_rng(0).uniform(-100, 100, (1_000_000, 3))


def _big_boxes():
    corners = _big_corners()
    return np.stack([corners, corners + 1.0], axis=1)


def _many_instances():
    # a million instances: their boxes build into a Boxes as big as _big_boxes
    many = np.tile(np.eye(4), (1_000_000, 1, 1))
    many[:, :3, 3] = _big_corners()
    return many, np.zeros(len(many), np.int64)


def _near_spheres():
    # two big spheres a hair apart, the one pair tested many times: every test
    # descends both hierarchies to the leaves nearest the gap
    near = np.tile(np.eye(4), (2, 1, 1))
    near[1, 0, 3] = 2.0001
    return sphere(200, 400), near, np.tile(np.array([[0, 1]], dtype=np.int64), (2_000, 1))


def _big_path():
    return "M0 0 " + "c1 2 3 4 5 6 a 3 2 30 1 0 4 4 " * 1_000_000


def _big_draco():
    n = 2_000_000
    position = [np.random.default_rng(0).random((n, 3)).astype(np.float32)]
    faces = np.arange(3 * (n // 3), dtype=np.int64).reshape((-1, 3))
    return faces, position, np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)


def _big_buffer():
    return trimkit.exchange.draco_encode(*_big_draco()) if LITTLE_ENDIAN else b""


# each name's input and what it calls on that input
_SLOW_CALLS = {
    "RayScene": (lambda: sphere(500, 1000), lambda mesh: len(trimkit.RayScene([mesh]))),
    "RayScene.boxes": (_many_instances, lambda i: trimkit.RayScene([box()]).boxes(*i).boxes()),
    "RayScene.intersect": (
        _big_rays,
        lambda i: trimkit.RayScene([i[0]]).intersect(*i[1], i[2], np.zeros(8, dtype=np.int64)),
    ),
    "RayScene.intersect_any": (_big_rays, lambda i: trimkit.RayScene([i[0]]).intersect_any(*i[1])),
    "Boxes": (_big_boxes, lambda b: len(trimkit.Boxes(b))),
    "Boxes.bounds": (_big_boxes, lambda b: trimkit.Boxes(b).bounds),
    "Boxes.boxes": (_big_boxes, lambda b: trimkit.Boxes(b).boxes()),
    "Boxes.intersect": (_big_boxes, lambda b: trimkit.Boxes(b).intersect(b)),
    "CollisionScene.boxes": (_many_instances, lambda i: trimkit.CollisionScene([box()]).boxes(*i).boxes()),
    "CollisionScene": (
        _near_spheres,
        lambda i: trimkit.CollisionScene([i[0]]).collides(i[1], np.zeros(2, np.int64), pairs=i[2]),
    ),
    "CollisionScene.collides": (
        _near_spheres,
        lambda i: trimkit.CollisionScene([i[0]]).collides(i[1], np.zeros(2, np.int64), pairs=i[2]),
    ),
    "CollisionScene.distances": (
        _near_spheres,
        lambda i: trimkit.CollisionScene([i[0]]).distances(i[1], np.zeros(2, np.int64), pairs=i[2][:200]),
    ),
    "triangulate": (lambda: circle(2_000_000), trimkit.triangulate),
    "exchange.svg_parse": (_big_path, trimkit.exchange.svg_parse),
    "exchange.svg_string": (
        lambda: trimkit.exchange.svg_parse(_big_path()),
        lambda parsed: trimkit.exchange.svg_string(*parsed),
    ),
    "decompose": (_big_mesh, lambda mesh: trimkit.decompose(*mesh, resolution=50_000)),
    "parametrize": (_big_mesh, lambda mesh: trimkit.parametrize(*mesh)),
    "simplify": (lambda: sphere(400, 800), lambda mesh: trimkit.simplify(*mesh, 1000)),
    "exchange.draco_encode": (_big_draco, lambda i: trimkit.exchange.draco_encode(*i)),
    "exchange.draco_decode": (
        _big_buffer,
        lambda buffer: trimkit.exchange.draco_decode(buffer, np.array([0], dtype=np.int64)),
    ),
}


def slow_call(name: str):
    # the call for `name`, its input built now and freed along with the call
    build, call = _SLOW_CALLS[name]
    given = build()
    return lambda: call(given)


@functools.cache
def empty_calls():
    # each function on an empty input, or None where none can exist
    no_vertices, no_faces = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    no_rays = (no_vertices, no_vertices)
    return {
        "RayScene": lambda: len(trimkit.RayScene([])),
        "RayScene.boxes": lambda: trimkit.RayScene([]).boxes().boxes(),
        "RayScene.intersect": lambda: trimkit.RayScene([(no_vertices, no_faces)]).intersect(
            *no_rays, np.zeros((0, 4, 4)), np.zeros(0, dtype=np.int64)
        ),
        "RayScene.intersect_any": lambda: trimkit.RayScene([]).intersect_any(*no_rays),
        "Boxes": lambda: len(trimkit.Boxes(np.zeros((0, 2, 3)))),
        "Boxes.bounds": lambda: trimkit.Boxes(np.zeros((0, 2, 3))).bounds,
        "Boxes.boxes": lambda: trimkit.Boxes(np.zeros((0, 2, 3))).boxes(),
        "Boxes.intersect": lambda: trimkit.Boxes(np.zeros((0, 2, 3))).intersect(np.zeros((0, 2, 3))),
        "CollisionScene.boxes": lambda: trimkit.CollisionScene([]).boxes().boxes(),
        "CollisionScene": lambda: trimkit.CollisionScene([]).collides(),
        "CollisionScene.collides": lambda: trimkit.CollisionScene([(no_vertices, no_faces)] * 2).collides(),
        "CollisionScene.distances": lambda: trimkit.CollisionScene([]).distances(),
        "triangulate": lambda: trimkit.triangulate(np.zeros((0, 2))),
        "exchange.svg_parse": lambda: trimkit.exchange.svg_parse(""),
        "exchange.svg_string": lambda: trimkit.exchange.svg_string(*trimkit.exchange.svg_parse("")),
        "decompose": lambda: trimkit.decompose(no_vertices, no_faces),
        "parametrize": lambda: trimkit.parametrize(no_vertices, no_faces),
        "simplify": lambda: trimkit.simplify(no_vertices, no_faces, 0),
        # draco has no empty primitive: it refuses one without vertices
        "exchange.draco_encode": None,
        "exchange.draco_decode": None,
    }


def test_tables_cover_the_api():
    # every public function, class and method is in every table, and nothing
    # else: a new one can't go untested, and a renamed one can't linger
    for table in (small_calls(), _SLOW_CALLS, empty_calls(), _array_arguments()):
        assert sorted(table) == API


@pytest.mark.parametrize("name", API)
@pytest.mark.parametrize("which", ["small", "empty"])
def test_outputs_are_fresh(name, which):
    # every output is a fresh, writable, C-contiguous array (or bytes, a str
    # or a number), empty ones included: nothing else holds its memory, so
    # writing to it is safe and nothing has to copy it first
    runs_here(name)
    call = (small_calls() if which == "small" else empty_calls())[name]
    if call is None:
        pytest.skip("no empty input exists")
    for a in [a for a in leaves(call()) if isinstance(a, np.ndarray)]:
        assert a.flags.writeable and a.flags.c_contiguous
        a.flat[:1] = 0


@pytest.mark.timing
@pytest.mark.skipif(not getattr(sys, "_is_gil_enabled", lambda: True)(), reason="no GIL to hold")
def test_stall_detector_catches_a_held_gil():
    # the control: sorting a big list is one C call that never releases it,
    # held off for most of the call (the rest is the call's own setup).
    # Shuffled, as timsort sorts a run (a reversed range) in linear time
    data = np.random.default_rng(0).permutation(1_000_000).tolist()
    assert longest_gil_stall(lambda: sorted(data), minimum=0.05) > 0.5


@pytest.mark.timing
@pytest.mark.parametrize("name", API)
def test_releases_gil(name):
    runs_here(name)
    # held off for at most a third of the call: Python's checks before and
    # after the work hold the GIL. The least of three runs, as a held GIL
    # stalls every run and a busy machine's scheduler only some of them
    call = slow_call(name)
    stall = min(longest_gil_stall(call, minimum=0.05) for _ in range(3))
    assert stall < 1 / 3, f"the GIL was held for {stall:.0%} of the call at once"


def test_only_numpy():
    requires = importlib.metadata.requires("trimkit") or []
    assert [r for r in requires if "extra ==" not in r] == ["numpy>=2"]


def test_the_package_is_the_extension():
    # no Python at all: `trimkit` is one extension, the package's __init__,
    # beside only its stubs and typing marker
    root = pathlib.Path(trimkit.__file__).parent
    assert pathlib.Path(trimkit.__file__).name.startswith("__init__.")
    assert not pathlib.Path(trimkit.__file__).name.endswith(".py")
    assert sorted(
        p.name for p in root.iterdir() if p.name != "__pycache__" and p != pathlib.Path(trimkit.__file__)
    ) == [
        "__init__.pyi",
        "exchange.pyi",
        "py.typed",
    ]


@pytest.mark.parametrize("name", API)
def test_values_must_be_bounded(name):
    # every float input is refused unless finite, and float64 ones unless
    # within 1e30 too: past it, products downstream overflow (rays silently
    # missed past 5.6e102), and NaN reaches undefined float-to-int casts
    runs_here(name)
    function, arguments = _array_arguments()[name]
    for argument, value in arguments.items():
        for path, array, _ in _paths(value):
            if array.dtype.kind != "f" or array.size == 0:
                continue
            bad_values = [np.nan, np.inf, -np.inf] + ([1e31, -1e31] if array.dtype == np.float64 else [])
            for bad in bad_values:
                broken = array.copy()
                broken.flat[0] = bad
                with pytest.raises(ValueError, match="must be finite"):
                    function(**{**arguments, argument: _replace(value, path, broken)})


def test_the_bound_itself_is_accepted():
    # exactly 1e30 is in range, and works: a mesh at that scale is hit by rays
    # and simplifies like the unit one
    vertices, faces = sphere(8, 16)
    huge = vertices * 1e30
    assert np.abs(huge).max() == 1e30
    _, hit_faces, _ = first_hits(
        trimkit.RayScene([(huge, faces)]), np.zeros((50, 3)), np.random.default_rng(0).normal(size=(50, 3))
    )
    assert (hit_faces >= 0).all()
    assert len(trimkit.simplify(huge, faces, 20)[1]) == len(trimkit.simplify(vertices, faces, 20)[1])


def _array_arguments(big: bool = False) -> dict:
    """
    Every public call with valid keyword arguments, by name: built inside
    each test, so no scene outlives it (nanobind reports those as leaks).
    `big` makes every array argument hundreds of kilobytes.
    """
    vertices, faces = sphere(100, 200) if big else box()
    rays = np.random.default_rng(0).normal(size=(2, 20_000 if big else 8, 3))
    scene = trimkit.RayScene([(vertices, faces)])
    # big ones narrow, so their overlaps are a few per box rather than all pairs
    bounds = np.stack([vertices, vertices + (1e-3 if big else 1.0)], axis=1)
    position = vertices.astype(np.float32)
    kind = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
    buffer = trimkit.exchange.draco_encode(faces, [position], kind) if LITTLE_ENDIAN else b""
    segments = trimkit.exchange.svg_parse("M0 0 L1 1 L2 0" * (10_000 if big else 1))
    instance_count = 5_000 if big else 2
    placed = np.tile(np.eye(4), (instance_count, 1, 1))
    placed[:, 0, 3] = np.arange(instance_count) * 3.0
    origins_directions = {"origins": rays[0], "directions": rays[1]}
    ray_instances = {"instance_matrices": placed, "instance_index": np.zeros(instance_count, np.int64)}
    collision_scene = trimkit.CollisionScene([(vertices, faces)])
    neighbors = np.column_stack([np.arange(instance_count - 1), np.arange(1, instance_count)]).astype(
        np.int64
    )
    instances = {
        "instance_matrices": placed,
        "instance_index": np.zeros(instance_count, np.int64),
        "pairs": neighbors,
    }
    calls = [
        (
            "RayScene",
            trimkit.RayScene,
            {"meshes": [(vertices, faces)]},
        ),
        ("RayScene.boxes", scene.boxes, ray_instances),
        ("RayScene.intersect", scene.intersect, {**origins_directions, **ray_instances}),
        ("RayScene.intersect_any", scene.intersect_any, {**origins_directions, **ray_instances}),
        ("Boxes", trimkit.Boxes, {"boxes": bounds}),
        ("Boxes.bounds", lambda: trimkit.Boxes(bounds).bounds, {}),
        ("Boxes.boxes", trimkit.Boxes(bounds).boxes, {}),
        ("Boxes.intersect", trimkit.Boxes(bounds).intersect, {"queries": bounds}),
        (
            "CollisionScene",
            trimkit.CollisionScene,
            {"meshes": [(vertices, faces)], "convex": np.array([True])},
        ),
        (
            "CollisionScene.boxes",
            collision_scene.boxes,
            {"instance_matrices": placed, "instance_index": np.zeros(instance_count, np.int64)},
        ),
        ("CollisionScene.collides", collision_scene.collides, instances),
        ("CollisionScene.distances", collision_scene.distances, instances),
        ("triangulate", trimkit.triangulate, {"exterior": circle(40_000 if big else 8)}),
        ("decompose", trimkit.decompose, {"vertices": vertices, "faces": faces, "resolution": 20_000}),
        ("parametrize", trimkit.parametrize, {"vertices": vertices, "faces": faces}),
        (
            "simplify",
            trimkit.simplify,
            {"vertices": vertices, "faces": faces, "face_count": 4},
        ),
        (
            "exchange.draco_encode",
            trimkit.exchange.draco_encode,
            {"faces": faces, "attributes": [position], "kinds": np.array([0], dtype=np.int64)},
        ),
        (
            "exchange.draco_decode",
            trimkit.exchange.draco_decode,
            {"buffer": np.frombuffer(buffer, np.uint8), "draco_attributes": np.array([0], dtype=np.int64)},
        ),
        ("exchange.svg_parse", trimkit.exchange.svg_parse, {"path": "M0 0 L1 1 L2 0"}),
        (
            "exchange.svg_string",
            trimkit.exchange.svg_string,
            {"vertices": segments[0], "segments": segments[1]},
        ),
    ]
    return {name: (function, arguments) for name, function, arguments in calls}


def _paths(value, path=(), in_tuple=False):
    """
    (path, array, in_tuple) for every array in an argument, through lists and
    tuples: whether it is a tuple's item, as a mesh's arrays are.
    """
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _paths(item, (*path, i), isinstance(value, tuple))
    elif isinstance(value, np.ndarray):
        yield path, value, in_tuple


def _replace(value, path, new):
    # `value` with the item at `path` replaced, lists and tuples rebuilt
    if not path:
        return new
    items = list(value)
    items[path[0]] = _replace(items[path[0]], path[1:], new)
    return type(value)(items)


def _converted(value: np.ndarray):
    # the same values as a narrower dtype of the same kind (a safe cast back,
    # converted with a warning), as a list (the same), and as a dtype that
    # casts back only with loss (refused). A uint8 buffer as a list reads as
    # int64, which doesn't cast back safely either
    narrower = {np.dtype(np.float64): np.float32, np.dtype(np.int64): np.int32, np.dtype(np.uint8): np.int8}
    lossy = {np.dtype(np.float64): np.complex128, np.dtype(np.int64): np.float64, np.dtype(np.uint8): np.int8}
    yield "narrower", value.astype(narrower.get(value.dtype, np.float64))
    yield "list", value.tolist()
    yield "lossy", value.astype(lossy.get(value.dtype, np.int64))


def _relaid(value: np.ndarray):
    # exactly these values in every other layout: strided, Fortran-ordered,
    # misaligned, and reversed views of reversed copies (negative strides, as
    # `np.flipud` and `np.fliplr` make) that read back as the original
    yield "strided", np.repeat(value, 2, axis=0)[::2]
    yield "reversed", np.ascontiguousarray(value[::-1])[::-1]
    if value.ndim > 1:
        yield "fortran", np.asfortranarray(value)
        yield "flipped", np.ascontiguousarray(value[..., ::-1])[..., ::-1]
    raw = np.zeros(value.nbytes + 1, dtype=np.uint8)
    shifted = np.frombuffer(raw.data, dtype=value.dtype, count=value.size, offset=1).reshape(value.shape)
    shifted[...] = value
    yield "misaligned", shifted


@pytest.mark.parametrize("name", API)
def test_safe_conversions_warn_and_match(name):
    # every array argument, and each array of a mesh pair, given as another
    # dtype or a list: converted with a UserWarning naming the argument and
    # answering exactly as the exact dtype does when numpy casts it back
    # safely, a TypeError naming the argument otherwise (the bare arrays of a
    # list, draco's attributes, take any dtype as they are)
    runs_here(name)
    function, arguments = _array_arguments()[name]
    for argument, value in arguments.items():
        for path, array, in_tuple in _paths(value):
            if array.dtype.names or not (in_tuple or not path):
                continue
            for how, wrong in _converted(array):
                given = {**arguments, argument: _replace(value, path, wrong)}
                if not np.can_cast(np.asarray(wrong).dtype, array.dtype, "safe"):
                    with pytest.raises(TypeError, match=argument):
                        function(**given)
                    continue
                # the same values as the exact dtype (narrowing rounded them)
                same = np.asarray(wrong).astype(array.dtype).reshape(array.shape)
                expected = _results(function(**{**arguments, argument: _replace(value, path, same)}))
                with pytest.warns(UserWarning, match=f"{argument}.*converted to {array.dtype.name}"):
                    got = _results(function(**given))
                for a, b in zip(got, expected, strict=True):
                    same = np.array_equal(a, b, equal_nan=True) if isinstance(a, np.ndarray) else a == b
                    assert same, (argument, how)


def test_a_mesh_is_any_pair():
    # a mesh is any sequence of two: a list is as good as a tuple, a triple isn't
    vertices, faces = box()
    expected = trimkit.RayScene([(vertices, faces)]).boxes().boxes()
    assert np.array_equal(trimkit.RayScene([[vertices, faces]]).boxes().boxes(), expected)
    assert np.array_equal(trimkit.CollisionScene([[vertices, faces]]).boxes().boxes(), expected)
    with pytest.raises(TypeError, match=r"meshes\[0\] must be a \(vertices, faces\) pair"):
        trimkit.RayScene([(vertices, faces, faces)])


def _results(result) -> list:
    # what a call answers, comparable: its arrays, bytes, strings and numbers
    if isinstance(result, (tuple, list)):
        return [r for item in result for r in _results(item)]
    if isinstance(result, trimkit.Boxes):
        return [result.boxes()]
    # what a scene holds, so its constructor's layouts are compared too
    if isinstance(result, (trimkit.RayScene, trimkit.CollisionScene)):
        return [result.boxes().boxes()]
    assert isinstance(result, (np.ndarray, bytes, str, int, float)), type(result)
    return [result]


@pytest.mark.parametrize("name", API)
def test_any_layout_answers_the_same(name):
    # every array in any layout (strided, reversed, Fortran-ordered,
    # misaligned, the records of a structured dtype too) gives exactly the
    # answer its C-contiguous copy does: copied once, as the caller would
    runs_here(name)
    function, arguments = _array_arguments()[name]
    expected = _results(function(**arguments))
    for argument, value in arguments.items():
        for path, array, _ in _paths(value):
            for how, relaid in _relaid(array):
                got = _results(function(**{**arguments, argument: _replace(value, path, relaid)}))
                assert len(got) == len(expected), (argument, how)
                for a, b in zip(got, expected, strict=True):
                    same = np.array_equal(a, b, equal_nan=True) if isinstance(a, np.ndarray) else a == b
                    assert same, (argument, how)


def _arrays(value) -> list:
    # every array in an argument or a result, through tuples and lists
    if isinstance(value, (tuple, list)):
        return [a for item in value for a in _arrays(item)]
    return [value] if isinstance(value, np.ndarray) else []


@pytest.mark.parametrize("name", API)
def test_outputs_never_alias_inputs(name):
    runs_here(name)
    function, arguments = _array_arguments()[name]
    inputs = _arrays(list(arguments.values()))
    for output in _arrays(function(**arguments)):
        assert not any(np.shares_memory(output, given) for given in inputs)


def _peak_python_bytes(call) -> int:
    """
    The most memory Python and numpy held at once during `call`, above what
    they held before it: numpy reports every array buffer it allocates to
    tracemalloc, and our own C++ allocations and outputs don't go through it.
    """
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak - before


def test_copy_detector_catches_copies():
    # the control: a copy the size of the input is plainly visible
    data = np.random.default_rng(0).random((50_000, 3))
    assert _peak_python_bytes(lambda: np.ascontiguousarray(data[::-1])) >= data.nbytes


@pytest.mark.parametrize("name", API)
def test_inputs_are_never_copied(name):
    # nothing converts or copies an input on the way in: during the call numpy
    # allocates far less than the largest input
    runs_here(name)
    function, arguments = _array_arguments(big=True)[name]
    largest = max((a.nbytes for a in _arrays(list(arguments.values()))), default=0)
    if largest == 0:
        pytest.skip("no array arguments to copy")
    assert largest > 200_000
    assert _peak_python_bytes(lambda: function(**arguments)) < largest // 2


def _long_query(name: str, scale: int):
    # a query whose structure is built outside it, so only the polling loop is
    # timed: a tinybvh build has no point at which to check for Ctrl-C
    rng = np.random.default_rng(0)
    if name == "RayScene.intersect":
        scene = trimkit.RayScene([sphere(100, 200)])
        origins, directions = rng.normal(size=(2, 250_000 * scale, 3)) * 0.1
        return lambda: scene.intersect(origins, directions)
    corners = rng.uniform(0, 100, (200_000, 3))
    tree = trimkit.Boxes(np.stack([corners, corners + 1.0], axis=1))
    queries = rng.uniform(0, 100, (25_000 * scale, 3))
    bounds = np.stack([queries, queries + 3.0], axis=1)
    return lambda: tree.intersect(bounds)


@pytest.mark.timing
@pytest.mark.parametrize("name", ["decompose", "RayScene.intersect", "Boxes.intersect"])
def test_ctrl_c_stops_long_calls(name):
    # every long loop polls for Ctrl-C with the GIL released: an interrupt
    # 50 ms in stops the call well before it would finish. Queries double
    # until a call takes 1 s, so this holds on fast and slow machines alike:
    # stopping takes the 50 ms timer plus up to one 50 ms poll period, and at
    # 0.5 s a loaded machine missed the half-call bound by 5 ms (2026-11-01).
    # (xatlas polls only from its worker threads, which can't check signals:
    # it stops at the end of the call, a known gap.)
    scale = 1
    while True:
        call = slow_call(name) if name == "decompose" else _long_query(name, scale)
        tic = time.perf_counter()
        call()
        uninterrupted = time.perf_counter() - tic
        if uninterrupted > 1.0 or name == "decompose" or scale >= 128:
            break
        scale *= 2
    assert uninterrupted > 0.5, "the input couldn't be made slow enough to interrupt mid-call"
    timer = threading.Timer(0.05, _thread.interrupt_main)
    tic = time.perf_counter()
    timer.start()
    with pytest.raises(KeyboardInterrupt):
        call()
        time.sleep(10)  # only if the call ignored it: the interrupt lands here instead
    timer.join()
    assert time.perf_counter() - tic < uninterrupted / 2


@pytest.mark.timing
def test_random_ctrl_c_leaves_the_pool_healthy():
    # Ctrl-C at random moments in a long parallel query, workers mid-range
    # or not: each raises promptly, and the next call is whole and the same
    call = _long_query("Boxes.intersect", 8)
    expected = call()
    for delay in np.random.default_rng(7).uniform(0.005, 0.2, 6):
        timer = threading.Timer(delay, _thread.interrupt_main)
        tic = time.perf_counter()
        timer.start()
        with pytest.raises(KeyboardInterrupt):
            call()
            # the call finished first: the interrupt lands here, between
            # sleeps, as interrupt_main sends no signal to cut one short
            for _ in range(1000):
                time.sleep(0.01)
        timer.join()
        # a 50 ms poll and one range at most; a second covers a loaded machine
        assert time.perf_counter() - tic < delay + 1.0
    assert np.array_equal(call(), expected)


# big enough that every parallel query wakes workers, hashed to compare runs
_PARALLEL_QUERIES = """
import hashlib, os, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
from helpers import sphere
import trimkit
if sys.argv[1] == "one":
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
rng = np.random.default_rng(0)
corners = rng.uniform(0, 100, (50_000, 3))
queries = rng.uniform(0, 100, (50_000, 3))
scene = trimkit.RayScene([sphere(60, 120)])
origins, directions = rng.normal(size=(2, 100_000, 3)) * 0.1
solids = trimkit.CollisionScene([sphere(10, 20)], convex=True)
matrices = np.tile(np.eye(4), (300, 1, 1))
matrices[:, :3, 3] = rng.uniform(0, 15, (300, 3))
index = np.zeros(300, np.int64)
boxes = trimkit.Boxes(np.stack([corners, corners + 1.0], axis=1))
outputs = [
    boxes.intersect(np.stack([queries, queries + 3.0], axis=1)),
    *scene.intersect(origins, directions, maximum_hits=1),
    *scene.intersect(origins, directions),
    *solids.distances(matrices, index),
    solids.collides(matrices, index),
]
print(hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in outputs)).hexdigest())
"""


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="needs CPU affinity (Linux)")
def test_one_core_matches_every_core():
    # however many workers a query gets, down to none, the answers are the
    # same bytes: each item's output depends on that item alone
    tests = str(pathlib.Path(__file__).parent)
    digests = {
        cores: subprocess.run(
            [sys.executable, "-c", _PARALLEL_QUERIES, cores, tests],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for cores in ("one", "every")
    }
    assert digests["one"] == digests["every"] != ""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_fork_after_a_parallel_call():
    # a child forked while the parent's workers are alive starts its own and
    # answers the same; Python warns about forking with threads alive
    call = _long_query("Boxes.intersect", 2)
    expected = call()
    read, write = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        try:
            os.write(write, b"1" if np.array_equal(call(), expected) else b"0")
        finally:
            os._exit(0)
    os.close(write)
    try:
        # a child that hangs fails this rather than the whole run
        assert select.select([read], [], [], 120.0)[0], "the forked child hung"
        assert os.read(read, 1) == b"1"
    finally:
        os.close(read)
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)


@pytest.mark.timing
@pytest.mark.skipif(not pathlib.Path("/proc/self/task").is_dir(), reason="counts threads through /proc")
def test_workers_exit_when_idle():
    # parallel.hpp's workers sleep for 10 s with nothing to do, then exit, so
    # an idle process holds no threads of ours
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("one CPU: queries never wake workers")
    call = _long_query("Boxes.intersect", 2)

    def threads() -> int:
        return len(os.listdir("/proc/self/task"))

    time.sleep(11)
    before = threads()
    call()
    assert threads() > before
    time.sleep(11)
    assert threads() == before


def test_enums_are_int_enums_named_for_their_function():
    # every enumeration is an IntEnum beside the functions taking it, none at the
    # top level, each built from its int and an int itself, and named for them:
    # DracoKind for draco_*, SvgKind for svg_*
    import enum
    import re

    from trimkit.exchange import DracoKind

    exchange = trimkit.exchange
    families = [v for v in vars(exchange).values() if isinstance(v, type) and issubclass(v, enum.Enum)]
    assert sorted(f.__name__ for f in families) == ["DracoKind", "SvgKind"]
    assert DracoKind is exchange.DracoKind and sys.modules["trimkit.exchange"] is exchange
    assert not [v for v in vars(trimkit).values() if isinstance(v, type) and issubclass(v, enum.Enum)]
    functions = [n for n, v in vars(exchange).items() if callable(v) and not isinstance(v, type)]
    for family in families:
        assert issubclass(family, enum.IntEnum)
        for member in family:
            assert family(int(member)) is member
            assert member == int(member)
        prefix = re.match(r"[A-Z][a-z]+", family.__name__)[0].lower()
        assert [f for f in functions if f.startswith(prefix + "_")], family
    assert int(exchange.SvgKind.CUBIC) == 2
    assert exchange.SvgSegment.fields["kind"][0] == np.uint8


def test_draco_refuses_only_big_endian():
    # on a big-endian machine both draco calls refuse before reading anything
    # (every value would come out byte-swapped); anywhere else they work
    vertices, faces = box()
    kind = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
    position = [vertices.astype(np.float32)]
    if LITTLE_ENDIAN:
        buffer = trimkit.exchange.draco_encode(faces, position, kind)
        assert len(
            trimkit.exchange.draco_decode(np.frombuffer(buffer, np.uint8), np.array([0], dtype=np.int64))[0]
        ) == len(faces)
        return
    message = "draco reads and writes only little-endian data, and this machine is big-endian"
    with pytest.raises(NotImplementedError, match=message):
        trimkit.exchange.draco_encode(faces, position, kind)
    with pytest.raises(NotImplementedError, match=message):
        trimkit.exchange.draco_decode(np.zeros(8, np.uint8), np.array([0], dtype=np.int64))


def _public(name: str):
    # a public name of `trimkit`, `X` or `X.member`, as the object documented
    import trimkit

    if name == "trimkit":
        return trimkit
    if name == "trimkit.exchange":
        return trimkit.exchange
    owner, _, member = name.partition(".")
    value = getattr(trimkit, owner)
    return vars(value)[member] if member else value


@pytest.mark.parametrize("name", [*API, "trimkit", "trimkit.exchange"])
def test_examples_run(name):
    # every public callable's numpy Examples section runs and prints exactly
    # what it says: no example in a docstring or the README can rot
    import doctest

    runs_here(name)
    doc = _public(name).__doc__
    test = doctest.DocTestParser().get_doctest(doc, {}, name, None, 0)
    assert test.examples, f"{name} has no Examples section"
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    runner.run(test)
    assert runner.failures == 0


def _attributes(doc: str) -> list[str]:
    # the names a numpy Attributes section lists
    section = doc.split("Attributes\n----------\n", 1)[1].split("\n\n", 1)[0]
    return [line.split(" :")[0] for line in section.splitlines() if line and not line[0].isspace()]


def test_docs_name_everything():
    # the module names its every non-callable public name, each enum its
    # every member (each with its own docstring), and each __len__ what it
    # counts: the numpy sections the README tables are built from
    import enum
    import types

    plain = [
        name
        for name in dir(trimkit)
        if not name.startswith("_")
        and (not callable(getattr(trimkit, name)) or isinstance(getattr(trimkit, name), types.GenericAlias))
    ]
    assert sorted(_attributes(trimkit.__doc__)) == sorted([*plain, "__version__"])
    assert _attributes(trimkit.exchange.__doc__) == ["DracoKind", "SvgKind", "SvgSegment"]
    families = [
        v for v in vars(trimkit.exchange).values() if isinstance(v, type) and issubclass(v, enum.Enum)
    ]
    for family in families:
        assert _attributes(family.__doc__) == [m.name for m in family]
        assert all(m.__doc__ and m.__doc__ != family.__doc__ for m in family)
    for name in API:
        value = _public(name)
        if isinstance(value, type) and "__len__" in vars(value):
            assert "Returns\n-------\ncount : int" in value.__len__.__doc__, name


def test_version_is_the_packages():
    assert trimkit.__version__ == importlib.metadata.version("trimkit")


def _scalar_calls():
    # every scalar argument of the package, as a call taking it, with a value
    vertices, faces = sphere(8, 12)
    position = vertices.astype(np.float32)
    kind = np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
    placed = np.tile(np.eye(4), (2, 1, 1))
    placed[1, 0, 3] = 3.0
    which = np.zeros(2, np.int64)

    def decompose(**scalar):
        return trimkit.decompose(vertices, faces, resolution=5000, **scalar)

    def scene():
        # built inside the call: this table outlives the tests, and no scene may
        return trimkit.CollisionScene([box()], convex=True)

    return {
        "face_count": (lambda v: trimkit.simplify(vertices, faces, v), 8),
        "maximum_error": (lambda v: trimkit.simplify(vertices, faces, 8, maximum_error=v), 1),
        "lock_border": (lambda v: trimkit.simplify(vertices, faces, 8, lock_border=v), True),
        "decompose maximum_hulls": (lambda v: decompose(maximum_hulls=v), 2),
        "decompose maximum_vertices": (lambda v: decompose(maximum_vertices=v), 8),
        "resolution": (lambda v: trimkit.decompose(vertices, faces, resolution=v), 5000),
        "decompose maximum_error": (lambda v: decompose(maximum_error=v), 4),
        "quantization_bits": (
            lambda v: (trimkit.exchange.draco_encode(faces, [position], kind, quantization_bits=v),),
            9,
        ),
        "convex": (lambda v: trimkit.CollisionScene([box()], convex=v).distances(placed, which), True),
        "margin": (lambda v: (scene().collides(placed, which, margin=v),), 1),
        "maximum_hits": (
            lambda v: trimkit.RayScene([(vertices, faces)]).intersect(
                np.zeros((2, 3)), np.eye(3)[:2], maximum_hits=v
            ),
            2,
        ),
        "upper_bound": (lambda v: scene().distances(placed, which, upper_bound=v), 4),
        # False never touches a device, so this runs everywhere
        "use_gpu": (lambda v: len(trimkit.RayScene([(vertices, faces)], use_gpu=v)), False),
    }


def _numpy_kinds(value) -> list:
    # the same value as each numpy scalar that can hold it, and as a float
    if isinstance(value, bool):
        return [np.bool_(value)]
    kinds = [np.int64, np.int32, np.int16, np.uint8, np.uint64]
    integers = [kind(value) for kind in kinds if value <= np.iinfo(kind).max]
    return [*integers, float(value), np.float64(value), np.float32(value)]


@pytest.mark.parametrize("name", _scalar_calls())
def test_numpy_scalars_are_python_scalars(name):
    # no argument ever needs numpy.float64(10.0), or refuses it: what numpy's
    # own arithmetic returns answers exactly as Python's number does
    runs_here("exchange.draco_encode" if name == "quantization_bits" else name)
    call, value = _scalar_calls()[name]
    expected = call(value)
    integral = name in (
        "face_count",
        "decompose maximum_hulls",
        "decompose maximum_vertices",
        "resolution",
        "quantization_bits",
        "maximum_hits",
    )
    for scalar in _numpy_kinds(value):
        if integral and isinstance(scalar, (float, np.floating)):
            with pytest.raises(TypeError):
                call(scalar)
            continue
        for got, same in zip(_results(call(scalar)), _results(expected), strict=True):
            assert np.array_equal(got, same, equal_nan=True) if isinstance(got, np.ndarray) else got == same


@pytest.mark.parametrize("name", ["lock_border", "convex", "use_gpu"])
@pytest.mark.parametrize("wrong", [0, 1, None, "yes", np.int64(1)])
def test_nothing_else_is_a_bool(name, wrong):
    with pytest.raises(TypeError):
        _scalar_calls()[name][0](wrong)
