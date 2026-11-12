"""
RayScene(use_gpu=True): the same answers as the CPU, to the byte, and the
CPU answering with a warning when no device can.
"""

import os
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest
from helpers import box, camera, cycloidal, device_scene, exploded, same, sphere, subdivided

import trimkit

pytestmark = pytest.mark.device


def rays(count, seed):
    rng = np.random.default_rng(seed)
    origins = rng.normal(size=(count, 3)) * 2.0
    directions = rng.normal(size=(count, 3))
    # some from inside, some aimed exactly at vertices, a few standing still
    origins[: count // 4] *= 0.1
    directions[count // 2 : count // 2 + 2] = 0.0
    return origins, directions


def instances(count, seed):
    rng = np.random.default_rng(seed)
    matrices = np.tile(np.eye(4), (count, 1, 1))
    matrices[:, :3, 3] = rng.uniform(-3, 3, (count, 3))
    # rotations about z and non-uniform scales
    angles = rng.uniform(0, 2 * np.pi, count)
    matrices[:, 0, 0] = np.cos(angles) * rng.uniform(0.5, 2.0, count)
    matrices[:, 0, 1] = -np.sin(angles)
    matrices[:, 1, 0] = np.sin(angles)
    matrices[:, 1, 1] = np.cos(angles) * rng.uniform(0.5, 2.0, count)
    return matrices


@pytest.mark.parametrize("mesh", [sphere(40, 80), box()], ids=["sphere", "box"])
def test_device_answers_the_cpu_answers(mesh, device_required):
    cpu = trimkit.RayScene([mesh])
    gpu = device_scene([mesh], device_required)
    for seed in range(3):
        origins, directions = rays(5000, seed)
        for maximum_hits in (None, 1, 3):
            same(
                cpu.intersect(origins, directions, maximum_hits=maximum_hits),
                gpu.intersect(origins, directions, maximum_hits=maximum_hits),
            )
        same(cpu.intersect_any(origins, directions), gpu.intersect_any(origins, directions))


def test_device_answers_through_instances(device_required):
    meshes = [sphere(20, 40), box(), (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))]
    cpu = trimkit.RayScene(meshes)
    gpu = device_scene(meshes, device_required)
    matrices = instances(30, 7)
    index = np.arange(30, dtype=np.int64) % 3
    origins, directions = rays(20000, 11)
    for maximum_hits in (None, 1, 2):
        same(
            cpu.intersect(origins, directions, matrices, index, maximum_hits=maximum_hits),
            gpu.intersect(origins, directions, matrices, index, maximum_hits=maximum_hits),
        )
    same(
        cpu.intersect_any(origins, directions, matrices, index),
        gpu.intersect_any(origins, directions, matrices, index),
    )
    # and the default placement, each mesh once
    same(cpu.intersect(origins, directions), gpu.intersect(origins, directions))


def adversarial_rays(vertices, faces):
    """
    Rays that sit on every decision boundary: axis-aligned with -0.0 components
    from the mesh's own vertices, through every vertex and edge midpoint from
    inside, grazing box faces, tiny and huge directions.
    """
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    midpoints = vertices[edges].mean(axis=1)
    inside = np.array([0.05, -0.1, 0.02])
    axes = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    # the sign of a zero component is part of the decision (copysign in tinybvh)
    negative_zero = -axes * 0.0 + axes
    origins = [
        np.tile(inside, (len(vertices) + len(midpoints), 1)),
        vertices,
        np.repeat(vertices, len(axes), axis=0),
        vertices[: len(vertices) // 2] * 1.5,
        vertices * 1e-6,
        vertices * 1e4,
    ]
    directions = [
        np.vstack([vertices, midpoints]) - inside,
        -vertices,
        np.tile(negative_zero, (len(vertices), 1)),
        -vertices[: len(vertices) // 2] * 1e6,
        -vertices * 1e-6,
        -vertices * 1e4,
    ]
    return np.vstack(origins), np.vstack(directions)


@pytest.mark.parametrize("mesh", [sphere(24, 48), box()], ids=["sphere", "box"])
def test_device_agrees_on_every_decision_boundary(mesh, device_required):
    cpu = trimkit.RayScene([mesh])
    gpu = device_scene([mesh], device_required)
    origins, directions = adversarial_rays(*mesh)
    for maximum_hits in (None, 1):
        same(
            cpu.intersect(origins, directions, maximum_hits=maximum_hits),
            gpu.intersect(origins, directions, maximum_hits=maximum_hits),
        )
    same(cpu.intersect_any(origins, directions), gpu.intersect_any(origins, directions))
    # and through instances that move the boundaries by exact amounts
    matrices = np.tile(np.eye(4), (3, 1, 1))
    matrices[1, :3, 3] = [0.5, -0.25, 0.125]
    matrices[2, 0, 0] = 2.0
    index = np.zeros(3, dtype=np.int64)
    same(
        cpu.intersect(origins, directions, matrices, index),
        gpu.intersect(origins, directions, matrices, index),
    )


def test_one_origin_or_direction_is_every_rays(device_required):
    # a (3,) origin or direction is every ray's: the answers of the array that
    # repeats it, on the CPU and on the device, which reads one row for all
    mesh = sphere(30, 60)
    cpu = trimkit.RayScene([mesh])
    gpu = device_scene([mesh], device_required)
    origins, directions = rays(20000, 5)
    origin, direction = np.array([0.1, 0.2, 3.0]), np.array([0.03, -0.02, -1.0])
    every_origin, every_direction = np.tile(origin, (20000, 1)), np.tile(direction, (20000, 1))
    for one, repeated in (
        ((origin, directions), (every_origin, directions)),
        ((origins, direction), (origins, every_direction)),
        ((origin, direction), (origin[None], direction[None])),
    ):
        for maximum_hits in (None, 1):
            expected = cpu.intersect(*repeated, maximum_hits=maximum_hits)
            assert len(expected[0]) > 0
            for scene in (cpu, gpu):
                same(expected, scene.intersect(*one, maximum_hits=maximum_hits))
                same(expected, scene.intersect(*repeated, maximum_hits=maximum_hits))
        for scene in (cpu, gpu):
            same(cpu.intersect_any(*repeated), scene.intersect_any(*one))


def test_device_answers_many_crossings_per_ray(device_required):
    # rays through a stack of twenty boxes cross forty surfaces each, through
    # instances
    meshes = [box((-5, -5, -5), (5, 5, 5))]
    matrices = np.tile(np.eye(4), (20, 1, 1))
    matrices[:, 2, 2] = 0.04
    matrices[:, 2, 3] = np.linspace(-4, 4, 20)
    index = np.zeros(20, dtype=np.int64)
    cpu = trimkit.RayScene(meshes)
    gpu = device_scene(meshes, device_required)
    rng = np.random.default_rng(13)
    origins = rng.uniform(-4, 4, (3000, 3))
    origins[:, 2] = -20.0
    directions = rng.normal(size=(3000, 3)) * 0.01
    directions[:, 2] = 1.0
    for maximum_hits in (None, 2, 5):
        same(
            cpu.intersect(origins, directions, matrices, index, maximum_hits=maximum_hits),
            gpu.intersect(origins, directions, matrices, index, maximum_hits=maximum_hits),
        )
    hits, _, _, _ = gpu.intersect(origins, directions, matrices, index)
    assert np.array_equal(np.bincount(hits, minlength=3000), np.full(3000, 40))


def sheets(spans):
    """
    One mesh of flat sheets across y, sheet k at z = k + 1 over x in spans[k].
    """
    corners = ((0, -9), (1, -9), (1, 9), (0, 9))
    vertices = np.array([[span[c], y, k + 1.0] for k, span in enumerate(spans) for c, y in corners])
    # int64 by name: numpy's default integer is int32 on 32-bit Windows
    first = 4 * np.arange(len(spans), dtype=np.int64)[:, None, None]
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64) + first
    return vertices, faces.reshape(-1, 3)


@pytest.mark.parametrize("count", [63, 64, 65, 255, 256, 257, 1024, 1025])
def test_device_lays_out_crossings_at_group_edges(count, device_required):
    # every ray crosses three sheets and only the last a fourth: ray counts
    # and totals that fill whole groups of the device's lanes, and that
    # don't, each crossing in its ray's order
    mesh = sheets([(-9, 9)] * 3 + [(5, 9)])
    origins = np.column_stack([np.linspace(-8, -1, count), np.full(count, 2.5), np.zeros(count)])
    origins[-1, 0] = 7.0
    directions = np.array([0.0, 0.0, 1.0])
    cpu = trimkit.RayScene([mesh]).intersect(origins, directions)
    same(cpu, device_scene([mesh], device_required).intersect(origins, directions))
    assert np.array_equal(np.bincount(cpu[0]), [3] * (count - 1) + [4])
    assert np.array_equal(cpu[3][-4:], [1.0, 2.0, 3.0, 4.0])


def test_device_runs_a_chunk_again_when_its_log_is_too_small(device_required):
    # six crossings a ray is more than a chunk's log holds when a device
    # opens, and ten more than it holds after that: each time the chunk runs
    # again with room, and a query that needs less afterwards is unaffected
    count = 300_003
    origins = np.column_stack([np.linspace(-8, 8, count), np.full(count, 2.5), np.zeros(count)])
    directions = np.array([0.0, 0.0, 1.0])
    for crossings in (6, 10, 2):
        mesh = sheets([(-9, 9)] * crossings)
        expected = trimkit.RayScene([mesh]).intersect(origins, directions)
        assert len(expected[0]) == crossings * count
        gpu = device_scene([mesh], device_required)
        same(expected, gpu.intersect(origins, directions))
        same(expected, gpu.intersect(origins, directions))


def test_device_orders_mixed_crossing_counts_across_chunks(device_required):
    # rays crossing 0, 1, 5, 16, 17, 33 and 44 sheets, mixed through chunks:
    # the device logs crossings in whatever order its lanes run, and each
    # ray's come back together, nearest first, capped or not
    crossings = np.array([0, 1, 5, 16, 17, 33, 44])
    mesh = sheets([(k, 44) for k in range(44)])
    rng = np.random.default_rng(23)
    count = 300_003
    wanted = rng.choice(crossings, count, p=[0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01])
    # a ray at x crosses the sheets that begin at or before it
    origins = np.column_stack([wanted - 0.5, rng.uniform(-8, 8, count), np.zeros(count)])
    directions = np.array([0.0, 0.0, 1.0])
    cpu = trimkit.RayScene([mesh])
    gpu = device_scene([mesh], device_required)
    expected = cpu.intersect(origins, directions)
    assert np.array_equal(np.bincount(expected[0], minlength=count), wanted)
    same(expected, gpu.intersect(origins, directions))
    for maximum_hits in (1, 5, 40):
        same(
            cpu.intersect(origins, directions, maximum_hits=maximum_hits),
            gpu.intersect(origins, directions, maximum_hits=maximum_hits),
        )
    same(cpu.intersect_any(origins, directions), gpu.intersect_any(origins, directions))


@pytest.mark.parametrize("count", [5008, 300_016])
def test_device_reads_only_this_querys_rays(count, device_required):
    # every ray of both queries misses, but a ray of the second read with the
    # first 16 bytes of its origin and direction left from the first would
    # hit: the device answers from the rays it was just given, whatever a
    # count does to where its chunks of them begin
    scene = device_scene([box((-1, -1, -1), (1, 1, 1))], device_required)
    rng = np.random.default_rng(3)
    near = rng.uniform(-1e-3, 1e-3, (count, 3))
    away = np.array([0.0, 0.0, 1.0])
    assert len(scene.intersect(near + 50.0 * away, near[::-1] * 1e-3 + away)[0]) == 0
    origins = rng.uniform(5, 6, (count, 3))
    directions = rng.uniform(0.5, 1.5, (count, 3)) * np.array([1.0, 1.0, -1.0])
    assert len(scene.intersect(origins, directions)[0]) == 0
    assert not scene.intersect_any(origins, directions).any()


def test_device_holds_a_tree_built_with_splits(device_required):
    # from 100,000 faces a mesh's tree is built with spatial splits, which
    # leave slots no leaf uses holding indices of nothing: following those
    # crashed on this part of the assembly, split once to 169,176 faces
    meshes, _, _ = cycloidal()
    mesh = subdivided(max(meshes, key=lambda m: len(m[1])))
    assert len(mesh[1]) == 169_176
    cpu = trimkit.RayScene([mesh])
    gpu = device_scene([mesh], device_required)
    rng = np.random.default_rng(17)
    middle, size = mesh[0].mean(axis=0), np.ptp(mesh[0], axis=0).max()
    origins = middle + rng.normal(size=(20000, 3)) * size
    directions = middle - origins + rng.normal(size=(20000, 3)) * size * 0.2
    expected = cpu.intersect(origins, directions)
    assert len(expected[0]) > 20000
    same(expected, gpu.intersect(origins, directions))
    same(cpu.intersect_any(origins, directions), gpu.intersect_any(origins, directions))


def test_device_answers_a_real_assembly(device_required):
    # cameras around the cycloidal drive, as loaded and pulled apart: one
    # origin for every ray of a view, most of them missing everything, a few
    # crossing dozens of surfaces
    meshes, matrices, index = cycloidal()
    cpu = trimkit.RayScene(meshes)
    gpu = device_scene(meshes, device_required)
    for placed in (matrices, exploded(meshes, matrices, index, 3.0)):
        box = cpu.boxes(placed, index).bounds
        middle, radius = box.mean(axis=0), np.linalg.norm(np.ptp(box, axis=0)) / 2
        for angle in (0.3, 2.1, 4.4):
            position = middle + 2.6 * radius * np.array([np.cos(angle), np.sin(angle), 0.5])
            directions = camera(position, middle, 200)
            expected = cpu.intersect(position, directions, placed, index)
            assert len(np.unique(expected[0])) > 1000 and np.bincount(expected[0]).max() > 4
            same(expected, gpu.intersect(position, directions, placed, index))
            # the rays that hit at all, by every query
            hit = np.unique(expected[0])
            blocked = cpu.intersect_any(position, directions, placed, index)
            assert np.array_equal(np.flatnonzero(blocked), hit)
            same(blocked, gpu.intersect_any(position, directions, placed, index))
            for scene in (cpu, gpu):
                same(hit, scene.intersect(position, directions, placed, index, maximum_hits=1)[0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e31])
def test_device_refuses_what_the_cpu_refuses(bad, device_required):
    # the device looks at each value as it copies it, a chunk at a time: a
    # value the CPU refuses is refused the same, wherever it is, and no
    # answer comes back
    cpu = trimkit.RayScene([box()])
    gpu = device_scene([box()], device_required)
    count = 300_000
    origins = np.full((count, 3), 2.0)
    directions = np.tile([-1.0, -0.9, -0.8], (count, 1))
    for name, at in (("origins", 0), ("origins", count - 1), ("directions", 7), ("directions", count // 2)):
        spoiled = {"origins": origins.copy(), "directions": directions.copy()}
        spoiled[name][at, 1] = bad
        for query in ("intersect", "intersect_any"):
            raised = []
            for scene in (cpu, gpu):
                with pytest.raises(ValueError, match=f"^{name} must be finite") as error:
                    getattr(scene, query)(**spoiled)
                raised.append(str(error.value))
            assert raised[0] == raised[1]
    # one row for every ray, and no rays at all
    for scene in (cpu, gpu):
        with pytest.raises(ValueError, match=r"^origins must be finite"):
            scene.intersect(np.array([bad, 0.0, 0.0]), directions)
        with pytest.raises(ValueError, match=r"^directions must be finite"):
            scene.intersect_any(np.zeros((0, 3)), np.array([0.0, bad, 0.0]))


def axis_rays(lo, hi, seed):
    """
    Rays with one or two direction components exactly zero, of either sign,
    along every axis: grids over a box from `lo` to `hi`, origins on its
    planes, an ulp of a double and of a float either side of them, at
    float32's rounding breakpoints, and far enough out that tinybvh's own
    arithmetic overflows.
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    size = hi - lo
    origins, directions = [], []
    planes = np.concatenate([lo, hi, (lo + hi) / 2])
    near = np.concatenate([
        planes,
        np.nextafter(planes, np.inf),
        np.nextafter(planes, -np.inf),
        np.nextafter(planes.astype(np.float32), np.float32(np.inf)).astype(float),
        np.nextafter(planes.astype(np.float32), np.float32(-np.inf)).astype(float),
        planes * (1 + 2.0**-24),
        [0.0, -0.0, 1e-40, -1e-40, 1e-310],
    ])  # fmt: skip
    far = np.array([2.0**24 - 1, 2.0**24, 2.0**24 + 2, 1e9, 1e29])
    for axis in range(3):
        others = [k for k in range(3) if k != axis]
        for sign in (1.0, -1.0):
            for zero in (0.0, -0.0):
                # along the axis, both other components zero
                grid = lo + rng.uniform(-0.2, 1.2, (400, 3)) * size
                grid[:200, others[0]] = rng.choice(near, 200)
                grid[100:300, others[1]] = rng.choice(near, 200)
                grid[:, axis] = lo[axis] - sign * size[axis]
                grid[300:350, others[0]] = rng.choice(far, 50) * rng.choice([-1, 1], 50)
                along = np.full((400, 3), zero)
                along[:, axis] = sign * rng.choice([1.0, 0.5, 3.0, 1e-3, 1e-200], 400)
                origins.append(grid)
                directions.append(along)
                # one zero component, crossing the other two obliquely
                for still in others:
                    oblique = rng.normal(size=(300, 3))
                    oblique[:, still] = zero
                    oblique[250:, others[1 - others.index(still)]] = 2.0**-101
                    start = lo + rng.uniform(-1, 2, (300, 3)) * size
                    start[:150, still] = rng.choice(near, 150)
                    start[280:, still] = rng.choice(far, 20)
                    origins.append(start)
                    directions.append(oblique)
    return np.vstack(origins), np.ascontiguousarray(np.vstack(directions))


AXIS_MESHES = {"box": box(), "far box": box((-3, 2, 1 + 2.0**-24), (-1, 2.5, 7)), "sphere": sphere(24, 48)}


@pytest.mark.parametrize("mesh", AXIS_MESHES.values(), ids=AXIS_MESHES.keys())
def test_device_agrees_on_axis_aligned_rays(mesh, device_required):
    # a zero direction component is decided by where the origin lies between
    # a box's planes, in float32 behind a margin (docs/trace.md 3.6): the same
    # bytes as the CPU on, beside and far from every plane, either sign of zero
    cpu = trimkit.RayScene([mesh])
    gpu = device_scene([mesh], device_required)
    origins, directions = axis_rays(mesh[0].min(axis=0), mesh[0].max(axis=0), 29)
    expected = cpu.intersect(origins, directions)
    assert len(np.unique(expected[0])) > 1000
    same(expected, gpu.intersect(origins, directions))
    nearest = cpu.intersect(origins, directions, maximum_hits=1)
    same(nearest, gpu.intersect(origins, directions, maximum_hits=1))
    same(cpu.intersect_any(origins, directions), gpu.intersect_any(origins, directions))
    # one direction for every ray: an orthographic view down each axis
    for axis in range(3):
        for down in ([0.0, 0.0], [-0.0, 0.0], [-0.0, -0.0]):
            direction = np.insert(down, axis, -1.0)
            same(cpu.intersect(origins, direction), gpu.intersect(origins, direction))


def test_device_agrees_on_axis_aligned_rays_through_instances(device_required):
    # translations and scales keep a zero component zero in an instance's
    # frame, a quarter turn moves it to another axis, any other turn loses it
    meshes = [box(), sphere(12, 24)]
    rng = np.random.default_rng(31)
    matrices = np.tile(np.eye(4), (12, 1, 1))
    matrices[:, :3, 3] = rng.integers(-4, 5, (12, 3)) * 0.75
    matrices[4:8, :3, :3] *= rng.choice([0.5, 2.0, 1.0], (4, 1, 3))
    matrices[8:10, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    angle = 0.3
    matrices[10:, :2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    index = np.arange(12) % 2
    cpu = trimkit.RayScene(meshes)
    gpu = device_scene(meshes, device_required)
    origins, directions = axis_rays([-4, -4, -4], [5, 5, 5], 37)
    expected = cpu.intersect(origins, directions, matrices, index)
    assert len(np.unique(expected[0])) > 500
    same(expected, gpu.intersect(origins, directions, matrices, index))
    blocked = cpu.intersect_any(origins, directions, matrices, index)
    same(blocked, gpu.intersect_any(origins, directions, matrices, index))


def test_rays_are_float64_or_float32(device_required):
    # an array goes to the device as its dtype, float32 or float64, whatever
    # its values: either way the answers are those of the same values as
    # float64, on the CPU and the device, for a row per ray and for one row
    # for every ray
    mesh = sphere(30, 60)
    cpu = trimkit.RayScene([mesh])
    gpu = device_scene([mesh], device_required)
    rng = np.random.default_rng(41)
    count = 30000
    origins = rng.normal(size=(count, 3)) * 2.0
    directions = -origins + rng.normal(size=(count, 3)) * 0.4
    kinds = {
        "float64": lambda a: a,
        "float32 values": lambda a: a.astype(np.float32).astype(np.float64),
        "float32": lambda a: a.astype(np.float32),
    }
    for origin_kind in kinds.values():
        for direction_kind in kinds.values():
            o, d = origin_kind(origins), direction_kind(directions)
            for given in ((o, d), (o[7], d), (o, d[7])):
                wide = tuple(np.asarray(a, dtype=np.float64) for a in given)
                expected = cpu.intersect(*wide)
                assert len(expected[0]) > 100
                for scene in (cpu, gpu):
                    same(expected, scene.intersect(*given))
                    same(cpu.intersect_any(*wide), scene.intersect_any(*given))


def test_device_widens_what_float32_cannot_hold(device_required):
    # float64 values that are exactly float32 until one, late, is not;
    # float32 denormals, which a device may read as zero, so a chunk holding
    # one goes widened; and both zeros: the answers stay
    mesh = box((-1, -1, -1), (1, 1, 1))
    cpu = trimkit.RayScene([mesh])
    gpu = device_scene([mesh], device_required)
    rng = np.random.default_rng(43)
    count = 300_000
    origins = (rng.normal(size=(count, 3)) * 3.0).astype(np.float32).astype(np.float64)
    directions = (-origins + rng.normal(size=(count, 3))).astype(np.float32).astype(np.float64)
    losses = (("origins", 280_000, 0.1), ("directions", 299_999, 1e-45 / 3), ("origins", 5, 1e-40))
    for name, at, value in losses:
        given = {"origins": origins.copy(), "directions": directions.copy()}
        given[name][at, 1] = value
        same(cpu.intersect(**given), gpu.intersect(**given))
    # float32 arrays with denormals and zeros of both signs in them
    narrow = {"origins": origins.astype(np.float32), "directions": directions.astype(np.float32)}
    narrow["origins"][::7, 0] = np.resize(np.float32([1e-40, -1e-40, 0.0, -0.0, 1e-45]), len(origins[::7]))
    narrow["directions"][3::11, 2] = np.float32(-0.0)
    expected = cpu.intersect(narrow["origins"].astype(np.float64), narrow["directions"].astype(np.float64))
    for scene in (cpu, gpu):
        same(expected, scene.intersect(**narrow))
    # and a query of each width after the other, in the same chunks
    wide = (origins + 0.1, directions)
    for given in ((origins, directions), wide, (narrow["origins"], narrow["directions"])):
        same(cpu.intersect_any(*given), gpu.intersect_any(*given))


def test_closed_mesh_is_watertight_on_the_device(device_required):
    # the containment-by-parity property, straight from the device
    vertices, faces = sphere(40, 80)
    scene = device_scene([(vertices, faces)], device_required)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    targets = np.vstack([vertices, vertices[edges].mean(axis=1)])
    origin = np.array([0.1, -0.2, 0.05])
    origins = np.tile(origin, (len(targets), 1))
    hits, _, _, _ = scene.intersect(origins, targets - origin)
    assert np.array_equal(np.bincount(hits, minlength=len(targets)), np.ones(len(targets), dtype=np.int64))


def test_threads_share_a_device_scene(device_required):
    mesh = sphere(30, 60)
    scene = device_scene([mesh], device_required)
    origins, directions = rays(20000, 3)
    expected = scene.intersect(origins, directions)
    results = [None] * 4
    errors = []

    def work(i):
        try:
            results[i] = scene.intersect(origins, directions)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for result in results:
        same(expected, result)


def test_threads_never_deadlock_on_the_device(device_required):
    # One thread runs long queries of one scene while another moves its own
    # scene's instances every query, which frees the top level it replaces
    # with the GIL held. Freeing once waited for the device while the query
    # holding the device waited for the GIL to look for Ctrl-C: both threads
    # stopped for good. A subprocess, so that is a timeout here, not a hang
    device_scene([box()], device_required)
    script = textwrap.dedent(
        """
        import sys, threading, time, warnings
        sys.path.insert(0, sys.argv[1])
        import numpy as np
        import trimkit
        from helpers import sphere
        warnings.simplefilter("error")
        mesh = sphere(60, 120)
        long_scene = trimkit.RayScene([mesh], use_gpu=True)
        moving_scene = trimkit.RayScene([mesh], use_gpu=True)
        rng = np.random.default_rng(0)
        origins = rng.normal(size=(400_000, 3)) * 3.0
        directions = -origins
        places = np.tile(np.eye(4), (2, 2, 1, 1))
        places[1, :, :3, 3] = 0.5
        index = np.zeros(2, dtype=np.int64)
        stop = time.monotonic() + 3.0
        done = {"long": 0, "moving": 0}
        def long_queries():
            while time.monotonic() < stop:
                long_scene.intersect(origins, directions)
                done["long"] += 1
        def moving_queries():
            while time.monotonic() < stop:
                moving_scene.intersect_any(origins[:64], directions[:64], places[done["moving"] % 2], index)
                done["moving"] += 1
        threads = [threading.Thread(target=long_queries), threading.Thread(target=moving_queries)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert done["long"] > 0 and done["moving"] > 0, done
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, os.path.dirname(__file__)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"


def test_use_gpu_is_keyword_only():
    # (what it accepts is covered with every other bool in test_invariants)
    with pytest.raises(TypeError):
        trimkit.RayScene([box()], True)


def test_without_a_device_the_cpu_answers_with_a_warning():
    # In a process whose Vulkan loader finds no driver (its own knob, pointed
    # at nothing: not ours), use_gpu warns once naming why, and every answer is
    # the CPU's. A subprocess, because a process opens its device once
    script = textwrap.dedent(
        """
        import sys, warnings
        sys.path.insert(0, sys.argv[1])
        import numpy as np
        import trimkit
        from helpers import sphere
        mesh = sphere(20, 40)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scene = trimkit.RayScene([mesh], use_gpu=True)
            rng = np.random.default_rng(0)
            origins, directions = rng.normal(size=(1000, 3)), rng.normal(size=(1000, 3))
            answer = scene.intersect(origins, directions)
            again = scene.intersect(origins, directions)
            hit = scene.intersect_any(origins, directions)
        assert len(caught) == 1, [str(w.message) for w in caught]
        message = str(caught[0].message)
        assert message.startswith("use_gpu: no Vulkan"), message
        assert "the CPU" in message, message
        cpu = trimkit.RayScene([mesh])
        for a, b in zip(answer, cpu.intersect(origins, directions), strict=True):
            assert a.tobytes() == b.tobytes()
        assert hit.tobytes() == cpu.intersect_any(origins, directions).tobytes()
        print("ok")
        """
    )
    env = {**os.environ, "VK_ICD_FILENAMES": os.devnull, "VK_DRIVER_FILES": os.devnull}
    result = subprocess.run(
        [sys.executable, "-c", script, os.path.dirname(__file__)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"
