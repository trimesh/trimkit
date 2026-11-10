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
from helpers import box, device_scene, same, sphere

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


def test_device_answers_for_one_origin_or_one_direction(device_required):
    # every ray from one point, every ray one way, and both: the device reads
    # one row for all (the broadcast path), the answers are the CPU's
    mesh = sphere(30, 60)
    cpu = trimkit.RayScene([mesh])
    gpu = device_scene([mesh], device_required)
    origins, directions = rays(20000, 5)
    one_origin = np.tile(origins[:1], (len(origins), 1))
    one_direction = np.tile(directions[:1], (len(directions), 1))
    for o, d in ((one_origin, directions), (origins, one_direction), (one_origin, one_direction)):
        for maximum_hits in (None, 1):
            same(
                cpu.intersect(o, d, maximum_hits=maximum_hits),
                gpu.intersect(o, d, maximum_hits=maximum_hits),
            )
        same(cpu.intersect_any(o, d), gpu.intersect_any(o, d))


def test_device_answers_many_crossings_per_ray(device_required):
    # rays through a stack of twenty boxes cross forty surfaces each: more than
    # the device's first guess at the room an unbounded query needs, so it
    # grows the room and writes again
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
