"""
The same answers everywhere: a corpus of exact inputs (dyadic coordinates,
directions of exactly representable length, meshes from integer lattices)
whose answers are hashed into tests/golden.json by `uv run scripts generate`
and checked on every platform the suite runs on, on the CPU and on a device.
A difference is a change in what the extension computes, not just in speed.

    python tests/test_golden.py --write    rewrite tests/golden.json
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from helpers import box, device_scene

import trimkit

GOLDEN = Path(__file__).with_name("golden.json")


def lattice_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The surface of the integer cube [-n, n]^3 projected to the unit sphere:
    squares and sums are exact, and sqrt and division are correctly rounded
    everywhere, so every vertex is the same double on every platform.
    """
    side = np.arange(-n, n + 1)
    grid = np.stack(np.meshgrid(side, side, side, indexing="ij"), axis=-1).reshape(-1, 3)
    on_surface = (np.abs(grid) == n).any(axis=1)
    points = grid[on_surface].astype(np.float64)
    index = {tuple(p): i for i, p in enumerate(grid[on_surface])}
    faces = []

    def at(face, u, v, i, j):
        axis, sign = face
        c = [0, 0, 0]
        c[axis] = sign
        c[u] = i
        c[v] = j
        return index[tuple(c)]

    # each cube face is a grid of quads, each split into two triangles
    for axis in range(3):
        for sign in (-n, n):
            u, v = [a for a in range(3) if a != axis]
            for i in range(-n, n):
                for j in range(-n, n):
                    face = (axis, sign)
                    a, b = at(face, u, v, i, j), at(face, u, v, i + 1, j)
                    c, d = at(face, u, v, i + 1, j + 1), at(face, u, v, i, j + 1)
                    if (sign > 0) == ((axis % 2) == 0):
                        faces += [[a, b, c], [a, c, d]]
                    else:
                        faces += [[a, c, b], [a, d, c]]
    norms = np.sqrt((points * points).sum(axis=1))
    return points / norms[:, None], np.array(faces, dtype=np.int64)


def exact_directions() -> np.ndarray:
    # integer directions whose length is an integer: hypot is exact on them
    quadruples = [(1, 0, 0), (1, 2, 2), (2, 3, 6), (1, 4, 8), (4, 4, 7), (2, 6, 9), (6, 6, 7), (3, 4, 0)]
    directions = []
    for q in quadruples:
        for perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for signs in ((1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, -1)):
                directions.append([q[perm[k]] * signs[k] for k in range(3)])
    return np.unique(np.array(directions, dtype=np.float64), axis=0)


def exact_rays() -> tuple[np.ndarray, np.ndarray]:
    side = np.arange(-6, 7) / 4.0  # dyadic
    origins = np.stack(np.meshgrid(side, side, side, indexing="ij"), axis=-1).reshape(-1, 3)
    directions = exact_directions()
    # every origin with a direction, cycling through them deterministically
    o = origins
    d = directions[np.arange(len(o)) % len(directions)]
    # and each origin with a second direction, reversed
    return np.vstack([o, o]), np.vstack([d, -directions[(np.arange(len(o)) * 7) % len(directions)]])


def exact_instances() -> tuple[np.ndarray, np.ndarray]:
    # dyadic translations and scales, and exact axis permutations
    matrices = []
    for k in range(12):
        m = np.eye(4)
        m[:3, 3] = [(k % 5) - 2, (k % 3) * 0.5 - 0.75, ((k * 3) % 7) * 0.25 - 0.5]
        m[0, 0] = 0.5 + (k % 4) * 0.25
        if k % 3 == 1:
            m[:3, :3] = m[:3, :3][[1, 2, 0]]  # a permutation of the axes
        if k % 4 == 2:
            m[1, 1] = -1.5
        matrices.append(m)
    return np.array(matrices), (np.arange(12) % 2).astype(np.int64)


def meshes() -> list:
    return [lattice_sphere(3), box((-0.5, -0.5, -0.5), (0.5, 0.75, 1.0))]


def cases(scene) -> dict[str, list[np.ndarray]]:
    """
    Every answer a RayScene of `meshes()` gives the corpus, CPU or device.
    """
    origins, directions = exact_rays()
    matrices, index = exact_instances()
    return {
        "boxes": [scene.boxes().boxes(), scene.boxes(matrices, index).boxes()],
        "intersect": list(scene.intersect(origins, directions)),
        "intersect nearest": list(scene.intersect(origins, directions, maximum_hits=1)),
        "intersect two": list(scene.intersect(origins, directions, maximum_hits=2)),
        "intersect_any": [scene.intersect_any(origins, directions)],
        "instances intersect": list(scene.intersect(origins, directions, matrices, index)),
        "instances nearest": list(scene.intersect(origins, directions, matrices, index, maximum_hits=1)),
        "instances intersect_any": [scene.intersect_any(origins, directions, matrices, index)],
    }


def digest(arrays: list[np.ndarray]) -> dict:
    h = hashlib.sha256()
    for a in arrays:
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return {"arrays": [[str(a.dtype), list(a.shape)] for a in arrays], "sha256": h.hexdigest()}


def golden() -> dict:
    # the CPU's answers, and Boxes over the instances' boxes
    scene = trimkit.RayScene(meshes())
    answers = cases(scene)
    matrices, index = exact_instances()
    tree = trimkit.Boxes(scene.boxes(matrices, index).boxes())
    origins, _ = exact_rays()
    answers["Boxes.intersection"] = [tree.intersection(np.stack([origins - 0.25, origins + 0.25], axis=1))]
    return {name: digest(arrays) for name, arrays in answers.items()}


def test_cpu_answers_are_golden():
    expected = json.loads(GOLDEN.read_text())
    actual = golden()
    assert list(actual) == list(expected)
    for name in expected:
        assert actual[name] == expected[name], f"{name} answers differently than tests/golden.json records"


@pytest.mark.device
def test_device_answers_are_golden(device_required):
    expected = json.loads(GOLDEN.read_text())
    for name, arrays in cases(device_scene(meshes(), device_required)).items():
        assert digest(arrays) == expected[name], f"{name} answers differently on the device"


def test_the_corpus_hits_and_misses():
    # how many hits each case has is pinned by its shape in golden.json; that
    # the corpus has rays on both sides of the answer is what this adds
    hits = cases(trimkit.RayScene(meshes()))["intersect_any"][0]
    assert hits.any() and not hits.all()


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit(__doc__)
    GOLDEN.write_text(json.dumps(golden(), indent=1) + "\n")
    print(f"wrote {GOLDEN}")
