"""
Ray queries on the CPU against a Vulkan device, on a real assembly: the
cycloidal drive of tests/models, as loaded, pulled apart, and with every face
split in four.

  python bench/rays.py            every row, a million rays each
  python bench/rays.py 500        a 500 x 500 view

Each view is a pinhole camera orbiting the assembly (one origin for every
ray, its directions float64 or float32) or an orthographic grid over it (one
direction for every ray). A row is
the best and the median of its views in milliseconds, CPU then device, and
whether the device answered the same bytes. Needs trimkit and numpy only.
"""

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from helpers import camera, cycloidal, exploded, subdivided

import trimkit

VIEWS = 8


@dataclass
class View:
    origins: np.ndarray
    directions: np.ndarray


@dataclass
class Timing:
    best: float
    median: float


def timed(call, views: list[View]) -> Timing:
    """
    The best and median milliseconds of `call(view)` over the views.
    """
    took = np.zeros(len(views))
    for i, view in enumerate(views):
        start = time.perf_counter()
        call(view)
        took[i] = (time.perf_counter() - start) * 1000
    return Timing(best=float(took.min()), median=float(np.median(took)))


def same(a, b) -> bool:
    a, b = (x if isinstance(x, tuple) else (x,) for x in (a, b))
    return all(x.shape == y.shape and x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))


def cameras(bounds: np.ndarray, pixels: int) -> list[View]:
    """
    Pinhole cameras around the middle of `bounds`, looking at it.
    """
    middle, radius = bounds.mean(axis=0), np.linalg.norm(np.ptp(bounds, axis=0)) / 2
    angles = 2 * np.pi * np.arange(VIEWS) / VIEWS
    heights = 0.45 + 0.1 * (np.arange(VIEWS) % 3)
    positions = middle + 2.6 * radius * np.column_stack([np.cos(angles), np.sin(angles), heights])
    return [View(origins=position, directions=camera(position, middle, pixels)) for position in positions]


def grids(bounds: np.ndarray, pixels: int, direction: np.ndarray) -> list[View]:
    """
    The same orthographic grid above `bounds`, looking along `direction`, once per view.
    """
    middle, radius = bounds.mean(axis=0), np.linalg.norm(np.ptp(bounds, axis=0)) / 2
    side = np.linspace(-1, 1, pixels) * radius
    x, y = np.meshgrid(side + middle[0], side + middle[1])
    height = np.full(x.size, bounds[1][2] + radius)
    origins = np.ascontiguousarray(np.column_stack([x.ravel(), y.ravel(), height]))
    return [View(origins=origins, directions=direction)] * VIEWS


def bench(label: str, meshes, matrices, index, pixels: int) -> None:
    cpu = trimkit.RayScene(meshes)
    gpu = trimkit.RayScene(meshes, use_gpu=True)
    bounds = cpu.boxes(matrices, index).bounds
    orbit = cameras(bounds, pixels)
    first = cpu.intersect(orbit[0].origins, orbit[0].directions, matrices, index)
    hit = len(np.unique(first[0])) / pixels**2
    faces = sum(len(meshes[i][1]) for i in index)
    crossings = len(first[0]) / pixels**2
    summary = f"{faces:,} faces placed, {hit:.0%} of a view's rays hit, {crossings:.2f} crossings a ray"
    print(f"\n{label}: {summary}")
    print(f"  {'':34s}{'cpu best / median':>20s}{'device best / median':>24s}   same")
    queries = {
        "nearest": lambda scene, v: scene.intersect(v.origins, v.directions, matrices, index, maximum_hits=1),
        "any hit": lambda scene, v: scene.intersect_any(v.origins, v.directions, matrices, index),
        "every crossing": lambda scene, v: scene.intersect(v.origins, v.directions, matrices, index),
    }
    rows = [(f"camera, {name}", query, orbit) for name, query in queries.items()]
    # the same views with their directions as float32: half the bytes to a device
    narrow = [View(origins=v.origins, directions=v.directions.astype(np.float32)) for v in orbit]
    rows += [(f"camera float32, {name}", queries[name], narrow) for name in ("nearest", "any hit")]
    down, tilted = np.array([0.0, 0.0, -1.0]), np.array([0.013, 0.021, -1.0])
    rows.append(("ortho along an axis, nearest", queries["nearest"], grids(bounds, pixels, down)))
    rows.append(("ortho off axis, nearest", queries["nearest"], grids(bounds, pixels, tilted)))
    for name, query, views in rows:
        # once each first, so neither pays for its first use
        identical = same(query(cpu, views[0]), query(gpu, views[0]))
        on_cpu = timed(lambda v, query=query: query(cpu, v), views)
        on_gpu = timed(lambda v, query=query: query(gpu, v), views)
        print(
            f"  {name:34s}{on_cpu.best:11.1f} /{on_cpu.median:7.1f}{on_gpu.best:15.1f} /{on_gpu.median:7.1f}"
            f"   {identical}   {on_cpu.median / on_gpu.median:.1f}x"
        )


if __name__ == "__main__":
    pixels = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    # a scene that can't use a device says so and would bench the CPU twice
    warnings.simplefilter("error")
    meshes, matrices, index = cycloidal()
    bench("as loaded", meshes, matrices, index, pixels)
    bench("pulled apart 3x", meshes, exploded(meshes, matrices, index, 3.0), index, pixels)
    bench("every face split in 16", [subdivided(subdivided(m)) for m in meshes], matrices, index, pixels)
