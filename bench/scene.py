"""
One real assembly as both scenes: the cycloidal drive of tests/models, 13
meshes placed 72 times, as loaded and pulled apart by degrees, so less and
less of it is hit or touching.

  python bench/scene.py

Rays, on the CPU against a Vulkan device, in batches from ten thousand rays to
four million. A batch is one pinhole camera's rays (one origin for every ray);
each time is the median over cameras orbiting the assembly, in milliseconds a
batch, CPU then device, then the CPU's time over the device's. The device's
answers are checked against the CPU's, byte for byte, before they are timed.

Collision, between every pair of the 72 instances at once: which touch, which
come within a millimetre, how far apart those within ten millimetres are, and
how far apart every pair is, each the best of five calls in milliseconds.

Prints two Markdown tables. Needs trimkit and numpy only.
"""

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from helpers import camera, cycloidal, exploded, same

import trimkit

# a batch is SIDE x SIDE rays
SIDES = (100, 316, 1000, 2000)
POSES = (1.0, 2.0, 5.0)
CAMERAS = 8
QUERIES = ("nearest", "any hit", "every crossing")


@dataclass
class Timing:
    cpu: float
    device: float


@dataclass
class Touching:
    pose: float
    pairs: int
    touching: int
    collides: float
    near: float
    bounded: float
    every: float


@dataclass
class Row:
    rays: int
    pose: float
    hitting: float
    timings: list[Timing]


def batches(bounds: np.ndarray, side: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    The (origin, directions) of pinhole cameras around the middle of `bounds`, looking at it.
    """
    middle, radius = bounds.mean(axis=0), np.linalg.norm(np.ptp(bounds, axis=0)) / 2
    angles = 2 * np.pi * np.arange(CAMERAS) / CAMERAS
    heights = 0.45 + 0.1 * (np.arange(CAMERAS) % 3)
    positions = middle + 2.6 * radius * np.column_stack([np.cos(angles), np.sin(angles), heights])
    return [(position, camera(position, middle, side)) for position in positions]


def median_milliseconds(call, rays: list) -> float:
    took = np.zeros(len(rays))
    for i, batch in enumerate(rays):
        start = time.perf_counter()
        call(*batch)
        took[i] = (time.perf_counter() - start) * 1000
    return float(np.median(took))


def bench() -> list[Row]:
    meshes, matrices, index = cycloidal()
    cpu = trimkit.RayScene(meshes)
    device = trimkit.RayScene(meshes, use_gpu=True)
    rows = []
    for side in SIDES:
        for pose in POSES:
            placed = exploded(meshes, matrices, index, pose)
            rays = batches(cpu.boxes(placed, index).bounds, side)
            given = (placed, index)
            queries = {
                "nearest": lambda scene, o, d, given=given: scene.intersect(o, d, *given, maximum_hits=1),
                "any hit": lambda scene, o, d, given=given: scene.intersect_any(o, d, *given),
                "every crossing": lambda scene, o, d, given=given: scene.intersect(o, d, *given),
            }
            timings = []
            for name in QUERIES:
                query = queries[name]
                # once each first, compared: neither is timed on its first use
                same(query(cpu, *rays[0]), query(device, *rays[0]))
                on_cpu = median_milliseconds(lambda o, d, query=query: query(cpu, o, d), rays)
                on_device = median_milliseconds(lambda o, d, query=query: query(device, o, d), rays)
                timings.append(Timing(cpu=on_cpu, device=on_device))
            hitting = float(np.mean([cpu.intersect_any(*batch, placed, index).mean() for batch in rays]))
            rows.append(Row(rays=side * side, pose=pose, hitting=hitting, timings=timings))
    return rows


def best_milliseconds(call, repeats: int = 5) -> float:
    took = np.zeros(repeats)
    for i in range(repeats):
        start = time.perf_counter()
        call()
        took[i] = (time.perf_counter() - start) * 1000
    return float(took.min())


def collisions() -> list[Touching]:
    meshes, matrices, index = cycloidal()
    scene = trimkit.CollisionScene(meshes)
    rows = []
    for pose in POSES:
        given = (exploded(meshes, matrices, index, pose), index)
        rows.append(
            Touching(
                pose=pose,
                pairs=len(index) * (len(index) - 1) // 2,
                touching=len(scene.collides(*given)),
                collides=best_milliseconds(lambda given=given: scene.collides(*given)),
                near=best_milliseconds(lambda given=given: scene.collides(*given, margin=1.0)),
                bounded=best_milliseconds(lambda given=given: scene.distances(*given, upper_bound=10.0)),
                every=best_milliseconds(lambda given=given: scene.distances(*given)),
            )
        )
    return rows


if __name__ == "__main__":
    # a scene that can't use a device says so and would bench the CPU twice
    warnings.simplefilter("error")
    headings = " | ".join(f"{query}: CPU ms | device ms | ratio" for query in QUERIES)
    print(f"| rays | pulled apart | hitting | {headings} |")
    print("|---|---|---|" + "---|" * (3 * len(QUERIES)))
    for row in bench():
        cells = " | ".join(f"{t.cpu:.2f} | {t.device:.2f} | {t.cpu / t.device:.1f}x" for t in row.timings)
        print(f"| {row.rays:,} | {row.pose:g}x | {row.hitting:.0%} | {cells} |")
    print()
    print(
        "| pulled apart | pairs | touching | `collides` ms | within 1 mm ms "
        "| distances within 10 mm ms | every distance ms |"
    )
    print("|---|---|---|---|---|---|---|")
    for touch in collisions():
        counts = f"{touch.pose:g}x | {touch.pairs:,} | {touch.touching}"
        times = f"{touch.collides:.2f} | {touch.near:.2f} | {touch.bounded:.1f} | {touch.every:.0f}"
        print(f"| {counts} | {times} |")
