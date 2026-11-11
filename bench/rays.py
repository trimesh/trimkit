"""
Ray queries on the CPU against a Vulkan device, on a real assembly: the
cycloidal drive of tests/models, as loaded and pulled apart by degrees, so
fewer and fewer rays hit anything, in batches from ten thousand rays to four
million.

  python bench/rays.py

A batch is one pinhole camera's rays (one origin for every ray); each time is
the median over cameras orbiting the assembly, in milliseconds a batch, CPU
then device, then the CPU's time over the device's. The device's answers are
checked against the CPU's, byte for byte, before they are timed. Prints a
Markdown table. Needs trimkit and numpy only.
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


if __name__ == "__main__":
    # a scene that can't use a device says so and would bench the CPU twice
    warnings.simplefilter("error")
    headings = " | ".join(f"{query}: CPU ms | device ms | ratio" for query in QUERIES)
    print(f"| rays | pulled apart | hitting | {headings} |")
    print("|---|---|---|" + "---|" * (3 * len(QUERIES)))
    for row in bench():
        cells = " | ".join(f"{t.cpu:.2f} | {t.device:.2f} | {t.cpu / t.device:.1f}x" for t in row.timings)
        print(f"| {row.rays:,} | {row.pose:g}x | {row.hitting:.0%} | {cells} |")
