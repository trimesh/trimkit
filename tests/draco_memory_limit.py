"""
Run inside a container with a memory limit, by `uv run scripts memory`: a
buffer that would fill it is a MemoryError with nothing touched, not the
kernel's SIGKILL, and a decode that fits still works. A SIGKILL ends this
process, which is what fails the stage.
"""

import pathlib
import resource
import sys

import numpy as np
from helpers import draco_points_bomb

import trimkit


def main() -> None:
    limit = pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip()
    assert limit != "max", "run this in a container with a memory limit"
    limit = int(limit)
    kind = np.array([trimkit.exchange.DracoKind.GENERIC], np.int64)
    wanted = np.zeros(1, np.int64)
    # what fits decodes, at up to a quarter of the limit
    points = limit // 4 // (4 * 3 * 2 + 4)
    faces = np.arange(3 * (points // 3), dtype=np.int64).reshape((-1, 3))
    value = np.zeros((points, 3), np.float32)
    decoded = trimkit.exchange.draco_decode(trimkit.exchange.draco_encode(faces, [value], kind), wanted)
    assert decoded[1][0].shape == value.shape
    del faces, value, decoded
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # from just past the limit, as 16 float32s a point, to everything a buffer can ask for
    for points in (limit // 128, limit // 16, limit, 2**32 - 1):
        buffer = draco_points_bomb(points)
        try:
            trimkit.exchange.draco_decode(buffer, wanted)
        except MemoryError:
            pass
        else:
            sys.exit(f"{points} points decoded in a limit of {limit} bytes")
    grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    assert grew < 16 * 1024, f"refusing touched {grew} KiB"
    print(f"refused every bomb within a limit of {limit} bytes, touching {grew} KiB")


main()
