"""
Run inside a container with a memory limit, by `uv run scripts test`: an input
that would fill it is a MemoryError, not the kernel's SIGKILL, and what fits
still works. A SIGKILL ends this process, which is what fails the stage.
"""

import contextlib
import pathlib

import numpy as np
from helpers import draco_faces_bomb, draco_points_bomb

import trimkit

decode = trimkit.exchange.draco_decode


def refused(call, *arguments) -> bool:
    try:
        call(*arguments)
    except MemoryError:
        return True
    return False


def main() -> None:
    limit = pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip()
    assert limit != "max", "run this in a container with a memory limit"
    limit = int(limit)
    wanted = np.zeros(1, np.int64)

    # a path past the limit: each `h1` is two bytes of an 80-byte segment and a
    # 16-byte vertex
    lines = "M0 0" + "h1" * (limit // 64)
    assert refused(trimkit.exchange.svg_parse, lines), "a path past the limit parsed"
    del lines

    # past the limit in one request, as 16 float32s a point. From 2**31 points
    # draco's own arithmetic refuses the count, a ValueError touching nothing
    for points in (limit // 16, 1_000_000_000):
        assert refused(decode, draco_points_bomb(points), wanted), f"{points} points decoded in {limit} bytes"
    assert refused(decode, draco_faces_bomb(), wanted), "every face a header can declare decoded"
    # these fill whatever is available, however it is asked for: either answer
    # will do, as what fails this is being killed
    for points in (limit // 128, limit // 32):
        with contextlib.suppress(MemoryError, ValueError):
            decode(draco_points_bomb(points), wanted)

    # what fits still works, at a quarter of the limit
    kind = np.array([trimkit.exchange.DracoKind.GENERIC], np.int64)
    points = limit // 4 // (4 * 3 * 2 + 4)
    faces = np.arange(3 * (points // 3), dtype=np.int64).reshape((-1, 3))
    value = np.zeros((points, 3), np.float32)
    decoded = decode(trimkit.exchange.draco_encode(faces, [value], kind), wanted)
    assert decoded[1][0].shape == value.shape
    del faces, value, decoded
    segments = limit // 4 // 96
    vertices, parsed = trimkit.exchange.svg_parse("M0 0" + " l1 2" * segments)
    assert len(parsed) == segments and len(vertices) == segments + 1
    print(f"refused or survived every bomb within a limit of {limit} bytes")


main()
