"""
Every function releases the GIL, so calls really do run at once: the same
inputs must give bit-identical answers on 8 threads as they do serially.
"""

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from helpers import leaves, runs_here, small_calls

CASES = small_calls()


def _flatten(result) -> list[np.ndarray]:
    # every output as raw bytes, so equal means bit-identical
    raw = [a.encode() if isinstance(a, str) else a for a in leaves(result)]
    return [np.frombuffer(a if isinstance(a, bytes) else np.atleast_1d(a), np.uint8) for a in raw]


@pytest.mark.parametrize("name", CASES)
def test_parallel_matches_serial(name):
    runs_here(name)
    call = CASES[name]
    expected = _flatten(call())
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: _flatten(call()), range(32)))
    for result in results:
        assert len(result) == len(expected)
        assert all(np.array_equal(a, b) for a, b in zip(result, expected, strict=True))


def test_free_threaded_stays_free():
    # importing trimkit must not turn the GIL back on
    if not hasattr(sys, "_is_gil_enabled") or sys.flags.gil == 1:
        pytest.skip("not a free-threaded interpreter")
    assert not sys._is_gil_enabled()


def test_shared_scene():
    # one scene queried from many threads at once, with batches big enough
    # that each query also splits across its own worker threads
    from helpers import first_hits, sphere

    import trimkit

    vertices, faces = sphere(30, 60)
    origins, directions = np.random.default_rng(0).normal(size=(2, 20_000, 3))
    scene = trimkit.RayScene([(vertices, faces)])
    expected = _flatten(first_hits(scene, origins, directions))
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: _flatten(first_hits(scene, origins, directions)), range(8)))
    del scene
    for result in results:
        assert all(np.array_equal(a, b) for a, b in zip(result, expected, strict=True))


def test_scheduler_stress():
    # eight threads at once, each mixing queries small enough to run alone
    # and big enough to wake the pool's workers: every caller shares the same
    # workers through parallel.hpp's slots, and still gets exactly the answer
    # it gets alone
    from helpers import first_hits, sphere

    import trimkit

    rng = np.random.default_rng(1)
    corners = rng.uniform(0, 100, (20_000, 3))
    tree = trimkit.Boxes(np.stack([corners, corners + 1.0], axis=1))
    scene = trimkit.RayScene([sphere(30, 60)])
    solids = trimkit.CollisionScene([sphere(8, 16)], convex=True)
    matrices = np.tile(np.eye(4), (120, 1, 1))
    matrices[:, :3, 3] = rng.uniform(0, 10, (120, 3))
    index = np.zeros(120, np.int64)
    queries = []
    for size in (10, 20_000):
        boxes = rng.uniform(0, 100, (size, 3))
        rays = rng.normal(size=(2, size, 3)) * 0.1
        queries += [
            lambda b=boxes: tree.intersection(np.stack([b, b + 3.0], axis=1)),
            lambda r=rays: first_hits(scene, *r),
            lambda r=rays: scene.intersect(*r),
        ]
    queries += [lambda: solids.distances(matrices, index), lambda: solids.collides(matrices, index)]
    expected = [_flatten(query()) for query in queries]

    def mixed(seed: int) -> bool:
        order = np.random.default_rng(seed).permutation(len(queries) * 3) % len(queries)
        return all(
            all(np.array_equal(a, b) for a, b in zip(_flatten(queries[q]()), expected[q], strict=True))
            for q in order
        )

    with ThreadPoolExecutor(8) as pool:
        assert all(pool.map(mixed, range(16)))


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork")
def test_fork_while_threads_swap_placements():
    # threads keep swapping one scene's remembered placement while the main
    # thread forks again and again: every child queries that same scene and
    # gets the parent's answer, never stuck on a lock a vanished thread held
    import threading

    from helpers import box, first_hits, sphere

    import trimkit

    rng = np.random.default_rng(11)
    meshes = [sphere(10, 20), box()]
    index = rng.integers(0, 2, 20)
    placements = []
    for _ in range(2):
        matrices = np.tile(np.eye(4), (20, 1, 1))
        matrices[:, :3, 3] = rng.uniform(-10, 10, (20, 3))
        placements.append(matrices)
    # few enough rays that a child answers on its own thread alone
    origins = rng.uniform(-15, 15, (32, 3))
    directions = placements[0][rng.integers(0, 20, 32), :3, 3] - origins
    scene = trimkit.RayScene(meshes)
    expected = first_hits(scene, origins, directions, placements[0], index)
    stop = threading.Event()

    def churn(k):
        while not stop.is_set():
            first_hits(scene, origins, directions, placements[k % 2], index)

    threads = [threading.Thread(target=churn, args=(k,)) for k in range(4)]
    for thread in threads:
        thread.start()
    try:
        for _ in range(20):
            child = lambda: first_hits(scene, origins, directions, placements[0], index)  # noqa: E731
            answer = _forked_answer(child, expected)
            assert answer == b"1", "the child hung" if answer == b"" else "the child answered differently"
    finally:
        stop.set()
        for thread in threads:
            thread.join()


def _forked_answer(query, expected, seconds: float = 30.0) -> bytes:
    # b"1" if a forked child's `query` gives exactly `expected`, b"0" if not,
    # b"" if it hangs (it is then killed)
    import os
    import select
    import warnings

    read, write = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        try:
            same = all(a.tobytes() == b.tobytes() for a, b in zip(query(), expected, strict=True))
            os.write(write, b"1" if same else b"0")
        finally:
            os._exit(0)
    os.close(write)
    try:
        ready, _, _ = select.select([read], [], [], seconds)
        if not ready:
            os.kill(pid, 9)
        return os.read(read, 1) if ready else b""
    finally:
        os.close(read)
        os.waitpid(pid, 0)
