"""
Every function releases the GIL, so calls really do run at once: the same
inputs must give bit-identical answers on 8 threads as they do serially.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from helpers import leaves, run_python, runs_here, small_calls

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
    queries += [
        lambda: solids.distances(matrices, index),
        lambda: solids.collides(matrices, index),
        lambda: solids.collides(matrices, index, return_distance=True, maximum_contacts=4),
    ]
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


def test_exit_while_a_daemon_thread_queries():
    # the interpreter exits while a daemon thread is inside a query big enough
    # to be on the pool: the module's atexit handler stops it at its next
    # poll, before CPython finalizes and ends a thread that takes the GIL
    # (3.12 and 3.13 unwind it on glibc, and end it without unwinding on musl
    # and Windows), so no worker is left inside a job on a vanished stack.
    # Each run must exit cleanly, never crash or hang
    import textwrap

    script = textwrap.dedent(
        """
        import sys, threading, time
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        from helpers import sphere
        import trimkit

        scene = trimkit.RayScene([sphere(60, 120)])
        origins, directions = np.random.default_rng(0).normal(size=(2, 400_000, 3))
        started = threading.Event()

        def query():
            while True:
                started.set()
                scene.intersect(origins, directions)

        threading.Thread(target=query, daemon=True).start()
        started.wait()
        print(time.monotonic(), flush=True)
        """
    )
    import time

    for _ in range(8):
        done = run_python(
            "-c",
            script,
            str(Path(__file__).parent),
            timeout=60,
        )
        exited = time.monotonic()
        assert done.returncode == 0, done.stderr.decode()[-2000:]
        # from the script's last line to the process gone, finalization included:
        # well under the 5 s limit the drain once waited out, with room for a
        # sanitizer's slower finalization (1.6 s under ThreadSanitizer)
        assert exited - float(done.stdout.split()[-1]) < 3.0, "the exit waited on the daemon's query"


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork")
def test_fork_from_inside_a_query():
    # a signal handler runs while a query polls for Ctrl-C, inside the pool
    # with its workers mid-range, and forks: in the child those workers are
    # gone with the items they held. The child's query must raise, never
    # return an answer missing them, and the child's pool must still answer
    import textwrap

    script = textwrap.dedent(
        """
        import os, signal, sys, time
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        from helpers import sphere
        import trimkit

        scene = trimkit.RayScene([sphere(60, 120)])
        origins, directions = np.random.default_rng(0).normal(size=(2, 2_000_000, 3))
        query = lambda: scene.intersect(origins, directions)
        same = lambda answer: all(a.tobytes() == b.tobytes() for a, b in zip(answer, expected, strict=True))
        expected = query()
        # how long a warm query takes: the first starts the workers and faults
        # its pages in, so a fraction of it can land after a warm one returned
        tic = time.perf_counter()
        query()
        took = time.perf_counter() - tic
        forked, child = [], []

        def fork(signum, frame):
            if not forked:
                forked.append(True)
                if os.fork() == 0:
                    child.append(True)

        signal.signal(signal.SIGALRM, fork)
        # the first fork at this fraction of the query, past its first poll; the
        # timer repeats, so one lands inside the query whatever this run's speed
        signal.setitimer(signal.ITIMER_REAL, float(sys.argv[2]) * took, took / 20)
        try:
            outcome = "same" if same(query()) else "different"
        except RuntimeError:
            outcome = "raised"
        signal.setitimer(signal.ITIMER_REAL, 0)
        if not child:
            # the parent: its answer is the answer, and the child exited cleanly
            assert forked, "the timer never fired inside the query"
            assert outcome == "same", outcome
            _, status = os.wait()
            if status != 0:
                # how it ended: an exit code, or minus the signal that killed it
                sys.exit(f"the child ended with {os.waitstatus_to_exitcode(status)}")
            sys.exit(0)
        os.write(1, f"{outcome} {same(query())}".encode())
        os._exit(0)
        """
    )
    outcomes = []
    for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        done = run_python(
            "-W",
            "ignore",
            "-c",
            script,
            str(Path(__file__).parent),
            str(fraction),
            timeout=120,
        )
        assert done.returncode == 0, done.stderr.decode()[-2000:]
        outcome, again = done.stdout.decode().split()
        assert outcome in ("same", "raised"), "the child's query returned an answer missing items"
        assert again == "True", "the child's pool answered differently afterwards"
        outcomes.append(outcome)
    # at least one fork landed inside a published job, or nothing here was tested
    if (__import__("os").cpu_count() or 1) > 1:
        assert "raised" in outcomes, f"no fork landed inside a published job: {outcomes}"


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork")
def test_forked_child_exits_promptly():
    # a parent thread is inside a query when the main thread forks: the
    # child has none of the parent's calls, so its exit waits for none of
    # them (it once waited out the whole drain limit for a call that
    # existed only in the parent)
    import textwrap

    script = textwrap.dedent(
        """
        import os, sys, threading, time
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        from helpers import sphere
        import trimkit

        scene = trimkit.RayScene([sphere(60, 120)])
        origins, directions = np.random.default_rng(0).normal(size=(2, 400_000, 3))
        busy = threading.Event()
        stop = threading.Event()

        def query():
            while not stop.is_set():
                busy.set()
                scene.intersect(origins, directions)

        thread = threading.Thread(target=query)
        thread.start()
        busy.wait()
        time.sleep(0.05)
        slowest = 0.0
        for _ in range(3):
            tic = time.monotonic()
            pid = os.fork()
            if pid == 0:
                sys.exit(0)
            _, status = os.waitpid(pid, 0)
            slowest = max(slowest, time.monotonic() - tic)
            assert status == 0, status
        stop.set()
        thread.join()
        print(f"{slowest:.3f}")
        """
    )
    done = run_python(
        "-W",
        "ignore",
        "-c",
        script,
        str(Path(__file__).parent),
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert float(done.stdout.split()[-1]) < 2.0, f"a forked child took {done.stdout.split()[-1]} s to exit"


@pytest.mark.skipif(not hasattr(__import__("os"), "sched_setaffinity"), reason="needs CPU affinity")
def test_workers_follow_affinity_as_it_widens():
    # the first call runs on one CPU, so it has no workers; once the process
    # may use more CPUs, the next big call after the pool reads them again
    # starts workers for them (the worker count was once fixed at whatever the
    # first call saw)
    import textwrap

    if len(__import__("os").sched_getaffinity(0)) < 2:
        pytest.skip("needs two CPUs")
    script = textwrap.dedent(
        """
        import os, sys, time
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        from helpers import sphere
        import trimkit

        every = os.sched_getaffinity(0)
        # one CPU before anything uses the pool, so no worker has ever started
        os.sched_setaffinity(0, {min(every)})
        scene = trimkit.RayScene([sphere(60, 120)])
        origins, directions = np.random.default_rng(0).normal(size=(2, 400_000, 3))
        scene.intersect(origins, directions)
        alone = len(os.listdir("/proc/self/task"))
        os.sched_setaffinity(0, every)
        # the CPUs allowed are read at most every 100 ms (CAP_REFRESH_NS)
        time.sleep(0.15)
        scene.intersect(origins, directions)
        print(alone, len(os.listdir("/proc/self/task")))
        """
    )
    done = run_python(
        "-W",
        "ignore",
        "-c",
        script,
        str(Path(__file__).parent),
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    alone, widened = map(int, done.stdout.split())
    assert widened > alone, f"{widened} threads after widening, {alone} on one CPU"
