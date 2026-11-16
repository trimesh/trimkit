import threading

import numpy as np
import pytest
from helpers import ball_runs, is_closed, volume

import trimkit

# the six voxels beside a voxel
BESIDE = np.concatenate([np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)])


def dense_runs(dense: np.ndarray) -> np.ndarray:
    # each filled voxel as a row of its own
    return np.argwhere(dense).astype(np.int64)


def crossings(dense: np.ndarray) -> np.ndarray:
    """
    The midpoint between every filled voxel and each empty voxel beside it,
    sorted: where the surface of an unweighted grid must have its vertices.
    """
    padded = np.pad(dense.astype(bool), 1)
    filled = np.argwhere(padded)
    points = []
    for step in BESIDE:
        beside = filled + step
        empty = ~padded[tuple(beside.T)]
        points.append(filled[empty] + step / 2.0 - 1.0)
    points = np.concatenate(points)
    return points[np.lexsort(points.T)]


def ordered(points: np.ndarray) -> np.ndarray:
    return points[np.lexsort(points.T)]


def test_every_pair_of_cells_closes():
    # Every occupancy of a 2x2x3 block, along each axis: 12,285 grids holding
    # every pair of cells that share a face, in every one of their cases. Each
    # must be closed and wound outward, with a vertex on every crossing: this
    # is what proves the case table src/contour.hpp computes.
    block = np.array([[k & 1, (k >> 1) & 1, k >> 2] for k in range(12)], dtype=np.int64)
    for axis in range(3):
        voxels = np.roll(block, axis, axis=1)
        shape = tuple(voxels.max(axis=0) + 1)
        for bits in range(1, 4096):
            dense = np.zeros(shape, dtype=bool)
            dense[tuple(voxels[[k for k in range(12) if bits >> k & 1]].T)] = True
            vertices, faces, _ = trimkit.contour(dense_runs(dense))
            assert is_closed(faces), (axis, bits)
            assert volume(vertices, faces) > 0.0, (axis, bits)
            assert np.array_equal(ordered(vertices), crossings(dense)), (axis, bits)


def test_random_grids_close_with_a_vertex_per_crossing():
    rng = np.random.default_rng(0)
    for fill in (0.1, 0.5, 0.9):
        for _ in range(20):
            dense = rng.random((6, 7, 8)) < fill
            vertices, faces, vertex_sources = trimkit.contour(dense_runs(dense))
            assert is_closed(faces)
            assert np.array_equal(ordered(vertices), crossings(dense))
            # the surface is inside the voxels' boxes and outside their centers
            assert 0.0 < volume(vertices, faces) < dense.sum()
            # each vertex is half a voxel from the voxel it came from
            offset = vertices - dense_runs(dense)[vertex_sources]
            assert np.array_equal(np.sort(np.abs(offset), axis=1), np.tile([0.0, 0.0, 0.5], (len(offset), 1)))


def test_one_voxel_is_an_octahedron():
    vertices, faces, vertex_sources = trimkit.contour(np.array([[4, -2, 7]]))
    assert len(vertices) == 6 and len(faces) == 8
    assert np.array_equal(ordered(vertices), ordered(np.array([4, -2, 7]) + BESIDE / 2.0))
    assert np.isclose(volume(vertices, faces), 1.0 / 6.0)
    assert vertex_sources.tolist() == [0] * 6


def test_voxels_sharing_only_an_edge_or_corner_stay_apart():
    # one closed octahedron each: nothing joins them
    for other in ([1, 1, 0], [1, 1, 1]):
        vertices, faces, _ = trimkit.contour(np.array([[0, 0, 0], other]))
        assert len(vertices) == 12 and len(faces) == 16
        assert is_closed(faces)
    # sharing a face they are one surface
    vertices, faces, _ = trimkit.contour(np.array([[0, 0, 0], [1, 0, 0]]))
    assert len(vertices) == 10 and is_closed(faces)


def test_runs_are_their_voxels_however_given():
    rng = np.random.default_rng(1)
    dense = rng.random((5, 6, 9)) < 0.6
    single = dense_runs(dense)
    expected = trimkit.contour(single)
    assert np.array_equal(ordered(expected[0]), crossings(dense))

    def same_surface(result):
        assert np.array_equal(ordered(result[0]), ordered(expected[0]))
        assert len(result[1]) == len(expected[1]) and is_closed(result[1])
        assert np.isclose(volume(result[0], result[1]), volume(expected[0], expected[1]))

    # three columns are four with a length of one, exactly
    ones = np.column_stack([single, np.ones(len(single), dtype=np.int64)])
    for got, want in zip(trimkit.contour(ones), expected, strict=True):
        assert np.array_equal(got, want)
    # merged along Z: a run starts where the voxel below is empty
    padded = np.pad(dense, ((0, 0), (0, 0), (1, 1)))
    starts = np.argwhere(padded[:, :, 1:-1] & ~padded[:, :, :-2])
    ends = np.argwhere(padded[:, :, 1:-1] & ~padded[:, :, 2:])
    merged = np.column_stack([starts, ends[:, 2] - starts[:, 2] + 1]).astype(np.int64)
    assert len(merged) < len(single) and merged[:, 3].sum() == dense.sum()
    same_surface(trimkit.contour(merged))
    # in any order
    same_surface(trimkit.contour(rng.permutation(merged)))


def test_a_solid_costs_its_surface():
    # a ball of 113 million voxels as 31 thousand runs: only its shell is visited
    runs = ball_runs(300)
    assert runs[:, 3].sum() > 100_000_000
    vertices, faces, vertex_sources = trimkit.contour(runs)
    assert is_closed(faces)
    assert len(vertices) < 2_000_000
    # the surface cuts the corners of the voxels' boxes
    ratio = volume(vertices, faces) / runs[:, 3].sum()
    assert 0.99 < ratio < 1.0
    # each vertex's source run holds the voxel half a step inside it
    source = runs[vertex_sources]
    inside_z = (vertices[:, 2] >= source[:, 2] - 0.5) & (vertices[:, 2] <= source[:, 2] + source[:, 3] - 0.5)
    assert inside_z.all()
    assert (np.abs(vertices[:, :2] - source[:, :2]).max(axis=1) <= 0.5).all()


def test_vertex_sources_name_the_inside_run():
    runs = np.array([[0, 0, 0, 3], [1, 0, 1, 1], [5, 5, 5, 2]])
    vertices, _, vertex_sources = trimkit.contour(runs)
    assert set(vertex_sources.tolist()) == {0, 1, 2}
    for vertex, source in zip(vertices, vertex_sources, strict=True):
        x, y, z, length = runs[source]
        # within half a voxel of a voxel of that run
        nearest = np.array([x, y, np.clip(round(vertex[2]), z, z + length - 1)])
        assert np.abs(vertex - nearest).sum() == 0.5


def test_all_ones_weigh_the_same_as_none():
    runs = ball_runs(9)
    weighted = trimkit.contour(runs, weights=np.ones(len(runs)))
    for got, want in zip(weighted, trimkit.contour(runs), strict=True):
        assert np.array_equal(got, want)


def test_weights_move_the_crossing_to_a_half():
    # between a voxel weighing `inside` and one weighing `outside` the vertex
    # sits where the line between them passes a half
    for inside, outside in ((1.0, 0.0), (0.75, 0.25), (1.0, 0.25), (0.625, 0.0), (1.0, 0.5)):
        runs = np.array([[0, 0, 0], [1, 0, 0]])
        vertices, faces, vertex_sources = trimkit.contour(runs, weights=np.array([inside, outside]))
        assert is_closed(faces)
        assert (vertex_sources == 0).all()
        between = vertices[(vertices[:, 1] == 0) & (vertices[:, 2] == 0) & (vertices[:, 0] > 0)]
        assert between[:, 0].tolist() == [(inside - 0.5) / (inside - outside)]
        # toward empty space, which weighs nothing
        assert vertices[:, 0].min() == -1.0 + 0.5 / inside
    # a half or less is outside: nothing inside, no surface
    for vertices_or_faces in trimkit.contour(np.array([[0, 0, 0]]), weights=np.array([0.5]))[:2]:
        assert len(vertices_or_faces) == 0


def test_a_weighted_ball_meshes_to_its_radius():
    # weights from a ball's distance field: a half exactly on the sphere and
    # falling by a half per voxel, so the clamp to [0, 1] is a whole voxel
    # from the sphere and never on a crossing. Each vertex is then where the
    # distance interpolated along its edge is the radius
    radius = 10.3
    grid = np.argwhere(np.ones((25, 25, 25), dtype=bool)) - 12
    distance = np.linalg.norm(grid, axis=1)
    weights = np.clip(0.5 + (radius - distance) / 2.0, 0.0, 1.0)
    keep = weights > 0.0
    vertices, faces, _ = trimkit.contour(grid[keep].astype(np.int64), weights=weights[keep])
    assert is_closed(faces)
    error = np.abs(np.linalg.norm(vertices, axis=1) - radius)
    # along a unit step the distance to the center bends from linear by at
    # most 1 / (8 d) for the nearest it comes, d, which is within a voxel
    assert error.max() < 1.0 / (8.0 * (radius - 1.0)) + 1e-12
    # against the binary ball, whose crossings are up to half a voxel off
    binary, _, _ = trimkit.contour(grid[distance <= radius].astype(np.int64))
    assert np.abs(np.linalg.norm(binary, axis=1) - radius).max() > 0.4
    assert np.isclose(volume(vertices, faces), 4.0 / 3.0 * np.pi * radius**3, rtol=0.01)


def test_overlapping_runs_are_refused():
    # the same voxel twice, however it is given and whatever it weighs
    for rows in (
        [[0, 0, 0], [0, 0, 0]],
        [[0, 0, 0, 4], [0, 0, 3, 1]],
        [[0, 0, 3, 1], [5, 5, 5, 1], [0, 0, 0, 4]],
        [[0, 0, 0, 4], [0, 0, 1, 2]],
    ):
        runs = np.array(rows)
        for weights in (None, np.zeros(len(runs)), np.linspace(0.0, 1.0, len(runs))):
            with pytest.raises(ValueError, match="must not overlap"):
                trimkit.contour(runs, weights=weights)
    # runs that only touch are apart
    touching = trimkit.contour(np.array([[0, 0, 0, 4], [0, 0, 4, 2], [0, 1, 0, 6]]))
    single = trimkit.contour(np.array([[0, y, z] for y in range(2) for z in range(6)]))
    assert np.array_equal(ordered(touching[0]), ordered(single[0])) and is_closed(touching[1])


def test_a_surface_too_large_is_refused_at_once():
    # as many cells as the run is long: counted without visiting each, then
    # refused before anything that size is made
    for runs in ([[0, 0, 0, 2**50]], [[x, 2 * x, -(2**51), 2**52] for x in range(400)]):
        with pytest.raises(MemoryError):
            trimkit.contour(np.array(runs))


def test_every_vertex_is_ringed_by_one_fan():
    # closed edges still allow two sheets pinched at a vertex: the faces
    # around each vertex must be one ring, whatever the grid
    rng = np.random.default_rng(3)
    for fill in (0.2, 0.5, 0.8):
        _, faces, _ = trimkit.contour(dense_runs(rng.random((7, 6, 8)) < fill))
        # each face gives its vertex the edge opposite: a ring is one cycle
        after = {}
        for a, b, c in faces.tolist():
            for vertex, start, end in ((a, b, c), (b, c, a), (c, a, b)):
                after.setdefault(vertex, {})[start] = end
        for ring in after.values():
            first = next(iter(ring))
            at, steps = ring[first], 1
            while at != first:
                at, steps = ring[at], steps + 1
            assert steps == len(ring)


def test_nothing_gives_nothing():
    for runs in (np.zeros((0, 3), dtype=np.int64), np.zeros((0, 4), dtype=np.int64)):
        vertices, faces, vertex_sources = trimkit.contour(runs)
        assert vertices.shape == (0, 3) and faces.shape == (0, 3) and vertex_sources.shape == (0,)
        assert (vertices.dtype, faces.dtype, vertex_sources.dtype) == (np.float64, np.int64, np.int64)
    # rows weighing nothing are no rows
    vertices, faces, _ = trimkit.contour(np.array([[0, 0, 0]]), weights=np.zeros(1))
    assert len(vertices) == 0 and len(faces) == 0


def test_the_limit_is_exact():
    # the farthest voxels allowed still give exact half-integer vertices
    limit = 2**52
    far = np.array([[limit - 1, -(limit - 1), limit - 3, 2]])
    vertices, faces, _ = trimkit.contour(far)
    assert is_closed(faces) and len(vertices) == 10
    assert vertices[:, 0].max() == limit - 0.5 and vertices[:, 1].min() == -(limit - 0.5)
    assert vertices[:, 2].max() == limit - 1.5
    for bad in ([[limit, 0, 0, 1]], [[0, -limit, 0, 1]], [[0, 0, limit - 1, 2]], [[0, 0, 0, limit + 1]]):
        with pytest.raises(ValueError, match="within 2\\^52"):
            trimkit.contour(np.array(bad))


def test_bad_input_is_refused():
    runs = np.array([[0, 0, 0, 1], [1, 0, 0, 2]])
    for length in (0, -1):
        with pytest.raises(ValueError, match="length of at least 1"):
            trimkit.contour(np.array([[0, 0, 0, length]]))
    for columns in (2, 5):
        with pytest.raises(ValueError, match=r"\(n, 3\) or \(n, 4\)"):
            trimkit.contour(np.zeros((1, columns), dtype=np.int64))
    with pytest.raises(TypeError, match="voxels must be int64"):
        trimkit.contour(np.zeros((2, 4)))
    for dimensions in (1, 4):
        with pytest.raises(TypeError, match="voxels must be int64 rows"):
            trimkit.contour(np.zeros((2,) * dimensions, dtype=np.int64))
    with pytest.raises(TypeError, match="voxels must be"):
        trimkit.contour("not voxels")
    with pytest.raises(ValueError, match="one per row"):
        trimkit.contour(runs, weights=np.ones(3))
    for weight in (np.nan, np.inf, 1e31):
        with pytest.raises(ValueError):
            trimkit.contour(runs, weights=np.array([1.0, weight]))
    for threshold in (-0.25, np.nan, np.inf, 1e31):
        with pytest.raises(ValueError, match="threshold must be at least 0"):
            trimkit.contour(runs, threshold=threshold)
    with pytest.raises(ValueError, match="volume holds its own"):
        trimkit.contour(np.ones((2, 2, 2), dtype=bool), weights=np.ones(8))
    with pytest.raises(ValueError):
        trimkit.contour(np.full((2, 2, 2), np.nan))
    with pytest.raises(TypeError):
        trimkit.contour(runs, np.ones(2))


def test_a_volume_is_its_voxels():
    # a bool volume is the surface of np.argwhere of it, bit for bit, with
    # each source the voxel's index into the flattened volume
    rng = np.random.default_rng(4)
    for shape in ((6, 7, 8), (1, 1, 1), (3, 1, 9), (9, 2, 1)):
        for fill in (0.2, 0.5, 0.9):
            dense = rng.random(shape) < fill
            got = trimkit.contour(dense)
            want = trimkit.contour(np.argwhere(dense))
            assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
            assert np.array_equal(got[2], np.flatnonzero(dense)[want[2]])
            assert is_closed(got[1])
    # whatever its layout
    dense = rng.random((8, 9, 10)) < 0.5
    expected = trimkit.contour(np.ascontiguousarray(dense[::2, 1:, ::-1]))
    for got, want in zip(trimkit.contour(dense[::2, 1:, ::-1]), expected, strict=True):
        assert np.array_equal(got, want)
    fortran = trimkit.contour(np.asfortranarray(dense))
    for got, want in zip(fortran, trimkit.contour(dense), strict=True):
        assert np.array_equal(got, want)
    # nothing gives nothing
    for shape in ((0, 0, 0), (0, 3, 3), (3, 3, 3)):
        vertices, faces, vertex_sources = trimkit.contour(np.zeros(shape, dtype=bool))
        assert vertices.shape == (0, 3) and faces.shape == (0, 3) and vertex_sources.shape == (0,)


def test_a_weight_volume_is_its_weighted_voxels():
    # a float64 volume is its non-zero voxels as weighted rows, bit for bit,
    # however far from the threshold its voxels inside and out are, which
    # are read only for their side
    rng = np.random.default_rng(5)
    for threshold in (0.0, 0.5, 2.0):
        for _ in range(10):
            field = rng.choice([-3.0, 0.0, 0.25, 0.5, 1.0, 4.0], size=(6, 5, 7)) * rng.random((6, 5, 7))
            field[rng.random(field.shape) < 0.3] = 0.0
            got = trimkit.contour(field, threshold=threshold)
            rows = np.argwhere(field != 0.0)
            want = trimkit.contour(rows, weights=field[field != 0.0], threshold=threshold)
            assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
            assert np.array_equal(got[2], np.flatnonzero(field)[want[2]])
            assert is_closed(got[1])
    # a bool volume weighs 1 where it is true
    dense = rng.random((5, 6, 7)) < 0.5
    for got, want in zip(trimkit.contour(dense), trimkit.contour(dense.astype(np.float64)), strict=True):
        assert np.array_equal(got, want)


def test_a_field_meshes_to_its_level_set():
    # a ball's signed distance, unclipped and in any units: the vertices sit
    # where the distance interpolated along each edge is the threshold
    radius = 10.3
    axis = np.arange(-13, 14)
    distance = np.sqrt(axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2)
    for scale in (1.0, 1000.0):
        field = scale * (radius - distance) + 2.0 * scale
        vertices, faces, _ = trimkit.contour(field, threshold=2.0 * scale)
        assert is_closed(faces)
        error = np.abs(np.linalg.norm(vertices - 13, axis=1) - radius)
        assert error.max() < 1.0 / (8.0 * (radius - 1.0)) + 1e-9
        assert np.isclose(volume(vertices, faces), 4.0 / 3.0 * np.pi * radius**3, rtol=0.01)


def test_threshold_moves_the_crossing():
    # between a voxel weighing `inside` and one weighing `outside` the
    # vertex sits where the line between them passes the threshold
    runs = np.array([[0, 0, 0], [1, 0, 0]])
    for inside, outside, threshold in ((4.0, -2.0, 1.0), (1.0, 0.0, 0.0), (0.75, 0.5, 0.6), (3.0, 1.0, 2.5)):
        vertices, faces, vertex_sources = trimkit.contour(
            runs, weights=np.array([inside, outside]), threshold=threshold
        )
        assert is_closed(faces) and (vertex_sources == 0).all()
        between = vertices[(vertices[:, 1] == 0) & (vertices[:, 2] == 0) & (vertices[:, 0] > 0)]
        assert between[:, 0].tolist() == [(threshold - inside) / (outside - inside)]
    # the default is a half
    default = trimkit.contour(ball_runs(5))
    for got, want in zip(trimkit.contour(ball_runs(5), threshold=0.5), default, strict=True):
        assert np.array_equal(got, want)
    # nothing weighs more than a threshold of 1 in a bool volume
    for got in trimkit.contour(np.ones((3, 3, 3), dtype=bool), threshold=1.0)[:2]:
        assert len(got) == 0


def test_a_volume_changed_during_the_call_is_survived():
    # A volume is read in place with the GIL released: a thread rewriting it
    # meanwhile may change the answer or have it refused, but whatever was
    # read is meshed whole, closed and indexing only what it made
    field = np.random.default_rng(6).random((48, 48, 48))
    stop = threading.Event()

    def scribble():
        rng = np.random.default_rng(7)
        values = (np.nan, 0.0, 1.0, 0.75, -1.0)
        while not stop.is_set():
            field[rng.integers(48)] = values[rng.integers(len(values))]

    thread = threading.Thread(target=scribble)
    thread.start()
    try:
        for _ in range(10):
            try:
                vertices, faces, vertex_sources = trimkit.contour(field)
            except ValueError:
                continue
            assert is_closed(faces)
            assert faces.min(initial=0) >= 0 and faces.max(initial=-1) < max(len(vertices), 1)
            assert vertex_sources.min(initial=0) >= 0 and vertex_sources.max(initial=-1) < field.size
    finally:
        stop.set()
        thread.join()
