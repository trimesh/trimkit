import numpy as np
import pytest
from helpers import box, sphere, volume

import trimkit


def test_simplifies_to_the_target_keeping_the_shape():
    vertices, faces = sphere(60, 120)
    kept_vertices, new_faces, vertex_sources = trimkit.simplify(vertices, faces, len(faces) // 10)
    assert 0 < len(new_faces) <= len(faces) // 10
    # a subset of the input vertices, unmoved and each used
    assert len(np.unique(vertex_sources)) == len(vertex_sources)
    assert np.array_equal(np.unique(new_faces), np.arange(len(vertex_sources)))
    assert np.array_equal(kept_vertices, vertices[vertex_sources])
    assert np.isclose(volume(kept_vertices, new_faces), volume(vertices, faces), rtol=0.05)


def test_maximum_error_stops_it_first():
    vertices, faces = sphere(60, 120)
    _, coarse, _ = trimkit.simplify(vertices, faces, 0)
    _, fine, _ = trimkit.simplify(vertices, faces, 0, maximum_error=1e-3)
    _, none, _ = trimkit.simplify(vertices, faces, 0, maximum_error=0.0)
    assert len(coarse) < len(fine) <= len(none)


def test_lock_border_keeps_the_boundary():
    # an open grid: its outline must survive untouched
    side = 30
    x, y = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    vertices = np.column_stack([x.ravel(), y.ravel(), np.sin(3 * x.ravel()) * 0.1])
    corner = (
        np.arange(side - 1, dtype=np.int64)[:, None] * side + np.arange(side - 1, dtype=np.int64)
    ).ravel()
    faces = np.concatenate(
        [
            np.column_stack([corner, corner + side, corner + 1]),
            np.column_stack([corner + 1, corner + side, corner + side + 1]),
        ]
    )
    edge = np.flatnonzero((x.ravel() == 0) | (x.ravel() == 1) | (y.ravel() == 0) | (y.ravel() == 1))
    _, _, vertex_sources = trimkit.simplify(vertices, faces, 50, lock_border=True)
    assert set(edge) <= set(vertex_sources.tolist())


@pytest.mark.parametrize("extent", [2.0**-30, 2.0**90])
def test_power_of_two_scales_are_exact(extent):
    # positions are centered and scaled by a power of two inside, which is
    # exact: the same mesh at any such scale within the 1e30 bound simplifies
    # identically
    vertices, faces = sphere(20, 40)
    reference = trimkit.simplify(vertices, faces, 100)
    scaled = trimkit.simplify(vertices * extent, faces, 100)
    assert np.array_equal(reference[0] * extent, scaled[0])
    assert np.array_equal(reference[1], scaled[1]) and np.array_equal(reference[2], scaled[2])


@pytest.mark.parametrize(("offset", "extent"), [(1e7, 1.0), (-1e12, 1e3), (0.0, 1e-6), (5.0, 1e29)])
def test_any_position_and_scale(offset, extent):
    # far from the origin, tiny, or near the 1e30 bound: float32 rounds
    # differently, and the greedy collapse order follows it, so the same
    # budget rather than the same triangles: nothing collapses to a point
    vertices, faces = sphere(20, 40)
    _, reference_faces, _ = trimkit.simplify(vertices, faces, 100)
    _, moved_faces, _ = trimkit.simplify(vertices * extent + offset, faces, 100)
    assert len(moved_faces) == len(reference_faces)


def test_nothing_to_do():
    vertices, faces = box()
    # a target above the input keeps it, a soup has no shared edges to collapse
    _, kept, _ = trimkit.simplify(vertices, faces, 1000)
    assert len(kept) == len(faces)
    soup = vertices[faces].reshape((-1, 3))
    _, kept, _ = trimkit.simplify(soup, np.arange(len(soup), dtype=np.int64).reshape((-1, 3)), 1)
    assert len(kept) == len(faces)


def test_bad_input():
    vertices, faces = box()
    with pytest.raises(ValueError, match="out of range"):
        trimkit.simplify(vertices, faces + len(vertices), 4)
    with pytest.raises(ValueError, match="face_count"):
        trimkit.simplify(vertices, faces, -1)
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="maximum_error"):
            trimkit.simplify(vertices, faces, 4, maximum_error=bad)
