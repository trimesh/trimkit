import numpy as np
import pytest
from helpers import box, is_closed, leaves, volume

import trimkit


def l_shape():
    # two overlapping boxes merged into one non-convex closed surface is fiddly,
    # so use disjoint boxes: a decomposition must still cover both
    a = box((0, 0, 0), (3, 1, 1))
    b = box((0, 2, 0), (1, 5, 1))
    return np.vstack([a[0], b[0]]), np.vstack([a[1], b[1] + len(a[0])])


def check_hulls(mesh, hulls):
    # a list of (vertices, faces) tuples: every hull is a closed convex surface
    # wound outward, and together they cover the mesh's volume
    assert type(hulls) is list
    total = 0.0
    for hull in hulls:
        assert type(hull) is tuple
        v, f = hull
        assert f.min() >= 0 and f.max() < len(v)
        assert is_closed(f)
        # convex and outward: every vertex is behind every face plane
        normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        distance = np.einsum("fj,vj->fv", normals, v) - np.einsum("fj,fj->f", normals, v[f[:, 0]])[:, None]
        assert distance.max() < 1e-9 * np.abs(normals).max()
        total += volume(v, f)
    assert total >= 0.95 * volume(*mesh)


@pytest.mark.parametrize("mesh", [box(), l_shape()])
def test_hulls_closed_convex_covering(mesh):
    check_hulls(mesh, trimkit.decompose(*mesh, resolution=100_000))


def test_how_many_hulls():
    assert [(len(v), len(f)) for v, f in trimkit.decompose(*box(), resolution=100_000)] == [(8, 12)]
    assert len(trimkit.decompose(*l_shape(), resolution=100_000)) >= 2


def test_hulls_go_straight_into_the_scenes():
    # a hull is a Mesh, so the result is what the scenes take
    hulls = trimkit.decompose(*box(), resolution=100_000)
    origins, directions = np.array([[0.5, 0.5, 5.0]]), np.array([[0.0, 0.0, -1.0]])
    instances, _, distances = trimkit.RayScene(hulls).intersect_first(origins, directions)
    assert instances.tolist() == [0]
    assert distances == pytest.approx([4.0])
    assert trimkit.CollisionScene(hulls, convex=True).boxes().bounds.tolist() == [[0, 0, 0], [1, 1, 1]]


def test_threads_change_nothing():
    # V-HACD always uses its thread pool, which must not change what it answers
    mesh = l_shape()
    first = trimkit.decompose(*mesh, resolution=50_000)
    for _ in range(3):
        for a, b in zip(leaves(trimkit.decompose(*mesh, resolution=50_000)), leaves(first), strict=True):
            assert np.array_equal(a, b)


def test_a_coarser_error_is_no_more_hulls():
    mesh = l_shape()
    fine = trimkit.decompose(*mesh, resolution=50_000, maximum_error=0.1)
    coarse = trimkit.decompose(*mesh, resolution=50_000, maximum_error=10.0)
    check_hulls(mesh, coarse)
    assert len(coarse) <= len(fine)


def test_caps_hold():
    mesh = l_shape()
    hulls = trimkit.decompose(*mesh, resolution=50_000, maximum_hulls=1, maximum_vertices=8)
    assert len(hulls) == 1
    assert len(hulls[0][0]) <= 8


def test_bad_input():
    vertices, faces = box()
    with pytest.raises(ValueError):
        trimkit.decompose(vertices, faces + 100)
    # parameters V-HACD would crash on or silently misread
    for bad in (
        {"resolution": 2**32 - 1},
        {"resolution": 0},
        {"maximum_hulls": 0},
        {"maximum_vertices": 3},
        {"maximum_error": float("nan")},
        {"maximum_error": -1.0},
        {"maximum_error": 100.5},
    ):
        with pytest.raises(ValueError):
            trimkit.decompose(vertices, faces, **bad)
    # empty input decomposes to nothing rather than crashing
    assert trimkit.decompose(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)) == []


def test_everything_welded_away():
    # V-HACD welds at 1/1000 of the mesh's scale, which an unused vertex sets
    # here: the tiny tetrahedron collapses entirely, which divided 0 by 0 and
    # cast the NaN to size_t (vendor/patches/vhacd-empty-volume.patch)
    vertices = np.array([[0, 0, 0], [1e-7, 0, 0], [0, 1e-7, 0], [0, 0, 1e-7], [1, 1, 1]], dtype=np.float64)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    assert trimkit.decompose(vertices, faces) == []
