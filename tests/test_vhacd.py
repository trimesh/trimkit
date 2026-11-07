import numpy as np
import pytest
from helpers import box, is_closed, volume

import trimkit


def l_shape():
    # two overlapping boxes merged into one non-convex closed surface is fiddly,
    # so use disjoint boxes: a decomposition must still cover both
    a = box((0, 0, 0), (3, 1, 1))
    b = box((0, 2, 0), (1, 5, 1))
    return np.vstack([a[0], b[0]]), np.vstack([a[1], b[1] + len(a[0])])


def check_hulls(mesh, vertices, faces, vertex_offsets, face_offsets):
    # every hull is a closed convex surface wound outward, and together they
    # cover the mesh's volume
    assert vertex_offsets[0] == face_offsets[0] == 0
    assert vertex_offsets[-1] == len(vertices) and face_offsets[-1] == len(faces)
    total = 0.0
    for hull in range(len(vertex_offsets) - 1):
        v = vertices[vertex_offsets[hull] : vertex_offsets[hull + 1]]
        f = faces[face_offsets[hull] : face_offsets[hull + 1]]
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
    result = trimkit.decompose(*mesh, resolution=100_000)
    assert len(result[2]) >= 2
    check_hulls(mesh, *result)


def test_threads_change_nothing():
    # V-HACD always uses its thread pool, which must not change what it answers
    mesh = l_shape()
    first = trimkit.decompose(*mesh, resolution=50_000)
    for _ in range(3):
        for a, b in zip(trimkit.decompose(*mesh, resolution=50_000), first, strict=True):
            assert np.array_equal(a, b)


def test_a_coarser_error_is_no_more_hulls():
    mesh = l_shape()
    fine = trimkit.decompose(*mesh, resolution=50_000, maximum_error=0.1)
    coarse = trimkit.decompose(*mesh, resolution=50_000, maximum_error=10.0)
    check_hulls(mesh, *coarse)
    assert len(coarse[2]) <= len(fine[2])


def test_caps_hold():
    mesh = l_shape()
    _, _, vertex_offsets, _ = trimkit.decompose(*mesh, resolution=50_000, maximum_hulls=1, maximum_vertices=8)
    assert len(vertex_offsets) - 1 == 1
    assert np.diff(vertex_offsets).max() <= 8


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
    vertices, faces, vertex_offsets, face_offsets = trimkit.decompose(
        np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    )
    assert len(vertices) == len(faces) == 0
    assert vertex_offsets.tolist() == face_offsets.tolist() == [0]


def test_everything_welded_away():
    # V-HACD welds at 1/1000 of the mesh's scale, which an unused vertex sets
    # here: the tiny tetrahedron collapses entirely, which divided 0 by 0 and
    # cast the NaN to size_t (vendor/patches/vhacd-empty-volume.patch)
    vertices = np.array([[0, 0, 0], [1e-7, 0, 0], [0, 1e-7, 0], [0, 0, 1e-7], [1, 1, 1]], dtype=np.float64)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    _, _, vertex_offsets, _ = trimkit.decompose(vertices, faces)
    assert len(vertex_offsets) - 1 == 0
