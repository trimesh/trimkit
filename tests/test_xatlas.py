import numpy as np
import pytest
from helpers import areas2d, sphere

import trimkit


def test_parametrize_sphere():
    vertices, faces = sphere()
    vertex_sources, new_faces, uv = trimkit.parametrize(vertices, faces)

    # seams duplicate vertices, but every triangle is exactly where it was
    assert len(new_faces) == len(faces)
    assert np.array_equal(vertices[vertex_sources][new_faces], vertices[faces])
    assert len(vertex_sources) == len(uv) >= len(vertices)
    assert uv.min() >= 0.0 and uv.max() <= 1.0

    # no triangle collapses to zero area, and charts don't overlap enough to
    # cover more than the unit square (xatlas may mirror a whole chart)
    areas = np.abs(areas2d(uv, new_faces))
    assert areas.min() > 0
    assert areas.sum() <= 1.0


def test_bad_input():
    vertices, faces = sphere(6, 8)
    with pytest.raises(ValueError):
        trimkit.parametrize(vertices, faces + len(vertices))


def test_no_faces_is_empty():
    # without indices xatlas reads vertices as a triangle list: no faces means nothing
    vertex_sources, faces, uv = trimkit.parametrize(np.zeros((9, 3)), np.zeros((0, 3), dtype=np.int64))
    assert len(vertex_sources) == len(faces) == len(uv) == 0


def test_far_from_origin():
    # float32 alone loses a mesh offset by 1e8 (every uv came out at zero):
    # centering first keeps it a full, valid parameterization like the original
    vertices, faces = sphere()
    near = trimkit.parametrize(vertices, faces)
    vertex_sources, new_faces, uv = trimkit.parametrize(vertices + 1e8, faces)
    assert np.array_equal(vertices[vertex_sources][new_faces], vertices[faces])
    assert len(vertex_sources) == len(near[0])
    area = np.abs(areas2d(uv, new_faces))
    assert area.min() > 0
    # packing efficiency varies with rounding, a collapsed atlas has no area at all
    assert np.isclose(area.sum(), np.abs(areas2d(near[2], near[1])).sum(), rtol=0.25)


def test_needle_triangle():
    # found by Atheris: a 4e9 : 255 triangle overflowed xatlas's packing bitmap
    vertices = np.array([[4283629332.0, 1563307566.0, 774778414.0], [255.0, 0, 0], [0, 0, 0]])
    _, faces, uv = trimkit.parametrize(vertices, np.array([[0, 1, 2]], dtype=np.int64))
    assert len(faces) == 1 and uv.min() >= 0 and uv.max() <= 1


@pytest.mark.parametrize("extent", [2.0**-20, 1e-6, 1e-2, 1024.0, 1e6])
def test_any_scale(extent):
    # xatlas's tolerances are absolute: a 1 cm mesh in metres used to get
    # every uv at one point. Scaled by a power of two inside, the answer is
    # the same at every scale
    vertices, faces = sphere(12, 24)
    _, _, reference = trimkit.parametrize(vertices, faces)
    _, _, uv = trimkit.parametrize(vertices * extent, faces)
    assert len(np.unique(uv.round(6), axis=0)) > len(vertices) // 2
    if np.log2(extent).is_integer():
        assert np.array_equal(uv, reference)
