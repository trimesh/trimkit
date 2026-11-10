import subprocess
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from helpers import box, check_boxes, first_hits, flatten, ray_scene, same, sphere, transform

import trimkit


def brute_force(vertices, faces, origins, directions):
    """
    Möller-Trumbore for every ray against every triangle, from either side:
    the (r, m) distance to each triangle hit ahead of its origin, else inf.
    """
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    e1, e2 = b - a, c - a
    p = np.cross(directions[:, None], e2[None])
    det = np.einsum("rmk,mk->rm", p, e1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = origins[:, None] - a[None]
        u = np.einsum("rmk,rmk->rm", s, p) / det
        q = np.cross(s, e1[None])
        v = np.einsum("rk,rmk->rm", directions, q) / det
        t = np.einsum("rmk,mk->rm", q, e2) / det
    hit = (det != 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return np.where(hit, t, np.inf)


def edge_distance(vertices, triangle, origin, direction) -> float:
    """
    How far inside its triangle a ray crosses the triangle's plane, in
    barycentric units: 0 on an edge or vertex.
    """
    a, b, c = vertices[triangle]
    e1, e2 = b - a, c - a
    p = np.cross(direction, e2)
    det = p @ e1
    s = origin - a
    u = (s @ p) / det
    v = (direction @ np.cross(s, e1)) / det
    return min(abs(u), abs(v), abs(1 - u - v))


def exact_t(vertices, triangle, origin, direction) -> Fraction:
    """
    The distance along a ray to a triangle's plane in exact arithmetic.
    """
    a, b, c = ([Fraction(x) for x in vertices[i]] for i in triangle)
    o, d = [Fraction(x) for x in origin], [Fraction(x) for x in direction]
    e1, e2 = [b[k] - a[k] for k in range(3)], [c[k] - a[k] for k in range(3)]
    n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
    return sum(n[k] * (a[k] - o[k]) for k in range(3)) / sum(n[k] * d[k] for k in range(3))


def meshes():
    rng = np.random.default_rng(0)
    vertices, faces = sphere(12, 24)
    soup = rng.uniform(-1, 1, (300, 3))
    yield "sphere", vertices, faces
    yield "soup", soup, np.arange(300, dtype=np.int64).reshape((-1, 3))
    # far from the origin, where float32 (Embree) has 8 m of resolution
    yield "offset", vertices + 1e8, faces


def scene_of(vertices, faces):
    """
    A single mesh as rays see it: a scene with one identity instance.
    """
    return ray_scene([(vertices, faces)])


def parameters(distances, directions):
    """
    Distances along each ray back in units of its direction, as brute force
    and exact arithmetic measure them.
    """
    return distances / np.linalg.norm(directions, axis=1)


@pytest.mark.parametrize("name, vertices, faces", list(meshes()))
def test_first_hits_match_brute_force(name, vertices, faces):
    rng = np.random.default_rng(1)
    center = vertices.mean(axis=0)
    # rays from a shell around the mesh at points scattered across it
    origins = center + rng.normal(size=(2000, 3)) * 3
    directions = center + rng.normal(size=(2000, 3)) * 0.6 - origins

    instances, faces_hit, distances = first_hits(scene_of(vertices, faces), origins, directions)
    t = parameters(distances, directions)
    every = brute_force(vertices, faces, origins, directions)
    best = every.min(axis=1)
    miss = np.isinf(best)
    assert miss.any() and not miss.all()
    assert np.array_equal(faces_hit == -1, miss) and np.array_equal(instances == -1, miss)
    assert (instances[~miss] == 0).all()
    assert np.isinf(distances[miss]).all()
    # every distance is within a few ulp of the exact one to its triangle
    # (11 is the worst measured, even 1e8 from the origin, plus one for the
    # conversion to a true distance and back), which the reference here,
    # Moller-Trumbore in float64, only manages to about 100
    hit = np.nonzero(~miss)[0]
    eps = np.finfo(float).eps
    for i in hit[:: max(1, len(hit) // 300)]:
        exact = exact_t(vertices, faces[faces_hit[i]], origins[i], directions[i])
        assert abs(Fraction(t[i]) - exact) <= 20 * eps * abs(exact)
    # and it is the nearest hit, its triangle really hit there: a different
    # face from the brute force could only be a tie on a shared edge
    assert np.allclose(t[hit], best[hit], rtol=1e-12, atol=0)
    assert np.allclose(every[hit, faces_hit[hit]], t[hit], rtol=1e-12, atol=0)


def test_distances_are_true_distances():
    # directions of every length give the same hits at the same distance
    vertices, faces = sphere(20, 40)
    scene = scene_of(vertices, faces)
    rng = np.random.default_rng(2)
    origins = rng.normal(size=(500, 3))
    origins *= 4 / np.linalg.norm(origins, axis=1)[:, None]
    unit = -origins / np.linalg.norm(origins, axis=1)[:, None]
    _, faces_unit, d_unit = first_hits(scene, origins, unit)
    scaled = unit * rng.uniform(1e-3, 1e3, (500, 1))
    _, faces_scaled, d_scaled = first_hits(scene, origins, scaled)
    assert np.array_equal(faces_unit, faces_scaled)
    assert np.allclose(d_unit, d_scaled, rtol=1e-12, atol=0)
    # aimed at the center of a unit sphere from 4 away: exactly the nearest
    # triangle a brute force over every triangle finds
    assert np.allclose(d_unit, brute_force(vertices, faces, origins, unit).min(axis=1), rtol=1e-12, atol=0)


def test_closed_mesh_is_watertight():
    # from inside a closed mesh, a ray aimed exactly at any vertex or the
    # middle of any edge must hit: with Moller-Trumbore, or with unpadded
    # traversal, about one in eight of these escaped through the cracks
    vertices, faces = sphere(40, 80)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    targets = np.vstack([vertices, vertices[edges].mean(axis=1)])
    scene = scene_of(vertices, faces)
    for origin in np.random.default_rng(0).uniform(-0.3, 0.3, (4, 3)):
        origins = np.tile(origin, (len(targets), 1))
        directions = targets - origin
        _, faces_hit, distances = first_hits(scene, origins, directions)
        assert (faces_hit >= 0).all()
        assert scene.intersect_any(origins, directions).all()
        # and on the way out each hits at most a hair past its target
        assert (parameters(distances, directions) <= 1 + 1e-9).all()


def test_empty_and_degenerate():
    vertices, faces = sphere(6, 8)
    origins, directions = np.zeros((3, 3)), np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 1]])
    # no faces hits nothing, and a ray without a direction goes nowhere
    empty = scene_of(vertices, np.zeros((0, 3), dtype=np.int64))
    instances, faces_hit, distances = first_hits(empty, origins, directions)
    assert instances.tolist() == faces_hit.tolist() == [-1, -1, -1] and np.isinf(distances).all()
    assert not empty.intersect_any(origins, directions).any()
    assert all(len(a) == 0 for a in empty.intersect(origins, directions))
    scene = scene_of(vertices, faces)
    instances, faces_hit, distances = first_hits(scene, origins, directions)
    assert faces_hit[1] == -1 and faces_hit[0] >= 0 and faces_hit[2] >= 0
    assert scene.intersect_any(origins, directions).tolist() == [True, False, True]
    rays, *_ = scene.intersect(origins, directions)
    assert 1 not in rays


def test_bad_input():
    vertices, faces = sphere(6, 8)
    with pytest.raises(ValueError, match=r"meshes\[0\]\[1\] index a vertex out of range"):
        trimkit.RayScene([(vertices, faces + len(vertices))])
    with pytest.raises(ValueError, match=r"meshes\[0\]\[0\].*finite"):
        trimkit.RayScene([(np.where(vertices == vertices.max(), np.nan, vertices), faces)])
    for mesh in ((vertices,), (vertices, faces, faces), vertices):
        with pytest.raises(TypeError, match=r"meshes\[0\] must be a \(vertices, faces\) pair"):
            trimkit.RayScene([mesh])
    # int32 faces cast safely to int64: converted with a warning, the same scene
    expected = trimkit.RayScene([(vertices, faces)]).boxes().boxes()
    with pytest.warns(UserWarning, match=r"meshes\[0\]\[1\]: int32 converted to int64"):
        converted = trimkit.RayScene([(vertices, faces.astype(np.int32))])
    assert np.array_equal(converted.boxes().boxes(), expected)
    # float64 faces would lose values: refused
    with pytest.raises(TypeError, match=r"meshes\[0\]\[1\]"):
        trimkit.RayScene([(vertices, faces.astype(np.float64))])
    scene = scene_of(vertices, faces)
    for query in (scene.intersect, scene.intersect_any):
        with pytest.raises(ValueError, match="same number"):
            query(np.zeros((2, 3)), np.ones((3, 3)))
        with pytest.raises(ValueError, match="finite"):
            query(np.full((1, 3), np.inf), np.ones((1, 3)))


def test_mesh_owns_its_geometry():
    # a mesh copies its inputs: changing them afterwards changes nothing
    vertices, faces = sphere(8, 16)
    vertices, faces = vertices.copy(), faces.copy()
    scene = trimkit.RayScene([(vertices, faces)])
    before_boxes = scene.boxes().boxes()
    rays = np.array([[0, 0, 5.0]]), np.array([[0, 0, -1.0]])
    before = first_hits(scene, *rays)
    vertices[:] = 1e6
    faces[:] = 0
    after = first_hits(scene, *rays)
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    assert np.array_equal(scene.boxes().boxes(), before_boxes)


def test_flat_mesh_with_spatial_splits():
    # over 100k triangles takes the spatial-split builder, and a flat grid
    # makes every node flat on z: that binned with 0 * inf, NaN cast to int
    # (UBSan aborts on it in the sanitizer build), until a vendored patch
    side = 230
    i, j = np.meshgrid(
        np.arange(side - 1, dtype=np.int64), np.arange(side - 1, dtype=np.int64), indexing="ij"
    )
    a = (i * side + j).ravel()
    faces = np.vstack(
        [np.column_stack([a, a + side, a + 1]), np.column_stack([a + 1, a + side, a + side + 1])]
    )
    assert len(faces) >= 100_000
    x, y = np.meshgrid(np.arange(side, dtype=float), np.arange(side, dtype=float), indexing="ij")
    vertices = np.column_stack([x.ravel(), y.ravel(), np.zeros(side * side)])
    rng = np.random.default_rng(0)
    origins = np.column_stack([rng.uniform(0, side - 1, (5000, 2)), rng.uniform(1, 10, 5000)])
    _, faces_hit, distances = first_hits(
        scene_of(vertices, faces), origins, np.tile([0.0, 0.0, -1.0], (5000, 1))
    )
    # straight down onto z = 0, every ray lands at its own height to an ulp
    assert (faces_hit >= 0).all()
    assert (np.abs(distances - origins[:, 2]) <= np.spacing(origins[:, 2])).all()


def _transforms(rng, k):
    # random rotations, non-uniform scales and translations
    out = np.tile(np.eye(4), (k, 1, 1))
    for m in out:
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        m[:3, :3] = q * np.sign(np.diag(r)) * rng.uniform(0.3, 2.0, 3)
        m[:3, 3] = rng.uniform(-20, 20, 3)
    return out


def test_defaults_and_fast_path():
    # a bare query is each mesh once where it is: one identity instance,
    # traced without a top level, the same as spelling it out, and as two
    # instances that need one
    mesh = sphere(16, 32)
    rng = np.random.default_rng(5)
    origins = rng.normal(size=(3000, 3)) * 3
    directions = rng.normal(size=(3000, 3)) * 0.3 - origins
    far = np.eye(4)
    far[0, 3] = 1000.0  # a second instance well out of the way
    one, two = np.eye(4)[None], np.stack([np.eye(4), far])
    scene = ray_scene([mesh])
    check_boxes(
        scene.boxes(two, np.array([0, 0], dtype=np.int64)), [mesh], two, np.array([0, 0], dtype=np.int64)
    )
    assert len(scene) == 1
    for query in (scene.intersect, scene.intersect_any):
        expected = query(origins, directions)
        for placed in (
            (one, np.array([0], dtype=np.int64)),
            (one, None),
            (two, np.array([0, 0], dtype=np.int64)),
        ):
            same(query(origins, directions, *placed), expected)
    # matrices alone default the meshes, one each
    pair = ray_scene([mesh, mesh], two)
    same(
        first_hits(pair, origins, directions, two),
        first_hits(scene, origins, directions, two, np.zeros(2, np.int64)),
    )
    # and every way of spelling the same instances places them the same
    assert np.array_equal(scene.boxes().boxes(), scene.boxes(one, np.array([0], dtype=np.int64)).boxes())


def test_remembered_instances_answer_the_same():
    # a scene builds a top level from exactly the instance arrays a query
    # gives and remembers the last: asking again, asking something else in
    # between, or changing the same array in place, every answer is the one
    # a fresh scene gives
    rng = np.random.default_rng(9)
    meshes = [sphere(10, 20), box()]
    index = rng.integers(0, 2, 40)
    first, second = _transforms(rng, 40), _transforms(rng, 40)
    origins = rng.uniform(-30, 30, (5000, 3))
    directions = first[rng.integers(0, 40, 5000), :3, 3] - origins
    scene = ray_scene(meshes, first, index)

    def fresh(matrices):
        return trimkit.RayScene(meshes).intersect(origins, directions, matrices, index)

    expected = fresh(first)
    other = fresh(second)
    assert expected[0].size and not all(np.array_equal(x, y) for x, y in zip(expected, other, strict=True))
    same(scene.intersect(origins, directions, first, index), expected)
    same(scene.intersect(origins, directions, first, index), expected)
    same(scene.intersect(origins, directions, second, index), other)
    same(scene.intersect(origins, directions, first.copy(), index), expected)
    # the same array, its values changed: the bytes say it moved
    moving = first.copy()
    same(scene.intersect(origins, directions, moving, index), expected)
    moving[...] = second
    same(scene.intersect(origins, directions, moving, index), other)
    # another index over the same matrices, and None for each mesh once
    swapped = 1 - index
    same(
        scene.intersect(origins, directions, first, swapped),
        trimkit.RayScene(meshes).intersect(origins, directions, first, swapped),
    )
    same(
        first_hits(scene, origins, directions),
        first_hits(trimkit.RayScene(meshes), origins, directions),
    )


def test_remembered_instances_from_many_threads():
    # threads alternating placements on one scene, each swapping what it
    # remembers under the others: every answer is the serial one
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(10)
    meshes = [sphere(12, 24), box()]
    index = rng.integers(0, 2, 30)
    placements = [_transforms(rng, 30) for _ in range(3)]
    origins = rng.uniform(-30, 30, (2000, 3))
    directions = placements[0][rng.integers(0, 30, 2000), :3, 3] - origins
    scene = ray_scene(meshes, placements[0], index)
    expected = [first_hits(trimkit.RayScene(meshes), origins, directions, m, index) for m in placements]
    with ThreadPoolExecutor(8) as pool:
        results = list(
            pool.map(
                lambda k: (k % 3, first_hits(scene, origins, directions, placements[k % 3], index)),
                range(48),
            )
        )
    for k, result in results:
        same(result, expected[k])


def test_instances_match_flattened_scene():
    # instancing is only a way to store a scene: the same rays cast at every
    # instance's transformed copy, concatenated into one mesh, must agree
    rng = np.random.default_rng(3)
    meshes = [sphere(10, 20), box(), sphere(6, 8)]
    index = rng.integers(0, len(meshes), 60)
    transforms = _transforms(rng, len(index))
    scene = ray_scene(meshes, transforms, index)
    vertices, faces, face_starts = flatten(meshes, transforms, index)
    # rays from all over aimed near random instances, so about half hit
    origins = rng.uniform(-40, 40, (20000, 3))
    directions = (
        transforms[rng.integers(0, len(index), 20000), :3, 3] + rng.normal(size=(20000, 3)) * 2 - origins
    )

    instances, faces_hit, distances = first_hits(scene, origins, directions, transforms, index)
    flattened = scene_of(vertices, faces)
    _, flat, d_flat = first_hits(flattened, origins, directions)
    miss = flat == -1
    assert miss.any() and not miss.all()
    assert np.array_equal(instances == -1, miss) and np.array_equal(faces_hit == -1, miss)
    assert np.isinf(distances[miss]).all()
    # the same distance, computed once in each instance's own frame
    assert np.allclose(distances[~miss], d_flat[~miss], rtol=1e-12, atol=0)
    # and the same triangle, unless another ties it at that distance: then
    # the flattened scene hits the instanced one's triangle there too
    mapped = np.where(miss, -1, face_starts[instances] + faces_hit)
    for ray in np.flatnonzero(mapped != flat):
        _, _, tied, at = flattened.intersect(origins[ray : ray + 1], directions[ray : ray + 1])
        assert mapped[ray] in tied[np.isclose(at, d_flat[ray], rtol=1e-12, atol=0)]
    assert np.array_equal(scene.intersect_any(origins, directions, transforms, index), ~miss)


def test_instances_are_watertight():
    # from inside a rotated, scaled closed mesh, rays aimed exactly at its
    # transformed vertices and edge midpoints all hit, as without instancing
    vertices, faces = sphere(30, 60)
    rng = np.random.default_rng(4)
    transforms = _transforms(rng, 3)
    index = np.zeros(3, dtype=np.int64)
    scene = ray_scene([(vertices, faces)], transforms, index)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    local = np.vstack([vertices, vertices[edges].mean(axis=1)])
    for m in transforms:
        targets = np.ascontiguousarray(transform(local, m))
        origin = m[:3, 3]  # the transformed center of the sphere
        origins = np.tile(origin, (len(targets), 1))
        instances, _, distances = first_hits(scene, origins, targets - origin, transforms, index)
        assert (instances >= 0).all()
        assert (parameters(distances, targets - origin) <= 1 + 1e-9).all()


def test_instances_of_nothing_and_lifetime():
    # an instance of a mesh without faces is never hit, and the rest keep
    # their own index; the scene holds its meshes once the caller lets go
    meshes = [(np.zeros((3, 3)), np.zeros((0, 3), dtype=np.int64)), box()]
    transforms = np.tile(np.eye(4), (3, 1, 1))
    transforms[:, 0, 3] = [0.0, 0.0, 5.0]
    index = np.array([0, 1, 1], dtype=np.int64)
    scene = ray_scene(meshes, transforms, index)
    del meshes
    origins = np.array([[5.5, 0.5, 10.0], [0.5, 0.5, 10.0], [50.0, 0, 0]])
    instances, faces_hit, distances = first_hits(
        scene, origins, np.tile([0, 0, -1.0], (3, 1)), transforms, index
    )
    assert instances.tolist() == [2, 1, -1] and faces_hit[2] == -1
    assert np.allclose(distances[:2], 9.0) and np.isinf(distances[2])
    # with nothing that can be hit, every ray misses
    none = ray_scene([])
    instances, faces_hit, distances = first_hits(none, np.zeros((2, 3)), np.ones((2, 3)))
    assert (instances == -1).all() and (faces_hit == -1).all() and np.isinf(distances).all()


def test_scene_bad_input():
    mesh = box()
    eye = np.eye(4)[None]
    for transforms, index, match in [
        (np.concatenate([eye, eye]), np.array([0], dtype=np.int64), "one row per instance"),
        (eye, np.array([1], dtype=np.int64), "out of range"),
        (eye * np.array([1, 1, 1, 2.0])[:, None], np.array([0], dtype=np.int64), "affine"),
        (eye * np.array([1, 0, 1, 1.0]), np.array([0], dtype=np.int64), "not invertible"),
        (eye * np.nan, np.array([0], dtype=np.int64), "finite"),
    ]:
        # every query, and boxes, place instances alike, refusals included
        scene = trimkit.RayScene([mesh])
        for query in (scene.intersect, scene.intersect_any):
            with pytest.raises(ValueError, match=match):
                query(np.zeros((1, 3)), np.ones((1, 3)), transforms, index)
        with pytest.raises(ValueError, match=match):
            scene.boxes(transforms, index)
        # and a refused placement is never remembered
        same(
            first_hits(scene, np.zeros((1, 3)), np.ones((1, 3))),
            first_hits(trimkit.RayScene([mesh]), np.zeros((1, 3)), np.ones((1, 3))),
        )
    with pytest.raises(ValueError, match="one instance per mesh"):
        first_hits(trimkit.RayScene([mesh]), np.zeros((1, 3)), np.ones((1, 3)), np.concatenate([eye, eye]))
    with pytest.raises(TypeError, match=r"meshes\[1\] must be a \(vertices, faces\) pair, not str"):
        trimkit.RayScene([mesh, "not a mesh"])


def test_all_hits_match_brute_force_and_first():
    rng = np.random.default_rng(6)
    meshes = [sphere(10, 20), box()]
    index = np.array([0, 1, 0, 1], dtype=np.int64)
    transforms = _transforms(rng, 4)
    scene = ray_scene(meshes, transforms, index)
    origins = rng.uniform(-40, 40, (3000, 3))
    directions = transforms[rng.integers(0, 4, 3000), :3, 3] + rng.normal(size=(3000, 3)) - origins
    rays, instances, faces_hit, distances = scene.intersect(origins, directions, transforms, index)
    assert len(rays) > 0
    # sorted by ray, then nearest first
    order = np.lexsort((distances, rays))
    assert np.array_equal(order, np.arange(len(rays)))
    # the first row of each ray is exactly its nearest hit, capped at one
    _, _, first_distances = first_hits(scene, origins, directions, transforms, index)
    heads = np.flatnonzero(np.r_[True, rays[1:] != rays[:-1]])
    assert np.array_equal(first_distances[rays[heads]], distances[heads])
    hit = np.isfinite(first_distances)
    assert np.array_equal(np.sort(rays[heads]), np.flatnonzero(hit))
    # every hit against brute force over each instance's transformed copy
    for k, (g, m) in enumerate(zip(index, transforms, strict=True)):
        vertices = transform(meshes[g][0], m)
        every = brute_force(vertices, meshes[g][1], origins, directions)
        expected = {tuple(row) for row in np.argwhere(np.isfinite(every)).tolist()}
        mine = rays[instances == k], faces_hit[instances == k]
        found = set(zip(*(a.tolist() for a in mine), strict=True))
        # the two disagree only on a ray through a shared edge or vertex: one
        # crossing, so one hit here, where the float64 reference finds every
        # triangle there, or none
        for ray, face in expected ^ found:
            assert edge_distance(vertices, meshes[g][1][face], origins[ray], directions[ray]) < 1e-9
        # and where both hit, at the same distance
        t = parameters(distances[instances == k], directions[mine[0]])
        agreed = np.array([(ray, face) in expected for ray, face in zip(*mine, strict=True)], dtype=bool)
        assert np.allclose(t[agreed], every[mine[0][agreed], mine[1][agreed]], rtol=1e-9, atol=0)


def test_all_hits_cross_closed_meshes_in_pairs():
    # from outside, a ray crosses a closed mesh as often going in as out, once
    # hits at one distance (a shared edge or vertex) count once
    vertices, faces = sphere(20, 40)
    scene = scene_of(vertices, faces)
    rng = np.random.default_rng(7)
    origins = rng.normal(size=(2000, 3))
    origins *= 5 / np.linalg.norm(origins, axis=1)[:, None]
    directions = rng.normal(size=(2000, 3)) * 0.5 - origins
    rays, _, _, distances = scene.intersect(origins, directions)
    distinct = np.unique(np.column_stack([rays, distances]), axis=0)[:, 0].astype(np.int64)
    counts = np.bincount(distinct, minlength=2000)
    assert (counts % 2 == 0).all() and counts.any()


def test_no_leaks_and_shared_meshes():
    # nanobind reports any instance alive at shutdown: build, share, query and
    # drop every kind of object, and a mesh deleted under a scene stays usable
    code = textwrap.dedent(
        """
        import numpy as np
        from helpers import sphere, box
        import trimkit
        mesh = sphere(8, 16)
        other = box()
        scenes = [trimkit.RayScene([mesh]), trimkit.RayScene([mesh, other])]
        del mesh, other
        rays = np.array([[0, 0, 5.0]]), np.array([[0, 0, -1.0]])
        # a remembered placement, replaced, and the default again
        placed = (None, np.array([0, 1, 0], dtype=np.int64))
        for scene, instances in [(scenes[0], ()), (scenes[1], placed), (scenes[1], ())]:
            assert len(scene.intersect(*rays, *instances, maximum_hits=1)[0]) == 1
            scene.intersect(*rays, *instances)
            scene.intersect_any(*rays, *instances)
            scene.boxes(*instances).boxes()
        boxes = trimkit.Boxes(np.zeros((4, 2, 3)))
        boxes.intersect(np.zeros((2, 2, 3)))
        boxes.bounds, boxes.boxes()
        del scenes, boxes
        """
    )
    tests = str(Path(__file__).parent)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=tests, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "leaked" not in result.stderr, result.stderr


def overlapping(boxes, queries):
    """
    Brute force: [query, box] for every overlap, touching included, sorted.
    """
    lo, hi = boxes[:, 0], boxes[:, 1]
    rows = [
        [q, b]
        for q, query in enumerate(queries)
        for b in np.flatnonzero(((lo <= query[1]) & (hi >= query[0])).all(axis=1))
    ]
    return np.array(rows, dtype=np.int64).reshape((-1, 2))


def random_boxes(rng, n, d, scale=1.0):
    corner = rng.uniform(-10, 10, (n, d))
    return np.stack([corner, corner + rng.uniform(0, 3 * scale, (n, d))], axis=1)


@pytest.mark.parametrize("d", [2, 3])
def test_boxes_match_brute_force(d):
    rng = np.random.default_rng(d)
    # sparse enough in either dimension that many queries hit nothing
    boxes = random_boxes(rng, 3000, d, scale=0.15 if d == 2 else 1.0)
    # big enough batches that queries split across threads
    queries = random_boxes(rng, 5000, d, scale=0.5)
    tree = trimkit.Boxes(boxes)
    assert len(tree) == len(boxes)
    pairs = tree.intersect(queries)
    assert pairs.shape[1] == 2 and pairs.dtype == np.int64
    assert np.array_equal(pairs, overlapping(boxes, queries))
    # some but not all queries hit something, so the comparison means something
    assert 0 < len(np.unique(pairs[:, 0])) < len(queries)


def test_boxes_touching_and_degenerate():
    # boxes that only share a face, an edge, a corner, and points, all overlap
    boxes = np.array(
        [[[0, 0, 0], [1, 1, 1]], [[1, 0, 0], [2, 1, 1]], [[1, 1, 0], [2, 2, 1]], [[1, 1, 1], [2, 2, 2]]],
        dtype=np.float64,
    )
    points = np.array([[[0.5, 0.5, 0.5]] * 2, [[5, 5, 5]] * 2], dtype=np.float64)
    tree = trimkit.Boxes(np.concatenate([boxes, points]))
    assert tree.intersect(boxes[:1]).tolist() == [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]
    assert tree.intersect(points).tolist() == [[0, 0], [0, 4], [1, 5]]


def test_boxes_deeply_nested():
    # boxes nested a thousand deep build a deep tree: the traversal stack
    # spills to the heap rather than overflowing
    sizes = 2.0 ** -np.arange(1000, dtype=np.float64)
    bounds = np.stack([np.zeros((1000, 3)), np.repeat(sizes[:, None], 3, axis=1)], axis=1)
    pairs = trimkit.Boxes(bounds).intersect(np.zeros((1, 2, 3)))
    assert pairs[:, 1].tolist() == list(range(1000))


@pytest.mark.parametrize("d", [2, 3])
def test_boxes_round_trip(d):
    # every box back exactly as given, -0.0 included, and their bounds
    boxes = random_boxes(np.random.default_rng(d), 500, d)
    boxes[7, 0, 0] = -0.0
    tree = trimkit.Boxes(boxes)
    copy = tree.boxes()
    assert copy.shape == boxes.shape and copy.tobytes() == boxes.tobytes()
    assert not np.shares_memory(copy, tree.boxes())
    assert np.array_equal(tree.bounds, [boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)])
    assert np.array_equal(trimkit.Boxes(copy).intersect(boxes), tree.intersect(boxes))


def test_boxes_empty():
    tree = trimkit.Boxes(np.zeros((0, 2, 3)))
    assert len(tree) == 0
    assert tree.boxes().shape == (0, 2, 3)
    assert tree.bounds.tolist() == [[np.inf] * 3, [-np.inf] * 3]
    assert trimkit.Boxes(np.zeros((0, 2, 2))).bounds.tolist() == [[np.inf] * 2, [-np.inf] * 2]
    assert tree.intersect(np.zeros((3, 2, 3))).shape == (0, 2)
    pairs = trimkit.Boxes(np.ones((2, 2, 2))).intersect(np.zeros((0, 2, 2)))
    assert pairs.shape == (0, 2)


def test_boxes_bad_input():
    good = np.zeros((1, 2, 3))
    for bounds, match in [
        (np.zeros((1, 2, 4)), r"\(n, 2, 2\) or \(n, 2, 3\)"),
        (np.array([[[1.0, 0, 0], [0, 1, 1]]]), "minimum above its maximum"),
        (good * np.nan, "finite"),
    ]:
        with pytest.raises(ValueError, match=match):
            trimkit.Boxes(bounds)
    with pytest.raises(ValueError, match="dimension"):
        trimkit.Boxes(good).intersect(np.zeros((1, 2, 2)))
    with pytest.raises(TypeError):
        trimkit.Boxes(np.zeros((1, 3, 3)))


@pytest.mark.parametrize("scale", [1e-170, 1e-5, 1.0, 1e5, 1e30])
def test_distance_for_any_direction_length(scale):
    # the direction's length must neither underflow nor overflow on the way
    # to a distance in the scene's units
    vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    scene = scene_of(vertices, np.array([[0, 1, 2]], dtype=np.int64))
    _, faces, distances = first_hits(scene, np.array([[0.0, 0.0, 5.0]]), np.array([[0.0, 0.0, -scale]]))
    assert faces.tolist() == [0]
    assert np.isclose(distances[0], 5.0)


def test_misaligned_arrays_are_copied():
    # C-contiguous at an odd byte offset, where every read in place would be
    # undefined: copied once, aligned, and the same scene as from the original
    vertices, faces = box()
    raw = np.zeros(vertices.nbytes + 1, dtype=np.uint8)
    shifted = np.frombuffer(raw.data, dtype=np.float64, count=vertices.size, offset=1).reshape((-1, 3))
    shifted[...] = vertices
    assert shifted.flags.c_contiguous and not shifted.flags.aligned
    assert np.array_equal(ray_scene([(shifted, faces)]).boxes().boxes(), ray_scene([box()]).boxes().boxes())


def test_transform_inverse_must_be_bounded():
    # invertible, but its inverse (scale 1e31) carries rays into the mesh's
    # frame past the bound, where the triangle test would overflow
    vertices, faces = box()
    tiny = np.tile(np.eye(4), (1, 1, 1))
    tiny[0, :3, :3] *= 1e-31
    with pytest.raises(ValueError, match="not invertible with an inverse within"):
        first_hits(
            trimkit.RayScene([(vertices, faces)]),
            np.zeros((1, 3)),
            np.ones((1, 3)),
            tiny,
            np.zeros(1, np.int64),
        )


def test_instanced_scene_near_the_bound_still_hits():
    # a sphere at 1e29 placed by a 10x transform reaches 1e30: every ray from
    # its center hits, where unbounded inputs used to silently miss
    vertices, faces = sphere(8, 16)
    scaled = np.tile(np.eye(4), (1, 1, 1))
    scaled[0, :3, :3] *= 10.0
    scene = ray_scene([(vertices * 1e29, faces)], scaled)
    directions = np.random.default_rng(0).normal(size=(100, 3))
    _, hit_faces, distances = first_hits(scene, np.zeros((100, 3)), directions, scaled)
    assert (hit_faces >= 0).all() and np.isfinite(distances).all()


def test_empty_mesh_instances_are_checked_too():
    # an instance of a mesh with no faces is never hit, but its transform is
    # still refused when it isn't invertible within the bound
    empty = (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    singular = np.zeros((1, 4, 4))
    singular[0, 3, 3] = 1.0
    with pytest.raises(ValueError, match="not invertible"):
        first_hits(trimkit.RayScene([empty]), np.zeros((1, 3)), np.ones((1, 3)), singular)


def test_scene_boxes_hold_every_instance():
    # random affine placements of several meshes, an empty one among them:
    # each box is between its placed faces and its mesh's box carried along
    rng = np.random.default_rng(8)
    meshes = [
        sphere(10, 20),
        box(),
        (np.zeros((0, 3)), np.zeros((0, 3), np.int64)),
        (box()[0] * 1e6, box()[1]),
    ]
    index = rng.integers(0, len(meshes), 200)
    transforms = _transforms(rng, len(index))
    scene = ray_scene(meshes, transforms, index)
    # boxes of other placements, before and after a query remembers one
    for count in (0, 1, 7):
        check_boxes(scene.boxes(transforms[:count], index[:count]), meshes, transforms[:count], index[:count])
    scene.intersect_any(np.zeros((1, 3)), np.ones((1, 3)), transforms, index)
    check_boxes(scene.boxes(transforms, index), meshes, transforms, index)
    check_boxes(scene.boxes(), meshes)


@pytest.mark.parametrize("x", [1e3, 1e7, 2e7, 1e9, 1e29])
def test_all_hits_on_axis_aligned_rays_far_out(x):
    # a ray along one axis has a zero direction on the others, so a node's
    # planes there are 0 * inf: NaN, which the slab test must drop exactly as
    # tinybvh's does. Kept, every hit past |x| ~1.7e7 was lost
    width = max(1.0, x * 1e-9)  # wide enough to be a triangle at x's precision
    triangle = np.array([[x - width, -1, 0], [x + width, -1, 0], [x, 1, 0]], dtype=np.float64)
    scene = ray_scene([(triangle, np.array([[0, 1, 2]], dtype=np.int64))])
    origins, directions = np.array([[x, 0.0, 10.0]]), np.array([[0.0, 0.0, -1.0]])
    first = first_hits(scene, origins, directions)
    every = scene.intersect(origins, directions)
    assert first[2].tolist() == [10.0]
    ray = np.zeros(1, dtype=np.int64)
    same((every[0], every[1], every[2], every[3]), (ray, first[0], first[1], first[2]))


def test_hit_counts_tell_inside_from_outside():
    # a closed mesh is crossed an odd number of times from inside and an even
    # number from outside, whichever way a ray goes: so hit counts are a
    # containment test. That needs a ray through a shared edge or vertex to
    # cross once there (one hit), not once per triangle meeting there
    vertices, faces = sphere(40, 80)
    scene = scene_of(vertices, faces)
    rng = np.random.default_rng(3)
    points = rng.uniform(-1.5, 1.5, (3000, 3))
    radius = np.linalg.norm(points, axis=1)
    # the faceted sphere lies just inside the unit sphere: stay clear of its skin
    points = points[(radius < 0.95) | (radius > 1.05)]
    inside = np.linalg.norm(points, axis=1) < 1.0
    rays, _, _, _ = scene.intersect(points, rng.normal(size=points.shape))
    assert np.array_equal(np.bincount(rays, minlength=len(points)) % 2 == 1, inside)
    # and straight at every vertex and every edge's midpoint from inside, the
    # rays that used to get two hits for one crossing: exactly one each
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))
    targets = np.vstack([vertices, vertices[edges].mean(axis=1)])
    origin = np.array([0.1, -0.2, 0.05])
    rays, _, _, distances = scene.intersect(np.tile(origin, (len(targets), 1)), targets - origin)
    assert np.array_equal(np.bincount(rays, minlength=len(targets)), np.ones(len(targets), dtype=np.int64))
    assert (parameters(distances, (targets - origin)[rays]) <= 1 + 1e-9).all()


def test_maximum_hits_caps_each_ray():
    # the first k hits of each ray, exactly the full answer's first k rows for
    # it; one hit per ray is the nearest; fewer than one is refused
    vertices, faces = sphere(10, 20)
    meshes = [(vertices, faces), (vertices + np.array([3.0, 0.0, 0.0]), faces)]
    scene = ray_scene(meshes)
    rng = np.random.default_rng(4)
    origins = np.column_stack([np.full(50, -5.0), rng.uniform(-0.5, 0.5, (50, 2))])
    directions = np.tile([1.0, 0.0, 0.0], (50, 1))
    full = scene.intersect(origins, directions)
    assert np.array_equal(np.bincount(full[0]), np.full(50, 4))
    for k in (1, 2, 3, 4, 10):
        keep = np.concatenate([np.flatnonzero(full[0] == ray)[:k] for ray in range(50)])
        for column, expected in zip(scene.intersect(origins, directions, maximum_hits=k), full, strict=True):
            assert np.array_equal(column, expected[keep])
    rays, instances, faces_hit, distances = scene.intersect(origins, directions, maximum_hits=1)
    first = first_hits(scene, origins, directions)
    assert np.array_equal(rays, np.arange(50))
    for mine, theirs in zip((instances, faces_hit, distances), first, strict=True):
        assert np.array_equal(mine, theirs)
    with pytest.raises(ValueError, match="maximum_hits"):
        scene.intersect(origins, directions, maximum_hits=0)
