"""
Collision and distance between placed meshes: what a bare query means, the
difference between convex solids and triangle surfaces, exact answers on
boxes, and agreement with an exact separating-axis test over random poses.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from helpers import box, collision_boxes, sphere

import trimkit


def placed(*offsets, rotations=None) -> np.ndarray:
    # one rigid transform per offset along x, or per (x, y, z) offset
    matrices = np.tile(np.eye(4), (len(offsets), 1, 1))
    for i, offset in enumerate(offsets):
        matrices[i, :3, 3] = offset if np.ndim(offset) else (offset, 0.0, 0.0)
        if rotations is not None:
            matrices[i, :3, :3] = rotations[i]
    return matrices


def rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def scene(convex, *meshes) -> trimkit.CollisionScene:
    """
    A scene of these meshes (by default one unit cube), all convex or all
    not, or convex by mesh, checked to know where its meshes are.
    """
    meshes = list(meshes) or [box()]
    built = trimkit.CollisionScene(meshes, convex=convex if isinstance(convex, bool) else np.array(convex))
    collision_boxes(built, meshes)
    return built


@pytest.mark.parametrize("convex", [False, True])
def test_bare_query_is_the_scene_as_built(convex):
    # no instances given: each mesh once, where it is, every pair
    touching = scene(convex, box(), box((0.5, 0, 0), (1.5, 1, 1)), box((5, 0, 0), (6, 1, 1)))
    assert touching.collides().tolist() == [[0, 1]]
    distances, _, _ = touching.distances()
    assert distances[1:] == pytest.approx([4.0, 3.5])


def test_found_pairs_match_named_pairs():
    # the tree finds the same touching pairs as naming every pair, ascending
    four = scene(False)
    matrices = placed(0.0, 0.5, 5.0, 5.5)
    index = np.zeros(4, np.int64)
    collision_boxes(four, [box()], matrices, index)
    explicit = np.column_stack(np.triu_indices(4, 1)).astype(np.int64)
    assert four.collides(matrices, index).tolist() == four.collides(matrices, index, pairs=explicit).tolist()
    assert four.collides(matrices, index).tolist() == [[0, 1], [2, 3]]
    # named pairs come back as named, in their order
    assert four.collides(
        matrices, index, pairs=np.array([[3, 2], [1, 3], [1, 0]], dtype=np.int64)
    ).tolist() == [
        [3, 2],
        [1, 0],
    ]


def test_convex_meshes_are_solids():
    # a small box inside a big one: their triangles never touch, but as
    # convex solids they overlap, with a negative signed distance
    inner, outer = box((0.4, 0.4, 0.4), (0.6, 0.6, 0.6)), box()
    assert scene(False, inner, outer).collides().tolist() == []
    assert scene(True, inner, outer).collides().tolist() == [[0, 1]]
    assert scene(False, inner, outer).distances()[0][0] == pytest.approx(0.4)
    # as solids: the least translation that separates them, 0.6 along an axis
    assert scene(True, inner, outer).distances()[0][0] == pytest.approx(-0.6, abs=1e-9)


@pytest.mark.parametrize("convex", [False, True])
def test_box_distance_points_and_normal(convex):
    two = scene(convex)
    distances, points, normals = two.distances(placed(0.0, 1.5), np.zeros(2, np.int64))
    assert distances == pytest.approx([0.5])
    # the closest points are on the facing sides, the normal from first to second
    assert points[0, 0, 0] == pytest.approx(1.0)
    assert points[0, 1, 0] == pytest.approx(1.5)
    assert normals[0] == pytest.approx([1.0, 0.0, 0.0])


def test_mesh_normal_is_nan_once_touching():
    _, _, normals = scene(False).distances(placed(0.0, 0.5), np.zeros(2, np.int64))
    assert np.isnan(normals).all()


@pytest.mark.parametrize("convex", [False, True])
def test_margin_counts_near_pairs(convex):
    two, matrices, index = scene(convex), placed(0.0, 1.5), np.zeros(2, np.int64)
    assert two.collides(matrices, index).tolist() == []
    assert two.collides(matrices, index, margin=0.4).tolist() == []
    assert two.collides(matrices, index, margin=0.6).tolist() == [[0, 1]]


def test_upper_bound_skips_far_pairs():
    two, matrices, index = scene(True), placed(0.0, 1.5), np.zeros(2, np.int64)
    distances, points, normals = two.distances(matrices, index, upper_bound=0.4)
    assert distances.tolist() == [np.inf]
    assert np.isnan(points).all() and np.isnan(normals).all()
    assert two.distances(matrices, index, upper_bound=0.6)[0] == pytest.approx([0.5])


def test_empty_meshes_never_touch():
    # placed anywhere, even right on the other mesh, and its box is the point
    empty = (np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    both = scene(False, empty, box())
    assert both.collides().tolist() == []
    assert both.distances()[0].tolist() == [np.inf]
    on_top = placed((0.5, 0.5, 0.5), 0.0)
    collision_boxes(both, [empty, box()], on_top)
    assert both.collides(on_top).tolist() == []


def separated(first: np.ndarray, second: np.ndarray) -> bool:
    """
    Whether two unit cubes placed by rigid transforms are apart: exact, by the
    separating axis theorem's fifteen axes.
    """
    corners = np.array([[x, y, z, 1.0] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    a = (corners @ first.T)[:, :3]
    b = (corners @ second.T)[:, :3]
    axes = [*first[:3, :3].T, *second[:3, :3].T]
    axes += [np.cross(u, v) for u in first[:3, :3].T for v in second[:3, :3].T]
    for axis in axes:
        if np.linalg.norm(axis) < 1e-12:
            continue
        pa, pb = a @ axis, b @ axis
        if pa.max() < pb.min() - 1e-9 or pb.max() < pa.min() - 1e-9:
            return True
    return False


@pytest.mark.parametrize("convex", [False, True])
def test_agrees_with_separating_axes(convex):
    rng = np.random.default_rng(1)
    count = 40
    matrices = placed(*rng.uniform(0, 3, (count, 3)), rotations=[rotation(rng) for _ in range(count)])
    cubes = scene(convex)
    collision_boxes(cubes, [box()], matrices, np.zeros(count, np.int64))
    found = cubes.collides(matrices, np.zeros(count, np.int64))
    pairs = np.column_stack(np.triu_indices(count, 1)).astype(np.int64)
    touching = {tuple(row) for row in found.tolist()}
    hits = [tuple(row) in touching for row in pairs.tolist()]
    expected = [not separated(matrices[i], matrices[j]) for i, j in pairs]
    # as surfaces too: triangles miss only a solid wholly inside the other,
    # which a cube can't be of another the same size
    assert hits == expected
    assert any(expected) and not all(expected)


def test_both_kernels_agree_on_distance():
    rng = np.random.default_rng(2)
    count = 20
    matrices = placed(*rng.uniform(0, 8, (count, 3)), rotations=[rotation(rng) for _ in range(count)])
    index = np.zeros(count, np.int64)
    meshes, _, _ = scene(False).distances(matrices, index)
    solids, _, _ = scene(True).distances(matrices, index)
    apart = solids > 1e-6
    assert apart.any()
    assert meshes[apart] == pytest.approx(solids[apart], abs=1e-6)


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        (np.diag([2.0, 1.0, 1.0, 1.0]), "rigid"),
        (np.diag([-1.0, 1.0, 1.0, 1.0]), "rigid"),
        (np.vstack([np.eye(4)[:3], [0.0, 0.0, 1.0, 1.0]]), "last row"),
    ],
)
def test_matrices_must_be_rigid(matrix, match):
    with pytest.raises(ValueError, match=match):
        scene(False).collides(matrix[None].copy())


def test_instances_are_checked():
    one = scene(False)
    with pytest.raises(ValueError, match="out of range"):
        one.collides(None, np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="one row per instance"):
        one.collides(placed(0.0, 1.0, 2.0), np.zeros(2, np.int64))
    with pytest.raises(ValueError, match="one instance per mesh"):
        one.collides(placed(0.0, 1.0))
    with pytest.raises(ValueError, match="out of range"):
        one.collides(placed(0.0, 1.0), np.zeros(2, np.int64), pairs=np.array([[0, 2]], dtype=np.int64))
    with pytest.raises(ValueError, match="twice"):
        one.collides(placed(0.0, 1.0), np.zeros(2, np.int64), pairs=np.array([[1, 1]], dtype=np.int64))
    with pytest.raises(ValueError, match="margin"):
        one.collides(margin=-1.0)
    with pytest.raises(ValueError, match="upper_bound"):
        one.distances(upper_bound=0.0)


def test_convex_scalar_is_every_mesh():
    meshes = [box(), box((0.5, 0.5, 0.5), (2, 2, 2))]
    matrices = placed(0.0, 0.0)
    for scalar in (True, False):
        mask = np.full(len(meshes), scalar)
        every = trimkit.CollisionScene(meshes, convex=scalar).distances(matrices)
        masked = trimkit.CollisionScene(meshes, convex=mask).distances(matrices)
        for a, b in zip(every, masked, strict=True):
            assert np.array_equal(a, b, equal_nan=True)
    # the default is none, and the two answers differ: one solid inside the
    # other is penetrated, two surfaces crossing touch
    default = trimkit.CollisionScene(meshes).distances(matrices)
    assert np.array_equal(default[0], trimkit.CollisionScene(meshes, convex=False).distances(matrices)[0])
    assert default[0][0] == 0.0
    assert trimkit.CollisionScene(meshes, convex=True).distances(matrices)[0][0] < 0.0


def test_convex_is_a_promise():
    # a box whose top face dips to a vertex inside it: not convex, but its
    # vertices' hull is the box, which is what a convex mesh is measured as
    vertices, faces = box()
    dented = np.vstack([vertices, [[0.5, 0.5, 0.6]]])
    top = [[1, 5, 8], [5, 7, 8], [7, 3, 8], [3, 1, 8]]
    dented = dented, np.vstack([faces[:-2], top])
    probe = box((0.4, 0.4, 0.9), (0.6, 0.6, 1.9))
    matrices, pairs = placed(0.0, 0.0), np.array([[0, 1]], dtype=np.int64)
    honest = scene(True, box(), probe).distances(matrices, pairs=pairs)
    promised = scene(True, dented, probe).distances(matrices, pairs=pairs)
    for a, b in zip(honest, promised, strict=True):
        assert np.array_equal(a, b)
    assert np.isclose(promised[0][0], -0.1)
    # measured as the surface it is, the probe sits in the dent untouched
    assert scene(False, dented, probe).collides(matrices, pairs=pairs).shape == (0, 2)
    with pytest.raises(ValueError, match=r"^meshes\[0\] is convex, so it needs faces$"):
        trimkit.CollisionScene([(vertices, faces[:0])], convex=True)
    with pytest.raises(ValueError, match="one row per mesh"):
        trimkit.CollisionScene([box()], convex=np.array([True, False]))
    with pytest.raises(TypeError, match="convex"):
        trimkit.CollisionScene([box()], convex=np.array([1], dtype=np.int64))
    with pytest.raises(TypeError, match="convex"):
        trimkit.CollisionScene([box()], convex=None)
    # a sphere's surface is a convex solid
    scene(True, sphere(10, 20))


def test_shared_scene():
    # one scene queried from many threads at once gives the serial answers
    rng = np.random.default_rng(3)
    count = 60
    matrices = placed(*rng.uniform(0, 4, (count, 3)), rotations=[rotation(rng) for _ in range(count)])
    index = rng.integers(0, 2, count)
    shared = scene([False, True], sphere(12, 24), sphere(12, 24))
    collision_boxes(shared, [sphere(12, 24)] * 2, matrices, index)
    expected = shared.distances(matrices, index)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: shared.distances(matrices, index), range(16)))
    for result in results:
        for got, want in zip(result, expected, strict=True):
            assert np.array_equal(got, want, equal_nan=True)


def test_one_mesh_many_instances():
    # instances place meshes: one cube built once, placed three times, two of
    # them overlapping, and the same answer as three separate cube meshes
    one = scene(False)
    three = scene(False, box(), box(), box())
    matrices = placed(0.0, 0.5, 5.0)
    assert one.collides(matrices, np.zeros(3, np.int64)).tolist() == [[0, 1]]
    assert three.collides(matrices).tolist() == [[0, 1]]


def test_convex_is_a_solid_against_every_mesh():
    # a small plain mesh inside a convex box touches it, either way round,
    # and its distance to the solid is negative; as two surfaces it doesn't
    inner, outer = box((0.4, 0.4, 0.4), (0.6, 0.6, 0.6)), box()
    mixed = scene([False, True], inner, outer)
    flipped = scene([True, False], outer, inner)
    assert mixed.collides().tolist() == [[0, 1]]
    assert flipped.collides().tolist() == [[0, 1]]
    # the inner surface is 0.4 inside the solid's boundary, either way round
    assert mixed.distances()[0][0] == pytest.approx(-0.4, abs=1e-9)
    assert flipped.distances()[0][0] == pytest.approx(-0.4, abs=1e-9)
    # apart, a mixed pair measures the same as two surfaces, either way round
    matrices = placed(0.0, 2.0)
    for order in ([False, True], [True, False]):
        two = scene(order, box(), box())
        distance, points, normals = two.distances(matrices)
        assert distance == pytest.approx([1.0])
        assert points[0, 0, 0] == pytest.approx(1.0) and points[0, 1, 0] == pytest.approx(2.0)
        assert normals[0] == pytest.approx([1.0, 0.0, 0.0])


def test_boxes_bound_tightly():
    # a long thin bar lying diagonally in its own frame, straightened by its
    # instance's rotation: its vertices' box, rotated, would reach a parallel
    # bar 3 away, while its oriented box doesn't, and either way a pair past
    # `upper_bound` is infinitely far
    vertices, faces = box((0, 0, 0), (10, 0.1, 0.1))
    diagonal = np.array([[np.sqrt(0.5), -np.sqrt(0.5), 0], [np.sqrt(0.5), np.sqrt(0.5), 0], [0, 0, 1]])
    turned = np.ascontiguousarray(vertices @ diagonal.T)
    straighten = np.eye(4)
    straighten[:3, :3] = diagonal.T
    beside = np.eye(4)
    beside[1, 3] = 3.0
    bars = scene(False, (turned, faces), (vertices, faces))
    matrices = np.stack([straighten, beside])
    collision_boxes(bars, [(turned, faces), (vertices, faces)], matrices)
    # the turned bar's box comes from its oriented box, so the two bars'
    # boxes don't meet, where its vertices' box, turned, reaches past y = 3
    turned_box, beside_box = bars.boxes(matrices).boxes()
    assert turned_box[1, 1] < beside_box[0, 1]
    assert bars.distances(matrices, upper_bound=1.0)[0].tolist() == [np.inf]
    assert bars.distances(matrices)[0] == pytest.approx([2.9])


@pytest.mark.parametrize("convex", [(False, False), (False, True), (True, False), (True, True)])
def test_upper_bound_is_exact(convex):
    # two spheres whose boxes come within the bound while the spheres don't:
    # past the bound is infinitely far, with nothing measured, and anything
    # at most that far is measured exactly as without a bound
    two = scene(list(convex), sphere(), sphere())
    matrices = placed(0.0, (2.2, 2.2, 2.2))
    measured, points, normals = two.distances(matrices)
    gap = measured[0]
    # the gap a brute-force minimum over every vertex of either sphere against
    # every triangle of the other gives (2026-11-09), from each kernel alike
    assert gap == pytest.approx(1.813354063143859, abs=1e-9)
    far, far_points, far_normals = two.distances(matrices, upper_bound=1.0)
    assert far.tolist() == [np.inf]
    assert np.isnan(far_points).all() and np.isnan(far_normals).all()
    for bound in (gap, np.nextafter(gap, np.inf), 2.0):
        near, near_points, near_normals = two.distances(matrices, upper_bound=bound)
        assert near.tolist() == measured.tolist()
        assert np.array_equal(near_points, points) and np.array_equal(near_normals, normals)
    assert two.distances(matrices, upper_bound=np.nextafter(gap, 0.0))[0].tolist() == [np.inf]


@pytest.mark.parametrize("offset", [1.7, 0.6], ids=["apart", "penetrating"])
def test_distance_gradient(offset):
    # what a planner differentiates: for two convex solids, moving the second
    # by a small step changes the signed distance by the normal along that
    # step, and turning it about its origin by the closest point's lever arm
    rng = np.random.default_rng(3)
    two = scene(True, *[box((-0.5,) * 3, (0.5,) * 3)] * 2)
    turned = placed(0.0, (offset, 0.2, 0.1), rotations=[np.eye(3), rotation(rng)])
    distance, points, normals = two.distances(turned)
    lever = points[0, 1] - turned[1, :3, 3]
    step = 1e-4
    for _ in range(4):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        moved = []
        for sign in (1.0, -1.0):
            matrices = turned.copy()
            matrices[1, :3, 3] += sign * step * direction
            moved.append(two.distances(matrices)[0][0])
        assert (moved[0] - moved[1]) / (2 * step) == pytest.approx(normals[0] @ direction, abs=1e-3)
        moved = []
        for sign in (1.0, -1.0):
            angle = sign * step
            axis = np.array(
                [
                    [0, -direction[2], direction[1]],
                    [direction[2], 0, -direction[0]],
                    [-direction[1], direction[0], 0],
                ]
            )
            spin = np.eye(3) + np.sin(angle) * axis + (1 - np.cos(angle)) * axis @ axis
            matrices = turned.copy()
            matrices[1, :3, :3] = spin @ turned[1, :3, :3]
            moved.append(two.distances(matrices)[0][0])
        assert (moved[0] - moved[1]) / (2 * step) == pytest.approx(
            np.cross(lever, normals[0]) @ direction, abs=1e-3
        )
    assert (distance[0] > 0) == (offset > 1.0)


def test_stacked_poses_are_one_call():
    # N poses of an L-link chain as N * L instances, each pose's pairs named:
    # one call answers exactly what N calls of one pose each do
    rng = np.random.default_rng(5)
    links = [box((0, 0, 0), (1.0, 0.2, 0.2)), box((0, 0, 0), (0.3, 0.3, 0.3)), sphere(8, 12)]
    chain = scene(True, *links)
    poses, count = 6, len(links)
    matrices = placed(
        *rng.uniform(-0.6, 0.6, size=(poses * count, 3)),
        rotations=[rotation(rng) for _ in range(poses * count)],
    )
    one = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    pairs = np.concatenate([one + count * pose for pose in range(poses)])
    index = np.tile(np.arange(count, dtype=np.int64), poses)
    collision_boxes(chain, links, matrices, index)
    stacked = chain.distances(matrices, index, pairs=pairs)
    touching = chain.collides(matrices, index, pairs=pairs)
    alone = [chain.distances(matrices[count * pose : count * (pose + 1)], pairs=one) for pose in range(poses)]
    for stacked_output, alone_output in zip(stacked, zip(*alone, strict=True), strict=True):
        assert np.array_equal(stacked_output, np.concatenate(alone_output), equal_nan=True)
    expected = [
        chain.collides(matrices[count * pose : count * (pose + 1)], pairs=one) + count * pose
        for pose in range(poses)
    ]
    assert len(touching) > 0
    assert touching.tolist() == np.concatenate(expected).tolist()


def test_pairs_and_options_are_keywords():
    # instances are positional; everything after them is named, so a call
    # reads the same in ten years
    two = scene(True, box(), box((2, 0, 0), (3, 1, 1)))
    with pytest.raises(TypeError):
        two.collides(None, None, np.array([[0, 1]], dtype=np.int64))
    with pytest.raises(TypeError):
        two.distances(None, None, np.array([[0, 1]], dtype=np.int64))
    assert two.collides(pairs=np.array([[0, 1]], dtype=np.int64), margin=1.5).tolist() == [[0, 1]]
