"""
Robot self-collision through trimkit.collision, called the way a user would:
one Python call per configuration. Checked against python-fcl per pair (mesh
mode) and against exact triangle distance between hulls (convex mode), and
MuJoCo on the same poses and pairs, timed with and without its Python.

usage: python trimkit_robot.py ROBOT MODE CONFIGS [DIRECTORY]   (MODE: mesh or convex)
  DIRECTORY: also write MuJoCo's models, poses and pairs there, for
  check/bench/collision/mujoco_kernel.cpp to time in C
needs: numpy trimesh yourdfpy robot_descriptions python-fcl mujoco, and trimkit
"""

import hashlib
import importlib
import itertools
import sys
import time
import warnings
from pathlib import Path

import fcl
import mujoco
import numpy as np
import yourdfpy

import trimkit

warnings.filterwarnings("ignore")
robot_name, mode, config_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = np.random.default_rng(1)
module = importlib.import_module(f"robot_descriptions.{robot_name}")
robot = yourdfpy.URDF.load(
    module.URDF_PATH, build_collision_scene_graph=True, load_collision_meshes=True, load_meshes=False
)
graph = robot.collision_scene.graph
nodes = list(graph.nodes_geometry)
link_of = {n: graph.transforms.parents[n] for n in nodes}
geometry = [robot.collision_scene.geometry[graph[n][1]] for n in nodes]
geometry = [g.convex_hull for g in geometry] if mode == "convex" else geometry
convex = mode == "convex"
meshes = [
    (np.ascontiguousarray(g.vertices, np.float64), np.ascontiguousarray(g.faces, np.int64)) for g in geometry
]
scene = trimkit.CollisionScene(meshes, convex=np.full(len(meshes), convex))
adjacent = {frozenset((j.parent, j.child)) for j in robot.robot.joints}
candidates = np.array(
    [
        (i, k)
        for i, k in itertools.combinations(range(len(nodes)), 2)
        if link_of[nodes[i]] != link_of[nodes[k]]
        and frozenset((link_of[nodes[i]], link_of[nodes[k]])) not in adjacent
    ],
    dtype=np.int64,
)
lower = np.array(
    [j.limit.lower if j.limit and j.limit.lower is not None else -np.pi for j in robot.actuated_joints]
)
upper = np.array(
    [j.limit.upper if j.limit and j.limit.upper is not None else np.pi for j in robot.actuated_joints]
)


def mask(found: np.ndarray, named: np.ndarray) -> np.ndarray:
    # which rows of `named` are among the touching pairs `collides` returned
    touching = {tuple(row) for row in found.tolist()}
    return np.array([tuple(row) in touching for row in named.tolist()], dtype=bool)


def transforms(configurations) -> np.ndarray:
    out = np.empty((len(configurations), len(nodes), 4, 4))
    for c, q in enumerate(configurations):
        robot.update_cfg(q)
        for i, n in enumerate(nodes):
            out[c, i] = graph.get(n)[0]
    return out


# MoveIt-style: drop pairs touching at zero or in >= 95% of 1000 random configurations
zero = transforms([np.clip(np.zeros(len(lower)), lower, upper)])[0]
always = np.mean(
    [
        mask(scene.collides(m, pairs=candidates).pairs, candidates)
        for m in transforms(rng.uniform(lower, upper, (1000, len(lower))))
    ],
    axis=0,
)
keep = ~mask(scene.collides(zero, pairs=candidates).pairs, candidates) & (always < 0.95)
pairs = np.ascontiguousarray(candidates[keep])
matrices = transforms(rng.uniform(lower, upper, (config_count, len(lower))))


def best_us(call, runs: int = 3) -> float:
    # the best of `runs` passes, in µs per configuration
    best = np.inf
    for _ in range(runs):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best / config_count * 1e6


found = [scene.collides(m, pairs=pairs) for m in matrices]
hits = np.array([mask(result.pairs, pairs) for result in found])
distances = np.array([scene.distances(m, pairs=pairs).distances for m in matrices])
collide_us = best_us(lambda: [scene.collides(m, pairs=pairs) for m in matrices])
distance_us = best_us(lambda: [scene.distances(m, pairs=pairs) for m in matrices])
# what a call costs before any pair: its Python, checks and placing
no_pairs = np.empty((0, 2), dtype=np.int64)
overhead_us = best_us(lambda: [scene.collides(m, pairs=no_pairs) for m in matrices])
# every configuration in one call, as a path is asked: its poses' links as
# instances, each pose's pairs offset to its own links
stacked = np.ascontiguousarray(matrices.reshape(-1, 4, 4))
stacked_index = np.tile(np.arange(len(nodes), dtype=np.int64), config_count)
stacked_pairs = np.ascontiguousarray(
    (pairs[None] + len(nodes) * np.arange(config_count, dtype=np.int64)[:, None, None]).reshape(-1, 2)
)
batch = scene.distances(stacked, stacked_index, pairs=stacked_pairs).distances.reshape(config_count, -1)
assert np.array_equal(batch, distances), "one call answers differently than one per configuration"
batch_collide_us = best_us(lambda: scene.collides(stacked, stacked_index, pairs=stacked_pairs))
batch_distance_us = best_us(lambda: scene.distances(stacked, stacked_index, pairs=stacked_pairs))


# MuJoCo on the same meshes, poses and pairs, set up as its docs and source
# advise for geometry queries: built with MjSpec, one mocap body and mesh geom
# per instance, contype = conaffinity = 0 so only the explicit pairs are
# tested, native CCD (its default; libccd was 2.5x slower on panda), and one
# contact per pair (mjDSBL_MULTICCD: multiccd's extra contacts cost and add
# nothing to which pairs touch or how far apart they are). Poses go straight
# into geom_xpos / geom_xmat, composed with the offset its compiler gives each
# mesh (recentered on its own inertial frame), which is what mj_kinematics
# would write, so mj_collision is the whole query. It collides every mesh as
# its convex hull: in mesh mode it answers a different question than trimkit
def mujoco_scene(margin: float):
    spec = mujoco.MjSpec()
    spec.option.disableflags = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    for i, (vertices, faces) in enumerate(meshes):
        spec.add_mesh(name=f"mesh{i}", uservert=vertices.ravel().tolist(), userface=faces.ravel().tolist())
        body = spec.worldbody.add_body(name=f"body{i}", mocap=True)
        body.add_geom(
            name=f"geom{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"mesh{i}", contype=0, conaffinity=0
        )
    for a, b in pairs:
        spec.add_pair(geomname1=f"geom{a}", geomname2=f"geom{b}", margin=margin)
    model = spec.compile()
    return model, mujoco.MjData(model)


def mujoco_place(model, data, m: np.ndarray) -> None:
    rotation = m[:, :3, :3]
    data.geom_xmat[:] = (rotation @ geom_rotation).reshape(-1, 9)
    data.geom_xpos[:] = np.einsum("kij,kj->ki", rotation, model.geom_pos) + m[:, :3, 3]


start = time.perf_counter()
touch_model, touch_data = mujoco_scene(0.0)
mujoco_compile_ms = (time.perf_counter() - start) * 1e3
geom_rotation = np.empty((touch_model.ngeom, 3, 3))
for g in range(touch_model.ngeom):
    mujoco.mju_quat2Mat(geom_rotation[g].reshape(9), touch_model.geom_quat[g])
# every pair's row in `pairs`, by its geoms (geom i is instance i)
pair_row = np.full((len(nodes), len(nodes)), -1, dtype=np.int64)
rows = np.arange(len(pairs), dtype=np.int64)
pair_row[pairs[:, 0], pairs[:, 1]] = pair_row[pairs[:, 1], pairs[:, 0]] = rows

mujoco_hits = np.zeros_like(hits)
mujoco_collide_kernel = 0.0
start = time.perf_counter()
for c, m in enumerate(matrices):
    mujoco_place(touch_model, touch_data, m)
    kernel = time.perf_counter()
    mujoco.mj_collision(touch_model, touch_data)
    mujoco_collide_kernel += time.perf_counter() - kernel
    geoms = touch_data.contact.geom[: touch_data.ncon]
    mujoco_hits[c, pair_row[geoms[:, 0], geoms[:, 1]]] = True
mujoco_collide_us = (time.perf_counter() - start) / config_count * 1e6

# distances: every pair given a margin past any gap in the robot, so
# mj_collision measures each one apart (true GJK distance) in one call. Its
# depth of an overlap is wrong at such a margin (positive, ~0.45 on panda,
# while exact at 0.1 and below), so overlapping pairs take the margin-0
# model's depth: two mj_collision calls, the fastest correct way through it
reach = 2.0 * float(np.ptp(matrices[:, :, :3, 3])) + 1.0
far_model, far_data = mujoco_scene(reach)
mujoco_distances = np.full_like(distances, np.inf)
mujoco_distance_kernel = 0.0
start = time.perf_counter()
for c, m in enumerate(matrices):
    mujoco_place(far_model, far_data, m)
    mujoco_place(touch_model, touch_data, m)
    kernel = time.perf_counter()
    mujoco.mj_collision(far_model, far_data)
    mujoco.mj_collision(touch_model, touch_data)
    mujoco_distance_kernel += time.perf_counter() - kernel
    geoms = far_data.contact.geom[: far_data.ncon]
    np.minimum.at(
        mujoco_distances[c], pair_row[geoms[:, 0], geoms[:, 1]], far_data.contact.dist[: far_data.ncon]
    )
    geoms = touch_data.contact.geom[: touch_data.ncon]
    mujoco_distances[c, pair_row[geoms[:, 0], geoms[:, 1]]] = touch_data.contact.dist[: touch_data.ncon]
mujoco_distance_us = (time.perf_counter() - start) / config_count * 1e6
# and mj_geomDistance, its one-pair query (exact at any distmax), on a few
# configurations: a Python call per pair here, so check/bench/collision/
# mujoco_kernel.cpp times it in C
geom_count = min(20, config_count)
geom_distances = np.empty((geom_count, len(pairs)))
start = time.perf_counter()
for c in range(geom_count):
    mujoco_place(far_model, far_data, matrices[c])
    for p, (a, b) in enumerate(pairs):
        geom_distances[c, p] = mujoco.mj_geomDistance(far_model, far_data, a, b, reach, None)
geom_distance_us = (time.perf_counter() - start) / geom_count * 1e6
if len(sys.argv) > 4:
    out = Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(touch_model, str(out / "touch.mjb"), None)
    mujoco.mj_saveModel(far_model, str(out / "far.mjb"), None)
    poses = np.empty((config_count, len(nodes), 12))
    for c, m in enumerate(matrices):
        mujoco_place(far_model, far_data, m)
        poses[c, :, :3], poses[c, :, 3:] = far_data.geom_xpos, far_data.geom_xmat
    poses.tofile(out / "poses.bin")
    pairs.tofile(out / "pairs.bin")
    np.array([reach]).tofile(out / "reach.bin")
    # and trimkit's own inputs, for timing builds against each other quickly
    np.savez(
        out / "trimkit.npz",
        matrices=matrices,
        pairs=pairs,
        convex=np.full(len(meshes), mode == "convex"),
        **{f"vertices{i}": v for i, (v, _) in enumerate(meshes)},
        **{f"faces{i}": f for i, (_, f) in enumerate(meshes)},
    )

print(
    f"## {robot_name} [{mode}]: {len(nodes)} bodies, {sum(len(g.faces) for g in geometry)} faces, "
    f"{len(pairs)} pairs, {config_count} configurations"
)
print(
    f"collides: {collide_us:.0f} µs per configuration, distances: {distance_us:.0f} µs, "
    f"of each a call with no pairs {overhead_us:.1f} µs; in one call for every configuration: "
    f"collides {batch_collide_us:.1f} µs, distances {batch_distance_us:.1f} µs per configuration; "
    f"{np.mean(np.any(hits, axis=1)):.0%} of configurations collide; answers "
    f"{hashlib.sha256(hits.tobytes() + distances.tobytes()).hexdigest()[:12]}"
)
print(
    f"mujoco {mujoco.__version__} (compile {mujoco_compile_ms:.0f} ms): "
    f"collides {mujoco_collide_us:.0f} µs per configuration "
    f"({mujoco_collide_kernel / config_count * 1e6:.0f} in mj_collision), distances "
    f"{mujoco_distance_us:.0f} µs ({mujoco_distance_kernel / config_count * 1e6:.0f} in mj_collision), "
    f"mj_geomDistance from Python {geom_distance_us:.0f} µs; "
    f"{np.sum(mujoco_hits != hits)} hit disagreements with trimkit of {hits.size}"
)
print(
    f"mujoco distances vs trimkit: worst {np.max(np.abs(mujoco_distances - distances)) * 1e3:.4f} mm; "
    f"mj_collision vs mj_geomDistance worst "
    f"{np.max(np.abs(mujoco_distances[: len(geom_distances)] - geom_distances)) * 1e3:.4f} mm"
)

# python-fcl, per pair, on the first 50 configurations
if mode == "mesh":

    def bvh(g):
        model = fcl.BVHModel()
        model.beginModel(len(g.vertices), len(g.faces))
        model.addSubModel(g.vertices, g.faces)
        model.endModel()
        return model

    objects = [fcl.CollisionObject(bvh(g), fcl.Transform()) for g in geometry]
    disagree, worst = 0, 0.0
    for c in range(min(50, config_count)):
        for i, o in enumerate(objects):
            o.setTransform(fcl.Transform(matrices[c, i, :3, :3], matrices[c, i, :3, 3]))
        for p, (a, b) in enumerate(pairs):
            touching = fcl.collide(objects[a], objects[b], fcl.CollisionRequest(), fcl.CollisionResult()) > 0
            disagree += touching != hits[c, p]
            if not touching:
                gap = fcl.distance(objects[a], objects[b], fcl.DistanceRequest(), fcl.DistanceResult())
                worst = max(worst, abs(gap - distances[c, p]))
    print(
        f"vs python-fcl: {disagree} hit disagreements of {min(50, config_count) * len(pairs)}, "
        f"worst distance difference {worst:.1e}"
    )
else:
    # the same hulls as triangle surfaces: exact distance between separated solids
    surfaces = trimkit.CollisionScene(meshes)
    errors, mujoco_errors = [], []
    for c in range(min(100, config_count)):
        exact = surfaces.distances(matrices[c], pairs=pairs).distances
        apart = ~hits[c] & (exact > 1e-6)
        errors.append(np.abs(distances[c][apart] - exact[apart]))
        mujoco_errors.append(np.abs(mujoco_distances[c][apart] - exact[apart]))
    for name, error in [("trimkit", np.concatenate(errors)), ("mujoco", np.concatenate(mujoco_errors))]:
        print(
            f"{name} vs exact hull distance: worst {error.max() * 1e3:.4f} mm, "
            f"99th percentile {np.quantile(error, 0.99) * 1e3:.5f} mm"
        )
