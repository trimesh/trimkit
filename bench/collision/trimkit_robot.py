"""
Robot self-collision through trimkit.collision, called the way a user would:
one Python call per configuration. Checked against python-fcl per pair (mesh
mode) and against exact triangle distance between hulls (convex mode).

usage: python trimkit_robot.py ROBOT MODE CONFIGS   (MODE: mesh or convex)
needs: numpy trimesh yourdfpy robot_descriptions python-fcl, and trimkit
"""

import importlib
import itertools
import sys
import time
import warnings

import fcl
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
        mask(scene.collides(m, pairs=candidates), candidates)
        for m in transforms(rng.uniform(lower, upper, (1000, len(lower))))
    ],
    axis=0,
)
keep = ~mask(scene.collides(zero, pairs=candidates), candidates) & (always < 0.95)
pairs = np.ascontiguousarray(candidates[keep])
matrices = transforms(rng.uniform(lower, upper, (config_count, len(lower))))

start = time.perf_counter()
found = [scene.collides(m, pairs=pairs) for m in matrices]
collide_us = (time.perf_counter() - start) / config_count * 1e6
hits = np.array([mask(f, pairs) for f in found])
start = time.perf_counter()
any_hit = [len(scene.collides(m, pairs=pairs)) > 0 for m in matrices]
start = time.perf_counter()
measured = [scene.distances(m, pairs=pairs) for m in matrices]
distance_us = (time.perf_counter() - start) / config_count * 1e6
distances = np.array([d for d, _, _ in measured])

print(
    f"## {robot_name} [{mode}]: {len(nodes)} bodies, {sum(len(g.faces) for g in geometry)} faces, "
    f"{len(pairs)} pairs, {config_count} configurations"
)
print(
    f"collides: {collide_us:.0f} µs per configuration, distances: {distance_us:.0f} µs; "
    f"{np.mean(np.any(hits, axis=1)):.0%} of configurations collide"
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
    errors = []
    for c in range(min(100, config_count)):
        exact = surfaces.distances(matrices[c], pairs=pairs)[0]
        apart = ~hits[c] & (exact > 1e-6)
        errors.append(np.abs(distances[c][apart] - exact[apart]))
    errors = np.concatenate(errors)
    print(
        f"vs exact hull distance: worst {errors.max() * 1e3:.4f} mm, "
        f"99th percentile {np.quantile(errors, 0.99) * 1e3:.5f} mm"
    )
