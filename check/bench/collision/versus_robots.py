# every pair of a robot's collision meshes: trimesh's CollisionManager (python-fcl)
# against trimkit, with the same convex choices
import importlib
import sys
import time
import warnings

import numpy as np
import trimesh
import yourdfpy

import trimkit

warnings.filterwarnings("ignore")
for robot in sys.argv[1:]:
    r = yourdfpy.URDF.load(
        importlib.import_module(f"robot_descriptions.{robot}").URDF_PATH,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
        load_meshes=False,
    )
    cs = r.collision_scene
    nodes = list(cs.graph.nodes_geometry)
    geometry = [cs.geometry[cs.graph[n][1]] for n in nodes]
    lo = np.array(
        [j.limit.lower if j.limit and j.limit.lower is not None else -np.pi for j in r.actuated_joints]
    )
    hi = np.array(
        [j.limit.upper if j.limit and j.limit.upper is not None else np.pi for j in r.actuated_joints]
    )
    rng = np.random.default_rng(3)
    T = []
    for q in rng.uniform(lo, hi, (60, len(lo))):
        r.update_cfg(q)
        T.append(np.stack([cs.graph.get(n)[0] for n in nodes]))
    T = np.ascontiguousarray(T)
    manager = trimesh.collision.CollisionManager()
    for i, g in enumerate(geometry):
        manager.add_object(str(i), g)
    convex = [bool(g.is_convex) for g in geometry]
    scene = trimkit.CollisionScene(
        [
            (np.ascontiguousarray(g.vertices, np.float64), np.ascontiguousarray(g.faces, np.int64))
            for g in geometry
        ],
        convex=np.array(convex),
    )

    def trimesh_collides(m, manager=manager):
        for i in range(len(m)):
            manager.set_transform(str(i), m[i])
        manager.in_collision_internal(return_names=True)

    times = {}
    for name, call in [("trimesh", trimesh_collides), ("trimkit", scene.collides)]:
        best = np.inf
        for _ in range(3):
            t = time.perf_counter()
            for m in T:
                call(m)
            best = min(best, (time.perf_counter() - t) / len(T))
        times[name] = best * 1e6
    faces = sum(len(g.faces) for g in geometry)
    old, new = times["trimesh"], times["trimkit"]
    print(
        f"| {robot.split('_')[0]} | {len(nodes)} | {faces:,} | {old:.0f} | {new:.0f} | {old / new:.1f}x |",
        flush=True,
    )
