"""
trimkit against the package trimesh uses today, on the same numpy inputs:
best of N wall-clock runs each. Prints one markdown row per comparison.
"""

import time

import DracoPy
import fast_simplification
import mapbox_earcut
import numpy as np
import rtree
import svg.path
import trimesh
import vhacdx
import xatlas as legacy_xatlas
from trimesh.ray.ray_pyembree import RayMeshIntersector

import trimkit


def best(call, repeat=5):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return min(times)


def row(name, old_name, old, new, *, repeat=5, note=""):
    t_old, t_new = best(old, repeat), best(new, repeat)
    times = f"{t_old * 1e3:.1f} | {t_new * 1e3:.1f} | {t_old / t_new:.1f}x"
    print(f"| `{name}` | `{old_name}` | {times} | {note} |", flush=True)


rng = np.random.default_rng(0)
sphere = trimesh.creation.icosphere(subdivisions=6)  # 81,920 faces
v, f = np.ascontiguousarray(sphere.vertices), np.ascontiguousarray(sphere.faces, dtype=np.int64)
print("| trimkit | replaces | old (ms) | trimkit (ms) | speedup | input |")
print("|---|---|---|---|---|---|")

# polygon triangulation

theta = np.linspace(0, 2 * np.pi, 200_000, endpoint=False)
ring = np.ascontiguousarray(
    np.column_stack([np.cos(theta), np.sin(theta)]) * (1 + 0.1 * np.sin(17 * theta))[:, None]
)
row(
    "trimkit.triangulate",
    "mapbox-earcut",
    lambda: mapbox_earcut.triangulate_float64(ring, np.array([len(ring)], np.uint32)),
    lambda: trimkit.triangulate(ring),
    note="200k-vertex polygon",
)

# simplification

row(
    "trimkit.simplify",
    "fast-simplification",
    lambda: fast_simplification.simplify(v, f, target_reduction=0.9),
    lambda: trimkit.simplify(v, f, len(f) // 10),
    note="82k faces to 10%",
)

# UV atlas

small = trimesh.creation.icosphere(subdivisions=4)
sv, sf = np.ascontiguousarray(small.vertices), np.ascontiguousarray(small.faces, dtype=np.int64)
row(
    "trimkit.parametrize",
    "xatlas",
    lambda: legacy_xatlas.parametrize(sv, sf),
    lambda: trimkit.parametrize(sv, sf),
    repeat=3,
    note="5k faces",
)

# convex decomposition

bunny = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3)
tv, tf = np.ascontiguousarray(bunny.vertices), np.ascontiguousarray(bunny.faces, dtype=np.int64)
# vhacdx takes VTK-style faces, a count before each triangle, as trimesh passes them
vtk = np.column_stack([np.full(len(tf), 3), tf]).ravel().astype(np.uint32)
row(
    "trimkit.decompose",
    "vhacdx",
    lambda: vhacdx.compute_vhacd(tv, vtk),
    lambda: trimkit.decompose(tv, tf),
    repeat=2,
    note="torus",
)

# draco

positions = v.astype(np.float32)
buffer = trimkit.exchange.draco_encode(
    f, [positions], np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
)
legacy_buffer = DracoPy.encode(positions, f)
row(
    "trimkit.exchange.draco_encode",
    "DracoPy",
    lambda: DracoPy.encode(positions, f),
    lambda: trimkit.exchange.draco_encode(
        f, [positions], np.array([trimkit.exchange.DracoKind.POSITION], dtype=np.int64)
    ),
    note="82k faces",
)
row(
    "trimkit.exchange.draco_decode",
    "DracoPy",
    lambda: DracoPy.decode(legacy_buffer),
    lambda: trimkit.exchange.draco_decode(np.frombuffer(buffer, np.uint8), np.array([0], dtype=np.int64)),
    note="82k faces",
)

# svg

d = "M0 0 " + "c1 2 3 4 5 6 a 3 2 30 1 0 4 4 L 7 8 " * 20_000
row(
    "trimkit.exchange.svg_parse",
    "svg.path",
    lambda: svg.path.parse_path(d),
    lambda: trimkit.exchange.svg_parse(d),
    repeat=3,
    note="60k commands",
)

# boxes

corners = rng.uniform(0, 100, (100_000, 3))
bounds = np.stack([corners, corners + 1.0], axis=1)
queries = bounds[:20_000]


def rtree_both():
    tree = rtree.index.Index(
        ((i, (*b[0], *b[1]), None) for i, b in enumerate(bounds)),
        properties=rtree.index.Property(dimension=3),
    )
    for q in queries:
        list(tree.intersection((*q[0], *q[1])))


row(
    "trimkit.Boxes",
    "rtree",
    rtree_both,
    lambda: trimkit.Boxes(bounds).intersect(queries),
    repeat=2,
    note="build 100k boxes, query 20k",
)

# rays

origins = rng.normal(size=(1_000_000, 3)) * 0.1
directions = rng.normal(size=(1_000_000, 3))
embree = RayMeshIntersector(sphere)
scene = trimkit.RayScene([(v, f)])
row(
    "trimkit.RayScene.intersect",
    "embreex",
    lambda: embree.intersects_first(origins, directions),
    lambda: scene.intersect(origins, directions, maximum_hits=1),
    repeat=3,
    note="1M rays, 82k faces",
)

# collision: trimesh's CollisionManager over many spheres

ball = trimesh.creation.icosphere(subdivisions=3)
k = 200
centers = rng.uniform(0, 12, (k, 3))
matrices = np.tile(np.eye(4), (k, 1, 1))
matrices[:, :3, 3] = centers
manager = trimesh.collision.CollisionManager()
for i in range(k):
    manager.add_object(str(i), ball, transform=matrices[i])
cscene = trimkit.CollisionScene(
    [(np.ascontiguousarray(ball.vertices), np.ascontiguousarray(ball.faces, dtype=np.int64))]
)
index = np.zeros(k, np.int64)
found = cscene.collides(matrices, index)
_, names = manager.in_collision_internal(return_names=True)
assert len(found) == len(names), (len(found), len(names))
row(
    "trimkit.CollisionScene.collides",
    "python-fcl",
    lambda: manager.in_collision_internal(return_names=True),
    lambda: cscene.collides(matrices, index),
    note=f"{k} spheres, {len(found)} touching pairs",
)
