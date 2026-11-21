# MuJoCo and coal: notes

Would MuJoCo make a better collision engine for `CollisionScene` than coal? We benchmarked MuJoCo 4.0.1 as well as we could set it up. Its convex kernel was 2 to 3.5 times faster than ours per core. The two use the same algorithms (GJK for distance, EPA for depth, a hill climb over each hull's vertices), so we found where coal spent the difference and fixed it, mostly in coal. On one core coal is now faster than MuJoCo at everything but panda's `collides`, and on every core at everything but talos's distances, where MuJoCo's threads spin and ours sleep between calls. Coal also still does what MuJoCo can't: triangle meshes that aren't convex.

Measured 2026-11-20 on an i9-14900K (8 performance cores with 2 threads each, 16 efficiency cores: 32 threads). The machine had other jobs running, so every number is the minimum over interleaved rounds, each build timed in a fresh process (see Reproduce).

## The robots

Three robots from `robot_descriptions`, each link's collision mesh replaced by its convex hull, and the pairs a motion planner would check: links not joined to each other, minus pairs that always or never touch (`trimkit_robot.py`). Times are µs per configuration.

| robot | links | pairs | hull vertices (median, max) |
|---|---|---|---|
| panda | 11 | 42 | 102, 152 |
| talos | 52 | 1242 | 130, 953 |
| g1 | 31 | 438 | 724, 5583 |

One core, one call per configuration. MuJoCo is called from C, so it pays no Python; trimkit pays 1 to 4 µs of Python per call. MuJoCo's distances include the witness points, as trimkit's do.

| | trimkit before | trimkit now | MuJoCo |
|---|---|---|---|
| panda collides | 5.0 | 2.9 | 2.4 |
| panda distances | 53.5 | 27.1 | 35.4 |
| talos collides | 125 | 30.5 | 47.5 |
| talos distances | 1451 | 633 | 662 |
| g1 collides | 98.7 | 23.2 | 49.9 |
| g1 distances | 1091 | 344 | 400 |

Every core. trimkit takes all configurations in one call (instances stacked, pairs offset per pose), which is how its pool is meant to be used. MuJoCo is its best case: each configuration's pairs split over 1 to 16 threads that spin between configurations so waking them costs nothing, the best thread count kept.

| | trimkit before | trimkit now | MuJoCo, best threads |
|---|---|---|---|
| panda collides | 0.9 | 0.8 | 1.8 (2) |
| panda distances | 4.8 | 3.5 | 14.2 (8) |
| talos collides | 13.4 | 7.3 | 20.7 (4) |
| talos distances | 173 | 113 | 86.6 (16) |
| g1 collides | 11.4 | 7.0 | 18.6 (4) |
| g1 distances | 99.8 | 45.3 | 62.9 (16) |

Both now find the same touching pairs on all three robots; before, they disagreed on 4 talos and 3 g1 pair tests (see the bug below). Accuracy, against the exact distance between the hulls as triangle surfaces, for pairs that are apart:

| | trimkit worst, mm | trimkit 99th percentile, mm | MuJoCo worst, mm | MuJoCo 99th percentile, mm |
|---|---|---|---|---|
| panda | 0.0011 | 0.00019 | 0.0000 | 0.00001 |
| talos | 0.0015 | 0.00008 | 0.0038 | 0.00001 |
| g1 | 0.0020 | 0.00061 | 0.0001 | 0.00001 |

## What changed, step by step

One core, µs per configuration. Each step keeps the ones above it.

| step | talos collides | talos distances | g1 collides | g1 distances |
|---|---|---|---|---|
| before | 125 | 1454 | 99 | 1096 |
| 1. stop GJK once a pair is proven apart | 64 | 1453 | 65 | 1095 |
| 2. start GJK between the hulls' centers | 58 | 1187 | 62 | 924 |
| 3. warm start the climb on each pair's first step | 45 | 936 | 43 | 649 |
| 4. stamps instead of clearing every vertex | 45 | 883 | 42 | 602 |
| 5. MuJoCo's 27-direction grid of starting vertices | 42 | 831 | 41 | 577 |
| 6. a 16 by 16 grid on each face of a cube, and no per-pair clearing | 31 | 633 | 23 | 344 |

1. Coal's collision request has an infinite distance bound by default, so GJK measured every separated pair's distance in full before saying they don't touch. With the bound at 0 it stops at the first plane that separates them. The answers are bit for bit the same, contact depths and points included. This is a coal setting, two lines in `src/collision.cpp`.
2. GJK started every pair along (1, 0, 0). Coal can start it along the line between the two shapes' bounding box centers instead (`BoundingVolumeGuess`), as MuJoCo does, if each shape's box is computed when it is built (`computeLocalAABB`, once per mesh).
3. Coal finds a hull's farthest vertex in a direction by climbing its edges from a starting vertex, and keeps a small table of good starting vertices. It skipped the table on each pair's first step, so every pair climbed from vertex 0 across the hull.
4. Each climb cleared a visited flag for every vertex, so a climb of 2 steps cost the whole hull. A counter that only grows marks visits instead.
5. Coal checked all 14 table entries on every sharp turn. MuJoCo looks up one of 27 cells by the direction and compares 2 vertices.
6. More cells keep helping: a direction picks one of 16 by 16 cells on a face of a cube, 1536 in all. The gain grew with hull size and was still growing at 16, and the table costs 6 KB per hull. Coal also cleared the visited flags of both hulls for every pair, for a function nothing here calls; that function now sizes them itself.

Steps 3 to 6 are one vendored patch (`vendor/patches/coal-support-warm-start.patch`: 57 lines of code added, 93 of coal's removed). Steps 1 and 2 are 6 lines of code in `src/collision.cpp`.

Two things tried did nothing and were dropped: starting from the table only on a pair's first step (slower), and avoiding a square root per step (no change).

## A bug fixed on the way

Step 2 exposed a coal bug. When GJK stops because it is sure two shapes overlap, and nobody asked for the depth, coal reported the gap GJK had left, up to its tolerance of 1e-6, as the distance. Collision then read that as "just apart" and found no contact, however deep the overlap. Before any of this, `collides` already missed 2 of 2222 overlapping pairs on talos and 2 of 1452 on g1, overlapping by up to 2.5 mm; starting between centers made it common between like shapes. The fix sets the distance to 0 (`vendor/patches/coal-gjk-collision-distance.patch`, one line), and it covers meshes against convex solids too. `check/tests/test_collision.py::test_overlapping_convex_solids_collide` fails without it once GJK starts between centers.

A fuzz over 120 scenes (boxes, gridded cubes, spheres, thin ellipsoids, needles, flat sheets, 1e-4 and 1e3 scale, coincident and nested copies, margins 0 to 0.2, up to 4 contacts, bounded distances) found 140 wrong answers before and none after. Distances moved by at most 2e-6. Where contact points moved, they slid across a flat contact patch, where any point is right, and normals moved less than 0.12 degrees.

Not fixed: below about 1e-12 in size, coal's convex distances come back as -1.8e308, before and after. Its tolerances are absolute.

## Threading

The pool was not the problem. Called once per configuration, each call wakes sleeping helper threads, and on talos that costs more than the work: collides takes 35 µs per call on all cores but 7.3 µs per configuration when every configuration goes in one call. A path or a batch of samples should be one call. For distances the pool gets about 7.5 times one core, which is what a spinning pool got from this CPU as well: its 16 efficiency cores and shared performance cores add up to roughly 8 fast ones.

Letting idle helpers spin 50 µs instead of 5 made per-call collides about 20% faster, but burns CPU after every call, and lives in `src/parallel.hpp`. It is left as a measured option.

## The cycloidal assembly

13 meshes placed 72 times (377k faces), none closed, so all of it runs coal's triangle mesh path, which none of this touches. Milliseconds, before / after:

| pulled apart | touching | `collides` | within 1 mm | distances within 10 mm | every distance |
|---|---|---|---|---|---|
| 1x | 119 | 0.39 / 0.42 | 0.46 / 0.47 | 32.5 / 31.8 | 91 / 92 |
| 2x | 67 | 0.46 / 0.38 | 0.23 / 0.21 | 33.0 / 31.5 | 120 / 119 |
| 5x | 25 | 0.14 / 0.08 | 0.23 / 0.20 | 3.7 / 4.4 | 146 / 142 |

MuJoCo has no row here: it collides every mesh as its convex hull, so it answers a different question. On the robots' own meshes it disagreed with trimkit on 8 of 84,000 (panda), 1,403 of 373,800 (talos) and 191 of 43,800 (g1) pair tests.

## Why not MuJoCo itself

- Size. Its library is 6.8 MB stripped. Collision is 132 KB of that; the rest is mostly its model compiler (2.1 MB, C++20, which our toolchain doesn't build), the engine (1.2 MB), its XML parser and qhull. Coal, Eigen and our binding are 474 KB of trimkit's 2.2 MB.
- Every mesh is a convex hull to it.
- Its distance through `mj_collision` with a large margin gets overlap depths wrong (+0.45 for a true -0.02 at a 5 m margin; right at 0.1 m and below). `mj_geomDistance` is right.
- By default it writes every warning to `MUJOCO_LOG.TXT` in the working directory and calls `exit()` on an error. A copy inside trimkit would need its own log handler to sit beside `import mujoco`; its symbols would be hidden by our version script either way.

MuJoCo's Python contacts are numpy arrays per field (`d.contact.dist`, `pos`, `frame`, `geom`), and forces exist only after a physics step. `CollisionResult` already has the geometric part: pairs, signed distances, witness points and normals.

## Reproduce

```
# a venv with trimkit, mujoco, trimesh, yourdfpy, robot_descriptions, python-fcl
python check/bench/collision/trimkit_robot.py talos_description convex 300 DIR
clang++ -O2 -std=c++17 -pthread check/bench/collision/mujoco_kernel.cpp -I$M/include $M/libmujoco.so.* -Wl,-rpath,$M -o mujoco_kernel
./mujoco_kernel DIR
```

`trimkit_robot.py` times trimkit per call and batched, checks it against python-fcl (meshes) or exact hull distance (hulls), and times MuJoCo from Python. With `DIR` it also writes the models, poses and pairs for `mujoco_kernel.cpp`, which times MuJoCo in C on 1 to 32 threads, and trimkit's own inputs (`trimkit.npz`) for timing builds against each other.
