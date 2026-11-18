## panda_description [convex] dof=8 bodies=11 faces=2364 pairs=42 (of 46 non-adjacent) configs=2000
| query | FCL subset | coal subset | python-fcl per pair | trimesh CollisionManager |
|---|---|---|---|---|
| boolean collide, µs/config | 16 | 63 | 56 | 92 (all pairs) |
| all contacts, µs/config | 17 | 64 | | |
| distance, µs/config | 106 | 65 | 163 | |
configs in self-collision: 6%; colliding pairs fcl=262 coal=262 (disagree 0), python-fcl vs fcl subset disagree 0 of 4200
distance |fcl - python-fcl| max 0.00e+00, |coal - python-fcl| max 7.46e-04
## talos_description [convex] dof=32 bodies=52 faces=22770 pairs=1242 (of 1275 non-adjacent) configs=1000
| query | FCL subset | coal subset | python-fcl per pair | trimesh CollisionManager |
|---|---|---|---|---|
| boolean collide, µs/config | 618 | 1711 | 1258 | 862 (all pairs) |
| all contacts, µs/config | 650 | 1724 | | |
| distance, µs/config | 3090 | 1721 | 4076 | |
configs in self-collision: 91%; colliding pairs fcl=6961 coal=6957 (disagree 4), python-fcl vs fcl subset disagree 0 of 124200
distance |fcl - python-fcl| max 0.00e+00, |coal - python-fcl| max 4.02e-02
## talos_description [mesh] dof=32 bodies=52 faces=26134 pairs=1246 (of 1275 non-adjacent) configs=300
| query | FCL subset | coal subset | python-fcl per pair | trimesh CollisionManager |
|---|---|---|---|---|
| boolean collide, µs/config | 268 | 673 | 978 | 1486 (all pairs) |
| all contacts, µs/config | 617 | 1435 | | |
| distance, µs/config | 207016 | 199547 | 226630 | |
configs in self-collision: 84%; colliding pairs fcl=1424 coal=1424 (disagree 0), python-fcl vs fcl subset disagree 0 of 124600
distance |fcl - python-fcl| max 0.00e+00, |coal - python-fcl| max 6.66e-16
terminate called after throwing an instance of 'std::runtime_error'
  what():  Found errors in the Convex mesh:
 The mesh is not watertight.
  Edge between vertices 781 and 782 is shared by 1 faces (should be 2).
  Edge between vertices 781 and 783 is shared by 1 faces (should be 2).
  Edge between vertices 782 and 785 is shared by 1 faces (should be 2).
  Edge between vertices 783 and 785 is shared by 1 faces (should be 2).
/bin/bash: line 1: 3755116 Aborted                    (core dumped) .venv/bin/python -I $S/scripts/robot_bench.py g1_description convex 500 $S/proto
## g1_description [mesh] dof=29 bodies=31 faces=226100 pairs=438 (of 438 non-adjacent) configs=100
| query | FCL subset | coal subset | python-fcl per pair | trimesh CollisionManager |
|---|---|---|---|---|
| boolean collide, µs/config | 168 | 372 | 448 | 4315 (all pairs) |
| all contacts, µs/config | 3948 | 11067 | | |
| distance, µs/config | 228968 | 207415 | 234333 | |
configs in self-collision: 71%; colliding pairs fcl=469 coal=469 (disagree 0), python-fcl vs fcl subset disagree 0 of 43800
distance |fcl - python-fcl| max 0.00e+00, |coal - python-fcl| max 4.44e-16
