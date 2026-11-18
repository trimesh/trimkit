## panda_description [mesh] dof=8 bodies=11 faces=2396 pairs=42 (of 46 non-adjacent) configs=2000
| query | FCL subset | coal subset | python-fcl per pair | trimesh CollisionManager |
|---|---|---|---|---|
| boolean collide, µs/config | 2 | 3 | 40 | 432 (all pairs) |
| all contacts, µs/config | 9 | 18 | | |
| distance, µs/config | 4710 | 4445 | 5094 | |
configs in self-collision: 6%; colliding pairs fcl=254 coal=254 (disagree 0), python-fcl vs fcl subset disagree 0 of 4200
distance |fcl - python-fcl| max 0.00e+00, |coal - python-fcl| max 2.78e-16
## panda_description [convex] dof=8 bodies=11 faces=2364 pairs=42 (of 46 non-adjacent) configs=2000
| query | FCL subset | coal subset | python-fcl per pair | trimesh CollisionManager |
|---|---|---|---|---|
| boolean collide, µs/config | 3 | 10 | 54 | 86 (all pairs) |
| all contacts, µs/config | 3 | 11 | | |
| distance, µs/config | 103 | 63 | 153 | |
configs in self-collision: 6%; colliding pairs fcl=262 coal=262 (disagree 0), python-fcl vs fcl subset disagree 0 of 4200
distance |fcl - python-fcl| max 0.00e+00, |coal - python-fcl| max 7.46e-04
convex distance vs exact triangle distance, fcl: max 1.03e+00, p99 4.52e-06, median 4.16e-17
convex distance vs exact triangle distance, coal: max 1.02e-01, p99 2.10e-07, median 2.78e-17
## talos_description [convex] dof=32 bodies=52 faces=22770 pairs=1242 (of 1275 non-adjacent) configs=1000
| query | FCL subset | coal subset | python-fcl per pair | trimesh CollisionManager |
|---|---|---|---|---|
| boolean collide, µs/config | 100 | 262 | 1238 | 858 (all pairs) |
| all contacts, µs/config | 119 | 289 | | |
| distance, µs/config | 2964 | 1785 | 4008 | |
configs in self-collision: 91%; colliding pairs fcl=6961 coal=6957 (disagree 4), python-fcl vs fcl subset disagree 0 of 124200
distance |fcl - python-fcl| max 0.00e+00, |coal - python-fcl| max 4.02e-02
convex distance vs exact triangle distance, fcl: max 1.12e+00, p99 1.79e-06, median 5.55e-17
convex distance vs exact triangle distance, coal: max 2.65e-01, p99 1.03e-07, median 3.47e-17
## talos_description [mesh] dof=32 bodies=52 faces=26134 pairs=1246 (of 1275 non-adjacent) configs=300
| query | FCL subset | coal subset | python-fcl per pair | trimesh CollisionManager |
|---|---|---|---|---|
| boolean collide, µs/config | 220 | 566 | 969 | 1384 (all pairs) |
| all contacts, µs/config | 551 | 1309 | | |
| distance, µs/config | 203140 | 190223 | 214887 | |
configs in self-collision: 84%; colliding pairs fcl=1424 coal=1424 (disagree 0), python-fcl vs fcl subset disagree 0 of 124600
distance |fcl - python-fcl| max 0.00e+00, |coal - python-fcl| max 6.66e-16
## g1_description [convex] dof=29 bodies=31 faces=60264 pairs=438 (of 438 non-adjacent) configs=500
| query | FCL subset | coal subset | python-fcl per pair | trimesh CollisionManager |
|---|---|---|---|---|
| boolean collide, µs/config | 146 | 257 | 989 | 1199 (all pairs) |
| all contacts, µs/config | 201 | 292 | | |
| distance, µs/config | 2898 | 1880 | 3261 | |
configs in self-collision: 92%; colliding pairs fcl=3516 coal=3513 (disagree 3), python-fcl vs fcl subset disagree 0 of 43800
distance |fcl - python-fcl| max 0.00e+00, |coal - python-fcl| max 2.23e-02
convex distance vs exact triangle distance, fcl: max 1.03e+00, p99 6.82e-06, median 5.55e-17
convex distance vs exact triangle distance, coal: max 1.39e-01, p99 6.32e-07, median 2.78e-17
## g1_description [mesh] dof=29 bodies=31 faces=226100 pairs=438 (of 438 non-adjacent) configs=100
| query | FCL subset | coal subset | python-fcl per pair | trimesh CollisionManager |
|---|---|---|---|---|
| boolean collide, µs/config | 147 | 388 | 451 | 4176 (all pairs) |
| all contacts, µs/config | 3906 | 11205 | | |
| distance, µs/config | 246900 | 205461 | 229637 | |
configs in self-collision: 71%; colliding pairs fcl=469 coal=469 (disagree 0), python-fcl vs fcl subset disagree 0 of 43800
distance |fcl - python-fcl| max 0.00e+00, |coal - python-fcl| max 4.44e-16
