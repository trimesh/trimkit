# trimkit

`trimkit` is an experimental packaging effort to make a single MIT-compatible `abi3` package for the stable, lightweight C/C++ [trimesh](https://github.com/mikedh/trimesh) dependencies. This is *not* a mesh library: it is a single "kit" for functional questions a mesh library might ask, that have mature iterative solutions. Hopefully it is easier to maintain and lock down than the previous one-Python-package-per-project. Install with:

```
pip install trimkit
```

It's small (~1MB wheels!), fast, and has simple and strict numpy-only interfaces for users who know exactly what they want. The idea is that `f(numpy) -> numpy` interfaces to well proven and mostly unchanging C/C++ implementations are likely to age better than other options. It isn't `trimesh` specific in any way, and the only dependency is `numpy`. Meshes are accepted as numpy arrays of `(n, 3) float64` vertices and `(m, 3) int64` faces.

Most of the Python packaging ecosystem churn has finally stabilized into `cibuildwheel`, `nanobind`, and `abi3`, and the geometry implementations used here are also mostly stable. `trimkit` offers lowest common denominator interfaces to [coal](https://github.com/coal-library/coal) for collision detection, [tinybvh](https://github.com/jbikker/tinybvh) for ray checks and AABB queries, [earcut.hpp](https://github.com/mapbox/earcut.hpp) for polygon tessellation, [VHACD](https://github.com/kmammou/v-hacd) for convex decomposition, [meshoptimizer](https://github.com/zeux/meshoptimizer) for mesh simplification, [xatlas](https://github.com/jpcy/xatlas) for UV atlases, [draco](https://github.com/google/draco) for buffer compression, and a hand rolled SVG path string to buffer parser.

There's been many bindings to mostly the same underlying implementations of polygon tessellation ([a](https://pypi.org/project/triangle/) [b](https://pypi.org/project/mapbox-earcut/) [c](https://pypi.org/project/earcutx/)), embree ([a](https://github.com/scopatz/pyembree) [b](https://pypi.org/project/pyembree/) [c](https://pypi.org/project/embreex/) [d](https://pypi.org/project/embree/) [e](https://pypi.org/project/raybender/)), FCL ([a](https://pypi.org/project/python-fcl) [b](https://pypi.org/project/fclx/) [c](https://github.com/jf---/python-fcl) [d](https://github.com/benureau/python-fcl)), quadric edge collapse ([a](https://pypi.org/project/open3d/) [b](https://pypi.org/project/fast-simplification/)), VHACD ([a](https://pypi.org/project/pyVHACD/) [b](https://pypi.org/project/vhacdx/)), and draco ([a](https://pypi.org/project/pydracogltf/) [b](https://pypi.org/project/DracoPy/) [c](https://pypi.org/project/dracox/)) over the years. It would be nice to stop having to rotate when a binding stops releasing and there's a new Python to support.

Most of the underlying geometry implementations in this package are mature and currently used by `trimesh` with different bindings. New to the pod are `tinybvh`, `coal`, `meshoptimizer`, and the SVG parser. `tinybvh` is the biggest change, and was picked because it looks well written, benchmarks look good, packaging embree has been a continuous source of pain, and it also provides an `rtree` alternative that benchmarks about 10x faster on numpy arrays. `coal` is an actively maintained fork of the flexible collision library (currently used by trimesh as `python-fcl`), that has more accurate convex distances. `meshoptimizer` is well-regarded, and easier to package than `open3d` or `fast-simplification`, small, and it never moves vertices.

The weekly GitHub Actions compromise and overall threat environment has made the zoo less attractive to maintain. With half a dozen packages there's half-a-dozen places we might have not pinned something; this is an attempt to lint, fuzz, and lock down in one place.

## Example


```python
import trimkit
import numpy as np

# a tetrahedron
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)

# place 2 tetrahedrons
instance_matrices = np.tile(np.eye(4), (2, 1, 1))
# translate the second one 0.5 units from the origin
instance_matrices[1, 0, 3] = 0.5
# the instance will be 2 copies of mesh `0`
instance_index = np.zeros(2, dtype=np.int64)

# the constructor takes numpy arrays
rays = trimkit.RayScene([(vertices, faces)])

# ray queries also take numpy arrays
origin, direction = np.array([-1.0, 0.2, 0.2]), np.array([1.0, 0.0, 0.0])
# the scene instancing is per-query
ray, instance, face, distance = rays.intersect(
     origin, direction, instance_matrices, instance_index)

# the same constructor arguments work for collisions!
# convex can be per-mesh or every mesh
# if you're using trimesh you probably want this as:
# [m.is_convex for m in meshes]
# if you say it is, but it isn't, it won't break, just the
# returned queries will be wrong 
solids = trimkit.CollisionScene([(vertices, faces)], convex=True)
touching = solids.collides(matrices, index)
distances, points, normals = solids.distances(matrices, index)
```

## Packaging

This uses a Python script to construct a `ninja` build file from a declarative `TOML` file, and then builds with `clang` and `lld`. Ninja is awesome, and this lets us retire `cmake`. It runs `clang` with profile-guided-optimization, and tries to get everything it can out of performance.

This builds `abi3` wheels on `python >= 312`, for Windows, Mac, and Linux. Linux has 4 (`glibc`/`MUSL`,`x86_64`/`aarch64`),  Mac has 2 (`x86_64`/`arm`), and Windows has 3 (`x86_64`/`arm`/`x86`), or 9 wheels per release.

This releases an `sdist` that is tested on free-threaded builds, and will release `abi3t` wheels when `nanobind` supports it cleanly. As a policy there will never be per-Python releases; this is an `abi3` packaging effort.


## API Choices

This does not and will never load meshes, parse XML, or support anything that isn't a pure data operation. For example it parses SVG path strings into numpy arrays fast; it does not parse the XML holding the path strings. It sits below `trimesh` or similar libraries in the Python ecosystem.

- Functions take numpy arrays, don't play precision games, and are biased toward uniformity: everything comes in and goes out as `float64` or `int64`.
- Generic names: functions (outside `trimkit.exchange`) are intended to be doors to the question class, not implementation specific. For example VHACD is the convex decomposition algorithm, but the function is called `decompose`. And if someone vibes the perfect header-only convex decomposition algorithm, we would add an `engine: None | EngineKind = None` argument.
- Honest about work in constructors: why not merge `CollisionScene` and `RayScene`? Well they both have potentially expensive acceleration structures, are different under the hood, and are different classes of question.
- Scene instancing is per-query, NOT saved in scene state. Rather than the usual stateful "scene.set_instances", queries are  `CollisionScene.distances(instance_matrices, instance_index)`. The caller passes instances on every query; the scene holds only the per-mesh acceleration structures.

## AI Preferences

- The build system, lints, packaging, hardening, optimizing: AI go wild. Generally try to make it faster, smaller, more tested, cleaner.
- API: never expand without permission.
- Geometry: these algorithms were mostly written by domain experts and have been run by millions of users. The standard of proof is extremely high for any change. Start with multiple failing tests, an absolutely solid root causea THIN fix that doesn't require new subsystems, and is justified from multiple angles and multiple cases, including performance data. And present the case coherently to a user: do not make changes to this without an explicit go-ahead, with success criteria established before implementing including failing tests; and a plan on how the change is going to be 50 lines or less to hot geometry code. 

## // Claude Slop

Everything below this line is LLM output.

<!-- api:start -->
<!-- generated by `uv run scripts generate` from the docstrings and trimkit.toml: don't edit -->

| `trimkit.` | what it is |
|---|---|
| `RayScene` | Meshes, each built once into a double precision BVH, and the only thing rays are cast against.<br>`scene = trimkit.RayScene([(vertices, faces)])` |
| `RayScene.boxes` | The box each instance is culled by, as a query with these instances places it: what the scene thinks it holds.<br>`boxes = trimkit.RayScene([(vertices, faces)]).boxes()` |
| `RayScene.intersect` | Where each ray crosses the scene's surface, nearest first, from either side.<br>`rays, instances, faces, distances = scene.intersect(origins, directions)` |
| `RayScene.intersect_any` | Whether each ray hits anything, stopping at the first triangle found.<br>`hits = scene.intersect_any(origins, directions)` |
| `Boxes` | Axis-aligned boxes built into a BVH for overlap queries: a static, numpy-native replacement for a bulk-loaded R-tree.<br>`tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))` |
| `Boxes.bounds` | The box around every box.<br>`bounds = tree.bounds` |
| `Boxes.boxes` | Every box, exactly as given.<br>`boxes = tree.boxes()` |
| `Boxes.intersection` | Every box overlapping one query box or each of many, touching included.<br>`hits = tree.intersection(np.array([0.5, 0.5, 2.5, 2.5]))` |
| `triangulate` | Triangulate a polygon with holes.<br>`vertices, faces = trimkit.triangulate(exterior)` |
| `decompose` | Decompose a closed triangle mesh into approximately convex hulls.<br>`hulls = trimkit.decompose(vertices, faces, maximum_hulls=1)` |
| `parametrize` | Cut a mesh into charts and pack them into a UV atlas.<br>`vertex_sources, faces, uv = trimkit.parametrize(vertices, faces)` |
| `simplify` | Collapse edges by quadric error until at most `face_count` faces remain.<br>`vertices, faces, vertex_sources = trimkit.simplify(vertices, faces, 4)` |
| `CollisionScene` | Meshes to place and query against each other. A scene never changes: each query names its own instances, so threads may share one.<br>`scene = trimkit.CollisionScene([(vertices, faces)], convex=True)` |
| `CollisionScene.boxes` | The box each instance is culled by, as a query with these instances places it: what the scene thinks it holds.<br>`boxes = scene.boxes()` |
| `CollisionScene.collides` | The pairs of instances that touch.<br>`pairs = scene.collides(placed, np.zeros(2, dtype=np.int64))` |
| `CollisionScene.distances` | The distance between each pair of instances, and where it is.<br>`distances, points, normals = scene.distances(placed, np.zeros(2, dtype=np.int64))` |
| `__version__` | This package's version.<br>`isinstance(trimkit.__version__, str)` |
| `Mesh` | A triangle mesh as the scenes take it and `decompose` returns it: a (vertices, faces) pair of (n, 3) float64 vertices and (m, 3) int64 triangle indices into them, given as a tuple, list or any sequence of two.<br>`mesh: trimkit.Mesh = (np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64))` |
| `exchange.DracoKind` | What a draco attribute holds, as `draco_encode` takes it.<br>`POSITION, NORMAL, TEXCOORD, COLOR, GENERIC` |
| `exchange.draco_encode` | Compress a glTF triangle primitive, keeping vertex order.<br>`buffer = trimkit.exchange.draco_encode(faces, [vertices.astype(np.float32)], kinds)` |
| `exchange.draco_decode` | Decompress a glTF draco buffer, decoding it once.<br>`faces, attributes = trimkit.exchange.draco_decode(buffer, np.array([0], dtype=np.int64))` |
| `exchange.SvgKind` | What drew an SvgSegment, its `kind` field.<br>`LINE, QUADRATIC, CUBIC, ARC` |
| `exchange.svg_parse` | Parse SVG path data into vertices and one segment per drawing command.<br>`vertices, segments = trimkit.exchange.svg_parse("M 0 0 L 1 1")` |
| `exchange.svg_string` | Write vertices and segments back as SVG path data, the inverse of `svg_parse`.<br>`path = trimkit.exchange.svg_string(*trimkit.exchange.svg_parse("M 0 0 L 1 1 Z"))` |
| `exchange.SvgSegment` | One drawing command of an SVG path, as `svg_parse` returns them and `svg_string` takes them.<br>`trimkit.exchange.svg_parse("M 0 0 L 1 1")[1].dtype == trimkit.exchange.SvgSegment` |

| part | upstream | license |
|---|---|---|
| `bvh` | [jbikker/tinybvh](https://github.com/jbikker/tinybvh), [KhronosGroup/Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | MIT, Apache-2.0 OR MIT |
| `earcut` | [mapbox/earcut.hpp](https://github.com/mapbox/earcut.hpp) | ISC |
| `vhacd` | [kmammou/v-hacd](https://github.com/kmammou/v-hacd) | BSD-3-Clause |
| `xatlas` | [jpcy/xatlas](https://github.com/jpcy/xatlas) | MIT |
| `draco` | [google/draco](https://github.com/google/draco) | Apache-2.0 |
| `svg` | [fastfloat/fast_float](https://github.com/fastfloat/fast_float), [jk-jeon/dragonbox](https://github.com/jk-jeon/dragonbox) | MIT OR Apache-2.0 OR BSL-1.0, Apache-2.0 WITH LLVM-exception OR BSL-1.0 |
| `meshoptimizer` | [zeux/meshoptimizer](https://github.com/zeux/meshoptimizer) | MIT |
| `collision` | [coal-library/coal](https://github.com/coal-library/coal), [libeigen/eigen](https://gitlab.com/libeigen/eigen) | BSD-3-Clause, MPL-2.0 |
| every part | [wjakob/nanobind](https://github.com/wjakob/nanobind), [Tessil/robin-map](https://github.com/Tessil/robin-map) | BSD-3-Clause, MIT |
<!-- api:end -->

Each docstring has the full parameters and returns (`help(trimkit.RayScene.intersect)`).

Scalar arguments take Python's numbers or numpy's. Every array argument is read in
place when it is exactly the dtype shown (float64 finite and within ±1e30) and
C-contiguous, and copied once, as `np.ascontiguousarray` would, in any other layout.
Another dtype numpy casts safely (float32, int32), or a list, is converted to that
dtype once with a `UserWarning` naming the argument: the copy is what you would have
written, and the warning says so. A dtype that would lose values, or anything that
isn't array-like, is a `TypeError`, and a bad value a `ValueError`, saying what was
expected. Whatever indexes memory is always copied once and the copy checked: faces,
into each library's index type, and a query's `instance_index` and `pairs`. Scene
constructors keep their own copy of their meshes.
Everything returned is freshly allocated, and the GIL is released while computing.
Answers are dense when you name the rows (`intersect_any`'s rays, `distances`'
pairs) and sparse rows of what was found otherwise (hits, touching pairs,
overlapping boxes), and `decompose` answers a list of `Mesh`, one per hull. Nothing a
scene answers ever depends on what it was asked before, so threads can share one.

`RayScene`, `Boxes` and `CollisionScene` queries spread big batches, and
`decompose` its hull splits and merge costs, across the cores they may run on
(their affinity, within a cgroup's CPU quota), on one pool of workers that exit
after 10 s idle (so a `fork` within that time gets CPython's multi-threaded
warning); `parametrize` uses xatlas's own threads, and the rest run on the
calling thread.

`RayScene(meshes, use_gpu=True)` traces on a Vulkan device instead: the first
GPU that computes in double precision, found through the system's Vulkan loader
at run time (no wheel links or needs Vulkan, and Vulkan 1.0 is enough). It holds
the same trees and runs a compute shader that makes every decision tinybvh's
traversal makes, most of them in float32 behind a proven error bound and the
rest with tinybvh's own float64 operations (`docs/trace.md`), so it answers the
same bytes the CPU does. Rays go as given: a `(3,)` origin or direction is one
row for every ray, and an array goes as its dtype: float32 at half the bytes
of float64, widened on the device. When no such device can be used,
at construction or later, the scene warns once (a `UserWarning` naming why)
and the
CPU answers. `tests/golden.json` holds the hashed answers of an exact corpus
(dyadic coordinates, integer-length directions, lattice meshes) that every
platform and the device are checked against: the same answers everywhere.

What the device is worth depends on the rays. `python bench/scene.py` times a
real assembly (`tests/models/cycloidal.npz`: 13 meshes placed 72 times, 377k
faces) as loaded and pulled apart, so fewer rays hit anything. A batch is one
camera's rays, one origin for all of them; times are the median over eight
cameras, in milliseconds a batch, twenty CPU threads against an RTX 3090:

| rays | pulled apart | hitting | nearest: CPU ms | device ms | ratio | any hit: CPU ms | device ms | ratio | every crossing: CPU ms | device ms | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 1x | 12% | 1.13 | 0.65 | 1.7x | 0.72 | 0.27 | 2.7x | 4.35 | 13.73 | 0.3x |
| 10,000 | 2x | 11% | 1.01 | 0.59 | 1.7x | 0.65 | 0.30 | 2.1x | 5.11 | 6.83 | 0.7x |
| 10,000 | 5x | 3% | 0.60 | 0.44 | 1.4x | 0.42 | 0.24 | 1.7x | 0.50 | 1.43 | 0.4x |
| 99,856 | 1x | 12% | 2.05 | 1.22 | 1.7x | 1.30 | 0.76 | 1.7x | 16.40 | 18.99 | 0.9x |
| 99,856 | 2x | 11% | 1.79 | 1.22 | 1.5x | 1.03 | 0.92 | 1.1x | 6.89 | 10.22 | 0.7x |
| 99,856 | 5x | 3% | 1.05 | 1.12 | 0.9x | 0.74 | 0.89 | 0.8x | 1.92 | 3.01 | 0.6x |
| 1,000,000 | 1x | 12% | 15.42 | 5.81 | 2.7x | 15.50 | 5.17 | 3.0x | 102.86 | 54.31 | 1.9x |
| 1,000,000 | 2x | 11% | 12.84 | 5.87 | 2.2x | 7.91 | 4.74 | 1.7x | 46.61 | 27.28 | 1.7x |
| 1,000,000 | 5x | 3% | 7.16 | 5.06 | 1.4x | 5.16 | 4.82 | 1.1x | 12.47 | 10.46 | 1.2x |
| 4,000,000 | 1x | 12% | 61.71 | 55.58 | 1.1x | 45.36 | 19.69 | 2.3x | 446.58 | 142.95 | 3.1x |
| 4,000,000 | 2x | 11% | 46.27 | 23.26 | 2.0x | 32.19 | 20.23 | 1.6x | 191.07 | 86.65 | 2.2x |
| 4,000,000 | 5x | 3% | 34.02 | 20.78 | 1.6x | 24.05 | 20.71 | 1.2x | 49.97 | 34.44 | 1.5x |

That run shared the machine with other work (a load of about 10 on its twenty
threads). Three things hold either way.
A camera's float64 directions are 24 MB a million rays, and moving them to the
device (about 0.17 ms a megabyte here, by any route measured, and nearly all
of a query's time: tracing is a few percent of it) takes about as long as the
CPU takes to find that most of them miss, so a sparse scene gains least;
float32 directions go at half that. Every crossing
runs each ray's crossings one after another on its own lane, so a small batch
with long rays is slower on the device than on the CPU, and a large one
faster. And the device's answers are the CPU's, which the bench checks before
it times anything.

The same script times the same assembly as a `CollisionScene`: every pair of
its 72 instances in one call, as triangle meshes (none of them is closed, so
none is convex), in milliseconds, the best of five calls:

| pulled apart | pairs | touching | `collides` ms | within 1 mm ms | distances within 10 mm ms | every distance ms |
|---|---|---|---|---|---|---|
| 1x | 2,556 | 119 | 0.55 | 0.56 | 29.9 | 97 |
| 2x | 2,556 | 67 | 0.38 | 0.31 | 36.1 | 147 |
| 5x | 2,556 | 25 | 0.17 | 0.34 | 4.8 | 151 |

`collides` culls by each instance's box, then by each mesh's hierarchy, so
which of 2,556 pairs touch among 377k placed faces is about half a
millisecond; a margin asks which come within that distance. `distances`
measures every pair it is asked about, and `upper_bound` lets it stop at pairs
farther apart than that, which is what a planner asks.

### Example

`examples/ur10.py`: a UR10e with a box gripper reaches into a cubby
for a block. The planner is one function, forward kinematics then one
`CollisionScene.distances` call over every waypoint of a joint-space spline, handed
to `scipy.optimize.minimize`, its gradient from the normals and closest points
`distances` returns.

```
planned in 92 ms, 59 evaluations of 101 waypoints
least clearance over 2001 waypoints: 18.4 mm
```

### Against what trimesh used before

| `trimkit.` | replaces | measured against it |
|---|---|---|
| `exchange.svg_parse`, `exchange.svg_string` | `svg.path` | 56x faster parsing 60k commands |
| `Boxes` | `rtree` | 8-10x faster: build 100k boxes, query 20k |
| `CollisionScene` | `python-fcl` | 6x faster: every touching pair of 200 spheres, through trimesh's `CollisionManager`; convex distances more accurate |
| `exchange.draco_encode`, `exchange.draco_decode` | `DracoPy` | 3.6x faster encoding, 1.7x decoding |
| `RayScene` | `embreex` | 3-3.6x faster on 1M rays, in float64 and watertight |
| `simplify` | `fast-simplification` | 2.9x faster, and thread safe |
| `parametrize` | `xatlas` | 1.3x faster |
| `decompose` | `vhacdx` | the same speed: the same library |
| `triangulate` | `mapbox-earcut` | the same speed: the same library |

Measured 2026-11-06 with `bench/versus.py` on the same numpy inputs: best of several
runs on 8 cores of a busy machine, release build without PGO. A range is two runs.

### Wheels

One `cp312-abi3` wheel per platform serves every CPython from 3.12, so a new CPython
needs no new release: Linux on glibc (`manylinux_2_28`) and musl (`musllinux_1_2`) on
x86_64 and aarch64, macOS 11+ on x86_64 and arm64, and Windows on x64, x86 and arm64,
every platform numpy publishes wheels for. 9 wheels and an sdist.

Anywhere else the sdist builds with clang 19 or newer and lld: no machine is refused,
and `-Csystem=eigen,fast_float,robin_map` takes those from the system. That includes
free-threaded Python, where `uv run scripts sdist` builds and tests it for every release:
it gets no wheels until nanobind reaches `abi3t` without a second package, as this
package has no per-version wheels.
`uv run scripts distro` generates from `trimkit.toml` the `debian/` directory and Fedora
spec a distribution's maintainer commits, into `build/distro/`.

## Development

```
uv run scripts check       # the gate: build, tests, static checks, sanitizers, fuzzers
uv run scripts update      # every upstream and dependency to its newest
uv run scripts format      # our C++, as scripts/clang-format.yaml says
uv run scripts generate    # regenerate the shaders' SPIR-V, the stubs, this README's generated block and tests/golden.json
uv run scripts wheel       # this platform's release wheels
uv run scripts distro      # Debian and Fedora packaging (--run: build, test and lint them)
```

`check` runs before pushing, and prints each stage's minutes. CI only releases:
every push to main runs `.github/workflows/release.yml`, which lints, builds and
tests the sdist and every wheel with these same commands, then publishes the
version `pyproject.toml` names (one PyPI already has is left as it is). A push to
`release/*` does all of it but the upload. Everything but `lock` and `wheel` runs
inside a pinned Debian trixie image (`docker/Dockerfile`), with the sources mounted
read-only unless the command writes them, so the host needs only Docker and uv.

`trimkit.toml` is the one hand-edited build file: every upstream (pin, patches,
and how it compiles, each deviation with its reason), every part of the one
extension, its wheel-size budget, and the toolchains a release accepts. There is
no CMake: a wheel is one shared library from a fixed list of sources, so the
backend (`scripts/backend.py`) has `scripts/builder.py` write that list as one
ninja file. To see exactly what a build does, read its `build.ninja`.

Every build is LLVM: clang with ThinLTO and PGO, linked by lld with safe
identical-code folding, or on macOS by Xcode's linker. The builder asks clang for
its own linker, archiver and `llvm-profdata` and stops unless they are all one
LLVM, since ThinLTO's bitcode and PGO's profiles are only read by their own.
`-Cstrict=true`, which the gate and the release use, also stops on a clang newer
than `[toolchain]` has passed with.

## Vendoring

`vendor/` holds what a build needs and nothing else, committed, so nothing is
downloaded to build, and this keeps building if GitHub goes away. One rule picks
the files, at the upstream's own paths: what `keep` names in `trimkit.toml`, and
whatever a preprocessor dependency scan of the C++ compiling it reaches on every
wheel target, where a reached header brings its same-named `sources`. `check`
rescans, so it can't drift. It's about 10 MB in a checkout, half of it Eigen's
headers, and 2.3 MB as an sdist.

`uv run scripts fetch` re-derives all of it from the pinned commits: `git status
vendor/` then shows nothing, so anyone can check that `vendor/` is what it says.
Our changes are unified diffs in `vendor/patches/`, each tagged `fix` or `perf`
with its reason, listed by sha256 and applied strictly, hunk for hunk.

## Supply chain

Every input that reaches a wheel, and what pins it:

- Compiled code: vendored, each upstream pinned by commit and by one `tree_sha256`
  of everything vendored for it, patches applied, which `check` verifies.
- Python tools and dependencies: hash-locked in `uv.lock`, with a one-week
  cooldown (`exclude-newer`); builds install them with `--require-hashes` from
  requirements exported from the lock, and cibuildwheel itself comes from it.
- The image: base by digest, Debian packages by snapshot date, uv by digest, and
  the TSan CPython by sha256. Every check runs from the image's own
  environment: nothing on the host can steer it.
- CI: actions pinned by commit; cibuildwheel pins its Linux images by digest and
  its macOS and Windows Pythons by sha256. Dependabot moves pins with the same
  cooldown.

Pinned by version or signature only, as nothing finer exists for them: clang and
lld from the manylinux and musllinux images' own package repositories (signed),
the macOS and Windows runners' compilers and SDKs, each held to `[toolchain]`;
cibuildwheel's own pip, build, auditwheel and delocate; and, for someone building
the sdist with build isolation, ninja from PyPI by version.

## Licensing

A wheel's `License-Expression` names everything compiled in, and `check` derives
it from `trimkit.toml`. All of it is permissive (MIT, ISC, BSD, Apache-2.0,
BSL-1.0) except Eigen's headers, MPL-2.0: file-level copyleft that binds only
those files, which are vendored unmodified (`check` refuses a patch to them),
built with `EIGEN_MPL2_ONLY`, and shipped as source in the sdist. Every license
ships in the wheel, with a PEP 770 SBOM generated from `trimkit.toml`.
