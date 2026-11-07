# trimkit

`trimkit` is an experimental packaging effort to make a single MIT-compatible `abi3` package for the stable, lightweight C/C++ [trimesh](https://github.com/mikedh/trimesh) dependencies. This is *not* a mesh library: it is a single "kit" for functional questions a mesh library might ask, that have mature iterative solutions. Hopefully it is easier to maintain and lock down than the previous one-Python-package-per-header. Install with:

```
pip install trimkit
```

It is small (~1MB wheels!), fast, and has simple and strict numpy-only interfaces for users who know exactly what they want. The idea is that `f(numpy) -> numpy` interfaces to well proven and mostly unchanging C/C++ implementations are likely to age better than other options. It isn't and shouldn't be `trimesh` specific in any way, and the only dependency is `numpy`. Meshes are accepted as `vertices: NDArray2D[np.float64], faces: NDArray2D[np.int64]` arrays.

This offers lowest common denominator interfaces to [coal](https://github.com/coal-library/coal) for collision detection, [tinybvh](https://github.com/jbikker/tinybvh) for ray checks and AABB queries, [earcut.hpp](https://github.com/mapbox/earcut.hpp) for polygon tessellation, [VHACD](https://github.com/kmammou/v-hacd) for convex decomposition, [meshoptimizer](https://github.com/zeux/meshoptimizer) for mesh simplification, [xatlas](https://github.com/jpcy/xatlas) for UV atlases, [draco](https://github.com/google/draco) for buffer compression, and a handrolled SVG path string to buffer parser.

Most of the Python packaging ecosystem churn has finally stabilized into `cibuildwheel`, `nanobind`, and `abi3`, and the geometry implementations used here are also mostly stable.

I've cycled through many bindings of polygon tessellation [a](https://pypi.org/project/triangle/) [b](https://pypi.org/project/mapbox-earcut/) [c](https://pypi.org/project/earcutx/), embree [a](https://github.com/scopatz/pyembree) [b](https://pypi.org/project/pyembree/) [c](https://pypi.org/project/embreex/) [d](https://pypi.org/project/embree/) [e](https://pypi.org/project/raybender/), FCL [a](https://pypi.org/project/python-fcl) [b](https://pypi.org/project/fclx/) [c](https://github.com/jf---/python-fcl) [d](https://github.com/benureau/python-fcl), quadric edge collapse [a](https://pypi.org/project/open3d/) [b](https://pypi.org/project/fast-simplification/), VHACD [a](https://pypi.org/project/pyVHACD/) [b](https://pypi.org/project/vhacdx/), and draco [a](https://pypi.org/project/pydracogltf/) [b](https://pypi.org/project/DracoPy/) [c](https://pypi.org/project/dracox/) over the years, and am looking for a way to stop having to rotate every time there's a new Python release and the maintainer (or me) hasn't released a specific wheel for it. 

Most of the underlying geometry implementations in this package are mature, and familiar to trimesh users. Less familiar options are `tinybvh`, `coal`, `meshoptimizer`, and the SVG parser, which is simple enough to just do here. `tinybvh` looks well written, benchmarks look good, packaging embree is quite rough, and it provides an `rtree` alternative that benchmarks about 10x faster on numpy arrays. `coal` is an actively maintained fork of FCL which has more accurate convex distances compared to `python-fcl`. `meshoptimizer` is easier to package, and it never moves vertices.

The weekly Github Actions compromise and overall threat environment has made the zoo less attractive to maintain. With half a dozen packages there's half-a-dozen places we might have not pinned something; this is an attempt to batten down the hatches.

## Packaging

This uses a Python script to construct a `ninja` build file from our declarative `TOML`, and then builds with `clang` and `lld`. Ninja is awesome, and this lets us avoid `cmake`.

This builds wheels for `abi3` on `>= 312`, for Windows, Mac, and Linux. Linux has 4 (`glibc`/`MUSL`,`x86_64`/`arm`), Mac has 2 (`x86_64`/`arm`), and Windows has 3 (`x86_64`/`arm`/`x86`), making the total number of wheels per release 9.

We release `sdist` packages that are tested on free-thread builds, and will release `abi3t` wheels when `nanobind` supports it cleanly. As a policy there will never be per-Python releases; this is an `abi3` packaging effort.

## AI Preferences

- The build system, lints, packaging, hardening, optimizing: AI go wild. Generally try to make it faster, smaller, more tested, cleaner.
- API: never expand without permission.
- Geometry: these algorithms were mostly written by domain experts and have been run by millions of users. The standard of proof is extremely high to make any changes. Start with multiple failing tests, an absolutely solid root-cause, a THIN fix that doesn't require new subsystems, and is justified from multiple angles and multiple cases, including performance data. And present the case coherently to a user: do not make changes to this without an explicit go-ahead, with success criteria established before implementing including failing tests; and a plan on how the change is going to be 50 lines or less to hot geometry code. 

## Place In The Ecosystem

This does not and will never load meshes, parse XML, or support anything that isn't pure data operation. For example it parses SVG path strings into numpy arrays fast; it does not parse the XML holding the path strings. It sits below `trimesh` or similar library in the ecosystem.

## // Claude Slop

<!-- api:start -->
<!-- generated by `uv run scripts generate` from the docstrings and trimkit.toml: don't edit -->

| `trimkit.` | what it is |
|---|---|
| `RayScene` | Meshes, each built once into a double precision BVH, and the only thing rays are cast against.<br>`scene = trimkit.RayScene([(triangle, faces)])` |
| `RayScene.boxes` | The box each instance is culled by, as a query with these instances places it: what the scene thinks it holds.<br>`boxes = trimkit.RayScene([(triangle, faces)]).boxes()` |
| `RayScene.intersect_first` | The nearest triangle each ray hits, from either side.<br>`instances, faces, distances = scene.intersect_first(origins, directions)` |
| `RayScene.intersect_all` | Every triangle each ray hits, from either side.<br>`rays, instances, faces, distances = scene.intersect_all(origins, directions)` |
| `RayScene.intersect_any` | Whether each ray hits anything, stopping at the first triangle found.<br>`hits = scene.intersect_any(origins, directions)` |
| `Boxes` | Axis-aligned boxes built into a BVH for overlap queries: a static, numpy-native replacement for a bulk-loaded R-tree.<br>`tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))` |
| `Boxes.bounds` | The box around every box.<br>`bounds = tree.bounds` |
| `Boxes.boxes` | Every box, exactly as given.<br>`boxes = tree.boxes()` |
| `Boxes.intersect` | Every box overlapping each query box, touching included.<br>`pairs = tree.intersect(np.array([[[0.5, 0.5], [2.5, 2.5]]]))` |
| `triangulate` | Triangulate a polygon with holes.<br>`vertices, faces = trimkit.triangulate(square)` |
| `decompose` | Decompose a closed triangle mesh into approximately convex hulls.<br>`hulls = trimkit.decompose(cube, faces, maximum_hulls=1)` |
| `parametrize` | Cut a mesh into charts and pack them into a UV atlas.<br>`vertex_sources, faces, uv = trimkit.parametrize(triangle, faces)` |
| `simplify` | Collapse edges by quadric error until at most `face_count` faces remain.<br>`vertices, faces, vertex_sources = trimkit.simplify(tetrahedron, faces, 4)` |
| `CollisionScene` | Meshes to place and query against each other. A scene never changes: each query names its own instances, so threads may share one.<br>`scene = trimkit.CollisionScene([(tetrahedron, faces)], convex=True)` |
| `CollisionScene.boxes` | The box each instance is culled by, as a query with these instances places it: what the scene thinks it holds.<br>`boxes = scene.boxes()` |
| `CollisionScene.collides` | The pairs of instances that touch.<br>`pairs = scene.collides(placed, np.zeros(2, dtype=np.int64))` |
| `CollisionScene.distances` | The distance between each pair of instances, and where it is.<br>`distances, points, normals = scene.distances(placed, np.zeros(2, dtype=np.int64))` |
| `__version__` | This package's version.<br>`isinstance(trimkit.__version__, str)` |
| `Mesh` | A triangle mesh as the scenes take it and `decompose` returns it: a (vertices, faces) tuple of (n, 3) float64 vertices and (m, 3) int64 triangle indices into them.<br>`mesh: trimkit.Mesh = (np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64))` |
| `exchange.DracoKind` | What a draco attribute holds, as `draco_encode` takes it.<br>`POSITION, NORMAL, TEXCOORD, COLOR, GENERIC` |
| `exchange.draco_encode` | Compress a glTF triangle primitive, keeping vertex order.<br>`buffer = trimkit.exchange.draco_encode(faces, [triangle.astype(np.float32)], kinds)` |
| `exchange.draco_decode` | Decompress a glTF draco buffer, decoding it once.<br>`faces, attributes = trimkit.exchange.draco_decode(buffer, np.array([0], dtype=np.int64))` |
| `exchange.SvgKind` | What drew an SvgSegment, its `kind` field.<br>`LINE, QUADRATIC, CUBIC, ARC` |
| `exchange.svg_parse` | Parse SVG path data into vertices and one segment per drawing command.<br>`vertices, segments = trimkit.exchange.svg_parse("M 0 0 L 1 1")` |
| `exchange.svg_string` | Write vertices and segments back as SVG path data, the inverse of `svg_parse`.<br>`path = trimkit.exchange.svg_string(*trimkit.exchange.svg_parse("M 0 0 L 1 1 Z"))` |
| `exchange.SvgSegment` | One drawing command of an SVG path, as `svg_parse` returns them and `svg_string` takes them.<br>`trimkit.exchange.svg_parse("M 0 0 L 1 1")[1].dtype == trimkit.exchange.SvgSegment` |

| part | upstream | license |
|---|---|---|
| `bvh` | [jbikker/tinybvh](https://github.com/jbikker/tinybvh) | MIT |
| `earcut` | [mapbox/earcut.hpp](https://github.com/mapbox/earcut.hpp) | ISC |
| `vhacd` | [kmammou/v-hacd](https://github.com/kmammou/v-hacd) | BSD-3-Clause |
| `xatlas` | [jpcy/xatlas](https://github.com/jpcy/xatlas) | MIT |
| `draco` | [google/draco](https://github.com/google/draco) | Apache-2.0 |
| `svg` | ours, [fastfloat/fast_float](https://github.com/fastfloat/fast_float), [jk-jeon/dragonbox](https://github.com/jk-jeon/dragonbox) | MIT, MIT OR Apache-2.0 OR BSL-1.0, Apache-2.0 WITH LLVM-exception OR BSL-1.0 |
| `meshoptimizer` | [zeux/meshoptimizer](https://github.com/zeux/meshoptimizer) | MIT |
| `collision` | [coal-library/coal](https://github.com/coal-library/coal), [libeigen/eigen](https://gitlab.com/libeigen/eigen) | BSD-3-Clause, MPL-2.0 |
| every part | [wjakob/nanobind](https://github.com/wjakob/nanobind), [Tessil/robin-map](https://github.com/Tessil/robin-map) | BSD-3-Clause, MIT |
<!-- api:end -->

Each docstring has the full parameters and returns (`help(trimkit.RayScene.intersect_first)`).

Scalar arguments take Python's numbers or numpy's. Every array argument takes
exactly the dtype shown (float64 finite and within ±1e30) and is never converted:
another dtype or a list is a `TypeError`, and a bad value a `ValueError`, saying what
was expected. A C-contiguous array is read in place, and any other layout copied
once, as `np.ascontiguousarray` would; faces are always copied once, into each
library's index type, and scene constructors keep their own copy of their meshes.
Everything returned is freshly allocated, and the GIL is released while computing.
Answers are dense when you name the rows (rays, pairs) and sparse rows of what was
found otherwise, and `decompose` answers a list of `Mesh`, one per hull. Nothing a
scene answers ever depends on what it was asked before, so threads can share one.

`RayScene`, `Boxes` and `CollisionScene` queries spread big batches
across the cores they may run on, on one pool of workers that exit after 10 s
idle (so a `fork` within that time gets CPython's multi-threaded warning);
`decompose` and `parametrize` use their libraries' own threads, and the rest run
on the calling thread.

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
uv run scripts generate    # regenerate the stubs and this README's generated block
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
  actionlint and the TSan CPython by sha256. Every check runs from the image's
  own environment: nothing on the host can steer it.
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
