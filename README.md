# trimkit
 
`trimkit` is an experimental packaging effort to make a single MIT-compatible `abi3` package for the stable, lightweight C/C++ [trimesh](https://github.com/mikedh/trimesh) dependencies. This is *not* a mesh library: it is a single "kit" for functional questions a mesh library might ask. It provides bindings to mature, stable implementations of iterative algorithms that are not vectorizable and untenably slow in Python, like quadric edge collapse, collisions, marching cubes, and ray checks. Hopefully it is easier to maintain and lock down than the previous one-Python-package-per-project, and it saves 100MiB+ for every `trimesh[easy,recommend]` install. Install with:

```
pip install trimkit
```

It's small (~1MB wheels!), fast, and has simple and strict `numpy`-only interfaces for users who know exactly what they want. The idea is that `f(numpy) -> numpy` interfaces to well proven and mostly unchanging C/C++ implementations are likely to age better than other options. It isn't `trimesh` specific in any way, and the only dependency is `numpy`. Meshes are accepted as numpy arrays of `(n, 3) float64` vertices and `(m, 3) int64` faces.


### Motivation

There have been many bindings to mostly the same underlying implementations of polygon tessellation ([a](https://pypi.org/project/triangle/) [b](https://pypi.org/project/mapbox-earcut/) [c](https://pypi.org/project/earcutx/)), embree ([a](https://github.com/scopatz/pyembree) [b](https://pypi.org/project/pyembree/) [c](https://pypi.org/project/embreex/) [d](https://pypi.org/project/embree/) [e](https://pypi.org/project/raybender/)), FCL ([a](https://pypi.org/project/python-fcl) [b](https://pypi.org/project/fclx/) [c](https://github.com/jf---/python-fcl) [d](https://github.com/benureau/python-fcl)), quadric edge collapse ([a](https://pypi.org/project/open3d/) [b](https://pypi.org/project/fast-simplification/)), VHACD ([a](https://pypi.org/project/pyVHACD/) [b](https://pypi.org/project/vhacdx/)), and draco ([a](https://pypi.org/project/pydracogltf/) [b](https://pypi.org/project/DracoPy/) [c](https://pypi.org/project/dracox/)) over the years. It would be nice to stop having to rotate when a binding stops releasing and there's a new Python to support.

The regularly scheduled GitHub Actions compromise and overall threat environment has made the zoo less attractive to maintain. With half a dozen packages there's half-a-dozen places we might have not pinned something; this is an attempt to lint, fuzz, and lock down in one place.

Most of the Python packaging ecosystem churn has finally stabilized into `cibuildwheel`, `nanobind`, and `abi3`. The geometry implementations used here are also mostly stable, and `trimesh` currently calls one or more package options that bind them to Python. `trimkit` offers lowest common denominator interfaces to [coal](https://github.com/coal-library/coal) for collision detection, [tinybvh](https://github.com/jbikker/tinybvh) for ray checks and AABB queries, [earcut.hpp](https://github.com/mapbox/earcut.hpp) for polygon tessellation, [VHACD](https://github.com/kmammou/v-hacd) for convex decomposition, [meshoptimizer](https://github.com/zeux/meshoptimizer) for mesh simplification, [xatlas](https://github.com/jpcy/xatlas) for UV atlases, [draco](https://github.com/google/draco) for buffer compression, and a hand rolled SVG path string to buffer parser.

Implementations that are new to the trimesh dependency tree are `tinybvh`, `coal`, `meshoptimizer`, the SVG path string parser, and the marching cubes implementation. `tinybvh` is the biggest change, and was picked because it looks well written, benchmarks look good, packaging embree has been a continuous source of pain, and it also provides an `rtree` alternative that benchmarks about 10x faster on numpy arrays. `coal` is an actively maintained fork of the flexible collision library (currently used by trimesh as `python-fcl`), that has more accurate convex distances. `meshoptimizer` is well-regarded, small, and easier to package than `open3d` or `fast-simplification`.

### Example


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
# which pairs touch, and with `return_distance` where: faces, depths, points, normals
touching = solids.collides(matrices, index).pairs
contacts = solids.collides(matrices, index, return_distance=True)
# every pair's signed distance, witness points, normal and closest faces
measured = solids.distances(matrices, index)
distances, points, normals = measured.distances, measured.points, measured.normals
```

### Packaging

This uses a Python script to construct a `ninja` build file from a declarative `TOML` file, and then builds with `clang` and `lld`. Ninja is awesome, and this lets us retire `cmake`. It runs `clang` with profile-guided-optimization, and tries to get everything it can out of performance and bundle size.

This builds `abi3` wheels on `python >= 312`, for Windows, Mac, and Linux. Linux has 4 (`glibc`/`MUSL`,`x86_64`/`aarch64`),  Mac has 2 (`x86_64`/`arm`), and Windows has 3 (`x86_64`/`arm`/`x86`), or 9 wheels per release.

This releases an `sdist` that is tested on free-threaded builds, and will release `abi3t` wheels when `nanobind` supports it cleanly. As a policy there will never be per-Python releases; this is an `abi3` packaging effort.


### API Design Choices

This does not and will never load meshes, parse XML, or support anything that isn't a tight data operation. For example: this parses SVG path strings into numpy arrays fast; it does not parse the XML holding the path strings. It sits below `trimesh` or similar libraries in the Python ecosystem.

- Functions take numpy arrays, don't play precision games, and are biased toward uniformity: everything comes in and goes out as `float64` or `int64`.
- Generic names: functions (outside `trimkit.exchange`) are intended to be doors to the question class, not implementation specific. For example VHACD is the convex decomposition algorithm, but the function is called `decompose`. And if someone vibes the perfect header-only convex decomposition algorithm, we would add an `engine: None | DecomposeAlgorithm = None` argument.
- Honest about work in constructors: why not merge `CollisionScene` and `RayScene`? Well they both have expensive acceleration structures, are different under the hood, and are different classes of question.
- Scene instancing is per-query, NOT saved in scene state. Rather than the usual stateful "scene.set_instances", queries are `CollisionScene.distances(instance_matrices, instance_index)`. The caller passes instances on every query; the scene holds only the per-mesh acceleration structures.

## AI Preferences

- The build system, lints, packaging, hardening, optimizing: AI go wild. Generally try to make it faster, smaller, more tested, cleaner.
- API: never expand without permission.
- Geometry: these algorithms were mostly written by domain experts and have been run by millions of users. The standard of proof is extremely high for any change. Start with multiple failing tests, an absolutely solid root causea THIN fix that doesn't require new subsystems, and is justified from multiple angles and multiple cases, including performance data. And present the case coherently to a user: do not make changes to this without an explicit go-ahead, with success criteria established before implementing including failing tests; and a plan on how the change is going to be 50 lines or less to hot geometry code. 

Basically: `nanobind` and the underlying geometry header should be doing 99% of the work. The main exceptions to this are these vibed out pieces of core logic:
- `parallel.hpp`
  - This provides a multicore `map`. In rust `rayon::par_iter.map` is incredible. But this project is a C/C++ binding, and I couldn't find a thin "multiplatform rayon but for C++". This is the piece that is both most suspicious from a system design perspective, but also the most important for performance. Claude produced a TLA+ spec for this, and is checking against it. But if you were going to deep review one thing it should be the header, the TLA+ specification, and the hardness of the asserts in the GenMC and scheduler tests. 
- `svgpath.hpp`
  - This is very very easy to do in a C++ loop, and extremely slow in Python. The SVG path string is the lowest-common-denominator of 2D curve representation, and this parses it to and from a structured numpy array.
- `contour.hpp`
  - Marching cubes is a simple algorithm, `scikit-image` is great but it's quite heavy and was used in trimesh ONLY for marching cubes. Also this way we can support sparse, run-length, and dense voxels with one function signature, reducing call complexity from Python. Basically voxels in Python can be pure `numpy`: `(n, 3) int` for sparse voxels, `(n, 4) int` for sparse voxel runs (the last column is the run length along the Z column), and full `(n, m, p) bool` dense voxel matrices. 
- the tinybvh vulkan ray query
  - This adds a `use_gpu: bool` argument to rays. I did it because `tinybvh` is still doing most of the work, and because GPU rays are so much faster. On my machine the check is 150x faster, but readback/overhead brings it to only a 2x win. But the future is GPU and this can probably be improved, with the same API, and gives a nice Vulkan-only foundation. A lot of the slowness is because I insisted on float64 rays. Trimesh has had years and years of "the ray check was wrong" issues, which are pretty much all "float32 has a ULP of around 1e-7 for normalized coordinates which really isn't that small in the scheme of things." This pipeline does an f32 pass with an f64 refine which is only a 20% performance hit over pure f32. And the tests insist on bit-identical results to the CPU path.


## // Claude Slop

Everything below this line is LLM output.

<!-- api:start -->
<!-- generated by `uv run scripts generate` from the docstrings and trimkit.toml: don't edit -->

## Libraries included

Each part of trimkit wraps existing libraries, compiled into the wheel:

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
| `contour` | none: ours | MIT |
| every part | [wjakob/nanobind](https://github.com/wjakob/nanobind), [Tessil/robin-map](https://github.com/Tessil/robin-map) | BSD-3-Clause, MIT |

## API

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
| `CollisionResult` | What a collision or distance query found: one row per pair of instances, or per contact of a pair, as `CollisionScene.collides` and `CollisionScene.distances` return it.<br>`result = trimkit.CollisionResult(np.array([[0, 1]], dtype=np.int64))` |
| `CollisionResult.pairs` | The instances of each row.<br>`pairs = result.pairs` |
| `CollisionResult.faces` | The face of each instance's mesh each row is at, or None.<br>`faces = result.faces` |
| `CollisionResult.distances` | The signed distance of each row, or None.<br>`distances = result.distances` |
| `CollisionResult.points` | The witness point on each instance of each row, or None.<br>`points = result.points` |
| `CollisionResult.normals` | The unit normal of each row, or None.<br>`normals = result.normals` |
| `CollisionScene` | Meshes to place and query against each other. A scene never changes: each query names its own instances, so threads may share one.<br>`scene = trimkit.CollisionScene([(vertices, faces)], convex=True)` |
| `CollisionScene.boxes` | The box each instance is culled by, as a query with these instances places it: what the scene thinks it holds.<br>`boxes = scene.boxes()` |
| `CollisionScene.collides` | The pairs of instances that touch, and optionally where.<br>`result = scene.collides(placed, np.zeros(2, dtype=np.int64))` |
| `CollisionScene.distances` | The distance between each pair of instances, and where it is.<br>`result = scene.distances(placed, np.zeros(2, dtype=np.int64))` |
| `contour` | The closed surface of a set of voxels, as rows or as a volume.<br>`vertices, faces, vertex_sources = trimkit.contour(runs)` |
| `__version__` | This package's version.<br>`isinstance(trimkit.__version__, str)` |
| `Mesh` | A triangle mesh as the scenes take it and `decompose` returns it: a (vertices, faces) pair of (n, 3) float64 vertices and (m, 3) int64 triangle indices into them, given as a tuple, list or any sequence of two.<br>`mesh: trimkit.Mesh = (np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64))` |
| `exchange.DracoKind` | What a draco attribute holds, as `draco_encode` takes it.<br>`POSITION, NORMAL, TEXCOORD, COLOR, GENERIC` |
| `exchange.draco_encode` | Compress a glTF triangle primitive, keeping vertex order.<br>`buffer = trimkit.exchange.draco_encode(faces, [vertices.astype(np.float32)], kinds)` |
| `exchange.draco_decode` | Decompress a glTF draco buffer, decoding it once.<br>`faces, attributes = trimkit.exchange.draco_decode(buffer, np.array([0], dtype=np.int64))` |
| `exchange.SvgKind` | What drew an SvgSegment, its `kind` field.<br>`LINE, QUADRATIC, CUBIC, ARC` |
| `exchange.svg_parse` | Parse SVG path data into vertices and one segment per drawing command.<br>`vertices, segments = trimkit.exchange.svg_parse("M 0 0 L 1 1")` |
| `exchange.svg_string` | Write vertices and segments back as SVG path data, the inverse of `svg_parse`.<br>`path = trimkit.exchange.svg_string(*trimkit.exchange.svg_parse("M 0 0 L 1 1 Z"))` |
| `exchange.SvgSegment` | One drawing command of an SVG path, as `svg_parse` returns them and `svg_string` takes them.<br>`trimkit.exchange.svg_parse("M 0 0 L 1 1")[1].dtype == trimkit.exchange.SvgSegment` |
<!-- api:end -->

`help(trimkit.RayScene.intersect)` prints the full arguments and return values of any function.

### Arguments and results

Arrays go in and come out as 64-bit floats or 64-bit integers. Each docstring says which. What happens to an array you pass depends on its type:

| you pass | what happens |
|---|---|
| the exact type, in one unbroken block of memory | read in place, no copy |
| a list, or 32-bit floats or integers | converted once, with a `UserWarning` naming the argument |
| a type that would lose values, or something that is not an array | `TypeError` |
| a bad value (floats must be finite and within ±1e30) | `ValueError`, saying what was expected |

Two things are always copied. Anything used as an index (faces, `instance_index`, `pairs`) is copied and range-checked. And a scene keeps its own copy of the meshes you build it from.

Results are always new arrays. If you name the rows (the rays of `intersect_any`, the pairs of `distances`), you get one answer per row. If you don't, you get one row per thing found: each hit, each touching pair, each overlapping box. `decompose` returns a list of meshes, one per hull.

Scenes are safe to share between threads. A scene never changes after it is built, and every function releases the GIL (Python's lock that lets only one thread run Python code at a time) while it computes.

### Pickling

Pickling is Python's way to save an object or send it to another process. A pickled scene stores only what you passed to its constructor: `Boxes` stores its `boxes`, `RayScene` its `meshes` and `use_gpu`, `CollisionScene` its `meshes` and which are `convex`.

The search trees are not stored. (A scene's search tree, called a BVH, is a tree of boxes that lets a query skip most of a mesh.) Loading a pickle builds them again, so loading takes as long as building the scene did, and the loaded scene gives the same answers bit for bit. `copy.deepcopy` works the same way.

Because a pickle is only constructor arguments, it loads in any trimkit version whose constructor takes the same names. A `RayScene` pickled with `use_gpu=True` still loads on a machine with no usable graphics card: it warns once and uses the CPU.

### Parallelism

This is a library that operates on numpy arrays, and thus is a natural fit for splitting work across every CPU core. This is done by one file, `src/parallel.hpp`. It keeps a pool: helper threads that sleep until there is work, then each takes a slice of the batch. We wrote it ourselves mainly so that no wheel has to carry someone else's threading library. The usual choices are oneTBB (Intel's threading library) and OpenMP (threading built into compilers). Both are large, and both expect to be installed once per machine. Packed into a Python package, a second copy can land in the same process as another package's copy, and the two compete for the same cores.

The second reason is fork, which most users never call themselves. Python's `multiprocessing`, web servers like gunicorn and task queues like Celery make their worker processes by copying the running process, and that copy is a fork. Only one thread survives the copy. A pool's helper threads are gone, so a library that didn't plan for this hangs in the worker, with no error to explain it. oneTBB's documentation says it does not support fork, and GCC's OpenMP can hang this way. Our pool resets itself in the copy and starts new helpers. What we give up is oneTBB's many years of use in real programs. To make up for that, ours is checked three ways: a tool that tries every possible thread ordering on the real code (GenMC), a written model of the design checked the same way (TLA+), and bugs planted on purpose that the tests must catch.

| | ours | oneTBB | OpenMP | a single-file pool |
|---|---|---|---|---|
| ships inside the wheel | yes: one header, compiled in | a separate shared library; its build warns against linking it in | needs the compiler's runtime library; Apple's clang has none | yes |
| works in a forked copy | yes: a call cut off by the fork fails cleanly | no: "does not support fork()" | GCC's can hang | usually not |
| track record | new, about 1,500 lines, model checked | many years in real programs | many years in real programs | varies |

How it does, on 8 cores, against a plain loop and the best of three pools written by hand (`check/bench/parallel.cpp`, which `uv run scripts bench` runs; microseconds, the best run, 2026-11-18):

| work | plain loop | ours | best by hand |
|---|---|---|---|
| tiny: 100 items of 50 ns each | 3 | 4 | 11 |
| even: 100,000 items of 1 µs each | 90,363 | 11,367 | 11,374 |
| uneven: 10,000 items of 1 µs, the first eighth 20 times slower | 30,937 | 3,896 | 4,006 |

A tiny call wakes no helper thread, which is why it costs about what the plain loop costs.

The pool is used by big batches in `RayScene`, `Boxes` and `CollisionScene` queries, and by the heavy inner steps of `decompose` and `parametrize`. Everything else runs on the calling thread, and no vendored library starts threads of its own.

The pool only uses cores the process is allowed to use, and it respects a Linux container's CPU limit. Helper threads exit after 10 seconds with no work. If you fork within those 10 seconds, CPython prints its warning about forking a process that has threads.

### GPU rays

`RayScene(meshes, use_gpu=True)` traces rays on a graphics card (a GPU) instead of the CPU.

It is always safe to turn on. If the machine has no usable card, the scene prints one warning that says why (a `UserWarning` starting with `use_gpu:`) and the CPU answers instead. You get the same results either way, only slower, and nothing crashes. The same thing happens if the card stops working later, in the middle of a program.

The only way trimkit talks to a card is Vulkan, a standard interface that the card's driver provides. There is no CUDA, OpenCL or Metal path. A card is usable if its Vulkan driver can compute with 64-bit floats, and Vulkan 1.0 is enough. Nothing extra is installed with the wheel: Vulkan is looked up when the scene is built. To see what your machine has:

- Linux: run `vulkaninfo --summary`. It comes in a package usually named `vulkan-tools`. If it lists your card, trimkit can see it too.
- Windows: run `vulkaninfo --summary` in a terminal. Most graphics drivers install it. If the command is missing, update the graphics driver.
- macOS: Vulkan only exists through an add-on layer (MoltenVK), and Apple's cards generally can't compute in 64-bit floats, so expect the warning and the CPU.

The GPU returns exactly the same bytes as the CPU. It walks the same search trees and makes every decision the CPU code makes. Most decisions are made in fast 32-bit floats, with a proof that the rounding error can't change the outcome. The few that could change are redone in 64-bit (`check/verify/gpu.md` has details). To hold this, `check/tests/golden.json` stores hashed answers for a set of exact inputs, and every platform and the GPU must reproduce them.

Rays are sent to the card as you give them. A single `(3,)` origin or direction is shared by every ray. 32-bit arrays are sent as 32-bit, at half the size, and widened on the card.

Whether the GPU is faster depends on the rays. `python check/bench/scene.py` times a real assembly (`check/tests/models/cycloidal.npz`: 13 meshes placed 72 times, 377k faces). It runs the assembly as loaded (1x), and with its parts spread 2 and 5 times farther apart so that fewer rays hit anything. A batch is all the rays of one camera, from one origin. Times are milliseconds per batch, the median over eight cameras: the 32 threads of an i9-14900K against an RTX 5070, on an idle machine (2026-11-17).

| rays | pulled apart | hitting | nearest: CPU ms | device ms | ratio | any hit: CPU ms | device ms | ratio | every crossing: CPU ms | device ms | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 1x | 12% | 0.24 | 0.55 | 0.4x | 0.13 | 0.25 | 0.5x | 1.29 | 11.53 | 0.1x |
| 10,000 | 2x | 11% | 0.17 | 0.43 | 0.4x | 0.09 | 0.28 | 0.3x | 0.73 | 5.59 | 0.1x |
| 10,000 | 5x | 3% | 0.11 | 0.31 | 0.4x | 0.08 | 0.21 | 0.4x | 0.19 | 1.04 | 0.2x |
| 99,856 | 1x | 12% | 1.15 | 0.82 | 1.4x | 0.87 | 0.51 | 1.7x | 7.72 | 13.17 | 0.6x |
| 99,856 | 2x | 11% | 0.98 | 0.71 | 1.4x | 0.74 | 0.63 | 1.2x | 7.70 | 8.62 | 0.9x |
| 99,856 | 5x | 3% | 0.53 | 0.65 | 0.8x | 0.35 | 0.45 | 0.8x | 1.47 | 2.15 | 0.7x |
| 1,000,000 | 1x | 12% | 8.37 | 4.43 | 1.9x | 5.13 | 3.50 | 1.5x | 53.66 | 48.95 | 1.1x |
| 1,000,000 | 2x | 11% | 6.26 | 3.98 | 1.6x | 3.93 | 3.54 | 1.1x | 27.49 | 28.10 | 1.0x |
| 1,000,000 | 5x | 3% | 3.77 | 3.56 | 1.1x | 2.79 | 3.33 | 0.8x | 6.78 | 8.22 | 0.8x |
| 4,000,000 | 1x | 12% | 31.25 | 17.80 | 1.8x | 18.88 | 13.16 | 1.4x | 211.01 | 154.86 | 1.4x |
| 4,000,000 | 2x | 11% | 22.82 | 15.81 | 1.4x | 14.78 | 13.39 | 1.1x | 99.02 | 81.04 | 1.2x |
| 4,000,000 | 5x | 3% | 13.46 | 12.88 | 1.0x | 9.88 | 12.65 | 0.8x | 22.97 | 26.97 | 0.9x |

Three things have held on every machine measured:

- Most of the GPU's time goes to moving rays, not tracing them. A million 64-bit directions are 24 MB, and sending them took about 0.17 ms per megabyte on an RTX 3090, however we sent them. Tracing is a few percent of the total. So the GPU gains least when most rays miss, because the CPU can rule those out in about the time it takes to send them. Sending 32-bit directions costs half as much.
- "Every crossing" queries need big batches to win. The card works through one ray's crossings in order, so a small batch of long rays is slower on the card than on the CPU, and a large batch is faster.
- The answers match. The bench checks the card's answers against the CPU's before it times anything.

## Benchmarks

Ray timings are in the GPU section above. The rest are here.

### Collisions

The same script times the same assembly as a `CollisionScene`. One call checks every pair of its 72 instances as triangle meshes. (None of the meshes is closed, so none can be treated as convex.) Times are milliseconds, the best of five calls:

| pulled apart | pairs | touching | `collides` ms | within 1 mm ms | distances within 10 mm ms | every distance ms |
|---|---|---|---|---|---|---|
| 1x | 2,556 | 119 | 0.48 | 0.48 | 28.8 | 59 |
| 2x | 2,556 | 67 | 0.46 | 0.21 | 23.4 | 78 |
| 5x | 2,556 | 25 | 0.07 | 0.23 | 2.7 | 95 |

`collides` is fast because it rules out most pairs cheaply: first by each instance's bounding box, then by each mesh's tree of boxes. That is how it finds which of 2,556 pairs touch, among 377k faces, in about half a millisecond. Given a margin, it also reports pairs that come within that distance without touching (the "within 1 mm" column).

`distances` is slower because it measures every pair you ask about. Pass `upper_bound` and it stops early on pairs farther apart than that, which is usually all a motion planner needs.

### Example: a robot arm planner

`examples/ur10.py`: a UR10e robot arm with a box gripper reaches into a cubby for a block. The planner is one function. It places the arm at every waypoint of a path, makes one `CollisionScene.distances` call for all of them, and hands the result to `scipy.optimize.minimize`. The gradient comes from the normals and closest points that `distances` returns.

```
planned in 40 ms: 15 L-BFGS-B evaluations of 14,342 pair distances each, aiming for 27.5 mm over 1 round
then twenty times as many waypoints checked clear
```

### Against what trimesh used before

| `trimkit.` | replaces | measured against it | peak memory (MiB), old / trimkit |
|---|---|---|---|
| `exchange.svg_parse`, `exchange.svg_string` | `svg.path` | 63-64x faster parsing 60k commands | 19 / 13 |
| `contour` | `scikit-image` | 44x faster on a 14M-voxel bool volume, 12x on a 28M-voxel float64 field | 160 / 49, 159 / 74 |
| `Boxes` | `rtree` | 10-11x faster: build 100k boxes, query 20k | 19 / 21 |
| `RayScene` | `embreex` | 6.3-6.4x faster on 1M rays, in float64 and watertight | 105 / 57 |
| `CollisionScene` | `python-fcl` | 4x faster: every touching pair of 200 spheres, through trimesh's `CollisionManager`; convex distances more accurate | under 1 / under 1 |
| `exchange.draco_encode`, `exchange.draco_decode` | `DracoPy` | 3.4x faster encoding, 1.5-1.6x decoding | 9 / 5, 5 / 6 |
| `simplify` | `fast-simplification` | 2.6x faster, and thread safe | 31 / 10 |
| `decompose` | `vhacdx` | 1.2x faster: the same library | 380 / 469: V-HACD's work runs on every core at once, each with its own working set |
| `parametrize` | `xatlas` | 1.1x faster | 6 / 5 |
| `triangulate` | `mapbox-earcut` | 1.1-1.2x faster: the same library | 20 / 21 |

Measured 2026-11-17 with `check/bench/versus.py`, on the same numpy inputs: the best of five calls each, on the 32 threads of an idle i9-14900K, with a release build that skips profile-guided optimization. A range is two runs. Peak memory is the most memory one call adds, its output included, each call in a fresh process.

### Installed size

Installed size on top of `trimesh` + `numpy`, cheapest first, so a shared dependency is charged to the smaller package. `coal` is the library `collision` vendors.

| on top of `trimesh` + `numpy` | also brings | adds (MiB) | total (MiB) |
|---|---|---|---|
| `svg.path` |  | 0.1 | 0.1 |
| `mapbox-earcut` |  | 0.2 | 0.3 |
| `xatlas` |  | 0.6 | 0.8 |
| `fast-simplification` |  | 0.7 | 1.5 |
| `vhacdx` |  | 0.7 | 2.3 |
| `rtree` |  | 1.8 | 4.1 |
| `DracoPy` |  | 7.5 | 11.6 |
| `python-fcl` | `cython` | 41.8 | 53.4 |
| `embreex` |  | 44.3 | 97.7 |
| `scikit-image` | `imageio`, `lazy-loader`, `networkx`, `packaging`, `pillow`, `scipy`, `tifffile` | 204.2 | 301.9 |
| `coal` | `cmeel`, `cmeel-assimp`, `cmeel-boost`, `cmeel-octomap`, `cmeel-qhull`, `cmeel-zlib`, `eigenpy`, `libcoal` | 274.3 | 576.3 |
| `trimkit` 0.2.1, instead of all of the above |  | 2.3 | 2.3 |

`coal` was never a trimesh dependency, and `scipy`, `pillow` and `networkx` stay. So the real saving is per install option, old bindings (Python < 3.15) against trimkit (3.15+):

| `trimesh[...]` | old bindings (MiB) | with trimkit (MiB) | saved (MiB) |
|---|---|---|---|
| `trimesh[easy]` | 246.9 | 202.1 | 44.9 |
| `trimesh[easy,recommend]` | 504.3 | 380.6 | 123.6 |

Measured by `check/bench/sizes.py`, 2026-11-17, CPython 3.13.13 on `linux-x86_64`, with a fresh `uv venv` per row and bytecode compiled. The `trimesh` 5.2.1 + `numpy` 2.5.5 baseline is 65.5 MiB.

## Wheels

A wheel is a prebuilt package. An sdist is the source package, which you build yourself. Each wheel here is tagged `cp312-abi3`: one build works on CPython 3.12 and every later version, so a new Python needs no new release.

There are 9 wheels, one for every platform numpy publishes wheels for:

- Linux, for both of its C libraries (glibc as `manylinux_2_28`, musl as `musllinux_1_2`), on x86_64 and aarch64
- macOS 11 and later, on x86_64 and arm64
- Windows, on x64, x86 and arm64

On any other machine, build from the sdist. It needs clang 19 or newer and the lld linker, and it refuses no machine. To use your system's copies of some libraries, pass `-Csystem=eigen,fast_float,robin_map`.

Free-threaded Python (the build of Python without the GIL) has no wheel yet. Every release builds and tests it from the sdist with `uv run scripts sdist`. It will get wheels when nanobind, the binding library, can produce one wheel that covers every free-threaded version.

For Linux distributions, `uv run scripts distro` generates the `debian/` directory and the Fedora spec file from `trimkit.toml`, into `build/distro/`.

## Development

```
uv run scripts check       # the gate: lint, build, tests, static checks, sanitizers, fuzzers, quick bench, thread-pool checkers
uv run scripts lint        # the fast deterministic checks, nothing compiled
uv run scripts fuzz        # every fuzzer, native then Atheris (--seconds each)
uv run scripts bench       # the thread pool against hand-written pools, 144 cells (--quick: the short ones, seconds)
uv run scripts genmc       # the thread pool's real code in every thread ordering (GenMC), and bugs planted in it
uv run scripts model       # the thread pool's design as a model (TLA+), and real runs replayed against it
uv run scripts update      # every upstream and dependency to its newest
uv run scripts format      # our C++, as scripts/clang-format.yaml says
uv run scripts generate    # regenerate the shaders' SPIR-V, the stubs, this README's generated block and check/tests/golden.json
uv run scripts wheel       # this platform's release wheels
uv run scripts distro      # Debian and Fedora packaging (--run: build, test and lint them)
```

Run `check` before pushing; it prints how many minutes each stage took. Everything except `lock` and `wheel` runs inside a pinned Debian container (`scripts/docker/Dockerfile`), so your machine needs only Docker and uv. The sources are mounted read-only unless the command writes them.

CI (the automatic checks on GitHub) runs the same commands:

- Every push to main runs `.github/workflows/release.yml`. It lints, builds and tests the sdist and every wheel, then publishes the version `pyproject.toml` names. A version PyPI already has is left alone.
- A push to `release/*` does all of that except the upload.
- Once a week, if main has moved, `.github/workflows/fuzz.yml` fuzzes for ten minutes per fuzzer. A fuzzer feeds a function random inputs to find crashes. Each run starts from the seeds in the repository, keeps nothing, and fails on a finding.

`trimkit.toml` is the one build file edited by hand. It lists every upstream library (its pinned version, its patches, how it compiles, and the reason for anything unusual), every part of the extension, the wheel-size budget, and the compilers a release accepts. There is no CMake. A wheel is one shared library made from a fixed list of sources, so `scripts/builder.py` writes that list as one file for ninja, a small build tool. To see exactly what a build does, read its `build.ninja`.

Every build uses LLVM, the family of tools that includes the clang compiler and the lld linker (on macOS, Xcode's linker). Two optimizations are on: ThinLTO, which optimizes across source files, and profile-guided optimization (PGO), which runs the code first to learn which paths are hot. Both write files only the same LLVM version can read, so the builder stops unless the compiler, linker and profile tool all match. `-Cstrict=true`, used by the gate and releases, also stops on a clang newer than the ones `[toolchain]` lists as tested.

## Vendoring

Vendoring means copying a library's source into this repository. `vendor/` holds what a build needs and nothing else, committed. Nothing is downloaded to build, so this keeps building if GitHub goes away. It is about 10 MB in a checkout, half of it Eigen's headers, and 2.3 MB in the sdist.

One rule decides which files are vendored: the files `keep` names in `trimkit.toml`, plus every file the compiler reads when it builds our code for any wheel target. Files stay at the upstream's own paths. `check` repeats the scan, so the set can't drift.

Our changes to vendored code are made in place, each wrapped between `//// TRIMKIT PATCH: fix.name` (or `perf.name`) and `//// TRIMKIT END PATCH`. Two commands keep that honest:

- `uv run scripts patch` saves each change as a diff in `vendor/patches/`, with its reason on top and its sha256 fingerprint recorded. It then fetches the upstream again and applies the patches, to prove they rebuild `vendor/` byte for byte.
- `uv run scripts fetch` rebuilds all of `vendor/` from the pinned commits and those patches. Afterwards `git status vendor/` shows nothing, so anyone can confirm `vendor/` is what it claims to be.

A change outside a marked block is refused, naming its line, and `check` fails on a block edited since its patch was written.

## Supply chain

Every input that reaches a wheel, and what pins it:

- Compiled code is vendored. Each upstream is pinned by commit and by one `tree_sha256` fingerprint of everything vendored for it, patches applied. `check` verifies it.
- Python tools and dependencies are locked by hash in `uv.lock`, and nothing newer than one week is accepted (`exclude-newer`). Builds install them with `--require-hashes`.
- The container image pins its base by digest (a fingerprint of its exact contents), Debian packages by snapshot date, uv by digest, and a CPython built with ThreadSanitizer (a tool that finds two threads touching the same memory unsafely) by sha256. Every check runs from the image, so nothing on your machine can steer it.
- CI pins GitHub actions by commit. cibuildwheel, the tool that builds the wheels, pins its Linux images by digest and its macOS and Windows Pythons by sha256.

`uv run scripts update` moves every pin, with the same one-week wait.

A few inputs can only be pinned by version or signature, because nothing finer exists: clang and lld from the Linux images' signed package repositories; the macOS and Windows runners' compilers and SDKs (each held to `[toolchain]`); cibuildwheel's own pip, build, auditwheel and delocate; and ninja from PyPI, for someone building the sdist in an isolated environment.

## Licensing

A wheel's `License-Expression` names every license compiled into it, and `check` derives it from `trimkit.toml`. All are permissive (MIT, ISC, BSD, Apache-2.0, BSL-1.0) except Eigen's headers, which are MPL-2.0. That license asks only that changes to those files be shared. They are vendored unchanged (`check` refuses a patch to them), built with `EIGEN_MPL2_ONLY`, and shipped as source in the sdist. Every license text ships in the wheel, along with a machine-readable list of everything inside it (a software bill of materials, in the PEP 770 format), generated from `trimkit.toml`.
