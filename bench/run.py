"""
Benchmark trimkit against the packages it replaces, inside trimesh.

  python bench/run.py setup TRIMESH DIR    clone the trimesh checkout TRIMESH to DIR,
                                          codemod it, make DIR/.venv
  DIR/.venv/bin/python bench/run.py stress    big inputs through trimesh's API
  python bench/run.py report LOG   one row per call site from the JSON lines

Each codemodded site calls `trimesh._compare.compare`, see `bench/compare.py`.
Every call is appended to DIR/bench.jsonl, the LOG for `report`, so running
the clone's own tests records each one.
"""

import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGACY = ["mapbox_earcut==2.1.0", "vhacdx==0.1.1", "xatlas==0.0.11", "DracoPy==2.2.0", "svg.path"]
# not replaced, but the baseline the bvh module is measured against
BASELINES = ["embreex"]

# (file, first line of the block, last line of the block, replacement)
SITES = [
    (
        "trimesh/ray/ray_pyembree.py",
        "        self.scene = rtcore_scene.EmbreeScene()\n",
        "        self.scene = rtcore_scene.EmbreeScene()\n",
        "        from trimesh._compare import bvh_build\n\n"
        "        self._built = bvh_build(vertices, faces)\n"
        "        self.scene = rtcore_scene.EmbreeScene()\n",
    ),
    (
        "trimesh/ray/ray_pyembree.py",
        "    def run(self, origins, normals, **kwargs):\n",
        "            scaled.astype(_embree_dtype), normals.astype(_embree_dtype), **kwargs\n        )\n",
        "    def run(self, origins, normals, **kwargs):\n"
        "        from trimesh._compare import bvh_new, bvh_old, compare, same_first_hit\n\n"
        "        assert not kwargs\n"
        "        return compare(\n"
        '            "bvh",\n'
        "            lambda: bvh_old(self.scene, self.origin, self.scale, origins, normals),\n"
        "            lambda: bvh_new(self._built, origins, normals),\n"
        "            lambda a, b: same_first_hit(self._built, origins, normals, a, b),\n"
        "        )\n",
    ),
    (
        "trimesh/creation.py",
        "        from mapbox_earcut import triangulate_float64\n",
        "            .reshape((-1, 3))\n        )\n",
        "        from trimesh._compare import compare, earcut_new, earcut_old, same_tiling\n\n"
        "        vertices, faces = compare(\n"
        '            "earcut", lambda: earcut_old(polygon), lambda: earcut_new(polygon), same_tiling\n'
        "        )\n",
    ),
    (
        "trimesh/decomposition.py",
        "    from vhacdx import compute_vhacd\n",
        "        for v, f in compute_vhacd(mesh.vertices, faces, **kwargs)\n    ]\n",
        "    from trimesh._compare import compare, same_vhacd, vhacd_new, vhacd_old\n\n"
        "    return compare(\n"
        '        "vhacd",\n'
        "        lambda: vhacd_old(mesh, **kwargs),\n"
        "        lambda: vhacd_new(mesh, **kwargs),\n"
        "        lambda a, b: same_vhacd(mesh, a, b),\n"
        "    )\n",
    ),
    (
        "trimesh/base.py",
        "        import xatlas\n",
        "        vmap, faces, uv = xatlas.parametrize(self.vertices, self.faces)\n",
        "        from trimesh._compare import compare, same_xatlas, xatlas_new, xatlas_old\n\n"
        "        vmap, faces, uv = compare(\n"
        '            "xatlas",\n'
        "            lambda: xatlas_old(self.vertices, self.faces),\n"
        "            lambda: xatlas_new(self.vertices, self.faces),\n"
        "            lambda a, b: same_xatlas(self.faces, a, b),\n"
        "        )\n",
    ),
    (
        "trimesh/exchange/gltf/extensions.py",
        '    import DracoPy\n\n    data = context["data"]\n',
        "        accessors[indices] = faces\n",
        "    from trimesh._compare import compare, draco_decode_new, draco_decode_old, same_decoded\n\n"
        "    decoded = compare(\n"
        '        "draco.decode",\n'
        "        lambda: draco_decode_old(context),\n"
        "        lambda: draco_decode_new(context),\n"
        "        same_decoded,\n"
        "    )\n"
        "    for index, value in decoded.items():\n"
        '        context["accessors"][index] = value\n',
    ),
    (
        "trimesh/exchange/gltf/extensions.py",
        "    import DracoPy\n\n    from . import _buffer_append\n",
        '        "attributes": absorbed,\n    }\n    return True\n',
        "    from trimesh._compare import compare, draco_encode_new, draco_encode_old, same_encoded\n\n"
        "    from . import _buffer_append\n\n"
        "    encoded = compare(\n"
        '        "draco.encode",\n'
        "        lambda: draco_encode_old(context),\n"
        "        lambda: draco_encode_new(context),\n"
        "        same_encoded,\n"
        "    )\n"
        "    if encoded is None:\n"
        "        return None\n"
        '    context["primitive"].setdefault("extensions", {})["KHR_draco_mesh_compression"] = {\n'
        '        "bufferView": _buffer_append(context["buffer_items"], encoded[0]),\n'
        '        "attributes": encoded[1],\n'
        "    }\n"
        "    return True\n",
    ),
    (
        "trimesh/path/exchange/svg_io.py",
        "        raw = np.array(list(parse_path(path_string)))\n",
        "                counts[name] += len(v)\n",
        "        from trimesh._compare import compare, same_svg, svg_path_new, svg_path_old\n\n"
        "        start = counts[name]\n"
        "        found = compare(\n"
        '            "svg",\n'
        "            lambda: svg_path_old(path_string, start),\n"
        "            lambda: svg_path_new(path_string, start),\n"
        "            same_svg,\n"
        "        )\n"
        "        entity_meta = _attrib_metadata(attrib=attrib)\n"
        "        for e in found[0]:\n"
        "            e.metadata.update(entity_meta)\n"
        "        entities[name].extend(found[0])\n"
        "        vertices[name].append(transform_points(found[1], matrix))\n"
        "        counts[name] += len(found[1])\n",
    ),
]


def setup(source: Path, target: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(source), str(target)], check=True)
    shutil.copy(HERE / "compare.py", target / "trimesh" / "_compare.py")
    for name, first, last, replacement in SITES:
        path = target / name
        text = path.read_text()
        # each block must be exactly where we expect, or trimesh moved on
        assert text.count(first) == 1, f"{name}: start of block not found once"
        start = text.index(first)
        end = text.index(last, start) + len(last)
        path.write_text(text[:start] + replacement + text[end:])

    venv = target / ".venv"
    subprocess.run(["uv", "venv", "-q", "--python", "3.12", str(venv)], check=True)
    install = ["uv", "pip", "install", "-q", "--python", str(venv / "bin" / "python")]
    subprocess.run([*install, "-e", f"{target}[easy,test]", *LEGACY, *BASELINES], check=True)
    subprocess.run([*install, "-C", "strict=true", str(HERE.parent)], check=True)


def stress() -> None:
    # big inputs through trimesh's public API, so they hit the codemodded sites
    import io

    import numpy as np
    import shapely
    import trimesh

    rng = np.random.default_rng(0)

    def star(n, radius, center=(0, 0)):
        angle = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = radius * rng.uniform(0.7, 1.0, n)
        return np.column_stack([r * np.cos(angle), r * np.sin(angle)]) + center

    # holes on a grid far enough apart and inside the shell to keep it valid
    grid = np.arange(-40, 41, 10.0)
    holes = [star(200, 1.0, (x, y)) for x in grid for y in grid if np.hypot(x, y) < 55]
    big = shapely.Polygon(star(100_000, 100.0), holes)
    assert big.is_valid
    trimesh.creation.triangulate_polygon(big)
    # what trimesh makes from arcs: a dense regular circle, and an annulus
    circle = shapely.Point(0, 0).buffer(1.0, quad_segs=10_000)
    trimesh.creation.triangulate_polygon(circle)
    trimesh.creation.triangulate_polygon(circle.difference(shapely.Point(0, 0).buffer(0.5, quad_segs=5_000)))
    for _ in range(2000):
        trimesh.creation.extrude_polygon(shapely.Polygon(star(30, 1.0)), height=1.0)

    # the models directory of the trimesh checkout `setup` cloned
    root = Path(trimesh.__file__).resolve().parents[1] / "models"
    trimesh.load_mesh(root / "featuretype.STL").convex_decomposition()
    trimesh.creation.icosphere(subdivisions=5).unwrap()

    sphere = trimesh.creation.icosphere(subdivisions=8)
    sphere.visual.vertex_colors = rng.integers(0, 255, (len(sphere.vertices), 4))
    blob = sphere.export(file_type="glb", extension_draco=True)
    trimesh.load_scene(trimesh.util.wrap_as_stream(blob), file_type="glb")

    # big batches of first hits, and containment, which casts many rays
    ball = trimesh.creation.icosphere(subdivisions=7)
    origins = rng.normal(size=(500_000, 3)) * 3
    ball.ray.intersects_first(origins, -origins)
    ball.contains(rng.uniform(-1.2, 1.2, (200_000, 3)))

    d = "M0 0 " + "c1 2 3 4 5 6 l1 1 1 0 0 1 a 3 2 30 1 0 4 4 q 1 1 2 2 " * 20_000
    svg = f'<svg xmlns="http://www.w3.org/2000/svg"><path d="{d}"/></svg>'
    trimesh.load_path(io.BytesIO(svg.encode()), file_type="svg")


def report(log: Path) -> None:
    rows = defaultdict(list)
    for line in log.read_text().splitlines():
        record = json.loads(line)
        rows[record.pop("name")].append(record)
    print("| site | calls | old s | new s | speedup | median old ms | median new ms | notes |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    import numpy as np

    for name, records in sorted(rows.items()):
        old = np.array([r["old"] for r in records])
        new = np.array([r["new"] for r in records])
        notes = ""
        if "old_bytes" in records[0]:
            o, n = (sum(r[k] for r in records) for k in ("old_bytes", "new_bytes"))
            notes = f"bytes {o:,} -> {n:,}"
        if "old_covered" in records[0]:
            notes = "hulls, surface covered " + ", ".join(
                f"{r['old_hulls']}->{r['new_hulls']} {r['old_covered']:.3f}->{r['new_covered']:.3f}"
                for r in records
            )
        if "differ" in records[0]:
            o, n = (sum(r[k] for r in records) for k in ("differ", "rays"))
            k = sum(r["behind"] for r in records)
            notes = (
                f"first hit differs on {o:,} of {n:,} rays: {k:,} embree hits behind"
                " the origin, the rest ties or edge grazes"
            )
        if "old_used" in records[0]:
            o, n = (min(r[k] for r in records) for k in ("old_used", "new_used"))
            notes = f"least atlas area used {o:.3f} -> {n:.3f}"
        print(
            f"| {name} | {len(records)} | {old.sum():.3f} | {new.sum():.3f} | "
            f"{old.sum() / new.sum():.2f}x | {np.median(old) * 1e3:.3f} | "
            f"{np.median(new) * 1e3:.3f} | {notes} |"
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "setup":
        setup(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
    elif command == "stress":
        stress()
    elif command == "report":
        report(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
