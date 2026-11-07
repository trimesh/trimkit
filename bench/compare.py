"""
Injected into a temporary trimesh as `trimesh/_compare.py` by `bench/run.py`.

Every codemodded call site runs `compare(name, old, new, same)`: the code
trimesh has today against the same thing done with `trimkit`,
asserting they agree. The old result is returned, so trimesh's behaviour is
unchanged and its own tests still judge it. The `*_new` functions here are
the code trimesh would ship.
"""

import json
import time
from pathlib import Path

import numpy as np

# every call is appended here, next to the clone's `trimesh` package
_LOG = Path(__file__).resolve().parent.parent / "bench.jsonl"
# the machine may be busy: time each side this many times and keep the fastest
_REPEAT = 3


def compare(name, old, new, same):
    # alternate which runs first so neither always gets the warm cache
    best = [np.inf, np.inf]
    for i in range(_REPEAT):
        for side in (0, 1) if i % 2 == 0 else (1, 0):
            tic = time.perf_counter()
            result = (old, new)[side]()
            best[side] = min(best[side], time.perf_counter() - tic)
            if side == 0:
                a = result
            else:
                b = result
    extra = same(a, b) or {}
    print(name, np.array(best))
    with _LOG.open("a") as f:
        f.write(json.dumps({"name": name, "old": best[0], "new": best[1], **extra}) + "\n")
    return a


def same_tiling(a, b):
    # our vendored earcut.hpp is a newer version than mapbox_earcut binds, so
    # their triangles can differ: require instead that both tile the polygon
    # exactly with the same vertices
    (vertices, old), (other, new) = a, b
    assert np.array_equal(vertices, other)
    assert np.array_equal(np.unique(old), np.unique(new))
    areas = []
    for faces in (old, new):
        p, q, r = (vertices[faces[:, i]] for i in range(3))
        area = (q - p)[:, 0] * (r - p)[:, 1] - (q - p)[:, 1] * (r - p)[:, 0]
        # every triangle is wound the same way, so none overlap or flip; a
        # sliver over collinear vertices may come out a rounding error either way
        noise = 1e-9 * np.abs(area).max(initial=0.0)
        assert np.all(area >= -noise) or np.all(area <= noise)
        areas.append(np.abs(area).sum())
    assert np.isclose(*areas, rtol=1e-9, atol=1e-12), areas


# -- earcut: `trimesh.creation.triangulate_polygon` --------------------------


def earcut_old(polygon):
    from mapbox_earcut import triangulate_float64

    vertices = [np.array(polygon.exterior.coords)]
    vertices.extend(np.array(i.coords) for i in polygon.interiors)
    rings = np.cumsum([len(v) for v in vertices])
    vertices = np.vstack(vertices)
    faces = triangulate_float64(vertices, rings).astype(np.int64).reshape((-1, 3))
    return vertices, faces


def earcut_new(polygon):
    import shapely

    import trimkit

    # shapely's own rings, as arrays: no offsets, and the vertices come back
    exterior = shapely.get_coordinates(polygon.exterior)
    interiors = [shapely.get_coordinates(ring) for ring in polygon.interiors]
    return trimkit.triangulate(exterior, interiors)


# -- vhacd: `trimesh.decomposition.convex_decomposition` --------------------

# trimesh's vhacdx keywords that trimkit.decompose still has; the rest stay V-HACD's defaults
_VHACD_NAMES = {
    "maxConvexHulls": "maximum_hulls",
    "maxNumVerticesPerCH": "maximum_vertices",
    "resolution": "resolution",
    "minimumVolumePercentErrorAllowed": "maximum_error",
}


def vhacd_old(mesh, **kwargs):
    from vhacdx import compute_vhacd

    faces = (
        np.column_stack((np.ones(len(mesh.faces), dtype=np.int64) * 3, mesh.faces)).ravel().astype(np.uint32)
    )
    return [{"vertices": v, "faces": f} for v, f in compute_vhacd(mesh.vertices, faces, **kwargs)]


def vhacd_new(mesh, **kwargs):
    import trimkit

    kwargs = {_VHACD_NAMES[k]: v for k, v in kwargs.items() if k in _VHACD_NAMES}
    return [{"vertices": v, "faces": f} for v, f in trimkit.decompose(mesh.vertices, mesh.faces, **kwargs)]


def _covered(hulls, points, tol):
    # the fraction of points inside (within `tol` of) at least one hull
    from trimesh import Trimesh

    inside = np.zeros(len(points), dtype=bool)
    for hull in hulls:
        h = Trimesh(**hull, process=False)
        offset = np.einsum("fj,fj->f", h.face_normals, h.triangles[:, 0])
        inside |= (h.face_normals @ points.T - offset[:, None]).max(axis=0) <= tol
    return inside.mean()


def same_vhacd(mesh, a, b):
    # the vendored V-HACD header is newer than vhacdx's: upstream acbb79c9
    # stopped shrink-wrapping hulls inside the surface. So instead of equal
    # output, require the new hulls cover at least as much of the surface
    points = mesh.sample(5000, seed=0)
    covered = [_covered(r, points, mesh.scale * 1e-3) for r in (a, b)]
    assert covered[1] >= covered[0] - 1e-3, covered
    return {"old_hulls": len(a), "new_hulls": len(b), "old_covered": covered[0], "new_covered": covered[1]}


# -- xatlas: `trimesh.Trimesh.unwrap` ----------------------------------------


def xatlas_old(vertices, faces):
    import xatlas

    return xatlas.parametrize(vertices, faces)


def xatlas_new(vertices, faces):
    import trimkit

    return trimkit.parametrize(vertices, faces)


def same_xatlas(faces, a, b):
    # centering on the bounding box and capping chart size (see src/xatlas.cpp)
    # change xatlas's charts, so require each result be a valid atlas of the
    # same mesh and that the new one packs about as well
    used = []
    for vmap, new_faces, uv in (a, b):
        # every face corner maps back to exactly the original vertex
        assert np.array_equal(vmap[new_faces], faces)
        assert np.isfinite(uv).all() and uv.min() >= 0.0 and uv.max() <= 1.0
        p, q, r = (uv[new_faces[:, i]] for i in range(3))
        area = np.abs((q - p)[:, 0] * (r - p)[:, 1] - (q - p)[:, 1] * (r - p)[:, 0]).sum() / 2.0
        # charts that don't overlap can't cover more than the unit square
        assert area <= 1.0 + 1e-6, area
        used.append(float(area))
    assert used[1] >= 0.9 * used[0], used
    return {"old_used": used[0], "new_used": used[1]}


# -- bvh: `trimesh.ray.ray_pyembree._EmbreeWrap` -----------------------------
#
# not a replacement: embree stays, and this measures tinybvh alongside it on
# every query trimesh makes, built once per scene like embree's


def bvh_build(vertices, faces):
    import trimkit

    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    return trimkit.RayScene([(vertices, faces)]), vertices, faces


def bvh_old(scene, origin, scale, origins, directions):
    scaled = (np.array(origins, dtype=np.float64) - origin) * scale
    return scene.run(scaled.astype(np.float32), directions.astype(np.float32))


def bvh_new(built, origins, directions):
    origins = np.ascontiguousarray(origins, np.float64)
    directions = np.ascontiguousarray(directions, np.float64)
    return built[0].intersect_first(origins, directions)[1]


def _hit(vertices, faces, faces_hit, origins, directions):
    # distance to and smallest barycentric coordinate of each ray's triangle
    a, b, c = (vertices[faces[faces_hit, i]] for i in range(3))
    e1, e2 = b - a, c - a
    p = np.cross(directions, e2)
    det = np.einsum("ij,ij->i", p, e1)
    s = origins - a
    u = np.einsum("ij,ij->i", s, p) / det
    q = np.cross(s, e1)
    v = np.einsum("ij,ij->i", directions, q) / det
    t = np.einsum("ij,ij->i", q, e2) / det
    return t, np.minimum(np.minimum(u, v), 1 - u - v)


def same_first_hit(built, origins, directions, a, b):
    # where float32 embree and float64 tinybvh disagree, every ray is one of:
    # - a tie: both triangles really are hit at the same distance
    # - an edge graze closer than float32 resolves, by either one's hit
    # - embree re-hitting a triangle behind the origin: trimesh steps each ray
    #   just past its last hit, and float32 can't resolve that step, while
    #   tinybvh's hit, if any, is really ahead
    _, vertices, faces = built
    differ = np.nonzero(a != b)[0]
    o = np.asarray(origins, np.float64)[differ]
    d = np.asarray(directions, np.float64)[differ]
    t, margin = np.full((2, len(differ)), np.inf), np.full((2, len(differ)), np.inf)
    for side, faces_hit in enumerate((a[differ], b[differ])):
        has = faces_hit >= 0
        if has.any():
            t[side, has], margin[side, has] = _hit(vertices, faces, faces_hit[has], o[has], d[has])
    tie = np.isclose(t[0], t[1], rtol=1e-6, atol=0) & np.isfinite(t).all(axis=0)
    graze = (margin < 1e-4).any(axis=0)
    behind = (t[0] <= 0) & (t[0] > -1e-5 * np.ptp(vertices, axis=0).max()) & (t[1] > 0)
    ok = tie | graze | behind
    assert ok.all(), differ[~ok][:5]
    return {"rays": len(a), "differ": len(differ), "behind": int(behind.sum())}


# -- draco: `trimesh.exchange.gltf.extensions` --------------------------------


def draco_decode_old(context):
    import DracoPy

    data = context["data"]
    attributes = context["primitive"].get("attributes", {})
    decoded = DracoPy.decode(context["views"][data["bufferView"]])
    names = {ident: name for name, ident in data["attributes"].items()}
    out = {}
    for attr in decoded.attributes:
        name = names.get(attr["unique_id"])
        if name in attributes:
            out[attributes[name]] = attr["data"]
    indices = context["primitive"].get("indices")
    faces = getattr(decoded, "faces", None)
    if indices is not None and faces is not None:
        out[indices] = faces
    return out


def draco_decode_new(context):
    import trimkit

    data = context["data"]
    attributes = context["primitive"].get("attributes", {})
    # the extension's JSON maps glTF names to draco attributes: ask for those
    names = [name for name in data["attributes"] if name in attributes]
    wanted = np.array([data["attributes"][name] for name in names], dtype=np.int64)
    faces, values = trimkit.exchange.draco_decode(context["views"][data["bufferView"]], wanted)
    out = {attributes[name]: value for name, value in zip(names, values, strict=True)}
    if "indices" in context["primitive"]:
        out[context["primitive"]["indices"]] = faces
    return out


def draco_kind(name):
    """
    The DracoKind for a glTF attribute name.
    """
    import trimkit

    if name in ("POSITION", "NORMAL"):
        return trimkit.exchange.DracoKind[name]
    for prefix, kind in (
        ("TEXCOORD_", trimkit.exchange.DracoKind.TEXCOORD),
        ("COLOR_", trimkit.exchange.DracoKind.COLOR),
    ):
        if name.startswith(prefix):
            return kind
    return trimkit.exchange.DracoKind.GENERIC


def same_decoded(a, b):
    assert a.keys() == b.keys()
    for k in a:
        # DracoPy widens some integer types, so compare values not dtypes
        assert np.array_equal(np.asarray(a[k]).reshape(np.shape(b[k])), b[k]), k


_ABSORB = ("NORMAL", "TEXCOORD_0", "COLOR_0")


def draco_encode_old(context):
    import DracoPy
    from trimesh.exchange.gltf.extensions import (
        _DRACO_COMPRESSION,
        _DRACO_QUANTIZATION,
        _draco_optional,
    )

    primitive = context["primitive"]
    attributes = primitive["attributes"]
    arrays = context["arrays"]
    absorb, optional = [], {}
    for name, keyword, kind in _draco_optional:
        value = arrays.get(attributes.get(name))
        if value is None:
            continue
        absorb.append((name, kind))
        optional[keyword] = value if value.dtype == np.uint8 else value.astype(np.float64)
    position = arrays[attributes["POSITION"]]
    indices = arrays[primitive["indices"]]
    buffer = DracoPy.encode(
        points=position,
        faces=indices,
        preserve_order=True,
        quantization_bits=_DRACO_QUANTIZATION,
        compression_level=_DRACO_COMPRESSION,
        **optional,
    )
    check = DracoPy.decode(buffer)
    if len(check.points) != len(position) or len(check.faces) != len(indices):
        return None
    unique_ids = {attr["attribute_type"]: attr["unique_id"] for attr in check.attributes}
    absorbed = {
        name: unique_ids[getattr(DracoPy.AttributeType, kind)]
        for name, kind in [("POSITION", "POSITION"), *absorb]
    }
    return buffer, absorbed


def draco_encode_new(context):
    import trimkit

    primitive = context["primitive"]
    attributes = primitive["attributes"]
    arrays = context["arrays"]
    names = ["POSITION", *(n for n in _ABSORB if attributes.get(n) in arrays)]
    # glTF stores indices as a flat scalar accessor, often uint32: trimkit
    # takes exactly int64, so the caller converts, visibly
    indices = np.asarray(arrays[primitive["indices"]], dtype=np.int64).reshape((-1, 3))
    values = [np.ascontiguousarray(arrays[attributes[n]]) for n in names]
    buffer = trimkit.exchange.draco_encode(indices, values, np.array([draco_kind(n) for n in names]))
    # keep trimesh's guard: a round trip must preserve every count
    faces, (position,) = trimkit.exchange.draco_decode(buffer, np.array([0]))
    if len(position) != len(values[0]) or len(faces) != len(indices):
        return None
    # draco attribute i is attributes[i], which is the extension's JSON
    return buffer, {name: i for i, name in enumerate(names)}


def same_encoded(a, b):
    import trimkit

    assert (a is None) == (b is None)
    if a is None:
        return None
    old_faces, (old_position,) = trimkit.exchange.draco_decode(a[0], np.array([a[1]["POSITION"]]))
    new_faces, (position,) = trimkit.exchange.draco_decode(b[0], np.array([b[1]["POSITION"]]))
    assert np.array_equal(old_faces, new_faces)
    step = np.ptp(position, axis=0).max() / (2**14 - 1)
    assert np.abs(old_position - position).max() <= step
    assert set(a[1]) == set(b[1])
    return {"old_bytes": len(a[0]), "new_bytes": len(b[0])}


# -- svg: `trimesh.path.exchange.svg_io._svg_path_convert` -------------------


def svg_path_old(path_string, start):
    from svg.path import parse_path
    from trimesh import grouping, util
    from trimesh.path.entities import Arc, Bezier, Line

    count = start

    def complex_to_float(values):
        return np.array([[i.real, i.imag] for i in values], dtype=np.float64)

    def multiline(lines):
        points = [L.start for L in lines]
        points.append(lines[-1].end)
        return np.array([[i.real, i.imag] for i in points], dtype=np.float64)

    raw = np.array(list(parse_path(path_string)))
    if len(raw) == 0:
        return [], np.zeros((0, 2))
    kinds_lookup = {"Line": 1, "Close": 1, "Arc": 2}
    kinds = np.array([kinds_lookup.get(type(i).__name__, 0) for i in raw], dtype=int)
    blocks = grouping.blocks(kinds, min_len=1, only_nonzero=False)
    parsed = []
    for b in blocks:
        chunk = raw[b]
        current = type(raw[b[0]]).__name__
        if current in ("Line", "Close"):
            parsed.append(("MultiLine", multiline(chunk)))
        elif len(b) > 1 and current == "Arc":
            verts = np.array(
                [
                    [a.start.real, a.start.imag, a.end.real, a.end.imag, a.center.real,
                     a.center.imag, a.radius.real, a.radius.imag, a.rotation]
                    for a in chunk
                ],
                dtype=np.float64,
            )  # fmt: skip
            closed = False
            if np.ptp(verts[:, 4:], axis=0).mean() < 1e-3:
                closed = util.allclose(verts[:, :2], np.roll(verts[:, 2:4], 1, axis=0))
            if closed:
                chunk[0].closed = True
                parsed.append(chunk[0])
            else:
                parsed.extend(chunk)
        else:
            parsed.extend(chunk)

    entities, vertices = [], []
    for item in parsed:
        kind = item[0] if isinstance(item, tuple) else type(item).__name__
        if kind == "MultiLine":
            points = item[1]
            entity = Line(points=np.arange(len(points)) + count)
        elif kind == "Arc":
            points = complex_to_float([item.start, item.point(0.5), item.end])
            entity = Arc(points=np.arange(3) + count, closed=getattr(item, "closed", False))
        elif kind == "QuadraticBezier":
            points = complex_to_float([item.start, item.control, item.end])
            entity = Bezier(points=np.arange(3) + count)
        elif kind == "CubicBezier":
            points = complex_to_float([item.start, item.control1, item.control2, item.end])
            entity = Bezier(np.arange(4) + count)
        else:
            continue
        entities.append(entity)
        vertices.append(points)
        count += len(points)
    return entities, np.vstack(vertices) if vertices else np.zeros((0, 2))


def svg_path_new(path_string, start):
    from trimesh.path.entities import Arc, Bezier, Line

    import trimkit

    vertices, segments = trimkit.exchange.svg_parse(path_string)
    n = len(segments)
    if n == 0:
        return [], np.zeros((0, 2))
    kind = segments["kind"]
    index = segments["points"]
    line = kind == trimkit.exchange.SvgKind.LINE
    arc = kind == trimkit.exchange.SvgKind.ARC

    # a run is consecutive segments of one kind that each start on the last
    # one's end vertex: only a moveto breaks a run, as in trimesh's grouping,
    # so lines drawn on after a close continue the polyline through it
    first = np.ones(n, dtype=bool)
    first[1:] = (kind[1:] != kind[:-1]) | (index[1:, 0] != index[:-1, 3])
    last = np.ones(n, dtype=bool)
    last[:-1] = first[1:]

    # each segment's points as trimesh's entities take them: start, a control
    # point or an arc's halfway point, a cubic's second control point, end
    slots = index.copy()
    slots[arc, 1] = index[arc, 2]

    closed = np.zeros(n, dtype=bool)
    if arc.any():
        # a run of arcs sharing center, radius and rotation whose last end is
        # its first start is one closed circle, the test trimesh makes on
        # svg.path (within a run, each arc starts on the last one's end vertex)
        runs = np.flatnonzero(first)
        run_ends = np.append(runs[1:], n) - 1
        params = np.column_stack([vertices[index[:, 1]], segments["radius"], segments["rotation"]])
        spread = np.maximum.reduceat(params, runs) - np.minimum.reduceat(params, runs)
        gap = np.abs(vertices[index[runs, 0]] - vertices[index[run_ends, 3]]).max(axis=1)
        circle = (spread.mean(axis=1) < 1e-3) & (gap < 1e-8)
        closed = ((kind[runs] == trimkit.exchange.SvgKind.ARC) & (run_ends > runs) & circle)[
            np.cumsum(first) - 1
        ]

    # a polyline takes each segment's start plus its last end, a closed
    # circle only its first arc, anything else all of its own points
    keep = ~closed | first
    mask = np.column_stack(
        [keep, keep & ~line, kind == trimkit.exchange.SvgKind.CUBIC, keep & (last | ~line)]
    )
    vertices = vertices[slots[mask]]

    # the one remaining loop builds trimesh's entity objects
    heads = np.flatnonzero(first | ~(line | closed))
    bounds = np.zeros(len(heads) + 1, dtype=np.int64)
    np.cumsum(np.add.reduceat(mask.sum(axis=1), heads), out=bounds[1:])
    index = np.arange(start, start + len(vertices))
    entities = [
        Arc(points=index[a:b], closed=c)
        if k == trimkit.exchange.SvgKind.ARC
        else (Line if k == trimkit.exchange.SvgKind.LINE else Bezier)(points=index[a:b])
        for k, c, a, b in zip(
            kind[heads].tolist(),
            closed[heads].tolist(),
            bounds[:-1].tolist(),
            bounds[1:].tolist(),
            strict=True,
        )
    ]
    return entities, vertices


def same_svg(a, b):
    assert [type(e).__name__ for e in a[0]] == [type(e).__name__ for e in b[0]]
    for x, y in zip(a[0], b[0], strict=True):
        assert np.array_equal(x.points, y.points)
        assert getattr(x, "closed", False) == getattr(y, "closed", False)
    assert a[1].shape == b[1].shape
    assert np.allclose(a[1], b[1], atol=1e-7)
