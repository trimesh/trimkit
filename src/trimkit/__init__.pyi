"""
Thin, stable-ABI numpy bindings to pinned C and C++ geometry libraries.

They depend only on `numpy`, take exact-dtype arrays (float64 ones finite
and within +-1e30), read C-contiguous ones in place and copy any other layout
once, return fresh arrays that own their memory, and release the GIL during
compute.

Attributes
----------
__version__ : str
  This package's version.
Mesh : TypeAlias
  A triangle mesh as the scenes take it and `decompose` returns it: a
  (vertices, faces) tuple of (n, 3) float64 vertices and (m, 3) int64 triangle
  indices into them.
exchange : module
  Reading and writing data formats: draco and SVG paths.

Examples
--------
>>> import numpy as np
>>> import trimkit
>>> isinstance(trimkit.__version__, str)
True
>>> mesh: trimkit.Mesh = (np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64))
>>> len(mesh)
2

Notes
-----
Built from: mapbox/earcut.hpp 2fbafad +patches, kmammou/v-hacd f900e42 +patches, jpcy/xatlas f700c77 +patches, fastfloat/fast_float dbeabee, jk-jeon/dragonbox beeeef9, google/draco d429de8 +patches, jbikker/tinybvh c9be310 +patches, zeux/meshoptimizer 9e1f07b, coal-library/coal 83cdaa6 +patches, libeigen/eigen f6db5ea, wjakob/nanobind 68480a9, Tessil/robin-map 4ec1bf1, svgpath (ours).
"""

from collections.abc import Sequence
from typing import Annotated, TypeAlias

import numpy
from numpy.typing import NDArray

from trimkit import exchange as exchange


Mesh: TypeAlias = tuple[NDArray[numpy.float64], NDArray[numpy.int64]]

class RayScene:
    """
    Meshes, each built once into a double precision BVH, and the only
    thing rays are cast against.

    Every query names its instances (which mesh, placed where), as a
    CollisionScene's do, so a bare query casts at each mesh once, where it
    is. Instances are placed under a top-level BVH built from exactly the
    arrays a query gives; the scene remembers the last one, so a query naming
    the same instances again (the same bytes) costs only comparing them, and
    moving anything costs that top level, never a mesh. Answers never depend
    on what was asked before.

    Parameters
    ----------
    meshes : sequence of Mesh
      The meshes to place, each a (vertices, faces) tuple of (n, 3) float64
      vertices and (m, 3) int64 triangle indices into them. Copied once
      into tinybvh's padded float layout, which the scene owns, so a mesh
      can never change under the threads sharing the scene.

    Examples
    --------
    >>> import numpy as np
    >>> import trimkit
    >>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> faces = np.array([[0, 1, 2]], dtype=np.int64)
    >>> scene = trimkit.RayScene([(triangle, faces)])
    >>> len(scene)
    1
    """

    def __init__(self, meshes: Sequence[Mesh]) -> None: ...

    def __len__(self) -> int:
        """
        How many meshes the scene was built from.

        Returns
        -------
        count : int
          The number of meshes the scene was built from.
        """

    def boxes(self, instance_matrices: Annotated[NDArray[numpy.float64], dict(shape=(None, 4, 4), order='C', device='cpu', writable=False)] | None = None, instance_index: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> Boxes:
        """
        The box each instance is culled by, as a query with these instances
        places it: what the scene thinks it holds.

        Parameters
        ----------
        instance_matrices : (k, 4, 4) float64, optional
          Each instance's affine transform from its mesh into the scene.
          Defaults to the identity.
        instance_index : (k,) int64, optional
          Which mesh each instance places. Defaults to each mesh once.

        Returns
        -------
        boxes : Boxes
          One 3D box per instance, in its order, holding every face it places
          (an instance of a mesh without faces: a point at its translation).

        Examples
        --------
        >>> import numpy as np
        >>> import trimkit
        >>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> faces = np.array([[0, 1, 2]], dtype=np.int64)
        >>> boxes = trimkit.RayScene([(triangle, faces)]).boxes()
        >>> boxes.bounds.tolist()
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        """

    def intersect_first(self, origins: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], directions: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], instance_matrices: Annotated[NDArray[numpy.float64], dict(shape=(None, 4, 4), order='C', device='cpu', writable=False)] | None = None, instance_index: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> tuple[NDArray[numpy.int64], NDArray[numpy.int64], NDArray[numpy.float64]]:
        """
        The nearest triangle each ray hits, from either side.

        Parameters
        ----------
        origins : (r, 3) float64
          Where each ray starts.
        directions : (r, 3) float64
          Each ray's direction, which need not be unit length.
        instance_matrices : (k, 4, 4) float64, optional
          Each instance's affine transform from its mesh into the scene.
          Defaults to the identity.
        instance_index : (k,) int64, optional
          Which mesh each instance places. Defaults to each mesh once.

        Returns
        -------
        instances : (r,) int64
          Instance of each first hit, or -1 for a miss.
        faces : (r,) int64
          Face of each first hit in its instance's mesh, or -1.
        distances : (r,) float64
          Distance from the origin to the hit, or infinity.

        Examples
        --------
        >>> import numpy as np
        >>> import trimkit
        >>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> faces = np.array([[0, 1, 2]], dtype=np.int64)
        >>> scene = trimkit.RayScene([(triangle, faces)])
        >>> origins, directions = np.array([[0.2, 0.2, 1.0]]), np.array([[0.0, 0.0, -1.0]])
        >>> instances, faces, distances = scene.intersect_first(origins, directions)
        >>> instances.tolist(), faces.tolist(), distances.tolist()
        ([0], [0], [1.0])
        """

    def intersect_all(self, origins: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], directions: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], instance_matrices: Annotated[NDArray[numpy.float64], dict(shape=(None, 4, 4), order='C', device='cpu', writable=False)] | None = None, instance_index: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> tuple[NDArray[numpy.int64], NDArray[numpy.int64], NDArray[numpy.int64], NDArray[numpy.float64]]:
        """
        Every triangle each ray hits, from either side.

        Parameters
        ----------
        origins : (r, 3) float64
          Where each ray starts.
        directions : (r, 3) float64
          Each ray's direction, which need not be unit length.
        instance_matrices : (k, 4, 4) float64, optional
          Each instance's affine transform from its mesh into the scene.
          Defaults to the identity.
        instance_index : (k,) int64, optional
          Which mesh each instance places. Defaults to each mesh once.

        Returns
        -------
        rays : (h,) int64
          Ray of each hit, one row per hit, sorted by ray and then
          distance. A ray through a shared edge or vertex hits every triangle
          there.
        instances : (h,) int64
          Instance of each hit.
        faces : (h,) int64
          Face of each hit in its instance's mesh.
        distances : (h,) float64
          Distance from the ray's origin to each hit.

        Examples
        --------
        >>> import numpy as np
        >>> import trimkit
        >>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> faces = np.array([[0, 1, 2]], dtype=np.int64)
        >>> scene = trimkit.RayScene([(triangle, faces)])
        >>> origins, directions = np.array([[0.2, 0.2, 1.0]]), np.array([[0.0, 0.0, -1.0]])
        >>> rays, instances, faces, distances = scene.intersect_all(origins, directions)
        >>> rays.tolist(), distances.tolist()
        ([0], [1.0])
        """

    def intersect_any(self, origins: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], directions: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], instance_matrices: Annotated[NDArray[numpy.float64], dict(shape=(None, 4, 4), order='C', device='cpu', writable=False)] | None = None, instance_index: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> NDArray[numpy.bool_]:
        """
        Whether each ray hits anything, stopping at the first triangle found.

        Parameters
        ----------
        origins : (r, 3) float64
          Where each ray starts.
        directions : (r, 3) float64
          Each ray's direction, which need not be unit length.
        instance_matrices : (k, 4, 4) float64, optional
          Each instance's affine transform from its mesh into the scene.
          Defaults to the identity.
        instance_index : (k,) int64, optional
          Which mesh each instance places. Defaults to each mesh once.

        Returns
        -------
        hits : (r,) bool
          True for every ray that hits a triangle.

        Examples
        --------
        >>> import numpy as np
        >>> import trimkit
        >>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> faces = np.array([[0, 1, 2]], dtype=np.int64)
        >>> scene = trimkit.RayScene([(triangle, faces)])
        >>> origins, directions = np.array([[0.2, 0.2, 1.0]]), np.array([[0.0, 0.0, -1.0]])
        >>> hits = scene.intersect_any(origins, directions)
        >>> hits.tolist()
        [True]
        """

class Boxes:
    """
    Axis-aligned boxes built into a BVH for overlap queries: a static,
    numpy-native replacement for a bulk-loaded R-tree.

    Parameters
    ----------
    boxes : (n, 2, d) float64
      Each box's minimum and maximum corner, in 2 or 3 dimensions. Copied
      into the tree the object owns, so the boxes can't change under it.

    Examples
    --------
    >>> import numpy as np
    >>> import trimkit
    >>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
    >>> len(tree)
    2
    """

    def __init__(self, boxes: Annotated[NDArray[numpy.float64], dict(shape=(None, 2, None), order='C', device='cpu', writable=False)]) -> None: ...

    def __len__(self) -> int:
        """
        How many boxes it holds.

        Returns
        -------
        count : int
          The number of boxes it holds.
        """

    @property
    def bounds(self) -> NDArray[numpy.float64]:
        """
        The box around every box.

        Returns
        -------
        bounds : (2, d) float64
          The minimum and maximum corner over every box, or +inf then -inf
          with no boxes.

        Examples
        --------
        >>> import numpy as np
        >>> import trimkit
        >>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
        >>> bounds = tree.bounds
        >>> bounds.tolist()
        [[0.0, 0.0], [3.0, 3.0]]
        """

    def boxes(self) -> NDArray[numpy.float64]:
        """
        Every box, exactly as given.

        Returns
        -------
        boxes : (n, 2, d) float64
          Each box's minimum and maximum corner, a fresh copy.

        Examples
        --------
        >>> import numpy as np
        >>> import trimkit
        >>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
        >>> boxes = tree.boxes()
        >>> boxes.shape
        (2, 2, 2)
        """

    def intersect(self, queries: Annotated[NDArray[numpy.float64], dict(shape=(None, 2, None), order='C', device='cpu', writable=False)]) -> NDArray[numpy.int64]:
        """
        Every box overlapping each query box, touching included.

        Parameters
        ----------
        queries : (m, 2, d) float64
          Each query's minimum and maximum corner, in the boxes' dimension.

        Returns
        -------
        pairs : (k, 2) int64
          Rows of [query, box] for every overlap, sorted by query
          and then box.

        Examples
        --------
        >>> import numpy as np
        >>> import trimkit
        >>> tree = trimkit.Boxes(np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]]))
        >>> pairs = tree.intersect(np.array([[[0.5, 0.5], [2.5, 2.5]]]))
        >>> pairs.tolist()
        [[0, 0], [0, 1]]
        """

def triangulate(exterior: Annotated[NDArray[numpy.float64], dict(shape=(None, 2), order='C', device='cpu', writable=False)], interiors: Sequence[Annotated[NDArray[numpy.float64], dict(shape=(None, 2), order='C', device='cpu', writable=False)]] = ()) -> tuple[NDArray[numpy.float64], NDArray[numpy.int64]]:
    """
    Triangulate a polygon with holes.

    Parameters
    ----------
    exterior : (n, 2) float64
      The polygon's outer ring, as shapely's `exterior.coords`. A repeated
      closing vertex is fine.
    interiors : sequence of (n_i, 2) float64
      Each hole's ring, as shapely's `interiors[i].coords`.

    Returns
    -------
    vertices : (N, 2) float64
      Every ring stacked in order, the exterior then each interior.
    faces : (t, 3) int64
      Triangle indices into `vertices`.

    Examples
    --------
    >>> import numpy as np
    >>> import trimkit
    >>> square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> vertices, faces = trimkit.triangulate(square)
    >>> faces.tolist()
    [[2, 3, 0], [2, 0, 1]]
    """

def decompose(vertices: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], faces: Annotated[NDArray[numpy.int64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], *, maximum_hulls: int | numpy.integer = 64, maximum_vertices: int | numpy.integer = 64, resolution: int | numpy.integer = 400000, maximum_error: float | numpy.floating = 1.0) -> list[Mesh]:
    """
    Decompose a closed triangle mesh into approximately convex hulls.

    Parameters
    ----------
    vertices : (n, 3) float64
      Mesh vertices.
    faces : (m, 3) int64
      Triangle indices into `vertices`, a closed surface: one with holes
      comes out hollow.
    maximum_hulls : int
      The most hulls to return.
    maximum_vertices : int
      The most vertices in each hull.
    resolution : int
      How finely to sample the mesh, in voxels: raise it for thin features.
    maximum_error : float
      Stop splitting a hull once its volume is within this percent of what it
      covers, from 0 to 100.

    Returns
    -------
    hulls : list of Mesh
      One closed convex (vertices, faces) per hull, as `RayScene` and
      `CollisionScene` take them: empty for input with no volume.

    Examples
    --------
    >>> import numpy as np
    >>> import trimkit
    >>> cube = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    >>> faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
    ...                   [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int64)
    >>> hulls = trimkit.decompose(cube, faces, maximum_hulls=1)
    >>> vertices, faces = hulls[0]
    >>> len(hulls), len(vertices), len(faces)
    (1, 8, 12)
    """

def parametrize(vertices: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], faces: Annotated[NDArray[numpy.int64], dict(shape=(None, 3), order='C', device='cpu', writable=False)]) -> tuple[NDArray[numpy.int64], NDArray[numpy.int64], NDArray[numpy.float64]]:
    """
    Cut a mesh into charts and pack them into a UV atlas.

    Parameters
    ----------
    vertices : (n, 3) float64
      Mesh vertices.
    faces : (m, 3) int64
      Triangle indices into `vertices`.

    Returns
    -------
    vertex_sources : (k,) int64
      The input vertex each output vertex came from.
    faces : (m, 3) int64
      Triangles indexing the output vertices.
    uv : (k, 2) float64
      Texture coordinates in the unit square.

    Examples
    --------
    >>> import numpy as np
    >>> import trimkit
    >>> triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> faces = np.array([[0, 1, 2]], dtype=np.int64)
    >>> vertex_sources, faces, uv = trimkit.parametrize(triangle, faces)
    >>> vertex_sources.tolist(), uv.shape
    ([0, 1, 2], (3, 2))
    """

def simplify(vertices: Annotated[NDArray[numpy.float64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], faces: Annotated[NDArray[numpy.int64], dict(shape=(None, 3), order='C', device='cpu', writable=False)], face_count: int | numpy.integer, *, maximum_error: float | numpy.floating = float('inf'), lock_border: bool | numpy.bool_ = False) -> tuple[NDArray[numpy.float64], NDArray[numpy.int64], NDArray[numpy.int64]]:
    """
    Collapse edges by quadric error until at most `face_count` faces remain.

    Parameters
    ----------
    vertices : (n, 3) float64
      Mesh vertices, merged: edges only collapse across shared vertices, so an
      unmerged triangle soup comes back unchanged.
    faces : (m, 3) int64
      Triangle indices into `vertices`.
    face_count : int
      Stop once at most this many faces remain. Topology can stop it sooner,
      leaving more faces than asked for.
    maximum_error : float
      Stop before the surface moves farther than this, in the units of
      `vertices`. The default stops only on `face_count`.
    lock_border : bool
      Keep every vertex on an open boundary edge.

    Returns
    -------
    vertices : (k, 3) float64
      The output vertices.
    faces : (t, 3) int64
      Triangles indexing the output vertices.
    vertex_sources : (k,) int64
      The input vertex each output vertex is, or -1 for one that is new, so
      per-vertex attributes carry over by index. Today none is new: the
      output vertices are exactly `vertices[vertex_sources]`.

    Examples
    --------
    >>> import numpy as np
    >>> import trimkit
    >>> tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    >>> vertices, faces, vertex_sources = trimkit.simplify(tetrahedron, faces, 4)
    >>> len(faces), bool((vertices == tetrahedron[vertex_sources]).all())
    (4, True)
    """

class CollisionScene:
    """
    Meshes to place and query against each other. A scene never changes:
    each query names its own instances, so threads may share one.

    Every query takes its instances (which mesh, placed where) and the pairs
    of them to test, all optional, so a bare `scene.collides()` asks about the
    scene as built: each mesh once, where it is, every pair. A robot passes its
    links' transforms and the pairs of links not joined to each other.

    `pairs` names exactly the pairs to test, which is how anything is left
    out: a robot's ignored links, obstacles against each other. A path is one
    query: N poses of L links are N * L instances, each pose's pairs named
    (`pairs=None` would pair a link with itself at another pose), and the
    pairs barely change from call to call, so build them once. Distances over
    every pair are O(k**2) in memory. For gradients, use convex meshes: two
    surfaces are 0 apart with no normal once they touch.

    A convex mesh is tested as a solid and any other mesh as a surface, in
    every pair: a mesh inside a convex one touches it, and a distance is
    negative only when a convex solid is penetrated.

    Parameters
    ----------
    meshes : sequence of Mesh
      The meshes to place, each a (vertices, faces) tuple of (n, 3) float64
      vertices and (m, 3) int64 triangle indices into them. Copied once
      into coal's own storage, which the scene owns, so a mesh can never
      change under the threads sharing the scene.
    convex : bool or (n,) bool, optional
      Which meshes to measure as convex solids, such as convex hulls:
      all of them or none, or a mask of which. Not checked: a mesh that
      isn't convex gives wrong distances, never a crash. Defaults to
      False.

    Examples
    --------
    >>> import numpy as np
    >>> import trimkit
    >>> tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    >>> scene = trimkit.CollisionScene([(tetrahedron, faces)], convex=True)
    >>> len(scene)
    1
    """

    def __init__(self, meshes: Sequence[Mesh], convex: bool | numpy.bool_ | Annotated[NDArray[numpy.bool_], dict(shape=(None,), order='C', device='cpu', writable=False)] = False) -> None: ...

    def __len__(self) -> int:
        """
        How many meshes the scene was built from.

        Returns
        -------
        count : int
          The number of meshes the scene was built from.
        """

    def boxes(self, instance_matrices: Annotated[NDArray[numpy.float64], dict(shape=(None, 4, 4), order='C', device='cpu', writable=False)] | None = None, instance_index: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> Boxes:
        """
        The box each instance is culled by, as a query with these instances
        places it: what the scene thinks it holds.

        Parameters
        ----------
        instance_matrices : (k, 4, 4) float64, optional
          Each instance's rigid transform from its mesh into the scene.
          Defaults to the identity.
        instance_index : (k,) int64, optional
          Which mesh each instance places. Defaults to each mesh once.

        Returns
        -------
        boxes : Boxes
          One 3D box per instance, in its order, holding every face it places
          (an instance of a mesh without faces: a point at its translation).

        Examples
        --------
        >>> import numpy as np
        >>> import trimkit
        >>> tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
        >>> scene = trimkit.CollisionScene([(tetrahedron, faces)], convex=True)
        >>> placed = np.tile(np.eye(4), (2, 1, 1))
        >>> boxes = scene.boxes()
        >>> boxes.bounds.tolist()
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        """

    def collides(self, instance_matrices: Annotated[NDArray[numpy.float64], dict(shape=(None, 4, 4), order='C', device='cpu', writable=False)] | None = None, instance_index: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None, *, pairs: Annotated[NDArray[numpy.int64], dict(shape=(None, 2), order='C', device='cpu', writable=False)] | None = None, margin: float | numpy.floating = 0.0) -> NDArray[numpy.int64]:
        """
        The pairs of instances that touch.

        Parameters
        ----------
        instance_matrices : (k, 4, 4) float64, optional
          Each instance's rigid transform from its mesh into the scene.
          Defaults to the identity.
        instance_index : (k,) int64, optional
          Which mesh each instance places. Defaults to each mesh once.
        pairs : (p, 2) int64, optional
          The pairs of instances to test. Defaults to every pair, found by
          their bounding boxes.
        margin : float
          Also count pairs closer than this.

        Returns
        -------
        pairs : (c, 2) int64
          Each pair of instances that touches or overlaps: rows of `pairs` in
          its order, or by default each touching i < j in ascending order.
          Empty when nothing touches.

        Examples
        --------
        >>> import numpy as np
        >>> import trimkit
        >>> tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
        >>> scene = trimkit.CollisionScene([(tetrahedron, faces)], convex=True)
        >>> placed = np.tile(np.eye(4), (2, 1, 1))
        >>> placed[1, 0, 3] = 0.5
        >>> pairs = scene.collides(placed, np.zeros(2, dtype=np.int64))
        >>> pairs.tolist()
        [[0, 1]]
        """

    def distances(self, instance_matrices: Annotated[NDArray[numpy.float64], dict(shape=(None, 4, 4), order='C', device='cpu', writable=False)] | None = None, instance_index: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None, *, pairs: Annotated[NDArray[numpy.int64], dict(shape=(None, 2), order='C', device='cpu', writable=False)] | None = None, upper_bound: float | numpy.floating = float('inf')) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]:
        """
        The distance between each pair of instances, and where it is.

        Parameters
        ----------
        instance_matrices : (k, 4, 4) float64, optional
          Each instance's rigid transform from its mesh into the scene.
          Defaults to the identity.
        instance_index : (k,) int64, optional
          Which mesh each instance places. Defaults to each mesh once.
        pairs : (p, 2) int64, optional
          The pairs of instances to measure. Defaults to every pair, in the
          order of `numpy.triu_indices(k, 1)`.
        upper_bound : float
          Pairs farther apart than this are infinitely far, with no points or
          normal, and are measured only as far as it takes to tell.

        Returns
        -------
        distances : (p,) float64
          The distance between each pair. A pair with a convex mesh has a signed
          distance, negative by how deep it reaches into the solid; two surfaces
          are 0 once their triangles touch.
        points : (p, 2, 3) float64
          The closest point on each instance of a pair, in the scene.
        normals : (p, 3) float64
          The unit direction from the first instance toward the second along
          which the distance is measured, or NaN for two surfaces that touch.

        Examples
        --------
        >>> import numpy as np
        >>> import trimkit
        >>> tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
        >>> scene = trimkit.CollisionScene([(tetrahedron, faces)], convex=True)
        >>> placed = np.tile(np.eye(4), (2, 1, 1))
        >>> placed[1, 0, 3] = 2.0
        >>> distances, points, normals = scene.distances(placed, np.zeros(2, dtype=np.int64))
        >>> distances.tolist(), normals.tolist()
        ([1.0], [[1.0, 0.0, 0.0]])
        """
