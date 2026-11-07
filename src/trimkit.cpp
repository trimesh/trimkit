// The one extension: every part's functions and classes in one module,
// `import trimkit`, built as the package itself (trimkit/__init__ with the
// extension suffix), so there is no Python in it at all. The top level asks
// geometry questions whatever library answers them; its one submodule,
// `trimkit.exchange`, reads and writes specific formats.
#include "common.hpp"

NB_MODULE(trimkit, m) {  // NOLINT(performance-unnecessary-value-param): the macro passes it by value
    // the last paragraph names what this build compiled in, each upstream's
    // repository and commit, or `(system)` for a distribution's copy
    m.doc() = R"(Thin, stable-ABI numpy bindings to pinned C and C++ geometry libraries.

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
Built from: )" TRIMKIT_UPSTREAM ".";
    m.attr("__version__") = TRIMKIT_VERSION;
    // a triangle mesh as scenes take them: (n, 3) float64 vertices and
    // (m, 3) int64 faces, the shapes every docstring states
    const nb::object ndarray = nb::module_::import_("numpy.typing").attr("NDArray");
    const nb::module_ numpy = nb::module_::import_("numpy");
    m.attr("Mesh") = nb::module_::import_("builtins")
                         .attr("tuple")[nb::make_tuple(ndarray[numpy.attr("float64")], ndarray[numpy.attr("int64")])];
    // a module of the extension, importable as `trimkit.exchange` too
    nb::module_ exchange = m.def_submodule("exchange", R"(Reading and writing data formats: draco and SVG paths.

Attributes
----------
DracoKind : enum.IntEnum
  What a draco attribute holds, as `draco_encode` takes it.
SvgKind : enum.IntEnum
  What drew an SvgSegment, its `kind` field.
SvgSegment : numpy.dtype
  One drawing command of an SVG path, as `svg_parse` returns them and
  `svg_string` takes them.

Examples
--------
>>> import trimkit
>>> trimkit.exchange.svg_parse("M 0 0 L 1 1")[1].dtype == trimkit.exchange.SvgSegment
True)");
    nb::module_::import_("sys").attr("modules")["trimkit.exchange"] = exchange;
    bind_bvh(m);
    bind_earcut(m);
    bind_vhacd(m);
    bind_xatlas(m);
    bind_draco(exchange);
    bind_svg(exchange);
    bind_meshoptimizer(m);
    bind_collision(m);
}
