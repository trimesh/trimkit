// The one door between numpy and our C++: the only file that touches
// nanobind's ndarray, so every array in and out of the package passes the
// same checks, and a binding can't forget one.
//
// In: `read` (typed arrays), `read_indices` (faces), `read_any` (a runtime
// dtype) and `read_records` (a structured dtype) return views that are
// already known to be exactly the documented dtype and shape, C-contiguous,
// aligned and, for float64, finite and within COORDINATE_LIMIT (float32:
// finite). A C-contiguous, aligned array of the exact dtype is read in place,
// never copied. One of the exact dtype in any other layout (strided,
// reversed, Fortran-ordered, misaligned) is copied once, exactly as
// `numpy.ascontiguousarray` would: what a caller would have written, and
// lossless. An array of another dtype that numpy casts safely (float32 to
// float64, int32 to int64: lossless), or a list, is converted once with a
// UserWarning naming the argument: the copy is what the caller would have
// written, and the warning says so. Anything else, a dtype that would lose
// values or an object that isn't array-like, is a TypeError (or for values,
// a ValueError) naming the argument.
// Read-only, mmap'd and buffer-protocol inputs are all fine. Call them with
// the GIL held; long scans release it themselves. The one copy always made
// is `read_indices`': faces arrive as int64 and each library takes its own
// index type, so they are checked and narrowed into a vector as they're read.
//
// Out: `to_numpy` moves a std::vector into a capsule-owned numpy array: a
// fresh, writable array that owns its memory and shares none with an input.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace trimkit {

// Every float64 input lies within +-1e30, so no arithmetic downstream can
// overflow: the deepest products in any module are degree 6 (an instance
// transform times a coordinate, cubed by the watertight ray test, or squared
// twice by tinybvh's surface areas), and 1e180 is far from DBL_MAX. Unbounded,
// rays silently missed everything past ~5.6e102 and tinybvh's builder cast
// NaN to int near 1e300. No real geometry is near it: the observable universe
// is ~8.8e29 mm across.
constexpr double COORDINATE_LIMIT = 1e30;
// how messages name it: the UTF-8 bytes of the plus-minus sign, as bytes so
// the source stays ASCII, split so `1e3` isn't read as more hex digits
constexpr const char* COORDINATE_LIMIT_TEXT =
    "\xc2\xb1"
    "1e30";
static_assert(COORDINATE_LIMIT == 1e30, "COORDINATE_LIMIT_TEXT names the limit");

// a C-contiguous input view with a fixed trailing dimension
template <typename T, nb::ssize_t... Shape>
using In = nb::ndarray<const T, nb::shape<Shape...>, nb::c_contig, nb::device::cpu>;

// any dtype, C-contiguous: for inputs whose dtype is chosen at runtime
using AnyIn = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

// how a sequence of any-dtype arrays reads in a signature; `read_any` each item
using AnyArguments = nb::typed<nb::sequence, nb::ndarray<>>;

// An argument that must already be an `In<T, Shape...>`. It reads as that
// array type in the signature and stubs, but nanobind accepts any object for
// it, and `read` checks it itself: in place when exact, copied or converted
// once as the header says, and a TypeError that names the argument instead
// of an overload dump.
template <typename T, nb::ssize_t... Shape>
struct Array {
    nb::handle object;
    bool is_none() const { return object.is_none(); }
};

// Rays' origins or directions as an argument: a row of float64 or float32 per
// ray, (n, 3), or one row for every ray, (3,). `read_rays` reads it, and the
// signature shows them all
struct RaysArgument {
    nb::handle object;
};

// Boxes' queries as an argument: (2d,) or (2, d) for one query, (q, 2d) or
// (q, 2, d) for many, each (q, 2, d) in memory. `Boxes::intersection` reads
// it with `read`, as whichever of the three array shapes it is
struct BoxQueriesArgument {
    nb::handle object;
};

// What `read_rays` read: `count` rows of three, or one row that is every
// ray's (`one`, with a count of 1), float64 or, for a float32 array taken as
// it is, float32. Valid while it is: it holds the argument, or its one copy
struct Rays {
    AnyIn held;
    const void* data = nullptr;
    std::size_t count = 0;
    bool is_float32 = false;
    bool one = false;
};

// row `i` of rays as float64: a float32 widens to exactly itself
inline void ray_row(const Rays& rays, std::size_t i, double* row) {
    const std::size_t first = rays.one ? 0 : 3 * i;
    for (std::size_t k = 0; k < 3; k++) {
        row[k] = rays.is_float32 ? static_cast<double>(static_cast<const float*>(rays.data)[first + k])
                                 : static_cast<const double*>(rays.data)[first + k];
    }
}

// A bool argument: Python's, or numpy's scalar, which is what `array.all()` and
// every other numpy test returns. Nothing else counts as one: not 0, 1 or None
struct Flag {
    bool value = false;
    operator bool() const { return value; }  // NOLINT(google-explicit-constructor): read as the bool it is
};

// A number argument: Python's or numpy's, as nanobind's own conversion takes it,
// and shown as both in the stubs, as numpy's scalars aren't Python's to a type checker
template <typename T>
struct Number {
    T value{};
    operator T() const { return value; }  // NOLINT(google-explicit-constructor): read as the number it is
};

// a numpy output: fresh, writable, owned by a capsule
template <typename T>
using Out = nb::ndarray<nb::numpy, T>;

// a numpy output whose dtype is only known at runtime, and a tuple of them
using AnyOut = nb::ndarray<nb::numpy>;
using AnyOuts = nb::typed<nb::tuple, AnyOut, nb::ellipsis>;

// numpy's objects kept for every call after: called once, by NB_MODULE
inline void init_numpy();

// raise a Python exception from a built-up message
[[noreturn]] inline void value_error(const std::string& message) { throw nb::value_error(message.c_str()); }
[[noreturn]] inline void type_error(const std::string& message) { throw nb::type_error(message.c_str()); }
// what this machine can't do at all, whatever the input (nanobind maps no C++
// exception to NotImplementedError, so it is set as CPython's own)
[[noreturn]] inline void not_implemented_error(const std::string& message) {
    PyErr_SetString(PyExc_NotImplementedError, message.c_str());
    throw nb::python_error();
}

namespace detail {

// numpy's own objects, looked up once as the module is made (`init_numpy`,
// the first thing NB_MODULE does) and kept for good: never released, so
// nothing of ours runs when the interpreter ends, and only read afterwards
struct Numpy {
    nb::handle ndarray;
    nb::handle bool_;
    nb::handle asarray;
    nb::handle array;
    nb::handle dtype;
    nb::handle can_cast;
};
inline Numpy numpy;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// "(*, 3)" for shape<-1, 3>, the way the stubs write it
template <nb::ssize_t... Shape>
std::string shape_text() {
    std::string text = "(";
    bool first = true;
    for (const nb::ssize_t extent : {Shape...}) {
        text += first ? "" : ", ";
        text += extent < 0 ? "*" : std::to_string(extent);
        first = false;
    }
    return text + (sizeof...(Shape) == 1 ? ",)" : ")");
}

// Python's str() of any object, for messages
inline std::string text(nb::handle object) { return nb::str(object).c_str(); }

// a short dtype name for messages: "float64", "int64", ...
template <typename T>
constexpr const char* dtype_name() {
    if constexpr (std::is_same_v<T, double>) {
        return "float64";
    } else if constexpr (std::is_same_v<T, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return "int64";
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        return "uint8";
    } else {
        static_assert(std::is_same_v<T, bool>, "a dtype numpy names");
        return "bool";
    }
}

// whether a view starts on an address `alignment` divides
inline bool aligned(const void* data, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(data) % alignment == 0;
}

}  // namespace detail

// The one scan of values there is. How many of `count` values are not within
// `limit`: beyond it, an infinity or not a number, one comparison that is
// false for all three, counted without branching so it vectorizes. Each value
// is written to `copy` as it is read, when there is one. Python-free
template <typename T>
std::size_t count_beyond(const T* values, std::size_t count, double limit, T* copy) {
    std::size_t beyond = 0;
    for (std::size_t i = 0; i < count; i++) {
        const T value = values[i];
        if (copy != nullptr) {
            copy[i] = value;
        }
        beyond += !(static_cast<double>(std::abs(value)) <= limit) ? 1U : 0U;
    }
    return beyond;
}

// refuse values not within `limit`, scanned without the GIL, naming the argument
template <typename T>
void check_values(const T* values, std::size_t count, double limit, const std::string& name) {
    std::size_t beyond = 0;
    {
        const nb::gil_scoped_release release;
        beyond = count_beyond<T>(values, count, limit, nullptr);
    }
    if (beyond != 0) {
        value_error(name + " must be finite" +
                    (limit == COORDINATE_LIMIT ? std::string(" and within ") + COORDINATE_LIMIT_TEXT : std::string()));
    }
}

// what an object that isn't the expected array actually is, for messages
inline std::string describe(nb::handle object) {
    if (nb::hasattr(object, "dtype") && nb::hasattr(object, "shape")) {
        const std::string dtype = detail::text(object.attr("dtype"));
        const bool vowel = !dtype.empty() && std::string("aeiou").find(dtype[0]) != std::string::npos;
        std::string description =
            (vowel ? "an " : "a ") + dtype + " array of shape " + detail::text(object.attr("shape"));
        if (nb::hasattr(object, "flags") && PyObject_IsTrue(object.attr("flags").attr("c_contiguous").ptr()) == 0) {
            description += " that isn't C-contiguous";
        }
        return description;
    }
    return detail::text(object.type().attr("__name__"));
}

// The one warning the extension gives, a UserWarning at the caller: something
// asked for happened another way (a conversion, the CPU answering for a
// device). A filter that makes warnings errors raises here, before anything runs
inline void warn(const std::string& message) {
    if (PyErr_WarnEx(PyExc_UserWarning, message.c_str(), 1) != 0) {
        throw nb::python_error();
    }
}

namespace detail {

// `name` arrived as `source` and is read as a fresh `dtype` copy
inline void warn_converted(const std::string& name, const std::string& source, const char* dtype) {
    warn(name + ": " + source + " converted to " + dtype + " (a copy); pass a " + dtype + " array to avoid it");
}

// What numpy reads an object as, without converting or warning: an ndarray
// itself, a list or any other sequence as `numpy.asarray` makes it, or
// nothing (a ragged list, a scene, None)
inline nb::object seen_as_array(nb::handle object) {
    if (PyObject_IsInstance(object.ptr(), numpy.ndarray.ptr()) == 1) {
        return nb::borrow(object);
    }
    try {
        return numpy.asarray(object);
    } catch (const nb::python_error&) {
        return {};
    }
}

// whether an array has the shape an In<T, Shape...> takes: its dimensions,
// and each extent the type fixes
template <nb::ssize_t... Shape>
bool shaped(const nb::object& array) {
    const nb::tuple shape = nb::borrow<nb::tuple>(array.attr("shape"));
    if (nb::len(shape) != sizeof...(Shape)) {
        return false;
    }
    std::size_t dimension = 0;
    for (const nb::ssize_t extent : {Shape...}) {
        if (extent >= 0 && nb::cast<nb::ssize_t>(shape[dimension]) != extent) {
            return false;
        }
        dimension++;
    }
    return true;
}

// An object numpy sees as `array` (`seen_as_array`'s answer for it), of
// exactly `dtype` (any dtype for null), that can't be read
// in place, as one fresh C-contiguous copy, aligned as every fresh numpy
// buffer is (a view at an odd byte offset would make every read of it
// undefined; a packed structured dtype counts as aligned to numpy there, so
// this always copies):
// - an ndarray of that dtype in another layout, what `numpy.ascontiguousarray`
//   gives, silently: lossless, and what a caller would have written
// - with a dtype named, an ndarray of another dtype that numpy casts safely
//   (`can_cast(..., "safe")`: float32 to float64, int32 to int64, never the
//   other way), or a list or any other sequence `numpy.asarray` reads,
//   converted with a UserWarning naming the argument
// Anything else gets nothing: it stays refused
inline nb::object relaid(nb::handle object, const nb::object& array, const char* dtype, const std::string& name) {
    const bool is_array = array.is(object);
    if (!array.is_valid() || (!is_array && dtype == nullptr)) {
        return {};
    }
    const std::string source = is_array ? text(array.attr("dtype")) : "a " + text(object.type().attr("__name__"));
    if (dtype == nullptr || array.attr("dtype").equal(numpy.dtype(dtype))) {
        if (!is_array) {
            warn_converted(name, source, dtype);
        }
        return numpy.array(array, nb::arg("copy") = true, nb::arg("order") = "C");
    }
    const nb::object wanted = numpy.dtype(dtype);
    if (!nb::cast<bool>(numpy.can_cast(array.attr("dtype"), wanted, "safe"))) {
        return {};
    }
    warn_converted(name, source, dtype);
    return numpy.array(array, nb::arg("dtype") = wanted, nb::arg("copy") = true, nb::arg("order") = "C");
}

// view an object as exactly an In<T, Shape...>, aligned for T: in place, or
// copied once when its layout is wrong, or converted once (with a warning)
// when its dtype casts safely. Its shape is checked first, so a wrong one is
// refused before anything is copied or warned about. The view holds what it
// reads
template <typename T, nb::ssize_t... Shape>
In<T, Shape...> view(nb::handle object, const std::string& name) {
    In<T, Shape...> array;
    bool exact = nb::try_cast(object, array, /*convert=*/false) && aligned(array.data(), alignof(T));
    if (!exact) {
        const nb::object seen = seen_as_array(object);
        if (seen.is_valid() && shaped<Shape...>(seen)) {
            const nb::object copy = relaid(object, seen, dtype_name<T>(), name);
            exact =
                copy.is_valid() && nb::try_cast(copy, array, /*convert=*/false) && aligned(array.data(), alignof(T));
        }
    }
    if (!exact) {
        type_error(name + " must be " + dtype_name<T>() + " of shape " + shape_text<Shape...>() + ", not " +
                   describe(object));
    }
    return array;
}

}  // namespace detail

// An array argument, checked: exactly T and Shape, C-contiguous, aligned, and
// finite for floating T. The view is valid while the argument is.
template <typename T, nb::ssize_t... Shape>
In<T, Shape...> read(nb::handle object, const std::string& name) {
    In<T, Shape...> array = detail::view<T, Shape...>(object, name);
    if constexpr (std::is_same_v<T, double>) {
        check_values(array.data(), array.size(), COORDINATE_LIMIT, name);
    }
    return array;
}

template <typename T, nb::ssize_t... Shape>
In<T, Shape...> read(const Array<T, Shape...>& argument, const std::string& name) {
    return read<T, Shape...>(argument.object, name);
}

namespace detail {

// whether `candidate` is an array of any dtype readable in place, viewed into `array`
inline bool any_readable(nb::handle candidate, AnyIn* array) {
    if (!nb::try_cast(candidate, *array, /*convert=*/false)) {
        return false;
    }
    const std::size_t item_size = (array->dtype().bits + 7U) / 8U;
    return item_size == 0 || aligned(array->data(), item_size);
}

// whether an array is float64 or float32 rays: (n, 3), or (3,) for one row
inline bool holds_rays(const AnyIn& array) {
    const bool floats = array.dtype() == nb::dtype<double>() || array.dtype() == nb::dtype<float>();
    return floats && ((array.ndim() == 1 && array.shape(0) == 3) || (array.ndim() == 2 && array.shape(1) == 3));
}

}  // namespace detail

// A rays argument: float64 or float32, a row per ray or one row for every
// ray, read in place when it is already that (no Python is called at all)
// and otherwise as `read` takes any array: its shape checked first, then one
// copy, converted with the warning when its dtype isn't one of the two. Its
// values are not checked here: a device checks them as it copies them
// (src/vulkan/scene.hpp), and anything else calls `check_rays`
inline Rays read_rays(const RaysArgument& argument, const std::string& name) {
    Rays rays;
    bool taken = detail::any_readable(argument.object, &rays.held) && detail::holds_rays(rays.held);
    if (!taken) {
        const nb::object seen = detail::seen_as_array(argument.object);
        if (seen.is_valid() && (detail::shaped<-1, 3>(seen) || detail::shaped<3>(seen))) {
            const bool narrow = seen.attr("dtype").equal(detail::numpy.dtype("float32"));
            const nb::object copy = detail::relaid(argument.object, seen, narrow ? "float32" : "float64", name);
            taken = copy.is_valid() && detail::any_readable(copy, &rays.held) && detail::holds_rays(rays.held);
        }
    }
    if (!taken) {
        type_error(name + " must be float64 or float32 of shape (*, 3) or (3,), not " + describe(argument.object));
    }
    rays.data = rays.held.data();
    rays.is_float32 = rays.held.dtype() == nb::dtype<float>();
    rays.one = rays.held.ndim() == 1;
    rays.count = rays.one ? 1 : rays.held.shape(0);
    return rays;
}

// the values of rays `read_rays` read, refused as `read` refuses them:
// float32 ones are held to the same bound as float64 ones
inline void check_rays(const Rays& rays, const std::string& name) {
    if (rays.is_float32) {
        check_values(static_cast<const float*>(rays.data), 3 * rays.count, COORDINATE_LIMIT, name);
    } else {
        check_values(static_cast<const double*>(rays.data), 3 * rays.count, COORDINATE_LIMIT, name);
    }
}

// An array argument viewed: its dtype, shape and layout as `read` checks
// them, its values not looked at. For what copies the values and checks the
// copy (a query's instances)
template <typename T, nb::ssize_t... Shape>
In<T, Shape...> view(const Array<T, Shape...>& argument, const std::string& name) {
    return detail::view<T, Shape...>(argument.object, name);
}

// A bool array's values as bytes: a byte other than 0 or 1 in a numpy bool
// array (a uint8 array viewed as bool can hold one) is undefined as a C++
// bool, so they are read as uint8 and tested `!= 0`, never as bool
inline const std::uint8_t* mask_bytes(const In<bool, -1>& mask) {
    return reinterpret_cast<const std::uint8_t*>(mask.data());  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

// (m, 3) int64 faces as the index type a vendored library uses, every index
// checked against `vertex_count` and, for uint32, the mesh against its range
template <typename Index>
std::vector<Index> read_indices(const Array<std::int64_t, -1, 3>& argument, const std::string& name,
                                std::size_t vertex_count) {
    const In<std::int64_t, -1, 3> faces = detail::view<std::int64_t, -1, 3>(argument.object, name);
    if constexpr (sizeof(Index) < sizeof(std::size_t)) {
        if (vertex_count >= std::numeric_limits<Index>::max() ||
            faces.shape(0) >= std::numeric_limits<Index>::max() / 3) {
            value_error("this mesh is too large to index with uint32");
        }
    }
    const std::int64_t* values = faces.data();
    std::vector<Index> indices(3 * faces.shape(0));
    // one branchless pass, which vectorizes: copy every index and count those
    // out of range (a negative index wraps to a huge unsigned one)
    std::size_t violations = 0;
    {
        const nb::gil_scoped_release release;
        for (std::size_t i = 0; i < indices.size(); i++) {
            // read once: what is checked is what is kept
            const std::int64_t index = values[i];
            violations += static_cast<std::uint64_t>(index) >= vertex_count ? 1U : 0U;
            indices[i] = static_cast<Index>(index);
        }
    }
    if (violations != 0) {
        // the rare refusal pays for finding the row to name, in the array as
        // it is now: the last row, if someone has changed it since
        std::size_t bad = 0;
        while (bad + 1 < indices.size() && static_cast<std::uint64_t>(values[bad]) < vertex_count) {
            bad++;
        }
        value_error(name + " index a vertex out of range at row " + std::to_string(bad / 3));
    }
    return indices;
}

// A mesh argument: a `(vertices, faces)` pair, shown in signatures as
// `Mesh`, the alias src/trimkit.cpp defines. `read_mesh` checks it
struct MeshArgument {
    nb::handle object;
};

// how a sequence of meshes reads in a signature: `Sequence[Mesh]`
using Meshes = nb::typed<nb::sequence, MeshArgument>;
// and a list of fresh meshes as a result: `list[Mesh]`
using MeshOuts = nb::typed<nb::list, MeshArgument>;

// a mesh as read: its vertices in place, its faces as a library's indices
template <typename Index>
struct Mesh {
    In<double, -1, 3> vertices;
    std::vector<Index> faces;
};

// One mesh of a sequence, named `name` (as "meshes[3]"): any sequence of
// exactly two, whose (n, 3) float64 vertices and (m, 3) int64 faces read as
// every other array does, the faces checked against the vertices
template <typename Index>
Mesh<Index> read_mesh(nb::handle object, const std::string& name) {
    if (!nb::isinstance<nb::sequence>(object) || nb::len(object) != 2) {
        type_error(name + " must be a (vertices, faces) pair, not " + describe(object));
    }
    Mesh<Index> mesh;
    mesh.vertices = read<double, -1, 3>(object[0], name + "[0]");
    mesh.faces = read_indices<Index>(Array<std::int64_t, -1, 3>{object[1]}, name + "[1]", mesh.vertices.shape(0));
    return mesh;
}

// a mesh given as two arguments, `vertices` and `faces`, read the same way
template <typename Index>
Mesh<Index> read_mesh(const Array<double, -1, 3>& vertices, const Array<std::int64_t, -1, 3>& faces) {
    Mesh<Index> mesh;
    mesh.vertices = read(vertices, "vertices");
    mesh.faces = read_indices<Index>(faces, "faces", mesh.vertices.shape(0));
    return mesh;
}

// every mesh of a sequence, read as `read_mesh` reads each, before anything
// is built from any of them
template <typename Index>
std::vector<Mesh<Index>> read_meshes(const Meshes& meshes) {
    std::vector<Mesh<Index>> read(nb::len(meshes));
    for (std::size_t m = 0; m < read.size(); m++) {
        read[m] = read_mesh<Index>(meshes[m], "meshes[" + std::to_string(m) + "]");
    }
    return read;
}

// An array of any numeric dtype, for inputs whose dtype is chosen at runtime:
// C-contiguous, aligned for its own item size, and finite if floating.
inline AnyIn read_any(nb::handle object, const std::string& name) {
    // in place, or its own dtype copied once into a readable layout
    AnyIn array;
    if (!detail::any_readable(object, &array)) {
        const nb::object copy = detail::relaid(object, detail::seen_as_array(object), nullptr, name);
        if (!copy.is_valid() || !detail::any_readable(copy, &array)) {
            type_error(name + " must be a numeric array, not " + describe(object));
        }
    }
    const nb::dlpack::dtype dtype = array.dtype();
    // integers, bools, or exactly float32 or float64: no other floating type
    // (half, bfloat, float8, extended, complex) is checked for its values
    const auto code = static_cast<nb::dlpack::dtype_code>(dtype.code);
    const bool integral = code == nb::dlpack::dtype_code::Int || code == nb::dlpack::dtype_code::UInt ||
                          code == nb::dlpack::dtype_code::Bool;
    if (!integral && dtype != nb::dtype<float>() && dtype != nb::dtype<double>()) {
        type_error(name + " must be a float32, float64 or integer array, not " + describe(object));
    }
    // float32 can't exceed its own range, so finite is its whole rule
    if (dtype == nb::dtype<float>()) {
        check_values(static_cast<const float*>(array.data()), array.size(),
                     static_cast<double>(std::numeric_limits<float>::max()), name);
    } else if (dtype == nb::dtype<double>()) {
        check_values(static_cast<const double*>(array.data()), array.size(), COORDINATE_LIMIT, name);
    }
    return array;
}

// A 1D array of exactly the structured `dtype`, read in place as rows of
// `Record`: C-contiguous and aligned for it, so anything else can't be
// reinterpreted as records it doesn't hold. The rows live as long as `view`,
// which holds the array or the one copy made of it
template <typename Record>
struct Records {
    In<std::uint8_t, -1> view;
    const Record* rows() const { return reinterpret_cast<const Record*>(view.data()); }
    std::size_t size() const { return view.shape(0) / sizeof(Record); }
};

template <typename Record>
Records<Record> read_records(nb::handle object, const nb::object& dtype, const char* dtype_name,
                             const std::string& name) {
    // the dtype's own size is the record's, so rows can't be misread
    const bool exact =
        PyObject_IsInstance(object.ptr(), detail::numpy.ndarray.ptr()) == 1 && object.attr("dtype").equal(dtype) &&
        nb::cast<std::size_t>(dtype.attr("itemsize")) == sizeof(Record) && nb::cast<int>(object.attr("ndim")) == 1;
    if (!exact) {
        type_error(name + " must be a 1D array of dtype " + dtype_name + ", not " + describe(object));
    }
    // in place when C-contiguous and aligned for `Record`, else copied once
    In<std::uint8_t, -1> bytes;
    const bool in_place = nb::cast<bool>(object.attr("flags").attr("c_contiguous")) &&
                          nb::try_cast(object.attr("view")("u1"), bytes, /*convert=*/false) &&
                          detail::aligned(bytes.data(), alignof(Record));
    if (!in_place) {
        const nb::object copy = detail::relaid(object, detail::seen_as_array(object), nullptr, name);
        bytes = detail::view<std::uint8_t, -1>(copy.attr("view")("u1"), name);
    }
    return {bytes};
}

// move a vector to the heap and return its data with a capsule that owns it
template <typename T>
std::pair<T*, nb::capsule> own(std::vector<T>&& values) {
    // an empty vector's data is null, which nanobind would export as a fresh
    // array of its own: one element of capacity keeps every output capsule-owned
    if (values.capacity() == 0) {
        values.reserve(1);
    }
    auto owned = std::make_unique<std::vector<T>>(std::move(values));
    nb::capsule capsule(owned.get(), [](void* pointer) noexcept { delete static_cast<std::vector<T>*>(pointer); });
    // only once the capsule exists does it own the vector, so a failure leaks nothing
    return {owned.release()->data(), std::move(capsule)};
}

// An output element left unwritten when allocated: for an output whose every
// element the binding writes, often from many threads, so it isn't filled
// once first. `to_numpy<T>` hands a vector of these to numpy as T, and
// `elements` writes them as T
template <typename T>
struct Unset {
    T value;
    // NOLINTNEXTLINE(modernize-use-equals-default,cppcoreguidelines-pro-type-member-init): left unwritten on purpose
    Unset() {}
};

template <typename T>
T* elements(std::vector<Unset<T>>& values) {
    static_assert(sizeof(Unset<T>) == sizeof(T) && std::is_standard_layout_v<Unset<T>>);
    return reinterpret_cast<T*>(values.data());  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): the same layout
}

// An output's shape must be exactly what its buffer holds: checked here, once,
// for every array the package returns, so no binding can hand numpy a shape
// past its memory
inline void check_shape(std::initializer_list<std::size_t> shape, std::size_t items) {
    std::size_t product = 1;
    for (const std::size_t extent : shape) {
        product *= extent;
    }
    if (product != items) {
        throw std::logic_error("an output's shape is not its buffer's size");
    }
}

// hand a vector to numpy without copying, optionally viewed `As` a type of
// the same size, e.g. uint64 indices that all fit in int64
template <typename As = void, typename T, typename R = std::conditional_t<std::is_void_v<As>, T, As>>
Out<R> to_numpy(std::vector<T>&& values, std::initializer_list<std::size_t> shape) {
    static_assert(sizeof(R) == sizeof(T));
    check_shape(shape, values.size());
    auto [data, owner] = own(std::move(values));
    return {reinterpret_cast<R*>(data), shape, owner};
}

// A vector as rows, however many it holds: `row` is one row's shape ({3} for
// faces, {2, 3} for pairs of points), and the leading extent is the vector's
// size over a row's, which `to_numpy` checks divides it
template <typename As = void, typename T, typename R = std::conditional_t<std::is_void_v<As>, T, As>>
Out<R> to_numpy_rows(std::vector<T>&& values, std::initializer_list<std::size_t> row) {
    std::size_t width = 1;
    for (const std::size_t extent : row) {
        width *= extent;
    }
    const std::size_t rows = width == 0 ? 0 : values.size() / width;
    const std::size_t* inner = row.begin();
    if (row.size() == 1) {
        return to_numpy<As>(std::move(values), {rows, inner[0]});
    }
    return to_numpy<As>(std::move(values), {rows, inner[0], inner[1]});
}

// records handed to numpy without copying, as a 1D array of the structured
// `dtype`, whose rows are exactly `Record`s
template <typename Record>
nb::object to_numpy_records(std::vector<Record>&& records, const nb::object& dtype) {
    const std::size_t bytes = records.size() * sizeof(Record);
    auto [data, owner] = own(std::move(records));
    const Out<std::uint8_t> raw(reinterpret_cast<std::uint8_t*>(data), {bytes}, owner);
    return nb::cast(raw).attr("view")(dtype);
}

// the same, for bytes whose dtype is only known at runtime
inline AnyOut to_numpy(std::vector<std::uint8_t>&& bytes, std::initializer_list<std::size_t> shape,
                       nb::dlpack::dtype dtype) {
    const std::size_t item_bytes = dtype.bits / 8U;
    check_shape(shape, item_bytes == 0 ? 0 : bytes.size() / item_bytes);
    auto [data, owner] = own(std::move(bytes));
    return {data, shape, owner, {}, dtype};
}

}  // namespace trimkit

// `Array<T, Shape...>` takes any object and shows up as In<T, Shape...>
namespace nanobind::detail {

// An argument nanobind hands over untouched, for the door to read itself:
// `Argument` holds whatever object was passed, and shows in signatures and
// stubs under `Shown::Name`
template <typename Argument, typename Shown>
struct passed_through {
    // cppcheck-suppress unknownMacro ; nanobind's, and cppcheck doesn't read nanobind's headers
    NB_TYPE_CASTER(Argument, Shown::Name)
    bool from_python(handle source, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
        value.object = source;
        return true;
    }
    static handle from_cpp(const Argument& argument, rv_policy /*policy*/, cleanup_list* /*cleanup*/) noexcept {
        return handle(argument.object).inc_ref();
    }
};

// `Array<T, Shape...>` shows up as In<T, Shape...>
template <typename T, nb::ssize_t... Shape>
struct type_caster<trimkit::Array<T, Shape...>>
    : passed_through<trimkit::Array<T, Shape...>, make_caster<trimkit::In<T, Shape...>>> {};

// `RaysArgument` shows up as each array it reads
struct rays_shown {
    // NOLINTNEXTLINE(readability-identifier-naming): the name nanobind reads
    static constexpr auto Name = make_caster<trimkit::In<double, -1, 3>>::Name + const_name(" | ") +
                                 make_caster<trimkit::In<double, 3>>::Name + const_name(" | ") +
                                 make_caster<trimkit::In<float, -1, 3>>::Name + const_name(" | ") +
                                 make_caster<trimkit::In<float, 3>>::Name;
};
template <>
struct type_caster<trimkit::RaysArgument> : passed_through<trimkit::RaysArgument, rays_shown> {};

// `BoxQueriesArgument` shows up as each array shape it reads
struct box_queries_shown {
    // NOLINTNEXTLINE(readability-identifier-naming): the name nanobind reads
    static constexpr auto Name = make_caster<trimkit::In<double, -1>>::Name + const_name(" | ") +
                                 make_caster<trimkit::In<double, -1, -1>>::Name + const_name(" | ") +
                                 make_caster<trimkit::In<double, -1, 2, -1>>::Name;
};
template <>
struct type_caster<trimkit::BoxQueriesArgument> : passed_through<trimkit::BoxQueriesArgument, box_queries_shown> {};

// `MeshArgument` shows up as the `Mesh` alias
struct mesh_shown {
    // NOLINTNEXTLINE(readability-identifier-naming): the name nanobind reads
    static constexpr auto Name = const_name("Mesh");
};
template <>
struct type_caster<trimkit::MeshArgument> : passed_through<trimkit::MeshArgument, mesh_shown> {};

template <typename T>
struct type_caster<trimkit::Number<T>> {
    using Inner = make_caster<T>;
    // cppcheck-suppress unknownMacro ; nanobind's, and cppcheck doesn't read nanobind's headers
    NB_TYPE_CASTER(trimkit::Number<T>,
                   const_name<std::is_integral_v<T>>("int | numpy.integer", "float | numpy.floating"))
    bool from_python(handle source, uint8_t flags, cleanup_list* cleanup) noexcept {
        Inner inner;
        if (!inner.from_python(source, flags, cleanup)) {
            return false;
        }
        value.value = inner.value;
        return true;
    }
    static handle from_cpp(const trimkit::Number<T>& number, rv_policy policy, cleanup_list* cleanup) noexcept {
        return Inner::from_cpp(number.value, policy, cleanup);
    }
};

// `Flag` takes a bool or a numpy.bool_, and only those
template <>
struct type_caster<trimkit::Flag> {
    // cppcheck-suppress unknownMacro ; nanobind's, and cppcheck doesn't read nanobind's headers
    NB_TYPE_CASTER(trimkit::Flag, const_name("bool | numpy.bool_"))
    bool from_python(handle source, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
        PyObject* object = source.ptr();
        const bool plain = object == Py_True || object == Py_False;
        if (!plain && PyObject_IsInstance(object, trimkit::detail::numpy.bool_.ptr()) != 1) {
            PyErr_Clear();
            return false;
        }
        const int truth = PyObject_IsTrue(object);
        PyErr_Clear();
        value.value = truth == 1;
        return truth >= 0;
    }
    static handle from_cpp(const trimkit::Flag& flag, rv_policy /*policy*/, cleanup_list* /*cleanup*/) noexcept {
        return handle(flag.value ? Py_True : Py_False).inc_ref();
    }
};
}  // namespace nanobind::detail

// numpy's objects, kept: a new reference to each, never given back
inline void trimkit::init_numpy() {
    const nb::module_ module = nb::module_::import_("numpy");
    detail::numpy.ndarray = nb::object(module.attr("ndarray")).release();
    detail::numpy.bool_ = nb::object(module.attr("bool_")).release();
    detail::numpy.asarray = nb::object(module.attr("asarray")).release();
    detail::numpy.array = nb::object(module.attr("array")).release();
    detail::numpy.dtype = nb::object(module.attr("dtype")).release();
    detail::numpy.can_cast = nb::object(module.attr("can_cast")).release();
}
