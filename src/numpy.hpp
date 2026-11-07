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
// lossless. Nothing is ever converted: any other dtype, a list or anything
// else is a TypeError (or for values, a ValueError) naming the argument.
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
// it, and `read` checks it without converting: no copy, ever, and a TypeError
// that names the argument instead of an overload dump.
template <typename T, nb::ssize_t... Shape>
struct Array {
    nb::handle object;
    bool is_none() const { return object.is_none(); }
};

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
inline std::string text(nb::handle object) {
    const nb::object converted = nb::steal(PyObject_Str(object.ptr()));
    if (!converted.is_valid()) {
        PyErr_Clear();
        return "?";
    }
    const char* utf8 = PyUnicode_AsUTF8AndSize(converted.ptr(), nullptr);  // stable ABI since 3.10
    if (utf8 == nullptr) {
        PyErr_Clear();
        return "?";
    }
    return utf8;
}

// a short dtype name for messages: "float64", "int64", ...
template <typename T>
constexpr const char* dtype_name() {
    if constexpr (std::is_same_v<T, double>) {
        return "float64";
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

// Refuse any value whose magnitude isn't at most `limit`: one comparison,
// false for NaN and infinity too, counted without branching so the scan
// vectorizes, and run without the GIL.
template <typename T>
void check_bounded(const T* values, std::size_t count, T limit, const std::string& name, const char* bound) {
    std::size_t violations = 0;
    {
        const nb::gil_scoped_release release;
        for (std::size_t i = 0; i < count; i++) {
            violations += !(std::abs(values[i]) <= limit) ? 1U : 0U;
        }
    }
    if (violations != 0) {
        value_error(name + " must be finite" + bound);
    }
}

// float64 values, refused unless within COORDINATE_LIMIT
inline void check_coordinates(const double* values, std::size_t count, const std::string& name) {
    check_bounded(values, count, COORDINATE_LIMIT, name, (std::string(" and within ") + COORDINATE_LIMIT_TEXT).c_str());
}

}  // namespace detail

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

namespace detail {

// A numpy array of exactly `dtype` (any dtype for null) that can't be read
// in place, as one fresh C-contiguous copy: what `numpy.ascontiguousarray`
// gives, and aligned as every fresh numpy buffer is, where a view at an odd
// byte offset would make every read of it undefined (a packed structured
// dtype counts as aligned to numpy there, so this always copies). Anything
// else, another dtype or a list, gets nothing: it stays refused
inline nb::object relaid(nb::handle object, const char* dtype) {
    const nb::module_ numpy = nb::module_::import_("numpy");
    const bool exact = PyObject_IsInstance(object.ptr(), numpy.attr("ndarray").ptr()) == 1 &&
                       (dtype == nullptr || object.attr("dtype").equal(numpy.attr("dtype")(dtype)));
    if (!exact) {
        return {};
    }
    return numpy.attr("array")(object, nb::arg("copy") = true, nb::arg("order") = "C");
}

// view an object as exactly an In<T, Shape...>, aligned for T: in place, or
// copied once when only its layout is wrong. The view holds what it reads
template <typename T, nb::ssize_t... Shape>
In<T, Shape...> view(nb::handle object, const std::string& name) {
    In<T, Shape...> array;
    bool exact = nb::try_cast(object, array, /*convert=*/false) && aligned(array.data(), alignof(T));
    if (!exact) {
        const nb::object copy = relaid(object, dtype_name<T>());
        exact = copy.is_valid() && nb::try_cast(copy, array, /*convert=*/false) && aligned(array.data(), alignof(T));
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
        detail::check_coordinates(array.data(), array.size(), name);
    }
    return array;
}

template <typename T, nb::ssize_t... Shape>
In<T, Shape...> read(const Array<T, Shape...>& argument, const std::string& name) {
    return read<T, Shape...>(argument.object, name);
}

// Whether an array argument holds exactly `stored`, byte for byte, checked
// for dtype, shape and layout as `read` checks it but not for its values:
// `stored` came from `read`, so equal bytes are valid ones. For comparing
// with what was read before (a scene's remembered instances), not computing
template <typename T, nb::ssize_t... Shape>
bool same_bytes(const Array<T, Shape...>& argument, const std::vector<T>& stored, const std::string& name) {
    const In<T, Shape...> array = detail::view<T, Shape...>(argument.object, name);
    return array.size() == stored.size() && std::memcmp(array.data(), stored.data(), stored.size() * sizeof(T)) == 0;
}

// A bool array argument, as its bytes: a byte other than 0 or 1 in a numpy
// bool array (a uint8 array viewed as bool can hold one) is undefined as a
// C++ bool, so its values are read as uint8 and tested `!= 0`
inline In<std::uint8_t, -1> read(const Array<bool, -1>& argument, const std::string& name) {
    const nb::handle object = argument.object;
    const nb::module_ numpy = nb::module_::import_("numpy");
    if (PyObject_IsInstance(object.ptr(), numpy.attr("ndarray").ptr()) != 1 ||
        !object.attr("dtype").equal(numpy.attr("dtype")("bool")) || nb::cast<int>(object.attr("ndim")) != 1) {
        type_error(name + " must be a bool array of shape (*,), not " + describe(object));
    }
    return detail::view<std::uint8_t, -1>(object.attr("view")("u1"), name);
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
            violations += static_cast<std::uint64_t>(values[i]) >= vertex_count ? 1U : 0U;
            indices[i] = static_cast<Index>(values[i]);
        }
    }
    if (violations != 0) {
        // the rare refusal pays for finding the row to name
        std::size_t bad = 0;
        while (static_cast<std::uint64_t>(values[bad]) < vertex_count) {
            bad++;
        }
        value_error(name + " index a vertex out of range at row " + std::to_string(bad / 3));
    }
    return indices;
}

// A mesh argument: a `(vertices, faces)` tuple, shown in signatures as
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

// One mesh of a sequence, named `name` (as "meshes[3]"): exactly a 2-tuple,
// whose (n, 3) float64 vertices and (m, 3) int64 faces read as every other
// array does, the faces checked against the vertices
template <typename Index>
Mesh<Index> read_mesh(nb::handle object, const std::string& name) {
    if (!nb::isinstance<nb::tuple>(object) || nb::len(object) != 2) {
        type_error(name + " must be a (vertices, faces) tuple, not " + describe(object));
    }
    Mesh<Index> mesh;
    mesh.vertices = read<double, -1, 3>(object[0], name + "[0]");
    mesh.faces = read_indices<Index>(Array<std::int64_t, -1, 3>{object[1]}, name + "[1]", mesh.vertices.shape(0));
    return mesh;
}

// An array of any numeric dtype, for inputs whose dtype is chosen at runtime:
// C-contiguous, aligned for its own item size, and finite if floating.
namespace detail {

// whether `candidate` is an array of any dtype readable in place, viewed into `array`
inline bool any_readable(nb::handle candidate, AnyIn* array) {
    if (!nb::try_cast(candidate, *array, /*convert=*/false)) {
        return false;
    }
    const std::size_t item_size = (array->dtype().bits + 7U) / 8U;
    return item_size == 0 || aligned(array->data(), item_size);
}

}  // namespace detail

inline AnyIn read_any(nb::handle object, const std::string& name) {
    // in place, or its own dtype copied once into a readable layout
    AnyIn array;
    if (!detail::any_readable(object, &array)) {
        const nb::object copy = detail::relaid(object, nullptr);
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
        detail::check_bounded(static_cast<const float*>(array.data()), array.size(), std::numeric_limits<float>::max(),
                              name, "");
    } else if (dtype == nb::dtype<double>()) {
        detail::check_coordinates(static_cast<const double*>(array.data()), array.size(), name);
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
    const bool exact = PyObject_IsInstance(object.ptr(), nb::module_::import_("numpy").attr("ndarray").ptr()) == 1 &&
                       object.attr("dtype").equal(dtype) &&
                       nb::cast<std::size_t>(dtype.attr("itemsize")) == sizeof(Record) &&
                       nb::cast<int>(object.attr("ndim")) == 1;
    if (!exact) {
        type_error(name + " must be a 1D array of dtype " + dtype_name + ", not " + describe(object));
    }
    // in place when C-contiguous and aligned for `Record`, else copied once
    In<std::uint8_t, -1> bytes;
    const bool in_place = nb::cast<bool>(object.attr("flags").attr("c_contiguous")) &&
                          nb::try_cast(object.attr("view")("u1"), bytes, /*convert=*/false) &&
                          detail::aligned(bytes.data(), alignof(Record));
    if (!in_place) {
        bytes = detail::view<std::uint8_t, -1>(detail::relaid(object, nullptr).attr("view")("u1"), name);
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

// hand a vector to numpy without copying, optionally viewed `As` a type of
// the same size, e.g. uint64 indices that all fit in int64
template <typename As = void, typename T, typename R = std::conditional_t<std::is_void_v<As>, T, As>>
Out<R> to_numpy(std::vector<T>&& values, std::initializer_list<std::size_t> shape) {
    static_assert(sizeof(R) == sizeof(T));
    auto [data, owner] = own(std::move(values));
    return {reinterpret_cast<R*>(data), shape, owner};
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
    auto [data, owner] = own(std::move(bytes));
    return {data, shape, owner, {}, dtype};
}

}  // namespace trimkit

// `Array<T, Shape...>` takes any object and shows up as In<T, Shape...>
namespace nanobind::detail {
template <typename T, nb::ssize_t... Shape>
struct type_caster<trimkit::Array<T, Shape...>> {
    using Argument = trimkit::Array<T, Shape...>;
    using Shown = make_caster<trimkit::In<T, Shape...>>;
    // cppcheck-suppress unknownMacro ; nanobind's, and cppcheck doesn't read nanobind's headers
    NB_TYPE_CASTER(Argument, Shown::Name)
    bool from_python(handle source, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
        value.object = source;
        return true;
    }
    static handle from_cpp(const Argument& array, rv_policy /*policy*/, cleanup_list* /*cleanup*/) noexcept {
        return array.object.inc_ref();
    }
};

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
        if (object == Py_True || object == Py_False) {
            value.value = object == Py_True;
            return true;
        }
        PyObject* numpy = PyImport_ImportModule("numpy");
        PyObject* scalar = numpy != nullptr ? PyObject_GetAttrString(numpy, "bool_") : nullptr;
        const int truth = scalar != nullptr && PyObject_IsInstance(object, scalar) == 1 ? PyObject_IsTrue(object) : -1;
        Py_XDECREF(scalar);
        Py_XDECREF(numpy);
        PyErr_Clear();
        value.value = truth == 1;
        return truth >= 0;
    }
    static handle from_cpp(const trimkit::Flag& flag, rv_policy /*policy*/, cleanup_list* /*cleanup*/) noexcept {
        return handle(flag.value ? Py_True : Py_False).inc_ref();
    }
};

// `MeshArgument` takes any object and shows up as the `Mesh` alias
template <>
struct type_caster<trimkit::MeshArgument> {
    // cppcheck-suppress unknownMacro ; nanobind's, and cppcheck doesn't read nanobind's headers
    NB_TYPE_CASTER(trimkit::MeshArgument, const_name("Mesh"))
    bool from_python(handle source, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
        value.object = source;
        return true;
    }
    static handle from_cpp(const trimkit::MeshArgument& mesh, rv_policy /*policy*/,
                           cleanup_list* /*cleanup*/) noexcept {
        return mesh.object.inc_ref();
    }
};
}  // namespace nanobind::detail
