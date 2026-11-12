// A RayScene on a Vulkan device, behind four functions: put a scene's meshes
// on it, put a top level beside them, trace rays against both, and free what
// was put. src/bvh.cpp is the only caller and knows nothing else of Vulkan;
// src/vulkan/vulkan.hpp is the device itself.
//
// One lock, and the rule that keeps it from ever deadlocking: the device is
// taken only by a thread that has released the GIL, and its holder calls
// nothing but Vulkan and the pool, never Python. So no thread waits for the
// device while holding the GIL, and no holder waits for the GIL. Freeing
// takes no lock at all (see `DeviceArrays`).
#pragma once

#include "../common.hpp"
#include "vulkan.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace trimkit::vulkan {

// a device takes one submitter at a time
inline Exclusive& device_lock() {
    static Exclusive lock;
    return lock;
}

// A scene's meshes, or one top level, on the device: uploaded once from the
// trees the CPU traces, freed with the scene or top level they mirror, and
// only in the process that opened the device (a forked child holds the
// parent's handles, which are not its to free).
//
// Freeing takes no lock. It runs wherever the last reference drops, often
// with the GIL held, and a thread holding the GIL must never wait for the
// device: whoever holds the device may be a long query. Vulkan asks for none
// either: destroying a buffer and freeing its memory are synchronized on that
// buffer and that memory alone, which nothing else names once their owner is
// gone (a query keeps its scene and its top level alive until it returns).
struct DeviceArrays {
    Device* device = nullptr;
    Arrays arrays;
    DeviceArrays() = default;
    DeviceArrays(const DeviceArrays&) = delete;
    DeviceArrays& operator=(const DeviceArrays&) = delete;
    ~DeviceArrays() {
        if (device->pid == process_id()) {
            destroy(*device, &arrays);
        }
    }
};

// One mesh as tinybvh holds it, each array as its bytes: `nodes` of 64 bytes
// (Node64), the leaf `slots` (each a face), three `corners` a face (each a
// vertex) and `vertices` of three float64. A mesh without faces has no nodes
struct MeshTree {
    const void* nodes = nullptr;
    std::size_t node_count = 0;
    const std::uint64_t* slots = nullptr;
    std::size_t slot_count = 0;
    const std::uint64_t* corners = nullptr;
    std::size_t corner_count = 0;
    const void* vertices = nullptr;
    std::size_t vertex_count = 0;
};

// a vector's bytes, to upload
template <typename T>
Span bytes_of(const std::vector<T>& values) {
    return {values.data(), values.size() * sizeof(T)};
}

// The f32 image of one triangle for the device's filter (src/vulkan/trace.comp,
// docs/trace.md 4.1, 5.2): the corners, a radius never below any coordinate's
// magnitude, the f64 normal and `nu`, a bound on that normal's distance from
// the exact one: the edges' roundings, the products' and difference's, the
// result's and the f32 conversion's, inflated by 2^-20 for this arithmetic's own
inline Tri32 triangle32(const double* vertices, const std::uint64_t* corners, std::size_t face) {
    constexpr double U64 = 1.1102230246251565e-16;         // 2^-53
    constexpr double U32 = 5.960464477539063e-08;          // 2^-24
    constexpr double INFLATE = 1.0 + 9.5367431640625e-07;  // 1 + 2^-20
    const double* v0 = vertices + (3 * static_cast<std::size_t>(corners[3 * face]));
    const double* v1 = vertices + (3 * static_cast<std::size_t>(corners[(3 * face) + 1]));
    const double* v2 = vertices + (3 * static_cast<std::size_t>(corners[(3 * face) + 2]));
    Tri32 image{};
    double radius = 0.0;
    for (std::size_t axis = 0; axis < 3; axis++) {
        image.p0[axis] = static_cast<float>(v0[axis]);
        image.p1[axis] = static_cast<float>(v1[axis]);
        image.p2[axis] = static_cast<float>(v2[axis]);
        radius = std::max({radius, std::abs(v0[axis]), std::abs(v1[axis]), std::abs(v2[axis])});
    }
    image.radius = std::nextafter(static_cast<float>(radius), std::numeric_limits<float>::infinity());
    image.prim = static_cast<std::uint32_t>(face);
    const double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    const double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    const double n[3] = {(e1[1] * e2[2]) - (e1[2] * e2[1]), (e1[2] * e2[0]) - (e1[0] * e2[2]),
                         (e1[0] * e2[1]) - (e1[1] * e2[0])};
    const double a1[3] = {std::abs(e1[0]), std::abs(e1[1]), std::abs(e1[2])};
    const double a2[3] = {std::abs(e2[0]), std::abs(e2[1]), std::abs(e2[2])};
    const double products[3] = {(a1[1] * a2[2]) + (a1[2] * a2[1]), (a1[2] * a2[0]) + (a1[0] * a2[2]),
                                (a1[0] * a2[1]) + (a1[1] * a2[0])};
    double nu = 0.0;
    for (std::size_t axis = 0; axis < 3; axis++) {
        image.n[axis] = static_cast<float>(n[axis]);
        nu = std::max(nu, (4 * U64 * products[axis]) + (2 * U64 * std::abs(n[axis])) + (U32 * std::abs(n[axis])));
    }
    image.nu = std::nextafter(static_cast<float>(nu * INFLATE), std::numeric_limits<float>::infinity());
    return image;
}

// Every mesh on the process's device, as src/vulkan/trace.comp reads them:
// each kind of array of every mesh in one buffer, with a table of where each
// mesh's begin. Null with the reason when there is no device or it can't.
// Called with the GIL released
inline std::unique_ptr<DeviceArrays> put_scene(Device* device, const MeshTree* meshes, std::size_t mesh_count,
                                               std::string* reason) {
    // `table` holds each mesh's first node, leaf slot, corner and vertex;
    // `totals` their sizes, which the shader indexes with 32 bits
    std::vector<std::uint32_t> table(4 * mesh_count, 0);
    std::size_t totals[4] = {0, 0, 0, 0};
    for (std::size_t m = 0; m < mesh_count; m++) {
        const std::size_t sizes[4] = {meshes[m].node_count, meshes[m].slot_count, meshes[m].corner_count,
                                      meshes[m].vertex_count};
        for (std::size_t k = 0; k < 4; k++) {
            table[(4 * m) + k] = static_cast<std::uint32_t>(totals[k]);
            totals[k] += sizes[k];
        }
    }
    if (*std::max_element(totals, totals + 4) >= INTERIOR) {
        *reason = "the scene has more nodes, faces or vertices than the device indexes";
        return nullptr;
    }
    std::vector<Node64> nodes(totals[0]);
    std::vector<std::uint64_t> corners(totals[2]);
    std::vector<double> vertices(3 * totals[3]);
    std::vector<Node32> node32(totals[0]);
    // a triangle per leaf slot: a slot no leaf reaches stays zero, unread
    std::vector<Tri32> tri32(totals[1]);
    for (std::size_t m = 0; m < mesh_count; m++) {
        const MeshTree& mesh = meshes[m];
        if (mesh.node_count == 0) {
            continue;
        }
        const std::uint32_t* first = table.data() + (4 * m);
        // the nodes as the bytes tinybvh reads them as
        std::memcpy(nodes.data() + first[0], mesh.nodes, mesh.node_count * sizeof(Node64));
        std::copy_n(mesh.corners, mesh.corner_count, corners.data() + first[2]);
        double* points = vertices.data() + (3 * std::size_t{first[3]});
        std::memcpy(points, mesh.vertices, mesh.vertex_count * 3 * sizeof(double));
        const Tree tree{nodes.data() + first[0], mesh.node_count,       mesh.slots,
                        mesh.slot_count,         mesh.corner_count / 3, TRACE_STACK_SIZE};
        const Node32* images = node32.data() + first[0];
        if (!node_images(tree, node32.data() + first[0], reason)) {
            return nullptr;
        }
        // a triangle for each slot of each leaf the walk reached: the slots it checked
        Tri32* triangles = tri32.data() + first[1];
        for (std::size_t n = 0; n < mesh.node_count; n++) {
            const std::uint32_t slots = (images[n].count_word & INTERIOR) != 0 ? 0 : images[n].count_word;
            for (std::uint32_t slot = images[n].left; slot < images[n].left + slots; slot++) {
                triangles[slot] = triangle32(points, mesh.corners, static_cast<std::size_t>(mesh.slots[slot]));
            }
        }
    }
    const Span spans[SCENE_BUFFERS] = {
        bytes_of(nodes), bytes_of(corners), bytes_of(vertices), bytes_of(table), bytes_of(node32), bytes_of(tri32),
    };
    std::unique_ptr<DeviceArrays> arrays = std::make_unique<DeviceArrays>();
    arrays->device = device;
    const Exclusively holding(device_lock());
    if (!upload(*device, spans, SCENE_BUFFERS, &arrays->arrays, reason)) {
        return nullptr;
    }
    return arrays;
}

// A top level on the device beside its scene: its tree as tinybvh holds it
// (`nodes` as bytes, the leaf `slots` each an instance), and each instance's
// inverse transform (16 float64) and mesh. Null with the reason when it
// can't. Called with the GIL released
inline std::unique_ptr<DeviceArrays> put_top(Device* device, const void* tree_nodes, std::size_t node_count,
                                             const std::uint64_t* slots, std::size_t slot_count,
                                             const std::vector<double>& inverses,
                                             const std::vector<std::uint32_t>& instance_meshes, std::string* reason) {
    std::vector<Node64> nodes(node_count);
    std::memcpy(nodes.data(), tree_nodes, node_count * sizeof(Node64));
    std::vector<Node32> node32(node_count);
    const Tree tree{nodes.data(), node_count, slots, slot_count, instance_meshes.size(), TRACE_TOP_STACK_SIZE};
    if (!node_images(tree, node32.data(), reason)) {
        return nullptr;
    }
    const Span spans[TOP_BUFFERS] = {
        bytes_of(nodes),  {slots, slot_count * sizeof(std::uint64_t)}, bytes_of(inverses), bytes_of(instance_meshes),
        bytes_of(node32),
    };
    std::unique_ptr<DeviceArrays> arrays = std::make_unique<DeviceArrays>();
    arrays->device = device;
    const Exclusively holding(device_lock());
    if (!upload(*device, spans, TOP_BUFFERS, &arrays->arrays, reason)) {
        return nullptr;
    }
    return arrays;
}

// `count` values copied as they are, and whether every one is a value a ray
// may hold: finite and within the bound every float64 input is held to
inline bool copy_checked(const double* values, std::size_t count, double* out) {
    std::size_t refused = 0;
    for (std::size_t i = 0; i < count; i++) {
        out[i] = values[i];
        refused += !(std::abs(values[i]) <= COORDINATE_LIMIT) ? 1U : 0U;
    }
    return refused == 0;
}

// `count` float32 values copied as they are: whether every one may be held,
// and in `*plain` whether every one is zero or normal (a device may read a
// float32 denormal as zero, so a chunk holding one goes widened instead)
inline bool copy_checked(const float* values, std::size_t count, float* out, bool* plain) {
    std::size_t refused = 0;
    std::size_t denormal = 0;
    for (std::size_t i = 0; i < count; i++) {
        const float size = std::abs(values[i]);
        out[i] = values[i];
        refused += !(static_cast<double>(size) <= COORDINATE_LIMIT) ? 1U : 0U;
        denormal += size != 0.0F && size < std::numeric_limits<float>::min() ? 1U : 0U;
    }
    *plain = denormal == 0;
    return refused == 0;
}

// `count` of a query's origins or directions from row `begin`, written into
// a chunk, and how (STRIDE_*): one row for every ray; a float32 array as
// float32, which the device widens; a float64 array as float64. The array's
// dtype decides, and nothing about one chunk decides another's. False for a
// value no ray may hold: the one look these values get on their way to a
// device. One pass on the calling thread: the device, not this copy, is what
// a query waits for
inline bool fill_rows(const Rows<3>& rays, void* chunk, std::size_t begin, std::size_t count, std::uint32_t* stride) {
    double* wide = static_cast<double*>(chunk);
    if (rays.one) {
        double row[3];
        rays.read(0, row);
        *stride = STRIDE_ONE;
        return copy_checked(row, 3, wide);
    }
    std::atomic<bool> taken{true};
    if (rays.wide != nullptr) {
        *stride = STRIDE_WIDE;
        const double* given = rays.wide + (3 * begin);
        in_blocks_without_gil(count, [&](std::size_t from, std::size_t to) {
            if (!copy_checked(given + (3 * from), 3 * (to - from), wide + (3 * from))) {
                taken.store(false, std::memory_order_relaxed);
            }
        });
        return taken.load();
    }
    const float* narrow = rays.narrow + (3 * begin);
    std::atomic<bool> every_plain{true};
    *stride = STRIDE_NARROW;
    in_blocks_without_gil(count, [&](std::size_t from, std::size_t to) {
        bool block_plain = true;
        if (!copy_checked(narrow + (3 * from), 3 * (to - from), static_cast<float*>(chunk) + (3 * from),
                          &block_plain)) {
            taken.store(false, std::memory_order_relaxed);
        }
        if (!block_plain) {
            every_plain.store(false, std::memory_order_relaxed);
        }
    });
    if (!taken.load()) {
        return false;
    }
    const bool plain = every_plain.load();
    if (!plain) {
        // a float32 widens to exactly itself, and was just checked
        *stride = STRIDE_WIDE;
        for (std::size_t i = 0; i < 3 * count; i++) {
            wide[i] = static_cast<double>(narrow[i]);
        }
    }
    return true;
}

// One query traced on the device over `count` rays: the scene, and either
// its `top` level or, with none, its one `mesh` directly; whether it only
// asks if anything is hit, else up to `most` crossings a ray; `extent` the
// scene's size. `consume(answers)` reads each chunk's answers, in ray order.
// False when the device failed (with the reason) or a ray holds a value no
// ray may (no reason: the caller's own check raises it), and nothing is left
// running on the device either way. Called with the GIL released
template <typename Consume>
bool trace(const DeviceArrays& scene, const DeviceArrays* top, std::uint32_t mesh, double extent, bool occlusion,
           std::size_t most, const Rows<3>& origins, const Rows<3>& directions, std::size_t count,
           const Consume& consume, std::string* reason) {
    Query query;
    query.scene = &scene.arrays;
    query.top = top != nullptr ? &top->arrays : nullptr;
    query.mesh = mesh;
    query.occlusion = occlusion;
    query.most = most < UNBOUNDED ? static_cast<std::uint32_t>(most) : UNBOUNDED;
    query.extent = extent;
    const Exclusively holding(device_lock());
    return run_query(
        *scene.device, query, count,
        [&](void* chunk_origins, void* chunk_directions, std::size_t begin, std::size_t rays, std::uint32_t* strides) {
            return fill_rows(origins, chunk_origins, begin, rays, &strides[0]) &&
                   fill_rows(directions, chunk_directions, begin, rays, &strides[1]);
        },
        consume, reason);
}

}  // namespace trimkit::vulkan
