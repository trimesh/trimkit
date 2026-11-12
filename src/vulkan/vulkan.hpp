// Ray queries on a Vulkan device: src/vulkan/trace.comp (tinybvh's traversals,
// filtered in f32 and exact in f64, so the device answers byte for byte what
// the CPU answers) over a RayScene's meshes, and src/vulkan/layout.comp laying
// its answers out. The loader is opened at run time by name, never
// linked, so every wheel works on a machine without Vulkan: anything missing
// or failing is a reason string the caller warns with and answers on the CPU.
// One device per process, opened on first use and kept; the caller serializes
// its use (a queue is not thread safe).
// Rays go through in chunks of a quarter million, two in flight, each one
// submission: the host fills a chunk's rays into mapped host memory, submits,
// and fills the next while the device traces this one. Host memory is written
// only before the submission that reads it. The kernel reads the rays where
// the host put them (as float64 or float32 rows, or one row for every ray),
// counts each ray's crossings and appends every crossing to the chunk's log
// on the device; a scan turns the counts into offsets; and two more steps lay
// the log out in ray order into mapped host columns, sized by the device from
// its own total. The host reads that total back: when the log was too small
// for it, nothing was laid out, and the chunk runs again with room. An
// occlusion query is the kernel alone. Python-free: src/vulkan/scene.hpp is
// the only user, and what src/bvh.cpp calls. Vulkan 1.0 core, no extensions.
#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

#include "spirv.hpp"

#if defined(_WIN32)
#include "../windows.hpp"

#include <process.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace trimkit::vulkan {

// -- the loader ---------------------------------------------------------------

// every entry point used, loaded by name through the loader
#define TRIMKIT_VULKAN_INSTANCE_FUNCTIONS(X)    \
    X(vkEnumeratePhysicalDevices)               \
    X(vkGetPhysicalDeviceProperties)            \
    X(vkGetPhysicalDeviceFeatures)              \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties)      \
    X(vkCreateDevice)                           \
    X(vkGetDeviceProcAddr)
#define TRIMKIT_VULKAN_DEVICE_FUNCTIONS(X) \
    X(vkGetDeviceQueue)                    \
    X(vkCreateBuffer)                      \
    X(vkDestroyBuffer)                     \
    X(vkGetBufferMemoryRequirements)       \
    X(vkAllocateMemory)                    \
    X(vkFreeMemory)                        \
    X(vkBindBufferMemory)                  \
    X(vkMapMemory)                         \
    X(vkCreateShaderModule)                \
    X(vkDestroyShaderModule)               \
    X(vkCreateDescriptorSetLayout)         \
    X(vkCreatePipelineLayout)              \
    X(vkCreateComputePipelines)            \
    X(vkCreateDescriptorPool)              \
    X(vkAllocateDescriptorSets)            \
    X(vkUpdateDescriptorSets)              \
    X(vkCreateCommandPool)                 \
    X(vkAllocateCommandBuffers)            \
    X(vkResetCommandBuffer)                \
    X(vkBeginCommandBuffer)                \
    X(vkEndCommandBuffer)                  \
    X(vkCmdCopyBuffer)                     \
    X(vkCmdPipelineBarrier)                \
    X(vkCmdBindPipeline)                   \
    X(vkCmdBindDescriptorSets)             \
    X(vkCmdPushConstants)                  \
    X(vkCmdDispatch)                       \
    X(vkCmdDispatchIndirect)               \
    X(vkCmdFillBuffer)                     \
    X(vkCreateFence)                       \
    X(vkResetFences)                       \
    X(vkQueueSubmit)                       \
    X(vkWaitForFences)

struct Functions {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
#define TRIMKIT_VULKAN_MEMBER(name) PFN_##name name = nullptr;
    TRIMKIT_VULKAN_INSTANCE_FUNCTIONS(TRIMKIT_VULKAN_MEMBER)
    TRIMKIT_VULKAN_DEVICE_FUNCTIONS(TRIMKIT_VULKAN_MEMBER)
#undef TRIMKIT_VULKAN_MEMBER
};

// -- what the shaders read ----------------------------------------------------

// src/vulkan/trace.comp's bindings, in its order: a scene's arrays (tinybvh's
// nodes, the corners, the vertices, each mesh's first node, leaf slot, corner
// and vertex in those, and the f32 images of the nodes and of the triangle
// in each leaf slot), a top level's (nodes, primitives, inverse transforms,
// each instance's mesh, the nodes' f32 image), and a chunk's (origins,
// directions, the crossings' words and distances, counts, offsets)
constexpr std::uint32_t SCENE_BUFFERS = 6;
constexpr std::uint32_t TOP_BUFFERS = 5;
constexpr std::uint32_t CHUNK_BUFFERS = 6;
constexpr std::uint32_t TRACE_BINDINGS = SCENE_BUFFERS + TOP_BUFFERS + CHUNK_BUFFERS;
// src/vulkan/layout.comp: counts, offsets, block sums, the header, the log
// (words and distances), the order, and the three columns the host reads
constexpr std::uint32_t LAYOUT_BINDINGS = 10;
constexpr std::uint32_t SETS = 2;  // descriptor sets per chunk in flight: one per shader

// A node as the shader's f32 filter reads it (docs/trace.md 3): the f64 box
// rounded to nearest, and for an interior node bit 31 with a mask of which of
// its two children's six f64 coordinates are exactly equal (the filter's
// tie-break); for a leaf, its count
struct Node32 {
    float lo[3];
    std::uint32_t left;
    float hi[3];
    std::uint32_t count_word;
};
constexpr std::uint32_t INTERIOR = 0x80000000U;

// The triangle in a leaf slot as the filter reads it (docs/trace.md 4, 5):
// corners in tinybvh's order as f32, the largest coordinate magnitude rounded
// up, the face, the f64 normal as f32 and a bound on how far that is from
// the exact normal
struct Tri32 {
    float p0[3];
    float radius;
    float p1[3];
    std::uint32_t prim;
    float p2[3];
    float nu;
    float n[3];
    float spare;
};
static_assert(sizeof(Node32) == 32 && sizeof(Tri32) == 64);

// A node as tinybvh's BVH<double, uint64_t> lays it out and the shader reads
// it: a leaf has a count, of the slots from `left_first`; an interior node's
// children are `left_first` and the node after
struct Node64 {
    double lo[3];
    std::uint64_t left_first;
    double hi[3];
    std::uint64_t count;
};
static_assert(sizeof(Node64) == 64);

// A tree to put on the device: its nodes, the slots its leaves index (which
// hold faces, or a top level's instances), what every slot a leaf reaches
// must be under, and the deepest leaf the shader can reach
struct Tree {
    const Node64* nodes;
    std::size_t node_count;
    const std::uint64_t* slots;
    std::size_t slot_count;
    std::uint64_t limit;
    std::uint32_t depth;
};

// The f32 images of a tree's nodes (docs/trace.md 3: each box rounded to
// nearest, and for an interior node which of its children's coordinates tie
// exactly, which decides their order), written by walking the tree from its
// root as the shader will. Nothing is followed before it is checked: a child
// is a node, a leaf's slots are slots, each of those slots holds a value
// under the limit, no leaf is deeper than the shader's stack, and no more
// nodes are reached than the tree has, so the walk ends whatever the nodes
// say. A node the walk doesn't reach keeps a zeroed image, a leaf of no
// slots: so whoever reads leaf images reads only slots checked here, and a
// slot no leaf reaches (a split build leaves stale ones) is never read at
// all. False with the reason for a tree that fails any of it
inline bool node_images(const Tree& tree, Node32* node32, std::string* reason) {
    struct Pending {
        std::uint32_t node;
        std::uint32_t depth;
    };
    Pending stack[TRACE_STACK_SIZE + 2];
    std::size_t pending = 0;
    std::size_t reached = 0;
    bool sound = tree.node_count > 0 && tree.node_count < INTERIOR && tree.slot_count < INTERIOR;
    stack[pending++] = {0, 0};
    while (sound && pending > 0) {
        const Pending at = stack[--pending];
        const Node64& node = tree.nodes[at.node];
        Node32& image = node32[at.node];
        for (std::size_t axis = 0; axis < 3; axis++) {
            image.lo[axis] = static_cast<float>(node.lo[axis]);
            image.hi[axis] = static_cast<float>(node.hi[axis]);
        }
        image.left = static_cast<std::uint32_t>(node.left_first);
        sound = ++reached <= tree.node_count;
        if (node.count > 0) {
            sound = sound && node.left_first <= tree.slot_count && node.count <= tree.slot_count - node.left_first;
            for (std::uint64_t slot = node.left_first; sound && slot < node.left_first + node.count; slot++) {
                sound = tree.slots[slot] < tree.limit;
            }
            // an unsound leaf stays the leaf of no slots
            image.count_word = sound ? static_cast<std::uint32_t>(node.count) : 0;
            continue;
        }
        if (at.depth == tree.depth) {
            *reason = "a tree is deeper than the " + std::to_string(tree.depth) + " levels the device traverses";
            return false;
        }
        sound = sound && node.left_first < tree.node_count - 1;
        if (!sound) {
            break;
        }
        const Node64& a = tree.nodes[node.left_first];
        const Node64& b = tree.nodes[node.left_first + 1];
        std::uint32_t ties = 0;
        for (std::uint32_t axis = 0; axis < 3; axis++) {
            ties |= (a.lo[axis] == b.lo[axis] ? 1U : 0U) << axis;
            ties |= (a.hi[axis] == b.hi[axis] ? 1U : 0U) << (3 + axis);
        }
        image.count_word = INTERIOR | ties;
        stack[pending++] = {image.left + 1, at.depth + 1};
        stack[pending++] = {image.left, at.depth + 1};
    }
    if (!sound) {
        *reason = "a tree is not one the device can traverse";
    }
    return sound;
}
// the most rays in one chunk: small enough that a million rays are four
// chunks, so the host's copy of one and reading of another overlap the
// device's tracing (measured against 2^16 to 2^22)
constexpr std::size_t CHUNK = std::size_t{1} << 18U;
// crossings per ray a chunk's log holds when a device opens: a chunk that
// logs more runs again with a log that holds them, kept from then on
constexpr std::size_t LOG_PER_RAY = 4;
// a chunk's header on the device: the crossings logged, then the three group
// counts of the dispatches that lay them out
constexpr VkDeviceSize HEADER_BYTES = 4 * sizeof(std::uint32_t);
// counts per block of the scan, as src/vulkan/layout.comp groups them
constexpr std::uint32_t SCAN_BLOCK = LAYOUT_PER_LANE * LAYOUT_GROUP;
// chunks in flight at once: the host fills and reads one while the device
// traces the other (measured 11 to 23% faster than one, at 16M rays)
constexpr std::size_t IN_FLIGHT = 2;
constexpr std::uint32_t UNBOUNDED = 0xffffffffU;

// How a chunk holds an array of rays' origins or directions, as the words from
// one ray's row to the next: one float64 row for every ray, a float32 row
// each, or a float64 row each. The host's fill decides it per array and per
// chunk, and src/vulkan/trace.comp reads through it
constexpr std::uint32_t STRIDE_ONE = 0;
constexpr std::uint32_t STRIDE_NARROW = 3;
constexpr std::uint32_t STRIDE_WIDE = 6;

struct PushConstants {
    std::uint32_t ray_count;
    std::uint32_t top;  // 1: through the top level, 0: one mesh
    std::uint32_t mesh;
    std::uint32_t occlusion;
    std::uint32_t most;              // crossings per ray at most
    std::uint32_t origin_stride;     // words between rays' rows: STRIDE_ONE, STRIDE_NARROW or STRIDE_WIDE
    std::uint32_t direction_stride;  // the same, of the directions
    std::uint32_t capacity;          // crossings the chunk's log holds
    double extent;                   // the scene's size
};
struct LayoutConstants {
    std::uint32_t count;
    std::uint32_t phase;     // 0 to 2 scan the counts into offsets, 3 orders the log, 4 writes the columns
    std::uint32_t capacity;  // crossings the chunk's log holds
};

// -- buffers, a chunk's and the device ----------------------------------------

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

struct Pipeline {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

// What one chunk in flight needs, whichever chunk it holds. How the rays
// cross was measured three ways (2026-11-12, RTX 3090, a million rays): the
// kernel reading host memory as here, 4.4 ms for 24 MB; the submission
// copying them into device memory first, the same; the host writing straight
// into device memory it can see, 16 ms. About 0.17 ms a megabyte either of
// the first two ways, whatever the rays hit, so this is the simplest. Mapped host
// memory the CPU caches for what the host writes or reads: the rays, read by
// the kernel over the bus as it traces, the occlusion words, the header as read back, and the
// three columns the layout writes. The device's own memory for what only
// shaders touch: the counts, their scan, the header, the log and its order.
// A crossing in the log is four words (its ray, its place along the ray, its
// instance and face) and a distance; in the host's columns its ray, its
// instance and face, and its distance. The ray buffers are a chunk's size;
// the log, its order and the columns hold `capacity` crossings
struct Chunk {
    Buffer origins;
    Buffer directions;
    Buffer occlusion;
    Buffer counts;
    Buffer offsets;
    Buffer blocks;
    Buffer header;
    Buffer readback;
    Buffer log_words;
    Buffer log_distances;
    Buffer order;
    Buffer out_rays;
    Buffer out_words;
    Buffer out_distances;
    std::size_t capacity = 0;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSet trace_set = VK_NULL_HANDLE;
    VkDescriptorSet layout_set = VK_NULL_HANDLE;
    // how the chunk it holds has its origins and directions: STRIDE_*
    std::uint32_t strides[2] = {STRIDE_WIDE, STRIDE_WIDE};
};

struct Device {
    Functions fn;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkPhysicalDeviceMemoryProperties memory{};
    VkDeviceSize range = 0;  // the largest storage buffer the device binds (lavapipe: 128 MB)
    Pipeline trace;
    Pipeline layout;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    // the commands and fence uploads use, and the chunks in flight
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Chunk chunks[IN_FLIGHT];
    int pid = 0;
};

inline int process_id() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

// whether a Vulkan call succeeded: if not, the reason names it and its result
inline bool ok(VkResult result, const char* what, std::string* reason) {
    if (result != VK_SUCCESS) {
        *reason = std::string(what) + " failed (VkResult " + std::to_string(static_cast<int>(result)) + ")";
    }
    return result == VK_SUCCESS;
}

// the loader's one exported entry point, from whichever library name this
// platform gives it, or null
inline PFN_vkGetInstanceProcAddr open_loader() {
#if defined(_WIN32)
    void* const library = LoadLibraryA("vulkan-1.dll");
    if (library == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(library, "vkGetInstanceProcAddr"));
#else
#if defined(__APPLE__)
    const char* names[] = {"libvulkan.1.dylib", "libvulkan.dylib"};
#else
    const char* names[] = {"libvulkan.so.1", "libvulkan.so"};
#endif
    for (const char* name : names) {
        void* library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (library != nullptr) {
            return reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
        }
    }
    return nullptr;
#endif
}

// The shader reads each 64-bit word it is handed (tinybvh's doubles and
// indices, the rays) as two 32-bit words, low word first: the host's bytes
// must already lie that way
inline bool little_endian() {
    const std::uint32_t one = 1;
    unsigned char first = 0;
    std::memcpy(&first, &one, 1);
    return first == 1;
}

// the index of a memory type among `type_bits` with every `required` flag,
// preferring one with the `preferred` flags too, or -1
inline int find_memory(const VkPhysicalDeviceMemoryProperties& memory, std::uint32_t type_bits,
                       VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
    for (const VkMemoryPropertyFlags wanted : {required | preferred, required}) {
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; i++) {
            const bool allowed = (type_bits & (std::uint32_t{1} << i)) != 0;
            if (allowed && (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
                return static_cast<int>(i);
            }
        }
    }
    return -1;
}

inline void destroy(const Device& device, Buffer* buffer) {
    if (buffer->buffer != VK_NULL_HANDLE) {
        device.fn.vkDestroyBuffer(device.device, buffer->buffer, nullptr);
    }
    if (buffer->memory != VK_NULL_HANDLE) {
        device.fn.vkFreeMemory(device.device, buffer->memory, nullptr);
    }
    *buffer = Buffer{};
}

// a buffer of at least `size` bytes (a storage buffer can't be empty) in
// memory with `required` flags, mapped when host visible
inline bool make_buffer(const Device& device, VkDeviceSize size, VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred, Buffer* out,
                        std::string* reason) {
    // past the device's range a bound buffer is undefined, not an error
    if (size > device.range) {
        *reason = "a buffer of " + std::to_string(size) + " bytes exceeds the device's storage buffer range";
        return false;
    }
    const Functions& fn = device.fn;
    Buffer buffer;
    buffer.size = std::max<VkDeviceSize>(size, 16);
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = buffer.size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bool made = ok(fn.vkCreateBuffer(device.device, &info, nullptr, &buffer.buffer), "vkCreateBuffer", reason);
    if (made) {
        VkMemoryRequirements requirements{};
        fn.vkGetBufferMemoryRequirements(device.device, buffer.buffer, &requirements);
        const int type = find_memory(device.memory, requirements.memoryTypeBits, required, preferred);
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = static_cast<std::uint32_t>(type);
        made = type >= 0;
        if (!made) {
            *reason = "no device memory of the kind needed";
        }
    }
    const bool mapped = (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    made = made &&
           ok(fn.vkAllocateMemory(device.device, &allocate, nullptr, &buffer.memory), "vkAllocateMemory", reason) &&
           ok(fn.vkBindBufferMemory(device.device, buffer.buffer, buffer.memory, 0), "vkBindBufferMemory", reason) &&
           (!mapped || ok(fn.vkMapMemory(device.device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped),
                          "vkMapMemory", reason));
    if (!made) {
        destroy(device, &buffer);
        return false;
    }
    *out = buffer;
    return true;
}

constexpr VkMemoryPropertyFlags HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkBufferUsageFlags TRANSFER = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
// what any buffer a shader touches may also be: copied, filled, and read for a dispatch's size
constexpr VkBufferUsageFlags STORAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | TRANSFER;

inline bool begin_commands(const Device& device, VkCommandBuffer command, std::string* reason) {
    if (!ok(device.fn.vkResetCommandBuffer(command, 0), "vkResetCommandBuffer", reason)) {
        return false;
    }
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return ok(device.fn.vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer", reason);
}

// end the recorded commands and submit them, to signal `fence`
inline bool submit_commands(const Device& device, VkCommandBuffer command, VkFence fence, std::string* reason) {
    if (!ok(device.fn.vkEndCommandBuffer(command), "vkEndCommandBuffer", reason)) {
        return false;
    }
    if (!ok(device.fn.vkResetFences(device.device, 1, &fence), "vkResetFences", reason)) {
        return false;
    }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    return ok(device.fn.vkQueueSubmit(device.queue, 1, &submit, fence), "vkQueueSubmit", reason);
}

inline bool wait(const Device& device, VkFence fence, std::string* reason) {
    return ok(device.fn.vkWaitForFences(device.device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences", reason);
}

inline void barrier(const Device& device, VkCommandBuffer command, VkAccessFlags from, VkPipelineStageFlags from_stage,
                    VkAccessFlags to, VkPipelineStageFlags to_stage) {
    VkMemoryBarrier memory{};
    memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory.srcAccessMask = from;
    memory.dstAccessMask = to;
    device.fn.vkCmdPipelineBarrier(command, from_stage, to_stage, 0, 1, &memory, 0, nullptr, 0, nullptr);
}

// one shader's writes made visible to the next shader's reads, and to the
// device reading a dispatch's size
inline void compute_barrier(const Device& device, VkCommandBuffer command) {
    barrier(device, command, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
}

// `size` bytes at `data` as a device-local storage buffer, copied through a
// staging buffer; an empty array becomes a small buffer the shader never reads
inline bool upload(const Device& device, const void* data, std::size_t size, Buffer* out, std::string* reason) {
    Buffer staging;
    if (!make_buffer(device, size, TRANSFER, HOST, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &staging, reason)) {
        return false;
    }
    Buffer target;
    if (!make_buffer(device, size, STORAGE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &target, reason)) {
        destroy(device, &staging);
        return false;
    }
    bool copied = true;
    if (size > 0) {
        std::memcpy(staging.mapped, data, size);
        copied = begin_commands(device, device.command, reason);
        if (copied) {
            VkBufferCopy region{};
            region.size = size;
            device.fn.vkCmdCopyBuffer(device.command, staging.buffer, target.buffer, 1, &region);
            copied =
                submit_commands(device, device.command, device.fence, reason) && wait(device, device.fence, reason);
        }
    }
    destroy(device, &staging);
    if (!copied) {
        destroy(device, &target);
        return false;
    }
    *out = target;
    return true;
}

// a buffer of `bytes`, mapped host memory the CPU caches or the device's own
inline bool make_sized(const Device& device, VkDeviceSize bytes, bool mapped, Buffer* out, std::string* reason) {
    return make_buffer(device, bytes, STORAGE, mapped ? HOST : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       mapped ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0, out, reason);
}

// A chunk's log, its order and the host's columns, all for `capacity`
// crossings: the new ones made before the old ones go, so a chunk that can't
// grow keeps what it has
inline bool size_log(const Device& device, Chunk* chunk, std::size_t capacity, std::string* reason) {
    const VkDeviceSize word = sizeof(std::uint32_t);
    Buffer* const held[6] = {&chunk->log_words, &chunk->log_distances, &chunk->order,
                             &chunk->out_rays,  &chunk->out_words,     &chunk->out_distances};
    const VkDeviceSize sizes[6] = {4 * word, sizeof(double), word, word, 2 * word, sizeof(double)};
    Buffer made[6];
    bool sized = true;
    for (std::size_t b = 0; b < 6 && sized; b++) {
        sized = make_sized(device, capacity * sizes[b], b >= 3, &made[b], reason);
    }
    for (std::size_t b = 0; b < 6; b++) {
        destroy(device, sized ? held[b] : &made[b]);
        if (sized) {
            *held[b] = made[b];
        }
    }
    if (sized) {
        chunk->capacity = capacity;
    }
    return sized;
}

constexpr std::size_t scan_blocks(std::size_t rays) { return (rays + SCAN_BLOCK - 1) / SCAN_BLOCK; }
// src/vulkan/layout.comp finds a scan block in one dimension of groups, which a device may cap at 65535
static_assert(scan_blocks(CHUNK) <= LAYOUT_WIDEST);

// -- opening the device -------------------------------------------------------

// a shader as a pipeline: `bindings` storage buffers in one set and
// `push_bytes` of push constants
inline bool make_pipeline(const Device& device, const std::uint32_t* code, std::size_t code_bytes,
                          std::uint32_t bindings, std::uint32_t push_bytes, Pipeline* out, std::string* reason) {
    const Functions& fn = device.fn;
    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = code_bytes;
    shader_info.pCode = code;
    VkShaderModule shader = VK_NULL_HANDLE;
    if (!ok(fn.vkCreateShaderModule(device.device, &shader_info, nullptr, &shader), "vkCreateShaderModule", reason)) {
        return false;
    }
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings(bindings);
    for (std::uint32_t b = 0; b < bindings; b++) {
        layout_bindings[b].binding = b;
        layout_bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layout_bindings[b].descriptorCount = 1;
        layout_bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings;
    layout_info.pBindings = layout_bindings.data();
    if (!ok(fn.vkCreateDescriptorSetLayout(device.device, &layout_info, nullptr, &out->set_layout),
            "vkCreateDescriptorSetLayout", reason)) {
        return false;
    }
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = push_bytes;
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &out->set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &range;
    if (!ok(fn.vkCreatePipelineLayout(device.device, &pipeline_layout_info, nullptr, &out->layout),
            "vkCreatePipelineLayout", reason)) {
        return false;
    }
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = out->layout;
    const VkResult result =
        fn.vkCreateComputePipelines(device.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &out->pipeline);
    fn.vkDestroyShaderModule(device.device, shader, nullptr);
    return ok(result, "vkCreateComputePipelines", reason);
}

// What src/vulkan/trace.comp needs of a device beyond Vulkan 1.0's floor, or null:
// double precision, a compute queue (which may carry transfers whether or not
// it says so), and the bindings in one set (the floor is 4)
inline const char* lacking(const VkPhysicalDeviceProperties& properties, const VkPhysicalDeviceFeatures& features,
                           const std::vector<VkQueueFamilyProperties>& queues, std::uint32_t* family) {
    if (features.shaderFloat64 != VK_TRUE) {
        return "no float64";
    }
    if (features.robustBufferAccess != VK_TRUE) {
        return "no robust buffer access";
    }
    if (properties.limits.maxPerStageDescriptorStorageBuffers < TRACE_BINDINGS) {
        return "too few storage buffers";
    }
    for (std::uint32_t q = 0; q < queues.size(); q++) {
        if ((queues[q].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            *family = q;
            return nullptr;
        }
    }
    return "no compute queue";
}

// Open the device: the loader, an instance, the first device that lacks
// nothing (a discrete GPU before an integrated one before a software device),
// its queue, the shaders as pipelines, and the chunks' buffers. Returns the
// reason it couldn't
inline bool open(Device* device, std::string* reason) {
    if (!little_endian()) {
        *reason = "the device path reads 64-bit words as little-endian pairs, which this machine doesn't write";
        return false;
    }
    Functions& fn = device->fn;
    fn.vkGetInstanceProcAddr = open_loader();
    if (fn.vkGetInstanceProcAddr == nullptr) {
        *reason = "no Vulkan loader is installed";
        return false;
    }
    fn.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(fn.vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
    if (fn.vkCreateInstance == nullptr) {
        *reason = "the Vulkan loader has no vkCreateInstance";
        return false;
    }
    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "trimkit";
    application.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    const VkResult created = fn.vkCreateInstance(&instance_info, nullptr, &device->instance);
    if (created == VK_ERROR_INCOMPATIBLE_DRIVER) {
        // what a loader with no driver behind it says
        *reason = "no Vulkan driver is installed";
        return false;
    }
    if (!ok(created, "vkCreateInstance", reason)) {
        return false;
    }
#define TRIMKIT_VULKAN_LOAD(name)                                                              \
    fn.name = reinterpret_cast<PFN_##name>(fn.vkGetInstanceProcAddr(device->instance, #name)); \
    if (fn.name == nullptr) {                                                                  \
        *reason = "the Vulkan loader has no " #name;                                           \
        return false;                                                                          \
    }
    TRIMKIT_VULKAN_INSTANCE_FUNCTIONS(TRIMKIT_VULKAN_LOAD)
#undef TRIMKIT_VULKAN_LOAD

    std::uint32_t count = 0;
    if (!ok(fn.vkEnumeratePhysicalDevices(device->instance, &count, nullptr), "vkEnumeratePhysicalDevices", reason)) {
        return false;
    }
    std::vector<VkPhysicalDevice> candidates(count);
    if (count > 0) {
        if (!ok(fn.vkEnumeratePhysicalDevices(device->instance, &count, candidates.data()),
                "vkEnumeratePhysicalDevices", reason)) {
            return false;
        }
    }
    // the kinds in the order preferred
    const VkPhysicalDeviceType kinds[] = {VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                                          VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU, VK_PHYSICAL_DEVICE_TYPE_CPU,
                                          VK_PHYSICAL_DEVICE_TYPE_OTHER};
    std::string seen;
    for (const VkPhysicalDeviceType kind : kinds) {
        for (std::uint32_t i = 0; i < count && device->physical == VK_NULL_HANDLE; i++) {
            VkPhysicalDeviceProperties properties{};
            fn.vkGetPhysicalDeviceProperties(candidates[i], &properties);
            if (properties.deviceType != kind) {
                continue;
            }
            VkPhysicalDeviceFeatures features{};
            fn.vkGetPhysicalDeviceFeatures(candidates[i], &features);
            std::uint32_t families = 0;
            fn.vkGetPhysicalDeviceQueueFamilyProperties(candidates[i], &families, nullptr);
            std::vector<VkQueueFamilyProperties> queues(families);
            fn.vkGetPhysicalDeviceQueueFamilyProperties(candidates[i], &families, queues.data());
            const char* lack = lacking(properties, features, queues, &device->queue_family);
            if (lack == nullptr) {
                device->physical = candidates[i];
                device->range = properties.limits.maxStorageBufferRange;
            } else {
                seen += (seen.empty() ? "" : ", ") + std::string(properties.deviceName) + " (" + lack + ")";
            }
        }
    }
    if (device->physical == VK_NULL_HANDLE) {
        *reason = count == 0 ? "no Vulkan device is present" : "no Vulkan device can run the kernel: " + seen;
        return false;
    }
    fn.vkGetPhysicalDeviceMemoryProperties(device->physical, &device->memory);

    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = device->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures wanted{};
    wanted.shaderFloat64 = VK_TRUE;
    // a shader's access past a buffer's end reads or writes inside the buffer
    // or nothing: a wrong answer the tests see, never another buffer's memory
    wanted.robustBufferAccess = VK_TRUE;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &wanted;
    if (!ok(fn.vkCreateDevice(device->physical, &device_info, nullptr, &device->device), "vkCreateDevice", reason)) {
        return false;
    }
#define TRIMKIT_VULKAN_LOAD(name)                                                          \
    fn.name = reinterpret_cast<PFN_##name>(fn.vkGetDeviceProcAddr(device->device, #name)); \
    if (fn.name == nullptr) {                                                              \
        *reason = "the Vulkan device has no " #name;                                       \
        return false;                                                                      \
    }
    TRIMKIT_VULKAN_DEVICE_FUNCTIONS(TRIMKIT_VULKAN_LOAD)
#undef TRIMKIT_VULKAN_LOAD
    fn.vkGetDeviceQueue(device->device, device->queue_family, 0, &device->queue);

    // the shaders, their bindings and their pipelines
    if (!make_pipeline(*device, TRACE_SPIRV, sizeof(TRACE_SPIRV), TRACE_BINDINGS, sizeof(PushConstants), &device->trace,
                       reason) ||
        !make_pipeline(*device, LAYOUT_SPIRV, sizeof(LAYOUT_SPIRV), LAYOUT_BINDINGS, sizeof(LayoutConstants),
                       &device->layout, reason)) {
        return false;
    }
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = (TRACE_BINDINGS + LAYOUT_BINDINGS) * IN_FLIGHT;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = SETS * IN_FLIGHT;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (!ok(fn.vkCreateDescriptorPool(device->device, &pool_info, nullptr, &device->descriptor_pool),
            "vkCreateDescriptorPool", reason)) {
        return false;
    }
    VkDescriptorSetLayout layouts[SETS * IN_FLIGHT];
    for (std::size_t s = 0; s < IN_FLIGHT; s++) {
        layouts[(SETS * s)] = device->trace.set_layout;
        layouts[(SETS * s) + 1] = device->layout.set_layout;
    }
    VkDescriptorSet sets[SETS * IN_FLIGHT]{};
    VkDescriptorSetAllocateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = device->descriptor_pool;
    set_info.descriptorSetCount = SETS * IN_FLIGHT;
    set_info.pSetLayouts = layouts;
    if (!ok(fn.vkAllocateDescriptorSets(device->device, &set_info, sets), "vkAllocateDescriptorSets", reason)) {
        return false;
    }

    // commands: one buffer for uploads and one per chunk in flight, each
    // reset on its own, and a fence for each
    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = device->queue_family;
    if (!ok(fn.vkCreateCommandPool(device->device, &command_pool_info, nullptr, &device->command_pool),
            "vkCreateCommandPool", reason)) {
        return false;
    }
    VkCommandBuffer commands[IN_FLIGHT + 1]{};
    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = device->command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = IN_FLIGHT + 1;
    if (!ok(fn.vkAllocateCommandBuffers(device->device, &command_info, commands), "vkAllocateCommandBuffers", reason)) {
        return false;
    }
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    device->command = commands[IN_FLIGHT];
    if (!ok(fn.vkCreateFence(device->device, &fence_info, nullptr, &device->fence), "vkCreateFence", reason)) {
        return false;
    }
    for (std::size_t s = 0; s < IN_FLIGHT; s++) {
        Chunk& chunk = device->chunks[s];
        chunk.command = commands[s];
        chunk.trace_set = sets[SETS * s];
        chunk.layout_set = sets[(SETS * s) + 1];
        if (!ok(fn.vkCreateFence(device->device, &fence_info, nullptr, &chunk.fence), "vkCreateFence", reason)) {
            return false;
        }
        // a chunk's rays and counts, and a log for LOG_PER_RAY crossings of each
        const VkDeviceSize word = sizeof(std::uint32_t);
        const VkDeviceSize point = 3 * sizeof(double);
        if (!make_sized(*device, CHUNK * point, true, &chunk.origins, reason) ||
            !make_sized(*device, CHUNK * point, true, &chunk.directions, reason) ||
            !make_sized(*device, CHUNK * word, true, &chunk.occlusion, reason) ||
            !make_sized(*device, CHUNK * word, false, &chunk.counts, reason) ||
            !make_sized(*device, CHUNK * word, false, &chunk.offsets, reason) ||
            !make_sized(*device, scan_blocks(CHUNK) * word, false, &chunk.blocks, reason) ||
            !make_sized(*device, HEADER_BYTES, false, &chunk.header, reason) ||
            !make_sized(*device, HEADER_BYTES, true, &chunk.readback, reason) ||
            !size_log(*device, &chunk, CHUNK * LOG_PER_RAY, reason)) {
            return false;
        }
    }
    device->pid = process_id();
    return true;
}

struct Opened {
    Device device;
    bool usable = false;
    std::string reason;
};

// The process's device, opened on first use and never closed: null with the
// reason when there is none. A child of a fork inherits the parent's handles
// but not its device, so a use there is refused with a reason too
inline Device* shared_device(std::string* reason) {
    static Opened opened = [] {
        Opened result;
        result.usable = open(&result.device, &result.reason);
        return result;
    }();
    if (!opened.usable) {
        *reason = opened.reason;
        return nullptr;
    }
    if (opened.device.pid != process_id()) {
        *reason = "the Vulkan device was opened before this process was forked";
        return nullptr;
    }
    return &opened.device;
}

// -- a scene on the device, and a query of it ---------------------------------

// one array to put on the device: its bytes
struct Span {
    const void* data;
    std::size_t size;
};

// A scene's meshes, or a top level, on the device: its arrays in the order
// src/vulkan/trace.comp binds them, SCENE_BUFFERS or TOP_BUFFERS of them
struct Arrays {
    Buffer buffers[SCENE_BUFFERS];
};

// `count` arrays uploaded as the first buffers of `out`: false with the
// reason, what was made left for `destroy`
inline bool upload(const Device& device, const Span* spans, std::uint32_t count, Arrays* out, std::string* reason) {
    for (std::uint32_t b = 0; b < count; b++) {
        if (!upload(device, spans[b].data, spans[b].size, &out->buffers[b], reason)) {
            return false;
        }
    }
    return true;
}

inline void destroy(const Device& device, Arrays* arrays) {
    for (Buffer& buffer : arrays->buffers) {
        destroy(device, &buffer);
    }
}

// What one query traces: the scene, and either one mesh directly or the top
// level; whether it only asks if anything is hit; how many crossings per ray
// at most
struct Query {
    const Arrays* scene = nullptr;
    const Arrays* top = nullptr;  // null traces `mesh` directly
    std::uint32_t mesh = 0;
    bool occlusion = false;
    std::uint32_t most = UNBOUNDED;
    double extent = 0.0;
};

// `count` buffers bound to `set`, in binding order
inline void bind_set(const Device& device, VkDescriptorSet set, const Buffer* const* buffers, std::uint32_t count) {
    VkDescriptorBufferInfo infos[TRACE_BINDINGS]{};
    VkWriteDescriptorSet writes[TRACE_BINDINGS]{};
    for (std::uint32_t b = 0; b < count; b++) {
        infos[b].buffer = buffers[b]->buffer;
        infos[b].range = VK_WHOLE_SIZE;
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = set;
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
    }
    device.fn.vkUpdateDescriptorSets(device.device, count, writes, 0, nullptr);
}

// bind what a query reads and the chunk's own buffers to the chunk's two sets:
// before a chunk is recorded, and again when its log has been resized
inline void bind(const Device& device, const Chunk& chunk, const Query& query) {
    const Buffer* trace[TRACE_BINDINGS];
    std::uint32_t b = 0;
    for (const Buffer& buffer : query.scene->buffers) {
        trace[b++] = &buffer;
    }
    // a query of one mesh never reads a top level: any buffer stands in
    for (std::uint32_t i = 0; i < TOP_BUFFERS; i++) {
        trace[b++] = query.top != nullptr ? &query.top->buffers[i] : &query.scene->buffers[0];
    }
    for (const Buffer* buffer : {&chunk.origins, &chunk.directions, &chunk.log_words, &chunk.log_distances,
                                 query.occlusion ? &chunk.occlusion : &chunk.counts, &chunk.header}) {
        trace[b++] = buffer;
    }
    const Buffer* layout[LAYOUT_BINDINGS] = {&chunk.counts,    &chunk.offsets,       &chunk.blocks, &chunk.header,
                                             &chunk.log_words, &chunk.log_distances, &chunk.order,  &chunk.out_rays,
                                             &chunk.out_words, &chunk.out_distances};
    bind_set(device, chunk.trace_set, trace, TRACE_BINDINGS);
    bind_set(device, chunk.layout_set, layout, LAYOUT_BINDINGS);
}

// a pipeline, its set and its constants, for the dispatch that follows
inline void use(const Device& device, VkCommandBuffer command, const Pipeline& pipeline, VkDescriptorSet set,
                const void* constants, std::uint32_t constants_size) {
    device.fn.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    device.fn.vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &set, 0, nullptr);
    device.fn.vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constants_size, constants);
}

// `groups` groups of what `use` chose, in two dimensions: a device may cap
// one dimension at 65535
inline void dispatch(const Device& device, VkCommandBuffer command, std::size_t groups) {
    const std::uint32_t width = static_cast<std::uint32_t>(std::min<std::size_t>(groups, LAYOUT_WIDEST));
    const std::uint32_t height = static_cast<std::uint32_t>((groups + width - 1) / width);
    device.fn.vkCmdDispatch(command, width, height, 1);
}

// Record and submit a chunk: the host has filled its `count` rays. The
// header is zeroed; the kernel counts each ray's crossings and logs them; for
// an occlusion query that is all. Then the scan of the counts into offsets,
// which also writes into the header how many groups lay the log out (none,
// when more was logged than the log holds); those two dispatches, sized by
// the device from the header; and the header copied to host memory
inline bool record_chunk(const Device& device, const Chunk& chunk, const Query& query, std::size_t count,
                         std::string* reason) {
    if (!begin_commands(device, chunk.command, reason)) {
        return false;
    }
    const Functions& fn = device.fn;
    // every count here is under 2^32: a chunk has at most CHUNK rays, and its
    // crossings are indexed with 32-bit words on the device
    PushConstants trace{};
    trace.ray_count = static_cast<std::uint32_t>(count);
    trace.top = query.top != nullptr ? 1 : 0;
    trace.mesh = query.mesh;
    trace.occlusion = query.occlusion ? 1 : 0;
    trace.most = query.most;
    trace.origin_stride = chunk.strides[0];
    trace.direction_stride = chunk.strides[1];
    trace.capacity = static_cast<std::uint32_t>(chunk.capacity);
    trace.extent = query.extent;
    if (!query.occlusion) {
        fn.vkCmdFillBuffer(chunk.command, chunk.header.buffer, 0, HEADER_BYTES, 0);
        barrier(device, chunk.command, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    use(device, chunk.command, device.trace, chunk.trace_set, &trace, sizeof(trace));
    dispatch(device, chunk.command, (count + TRACE_GROUP - 1) / TRACE_GROUP);
    if (query.occlusion) {
        return submit_commands(device, chunk.command, chunk.fence, reason);
    }
    const std::size_t blocks = scan_blocks(count);
    LayoutConstants layout{};
    layout.count = static_cast<std::uint32_t>(count);
    layout.capacity = static_cast<std::uint32_t>(chunk.capacity);
    for (const std::size_t groups : {blocks, std::size_t{1}, blocks}) {
        compute_barrier(device, chunk.command);
        use(device, chunk.command, device.layout, chunk.layout_set, &layout, sizeof(layout));
        dispatch(device, chunk.command, groups);
        layout.phase++;
    }
    // ordering the log, then writing the columns: the header's three counts, after its total
    for (std::uint32_t laid = 0; laid < 2; laid++) {
        compute_barrier(device, chunk.command);
        use(device, chunk.command, device.layout, chunk.layout_set, &layout, sizeof(layout));
        fn.vkCmdDispatchIndirect(chunk.command, chunk.header.buffer, sizeof(std::uint32_t));
        layout.phase++;
    }
    barrier(device, chunk.command, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferCopy region{};
    region.size = HEADER_BYTES;
    fn.vkCmdCopyBuffer(chunk.command, chunk.header.buffer, chunk.readback.buffer, 1, &region);
    return submit_commands(device, chunk.command, chunk.fence, reason);
}

// A chunk's answers as the caller reads them
struct Answers {
    std::size_t begin;               // the chunk's first ray of the query
    std::size_t count;               // and how many
    const std::uint32_t* occlusion;  // an occlusion query: 1 or 0 per ray
    std::size_t total;               // else the chunk's crossings, nearest first by ray:
    const std::uint32_t* rays;       // the ray, counted from the chunk's first,
    const std::uint32_t* words;      // two words: the instance in the top level (0 for one mesh) and the face,
    const double* distances;         // and the distance in the scene's units
};

// One query over `count` rays, chunk k in chunk k % IN_FLIGHT: `fill(origins,
// directions, begin, n, steps)` writes a chunk's rays before the chunk is
// submitted, each array as the STRIDE_* it names in `steps`, or returns false
// to end the query there (rays it won't hand a device: no reason is given),
// and `consume(answers)` reads each
// chunk's answers, in order, while the chunks after it run. False with the
// reason on any failure, with nothing left running on the device either way
template <typename Fill, typename Consume>
bool run_query(Device& device, const Query& query, std::size_t count, const Fill& fill, const Consume& consume,
               std::string* reason) {
    const auto rays_of = [&](std::size_t number) { return std::min(CHUNK, count - (number * CHUNK)); };
    // chunk `number` of the query filled and submitted
    const auto start = [&](std::size_t number) -> bool {
        Chunk& chunk = device.chunks[number % IN_FLIGHT];
        const std::size_t n = rays_of(number);
        bind(device, chunk, query);
        return fill(chunk.origins.mapped, chunk.directions.mapped, number * CHUNK, n, chunk.strides) &&
               record_chunk(device, chunk, query, n, reason);
    };
    // chunk `number` waited for, and with `read` its answers consumed
    const auto finish = [&](std::size_t number, bool read) -> bool {
        Chunk& chunk = device.chunks[number % IN_FLIGHT];
        if (!wait(device, chunk.fence, reason) || !read) {
            return false;
        }
        Answers answers{};
        answers.begin = number * CHUNK;
        answers.count = rays_of(number);
        answers.occlusion = static_cast<const std::uint32_t*>(chunk.occlusion.mapped);
        if (!query.occlusion) {
            const std::uint32_t* logged = static_cast<const std::uint32_t*>(chunk.readback.mapped);
            // more crossings than the log holds were counted, not laid out:
            // the chunk again, its rays still where they are, with a log that
            // holds them (at least twice the last, so few chunks do this)
            while (*logged > chunk.capacity) {
                if (!size_log(device, &chunk, std::max<std::size_t>(*logged, 2 * chunk.capacity), reason)) {
                    return false;
                }
                bind(device, chunk, query);
                if (!record_chunk(device, chunk, query, answers.count, reason) || !wait(device, chunk.fence, reason)) {
                    return false;
                }
            }
            answers.total = *logged;
            answers.rays = static_cast<const std::uint32_t*>(chunk.out_rays.mapped);
            answers.words = static_cast<const std::uint32_t*>(chunk.out_words.mapped);
            answers.distances = static_cast<const double*>(chunk.out_distances.mapped);
        }
        consume(answers);
        return true;
    };
    const std::size_t chunks = (count + CHUNK - 1) / CHUNK;
    std::size_t started = 0;
    std::size_t finished = 0;
    bool working = true;
    try {
        while (working && started < chunks) {
            // a chunk is free once the chunk IN_FLIGHT before this one has finished
            if (started - finished == IN_FLIGHT) {
                working = finish(finished++, true);
            }
            if (working) {
                working = start(started);
                started += working ? 1 : 0;
            }
        }
        // the chunks still in flight, oldest first: waited for even after a failure
        while (finished < started) {
            working = finish(finished++, working) && working;
        }
    } catch (...) {
        // `consume` threw (its columns didn't fit in memory): the chunks still
        // in flight are waited for all the same, so the next query finds the
        // device idle, and the caller gets its exception
        std::string unused;
        while (finished < started) {
            wait(device, device.chunks[finished++ % IN_FLIGHT].fence, &unused);
        }
        throw;
    }
    return working;
}

}  // namespace trimkit::vulkan
