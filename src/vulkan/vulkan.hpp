// Ray queries on a Vulkan device: src/vulkan/trace.comp (tinybvh's traversals,
// filtered in f32 and exact in f64, so the device answers byte for byte what
// the CPU answers) over a RayScene's meshes, and src/vulkan/layout.comp laying
// its answers out. The loader is opened at run time by name, never
// linked, so every wheel works on a machine without Vulkan: anything missing
// or failing is a reason string the caller warns with and answers on the CPU.
// One device per process, opened on first use and kept; the caller serializes
// its use (a queue is not thread safe).
// Rays go through in chunks of a quarter million, two in flight: the host
// fills a chunk's rays into mapped host memory, submits, and fills the next
// while the device traces this one. Host memory is written only before the
// submission that reads it: written behind events while a submission ran, a
// device was measured answering from bytes it had already fetched. The
// kernel reads the rays where the host put them (as float64 or float32 rows,
// or one row for every ray), counts each ray's crossings and writes its first DENSE
// into the ray's own room, and the device scans the counts into offsets:
// that is a chunk's first submission, and all of an occlusion query. The
// host reads back one total, makes room for exactly that many crossings, and
// a second submission writes them as the caller reads them, compacted and in
// order, into mapped host memory; a chunk with none skips it. Only the rays
// that crossed more than DENSE times are traced again, there, each writing
// every crossing at its offset. Python-free: src/bvh.cpp is the only
// user. Vulkan 1.0 core, no extensions.
#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

#include "spirv.hpp"

#if defined(_WIN32)
// windows.h for the loader's two calls, without its min/max macros and the
// headers we don't use
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>
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
// src/vulkan/layout.comp: counts, offsets, block sums, the kernel's crossings
// (each ray's room, and those at the offsets: words and distances of each),
// and the three columns the host reads
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

// A triangle per primitive slot as the filter reads it (docs/trace.md 4, 5):
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
// the deepest leaf src/vulkan/trace.comp's stacks hold a path to: its
// STACK_SIZE for a mesh, its TOP_STACK_SIZE for a top level
constexpr std::uint32_t MESH_DEPTH = 64;
constexpr std::uint32_t TOP_DEPTH = 32;

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

// Walk a tree from its root, as the shader will: each node reached gets its
// f32 image in `node32` (docs/trace.md 3: the box rounded to nearest, and for
// an interior node which of its children's coordinates tie exactly, which
// decides their order), and `leaf(slot, value)` is called for each slot of
// each leaf. Nothing is followed before it is checked: a child is a node, a
// leaf's slots are slots, a slot's value is under the limit, no node is
// reached twice and no leaf is deeper than the shader's stack. So the shader
// reads only what was written here, and a slot no leaf reaches (a split
// build leaves stale ones) is never read at all. False with the reason for a
// tree that fails any of it
template <typename Leaf>
bool walk(const Tree& tree, Node32* node32, const Leaf& leaf, std::string* reason) {
    struct Pending {
        std::uint32_t node;
        std::uint32_t depth;
    };
    Pending stack[MESH_DEPTH + 2];
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
                if (sound) {
                    leaf(static_cast<std::size_t>(slot), tree.slots[slot]);
                }
            }
            image.count_word = static_cast<std::uint32_t>(node.count);
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
// device's tracing of a third (measured against 2^16 to 2^22)
constexpr std::size_t CHUNK = std::size_t{1} << 18U;
constexpr std::uint32_t GROUP_SIZE = 64;
// crossings per ray the kernel keeps from its counting traversal, each ray
// in its own room: only a ray with more is traced again. Rays through a real
// assembly cross about nine surfaces when they hit it at all: one in ten of
// those more than 16 (traced again, that cost half as much as the first
// traversal), one in a thousand more than 32
constexpr std::uint32_t DENSE = 32;
constexpr std::uint32_t LAYOUT_GROUP = 256;
constexpr std::uint32_t SCAN_BLOCK = 4 * LAYOUT_GROUP;  // counts per scan block: 4 a lane
// chunks in flight at once: the host fills and reads one while the device
// traces the other (measured 11 to 23% faster than one, at 16M rays)
constexpr std::size_t SLOTS = 2;
constexpr std::uint32_t UNBOUNDED = 0xffffffffU;

// How a chunk holds an array of rays' origins or directions, as the words from
// one ray's row to the next: one float64 row for every ray, a float32 row
// each, or a float64 row each. The host's fill decides it per array and per
// chunk, and src/vulkan/trace.comp reads through it
constexpr std::uint32_t STEP_ONE = 0;
constexpr std::uint32_t STEP_NARROW = 3;
constexpr std::uint32_t STEP_WIDE = 6;

struct PushConstants {
    std::uint32_t ray_count;
    std::uint32_t mode;  // 0: one mesh, 1: the top level
    std::uint32_t mesh;
    std::uint32_t occlusion;
    std::uint32_t most;            // crossings per ray at most
    std::uint32_t origin_step;     // words between rays' rows: STEP_ONE, STEP_NARROW or STEP_WIDE
    std::uint32_t direction_step;  // the same, of the directions
    std::uint32_t dense;           // crossings per ray its room holds
    std::uint32_t retrace;         // 1: only the rays with more, each writing every crossing at its offset
    std::uint32_t spare;           // `extent` is aligned to 8 in the shader too
    double extent;                 // the scene's size
};
struct LayoutConstants {
    std::uint32_t count;
    std::uint32_t step;  // 0 to 2 scan the counts into offsets, 3 writes the crossings out
    std::uint32_t block_count;
    std::uint32_t dense;  // crossings per ray the kernel wrote dense (a count past it raises the overflow word), or 0
};

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

// What one chunk in flight needs, whichever chunk it holds. Mapped host
// memory the CPU caches for what the host writes or reads: the rays, read by
// the kernel over the bus as it traces (measured faster than copying them to
// the device first), the occlusion words, the chunk's total of crossings,
// and the three columns the layout writes. The device's own memory for what
// only shaders touch: the counts, their scan, each ray's room of DENSE
// crossings, and the crossings of the rays that had more, at their offsets.
// A crossing is two words, its instance and face, and a distance; in the
// host's columns its ray too. The ray buffers are a chunk's size from the
// start; the rest grow to the most a chunk of the query needs, and what grew
// large is let go when the query ends (`release`)
struct Slot {
    Buffer origins;
    Buffer directions;
    Buffer occlusion;
    Buffer counts;
    Buffer offsets;
    Buffer blocks;
    Buffer total;
    Buffer room_words;
    Buffer room_distances;
    Buffer more_words;
    Buffer more_distances;
    Buffer rays;
    Buffer words;
    Buffer distances;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSet trace_set = VK_NULL_HANDLE;
    VkDescriptorSet layout_set = VK_NULL_HANDLE;
    // how the chunk it holds has its origins and directions: STEP_*
    std::uint32_t steps[2] = {STEP_WIDE, STEP_WIDE};
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
    Slot slots[SLOTS];
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
    const HMODULE library = LoadLibraryA("vulkan-1.dll");
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
constexpr VkBufferUsageFlags STORAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | TRANSFER;

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

// one shader's writes made visible to the next shader's reads
inline void compute_barrier(const Device& device, VkCommandBuffer command) {
    barrier(device, command, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
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
    bool ok = true;
    if (size > 0) {
        std::memcpy(staging.mapped, data, size);
        ok = begin_commands(device, device.command, reason);
        if (ok) {
            VkBufferCopy region{};
            region.size = size;
            device.fn.vkCmdCopyBuffer(device.command, staging.buffer, target.buffer, 1, &region);
            ok = submit_commands(device, device.command, device.fence, reason) && wait(device, device.fence, reason);
        }
    }
    destroy(device, &staging);
    if (!ok) {
        destroy(device, &target);
        return false;
    }
    *out = target;
    return true;
}

// Room in `buffer` for `bytes`, mapped host memory the CPU caches or the
// device's own: kept if it has it, else replaced by one grown by doubling,
// whose contents are new
inline bool hold(const Device& device, Buffer* buffer, VkDeviceSize bytes, bool mapped, std::string* reason) {
    if (bytes <= buffer->size) {
        return true;
    }
    VkDeviceSize size = std::max<VkDeviceSize>(buffer->size, VkDeviceSize{1} << 16U);
    while (size < bytes) {
        size *= 2;
    }
    // doubled past what the device binds, what was asked is still tried
    size = std::max(bytes, std::min(size, device.range));
    destroy(device, buffer);
    return make_buffer(device, size, STORAGE, mapped ? HOST : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       mapped ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0, buffer, reason);
}

// A buffer a query grew past `KEEP` bytes, made small again: a device holds
// between queries a full chunk's rooms (allocating them costs a millisecond
// each, measured), not the columns and offsets of a chunk that crossed more.
// Kept as it is if the small one can't be made
constexpr VkDeviceSize KEEP = VkDeviceSize{64} << 20U;
inline void release(const Device& device, Buffer* buffer) {
    if (buffer->size <= KEEP) {
        return;
    }
    Buffer small;
    std::string ignored;
    if (hold(device, &small, 1, buffer->mapped != nullptr, &ignored)) {
        destroy(device, buffer);
        *buffer = small;
    }
}

constexpr std::size_t scan_blocks(std::size_t rays) { return (rays + SCAN_BLOCK - 1) / SCAN_BLOCK; }
// src/vulkan/layout.comp finds a scan block in one dimension of groups, which a device may cap at 65535
static_assert(scan_blocks(CHUNK) <= 32768);

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
        *reason = count == 0 ? "no Vulkan device is present" : "no Vulkan device computes in double precision: " + seen;
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
    pool_size.descriptorCount = (TRACE_BINDINGS + LAYOUT_BINDINGS) * SLOTS;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = SETS * SLOTS;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (!ok(fn.vkCreateDescriptorPool(device->device, &pool_info, nullptr, &device->descriptor_pool),
            "vkCreateDescriptorPool", reason)) {
        return false;
    }
    VkDescriptorSetLayout layouts[SETS * SLOTS];
    for (std::size_t s = 0; s < SLOTS; s++) {
        layouts[(SETS * s)] = device->trace.set_layout;
        layouts[(SETS * s) + 1] = device->layout.set_layout;
    }
    VkDescriptorSet sets[SETS * SLOTS]{};
    VkDescriptorSetAllocateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = device->descriptor_pool;
    set_info.descriptorSetCount = SETS * SLOTS;
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
    VkCommandBuffer commands[SLOTS + 1]{};
    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = device->command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = SLOTS + 1;
    if (!ok(fn.vkAllocateCommandBuffers(device->device, &command_info, commands), "vkAllocateCommandBuffers", reason)) {
        return false;
    }
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    device->command = commands[SLOTS];
    if (!ok(fn.vkCreateFence(device->device, &fence_info, nullptr, &device->fence), "vkCreateFence", reason)) {
        return false;
    }
    for (std::size_t s = 0; s < SLOTS; s++) {
        Slot& slot = device->slots[s];
        slot.command = commands[s];
        slot.trace_set = sets[SETS * s];
        slot.layout_set = sets[(SETS * s) + 1];
        if (!ok(fn.vkCreateFence(device->device, &fence_info, nullptr, &slot.fence), "vkCreateFence", reason)) {
            return false;
        }
        // a chunk's rays and counts, and something to bind for the rest
        const VkDeviceSize word = sizeof(std::uint32_t);
        const VkDeviceSize point = 3 * sizeof(double);
        if (!hold(*device, &slot.origins, CHUNK * point, true, reason) ||
            !hold(*device, &slot.directions, CHUNK * point, true, reason) ||
            !hold(*device, &slot.occlusion, CHUNK * word, true, reason) ||
            !hold(*device, &slot.counts, CHUNK * word, false, reason) ||
            !hold(*device, &slot.offsets, CHUNK * word, false, reason) ||
            !hold(*device, &slot.blocks, (scan_blocks(CHUNK) + 2) * word, false, reason) ||
            !hold(*device, &slot.total, 2 * word, true, reason)) {
            return false;
        }
        for (Buffer* buffer : {&slot.room_words, &slot.room_distances, &slot.more_words, &slot.more_distances}) {
            if (!hold(*device, buffer, 1, false, reason)) {
                return false;
            }
        }
        for (Buffer* buffer : {&slot.rays, &slot.words, &slot.distances}) {
            if (!hold(*device, buffer, 1, true, reason)) {
                return false;
            }
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

// crossings per ray a chunk's counting traversal keeps
inline std::uint32_t room_width(const Query& query) { return std::min(query.most, DENSE); }

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

// bind what a query reads and the slot's own buffers (which may have regrown)
// to the slot's two sets, before each submission is recorded. The kernel
// writes crossings to each ray's room, or tracing again, to the offsets
inline void bind(const Device& device, const Slot& slot, const Query& query, bool retrace) {
    const Buffer* trace[TRACE_BINDINGS];
    std::uint32_t b = 0;
    for (const Buffer& buffer : query.scene->buffers) {
        trace[b++] = &buffer;
    }
    // a query of one mesh never reads a top level: any buffer stands in
    for (std::uint32_t i = 0; i < TOP_BUFFERS; i++) {
        trace[b++] = query.top != nullptr ? &query.top->buffers[i] : &query.scene->buffers[0];
    }
    for (const Buffer* buffer : {&slot.origins, &slot.directions, retrace ? &slot.more_words : &slot.room_words,
                                 retrace ? &slot.more_distances : &slot.room_distances,
                                 query.occlusion ? &slot.occlusion : &slot.counts, &slot.offsets}) {
        trace[b++] = buffer;
    }
    const Buffer* layout[LAYOUT_BINDINGS] = {
        &slot.counts,     &slot.offsets,        &slot.blocks, &slot.room_words, &slot.room_distances,
        &slot.more_words, &slot.more_distances, &slot.rays,   &slot.words,      &slot.distances};
    bind_set(device, slot.trace_set, trace, TRACE_BINDINGS);
    bind_set(device, slot.layout_set, layout, LAYOUT_BINDINGS);
}

// `groups` groups of a pipeline, in two dimensions: a device may cap one
// dimension at 65535
inline void dispatch(const Device& device, VkCommandBuffer command, const Pipeline& pipeline, VkDescriptorSet set,
                     const void* constants, std::uint32_t constants_size, std::size_t groups) {
    device.fn.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    device.fn.vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &set, 0, nullptr);
    device.fn.vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constants_size, constants);
    const std::uint32_t width = static_cast<std::uint32_t>(std::min<std::size_t>(groups, 32768));
    const std::uint32_t height = static_cast<std::uint32_t>((groups + width - 1) / width);
    device.fn.vkCmdDispatch(command, width, height, 1);
}

// the kernel over the `count` rays of the slot's chunk. Every count here is
// under 2^32: a chunk has at most CHUNK rays, and its crossings are indexed
// with 32-bit words on the device (an offset of 4 billion crossings in one
// chunk would be 80 GB of answers)
inline void trace_rays(const Device& device, const Slot& slot, const Query& query, std::size_t count, bool retrace) {
    PushConstants constants{};
    constants.ray_count = static_cast<std::uint32_t>(count);
    constants.mode = query.top != nullptr ? 1 : 0;
    constants.mesh = query.mesh;
    constants.occlusion = query.occlusion ? 1 : 0;
    constants.most = query.most;
    constants.origin_step = slot.steps[0];
    constants.direction_step = slot.steps[1];
    constants.dense = room_width(query);
    constants.retrace = retrace ? 1 : 0;
    constants.extent = query.extent;
    dispatch(device, slot.command, device.trace, slot.trace_set, &constants, sizeof(constants),
             (count + GROUP_SIZE - 1) / GROUP_SIZE);
}

// A chunk's first submission, the host having filled its `count` rays: the
// kernel, counting each ray's crossings and writing its first DENSE to its
// room; for an occlusion query that is all. Then the scan of the counts into
// offsets, which raises the overflow word when a ray had more, and the
// chunk's total and that word copied to host memory
inline bool record_trace(const Device& device, const Slot& slot, const Query& query, std::size_t count,
                         std::string* reason) {
    if (!begin_commands(device, slot.command, reason)) {
        return false;
    }
    trace_rays(device, slot, query, count, false);
    if (query.occlusion) {
        return submit_commands(device, slot.command, slot.fence, reason);
    }
    const std::size_t blocks = scan_blocks(count);
    LayoutConstants layout{};
    layout.count = static_cast<std::uint32_t>(count);
    layout.block_count = static_cast<std::uint32_t>(blocks);
    layout.dense = room_width(query);
    for (const std::size_t groups : {blocks, std::size_t{1}, blocks}) {
        compute_barrier(device, slot.command);
        dispatch(device, slot.command, device.layout, slot.layout_set, &layout, sizeof(layout), groups);
        layout.step++;
    }
    // the total and the overflow word, after the last block
    barrier(device, slot.command, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferCopy region{};
    region.srcOffset = blocks * sizeof(std::uint32_t);
    region.size = 2 * sizeof(std::uint32_t);
    device.fn.vkCmdCopyBuffer(slot.command, slot.blocks.buffer, slot.total.buffer, 1, &region);
    return submit_commands(device, slot.command, slot.fence, reason);
}

// A chunk's second submission, for its `total` crossings, the host having
// made room for them: with `overflow`, the kernel again for the rays that
// crossed more than their room holds, each writing every crossing at its
// offset; then a lane per crossing writing it into the host's columns
inline bool record_layout(const Device& device, const Slot& slot, const Query& query, std::size_t count,
                          std::size_t total, bool overflow, std::string* reason) {
    if (!begin_commands(device, slot.command, reason)) {
        return false;
    }
    if (overflow) {
        trace_rays(device, slot, query, count, true);
        compute_barrier(device, slot.command);
    }
    LayoutConstants layout{};
    layout.count = static_cast<std::uint32_t>(count);
    layout.step = 3;
    layout.block_count = static_cast<std::uint32_t>(scan_blocks(count));
    layout.dense = room_width(query);
    dispatch(device, slot.command, device.layout, slot.layout_set, &layout, sizeof(layout),
             (total + LAYOUT_GROUP - 1) / LAYOUT_GROUP);
    return submit_commands(device, slot.command, slot.fence, reason);
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

// One query over `count` rays, chunk k in slot k % SLOTS: `fill(origins,
// directions, begin, n, steps)` writes a chunk's rays before the chunk is
// submitted, each array as the STEP_* it names in `steps`, or returns false
// to end the query there (rays it won't hand a device: no reason is given),
// and `consume(answers)` reads each
// chunk's answers, in order, while the chunks after it run. False with the
// reason on any failure, with nothing left running on the device either way
template <typename Fill, typename Consume>
bool run_query(Device& device, const Query& query, std::size_t count, const Fill& fill, const Consume& consume,
               std::string* reason) {
    const auto rays_of = [&](std::size_t chunk) { return std::min(CHUNK, count - (chunk * CHUNK)); };
    const VkDeviceSize word = sizeof(std::uint32_t);
    // chunk `chunk` filled and submitted
    const auto start = [&](std::size_t chunk) -> bool {
        Slot& slot = device.slots[chunk % SLOTS];
        const std::size_t n = rays_of(chunk);
        // each ray's room, which an occlusion query never writes
        const VkDeviceSize room = query.occlusion ? 0 : VkDeviceSize{n} * room_width(query);
        if (!hold(device, &slot.room_words, room * 2 * word, false, reason) ||
            !hold(device, &slot.room_distances, room * sizeof(double), false, reason)) {
            return false;
        }
        bind(device, slot, query, false);
        return fill(slot.origins.mapped, slot.directions.mapped, chunk * CHUNK, n, slot.steps) &&
               record_trace(device, slot, query, n, reason);
    };
    // chunk `chunk` waited for, and with `read` its crossings laid out and its answers consumed
    const auto finish = [&](std::size_t chunk, bool read) -> bool {
        Slot& slot = device.slots[chunk % SLOTS];
        if (!wait(device, slot.fence, reason) || !read) {
            return false;
        }
        Answers answers{};
        answers.begin = chunk * CHUNK;
        answers.count = rays_of(chunk);
        answers.occlusion = static_cast<const std::uint32_t*>(slot.occlusion.mapped);
        const std::uint32_t* total = static_cast<const std::uint32_t*>(slot.total.mapped);
        if (!query.occlusion && total[0] != 0) {
            answers.total = total[0];
            const bool overflow = total[1] != 0;
            // room for exactly the crossings counted: the columns, and for a
            // ray past its own room, every crossing at its offset
            const VkDeviceSize more = overflow ? answers.total : 0;
            if (!hold(device, &slot.rays, answers.total * word, true, reason) ||
                !hold(device, &slot.words, answers.total * 2 * word, true, reason) ||
                !hold(device, &slot.distances, answers.total * sizeof(double), true, reason) ||
                !hold(device, &slot.more_words, more * 2 * word, false, reason) ||
                !hold(device, &slot.more_distances, more * sizeof(double), false, reason)) {
                return false;
            }
            bind(device, slot, query, true);
            if (!record_layout(device, slot, query, answers.count, answers.total, overflow, reason) ||
                !wait(device, slot.fence, reason)) {
                return false;
            }
            answers.rays = static_cast<const std::uint32_t*>(slot.rays.mapped);
            answers.words = static_cast<const std::uint32_t*>(slot.words.mapped);
            answers.distances = static_cast<const double*>(slot.distances.mapped);
        }
        consume(answers);
        return true;
    };
    const std::size_t chunks = (count + CHUNK - 1) / CHUNK;
    std::size_t started = 0;
    std::size_t finished = 0;
    bool working = true;
    while (working && started < chunks) {
        // a slot is free once the chunk SLOTS before this one has finished
        if (started - finished == SLOTS) {
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
    for (Slot& slot : device.slots) {
        for (Buffer* buffer : {&slot.room_words, &slot.room_distances, &slot.more_words, &slot.more_distances,
                               &slot.rays, &slot.words, &slot.distances}) {
            release(device, buffer);
        }
    }
    return working;
}

}  // namespace trimkit::vulkan
