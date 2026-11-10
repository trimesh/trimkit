// Ray queries on a Vulkan device: src/trace.comp (tinybvh's traversals,
// filtered in f32 and exact in f64, so the device answers byte for byte what
// the CPU answers) over a RayScene's meshes, src/scan.comp and src/compact.comp
// laying its answers out. The loader is opened at run time by name, never
// linked, so every wheel works on a machine without Vulkan: anything missing
// or failing is a reason string the caller warns with and answers on the CPU.
// One device per process, opened on first use and kept; the caller serializes
// its use (a queue is not thread safe).
// Rays go through in chunks, two in flight, each one submission and one
// fence: the kernel reads a chunk's rays straight from the mapped host memory
// the caller fills piece by piece behind events (so the copy overlaps the
// tracing; a broadcast bit reads one row for every ray), counts each ray's
// crossings and, for a small cap, writes them dense; the device scans the
// counts into offsets; an unbounded query traces again to write every
// crossing at its offset; and a compaction writes the crossings as the caller
// reads them, compacted and in order, into mapped host memory. Nothing per
// ray crosses the host between steps. Python-free: src/bvh.cpp is the only
// user. Vulkan 1.0 core, no extensions.
#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

#include "trace_spirv.hpp"

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
    X(vkCreateEvent)                       \
    X(vkResetEvent)                        \
    X(vkSetEvent)                          \
    X(vkCmdWaitEvents)                     \
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

// the trace shader's bindings, in src/trace.comp's order: the scene's five,
// the top level's four, a chunk's origins and directions, the crossings'
// instances, faces and distances, the f32 shadows of the scene's nodes and
// triangles and of the top level's nodes, counts and offsets
constexpr std::uint32_t TRACE_BINDINGS = 19;
// src/scan.comp: counts, offsets, block sums
constexpr std::uint32_t SCAN_BINDINGS = 3;
// src/compact.comp: offsets, block sums, the kernel's three hit arrays, the
// four columns the host reads
constexpr std::uint32_t COMPACT_BINDINGS = 9;
constexpr std::uint32_t SETS = 3;  // descriptor sets per chunk in flight: one per shader

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
// a chunk's origins and directions as the shader reads them: 3 doubles each,
// in two arrays, so a caller's own arrays copy in whole
constexpr std::size_t RAY_DOUBLES = 3;
// the most rays in one chunk: 96 MB of origins, and with DENSE crossings per
// ray 128 MB of dense distances, under every device's storage buffer range
// (lavapipe's is exactly 128 MB). A chunk's buffers grow to what is asked,
// never shrink
constexpr std::size_t CHUNK = std::size_t{1} << 22U;
constexpr std::uint32_t GROUP_SIZE = 64;
// the most crossings per ray the kernel writes dense in its counting
// traversal, so a query capped there traces once; beyond, it traces twice
constexpr std::uint32_t DENSE = 4;
constexpr std::uint32_t SCAN_BLOCK = 1024;  // counts per scan block: 4 a lane
constexpr std::uint32_t COMPACT_GROUP = 256;
// chunks in flight at once
constexpr std::size_t SLOTS = 2;
// a chunk is filled and traced in pieces: the host fills one while the device
// traces the one before it
constexpr std::size_t PIECES = 8;
constexpr std::uint32_t UNBOUNDED = 0xffffffffU;

struct PushConstants {
    std::uint32_t ray_count;
    std::uint32_t mode;  // 0: one mesh, 1: the top level
    std::uint32_t mesh;
    std::uint32_t occlusion;
    std::uint32_t most;        // crossings per ray at most
    std::uint32_t pass_index;  // 0 counts, 1 writes at the offsets, 2 counts and writes dense
    std::uint32_t ray_first;   // this dispatch's first ray of the chunk
    std::uint32_t capacity;    // crossings the hit buffers hold
    std::uint32_t broadcast;   // bit 0: one origin for every ray, bit 1: one direction
    std::uint32_t padding;     // `extent` is aligned to 8 in the shader too
    double extent;             // the scene's size
};
struct ScanConstants {
    std::uint32_t count;
    std::uint32_t step;
    std::uint32_t block_count;
};
struct CompactConstants {
    std::uint32_t count;
    std::uint32_t block_count;
    std::uint32_t dense;  // slots per ray the kernel wrote dense, 0 for compacted
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

// What one chunk in flight needs. Mapped host memory the CPU caches for what
// the host writes or reads: the rays, read by the kernel over the bus as it
// traces (measured faster than copying them to the device first), the
// occlusion words, the chunk's total of crossings, and the four columns the
// compaction writes. The device's own memory for what only shaders touch:
// the counts, their scan, and the crossings as the kernel writes them. Every
// buffer is grown to the most a chunk has needed, never shrunk
struct Slot {
    Buffer origins;
    Buffer directions;
    Buffer occlusion;
    Buffer counts;
    Buffer offsets;
    Buffer blocks;
    Buffer total;
    std::size_t ray_capacity = 0;  // rays the ray buffers hold
    Buffer hit_instances;
    Buffer hit_faces;
    Buffer hit_distances;
    Buffer rays;
    Buffer instances;
    Buffer faces;
    Buffer distances;
    std::size_t hit_capacity = 0;  // crossings the hit buffers and columns hold
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkEvent events[PIECES] = {};  // set by the host as it fills each piece
    VkDescriptorSet trace_set = VK_NULL_HANDLE;
    VkDescriptorSet scan_set = VK_NULL_HANDLE;
    VkDescriptorSet compact_set = VK_NULL_HANDLE;
    bool pending = false;
    // the chunk it holds: which rays
    std::size_t begin = 0;
    std::size_t count = 0;
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
    Pipeline scan;
    Pipeline compact;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    // the commands and fence uploads use, and the chunks in flight
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Slot slots[SLOTS];
    // what a query of one mesh binds as the top level's buffers: never read
    Buffer empty;
    int pid = 0;
};

inline int process_id() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

inline std::string failed(const char* what, VkResult result) {
    return std::string(what) + " failed (VkResult " + std::to_string(static_cast<int>(result)) + ")";
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
    Buffer buffer;
    buffer.size = std::max<VkDeviceSize>(size, 16);
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = buffer.size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = device.fn.vkCreateBuffer(device.device, &info, nullptr, &buffer.buffer);
    if (result != VK_SUCCESS) {
        *reason = failed("vkCreateBuffer", result);
        return false;
    }
    VkMemoryRequirements requirements{};
    device.fn.vkGetBufferMemoryRequirements(device.device, buffer.buffer, &requirements);
    const int type = find_memory(device.memory, requirements.memoryTypeBits, required, preferred);
    if (type < 0) {
        destroy(device, &buffer);
        *reason = "no device memory of the kind needed";
        return false;
    }
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = static_cast<std::uint32_t>(type);
    result = device.fn.vkAllocateMemory(device.device, &allocate, nullptr, &buffer.memory);
    if (result != VK_SUCCESS) {
        destroy(device, &buffer);
        *reason = failed("vkAllocateMemory", result);
        return false;
    }
    result = device.fn.vkBindBufferMemory(device.device, buffer.buffer, buffer.memory, 0);
    if (result != VK_SUCCESS) {
        destroy(device, &buffer);
        *reason = failed("vkBindBufferMemory", result);
        return false;
    }
    if ((required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        result = device.fn.vkMapMemory(device.device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped);
        if (result != VK_SUCCESS) {
            destroy(device, &buffer);
            *reason = failed("vkMapMemory", result);
            return false;
        }
    }
    *out = buffer;
    return true;
}

constexpr VkMemoryPropertyFlags HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkBufferUsageFlags TRANSFER = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
constexpr VkBufferUsageFlags STORAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | TRANSFER;

inline bool begin_commands(const Device& device, VkCommandBuffer command, std::string* reason) {
    VkResult result = device.fn.vkResetCommandBuffer(command, 0);
    if (result != VK_SUCCESS) {
        *reason = failed("vkResetCommandBuffer", result);
        return false;
    }
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = device.fn.vkBeginCommandBuffer(command, &begin);
    if (result != VK_SUCCESS) {
        *reason = failed("vkBeginCommandBuffer", result);
        return false;
    }
    return true;
}

// end the recorded commands and submit them, to signal `fence`
inline bool submit_commands(const Device& device, VkCommandBuffer command, VkFence fence, std::string* reason) {
    VkResult result = device.fn.vkEndCommandBuffer(command);
    if (result != VK_SUCCESS) {
        *reason = failed("vkEndCommandBuffer", result);
        return false;
    }
    result = device.fn.vkResetFences(device.device, 1, &fence);
    if (result != VK_SUCCESS) {
        *reason = failed("vkResetFences", result);
        return false;
    }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    result = device.fn.vkQueueSubmit(device.queue, 1, &submit, fence);
    if (result != VK_SUCCESS) {
        *reason = failed("vkQueueSubmit", result);
        return false;
    }
    return true;
}

inline bool wait(const Device& device, VkFence fence, std::string* reason) {
    const VkResult result = device.fn.vkWaitForFences(device.device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        *reason = failed("vkWaitForFences", result);
        return false;
    }
    return true;
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

// a capacity grown by doubling to hold `needed`, never shrunk
inline std::size_t grown(std::size_t capacity, std::size_t needed) {
    std::size_t next = std::max<std::size_t>(capacity, std::size_t{1} << 16U);
    while (next < needed) {
        next *= 2;
    }
    return next;
}

// a mapped host buffer the CPU caches, of `elements` of `size` bytes each
inline bool make_mapped(const Device& device, std::size_t elements, std::size_t size, Buffer* out,
                        std::string* reason) {
    destroy(device, out);
    return make_buffer(device, VkDeviceSize{elements} * size, STORAGE, HOST, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, out,
                       reason);
}

// a buffer in the device's own memory
inline bool make_local(const Device& device, std::size_t elements, std::size_t size, Buffer* out, std::string* reason) {
    destroy(device, out);
    return make_buffer(device, VkDeviceSize{elements} * size, STORAGE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, out,
                       reason);
}

inline std::size_t scan_blocks(std::size_t rays) { return (rays + SCAN_BLOCK - 1) / SCAN_BLOCK; }

// room in a slot's ray buffers for `rays`
inline bool hold_rays(const Device& device, Slot* slot, std::size_t rays, std::string* reason) {
    if (rays <= slot->ray_capacity) {
        return true;
    }
    const std::size_t capacity = grown(slot->ray_capacity, rays);
    slot->ray_capacity = 0;
    const std::size_t point = RAY_DOUBLES * sizeof(double);
    if (!make_mapped(device, capacity, point, &slot->origins, reason) ||
        !make_mapped(device, capacity, point, &slot->directions, reason) ||
        !make_mapped(device, capacity, sizeof(std::uint32_t), &slot->occlusion, reason) ||
        !make_local(device, capacity, sizeof(std::uint32_t), &slot->counts, reason) ||
        !make_local(device, capacity, sizeof(std::uint32_t), &slot->offsets, reason) ||
        !make_local(device, scan_blocks(capacity) + 1, sizeof(std::uint32_t), &slot->blocks, reason)) {
        return false;
    }
    slot->ray_capacity = capacity;
    return true;
}

// room in a slot's hit buffers and columns for `hits` crossings
inline bool hold_hits(const Device& device, Slot* slot, std::size_t hits, std::string* reason) {
    if (hits <= slot->hit_capacity) {
        return true;
    }
    const std::size_t capacity = grown(slot->hit_capacity, hits);
    slot->hit_capacity = 0;
    // a crossing is a u32 instance, a u32 face and a double distance, and in
    // the columns its u32 ray too
    if (!make_local(device, capacity, 4, &slot->hit_instances, reason) ||
        !make_local(device, capacity, 4, &slot->hit_faces, reason) ||
        !make_local(device, capacity, 8, &slot->hit_distances, reason) ||
        !make_mapped(device, capacity, 4, &slot->rays, reason) ||
        !make_mapped(device, capacity, 4, &slot->instances, reason) ||
        !make_mapped(device, capacity, 4, &slot->faces, reason) ||
        !make_mapped(device, capacity, 8, &slot->distances, reason)) {
        return false;
    }
    slot->hit_capacity = capacity;
    return true;
}

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
    VkResult result = fn.vkCreateShaderModule(device.device, &shader_info, nullptr, &shader);
    if (result != VK_SUCCESS) {
        *reason = failed("vkCreateShaderModule", result);
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
    result = fn.vkCreateDescriptorSetLayout(device.device, &layout_info, nullptr, &out->set_layout);
    if (result != VK_SUCCESS) {
        *reason = failed("vkCreateDescriptorSetLayout", result);
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
    result = fn.vkCreatePipelineLayout(device.device, &pipeline_layout_info, nullptr, &out->layout);
    if (result != VK_SUCCESS) {
        *reason = failed("vkCreatePipelineLayout", result);
        return false;
    }
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = out->layout;
    result = fn.vkCreateComputePipelines(device.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &out->pipeline);
    fn.vkDestroyShaderModule(device.device, shader, nullptr);
    if (result != VK_SUCCESS) {
        *reason = failed("vkCreateComputePipelines", result);
        return false;
    }
    return true;
}

// What src/trace.comp needs of a device beyond Vulkan 1.0's floor, or null:
// double precision, a compute queue (which may carry transfers whether or not
// it says so), and the bindings in one set (the floor is 4)
inline const char* lacking(const VkPhysicalDeviceProperties& properties, const VkPhysicalDeviceFeatures& features,
                           const std::vector<VkQueueFamilyProperties>& queues, std::uint32_t* family) {
    if (features.shaderFloat64 != VK_TRUE) {
        return "no float64";
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
    VkResult result = fn.vkCreateInstance(&instance_info, nullptr, &device->instance);
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
        // what a loader with no driver behind it says
        *reason = "no Vulkan driver is installed";
        return false;
    }
    if (result != VK_SUCCESS) {
        *reason = failed("vkCreateInstance", result);
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
    result = fn.vkEnumeratePhysicalDevices(device->instance, &count, nullptr);
    if (result != VK_SUCCESS) {
        *reason = failed("vkEnumeratePhysicalDevices", result);
        return false;
    }
    std::vector<VkPhysicalDevice> candidates(count);
    if (count > 0) {
        result = fn.vkEnumeratePhysicalDevices(device->instance, &count, candidates.data());
        if (result != VK_SUCCESS) {
            *reason = failed("vkEnumeratePhysicalDevices", result);
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
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &wanted;
    result = fn.vkCreateDevice(device->physical, &device_info, nullptr, &device->device);
    if (result != VK_SUCCESS) {
        *reason = failed("vkCreateDevice", result);
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
        !make_pipeline(*device, SCAN_SPIRV, sizeof(SCAN_SPIRV), SCAN_BINDINGS, sizeof(ScanConstants), &device->scan,
                       reason) ||
        !make_pipeline(*device, COMPACT_SPIRV, sizeof(COMPACT_SPIRV), COMPACT_BINDINGS, sizeof(CompactConstants),
                       &device->compact, reason)) {
        return false;
    }
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = (TRACE_BINDINGS + SCAN_BINDINGS + COMPACT_BINDINGS) * SLOTS;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = SETS * SLOTS;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    result = fn.vkCreateDescriptorPool(device->device, &pool_info, nullptr, &device->descriptor_pool);
    if (result != VK_SUCCESS) {
        *reason = failed("vkCreateDescriptorPool", result);
        return false;
    }
    VkDescriptorSetLayout layouts[SETS * SLOTS];
    for (std::size_t s = 0; s < SLOTS; s++) {
        layouts[(SETS * s)] = device->trace.set_layout;
        layouts[(SETS * s) + 1] = device->scan.set_layout;
        layouts[(SETS * s) + 2] = device->compact.set_layout;
    }
    VkDescriptorSet sets[SETS * SLOTS]{};
    VkDescriptorSetAllocateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = device->descriptor_pool;
    set_info.descriptorSetCount = SETS * SLOTS;
    set_info.pSetLayouts = layouts;
    result = fn.vkAllocateDescriptorSets(device->device, &set_info, sets);
    if (result != VK_SUCCESS) {
        *reason = failed("vkAllocateDescriptorSets", result);
        return false;
    }

    // commands: one buffer for uploads and one per chunk in flight, each
    // reset on its own, a fence for each, and a chunk's events
    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = device->queue_family;
    result = fn.vkCreateCommandPool(device->device, &command_pool_info, nullptr, &device->command_pool);
    if (result != VK_SUCCESS) {
        *reason = failed("vkCreateCommandPool", result);
        return false;
    }
    VkCommandBuffer commands[SLOTS + 1]{};
    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = device->command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = SLOTS + 1;
    result = fn.vkAllocateCommandBuffers(device->device, &command_info, commands);
    if (result != VK_SUCCESS) {
        *reason = failed("vkAllocateCommandBuffers", result);
        return false;
    }
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkEventCreateInfo event_info{};
    event_info.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    device->command = commands[SLOTS];
    result = fn.vkCreateFence(device->device, &fence_info, nullptr, &device->fence);
    if (result != VK_SUCCESS) {
        *reason = failed("vkCreateFence", result);
        return false;
    }
    for (std::size_t s = 0; s < SLOTS; s++) {
        Slot& slot = device->slots[s];
        slot.command = commands[s];
        slot.trace_set = sets[SETS * s];
        slot.scan_set = sets[(SETS * s) + 1];
        slot.compact_set = sets[(SETS * s) + 2];
        result = fn.vkCreateFence(device->device, &fence_info, nullptr, &slot.fence);
        if (result != VK_SUCCESS) {
            *reason = failed("vkCreateFence", result);
            return false;
        }
        for (VkEvent& event : slot.events) {
            result = fn.vkCreateEvent(device->device, &event_info, nullptr, &event);
            if (result != VK_SUCCESS) {
                *reason = failed("vkCreateEvent", result);
                return false;
            }
        }
        if (!make_mapped(*device, 1, sizeof(std::uint32_t), &slot.total, reason) ||
            !hold_rays(*device, &slot, 1, reason) || !hold_hits(*device, &slot, 1, reason)) {
            return false;
        }
    }
    if (!upload(*device, nullptr, 0, &device->empty, reason)) {
        return false;
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

// A scene's meshes on the device, as src/trace.comp reads them: tinybvh's
// nodes and geometry for the exact path, their f32 shadows for the filter
struct Scene {
    Buffer nodes;
    Buffer primitives;
    Buffer corners;
    Buffer vertices;
    Buffer meshes;
    Buffer node32;
    Buffer tri32;
};

// A top level on the device: its nodes, instance indices, inverse
// transforms, each instance's mesh, and its nodes' f32 shadow
struct Top {
    Buffer nodes;
    Buffer primitives;
    Buffer inverses;
    Buffer instance_meshes;
    Buffer node32;
};

inline void destroy(const Device& device, Scene* scene) {
    for (Buffer* buffer : {&scene->nodes, &scene->primitives, &scene->corners, &scene->vertices, &scene->meshes,
                           &scene->node32, &scene->tri32}) {
        destroy(device, buffer);
    }
}

inline void destroy(const Device& device, Top* top) {
    for (Buffer* buffer : {&top->nodes, &top->primitives, &top->inverses, &top->instance_meshes, &top->node32}) {
        destroy(device, buffer);
    }
}

// What one query traces: the scene, and either one mesh directly or the top
// level; whether it only asks if anything is hit; how many crossings per ray
// at most; and whether every ray shares one origin or one direction (bits 0
// and 1), so the host fills and the device reads one row
struct Query {
    const Scene* scene = nullptr;
    const Top* top = nullptr;  // null traces `mesh` directly
    std::uint32_t mesh = 0;
    bool occlusion = false;
    std::uint32_t most = UNBOUNDED;
    std::uint32_t broadcast = 0;
    double extent = 0.0;
};

// a query capped at DENSE crossings traces once, writing them dense
inline bool dense(const Query& query) { return !query.occlusion && query.most <= DENSE; }

// piece `p` of a chunk of `count` rays: its first ray and how many
inline void piece_of(std::size_t count, std::size_t p, std::size_t* first, std::size_t* piece) {
    const std::size_t size = (count + PIECES - 1) / PIECES;
    *first = std::min(count, p * size);
    *piece = std::min(count, *first + size) - *first;
}

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
// to the slot's three sets: before each chunk is recorded
inline void bind(const Device& device, const Slot& slot, const Query& query) {
    const Top* top = query.top;
    const Buffer* trace[TRACE_BINDINGS] = {
        &query.scene->nodes,
        &query.scene->primitives,
        &query.scene->corners,
        &query.scene->vertices,
        &query.scene->meshes,
        top != nullptr ? &top->nodes : &device.empty,
        top != nullptr ? &top->primitives : &device.empty,
        top != nullptr ? &top->inverses : &device.empty,
        top != nullptr ? &top->instance_meshes : &device.empty,
        &slot.origins,
        &slot.directions,
        &slot.hit_instances,
        &slot.hit_faces,
        &slot.hit_distances,
        &query.scene->node32,
        &query.scene->tri32,
        top != nullptr ? &top->node32 : &device.empty,
        query.occlusion ? &slot.occlusion : &slot.counts,
        &slot.offsets,
    };
    const Buffer* scan[SCAN_BINDINGS] = {&slot.counts, &slot.offsets, &slot.blocks};
    const Buffer* compact[COMPACT_BINDINGS] = {&slot.offsets,   &slot.blocks,        &slot.hit_instances,
                                               &slot.hit_faces, &slot.hit_distances, &slot.rays,
                                               &slot.instances, &slot.faces,         &slot.distances};
    bind_set(device, slot.trace_set, trace, TRACE_BINDINGS);
    bind_set(device, slot.scan_set, scan, SCAN_BINDINGS);
    bind_set(device, slot.compact_set, compact, COMPACT_BINDINGS);
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

// the kernel over rays [first, first + count) of the slot's chunk. Every
// count here is under 2^32: a chunk has at most CHUNK rays, and its crossings
// are indexed with 32-bit words on the device (an offset of 4 billion
// crossings in one chunk would be 80 GB of answers)
inline void trace_rays(const Device& device, const Slot& slot, const Query& query, std::size_t first, std::size_t count,
                       std::uint32_t pass_index, std::size_t capacity) {
    PushConstants constants{};
    constants.ray_count = static_cast<std::uint32_t>(count);
    constants.mode = query.top != nullptr ? 1 : 0;
    constants.mesh = query.mesh;
    constants.occlusion = query.occlusion ? 1 : 0;
    constants.most = query.most;
    constants.pass_index = pass_index;
    constants.ray_first = static_cast<std::uint32_t>(first);
    constants.capacity = static_cast<std::uint32_t>(capacity);
    constants.broadcast = query.broadcast;
    constants.extent = query.extent;
    dispatch(device, slot.command, device.trace, slot.trace_set, &constants, sizeof(constants),
             (count + GROUP_SIZE - 1) / GROUP_SIZE);
}

// Record and submit a chunk's commands. From the start: each piece's kernel
// behind the event the host sets once it has filled the piece (counting, and
// writing dense when `dense`); for an occlusion query that is all. Then the
// scan of the counts, with the chunk's total copied to host memory; for a
// query past DENSE the kernel again over the whole chunk, writing every
// crossing at its offset up to the hit buffers' capacity; and the compaction
// of the crossings into the host's columns. `traced` false skips the pieces
// and the scan: the hit buffers were short and have grown, so only the
// writing and the compaction run again
inline bool record_chunk(const Device& device, Slot& slot, const Query& query, bool traced, std::string* reason) {
    if (!begin_commands(device, slot.command, reason)) {
        return false;
    }
    const std::size_t count = slot.count;
    const std::size_t dense_slots = dense(query) ? count * query.most : 0;
    if (!traced) {
        for (std::size_t p = 0; p < PIECES; p++) {
            std::size_t first = 0;
            std::size_t piece = 0;
            piece_of(count, p, &first, &piece);
            if (piece == 0) {
                break;
            }
            device.fn.vkResetEvent(device.device, slot.events[p]);
            // the host's writes of the piece's rays, visible to the kernel
            VkMemoryBarrier filled{};
            filled.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            filled.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            filled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            device.fn.vkCmdWaitEvents(slot.command, 1, &slot.events[p], VK_PIPELINE_STAGE_HOST_BIT,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1, &filled, 0, nullptr, 0, nullptr);
            trace_rays(device, slot, query, first, piece, dense_slots != 0 ? 2 : 0, dense_slots);
        }
        if (query.occlusion) {
            return submit_commands(device, slot.command, slot.fence, reason);
        }
        const std::size_t blocks = scan_blocks(count);
        ScanConstants scan{};
        scan.count = static_cast<std::uint32_t>(count);
        scan.block_count = static_cast<std::uint32_t>(blocks);
        compute_barrier(device, slot.command);
        dispatch(device, slot.command, device.scan, slot.scan_set, &scan, sizeof(scan), blocks);
        compute_barrier(device, slot.command);
        scan.step = 1;
        dispatch(device, slot.command, device.scan, slot.scan_set, &scan, sizeof(scan), 1);
        barrier(device, slot.command, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy region{};
        region.srcOffset = blocks * sizeof(std::uint32_t);
        region.size = sizeof(std::uint32_t);
        device.fn.vkCmdCopyBuffer(slot.command, slot.blocks.buffer, slot.total.buffer, 1, &region);
        scan.step = 2;
        dispatch(device, slot.command, device.scan, slot.scan_set, &scan, sizeof(scan), blocks);
        compute_barrier(device, slot.command);
    }
    if (dense_slots == 0) {
        trace_rays(device, slot, query, 0, count, 1, slot.hit_capacity);
        compute_barrier(device, slot.command);
    }
    CompactConstants compact{};
    compact.count = static_cast<std::uint32_t>(count);
    compact.block_count = static_cast<std::uint32_t>(scan_blocks(count));
    compact.dense = dense_slots != 0 ? query.most : 0;
    const std::size_t slots = dense_slots != 0 ? dense_slots : slot.hit_capacity;
    dispatch(device, slot.command, device.compact, slot.compact_set, &compact, sizeof(compact),
             (slots + COMPACT_GROUP - 1) / COMPACT_GROUP);
    return submit_commands(device, slot.command, slot.fence, reason);
}

// A chunk's rays as the caller fills them, a piece at a time: 3 doubles each,
// the chunk starting at ray `begin` of the query
struct Chunk {
    double* origins;
    double* directions;
    std::size_t begin;
};

// A chunk's answers as the caller reads them
struct Answers {
    std::size_t begin;
    std::size_t count;
    const std::uint32_t* occlusion;  // an occlusion query: 1 or 0 per ray
    std::size_t total;               // else the chunk's crossings, nearest first by ray:
    const std::uint32_t* rays;       // the ray, counted from the chunk's first,
    const std::uint32_t* instances;  // the instance in the top level (0 for one mesh),
    const std::uint32_t* faces;      // the face, and the distance in the scene's units
    const double* distances;
};

// One query over `count` rays, a chunk at a time with two in flight. After a
// chunk is submitted, `fill(chunk, first, piece)` writes each piece's rays
// (the device waits for each); `consume(answers)` reads a chunk's answers,
// in order. False with the reason on any failure
template <typename Fill, typename Consume>
bool run_query(Device& device, const Query& query, std::size_t count, const Fill& fill, const Consume& consume,
               std::string* reason) {
    const auto finish = [&](Slot& slot) -> bool {
        slot.pending = false;
        if (!wait(device, slot.fence, reason)) {
            return false;
        }
        Answers answers{};
        answers.begin = slot.begin;
        answers.count = slot.count;
        answers.occlusion = static_cast<const std::uint32_t*>(slot.occlusion.mapped);
        if (!query.occlusion) {
            answers.total = *static_cast<const std::uint32_t*>(slot.total.mapped);
            if (answers.total > slot.hit_capacity) {
                // the guess was short: room for what was counted, and the
                // writing traversal and the compaction again
                if (!hold_hits(device, &slot, answers.total, reason)) {
                    return false;
                }
                bind(device, slot, query);
                if (!record_chunk(device, slot, query, true, reason) || !wait(device, slot.fence, reason)) {
                    return false;
                }
            }
            answers.rays = static_cast<const std::uint32_t*>(slot.rays.mapped);
            answers.instances = static_cast<const std::uint32_t*>(slot.instances.mapped);
            answers.faces = static_cast<const std::uint32_t*>(slot.faces.mapped);
            answers.distances = static_cast<const double*>(slot.distances.mapped);
        }
        consume(answers);
        return true;
    };
    std::size_t chunks = 0;
    for (std::size_t begin = 0; begin < count; begin += CHUNK, chunks++) {
        Slot& slot = device.slots[chunks % SLOTS];
        if (slot.pending && !finish(slot)) {
            return false;
        }
        const std::size_t n = std::min(CHUNK, count - begin);
        // room for the rays, and for the crossings: exactly the dense slots,
        // or the guess for an unbounded query, a closed surface entered and
        // left, grown by what earlier chunks needed
        if (!hold_rays(device, &slot, n, reason) ||
            !hold_hits(device, &slot, dense(query) ? n * query.most : 2 * n, reason)) {
            return false;
        }
        slot.begin = begin;
        slot.count = n;
        bind(device, slot, query);
        if (!record_chunk(device, slot, query, false, reason)) {
            return false;
        }
        slot.pending = true;
        Chunk chunk{};
        chunk.origins = static_cast<double*>(slot.origins.mapped);
        chunk.directions = static_cast<double*>(slot.directions.mapped);
        chunk.begin = begin;
        // every event is set whatever a piece holds: the device waits for each
        for (std::size_t p = 0; p < PIECES; p++) {
            std::size_t first = 0;
            std::size_t piece = 0;
            piece_of(n, p, &first, &piece);
            fill(chunk, first, piece);
            device.fn.vkSetEvent(device.device, slot.events[p]);
        }
    }
    // the older chunk first
    for (std::size_t s = 0; s < SLOTS; s++) {
        Slot& slot = device.slots[(chunks + s) % SLOTS];
        if (slot.pending && !finish(slot)) {
            return false;
        }
    }
    return true;
}

}  // namespace trimkit::vulkan
