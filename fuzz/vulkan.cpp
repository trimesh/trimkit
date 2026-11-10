// Property: for any triangles, built by tinybvh with spatial splits or
// without, the walk that puts a tree on the device (src/vulkan/vulkan.hpp)
// accepts the tree, hands a leaf only faces that exist, reaches every face
// through some leaf slot, and writes an image for each node it reaches and
// no other: so the shader never follows a slot or a node this didn't check.
// A split build leaves stale slots between its leaves' ranges, which is what
// following every slot once crashed on.
#include "../src/vulkan/vulkan.hpp"

#include <fuzzer/FuzzedDataProvider.h>
#include <tiny_bvh.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    using V = tinybvh::bvhdbl3;
    using namespace trimkit::vulkan;
    FuzzedDataProvider in(data, size);
    const auto m = in.ConsumeIntegralInRange<std::uint64_t>(1, 96);
    // long thin triangles across one another are what a split build splits
    const double reach = in.ConsumeBool() ? 10.0 : 0.5;
    std::vector<V> vertices(3 * m);
    std::vector<std::uint64_t> faces(3 * m);
    for (std::uint64_t f = 0; f < m; f++) {
        const V at(in.ConsumeFloatingPointInRange(-10.0, 10.0), in.ConsumeFloatingPointInRange(-10.0, 10.0),
                   in.ConsumeFloatingPointInRange(-10.0, 10.0));
        for (std::uint64_t corner = 0; corner < 3; corner++) {
            vertices[(3 * f) + corner] =
                at + V(in.ConsumeFloatingPointInRange(-reach, reach), in.ConsumeFloatingPointInRange(-reach, reach),
                       in.ConsumeFloatingPointInRange(-0.5, 0.5));
            faces[(3 * f) + corner] = (3 * f) + corner;
        }
    }
    tinybvh::BVH_Double bvh;
    if (in.ConsumeBool()) {
        bvh.BuildHQ(vertices.data(), faces.data(), m);
    } else {
        bvh.Build(vertices.data(), faces.data(), m);
    }
    static_assert(sizeof(tinybvh::BVH_Double::BVHNode) == sizeof(Node64));
    std::vector<Node64> nodes(bvh.usedNodes);
    std::memcpy(nodes.data(), bvh.bvhNode, nodes.size() * sizeof(Node64));
    Tree tree{};
    tree.nodes = nodes.data();
    tree.node_count = nodes.size();
    tree.slots = bvh.primIdx;
    tree.slot_count = bvh.idxCount;
    tree.limit = m;
    tree.depth = MESH_DEPTH;
    std::vector<Node32> node32(nodes.size());
    std::vector<std::uint32_t> seen(m, 0);
    std::vector<std::uint32_t> slots(bvh.idxCount, 0);
    std::string reason;
    const bool sound = walk(
        tree, node32.data(),
        [&](std::size_t slot, std::uint64_t face) {
            if (face >= m || slot >= slots.size()) {
                __builtin_trap();
            }
            seen[face]++;
            slots[slot]++;
        },
        &reason);
    // under a hundred faces is never deeper than the shader's stack
    if (!sound) {
        __builtin_trap();
    }
    for (std::uint64_t f = 0; f < m; f++) {
        if (seen[f] == 0) {
            __builtin_trap();
        }
    }
    for (const std::uint32_t reached : slots) {
        if (reached > 1) {
            __builtin_trap();
        }
    }
    // a leaf's image names the slots the walk handed out; an interior node's, two nodes
    for (std::size_t n = 0; n < nodes.size(); n++) {
        const Node32& image = node32[n];
        const bool interior = (image.count_word & INTERIOR) != 0;
        const std::uint64_t end = std::uint64_t{image.left} + (interior ? 2 : image.count_word);
        if (end > (interior ? nodes.size() : slots.size())) {
            __builtin_trap();
        }
    }
    return 0;
}
