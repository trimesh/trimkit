// The thread indexes: no two threads running items of one call hold the same
// index at once, so per-thread scratch indexed by it (xatlas) is never shared
#include "harness.hpp"

std::atomic<int> holding[2];

int main() {
    const auto failure = trimkit::parallel::for_each_range(
        2,
        [](std::size_t /*begin*/, std::size_t /*end*/) {
            const std::uint32_t index = trimkit::parallel::thread_index();
            assert(index < trimkit::parallel::thread_slots());
            assert(holding[index].fetch_add(1, std::memory_order_relaxed) == 0);
            holding[index].fetch_sub(1, std::memory_order_relaxed);
        },
        trimkit::parallel::ItemCost::Coarse);
    assert(!failure);
    return 0;
}
