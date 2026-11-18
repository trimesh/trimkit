// T4: a body that fails (under GenMC, by a flag in place of a throw) makes
// the call fail, on whichever thread it ran; no item runs twice; and the call
// still returns only once no thread is inside it (T1: the plain int below is
// a race otherwise)
#include "harness.hpp"

std::atomic<int> runs[2];
int written[2];

int main() {
    const auto failure = trimkit::parallel::for_each_range(
        2,
        [](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; i++) {
                runs[i].fetch_add(1, std::memory_order_relaxed);
                written[i] = 1;
                if (i == 1) {
                    trimkit::platform::genmc_body_failed = true;
                }
            }
        },
        trimkit::parallel::ItemCost::Coarse);
    assert(failure);
    assert(runs[1].load(std::memory_order_relaxed) == 1);
    for (int i = 0; i < 2; i++) {
        assert(runs[i].load(std::memory_order_relaxed) <= 1);
        assert(written[i] == runs[i].load(std::memory_order_relaxed));
    }
    return 0;
}
