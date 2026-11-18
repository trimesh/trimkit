// T4: a poll that says stop, on the calling thread, stops the call while the
// worker may be inside it: the calling thread runs no range after its first,
// the call returns with no thread left inside (T1: the plain ints are a race
// otherwise), and no item runs twice
#include "harness.hpp"

std::atomic<int> runs[3];
int written[3];
int caller_ranges = 0;  // the calling thread's alone: index 0

int main() {
    trimkit::parallel::Stop stop;
    stop.poll = [](void* /*context*/) { return true; };
    const auto failure = trimkit::parallel::for_each_range(
        3,
        [](std::size_t begin, std::size_t end) {
            if (trimkit::parallel::thread_index() == 0) {
                caller_ranges++;
            }
            for (std::size_t i = begin; i < end; i++) {
                runs[i].fetch_add(1, std::memory_order_relaxed);
                written[i] = 1;
            }
        },
        trimkit::parallel::ItemCost::Coarse, 1, stop);
    assert(!failure);  // a stop is the caller's to report (common.hpp's Interrupt), not a failure
    assert(caller_ranges <= 1);
    for (int i = 0; i < 3; i++) {
        assert(runs[i].load(std::memory_order_relaxed) <= 1);
        assert(written[i] == runs[i].load(std::memory_order_relaxed));
    }
    return 0;
}
