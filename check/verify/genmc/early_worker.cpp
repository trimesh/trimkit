// T1 and T2 for a worker already running when the call publishes: its join,
// not its thread's start, is what orders the job's fields and the outputs
// (a worker the call itself starts is ordered by pthread_create, which would
// hide a weak publish or join)
#include "harness.hpp"

std::atomic<int> runs[2];
int written[2];

int main() {
    (void)trimkit::parallel::worker_cap();  // the cap, as a call reads it first
    (void)trimkit::parallel::spawn_worker();
    const auto failure = trimkit::parallel::for_each_range(
        2,
        [](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; i++) {
                runs[i].fetch_add(1, std::memory_order_relaxed);
                written[i] = 1;
            }
        },
        trimkit::parallel::ItemCost::Coarse);
    assert(!failure);
    for (int i = 0; i < 2; i++) {
        assert(runs[i].load(std::memory_order_relaxed) == 1);
        assert(written[i] == 1);
    }
    return 0;
}
