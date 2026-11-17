// Exclusively, the device lock: one holder at a time (the plain counter is a
// data race otherwise), and every waiter gets its turn (no unlock is lost:
// every thread ends)
#include "harness.hpp"

std::atomic<std::uint32_t> lock{0};
int counter = 0;

int main() {
    auto take = [] {
        const trimkit::parallel::Exclusively holding(lock);
        counter++;
    };
    const pthread_t thread = harness::start(take);
    take();
    harness::join(thread);
    assert(counter == 2);
    return 0;
}
