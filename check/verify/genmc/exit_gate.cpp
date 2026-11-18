// The exit gate alone, as gil.hpp uses it: a call either sees the gate closed
// (and keeps the GIL), or the closer sees the call and waits for it. Once the
// wait returns, no call that let go of the GIL is still without it
#include "harness.hpp"

std::atomic<int> released_now{0};

int main() {
    auto call = [] {
        for (int k = 0; k < 2; k++) {
            if (trimkit::parallel::enter_exit_gate()) {
                released_now.fetch_add(1, std::memory_order_relaxed);
                released_now.fetch_sub(1, std::memory_order_relaxed);
            }
            trimkit::parallel::leave_exit_gate();
        }
    };
    const pthread_t thread = harness::start(call);
    trimkit::parallel::close_exit_gate();
    trimkit::parallel::wait_for_exit_gate();
    assert(released_now.load(std::memory_order_relaxed) == 0);
    // and a call after the gate closed never lets go
    assert(!trimkit::parallel::enter_exit_gate());
    trimkit::parallel::leave_exit_gate();
    harness::join(thread);
    return 0;
}
