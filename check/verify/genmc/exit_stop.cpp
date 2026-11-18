// The exit: a call that may run as the interpreter exits counts itself into
// the exit gate (as gil.hpp's GilReleased does), and the exit closes the gate
// and waits. Every execution ends; the call either ran every item or failed,
// never returned part of an answer; and once the wait returns, no call is
// inside the gate
#include "harness.hpp"

std::atomic<int> runs[2];
std::atomic<int> inside{0};
std::atomic<bool> partial{false};

int main() {
    auto call = [] {
        const bool released = trimkit::parallel::enter_exit_gate();
        if (released) {
            inside.store(1, std::memory_order_relaxed);
        }
        const auto failure = trimkit::parallel::for_each_range(
            2,
            [](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; i++) {
                    runs[i].fetch_add(1, std::memory_order_relaxed);
                }
            },
            trimkit::parallel::ItemCost::Coarse);
        int ran = 0;
        for (auto& item : runs) {
            ran += item.load(std::memory_order_relaxed);
        }
        partial.store(!failure && ran != 2, std::memory_order_relaxed);
        if (released) {
            inside.store(0, std::memory_order_relaxed);
        }
        trimkit::parallel::leave_exit_gate();
    };
    const pthread_t thread = harness::start(call);
    trimkit::parallel::close_exit_gate();
    trimkit::parallel::wait_for_exit_gate();
    assert(inside.load(std::memory_order_relaxed) == 0);
    harness::join(thread);
    assert(!partial.load(std::memory_order_relaxed));
    return 0;
}
