// T1: what a body writes, on any thread, is visible to the caller once the
// call returns. The outputs are plain ints: a missing happens-before between
// a worker's writes and the caller's reads is a data race, which GenMC reports
#include "harness.hpp"

int out[3];

int main() {
    const auto failure = trimkit::parallel::for_each_range(3, [](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            out[i] = static_cast<int>(i) + 1;
        }
    });
    assert(!failure);
    for (int i = 0; i < 3; i++) {
        assert(out[i] == i + 1);
    }
    return 0;
}
