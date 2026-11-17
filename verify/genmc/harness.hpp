// What every harness shares: the pool exactly as src/parallel.hpp is, built
// for GenMC (platform.hpp's TRIMKIT_GENMC backend), and threads to call it
// from. GenMC runs each harness in every execution the C++ memory model
// allows, and fails on an assertion, a data race, a use after free, or (with
// --check-liveness) a spin that never ends. Every thread of an execution must
// end: idle workers exit at once under GenMC, so a caller left waiting forever
// is a run that never finishes, which `uv run scripts genmc` counts as a failure
#pragma once

// the header from the include path: src/, or a mutant of it (`uv run scripts genmc`)
#include "parallel.hpp"

#include <pthread.h>

#include <atomic>
#include <cassert>
#include <cstddef>

namespace harness {

template <typename Run>
void* run_thread(void* run) {
    (*static_cast<Run*>(run))();
    return nullptr;
}

// `run()` on a thread of its own
template <typename Run>
pthread_t start(Run& run) {
    pthread_t thread{};
    pthread_create(&thread, nullptr, run_thread<Run>, &run);
    return thread;
}

inline void join(pthread_t thread) { pthread_join(thread, nullptr); }

}  // namespace harness
