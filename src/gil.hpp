// gil.hpp: the one door to releasing the GIL, `GilReleased`, where every
// binding lets Python run while it computes, so the interpreter's exit can be
// made safe in one place.
//
// CPython runs atexit handlers, then finalizes, and once it finalizes 3.12
// and 3.13 end a daemon thread that takes the GIL: on glibc by unwinding it,
// on musl and Windows without unwinding at all. A thread ended while it takes
// the GIL back, in a destructor, is std::terminate on glibc; one ended
// mid-computation, as a poll for Ctrl-C takes the GIL (common.hpp,
// Interrupt), leaves its stack, and any pool job on it, to vanish under the
// threads still using it. So every GilReleased counts itself through the
// pool's exit gate (parallel.hpp, ExitGate), and the module's atexit handler
// closes the gate, which stops every pool job, then waits with the GIL
// released for every call inside to take it back, which each can while the
// handler runs: none is left to lose its thread. A call that releases the
// GIL after the gate closed keeps it instead, so it never takes it back once
// the interpreter finalizes (tests/test_threads.py,
// test_exit_while_a_daemon_thread_queries, test_forked_child_exits_promptly)
#pragma once

#include "parallel.hpp"

#include <nanobind/nanobind.h>

namespace trimkit {

// The GIL released for this scope, unless the interpreter is exiting
class GilReleased {
public:
    GilReleased() {
        if (parallel::enter_exit_gate()) {
            state_ = PyEval_SaveThread();
        }
    }
    ~GilReleased() {
        if (state_ != nullptr) {
            PyEval_RestoreThread(state_);
        }
        parallel::leave_exit_gate();
    }
    GilReleased(const GilReleased&) = delete;
    GilReleased& operator=(const GilReleased&) = delete;
    GilReleased(GilReleased&&) = delete;
    GilReleased& operator=(GilReleased&&) = delete;

private:
    PyThreadState* state_ = nullptr;
};

// What a call stopped by the interpreter's exit raises, with the GIL held:
// SystemExit, which is what the exit is, which a thread's excepthook ignores
// (printing a traceback from a daemon thread as the interpreter finalizes is
// itself a fatal error), and which `except Exception` doesn't swallow. Never
// what the call had so far
[[noreturn]] inline void raise_exiting() {
    PyErr_SetNone(PyExc_SystemExit);
    throw nanobind::python_error();
}

// The module's atexit handler: the exit gate closed, so every pool job stops
// at its next claim, then, with the GIL released, a wait for every call that
// released it to take it back, for as long as that takes: serial work can't
// be stopped part way, and a time limit would only leave it to take the GIL
// back as the interpreter finalizes. Exit may be slow, never a crash
inline void exit_at_interpreter_exit() {
    parallel::close_exit_gate();
    PyThreadState* state = PyEval_SaveThread();
    parallel::wait_for_exit_gate();
    PyEval_RestoreThread(state);
}

}  // namespace trimkit
