// gil.hpp: the one door to releasing the GIL, `GilReleased`, where every
// binding lets Python run while it computes, so the interpreter's exit can be
// made safe in one place.
//
// CPython runs atexit handlers, then finalizes, and once it finalizes 3.12
// and 3.13 end a daemon thread that takes the GIL: on glibc by unwinding it,
// on musl and Windows without unwinding at all. A thread ended while it takes
// the GIL back, in a destructor, is std::terminate on glibc; one ended
// mid-computation, as a poll for Ctrl-C takes the GIL (Interrupt,
// below), leaves its stack, and any pool job on it, to vanish under the
// threads still using it. So every GilReleased counts itself through the
// pool's exit gate (parallel.hpp, ExitGate), and the module's atexit handler
// closes the gate, which stops every pool job, then waits with the GIL
// released for every call inside to take it back, which each can while the
// handler runs: none is left to lose its thread. A call that releases the
// GIL after the gate closed keeps it instead, so it never takes it back once
// the interpreter finalizes (check/tests/test_threads.py,
// test_sequences_of_calls, test_forked_child_exits_promptly)
#pragma once

#include "trimkit.hpp"

#include <nanobind/nanobind.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <thread>

namespace nb = nanobind;

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

// A call's Stop (with_pool): Ctrl-C, polled while the GIL is released. Only
// the calling thread receives signals, so others return at once, and polls
// are rate limited. Once interrupted the error stays set on the calling
// thread, for `raise_if_stopped` to raise
class Interrupt {
public:
    static constexpr std::chrono::milliseconds POLL_PERIOD{parallel::POLL_MS};

    Interrupt() = default;
    Interrupt(const Interrupt&) = delete;
    Interrupt& operator=(const Interrupt&) = delete;

    // with the GIL held, once the work has stopped: Ctrl-C as the error it
    // set, the exit as StoppedAtExit, which install_fork_handler's translator
    // raises as SystemExit; never what the work had so far
    void raise_if_stopped() const {
        if (exiting_) {
            throw parallel::StoppedAtExit();
        }
        if (interrupted_) {
            throw nb::python_error();
        }
    }

    // returns whether the computation should stop: at Ctrl-C, or once the
    // interpreter is exiting, which stops it without taking the GIL, so the
    // exit isn't held up by it (gil.hpp)
    bool poll() {
        if (exiting_ || parallel::exit_gate_closed()) {
            exiting_ = true;
            return true;
        }
        const auto now = std::chrono::steady_clock::now();
        if (interrupted_ || std::this_thread::get_id() != caller_ || now - last_ < POLL_PERIOD) {
            return interrupted_;
        }
        last_ = now;
        parallel::checked(!parallel::holding_exclusively, "no poll takes the GIL while holding Exclusively");
        const nb::gil_scoped_acquire acquire;
        interrupted_ = PyErr_CheckSignals() != 0;
        return interrupted_;
    }

private:
    std::thread::id caller_ = std::this_thread::get_id();
    std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
    // read from the library's worker threads, written only by the caller
    std::atomic<bool> interrupted_{false};
    std::atomic<bool> exiting_{false};
};

// A part's work from Python: `work(stop)` with the GIL released, where `stop`
// is Ctrl-C (and the interpreter's exit), polled by every loop the part runs
// on the pool (trimkit.hpp, in_parallel). A loop it stopped throws Stopped,
// which comes back out of here as KeyboardInterrupt, or SystemExit at the
// exit; any other exception passes through
template <typename Work>
auto with_pool(const Work& work) {
    // from Python, holding the GIL: a body calling back in would release a
    // GIL it doesn't hold
    parallel::checked(parallel::bodies_running == 0, "no body enters the pool from Python");
    Interrupt interrupt;
    const parallel::Stop stop{[](void* context) { return static_cast<Interrupt*>(context)->poll(); }, &interrupt};
    try {
        const GilReleased release;
        return work(stop);
    } catch (const Stopped&) {
        interrupt.raise_if_stopped();
        throw;
    }
}

// What the module's init installs: the pool's fork handler (which resets the
// exit gate too), and two translations no other exception translator makes:
// - SystemExit for a call the interpreter's exit stopped (parallel.hpp,
//   StoppedAtExit): what the exit is, which a thread's excepthook ignores
//   (printing a traceback from a daemon thread as the interpreter finalizes
//   is itself a fatal error), and which `except Exception` doesn't swallow;
// - MemoryError for std::length_error, which a container throws only for a
//   size past its max_size, more than any memory holds. On 32 bits a vector
//   of more than PTRDIFF_MAX bytes is one, and nanobind would say ValueError
inline void install_fork_handler() {
    parallel::install_fork_handler();
    nb::register_exception_translator([](const std::exception_ptr& error, void* /*unused*/) {
        try {
            std::rethrow_exception(error);
        } catch (const parallel::StoppedAtExit&) {
            PyErr_SetNone(PyExc_SystemExit);
        } catch (const std::length_error& too_long) {
            PyErr_SetString(PyExc_MemoryError, too_long.what());
        }
    });
}

}  // namespace trimkit
