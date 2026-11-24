"""
Options for the suite.

`--device-required`: a test that needs a Vulkan device computing in double
precision fails instead of skipping when there is none. The gate passes it:
its image has Mesa's CPU device, so a skip there would hide a broken device
path. Elsewhere (a wheel's CI, a machine without Vulkan) the tests skip and
say why.

Every Python a test starts dumps its stack on a crash (SIGSEGV, SIGABRT,
SIGBUS, SIGILL, SIGFPE) to the stderr the test captures and prints when it
fails, and so does a child it forks: pytest's own faulthandler covers only
pytest's process. A test still running after `faulthandler_timeout`
(pyproject.toml) dumps every thread's stack, so a hang in CI says where.

Nothing a test runs writes to the terminal: whatever reaches file
descriptor 1 or 2 during a test, from Python or from any library's C,
fails it. A library never prints to its caller's terminal (xatlas did,
src/xatlas.cpp), and its warnings are Python's, which pytest records.
"""

import ctypes
import os

import pytest

os.environ.setdefault("PYTHONFAULTHANDLER", "1")


def pytest_addoption(parser):
    parser.addoption(
        "--device-required",
        action="store_true",
        default=False,
        help="fail, rather than skip, the tests that need a Vulkan device",
    )


@pytest.fixture(scope="session")
def device_required(request):
    return request.config.getoption("--device-required")


# C's stdio buffers what a library prints into a pipe until the process
# exits, after any test could see it: flushed here first, where there is one
# C library to ask (a Windows extension's static runtime keeps its own)
_C = ctypes.CDLL(None) if os.name == "posix" else None


@pytest.fixture(autouse=True)
def writes_nothing(capfd):
    yield
    if _C is not None:
        _C.fflush(None)
    out, err = capfd.readouterr()
    assert not out and not err, f"written to the terminal: {out!r} {err!r}"
