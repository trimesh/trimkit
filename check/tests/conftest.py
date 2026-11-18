"""
Options for the suite.

`--device-required`: a test that needs a Vulkan device computing in double
precision fails instead of skipping when there is none. The gate passes it:
its image has Mesa's CPU device, so a skip there would hide a broken device
path. Elsewhere (a wheel's CI, a machine without Vulkan) the tests skip and
say why.
"""

import pytest


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
