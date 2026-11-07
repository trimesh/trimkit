"""
The first stage of a profile-guided build, run by cibuildwheel as
`before-build` on every platform but 32-bit Windows (pyproject.toml), in the
build environment (`sys.executable`):

1. install the build requirements and test dependencies from
   requirements.txt, by hash
2. build an instrumented copy (`pgo=generate`) into the wheel's own build
   directory and run the test suite, but for its wall-clock bounds, which
   writes the profiles there
3. merge them with the build's own llvm-profdata, found and checked as
   every tool is (builder.Tools), into the one file `pgo=use` reads, and
   uninstall the instrumented copy, so the wheel is then built in that same
   directory (pyproject.toml) and tested from a clean environment

The profiles only steer the optimizer: they can't change what the code
computes, and the finished wheel is tested in full on its own afterwards.

Standard library only: the manylinux and musllinux images have pip, not uv.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from backend import DEFAULT_BUILD_DIR
from builder import ROOT, Config, Platform, Tools, load


def run(*args: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(list(args), cwd=cwd, check=True)


def main() -> None:
    config = Config(DEFAULT_BUILD_DIR, pgo="use")
    tools = Tools.current(Platform.current(), load())
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(*pip, "--require-hashes", "--no-deps", "-r", str(ROOT / "requirements.txt"))
    shutil.rmtree(config.profiles, ignore_errors=True)
    run(*pip, "--no-build-isolation", "--no-deps", "-C", "pgo=generate", ".")
    # from elsewhere, so the tests import the installed copy, not `src/`
    with tempfile.TemporaryDirectory() as elsewhere:
        run(
            sys.executable, "-m", "pytest", "-q", "-m", "not timing", str(ROOT / "tests"), cwd=Path(elsewhere)
        )
    raw = sorted(str(p) for p in config.profiles.glob("*.profraw"))
    if not raw:
        sys.exit(f"the training wrote no profiles to {config.profiles}")
    run(str(tools.profiler), "merge", "-o", str(config.profile), *raw)
    run(sys.executable, "-m", "pip", "uninstall", "-q", "-y", "trimkit")


if __name__ == "__main__":
    main()
