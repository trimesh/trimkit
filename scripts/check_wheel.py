"""
Check a built wheel:
- its metadata renders (twine)
- one stable-ABI extension, the package itself, within its size budget, and
  no other shared library
- nothing in it beyond the stable ABI (abi3audit)
- every license and the SBOM shipped

  python scripts/check_wheel.py dist/trimkit-*.whl

`uv run scripts static` runs it on the dev build's wheel, and cibuildwheel's
audit on every release wheel, with twine and abi3audit installed by hash.
"""

import stat
import subprocess
import sys
import zipfile

from backend import license_files
from builder import ROOT, load

# stable ABI only: `.abi3.so` on unix, and windows `.pyd` has no tag
EXTENSIONS = ("trimkit/__init__.abi3.so", "trimkit/__init__.pyd")
LIBRARIES = (".so", ".dylib", ".dll", ".pyd")


def tool(*args: str) -> list[str]:
    # a tool from this interpreter's environment: failures are reported, not raised
    result = subprocess.run(
        [sys.executable, "-m", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return [f"{args[0]}:\n{result.stdout}"] if result.returncode else []


def check(path: str) -> list[str]:
    wheel = zipfile.ZipFile(path)
    files = {i.filename: i.file_size for i in wheel.infolist()}
    errors = tool("twine", "check", "--strict", path)
    found = [f for f in EXTENSIONS if f in files]
    # bundled libraries are versioned, as `libstdc++-5d72f927.so.6.0.33`
    others = [f for f in files if f not in found and (f.endswith(LIBRARIES) or ".so." in f)]
    if len(found) != 1 or others:
        return [*errors, f"expected one stable-ABI extension and nothing else, found {found + others}"]
    # uncompressed bytes, from trimkit.toml's [extension] budget
    budget = load().budget
    if files[found[0]] > budget:
        errors.append(f"{found[0]}: {files[found[0]]:,} bytes is over its {budget:,} budget")
    # installed executable, which pip does only for an entry typed a regular file
    mode = wheel.getinfo(found[0]).external_attr >> 16
    if not (stat.S_ISREG(mode) and mode & 0o111):
        errors.append(f"{found[0]}: mode {mode:o} won't install executable")
    shipped = {f.split(".dist-info/licenses/", 1)[1] for f in files if ".dist-info/licenses/" in f}
    expected = {p.relative_to(ROOT).as_posix() for p in license_files()}
    errors += [f"license missing: {name}" for name in sorted(expected - shipped)]
    if not any(f.endswith(".dist-info/sboms/sbom.spdx.json") for f in files):
        errors.append("SBOM missing from .dist-info/sboms/")
    # the extension uses the stable ABI alone
    return errors + tool("abi3audit", "--strict", "--assume-minimum-abi3", "3.12", "--report", path)


if __name__ == "__main__":
    failed = False
    for wheel in sys.argv[1:]:
        for error in check(wheel):
            failed = True
            print(f"{wheel}: {error}")
    sys.exit(int(failed))
