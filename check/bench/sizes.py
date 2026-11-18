"""
How much disk the packages trimkit replaces take, installed on top of
`trimesh` + `numpy`, against trimkit's wheel, and trimesh's extras
installed with the bindings they take before Python 3.15 and with trimkit,
which they take from it. Prints two markdown tables.

`python bench/sizes.py [PYPROJECT]` reads the extras from trimesh's
pyproject.toml, by default a checkout beside this one. It runs on a Python
before 3.15, where both sets of bindings install.

Every row is a fresh `uv venv` holding the baseline, every row above it and
the row's package, with bytecode compiled as `pip install` leaves it, and
the size is every byte under its site-packages. Rows are ordered greedily:
each adds the package that adds the least given the rows above it, so a
dependency two packages share is charged to the smaller one.
"""

import datetime
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from pathlib import Path

# what trimesh installed for what trimkit does, and `coal`:
# the library `trimkit`'s `collision` vendors
REPLACED = (
    "svg.path",
    "rtree",
    "embreex",
    "vhacdx",
    "mapbox-earcut",
    "fast-simplification",
    "python-fcl",
    "scikit-image",
    "DracoPy",
    "xatlas",
    "coal",
)
BASELINE = ("trimesh", "numpy")
# each extra installed with every one before it, as users stack them
EXTRAS = ("easy", "recommend")
# the marker trimesh's extras split the bindings by: before it, after it
BEFORE, AFTER = "python_version<'3.15'", "python_version>='3.15'"
ENVIRONMENT = {**os.environ, "UV_NO_CONFIG": "1"}


def installed(requirements: tuple[str, ...]) -> tuple[int, dict[str, str]]:
    """
    Install `requirements` on top of the baseline into a fresh venv.

    Parameters
    ----------
    requirements
      Package names to install with the baseline.

    Returns
    ----------
    size
      Bytes of every file under the venv's site-packages.
    versions
      The version of every distribution installed, by lowercase name.
    """
    with tempfile.TemporaryDirectory() as directory:
        venv = Path(directory) / "venv"
        python = ("--python", str(venv))
        subprocess.run(
            ["uv", "venv", "--quiet", "--python", sys.executable, str(venv)],
            env=ENVIRONMENT,
            check=True,
        )
        subprocess.run(
            [
                *("uv", "pip", "install", "--quiet", "--compile-bytecode"),
                *(*python, *BASELINE, *requirements),
            ],
            env=ENVIRONMENT,
            check=True,
        )
        frozen = subprocess.run(
            ["uv", "pip", "freeze", *python],
            env=ENVIRONMENT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        site = next(venv.rglob("site-packages"))
        size = sum(path.stat().st_size for path in site.rglob("*") if path.is_file())
    pins = (line.split("==") for line in frozen.split())
    versions = {normalized(name): version for name, version in pins}
    return size, versions


def normalized(name: str) -> str:
    # the distribution name `uv pip freeze` prints, as PEP 503 compares them
    return re.sub(r"[-_.]+", "-", name).lower()


def mib(size: int) -> str:
    return f"{size / 2**20:,.1f}"


if sys.version_info >= (3, 15):
    sys.exit("run on a Python before 3.15, where both sets of bindings install")
if len(sys.argv) > 1:
    pyproject = Path(sys.argv[1])
else:
    pyproject = Path(__file__).resolve().parents[3] / "trimesh" / "pyproject.toml"
with pyproject.open("rb") as file:
    optional = tomllib.load(file)["project"]["optional-dependencies"]

base_size, base_versions = installed(())
print(
    f"`bench/sizes.py`, {datetime.date.today()}, CPython {sys.version.split()[0]} "
    f"`{sysconfig.get_platform()}`, a fresh `uv venv` per row with bytecode "
    f"compiled; the `trimesh` {base_versions['trimesh']} + `numpy` "
    f"{base_versions['numpy']} baseline is {mib(base_size)} MiB.\n"
)
print("| on top of `trimesh` + `numpy` | also brings | adds (MiB) | total (MiB) |")
print("|---|---|---|---|")

chosen, size, versions = (), base_size, base_versions
remaining = list(REPLACED)
while remaining:
    trials = {name: installed((*chosen, name)) for name in remaining}
    name = min(remaining, key=lambda candidate: trials[candidate][0])
    remaining.remove(name)
    added_size, added_versions = trials[name]
    brings = sorted(set(added_versions) - set(versions) - {normalized(name)})
    print(
        f"| `{name}` | {', '.join(f'`{other}`' for other in brings)} "
        f"| {mib(added_size - size)} | {mib(added_size - base_size)} |"
    )
    chosen, size, versions = (*chosen, name), added_size, added_versions

trimkit_size, trimkit_versions = installed(("trimkit",))
brings = sorted(set(trimkit_versions) - set(base_versions) - {"trimkit"})
print(
    f"| `trimkit` {trimkit_versions['trimkit']}, instead of all of the above "
    f"| {', '.join(f'`{other}`' for other in brings)} "
    f"| {mib(trimkit_size - base_size)} | {mib(trimkit_size - base_size)} |"
)

print("\n| `trimesh[...]` | old bindings (MiB) | with trimkit (MiB) | saved (MiB) |")
print("|---|---|---|---|")
stacked = ()
for extra in EXTRAS:
    stacked = (*stacked, *optional[extra])
    # before 3.15 this Python takes the old bindings' markers as written
    old_size, _ = installed(tuple(line for line in stacked if AFTER not in line))
    new_size, _ = installed(tuple(line.split(";")[0] for line in stacked if BEFORE not in line))
    names = ",".join(EXTRAS[: EXTRAS.index(extra) + 1])
    print(
        f"| `trimesh[{names}]` | {mib(old_size - base_size)} "
        f"| {mib(new_size - base_size)} | {mib(old_size - new_size)} |"
    )
