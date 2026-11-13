"""
The PEP 517 build backend (pyproject.toml's `build-backend`): wheels are
built by writing scripts/builder.py's plan and running ninja on it, sdists
are the source tree as it stands. Config settings are builder.Config's
fields (`-Cstrict=true`, `-Cbuild-dir=...`); anything else is an error.

Standard library only, beside ninja: the build requirement's, pinned, or for
a distribution building without it, the `ninja` on PATH.
"""

import base64
import gzip
import hashlib
import io
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path

import builder
import licenses

ROOT = builder.ROOT
PACKAGE = ROOT / "src" / "trimkit"
# the package's typing: its stubs and marker (the package itself is the extension)
PACKAGE_FILES = ("*.pyi", "py.typed")
# where wheels build unless `build-dir` says otherwise
DEFAULT_BUILD_DIR = ROOT / "build" / "wheel"
# left out of sdists: CI configuration, meaningless outside the repository,
# and build output
NOT_SHIPPED = (".github", "build")
# every archive entry gets this time, so the same inputs give the same bytes
EPOCH = (1980, 1, 1, 0, 0, 0)


def project() -> dict:
    return builder.pyproject()["project"]


def license_files() -> list[Path]:
    return sorted({p for pattern in project()["license-files"] for p in ROOT.glob(pattern) if p.is_file()})


def metadata() -> str:
    """
    Core metadata 2.4 from pyproject.toml's [project]: every key it has is
    one this knows, so nothing there is silently dropped.
    """
    meta = project()
    headers = [("Metadata-Version", "2.4"), ("Name", meta["name"]), ("Version", meta["version"])]
    render = {
        "name": lambda _: [],
        "version": lambda _: [],
        "description": lambda value: [("Summary", value)],
        "readme": lambda _: [("Description-Content-Type", "text/markdown")],
        "requires-python": lambda value: [("Requires-Python", value)],
        "license": lambda value: [("License-Expression", value)],
        "license-files": lambda _: [
            ("License-File", p.relative_to(ROOT).as_posix()) for p in license_files()
        ],
        "authors": lambda value: [("Author-email", f"{a['name']} <{a['email']}>") for a in value],
        "dependencies": lambda value: [("Requires-Dist", d) for d in value],
        "classifiers": lambda value: [("Classifier", c) for c in value],
        "urls": lambda value: [("Project-URL", f"{label}, {url}") for label, url in value.items()],
    }
    unknown = sorted(set(meta) - set(render))
    if unknown:
        sys.exit(f"pyproject.toml [project] keys the backend doesn't write: {unknown}")
    for key, value in meta.items():
        headers += render[key](value)
    readme = (ROOT / meta["readme"]).read_text()
    return "".join(f"{key}: {value}\n" for key, value in headers) + "\n" + readme


def tag(python: builder.Python, platform: builder.Platform) -> str:
    version = f"{sys.version_info[0]}{sys.version_info[1]}"
    interpreter = f"cp{version}-cp{version}t" if python.free_threaded else "cp312-abi3"
    if platform.os == "macos":
        # the deployment target the build used, not the interpreter's own
        machine = "arm64" if platform.machine == "aarch64" else "x86_64"
        system = f"macosx_{builder.macos_deployment_target().replace('.', '_')}_{machine}"
    elif platform.os == "linux":
        # the machine built for, not the kernel's: a 32-bit interpreter on a
        # 64-bit kernel builds i686 (which sysconfig reports as x86_64)
        system = f"linux_{platform.machine}"
    else:
        system = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"{interpreter}-{system}"


def write_wheel(wheel: Path, files: dict[str, bytes], record: str) -> None:
    """
    A wheel of `files`, by archive path, with the RECORD of them at `record`.
    """
    rows = []
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, data in sorted(files.items()):
            info = zipfile.ZipInfo(path, EPOCH)
            # shared libraries executable, as installers expect: pip keeps
            # the bits only of an entry typed as a regular file, and Fedora's
            # find-debuginfo splits only executable files
            executable = path.endswith((".so", ".dylib", ".dll", ".pyd"))
            info.external_attr = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            rows.append(f"{path},sha256={digest},{len(data)}")
        rows.append(f"{record},,")
        info = zipfile.ZipInfo(record, EPOCH)
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        archive.writestr(info, "\n".join(rows) + "\n")


def ninja_executable() -> Path:
    """
    The pinned build requirement's ninja, which every build of ours installs
    by hash; without it (a distribution's own ninja package), the one on PATH.
    Either holds the plan to its `ninja_required_version`.
    """
    try:
        import ninja
    except ImportError:
        found = shutil.which("ninja")
        if found is None:
            sys.exit("no ninja: install the `ninja` build requirement, or a ninja on PATH")
        return Path(found)
    return Path(ninja.BIN_DIR) / ("ninja.exe" if sys.platform == "win32" else "ninja")


def build(config: builder.Config) -> builder.Target:
    """
    Write the plan and run ninja on it: the extension, and in fuzz builds the
    harnesses, all in `config.build_dir`.
    """
    target = builder.Target.current(config)
    builder.write(target)
    # every failure in one run, rather than the first few
    subprocess.run([str(ninja_executable()), "-k", "0", "-C", str(config.build_dir)], check=True)
    return target


# -- PEP 517 --------------------------------------------------------------------


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    config = builder.Config.from_settings(config_settings, DEFAULT_BUILD_DIR)
    target = build(config)
    meta = project()
    wheel_tag = tag(target.python, target.platform)
    dist_info = f"{meta['name']}-{meta['version']}.dist-info"
    files = {f"trimkit/{p.name}": p.read_bytes() for pattern in PACKAGE_FILES for p in PACKAGE.glob(pattern)}
    extension = builder.extension_file(target.python)
    files[f"trimkit/{extension}"] = (config.build_dir / extension).read_bytes()
    files[f"{dist_info}/METADATA"] = metadata().encode()
    wheel = {"Wheel-Version": "1.0", "Generator": "trimkit scripts/backend.py", "Root-Is-Purelib": "false"}
    files[f"{dist_info}/WHEEL"] = "".join(
        f"{k}: {v}\n" for k, v in {**wheel, "Tag": wheel_tag}.items()
    ).encode()
    files |= {
        f"{dist_info}/licenses/{p.relative_to(ROOT).as_posix()}": p.read_bytes() for p in license_files()
    }
    # the SBOM of everything compiled in, as the wheel's PEP 770 SBOM: not
    # for a build against system copies, which it would describe wrongly
    if not target.system:
        files[f"{dist_info}/sboms/sbom.spdx.json"] = licenses.sbom_text().encode()
    filename = f"{meta['name']}-{meta['version']}-{wheel_tag}.whl"
    write_wheel(Path(wheel_directory) / filename, files, f"{dist_info}/RECORD")
    return filename


def build_sdist(sdist_directory, config_settings=None) -> str:
    """
    In a git checkout the tracked files, so its loose files never ship, and
    otherwise (an export, or an unpacked sdist) every file there is.
    """
    if config_settings:
        sys.exit(f"sdists take no config settings, not {config_settings}")
    meta = project()
    base = f"{meta['name']}-{meta['version']}"
    filename = f"{base}.tar.gz"
    candidates = builder.tracked() if (ROOT / ".git").exists() else ROOT.rglob("*")
    files = sorted(
        p
        for p in candidates
        if p.is_file() and p.relative_to(ROOT).parts[0] not in NOT_SHIPPED and "__pycache__" not in p.parts
    )
    entries = {f"{base}/{p.relative_to(ROOT).as_posix()}": p.read_bytes() for p in files}
    entries[f"{base}/PKG-INFO"] = metadata().encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, data in sorted(entries.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    with gzip.GzipFile(Path(sdist_directory) / filename, "wb", mtime=0) as archive:
        archive.write(buffer.getvalue())
    return filename
