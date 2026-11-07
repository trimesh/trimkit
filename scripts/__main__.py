"""
The one way to build, test and check this package, locally and in CI alike:

  uv run scripts check       the gate: the strict build and tests, every
                             static check, ASan, TSan, and every fuzzer
  uv run scripts distro      Debian and Fedora packaging (--run: build,
                             test and lint them), when packaging changes
  uv run scripts generate    regenerate the stubs and the README API summary
  uv run scripts update      move every upstream and dependency to its newest
  uv run scripts format      format our C++ with the pinned clang-format
  uv run scripts wheel       this platform's release wheels

`check` runs every stage even after one fails, then prints a table of them
with their times and the command that reruns any that failed on its own. Every `cmd_*` below
is a command: the stages and their parts exist for debugging one failure,
and `fetch` and `outdated` re-vendor and audit trimkit.toml's upstreams;
none of these are otherwise documented.

CI only releases: a push to main runs `static`, `sdist` and `wheel` exactly
as they run here, then publishes, the one CI-only step, as it needs GitHub's
token for PyPI. `check` is the gate, run before pushing. Everything but `lock` and `wheel` re-runs
itself inside the image built from docker/Dockerfile, with the repository
mounted read-only at /src (writable for `generate`) and builds under
`build/docker/`, so it behaves the same on every machine and needs only
Docker and uv on the host.

Standard library only, every subprocess an argument list, and nothing read
from the environment. The image is the isolation: `docker run` starts every
in-image command from the image's own environment, never the host's.
"""

import argparse
import functools
import hashlib
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import api
import builder
import distro
import licenses

import vendor

ROOT = builder.ROOT
MANIFEST = builder.load()
# the Python every build but `tsan` uses, the oldest the stable ABI targets
PYTHON = "3.12"
# the free-threaded Python `sdist` also builds and tests on
FREE_THREADED_PYTHON = "3.15t"
# The free-threaded CPython `tsan` builds with ThreadSanitizer, checked by
# hash. Built here, under `docker run`, rather than in the image: TSan's
# runtime re-executes itself without address randomization, which docker
# build's seccomp profile refuses (its build tools crash, 2026-11-01)
CPYTHON = "3.14.3"
CPYTHON_SHA256 = "a97d5549e9ad81fe17159ed02c68774ad5d266c72f8d9a0b5a9c371fe85d902b"


def run(*args, env: dict | None = None, cwd: Path = ROOT, capture: bool = False) -> str:
    """
    Run one command, failing loudly. `env` adds to the inherited environment
    only for the child: sanitizer runtimes are configured that way.
    """
    print("+", " ".join(str(a) for a in args), flush=True)
    result = subprocess.run(
        [str(a) for a in args],
        cwd=cwd,
        check=True,
        env=None if env is None else {**os.environ, **env},
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
    )
    return result.stdout if capture else ""


def tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        sys.exit(f"`{name}` is not installed: is this running outside the image? See `uv run scripts --help`")
    return path


class Build:
    """
    One named build under `<root>/<name>/`: a wheel our backend builds from
    the tracked files (or an sdist) with `settings`, builder.Config's fields,
    installed into its own venv. Its ninja build directory persists, so a
    rebuild compiles only what changed.
    """

    def __init__(self, root: Path, name: str):
        self.dir = root / name
        self.venv = self.dir / "venv"
        self.python = self.venv / "bin" / "python"
        self.build_dir = self.dir / "build"
        self.wheel = self.dir / "wheel"
        # where the build compiles from, unless it is given an sdist
        self.source = self.dir / "source"

    def install(self, python: str, source: Path | None = None, **settings) -> "Build":
        run(tool("uv"), "venv", "-q", "--allow-existing", "--python", python, self.venv)
        # every dependency, the build requirements included, by hash from the
        # lock, then the package built without isolation against exactly those
        requirements = export(self.dir / "requirements.txt", "build", "test", "audit")
        run(tool("uv"), "pip", "sync", "-q", "--python", self.python, "--require-hashes", requirements)
        # the tracked files alone, so no loose file is ever compiled
        source = source or builder.export_tracked(self.source)
        options = {"build-dir": self.build_dir}
        options |= {k.replace("_", "-"): "true" if v is True else v for k, v in settings.items()}
        wheels = self.dir / "wheels"
        shutil.rmtree(wheels, ignore_errors=True)
        run(
            tool("uv"),
            "build",
            "-q",
            "--wheel",
            "--no-build-isolation",
            "--python",
            self.python,
            "--out-dir",
            wheels,
            *(f"-C{key}={value}" for key, value in options.items()),
            source,
        )
        (self.wheel,) = wheels.glob("*.whl")
        uv_pip = [tool("uv"), "pip", "install", "-q", "--python", self.python, "--no-deps"]
        run(*uv_pip, "--reinstall-package", "trimkit", self.wheel)
        return self

    def pytest(self, *args, env: dict | None = None, cwd: Path = ROOT) -> None:
        # sanitizer runs (any `env`) go without ASLR, see `no_aslr`, and
        # capture only Python's output: a sanitizer reports on the process's
        # own stderr and then exits, before pytest could show a capture of it
        sanitized = [*no_aslr(), self.python, "-m", "pytest", "--capture=sys"] if env else []
        command = sanitized or [self.python, "-m", "pytest"]
        run(*command, "-q", *args, env=env, cwd=cwd)


@functools.cache
def dev(root: Path) -> Build:
    """
    The strict release-configuration build the tests, analyzers, stubs and
    wheel audit use, built once per run.
    """
    return Build(root, "dev").install(PYTHON, strict=True)


@functools.cache
def compile_commands(root: Path) -> list[Path]:
    """
    How the dev build compiled each of our sources, for clang-tidy: ninja
    writes compile_commands.json from the build it ran. Returns our sources
    as that build saw them, in its export of the tracked files.
    """
    build = dev(root)
    commands = run(
        build.venv / "bin" / "ninja", "-C", build.build_dir, "-t", "compdb", "compile", capture=True
    )
    (build.build_dir / "compile_commands.json").write_text(commands)
    return [build.source / p.relative_to(ROOT) for p in MANIFEST.sources]


def no_aslr() -> list[str]:
    """
    Sanitizer runtimes reserve fixed shadow regions that a randomized mapping
    can occasionally land in on kernels with high mmap entropy, aborting the
    run at random: run them with address randomization off.
    """
    return [tool("setarch"), platform.machine(), "--addr-no-randomize"]


def asan_runtime() -> str:
    machine = platform.machine()
    return run(tool("clang"), f"-print-file-name=libclang_rt.asan-{machine}.so", capture=True).strip()


def export(path: Path, *groups: str) -> Path:
    """
    uv.lock's `groups` as a hash-pinned requirements file, for pip, which the
    cibuildwheel containers have rather than uv, and for `uv pip sync`.
    """
    named = [word for group in groups for word in ("--group", group)]
    run(
        tool("uv"), "export", "-q", "--frozen", "--no-default-groups", *named, "--no-emit-project", "-o", path
    )
    return path


# -- commands ----------------------------------------------------------------


def test_files() -> list[Path]:
    # the tracked test modules: an untracked scratch test never runs
    return [p for p in builder.tracked_under(ROOT / "tests") if p.name.startswith("test_")]


def cmd_test(args) -> None:
    dev(args.root).pytest(*test_files())


def cmd_asan(args) -> None:
    build = Build(args.root, "asan").install(PYTHON, sanitize="address,undefined")
    # wall-clock bounds measure the sanitizer, not the code
    build.pytest(
        "-m",
        "not timing",
        *test_files(),
        env={
            "LD_PRELOAD": asan_runtime(),
            "ASAN_OPTIONS": "detect_leaks=0",
            "PYTHONMALLOC": "malloc",
        },
    )


def tsan_python(root: Path) -> Path:
    """
    A free-threaded CPython built with ThreadSanitizer, once per build root:
    TSan only sees races through an instrumented interpreter.
    """
    prefix = root / f"cpython-{CPYTHON}-tsan"
    python = prefix / "bin" / "python3.14t"
    if python.exists():
        return python
    source = root / f"cpython-{CPYTHON}-src"
    shutil.rmtree(source, ignore_errors=True)
    source.mkdir(parents=True)
    url = f"https://www.python.org/ftp/python/{CPYTHON}/Python-{CPYTHON}.tar.xz"
    print("+ fetch", url, flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        tarball = response.read()
    if hashlib.sha256(tarball).hexdigest() != CPYTHON_SHA256:
        sys.exit(f"{url} does not match its pinned sha256")
    with tarfile.open(fileobj=io.BytesIO(tarball)) as tar:
        tar.extractall(source, filter="data")
    (tree,) = source.iterdir()
    configure = ["./configure", "-q", "--disable-gil", "--with-thread-sanitizer", f"--prefix={prefix}"]
    run(*configure, cwd=tree, env={"CC": "clang", "CXX": "clang++"})
    run(tool("make"), "-s", f"-j{os.cpu_count()}", cwd=tree)
    run(tool("make"), "-s", "install", cwd=tree)
    shutil.copy(tree / "Tools" / "tsan" / "suppressions_free_threading.txt", prefix)
    shutil.rmtree(source)
    return python


def cmd_tsan(args) -> None:
    python = tsan_python(args.root)
    build = Build(args.root, "tsan").install(str(python), sanitize="thread")
    # only CPython's own suppressions for its free-threading internals
    suppressions = python.parent.parent / "suppressions_free_threading.txt"
    # a forked child may start threads (the pool's workers, when its query is
    # long enough under TSan's instrumentation), which TSan can't model and by
    # default kills the child for; glibc supports it, and the fork test needs it
    options = f"suppressions={suppressions} halt_on_error=1 die_after_fork=0"
    build.pytest("-s", "tests/test_threads.py", env={"TSAN_OPTIONS": options})


def harness(directory: Path, corpus: Path, findings: Path, name: str, options: list[str]) -> str:
    """
    Run one harness and return the lines worth reporting: it finding a crash
    is reported, not raised, so every harness gets its turn.
    """
    command = [
        *no_aslr(),
        directory / f"fuzz_{name}",
        *options,
        f"-artifact_prefix={findings}/",
        corpus / name,
    ]
    print("+", " ".join(str(c) for c in command), flush=True)
    result = subprocess.run(
        [str(c) for c in command],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    keep = re.compile(r"stat::number_of_executed_units|ERROR|SUMMARY|crash|timeout:")
    lines = [line for line in result.stdout.splitlines() if keep.search(line)][-2:]
    status = f"\n{name}: exit {result.returncode}" if result.returncode else ""
    return "\n".join(f"{name}: {line}" for line in lines) + status


def cmd_fuzz(args) -> None:
    # the harnesses and seed writers, built beside the sanitized extension
    directory = Build(args.root, "fuzz-env").install(PYTHON, fuzz=True).build_dir
    corpus = args.root / "fuzz" / "corpus"
    findings = args.root / "fuzz" / "findings"
    for name in MANIFEST.harnesses:
        (corpus / name).mkdir(parents=True, exist_ok=True)
    findings.mkdir(parents=True, exist_ok=True)
    for name, part in MANIFEST.parts.items():
        if part.seeds:
            run(directory / part.seeds, corpus / name)
    # a seed that exercises every svg command
    (corpus / "svgpath" / "0").write_text(
        "M0 0 h5 v5 l-5-5 Z m1 1 C1 2 3 4 5 6 s1 1 2 2 q1 1 2 2 t3 3 a5 5 0 1 0 10 0\n"
    )
    failed = False
    for name in MANIFEST.harnesses:
        options = [
            f"-max_total_time={args.seconds}",
            "-timeout=10",
            "-rss_limit_mb=4096",
            "-print_final_stats=1",
        ]
        if name == "draco":
            # draco's memory blowups are decompression bombs, unbounded by design
            # (draco_decode's Notes), not findings here
            options += [f"-fork={CORES}", "-ignore_ooms=1"]
        report = harness(directory, corpus, findings, name, options)
        print(report, flush=True)
        # libFuzzer's fork mode exits with its out-of-memory code, 71, when its
        # last job ran out of memory, even with those ignored: a crash is 77
        ignored_oom = name == "draco" and report.endswith(f"{name}: exit 71")
        failed |= "exit " in report and not ignored_oom
    if failed:
        sys.exit(f"fuzzing found something: see {findings}")


def cmd_atheris(args) -> None:
    # the extension with ASan and libFuzzer's coverage hooks, driven by Atheris
    # through Python: exercises our argument conversion and validation. Address
    # only: Atheris ships ASan and libFuzzer, not the UBSan runtime
    build = Build(args.root, "atheris").install(PYTHON, sanitize="fuzzer-no-link,address")
    locked = export(build.dir / "atheris.txt", "fuzz")
    run(tool("uv"), "pip", "install", "-q", "--python", build.python, "--require-hashes", "-r", locked)
    path = run(build.python, "-c", "import atheris; print(atheris.path())", capture=True).strip()
    findings = args.root / "fuzz" / "findings" / "atheris"
    findings.mkdir(parents=True, exist_ok=True)
    # with address randomization, unlike clang's own sanitizer runs: Atheris's
    # ASan runtime is a different build, and without randomization its
    # mappings collide often enough to crash a third of plain `import numpy`s
    run(
        build.python,
        ROOT / "fuzz" / "atheris_bindings.py",
        f"-max_total_time={args.seconds}",
        "-timeout=20",
        f"-artifact_prefix={findings}/",
        env={
            "LD_PRELOAD": f"{path}/asan_with_fuzzer.so",
            "ASAN_OPTIONS": "detect_leaks=0",
            "PYTHONMALLOC": "malloc",
        },
    )


def cmd_tidy(args) -> None:
    # our sources, and the headers in our own directories, as the dev build saw them
    sources = compile_commands(args.root)
    ours = [re.escape(str(dev(args.root).source / d.relative_to(ROOT))) for d in MANIFEST.directories]
    header_filter = f"--header-filter=^({'|'.join(ours)})/"
    run(tool("clang-tidy"), "--quiet", header_filter, "-p", dev(args.root).build_dir, *sources)


def cmd_cppcheck(args) -> None:
    # our sources and our own headers (clang's analyzer runs in `tidy`)
    own = [d for d in MANIFEST.directories if d != ROOT / "src"]
    headers = sorted(p for d in own for p in d.glob("*.hpp"))
    run(
        tool("cppcheck"),
        "--quiet",
        "--enable=warning,performance,portability",
        "--inline-suppr",
        "--std=c++17",
        "--error-exitcode=1",
        # every branch, not cppcheck's default sample of them, one line per finding
        "--check-level=exhaustive",
        "--template={file}:{line}: {severity}: {message} [{id}]",
        *(f"-I{d}" for d in own),
        *MANIFEST.sources,
        *headers,
    )


def cpp_files() -> list[Path]:
    # all of our C++, whichever directory it lives in
    directories = [*MANIFEST.directories, ROOT / "fuzz"]
    return [p for d in directories for p in builder.tracked_under(d) if p.suffix in (".cpp", ".hpp")]


def cmd_format(args) -> None:
    # our C++ as .clang-format says, with the image's clang-format
    run(tool("clang-format"), "-i", *cpp_files())


def cmd_clang_format(args) -> None:
    run(tool("clang-format"), "--dry-run", "-Werror", *cpp_files())


def cmd_ruff(args) -> None:
    # tracked Python only, still honoring pyproject's excludes
    python = [p for p in builder.tracked() if p.suffix in (".py", ".pyi")]
    run(tool("ruff"), "check", "--no-cache", "--force-exclude", *sorted(python))
    run(tool("ruff"), "format", "--no-cache", "--check", "--force-exclude", *sorted(python))


def cmd_actions(args) -> None:
    github = builder.tracked_under(ROOT / ".github")
    run(tool("zizmor"), "--offline", "--pedantic", *github)
    run(tool("actionlint"), *[p for p in github if p.parent.name == "workflows"])


def stubs(root: Path, write: bool) -> None:
    """
    Generate the stubs from the built extension, and the README's API summary
    from the stubs: write them, or fail if they drifted.
    """
    build = dev(root)
    package = ROOT / "src" / "trimkit"
    with tempfile.TemporaryDirectory() as fresh:
        # the package and its submodule: trimkit/__init__.pyi, trimkit/exchange.pyi
        stubgen = builder.NANOBIND / "src" / "stubgen.py"
        run(build.python, stubgen, "-q", "-r", "-m", "trimkit", "-O", fresh)
        generated = {p.name: p.read_text() for p in (Path(fresh) / "trimkit").glob("*.pyi")}
        existing = {p.name: p.read_text() for p in package.glob("*.pyi")}
        if write:
            for name in existing.keys() - generated.keys():
                (package / name).unlink()
            for name, text in generated.items():
                (package / name).write_text(text)
        elif existing != generated:
            sys.exit("the stubs drifted from the extension; run `uv run scripts generate`")
    (api.write if write else api.check)()


def cmd_stubs(args) -> None:
    stubs(args.root, write=False)


@functools.cache
def sdist(root: Path) -> Path:
    """
    The sdist, built once per run with the hash-locked build backend, and
    unpacked next to it.
    """
    out = root / "sdist"
    shutil.rmtree(out, ignore_errors=True)
    python = dev(root).python
    # from the tracked files alone, so nothing untracked can end up in it
    source = builder.export_tracked(root / "sdist-source")
    run(tool("uv"), "build", "--sdist", "--no-build-isolation", "--python", python, "--out-dir", out, source)
    (archive,) = out.glob("*.tar.gz")
    with tarfile.open(archive) as tar:
        tar.extractall(out / "unpacked", filter="data")
    return archive


def cmd_sdist(args) -> None:
    # the sdist alone must build and pass its own tests
    archive = sdist(args.root)
    (tree,) = (archive.parent / "unpacked").iterdir()
    build = Build(args.root, "sdist-test").install(PYTHON, source=archive, strict=True)
    build.pytest(tree / "tests", cwd=tree)
    # and on free-threaded Python, which gets no wheels: this is how it installs.
    # Wall-clock bounds aside, as they measure the GIL it doesn't have
    free = Build(args.root, "sdist-free-threaded").install(FREE_THREADED_PYTHON, source=archive, strict=True)
    free.pytest("-m", "not timing", tree / "tests", cwd=tree)


def cmd_licenses(args) -> None:
    # every component permissive, its license shipped, its SBOM entry whole
    licenses.check()


# what only src/numpy.hpp may do: touch nanobind's ndarray directly, or the
# checks behind its `read*` functions, so no binding can skip one
DOOR_ONLY = re.compile(r"nb::ndarray|try_cast|check_bounded|check_coordinates|trimkit::detail::")


def check_door() -> None:
    # every C++ file of ours but the door itself
    sources = [
        p
        for p in builder.tracked_under(ROOT / "src")
        if p.suffix in (".cpp", ".hpp") and p.name != "numpy.hpp"
    ]
    found = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in sources
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if DOOR_ONLY.search(line)
    ]
    if found:
        sys.exit("arrays must go through src/numpy.hpp:\n  " + "\n  ".join(found))
    # threads: only common.hpp reaches the pool, through in_parallel and gather_in_parallel
    found = [
        f"{path.relative_to(ROOT)}: {needle}"
        for path in sources
        if path.name not in ("common.hpp", "parallel.hpp")
        for needle in ("parallel.hpp", "parallel::")
        if needle in path.read_text()
    ]
    if found:
        sys.exit("threads must go through in_parallel in src/common.hpp:\n  " + "\n  ".join(found))
    print(f"ok: {len(sources)} sources use only src/numpy.hpp for arrays and src/common.hpp for threads")


def cmd_audit(args) -> None:
    # the dev build's wheel, audited as every release wheel is
    run(dev(args.root).python, ROOT / "scripts" / "check_wheel.py", dev(args.root).wheel)


def cmd_static(args) -> None:
    vendor.check()
    builder.check()
    check_door()
    # the lock matches pyproject.toml
    run(tool("uv"), "lock", "--check")
    for command in (
        cmd_clang_format,
        cmd_ruff,
        cmd_actions,
        cmd_licenses,
        # the sdist builds; building and testing from it alone is `sdist`,
        # which release CI runs, as it's a whole build and suite again
        lambda args: sdist(args.root),
        cmd_stubs,
        cmd_audit,
        cmd_tidy,
        cmd_cppcheck,
    ):
        command(args)


def cmd_wheel(args) -> None:
    # cibuildwheel by hash from uv.lock, configured by pyproject.toml, on the
    # tracked files alone (its containers copy whatever directory they build)
    # plus the requirements its hooks install by hash with pip
    source = builder.export_tracked(args.root / "wheel-source")
    # what the build and its tests install inside each image, and what the
    # audit installs on the host: apart, as the audit tools need wheels (such
    # as cryptography's) that not every image's platform has
    export(source / "requirements.txt", "build", "test")
    export(source / "audit.txt", "audit")
    run(
        tool("uv"),
        "run",
        "--project",
        ROOT,
        "--isolated",
        "--frozen",
        "--no-default-groups",
        "--only-group",
        "wheel",
        "cibuildwheel",
        "--output-dir",
        ROOT / "wheelhouse",
        cwd=source,
    )


def cmd_distro(args) -> None:
    # Debian and Fedora packaging from the sdist into build/distro/; with
    # --run, each built against the system's upstreams, tested and linted in
    # its distribution's own image
    source = builder.export_tracked(args.root / "distro-source")
    out = args.root / "distro-sdist"
    shutil.rmtree(out, ignore_errors=True)
    run(
        tool("uv"),
        "build",
        "-q",
        "--sdist",
        "--no-build-isolation",
        "--python",
        sys.executable,
        "--out-dir",
        out,
        source,
    )
    (archive,) = out.glob("*.tar.gz")
    distro.write(archive, args.root / "distro")
    if args.run:
        distro.run(
            args.root / "distro",
            [tool("docker"), "run", "--rm", *docker_cores()],
            f"{os.getuid()}:{os.getgid()}",
        )


def cmd_fetch(args) -> None:
    # re-derive the named upstreams (default all) from their pins, or --latest
    vendor.fetch(args.names, args.latest)


def cmd_outdated(args) -> None:
    vendor.outdated()


def cmd_lock(args) -> None:
    # every dependency to its newest release past the lock's one-week cooldown
    run(tool("uv"), "lock", "--upgrade")


def cmd_update(args) -> None:
    # every upstream to its ref's newest commit and every dependency to its
    # newest release, the stubs that quote those commits, then what moved: run
    # `check` on it. A patch that no longer applies stops here, naming it, and
    # `git restore vendor` undoes it
    vendor.fetch([], latest=True)
    cmd_lock(args)
    stubs(args.root, write=True)
    run(tool("git"), "-C", ROOT, "diff", "--stat", "--", "trimkit.toml", "uv.lock")


def cmd_generate(args) -> None:
    # everything derived from the sources: the stubs from the module the dev
    # build makes, and the README's API summary from the stubs
    stubs(args.root, write=True)


# what `check` runs, in order: every stage runs even after one fails
# the gate, which should take minutes after a small change: the distribution
# packages (`distro --run`, two more builds and suites) are checked when the
# packaging changes, and the sdist's own build (`sdist`) in release CI
STAGES = ["test", "static", "asan", "tsan", "fuzz", "atheris"]


def cmd_check(args) -> None:
    results = []
    for stage in STAGES:
        command = [sys.executable, str(ROOT / "scripts"), "--root", str(args.root), stage]
        if stage in ("fuzz", "atheris"):
            command += ["--seconds", str(args.fuzz_seconds)]
        print(f"\n==== {stage} ====", flush=True)
        tic = time.monotonic()
        code = subprocess.run(command, cwd=ROOT, check=False).returncode
        results.append((stage, code, time.monotonic() - tic))
    print("\n==== check ====")
    for stage, code, seconds in results:
        print(f"  {'pass' if code == 0 else 'FAIL'}  {stage:<8} {seconds / 60:5.1f} min")
    print(f"        total    {sum(s for _, _, s in results) / 60:5.1f} min")
    failed = [stage for stage, code, _ in results if code != 0]
    if failed:
        rerun = "\n".join(f"  uv run scripts {stage}" for stage in failed)
        sys.exit(f"check failed; rerun a stage on its own with:\n{rerun}")


IMAGE = "trimkit-dev"
# every container of ours gets this many cores, the last ones, so a check
# never takes over a shared machine: by affinity rather than a quota, so the
# thread pool (which counts the cores it may run on) sizes itself to them
CORES = 2


def docker_cores() -> list[str]:
    total = os.cpu_count() or 1
    first = max(0, total - CORES)
    return ["--cpuset-cpus", f"{first}-{total - 1}"]


# on the host: `check` dispatches stages that each enter the image, `lock`
# needs only uv, `wheel` and `distro` run their own containers, `image`
# builds it, and `outdated` only asks the upstreams
ON_HOST = {"check", "lock", "wheel", "image", "outdated", "distro"}
SANITIZED = {"asan", "tsan", "fuzz", "atheris"}
WRITES_SOURCES = {"generate", "fetch", "update", "format"}


def cmd_image(args) -> None:
    # docker's layer cache makes this quick unless the Dockerfile or lock changed
    run(tool("docker"), "build", "-q", "-f", ROOT / "docker" / "Dockerfile", "-t", IMAGE, ROOT, capture=True)


def in_image(args) -> None:
    """
    Run this same command inside the image, as the caller's uid so files it
    writes under build/ stay theirs, with the sources read-only.
    """
    cmd_image(args)
    forwarded = [args.command]
    if hasattr(args, "seconds"):
        forwarded += ["--seconds", str(args.seconds)]
    if args.command == "fetch":
        forwarded += [*args.names, *(["--latest"] if args.latest else [])]
    builds = ROOT / "build" / "docker"
    builds.mkdir(parents=True, exist_ok=True)
    access = "rw" if args.command in WRITES_SOURCES else "ro"
    # Sanitizer runs turn ASLR off through personality(2) (see `no_aslr`), which
    # docker's default seccomp profile refuses: only those runs drop seccomp,
    # still unprivileged, as the caller's uid, with no added capabilities
    seccomp = ["--security-opt", "seccomp=unconfined"] if args.command in SANITIZED else []
    docker = [
        tool("docker"),
        "run",
        "--rm",
        "--init",
        *docker_cores(),
        *seccomp,
        # Docker's default open-file limit is about 2**31, and sanitizers
        # close every descriptor up to it before starting llvm-symbolizer:
        # each symbolized line, every NEW_FUNC libFuzzer prints included,
        # then took seconds to minutes, and fuzzers ran a few dozen inputs
        "--ulimit",
        "nofile=65536:65536",
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/src:{access}",
        "-v",
        f"{builds}:/src/build/docker:rw",
        "-w",
        "/src",
    ]
    inside = ["python3", "/src/scripts", "--inside", "--root", "/src/build/docker", *forwarded]
    run(*docker, IMAGE, *inside)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT / "build", help="where builds and venvs go")
    # set when already running inside the image
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    # only the documented commands show in the usage and help
    documented = {
        "check": "the gate: build, tests, static checks, sanitizers and fuzzers",
        "generate": "regenerate the stubs and the README API summary",
        "update": "move every upstream and dependency to its newest",
        "format": "format our C++ with the pinned clang-format",
        "wheel": "this platform's release wheels",
    }
    sub = parser.add_subparsers(dest="command", required=True, metavar="{" + ",".join(documented) + "}")
    for name, function in sorted(globals().items()):
        if name.startswith("cmd_"):
            command = sub.add_parser(
                name[4:], **({"help": documented[name[4:]]} if name[4:] in documented else {})
            )
            command.set_defaults(function=function)
            if name in ("cmd_fuzz", "cmd_atheris"):
                command.add_argument("--seconds", type=int, default=60)
            if name == "cmd_fetch":
                command.add_argument("names", nargs="*")
                command.add_argument("--latest", action="store_true", help="move each pin to its ref's HEAD")
            if name == "cmd_distro":
                command.add_argument(
                    "--run", action="store_true", help="build, test and lint them in containers"
                )
            if name == "cmd_check":
                command.add_argument("--fuzz-seconds", type=int, default=10, help="seconds per fuzzer")
    # `uv run scripts -- static` reads the same as `uv run scripts static`
    argv = sys.argv[1:]
    args = parser.parse_args(argv[1:] if argv[:1] == ["--"] else argv)
    args.root = args.root.resolve()
    try:
        if args.command not in ON_HOST and not args.inside:
            in_image(args)
        else:
            args.function(args)
    except subprocess.CalledProcessError as error:
        # a captured command's output is the only record of why it failed
        if error.output:
            print(error.output, file=sys.stderr)
        sys.exit(f"failed ({error.returncode}): {' '.join(str(a) for a in error.cmd[:6])}")


if __name__ == "__main__":
    main()
