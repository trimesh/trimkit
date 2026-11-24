"""
The one way to build, test and check this package, locally and in CI alike:

  uv run scripts check       the gate: lint, test, static, ASan, TSan, fuzz,
                             bench --quick, genmc and model
  uv run scripts lint        the fast deterministic checks, nothing compiled:
                             vendor, doors, lock, formatting, ruff, zizmor,
                             licenses, shaders and cppcheck
  uv run scripts test        the strict build and the tests, in 2 GB, on the
                             last 2 cores (--cores N, or all)
  uv run scripts static      the checks that need a build: clang-tidy, the
                             sdist, stubs and the wheel audit
  uv run scripts fuzz        every fuzzer, native then Atheris (--seconds each)
  uv run scripts genmc       model check src/parallel.hpp itself with GenMC,
                             and that each bug planted in it and each memory
                             order weakened is caught
  uv run scripts model       check the pool's protocol (check/verify/tla) with TLC,
                             that each mutation of it is caught, and that
                             traced runs of the code are ones it allows
  uv run scripts bench       the pool against hand-written pools on 8 cores,
                             144 cells (--quick: the short calls, in seconds)
  uv run scripts distro      Debian and Fedora packaging (--run: build,
                             test and lint them, --machine i686|s390x on
                             another machine's images), when packaging changes
  uv run scripts generate    regenerate the shaders' SPIR-V, the stubs, the
                             README API summary and check/tests/golden.json
  uv run scripts patch       write vendor/patches/ from the blocks in vendor/
  uv run scripts update      move every upstream and dependency to its newest
  uv run scripts format      format our C++ with the pinned clang-format
  uv run scripts wheel       this platform's release wheels

`check` runs every stage even after one fails, then prints a table of them
with their times and the command that reruns any that failed on its own. Every `cmd_*` below
is a command: the stages and their parts exist for debugging one failure,
and `fetch` and `outdated` re-vendor and audit trimkit.toml's upstreams;
none of these are otherwise documented.

CI runs these same commands. A push to main runs `lint`, `static`, `sdist`
and `wheel` exactly as they run here, then publishes, the one CI-only step,
as it needs GitHub's token for PyPI; when main has moved, CI runs `fuzz` for
longer once a week and `distro --run --machine s390x` once a month. `check` is the gate, run
before pushing. Everything but what ON_HOST names re-runs itself inside the
image built from scripts/docker/Dockerfile, with the repository mounted
read-only at /src (writable for `generate`) and builds under `build/docker/`,
so it behaves the same on every machine and needs only Docker and uv on the
host.

Standard library only, every subprocess an argument list, and nothing read
from the environment. The image is the isolation: `docker run` starts every
in-image command from the image's own environment, never the host's.
"""

import argparse
import ast
import concurrent.futures
import doctest
import functools
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import xml.etree.ElementTree
from pathlib import Path
from typing import NamedTuple

import api
import builder
import distro
import licenses

import vendor

ROOT = builder.ROOT
CHECK = builder.CHECK
DOCKER = builder.DOCKER
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
    uv.lock's `groups` as a hash-pinned requirements file, for the `uv pip`
    of cibuildwheel's hooks and for `uv pip sync`.
    """
    named = [word for group in groups for word in ("--group", group)]
    run(
        tool("uv"), "export", "-q", "--frozen", "--no-default-groups", *named, "--no-emit-project", "-o", path
    )
    return path


# -- commands ----------------------------------------------------------------


def test_files() -> list[Path]:
    # the tracked test modules: an untracked scratch test never runs
    return [p for p in builder.tracked_under(CHECK / "tests") if p.name.startswith("test_")]


def cmd_build(args) -> None:
    # the dev build alone: `in_image` runs it before `test`, which has less
    # memory, and with --planted the builds of the bugs `test` plants too
    dev(args.root)
    if args.planted:
        planted_wheels(args.root)


def cmd_test(args) -> None:
    # in a container with a memory limit (`in_image`): a suite that holds on to
    # too much fails here, not one push later on a platform with less. Then the
    # bombs, where the kernel kills what touches too much
    build = dev(args.root)
    # the image has a Vulkan device (Mesa's CPU one), so the device tests run
    build.pytest("--device-required", *test_files())
    run(build.python, CHECK / "tests" / "memory_limit.py", cwd=CHECK / "tests")
    # decided once, on the host, by the cores asked for (`in_image`)
    cores = len(os.sched_getaffinity(0))
    if not args.planted:
        print(f"planted bugs: not tested on {cores} cores, the pool shows them on {PLANTED_TEST_CORES}")
    elif cores < PLANTED_TEST_CORES:
        sys.exit(f"planted bugs asked for on {cores} cores, where the pool can't show them")
    elif not planted_bugs_tested(args.root):
        sys.exit("the suite passed a build with a planted bug (CODE_MUTANTS)")


# the cores the suite needs to fail on every planted bug: on two the pool has
# one worker, which no wake leaves behind, so an idle respawn can't show
PLANTED_TEST_CORES = 4
# how long one planted bug's suite may run: the whole suite takes minutes
PLANTED_TEST_SECONDS = 3600


def planted_wheels(root: Path) -> dict[str, Path]:
    """
    A wheel for each bug planted for the suite (CODE_MUTANTS, "test"), by
    name, built one at a time in one build directory, so each compiles only
    what its bug changed: every file a bug edits is written fresh first, and
    again once its bug is out, so no build keeps an object compiled with a
    bug in it. Built outside `test`'s memory limit, as the dev build is.
    """
    planted = code_mutants("test")
    build = Build(root, "planted")
    kept = build.dir / "kept"
    shutil.rmtree(kept, ignore_errors=True)
    source = builder.export_tracked(build.source)
    for path in {path for path, _ in planted.values()}:
        (source / path).write_text((ROOT / path).read_text())
    wheels = {}
    for name, (path, mutated) in planted.items():
        (source / path).write_text(mutated)
        build.install(PYTHON, source=source, strict=True)
        (kept / name).mkdir(parents=True)
        wheels[name] = Path(shutil.copy(build.wheel, kept / name))
        (source / path).write_text((ROOT / path).read_text())
    # and the build with no bug in it: the control the suite must pass under
    # the same command (planted_bugs_tested)
    build.install(PYTHON, source=source, strict=True)
    (kept / "unplanted").mkdir(parents=True)
    wheels["unplanted"] = Path(shutil.copy(build.wheel, kept / "unplanted"))
    return wheels


def failing_tests(report: Path) -> list[str]:
    # the node of each test a JUnit report says failed or errored
    if not report.exists():
        return []
    failed = []
    # pytest's own report of the run just made, not untrusted input
    for case in xml.etree.ElementTree.parse(report).iter("testcase"):  # noqa: S314
        if case.find("failure") is not None or case.find("error") is not None:
            failed.append(f"{case.get('classname', '')}::{case.get('name', '')}")
    return failed


def planted_bugs_tested(root: Path) -> bool:
    """
    The suite, as the stage runs it but stopping at its first failure, on
    each planted bug's wheel (`planted_wheels`, which `build --planted`
    made), one at a time, must fail: its timing tests too, which see Ctrl-C
    ignored or the GIL held. The unplanted build's wheel goes first, under
    the same command, and must pass, so a failure says something of its bug.
    A catch is a test failing (pytest's 1, the test read from its JUnit
    report, matching the bug's test_verdict when it names one), a crash
    (a signal), or a hang for a bug whose verdict is HANG; never an error
    collecting or running the suite, or a hang of any other. Each prints the
    test that caught it. Returns whether every bug was caught.
    """
    build = Build(root, "planted")
    report = build.dir / "planted.xml"

    def suite(name: str) -> tuple[str, list[str], str]:
        # how the suite ended on `name`'s wheel, the tests that failed, and its last line
        (wheel,) = (build.dir / "kept" / name).glob("*.whl")
        run(tool("uv"), "pip", "install", "-q", "--python", build.python, "--no-deps", "--reinstall-package",
            "trimkit", wheel)  # fmt: skip
        report.unlink(missing_ok=True)
        command = [build.python, "-m", "pytest", "-q", "-x", "--device-required", f"--junitxml={report}",
                   *test_files()]  # fmt: skip
        try:
            tested = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, check=False, timeout=PLANTED_TEST_SECONDS
            )
        except subprocess.TimeoutExpired:
            return "hung", [], f"hung past {PLANTED_TEST_SECONDS} s"
        last = (tested.stdout.strip().splitlines()[-1:] or [""])[0]
        failed = failing_tests(report)
        if tested.returncode == 0:
            return "passed", failed, last
        if tested.returncode == 1 and failed:
            return "failed", failed, last
        if tested.returncode < 0:
            return "crashed", failed, last
        return f"errored ({tested.returncode})", failed, last

    ended, failed, last = suite("unplanted")
    if ended != "passed":
        sys.exit(
            f"the suite {ended} on the unplanted build, beside the planted bugs: no catch counts\n  {last}"
        )
    caught = {}
    for name in code_mutants("test"):
        verdict = CODE_MUTANTS[name].test_verdict
        ended, failed, last = suite(name)
        if verdict == HANG:
            caught[name] = ended == "hung"
        elif ended == "failed":
            caught[name] = not verdict or any(re.search(verdict, test) for test in failed)
        else:
            caught[name] = ended == "crashed" and not verdict
        shown = ", ".join(failed) or last
        print(f"  {'pass' if caught[name] else f'FAIL ({ended})'}  planted bug {name}: {shown}", flush=True)
    return all(caught.values())


def cmd_asan(args) -> None:
    # CFI too: it needs ThinLTO and hidden visibility, which every build has
    build = Build(args.root, "asan").install(PYTHON, sanitize="address,undefined,cfi")
    # wall-clock bounds measure the sanitizer, not the code. Every allocation
    # is filled with ASan's pattern, all of it rather than its first 4 KiB:
    # outputs are left unwritten when allocated (`Unset`, contour's
    # `UnwrittenValue`), each element written once, so one left unwritten
    # holds the pattern and the tests comparing answers see it
    build.pytest(
        "-m",
        "not timing",
        *test_files(),
        env={
            "LD_PRELOAD": asan_runtime(),
            "ASAN_OPTIONS": "detect_leaks=0:max_malloc_fill_size=2147483647",
            "PYTHONMALLOC": "malloc",
        },
    )


def fetch(url: str, sha256: str, into: Path) -> Path:
    """
    A release tarball, checked against its pinned sha256 as it streams in,
    extracted into `into`, emptied first: its one top directory.
    """
    # S310 warns of urlopen reading file: and other schemes: https only
    if not url.startswith("https://"):
        sys.exit(f"{url} isn't https")
    shutil.rmtree(into, ignore_errors=True)
    into.mkdir(parents=True)
    print("+ fetch", url, flush=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, tempfile.TemporaryFile() as tarball:  # noqa: S310
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            tarball.write(chunk)
        if digest.hexdigest() != sha256:
            sys.exit(f"{url} is {digest.hexdigest()}, not its pinned sha256 {sha256}")
        tarball.seek(0)
        with tarfile.open(fileobj=tarball) as tar:
            tar.extractall(into, filter="data")
    (tree,) = into.iterdir()
    return tree


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
    url = f"https://www.python.org/ftp/python/{CPYTHON}/Python-{CPYTHON}.tar.xz"
    tree = fetch(url, CPYTHON_SHA256, source)
    configure = ["./configure", "-q", "--disable-gil", "--with-thread-sanitizer", f"--prefix={prefix}"]
    run(*configure, cwd=tree, env={"CC": "clang", "CXX": "clang++"})
    run(tool("make"), "-s", f"-j{len(os.sched_getaffinity(0))}", cwd=tree)
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
    build.pytest("-s", "check/tests/test_threads.py", env={"TSAN_OPTIONS": options})


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
    # from the seeds alone, every run, as CI's: a run is the commit and nothing else
    shutil.rmtree(corpus, ignore_errors=True)
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
    # the pool's harness must fail on each bug planted for it
    failed |= not planted_bugs_fuzzed(args.root, corpus / "parallel")
    # the header-only harnesses again as 32-bit programs, over that corpus
    failed |= not replay_32bit(args.root, corpus, findings)
    # then the bindings, whatever the harnesses found
    atheris(args)
    if failed:
        sys.exit(f"fuzzing found something: see {findings}")


# how long check/fuzz/parallel.cpp gets to fail on a planted bug: its fixed
# checks, then inputs. Most fail in the first second; one that hangs (a wake
# that never comes) takes all of it
PLANTED_SECONDS = 30


def planted_bugs_fuzzed(root: Path, corpus: Path) -> bool:
    """
    check/fuzz/parallel.cpp, built on src/parallel.hpp with each bug planted
    for the fuzzer (CODE_MUTANTS), must catch it with its own verdict: print
    the `expect` message the bug names, or hang for a lost wake (HANG), and
    nothing else counts (a timeout of a bug that names a message, a build that
    fails, another check). The header as it is runs beside them, under the
    same load, and must pass: so the harness is shown able to fail on the real
    code. Returns whether every bug was caught.
    """
    harness = CHECK / "fuzz" / "parallel.cpp"

    def ran(planted: tuple[str, str]) -> tuple[str, str, str]:
        # (name, how it ended: "passed", "hung" or "failed", what it printed
        # of its checks)
        name, source = planted
        here = root / "fuzz" / "planted" / name
        shutil.rmtree(here, ignore_errors=True)
        (here / "src").mkdir(parents=True)
        (here / "check" / "fuzz").mkdir(parents=True)
        for header in (ROOT / "src").glob("*.hpp"):
            shutil.copy(header, here / "src")
        (here / "src" / "parallel.hpp").write_text(source)
        shutil.copy(harness, here / "check" / "fuzz")
        shutil.copytree(corpus, here / "corpus")
        binary = here / "fuzz_parallel"
        subprocess.run(
            [tool("clang++"), "-std=c++20", "-O1", "-g", "-pthread", "-fsanitize=fuzzer,address,undefined",
             "-fno-sanitize-recover=all", "-DTRIMKIT_CHECKED", str(here / "check" / "fuzz" / "parallel.cpp"),
             "-o", str(binary)],
            check=True, capture_output=True,
        )  # fmt: skip
        try:
            done = subprocess.run(
                [*no_aslr(), str(binary), str(here / "corpus"), f"-max_total_time={PLANTED_SECONDS}",
                 "-seed=1", "-timeout=10", f"-artifact_prefix={here}/"],
                cwd=here, capture_output=True, text=True, check=False, timeout=PLANTED_SECONDS + 30,
            )  # fmt: skip
        except subprocess.TimeoutExpired as hung:
            said = hung.stderr or ""
            said = said.decode(errors="replace") if isinstance(said, bytes) else said
            return name, "hung", said
        if "ERROR: libFuzzer: timeout" in done.stderr:
            return name, "hung", done.stderr  # an input that waited past -timeout: hung all the same
        return name, "passed" if done.returncode == 0 else "failed", done.stderr

    def checks_said(output: str) -> list[str]:
        return [line for line in output.splitlines() if line.startswith("check/fuzz/parallel.cpp: ")]

    planted = {name: source for name, (_, source) in code_mutants("fuzz").items()}
    control = ("unplanted", (ROOT / "src" / "parallel.hpp").read_text())
    with concurrent.futures.ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        (_, ended, said), *results = pool.map(ran, [control, *planted.items()])
    if ended != "passed":
        sys.exit(
            f"the fuzz harness {ended} on src/parallel.hpp as it is, beside the planted bugs:"
            " no catch counts\n  " + "\n  ".join(checks_said(said))
        )
    caught = True
    for name, ended, said in results:
        verdict = CODE_MUTANTS[name].fuzz_verdict
        lines = checks_said(said)
        found = ended == "hung" if verdict == HANG else f"check/fuzz/parallel.cpp: {verdict}" in lines
        caught = caught and found
        why = "hung" if ended == "hung" else (lines[-1] if lines else f"{ended}, saying nothing")
        expected = "" if found else f" (its verdict: {verdict})"
        print(f"  {'pass' if found else 'FAIL'}  planted bug {name}: {why}{expected}", flush=True)
    return caught


def replay_32bit(root: Path, corpus: Path, findings: Path) -> bool:
    """
    The harnesses of every part that compiles no vendored sources (its
    upstreams are headers alone), built again as 32-bit programs with ASan and
    UBSan and run over every input the 64-bit run kept: size_t is 32 bits
    there, as on the x86 Windows wheel, where a cast that cuts a 64-bit value
    shows. Returns whether every input passed.
    """
    directory = root / "fuzz-32"
    directory.mkdir(parents=True, exist_ok=True)
    passed = True
    for part in MANIFEST.parts.values():
        if any(upstream.sources for upstream in part.upstreams):
            continue
        includes = [word for u in part.upstreams for d in u.include for word in ("-isystem", str(ROOT / d))]
        defines = [f"-D{name}" for u in part.upstreams for name in u.defines]
        for name in part.fuzz:
            run(
                # SSE2 floating point, as the x86 Windows wheel has: x87's
                # 80-bit registers round twice, which no shipped build does
                tool("clang++"), "-m32", "-msse2", "-mfpmath=sse",
                "-std=c++17", "-O1", "-g", "-D_GLIBCXX_ASSERTIONS",
                "-fsanitize=fuzzer,address,undefined", "-fno-sanitize-recover=all",
                *includes, *defines, CHECK / "fuzz" / f"{name}.cpp", *([part.source] if part.source else []),
                "-o", directory / f"fuzz_{name}",
            )  # fmt: skip
            report = harness(directory, corpus, findings, name, ["-runs=0"])
            print(report.replace(f"{name}:", f"{name} (32-bit):"), flush=True)
            passed = passed and "exit " not in report
    return passed


def atheris(args) -> None:
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
        CHECK / "fuzz" / "atheris_bindings.py",
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
    config = f"--config-file={ROOT / 'scripts' / 'clang-tidy.yaml'}"
    run(tool("clang-tidy"), "--quiet", config, header_filter, "-p", dev(args.root).build_dir, *sources)


def cmd_cppcheck(args) -> None:
    # our sources, and the headers they include (clang's analyzer runs in `tidy`)
    run(
        tool("cppcheck"),
        "--quiet",
        "--enable=warning,style,performance,portability",
        "--inline-suppr",
        "--std=c++17",
        "--error-exitcode=1",
        # every branch, not cppcheck's default sample of them, one line per finding
        "--check-level=exhaustive",
        # and every configuration of the #ifdefs, not the 12 it samples, so
        # what it reports doesn't move when an include does
        "--force",
        f"-j{len(os.sched_getaffinity(0))}",
        "--template={file}:{line}: {severity}: {message} [{id}]",
        *MANIFEST.sources,
    )


def cpp_files() -> list[Path]:
    # all of our C++, whichever directory it lives in
    directories = [*MANIFEST.directories, CHECK / "fuzz"]
    return [p for d in directories for p in builder.tracked_under(d) if p.suffix in (".cpp", ".hpp")]


# how our C++ is laid out, for the image's clang-format
CLANG_FORMAT_STYLE = f"--style=file:{ROOT / 'scripts' / 'clang-format.yaml'}"


def cmd_format(args) -> None:
    run(tool("clang-format"), CLANG_FORMAT_STYLE, "-i", *cpp_files())


def cmd_clang_format(args) -> None:
    run(tool("clang-format"), CLANG_FORMAT_STYLE, "--dry-run", "-Werror", *cpp_files())


def cmd_ruff(args) -> None:
    # tracked Python only, still honoring pyproject's excludes
    python = [p for p in builder.tracked() if p.suffix in (".py", ".pyi")]
    run(tool("ruff"), "check", "--no-cache", "--force-exclude", *sorted(python))
    run(tool("ruff"), "format", "--no-cache", "--check", "--force-exclude", *sorted(python))


def cmd_actions(args) -> None:
    github = builder.tracked_under(ROOT / ".github")
    run(tool("zizmor"), "--offline", "--pedantic", *github)


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
    build.pytest(tree / "check" / "tests", cwd=tree)
    # and on free-threaded Python, which gets no wheels: this is how it installs.
    # Wall-clock bounds aside, as they measure the GIL it doesn't have
    free = Build(args.root, "sdist-free-threaded").install(FREE_THREADED_PYTHON, source=archive, strict=True)
    free.pytest("-m", "not timing", tree / "check" / "tests", cwd=tree)


def cmd_licenses(args) -> None:
    # every component permissive, its license shipped, its SBOM entry whole
    licenses.check()


# what starts or waits on a thread, allowed in src/parallel.hpp alone
THREADING = (
    "<mutex>",
    "<condition_variable>",
    "<future>",
    "std::mutex",
    "std::condition_variable",
    "std::thread(",
    "std::async",
    "pthread_create",
    "_beginthreadex",
)
# the vendored libraries with thread pools of their own, and the define that compiles each out
POOLS_COMPILED_OUT = {
    "vhacd": "VHACD_DISABLE_THREADING",
    "tinybvh": "NO_THREADED_BUILDS",
    "xatlas": "XA_MULTITHREADED=0",
}

# what only src/numpy.hpp may do: touch nanobind's ndarray directly, or the
# checks behind its `read*` functions, so no binding can skip one
DOOR_ONLY = re.compile(r"nb::ndarray|try_cast|check_values|trimkit::detail::|\b(malloc|calloc|realloc)\(")


# numpy's array makers whose dtype, left out, comes from the values: integers
# become the platform's default, int32 on 32-bit machines and int64 elsewhere
INT_MAKERS = {"array", "asarray", "ascontiguousarray", "fromiter", "arange", "full"}


def _constants(node: ast.AST) -> set[str]:
    # the kinds of number a literal holds, through lists, tuples, signs and a
    # comprehension's element
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return set()
        return {type(node.value).__name__} if isinstance(node.value, (int, float, complex)) else set()
    if isinstance(node, ast.UnaryOp):
        return _constants(node.operand)
    if isinstance(node, (ast.List, ast.Tuple)):
        return set().union(*(_constants(e) for e in node.elts))
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return _constants(node.elt)
    # a name may hold anything; a call or arithmetic, a float as easily
    return {"name"} if isinstance(node, ast.Name) else {"unknown"}


def _int_literal(node: ast.AST) -> bool:
    # a literal numpy gives its default integer dtype: integers in it, and
    # nothing that is or may be floating
    kinds = _constants(node)
    return "int" in kinds and kinds <= {"int", "name"}


def _default_ints(tree: ast.AST) -> list[int]:
    # the lines of `tree` that make an array of the platform's default integer
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        keywords = {k.arg: k.value for k in node.keywords}
        # dtype=int, astype(int) and their spellings are the platform's int too
        given = keywords.get("dtype")
        if isinstance(function, ast.Attribute) and function.attr == "astype" and node.args:
            given = node.args[0]
        if (isinstance(given, ast.Name) and given.id == "int") or (
            isinstance(given, ast.Constant) and given.value in ("int", "int_", "intp")
        ):
            lines.append(node.lineno)
            continue
        named = isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)
        if not (named and function.value.id in ("np", "numpy") and function.attr in INT_MAKERS):
            continue
        if "dtype" in keywords:
            continue
        if function.attr == "arange":
            # integers unless a float is given
            floats = any(isinstance(a, ast.Constant) and isinstance(a.value, float) for a in node.args)
            if not floats:
                lines.append(node.lineno)
        elif function.attr == "full":
            if len(node.args) > 1 and _int_literal(node.args[1]):
                lines.append(node.lineno)
        elif node.args and _int_literal(node.args[0]):
            lines.append(node.lineno)
    return lines


def check_default_ints() -> None:
    # Every integer array names its dtype: numpy's default integer is int32 on
    # 32-bit machines, so an array that is int64 here is converted, with a
    # warning, or refused on the x86 Windows wheel. In tracked Python, and in
    # the examples of every docstring in it (the stubs carry the bindings')
    found = []
    for path in sorted(p for p in builder.tracked() if p.suffix in (".py", ".pyi")):
        text = path.read_text()
        tree = ast.parse(text)
        found += [f"{path.relative_to(ROOT)}:{line}" for line in _default_ints(tree)]
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node, clean=False)
            for example in doctest.DocTestParser().get_examples(docstring or ""):
                try:
                    lines = _default_ints(ast.parse(example.source))
                except SyntaxError:
                    continue
                name = getattr(node, "name", "module")
                found += [
                    f"{path.relative_to(ROOT)}: {name} example `{example.source.strip()}`" for _ in lines
                ]
    if found:
        sys.exit(
            "integer arrays must name their dtype (numpy's default is int32 on 32-bit):\n  "
            + "\n  ".join(found)
        )


def check_door() -> None:
    # every C++ file of ours but the door itself, and the allocation door,
    # whose malloc is every allocation's
    every = [p for p in builder.tracked_under(ROOT / "src") if p.suffix in (".cpp", ".hpp")]
    sources = [p for p in every if p.name not in ("numpy.hpp", "allocator.cpp")]
    found = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in sources
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if DOOR_ONLY.search(line)
    ]
    if found:
        sys.exit("arrays must go through src/numpy.hpp:\n  " + "\n  ".join(found))
    # Windows: its calls are declared in src/windows.hpp, as windows.h's macros
    # (FAR, near, min, max) rename whatever of ours they match
    found = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in every
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"#\s*include\s*<windows\.h>", line, re.IGNORECASE)
    ]
    if found:
        sys.exit("Win32 calls must be declared in src/windows.hpp, never windows.h:\n  " + "\n  ".join(found))
    # numpy's bool bytes are read as bytes (numpy.hpp, mask_bytes): a byte
    # other than 0 or 1 is undefined as a C++ bool, and a uint8 array viewed
    # as bool holds one
    found = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in sources
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"\bbool\s*\*", line) and "NOLINT" not in line
    ]
    if found:
        sys.exit(
            "read bool arrays as bytes (numpy.hpp, mask_bytes), never through bool*:\n  " + "\n  ".join(found)
        )
    # threads, in every file, the two doors too: only trimkit.hpp reaches the
    # pool, through in_parallel and the rest of its door, and gil.hpp, for the pool's exit gate and Ctrl-C
    # (platform.hpp is the pool's own layer). Anything else names at most the
    # Stop it is handed, never the pool itself
    found = [
        f"{path.relative_to(ROOT)}: {needle}"
        for path in every
        if path.name not in ("trimkit.hpp", "gil.hpp", "parallel.hpp", "platform.hpp")
        for needle in sorted(set(re.findall(r'parallel\.hpp"|parallel::(?!Stop\b)\w+', path.read_text())))
    ]
    if found:
        sys.exit("threads must go through in_parallel in src/trimkit.hpp:\n  " + "\n  ".join(found))
    # and the platform only through the pool, but for the Vulkan code's fork
    # generation: platform.hpp is no second door to threads
    found = [
        f"{path.relative_to(ROOT)}: platform.hpp"
        for path in every
        if path.name not in ("parallel.hpp", "platform.hpp", "vulkan.hpp")
        and re.search(r'#\s*include\s*"[./]*platform\.hpp"', path.read_text())
    ]
    if found:
        sys.exit("src/platform.hpp is only for parallel.hpp and Vulkan:\n  " + "\n  ".join(found))
    # and the pool is the only thread, mutex or wait there is: nothing of ours
    # starts or blocks on anything else
    found = [
        f"{path.relative_to(ROOT)}: {needle}"
        for path in every
        if path.name not in ("parallel.hpp", "platform.hpp")
        for needle in THREADING
        if needle in path.read_text()
    ]
    if found:
        sys.exit("the pool in src/parallel.hpp is the only threading there is:\n  " + "\n  ".join(found))
    # a thread starts only at a call's wake that finds too few live
    # (wake_helpers), or in the chain it began (a new worker's first step):
    # never at an exit, which once started one for a wake other workers had
    # served (Scheduler.tla, P3, checks the protocol; this, that the code
    # has no other door)
    pool = (ROOT / "src" / "parallel.hpp").read_text()
    starts = []
    for call in re.finditer(r"\bspawn_worker\(\)", pool):
        before = pool[: call.start()]
        if before.endswith("inline bool "):  # its definition
            continue
        function = re.findall(r"^inline [^\n(]*?\b(\w+)\(", before, re.MULTILINE)[-1]
        case = re.findall(r"case (\w+::\w+):", before)
        starts.append(function if function != "worker_main" else f"worker_main {case[-1] if case else ''}")
    if sorted(starts) != ["wake_helpers", "worker_main WorkerStep::Start"]:
        sys.exit(
            f"src/parallel.hpp: spawn_worker() only from wake_helpers and WorkerStep::Start, not {starts}"
        )
    # the GIL: released only through GilReleased (src/gil.hpp), which makes the
    # interpreter's exit safe for every call that releases it
    found = [
        f"{path.relative_to(ROOT)}: {needle}"
        for path in every
        if path.name != "gil.hpp"
        for needle in ("gil_scoped_release", "PyEval_SaveThread", "Py_BEGIN_ALLOW_THREADS")
        if needle in path.read_text()
    ]
    if found:
        sys.exit("release the GIL only through GilReleased in src/gil.hpp:\n  " + "\n  ".join(found))
    # every vendored library with a pool of its own has it compiled out, and
    # runs its fan-outs on ours or on the calling thread
    for upstream, define in POOLS_COMPILED_OUT.items():
        if define not in MANIFEST.upstreams[upstream].defines:
            sys.exit(f"trimkit.toml: [upstream.{upstream}.defines] must set {define}: one pool, parallel.hpp")
    print(f"ok: {len(sources)} sources use only src/numpy.hpp for arrays and src/trimkit.hpp for threads")


def cmd_audit(args) -> None:
    # the dev build's wheel, audited as every release wheel is
    run(dev(args.root).python, ROOT / "scripts" / "check_wheel.py", dev(args.root).wheel)


# the model's steps that only a mutation takes (claim_load_then_store's): no
# scenario of the real protocol, nor any traced run, ever takes them
MUTATION_ONLY_STEPS = ("CallerClaimStore", "WorkerClaimStore")


def check_model_steps() -> None:
    """
    The model (check/verify/tla/Scheduler.tla) and the code name the same steps:
    each CallerStep and WorkerStep of src/parallel.hpp has an action named
    Caller<Step> or Worker<Step> (or <Step><Part>, an atomic part of it), and
    each such action is part of a step the code has.
    """
    header = (ROOT / "src" / "parallel.hpp").read_text()
    spec = (CHECK / "verify" / "tla" / "Scheduler.tla").read_text()
    problems = []
    for side in ("Caller", "Worker"):
        enum = re.search(rf"enum class {side}Step : std::uint8_t \{{(.*?)\}};", header, re.S)
        steps = set(re.findall(r"^\s*(\w+),", enum.group(1), re.M)) if enum else set()
        actions = set(re.findall(rf"^{side}(\w+)\(t, f\) ==", spec, re.M))
        if not steps:
            problems.append(f"src/parallel.hpp has no {side}Step")
        for action in sorted(actions):
            if not any(action.startswith(step) for step in steps):
                problems.append(f"{side}{action} in the model is no step of {side}Step")
        for step in sorted(steps):
            if not any(action.startswith(step) for action in actions):
                problems.append(f"{side}Step::{step} has no action {side}{step} in the model")
            if f"case {side}Step::{step}:" not in header:
                problems.append(f"{side}Step::{step} has no case in its switch")
    # and each traced step is one of the model's actions, and each action is
    # traced, but for running items, which is local, and a mutation's
    traced = set(re.findall(r'"((?:Caller|Worker)\w+)"', header))
    modelled = set(re.findall(r"^((?:Caller|Worker)\w+)\(t, f\) ==", spec, re.M))
    local = {"CallerRun", "CallerRunNested", "WorkerRun", "WorkerRunNested", *MUTATION_ONLY_STEPS}
    problems += [
        f"src/parallel.hpp traces {name}, which the model has no action for"
        for name in sorted(traced - modelled)
    ]
    problems += [
        f"the model's {name} is no step src/parallel.hpp traces" for name in sorted(modelled - traced - local)
    ]
    if problems:
        sys.exit("the model and src/parallel.hpp name different steps:\n  " + "\n  ".join(problems))


def cmd_lint(args) -> None:
    # fast and deterministic, nothing compiled: what every file must already be
    vendor.check()
    builder.check()
    check_door()
    check_default_ints()
    check_model_steps()
    # every planted bug still applies, whichever stage plants it: checked as
    # text here, so a stale one is seen on any machine, not only by its stage
    code_mutants("")
    # the lock matches pyproject.toml
    run(tool("uv"), "lock", "--check")
    for command in (cmd_clang_format, cmd_ruff, cmd_actions, cmd_licenses, cmd_shader, cmd_cppcheck):
        command(args)


# What starts, blocks on or sleeps a thread, or handles a signal: in the code a
# build compiles, after every #if and every patch, only src/parallel.hpp may
# (`cmd_threads`). A regular expression over preprocessed C++, so a name in a
# comment or a compiled-out branch is never seen
THREAD_CALLS = re.compile(
    r"\b(?:std::(?:thread|jthread|async|future|promise|mutex|recursive_mutex|timed_mutex|shared_mutex"
    r"|condition_variable(?:_any)?|counting_semaphore|binary_semaphore|latch|barrier)\b"
    r"|std::this_thread::(?:sleep_for|sleep_until|yield)"
    r"|pthread_(?:create|mutex_lock|mutex_trylock|cond_wait|cond_timedwait|sigmask|kill|cancel)"
    r"|_beginthreadex|CreateThread|WaitOnAddress|WakeByAddress\w*|__ulock_wait|__ulock_wake"
    r"|sched_yield|nanosleep|usleep|Sleep|SleepEx|sigaction|sigprocmask|SetConsoleCtrlHandler"
    r"|PyErr_CheckSignals|PyErr_SetInterrupt\w*|notify_one|notify_all)\b"
    r"|\bsignal\s*\(|\braise\s*\(\s*SIG"
)
# the pool itself, and what is still to move into it, each with why it may
THREAD_CALLS_ALLOWED = {
    "src/platform.hpp": "the operating system's threads, waits and signal masks, for the pool alone",
    "src/parallel.hpp": "the pool",
    "src/gil.hpp": "Interrupt polls for Ctrl-C on the calling thread (PyErr_CheckSignals)",
}
# Eigen is vendored unmodified (MPL-2.0, `check` refuses a patch to it) and
# isn't scanned: its thread pool is compiled only with EIGEN_USE_THREADS,
# which no build may define
THREAD_CALLS_UNSCANNED = ("vendor/eigen/",)
# What reaches past the process, by the same scan, Eigen included: a library
# opened or looked into, a process started, a file or socket opened, the kernel
# called. The loaders and fork count at any mention (an address taken is a
# call), the rest when called, each with the literal ending its first or
# second argument. Linux's names only, as only Linux's branches are seen. Not
# `system`, which a member initializer can't be told from, nor SYS_futex, a
# number once preprocessed: check_wheel.py refuses the one imported, and
# `syscall` is what is seen of the other
REACHING_CALLS = re.compile(
    r"(?<![\w.>])(?:(dlopen|dlmopen|dlsym|dlvsym|v?fork)\b"
    r"|(syscall|open|openat|fopen|freopen|creat|socket|connect|popen|exec[lv]p?e?|fexecve|posix_spawnp?)"
    r"(?=\s*\((?!\s*\))))"
    r"(?:\s*\(\s*(?:\w+\s*,\s*)?(\"[^\"]*\")(?=\s*[,)]))?"
)
# each file's calls and the literal each names, "*" for any arguments. The
# loader is pinned to its names; open and syscall only to their file, not to
# what is opened or which call is made. Windows' loader is check_wheel.py's
REACHING_ALLOWED = {
    "src/vulkan/vulkan.hpp": {
        ("dlopen", '"libvulkan.so.1"'),
        ("dlopen", '"libvulkan.so"'),
        ("dlsym", '"vkGetInstanceProcAddr"'),
    },
    "src/cgroup.hpp": {("open", "*")},  # limits from /proc and /sys, read-only
    "src/platform.hpp": {("syscall", "*")},  # futex
}


def cmd_threads(args) -> None:
    """
    Nothing the extension compiles starts, blocks on or sleeps a thread, or
    handles a signal, outside the pool, nor reaches past the process but as
    REACHING_ALLOWED says: every translation unit the dev build compiles is
    preprocessed with its own command, so what is seen is what is built,
    every vendored library's #if decided and every patch applied, and each
    call is traced by its line markers to the file it is written in. The dev
    build's platform only: another's branches go unread here, and what the
    binaries import is check_wheel.py's.
    """
    compile_commands(args.root)
    build = dev(args.root)
    entries = json.loads((build.build_dir / "compile_commands.json").read_text())
    defined = [d for u in MANIFEST.upstreams.values() for d in u.defines if d.startswith("EIGEN_USE_THREADS")]
    if defined:
        sys.exit(f"trimkit.toml defines {defined[0]}: Eigen's thread pool stays compiled out")

    def scan(entry: dict) -> set[tuple[str, str]]:
        command, skip = [], False
        for word in shlex.split(entry["command"]):
            if skip or word in ("-c", "-MD", "-MMD"):
                skip = False
                continue
            skip = word in ("-o", "-MF", "-MT", "-MQ")
            if not skip:
                command.append(word)
        text = subprocess.run(
            [*command, "-E"], cwd=entry["directory"], capture_output=True, text=True, check=True
        ).stdout
        found, where = set(), ""
        for line in text.splitlines():
            marker = re.match(r'# \d+ "([^"]+)"', line)
            if marker:
                path = (Path(entry["directory"]) / marker[1]).resolve()
                where = path.relative_to(build.source).as_posix() if path.is_relative_to(build.source) else ""
            elif where:
                if where not in THREAD_CALLS_ALLOWED and not where.startswith(THREAD_CALLS_UNSCANNED):
                    found |= {(where, call[0]) for call in THREAD_CALLS.finditer(line)}
                allowed = REACHING_ALLOWED.get(where, set())
                for call in REACHING_CALLS.finditer(line):
                    name, literal = call[1] or call[2], call[3] or ""
                    if (name, literal) not in allowed and (name, "*") not in allowed:
                        found.add((where, f"{name}({literal or '...'})"))
        return found

    with concurrent.futures.ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        found = sorted(set().union(*pool.map(scan, entries)))
    if found:
        sys.exit(
            "threads, waits and signals are src/parallel.hpp's alone, and libraries, processes, files,"
            " sockets and system calls are only as REACHING_ALLOWED says, in what the build compiles:\n  "
            + "\n  ".join(f"{where}: {call}" for where, call in found)
        )
    print(
        f"ok: {len(entries)} translation units start, wait on and signal no thread outside the pool, "
        "and, in the Linux build's source, open no library but the Vulkan loader"
    )


# the pool against the best of what people write by hand, in every cell of
# check/bench/parallel.cpp: no slower than this times the best of them, plus this,
# plus the tail no dynamic schedule can promise to beat, which the pool states
# as its bound: the cell's costliest single item and one first claim
# (src/parallel.hpp, policy::MIN_CLAIM_NS)
POOL_AGAINST_HAND = 1.10
POOL_SLACK_US = 20.0
POOL_TAIL_CLAIM_US = 4.0
# and no more CPU than this times the persistent hand-written pools', plus
# this: spinning on cores a neighbour could use is a cost too
POOL_CPU_AGAINST_HAND = 1.25
POOL_CPU_SLACK_US = 50.0
# the cores `bench` gets, by default: enough to see the pool scale, the last ones
BENCH_CORES = 8


def _bench_misses(cell: dict) -> list[str]:
    # why a cell misses the gate, if it does
    hand = min(cell["spawn"], cell["cv_static"], cell["cv_chunks"])
    allowed = POOL_AGAINST_HAND * hand + POOL_SLACK_US + cell["costliest_us"] + POOL_TAIL_CLAIM_US
    hand_cpu = min(cell["cv_static_cpu"], cell["cv_chunks_cpu"])
    reasons = []
    if cell["pool"] > allowed:
        reasons.append(f"pool {cell['pool']:.0f} us, best by hand {hand:.0f} us")
    if cell["pool"] > cell["serial"] + POOL_SLACK_US + cell["costliest_us"] + POOL_TAIL_CLAIM_US:
        reasons.append(f"pool {cell['pool']:.0f} us, serial {cell['serial']:.0f} us")
    if cell["pool_cpu"] > POOL_CPU_AGAINST_HAND * hand_cpu + POOL_CPU_SLACK_US:
        reasons.append(f"pool CPU {cell['pool_cpu']:.0f} us, by hand {hand_cpu:.0f} us")
    return reasons


# the longest serial time a cell's wake-up cost is read from (_wake_cost_us)
WAKE_COST_CELL_US = 1_000.0
# the most serial work a cell may be: all of them, or with `--quick` the short ones
BENCH_SERIAL_NS = 2e9
BENCH_QUICK_SERIAL_NS = 2e7


def _wake_cost_us(cell: dict) -> str:
    # what a wake-up cost here, against the C the pool's policy takes it to
    # (src/parallel.hpp, policy::WAKE_COST_NS): the CPU the pool took beyond
    # the serial loop's and its workers' spinning, for each worker it woke or
    # started, the work running slower on a core just woken included. Read
    # only where the work is short enough that its own CPU's noise, a few
    # percent, can't swamp a few wakes'
    roused = cell["woken"] + cell["spawns"]
    if roused == 0 or cell["serial"] > WAKE_COST_CELL_US:
        return "-"
    return f"{(cell['pool_cpu'] - cell['serial_cpu'] - cell['spin_us']) / roused:.1f}"


def cmd_bench(args) -> None:
    """
    src/parallel.hpp against a serial loop and three pools written by hand
    (check/bench/parallel.cpp), with 1 and 4 callers, alone and with a noisy
    neighbour, on `--cores` cores. In every cell the pool must be within
    POOL_AGAINST_HAND of the best hand-written way plus POOL_SLACK_US and the
    costliest item, never slower than the serial loop by more than that, and
    use no more than POOL_CPU_AGAINST_HAND of the persistent pools' CPU plus
    POOL_CPU_SLACK_US. A cell that misses is measured twice more, and misses
    only if it does every time: a busy machine slows a round, not every round.
    Prints each cell, then fails naming every cell that missed, and why.
    `--quick` measures only the cells of at most BENCH_QUICK_SERIAL_NS of
    serial work: the short calls, where scheduling costs show.
    """
    binary = args.root / "bench" / "parallel"
    binary.parent.mkdir(parents=True, exist_ok=True)
    run(
        tool("clang++"),
        "-std=c++20",
        "-O2",
        "-DNDEBUG",
        "-pthread",
        CHECK / "bench" / "parallel.cpp",
        "-o",
        binary,
    )
    threads = str(len(os.sched_getaffinity(0)))
    most = f"{BENCH_QUICK_SERIAL_NS if args.quick else BENCH_SERIAL_NS:.0f}"
    missed = []
    costs: list[str] = []
    for callers in ("1", "4"):
        for noise in ("0", threads):
            measured = run(binary, threads, callers, noise, most, capture=True)
            cells = [json.loads(line) for line in measured.splitlines()]
            for index, first in enumerate(cells):
                best = first
                for _ in range(2):
                    if not _bench_misses(best):
                        break
                    only = f"{best['items']},{best['item_ns']:.0f},{best['skew_index']}"
                    again = json.loads(
                        run(binary, threads, callers, noise, most, only, capture=True).splitlines()[-1]
                    )
                    best = min(
                        best, again, key=lambda c: len(_bench_misses(c)) + c["pool"] / (c["serial"] + 1)
                    )
                cells[index] = best
            print(
                f"\n{threads} threads, {callers} callers, noise {noise}\n"
                f"{'items':>8} {'item ns':>8} {'skew':>15} {'serial':>10} {'pool':>10} {'best hand':>10} "
                f"{'pool/best':>9} {'cpu/hand cpu':>12} {'threads':>7} {'woken':>5} {'spin us':>7} "
                f"{'us/wake':>7}"
            )
            for cell in cells:
                hand = min(cell["spawn"], cell["cv_static"], cell["cv_chunks"])
                hand_cpu = min(cell["cv_static_cpu"], cell["cv_chunks_cpu"])
                print(
                    f"{cell['items']:>8} {cell['item_ns']:>8.0f} {cell['skew']:>15} {cell['serial']:>10.0f} "
                    f"{cell['pool']:>10.0f} {hand:>10.0f} {cell['pool'] / hand:>9.2f} "
                    f"{cell['pool_cpu'] / hand_cpu:>12.2f} {cell['pool_threads']:>7} {cell['woken']:>5} "
                    f"{cell['spin_us']:>7.0f} {_wake_cost_us(cell):>7}"
                )
                reasons = _bench_misses(cell)
                if reasons:
                    missed.append(
                        f"{cell['items']} items of {cell['item_ns']:.0f} ns, {cell['skew']}, "
                        f"{callers} callers, noise {noise}: " + "; ".join(reasons)
                    )
            costs.extend(c for c in map(_wake_cost_us, cells) if c != "-")
            # a wheel's caller looks again each second for workers whose wake
            # was lost (src/parallel.hpp, RECHECK_MS): that it never had to
            if cells and cells[-1]["rechecks"]:
                missed.append(f"{callers} callers, noise {noise}: {cells[-1]['rechecks']} wakes were lost")
    if costs:
        ordered = sorted(float(c) for c in costs)
        print(
            f"\nwhat a wake-up cost, beyond the spinning, in the {len(ordered)} cells under "
            f"{WAKE_COST_CELL_US:.0f} us serial that woke workers: median "
            f"{ordered[len(ordered) // 2]:.1f} us, p90 {ordered[len(ordered) * 9 // 10]:.1f} us, "
            f"against the policy's C of {cells[-1]['wake_cost_us']:.1f} us"
        )
    if missed:
        sys.exit(f"the pool missed the gate in {len(missed)} cells:\n  " + "\n  ".join(missed))
    print("ok: the pool is within the gate in every cell")


# GenMC (scripts/docker/Dockerfile.verify) runs src/parallel.hpp itself, built for its
# backend (src/platform.hpp, TRIMKIT_GENMC), in every execution the C++
# memory model allows (RC11), and reports an assertion, a data race, a use
# after free, or a wait that can never end. Each harness in check/verify/genmc must
# explore at least its floor of executions, about half what it does now, so
# one that stops sharing work (and passes for want of anything to check)
# fails too. One call at a time, with its worker: two jobs at once (two
# callers, a nested call, or one caller's two calls) GenMC estimates at some
# 170 million executions, so those are check/verify/tla's, the fuzzer's and TSan's
GENMC_HARNESSES = {
    "visibility": 40,
    "early_worker": 20000,
    "failure": 290,
    "stop": 560,
    "exit_stop": 240,
    "thread_index": 260,
    "exclusively": 2,
    "chunk_list": 5,
    "exit_gate": 15,
}
GENMC_SECONDS = 1200
# the memory orders in src/parallel.hpp that may be weakened by one step with
# no harness failing, each with why it is still as strong as it is: keyed by
# the line's code, so a moved or edited line is looked at again
# No entry may rest on the model alone, which assumes sequentially consistent
# memory: each says what does check the code, or that nothing does
_LOST_WAKE = (
    "keeps a wake from being lost (the wake, spinning and live counts' pairings). Argued only under weak"
    " memory: GenMC's waits return at once, so it can't see a lost wake. On real threads the fuzzer does"
    " (a caller with no Stop waits with no timeout, and its fixed check counts sleepers woken), on x86 alone"
)
_LEFT = (
    "orders a worker's leave before its bump of `left`, so a caller that sees the bump sees the leave and"
    " doesn't sleep on it. Argued only under weak memory, as GenMC's waits return at once; on real threads"
    " the fuzzer's callers wait with no timeout, so sleeping through the last leave hangs it, on x86 alone"
)
_TWO_HOLDERS = (
    "orders one holder of a thread index before the next, which are two workers' lifetimes: no harness of one"
    " call reaches that. ThreadSanitizer checks the scratch they index (check/tests/test_threads.py)"
)
_TWO_JOBS = (
    "orders the last job's last leave before the next reservation of its slot: two jobs, which GenMC can't"
    " reach here (some 170 million executions). Argued only"
)
_COUNTED_ONLY = (
    "holds idle workers for check/fuzz/parallel.cpp's checks, in counted builds alone, which GenMC never"
    " builds. seq_cst because letting go is a store, then a bump of the wake, both before a worker's"
    " seq_cst look at the wake and then at the hold: one order, so a worker woken by the bump sees the"
    " hold let go, and never parks again without a timeout"
)
GENMC_MUTANTS_SURVIVING: dict[str, str] = {
    "return idle_held.load(std::memory_order_seq_cst);  [seq_cst -> acquire]": _COUNTED_ONLY,
    "idle_held.store(hold, std::memory_order_seq_cst);  [seq_cst -> release]": _COUNTED_ONLY,
    "pool.wake.fetch_add(1, std::memory_order_seq_cst);  [seq_cst -> acq_rel]": _COUNTED_ONLY,
    "if ((held_indexes[index / 64].fetch_or(bit, std::memory_order_acquire) & bit) == 0) {  [acquire -> relaxed]": _TWO_HOLDERS,  # noqa: E501
    "held_indexes[index / 64].fetch_and(~(std::uint64_t{1} << (index % 64)), std::memory_order_release);  [release -> relaxed]": _TWO_HOLDERS,  # noqa: E501
    "std::memory_order_acquire, std::memory_order_relaxed)) {  [acquire -> relaxed]": _TWO_JOBS,
    "slot.left.fetch_add(1, std::memory_order_release);  [release -> relaxed]": _LEFT,
    "left = slot.left.load(std::memory_order_acquire);  [acquire -> relaxed]": _LEFT,
    "moved = slot.left.load(std::memory_order_acquire) != left;  [acquire -> relaxed]": _LEFT,
    "pool.wake.fetch_add(1, std::memory_order_seq_cst);  // pairs with park_worker's, Holding's decrements  [seq_cst -> acq_rel]": _LOST_WAKE,  # noqa: E501
    "spinning = pool.spinning.load(std::memory_order_seq_cst);  [seq_cst -> acquire]": _LOST_WAKE,
    "if (pool.live.load(std::memory_order_seq_cst) < want) {  [seq_cst -> acquire]": _LOST_WAKE,
    "pool.spinning.fetch_add(1, std::memory_order_seq_cst);  [seq_cst -> acq_rel]": _LOST_WAKE,
    "saw = pool.wake.load(std::memory_order_acquire) != observed;  [acquire -> relaxed]": _LOST_WAKE,
    "pool.spinning.fetch_sub(1, std::memory_order_seq_cst);  // out of the count before the wake is compared  [seq_cst -> acq_rel]": _LOST_WAKE,  # noqa: E501
    "moved = pool.wake.load(std::memory_order_seq_cst) != observed;  [seq_cst -> acquire]": _LOST_WAKE,
    "pool.live.fetch_sub(1, std::memory_order_seq_cst);  [seq_cst -> acq_rel]": _LOST_WAKE,
    "if (pool.wake.load(std::memory_order_seq_cst) == observed || !count_worker_live()) {  [seq_cst -> acquire]": _LOST_WAKE,  # noqa: E501
    "observed = pool.wake.load(std::memory_order_acquire);  [acquire -> relaxed]": _LOST_WAKE,
}


# how GenMC words what it found in the code, its one verdict line ("Error:
# Safety violation!", "Error: Non-atomic race!"): never a compiler's error,
# GenMC's own INTERNAL FAILURE, a time limit, or "trace" in some other line
GENMC_FOUND = re.compile(r"^Error: ")


def _genmc(root: Path, harness: str) -> tuple[bool, int, str]:
    # one harness against the header under `root`: whether it passed, the
    # executions it explored, and the first line saying why it didn't
    try:
        done = subprocess.run(
            [
                "/opt/genmc/bin/genmc",
                "--rc11",
                "--check-liveness",
                "--disable-estimation",
                "--",
                "-std=c++17",
                "-ffp-contract=off",
                "-DTRIMKIT_GENMC",
                "-DTRIMKIT_CHECKED",  # the header's own assertions, each a failure GenMC reports
                f"-I{root}",
                str(CHECK / "verify" / "genmc" / f"{harness}.cpp"),
            ],
            capture_output=True,
            text=True,
            timeout=GENMC_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, 0, f"no answer in {GENMC_SECONDS} s"
    found = re.search(r"complete executions explored: (\d+)", done.stdout)
    why = next(
        (line for line in done.stdout.splitlines() if GENMC_FOUND.search(line)),
        done.stderr.strip().splitlines()[-1] if done.returncode != 0 and done.stderr.strip() else "",
    )
    return done.returncode == 0, int(found.group(1)) if found else 0, why


def _weakened(source: str) -> list[tuple[str, str]]:
    # every non-relaxed memory order in `source`, weakened one step, as (the
    # line it is on, the source with only it weakened): seq_cst to what its
    # operation can take below it, acq_rel to acquire and to release, and
    # acquire and release to relaxed
    mutants = []
    traced = re.search(r"#if defined\(TRIMKIT_TRACE\)\nclass TraceStep.*?#else", source, re.S)
    for found in re.finditer(r"std::memory_order_(seq_cst|acq_rel|acquire|release)\b", source):
        if traced and traced.start() <= found.start() < traced.end():
            continue  # a traced build's lock, which GenMC never builds
        order = found.group(1)
        line = source[source.rfind("\n", 0, found.start()) + 1 : source.find("\n", found.end())].strip()
        operation = re.findall(r"\.(\w+)\(", source[max(0, found.start() - 300) : found.start()])
        operation = operation[-1] if operation else ""
        if order == "seq_cst":
            weaker = {"load": ["acquire"], "store": ["release"]}.get(operation, ["acq_rel"])
        elif order == "acq_rel":
            weaker = ["acquire", "release"]
        else:
            weaker = ["relaxed"]
        for to in weaker:
            mutated = source[: found.start()] + f"std::memory_order_{to}" + source[found.end() :]
            mutants.append((f"{line}  [{order} -> {to}]", mutated))
    return mutants


def cmd_genmc(args) -> None:
    """
    Every harness in check/verify/genmc on src/parallel.hpp as it is; then on each
    bug planted for GenMC (CODE_MUTANTS), which must fail some harness; then
    on each mutant of it with one memory order weakened by one step, which
    must fail some harness unless GENMC_MUTANTS_SURVIVING says why it may not.
    """
    found = sorted(p.stem for p in (CHECK / "verify" / "genmc").glob("*.cpp"))
    if found != sorted(GENMC_HARNESSES):
        sys.exit(f"check/verify/genmc has {found}, GENMC_HARNESSES lists {sorted(GENMC_HARNESSES)}")
    workers = len(os.sched_getaffinity(0))
    failed = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        ran = pool.map(lambda h: _genmc(ROOT / "src", h), GENMC_HARNESSES)
        results = dict(zip(GENMC_HARNESSES, ran, strict=True))
    for harness, (passed, executions, why) in results.items():
        floor = GENMC_HARNESSES[harness]
        verdict = "pass" if passed and executions >= floor else "FAIL"
        print(f"  {verdict}  {harness:<14} {executions:>7} executions (floor {floor}) {why}")
        if not passed:
            failed.append(f"{harness}: {why}")
        elif executions < floor:
            failed.append(f"{harness}: {executions} executions, under its floor of {floor}")
    if failed:
        sys.exit("GenMC found:\n  " + "\n  ".join(failed))
    # fastest first, so most mutants are caught after a few seconds
    order = sorted(GENMC_HARNESSES, key=lambda h: results[h][1])
    source = (ROOT / "src" / "parallel.hpp").read_text()
    mutants = _weakened(source)

    def survives(mutant: tuple[str, str]) -> str | None:
        # caught only by what GenMC itself found: a mutant that doesn't
        # compile, or that no harness answers within GENMC_SECONDS, is
        # nothing caught, and stops the stage rather than count as caught
        line, mutated = mutant
        with tempfile.TemporaryDirectory() as scratch:
            # every header the unmutated run reads, that one replaced
            for header in builder.tracked_under(ROOT / "src"):
                if header.suffix == ".hpp" and header.parent == ROOT / "src":
                    shutil.copy(header, scratch)
            (Path(scratch) / "parallel.hpp").write_text(mutated)
            for harness in order:
                passed, _, why = _genmc(Path(scratch), harness)
                if not passed and GENMC_FOUND.search(why):
                    return None
                if not passed:
                    sys.exit(f"GenMC couldn't check {line!r} with {harness}: {why}")
        return line

    bugs = [(name, mutated) for name, (_, mutated) in code_mutants("genmc").items()]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        uncaught = [name for name in pool.map(survives, bugs) if name is not None]
        surviving = [line for line in pool.map(survives, mutants) if line is not None]
    print(f"  {len(bugs) - len(uncaught)} of {len(bugs)} planted bugs caught")
    if uncaught:
        sys.exit("GenMC passed src/parallel.hpp with a planted bug (CODE_MUTANTS): " + ", ".join(uncaught))
    print(f"  {len(mutants) - len(surviving)} of {len(mutants)} weakened memory orders caught")
    unexplained = [line for line in surviving if line not in GENMC_MUTANTS_SURVIVING]
    stale = [line for line in GENMC_MUTANTS_SURVIVING if line not in surviving]
    for line in surviving:
        print(f"  survives: {line}\n            {GENMC_MUTANTS_SURVIVING.get(line, 'UNEXPLAINED')}")
    if unexplained or stale:
        sys.exit(
            "every weakened memory order must fail a harness, or be explained in GENMC_MUTANTS_SURVIVING:\n  "
            + "\n  ".join(
                [f"survives, unexplained: {line}" for line in unexplained]
                + [f"explained, but caught now (remove it): {line}" for line in stale]
            )
        )


# check/verify/tla/Scheduler.tla, the pool's protocol, checked by TLC in every
# interleaving of each scenario: its constants over MODEL_BASE's, and the
# fewest distinct states it must reach, so a scenario that stops exploring
# fails. Then each mutation breaks one guard of the C++ in the model, and TLC
# must find the property it names broken in its scenario
MODEL_BASE = {
    "Callers": 1, "Calls": 2, "Cap": 1, "NSlots": 2, "Items": 2, "ThrowAt": 0, "NestAt": 0,
    "Interrupt": False, "SpawnFails": False, "IdleExit": False, "Fork": False, "AtExit": False,
}  # fmt: skip
MODEL_SCENARIOS = {
    "two sequential calls": ({}, 110_000),
    "two workers": ({"Calls": 1, "Cap": 2}, 130_000),
    "an item fails": ({"Calls": 1, "ThrowAt": 1}, 1_600),
    "ctrl-c": ({"Calls": 1, "Interrupt": True}, 1_500),
    "thread creation fails": ({"Calls": 1, "Cap": 2, "SpawnFails": True}, 130_000),
    "workers exit idle, then start again": ({"IdleExit": True}, 300_000),
    "fork": ({"Calls": 1, "Cap": 2, "Fork": True}, 390_000),
    "the interpreter exits": ({"Calls": 1, "AtExit": True}, 3_400),
    "a nested call": ({"Calls": 1, "NestAt": 1}, 70_000),
    "two callers, one slot": ({"Callers": 2, "Calls": 1, "NSlots": 1}, 390_000),
    "two callers, two slots": ({"Callers": 2, "Calls": 1}, 2_000_000),
    # features together: each alone passing says nothing of the pair
    "fork at the exit": ({"Calls": 1, "Fork": True, "AtExit": True}, 7_000),
    "fork in a nested call": ({"Calls": 1, "Fork": True, "NestAt": 1}, 270_000),
    "fork with two callers": ({"Callers": 2, "Calls": 1, "NSlots": 1, "Fork": True}, 430_000),
    "fork, a failure and ctrl-c": ({"Calls": 1, "Fork": True, "ThrowAt": 1, "Interrupt": True}, 3_000),
    "the exit in a nested call": ({"Calls": 1, "AtExit": True, "NestAt": 1}, 240_000),
}
MODEL_MUTATIONS = {
    "join_ignores_open": ("S1|SlotWord", {"Calls": 1}),
    "close_without_waiting": ("S1|L1|R", {"Calls": 1}),
    "publish_keeps_epoch": ("E", {}),
    "claim_load_then_store": ("S2", {"Calls": 1, "Cap": 2}),
    "park_ignores_wake": ("P1", {}),
    "wake_reads_spinning_first": ("P1", {}),
    "wait_checks_count_first": ("L1", {"Calls": 1}),
    "wait_ignores_epoch": ("L1", {"Callers": 2, "Calls": 1, "NSlots": 1}),
    "exit_without_recheck": ("P2", {"IdleExit": True}),
    # an exit starting another thread for a wake other workers served
    "exit_starts_replacement": ("P3", {"IdleExit": True}),
    "wake_counts_live_first": ("P2", {"IdleExit": True}),
    "fork_fails_nothing": ("R", {"Calls": 1, "Fork": True}),
    "fork_keeps_slots": ("S1", {"Calls": 1, "Fork": True}),
    "fork_closes_wiped_slot": ("SlotWord", {"Calls": 1, "Fork": True}),
    "claims_ignore_exit": ("X", {"Calls": 1, "AtExit": True}),
}
MODEL_PROPERTIES = ["S1", "S2", "S3", "P1", "P2", "R", "X", "E"]
# no run of TLC takes this long but one that can't end: a model whose state
# space has no bound (a mutation that bumps the wake for ever) fails here
MODEL_SECONDS = 3600


def tlc_java(here: Path) -> list[str]:
    # the JVM's options for one TLC, run in `here`. Its heap is bounded, as
    # many run at once as the stage has cores; and its temporary directory is
    # its own, as TLC unpacks its standard modules there under fixed names,
    # and two unpacking at once into /tmp fail to parse them
    (here / "tmp").mkdir(exist_ok=True)
    return ["-Xss64m", "-Xmx3g", "-XX:+UseParallelGC", f"-Djava.io.tmpdir={here / 'tmp'}"]


# The code held to the model (check/verify/tla/SchedulerTrace.tla): a traced build of
# src/parallel.hpp (check/verify/tla/trace.cpp) runs real calls on the real pool,
# round after round, and TLC must find that the model allows every step of
# each log. Then each bug planted for it (CODE_MUTANTS: a step moved, or
# dropped) must make some round's log one the model doesn't allow: so the
# replay is shown able to tell
TRACE_ROUNDS = 8
TRACE_CALLERS, TRACE_CALLS, TRACE_ITEMS = 2, 4, 32
# the cores the traced driver runs on, whatever the stage has: three workers,
# so more than one joins, leaves and is woken at once
TRACE_CORES = 4


class PlantedBug(NamedTuple):
    """
    A bug planted in one of our sources, `path`: each (old, new) of `edits`
    an exact text of it, found once, replaced. `caught_by` names every check that must fail on it:
    "genmc" (a harness of check/verify/genmc), "trace" (some round's log is
    one the model doesn't allow) or "fuzz" (check/fuzz/parallel.cpp crashes
    or hangs), all three on src/parallel.hpp alone; or "test" (the suite, on
    a build with it, `scripts test`).
    """

    caught_by: tuple[str, ...]
    edits: tuple[tuple[str, str], ...]
    path: str = "src/parallel.hpp"
    # What catching it looks like, so a failure for any other reason (a build
    # error, a timeout, an unrelated check under load) is no catch. For the
    # fuzzer: the `expect` message check/fuzz/parallel.cpp must print, or HANG
    # for a bug whose only sign is a wait that never ends (a lost wake), which
    # checked builds never time out of. For the suite: a pattern the failing
    # test's node id must match, or "" for any test the unplanted build passes
    fuzz_verdict: str = ""
    test_verdict: str = ""


# the verdict of a planted bug that only shows as a wait that never ends
HANG = "hang"


# Bugs planted one at a time, each with the checks that must fail on it. A
# check that passes a planted bug fails its stage, and so does an edit that
# no longer applies: so each check is shown able to fail on the real code,
# not only on the model. Not here, as nothing sees them but `bench` (they cost
# speed, never an answer): a spawn chain that stops early, and a caller's
# wait that ignores the epoch
CODE_MUTANTS = {
    "close_without_waiting": PlantedBug(
        ("genmc",),
        (
            (
                "            if (inside) {\n"
                "                wait_for_workers(*slot_, job_, epoch_, poll);\n"
                "            }\n",
                "",
            ),
        ),
    ),
    "join_ignores_open": PlantedBug(
        ("genmc",),
        (
            (
                "while (SlotWord{word}.open() && SlotWord{word}.epoch() == SlotWord{seen}.epoch()) {",
                "while (SlotWord{word}.epoch() == SlotWord{seen}.epoch()) {",
            ),
        ),
    ),
    "touch_job_after_leave": PlantedBug(
        ("genmc",),
        (
            (
                "                holding.leave();\n                job = nullptr;",
                "                holding.leave();\n"
                "                (void)job->stopped.load(std::memory_order_relaxed);\n"
                "                job = nullptr;",
            ),
        ),
    ),
    "claim_load_then_store": PlantedBug(
        ("genmc",),
        (
            (
                "        job.next.fetch_add(take, std::memory_order_relaxed);"
                "  // atomicity alone: each item claimed once (T2)",
                "        job.next.load(std::memory_order_relaxed);\n"
                "    job.next.store(first + take, std::memory_order_relaxed);",
            ),
        ),
    ),
    "open_before_store": PlantedBug(
        ("genmc",),
        (
            (
                """    {
        const TraceStep step("CallerPublishStore");
        reserved->job.store(&job, std::memory_order_relaxed);
    }
    const TraceStep step("CallerPublishOpen");
    // release: pairs with join_job's acquire, publishing the job and its pointer
    const SlotWord opened{reserved->word.fetch_or(SlotWord::OPEN, std::memory_order_release)};
""",
                """    const TraceStep step("CallerPublishOpen");
    const SlotWord opened{reserved->word.fetch_or(SlotWord::OPEN, std::memory_order_release)};
    reserved->job.store(&job, std::memory_order_relaxed);
""",
            ),
        ),
    ),
    "exit_live_before_index": PlantedBug(
        ("genmc",),
        (
            (
                """            give_thread_index();
            // out of the count before it reads the wake (pairs with wake_helpers)
            pool.live.fetch_sub(1, std::memory_order_seq_cst);""",
                """            pool.live.fetch_sub(1, std::memory_order_seq_cst);
            give_thread_index();""",
            ),
        ),
    ),
    # the spinning count read before the wake's bump: a spinner can miss it
    "wake_reads_spinning_first": PlantedBug(
        ("trace",),
        (
            (
                """    {
        const TraceStep step("CallerWake");
        pool.wake.fetch_add(1, std::memory_order_seq_cst);  // pairs with park_worker's, Holding's decrements
    }
    std::uint32_t spinning = 0;
    {
        const TraceStep step("CallerWakeSpinning");
        spinning = pool.spinning.load(std::memory_order_seq_cst);
    }
""",
                """    std::uint32_t spinning = 0;
    {
        const TraceStep step("CallerWakeSpinning");
        spinning = pool.spinning.load(std::memory_order_seq_cst);
    }
    {
        const TraceStep step("CallerWake");
        pool.wake.fetch_add(1, std::memory_order_seq_cst);  // pairs with park_worker's, Holding's decrements
    }
""",
            ),
        ),
    ),
    # no second look after reading `left`: the last leave can slip between
    "wait_skips_recheck": PlantedBug(
        ("trace",),
        (
            (
                """        {
            const TraceStep step("CallerCloseRecheck");
            if (workers_gone(slot, epoch)) {
                return;
            }
        }
""",
                "",
            ),
        ),
    ),
    # `left` bumped before the worker is out of the count
    "leave_bumps_first": PlantedBug(
        ("trace",),
        (
            (
                """    SlotWord before;
    {
        const TraceStep step("WorkerLeave");""",
                """    {
        const TraceStep bump("WorkerLeaveBump");
        slot.left.fetch_add(1, std::memory_order_relaxed);
    }
    SlotWord before;
    {
        const TraceStep step("WorkerLeave");""",
            ),
        ),
    ),
    # the last worker out tells no one: its caller sleeps for ever
    "leave_never_bumps": PlantedBug(
        ("fuzz",),
        (
            (
                "        slot.left.fetch_add(1, std::memory_order_release);\n"
                "        platform::wake_all(slot.left);\n",
                "",
            ),
        ),
        fuzz_verdict=HANG,
    ),
    "leave_never_wakes": PlantedBug(
        ("fuzz",),
        (("        platform::wake_all(slot.left);\n", ""),),
        fuzz_verdict=HANG,
    ),
    "never_wake_sleepers": PlantedBug(
        ("fuzz",),
        (
            (
                "const std::uint32_t woken = platform::wake_some(pool.wake, more - spinning);",
                "const std::uint32_t woken = 0;",
            ),
        ),
        fuzz_verdict="a call worth sharing woke no sleeping worker",
    ),
    "failure_not_stopping": PlantedBug(
        ("fuzz",),
        (
            (
                "    job.stopped.store(true, std::memory_order_relaxed);\n"
                "}\n\n// Items [begin, end) run on this thread",
                "}\n\n// Items [begin, end) run on this thread",
            ),
        ),
        fuzz_verdict="an item that threw left the job running to the end",
    ),
    "fork_fails_nothing": PlantedBug(
        ("fuzz",),
        (("        held->job->forked.store(true, std::memory_order_relaxed);\n", ""),),
        fuzz_verdict="a fork left a shared call to answer",
    ),
    "fork_closes_wiped_slot": PlantedBug(
        ("fuzz",),
        (("        if (generation_ == platform::fork_generation()) {", "        if (slot_ != nullptr) {"),),
        fuzz_verdict="a call forked from its poll answered in the child",
    ),
    # the idle respawn the pool once had: a parked worker gave up at its
    # timeout without comparing the wake, and an exit seeing the wake moved
    # started a replacement, so after a big call then a small one every
    # sleeping worker started another as it went idle
    "idle_exit_respawns": PlantedBug(
        ("fuzz", "test"),
        (
            (
                """        if (moved) {
            return true;
        }
        const std::int64_t now = platform::monotonic_ns();
        if (now >= idle_until) {
            return false;
        }
""",
                """        const std::int64_t now = platform::monotonic_ns();
        if (now >= idle_until) {
            return false;
        }
        if (moved) {
            return true;
        }
""",
            ),
            (
                "                if (pool.wake.load(std::memory_order_seq_cst) == observed ||"
                " !count_worker_live()) {\n"
                "                    return nullptr;  // gone: no caller counted it, or the cap"
                " is full without it\n"
                "                }\n"
                "                // a caller may have counted it live: it stays, and looks\n"
                "                traced.took(1);\n"
                "                if (!take_thread_index()) {  // as at its start: the cap leaves"
                " room for every live one\n"
                '                    checked(false, "a worker that stays finds a thread index'
                ' free");\n'
                "                    pool.live.fetch_sub(1, std::memory_order_seq_cst);\n"
                "                    return nullptr;\n"
                "                }\n"
                "                holding.counted = true;\n"
                "                step = WorkerStep::Scan;\n"
                "                break;\n",
                "                if (pool.wake.load(std::memory_order_seq_cst) != observed) {\n"
                "                    spawn_worker();\n"
                "                }\n"
                "                return nullptr;\n",
            ),
        ),
        fuzz_verdict="P3: a thread started with no call running",
    ),
    "exit_refusal_forgotten": PlantedBug(
        ("fuzz",),
        (("        job.refused_at_exit.store(true, std::memory_order_relaxed);\n", ""),),
        fuzz_verdict="a job stopped at exit returned without saying so",
    ),
    # RayScene.intersect's loop handed a Stop that never stops: Ctrl-C can't land in it
    "intersect_ignores_stop": PlantedBug(
        ("test",),
        (
            (
                "        warn_fallback(trimkit::with_pool([&](const trimkit::parallel::Stop& stop) {\n"
                "            return scene_.intersect(*top, origins.vectors, directions.vectors, n, most,"
                " stop, &columns);\n",
                "        warn_fallback(trimkit::with_pool([&](const trimkit::parallel::Stop& /*stop*/) {\n"
                "            return scene_.intersect(*top, origins.vectors, directions.vectors, n, most,"
                " trimkit::parallel::Stop{}, &columns);\n",
            ),
        ),
        "src/trimkit.cpp",
    ),
    # simplify's work done with the GIL held: no other thread runs until it answers
    "simplify_holds_gil": PlantedBug(
        ("test",),
        (
            (
                "        // converting reads every value, so it runs without the GIL too\n"
                "        const trimkit::GilReleased release;\n",
                "",
            ),
        ),
        "src/trimkit.cpp",
    ),
    # the pool's fork handler never installed: a child keeps its parent's slots, counts and exit gate
    "fork_handler_not_installed": PlantedBug(
        ("test",),
        (("    parallel::install_fork_handler();\n", ""),),
        "src/gil.hpp",
    ),
    # the exit handler never registered: the interpreter finalizes with calls still inside the gate
    "exit_handler_not_registered": PlantedBug(
        ("test",),
        (
            (
                '    nb::module_::import_("atexit").attr("register")'
                "(nb::cpp_function(&trimkit::exit_at_interpreter_exit));\n",
                "",
            ),
        ),
        "src/trimkit.cpp",
    ),
    # check_values refuses nothing: NaN, an infinity and a value past the limit are all read
    "check_values_checks_nothing": PlantedBug(
        ("test",),
        (
            (
                "    if (beyond != 0) {\n        refuse(name + ",
                "    if (beyond > count) {\n        refuse(name + ",
            ),
        ),
        "src/numpy.hpp",
    ),
    # faces compared as signed: an index of -1 is in range, and kept as the largest index there is
    "faces_compared_signed": PlantedBug(
        ("test",),
        (
            (
                "violations += static_cast<std::uint64_t>(index) >= vertex_count ? 1U : 0U;",
                "violations += index >= static_cast<std::int64_t>(vertex_count) ? 1U : 0U;",
            ),
        ),
        "src/numpy.hpp",
    ),
    # faces narrowed to the library's index before they are compared: 2**32 is read as vertex 0
    "faces_narrowed_before_compared": PlantedBug(
        ("test",),
        (
            (
                "violations += static_cast<std::uint64_t>(index) >= vertex_count ? 1U : 0U;",
                "violations += static_cast<Index>(index) >= vertex_count ? 1U : 0U;",
            ),
        ),
        "src/numpy.hpp",
    ),
    # any dtype numpy can cast at all is converted: float64 faces are truncated, not refused
    "converts_unsafe_casts": PlantedBug(
        ("test",),
        (
            (
                'numpy.can_cast(array.attr("dtype"), wanted, "safe")',
                'numpy.can_cast(array.attr("dtype"), wanted, "unsafe")',
            ),
        ),
        "src/numpy.hpp",
    ),
    # a dtype numpy casts safely converted without the warning
    "converts_without_warning": PlantedBug(
        ("test",),
        (
            (
                "    warn_converted(name, source, dtype);\n"
                '    return numpy.array(array, nb::arg("dtype") = wanted,',
                '    return numpy.array(array, nb::arg("dtype") = wanted,',
            ),
        ),
        "src/numpy.hpp",
    ),
    # an array's shape not checked before it is converted: a wrong one is copied, and warned about, first
    "converts_before_shape_checked": PlantedBug(
        ("test",),
        (("if (seen.is_valid() && shaped<Shape...>(seen)) {", "if (seen.is_valid()) {"),),
        "src/numpy.hpp",
    ),
    # an array in the other byte order viewed as this machine's, its bytes never swapped: other values
    "byte_order_viewed_not_swapped": PlantedBug(
        ("test",),
        (
            (
                'return numpy.array(array, nb::arg("dtype") = native, nb::arg("copy") = true,',
                'return numpy.array(array.attr("view")(native), nb::arg("copy") = true,',
            ),
        ),
        "src/numpy.hpp",
    ),
    # instance_index bounded one past the meshes: the mesh after the last is placed
    "instance_index_off_by_one": PlantedBug(
        ("test",),
        (
            (
                "!indexes(instances.index[i], mesh_count)",
                "!indexes(instances.index[i], mesh_count + 1)",
            ),
        ),
        "src/numpy.hpp",
    ),
    # a pair naming one instance twice is measured against itself, not refused
    "pair_of_one_instance_allowed": PlantedBug(
        ("test",),
        (
            (
                "        if (a == b) {\n"
                '            trimkit::refuse("pairs names one instance twice at row " + std::to_string(p));\n'
                "        }\n",
                "",
            ),
        ),
        "src/collision.cpp",
    ),
    # equal vertices each kept as their own: faces name the later ones, and CDT is handed duplicates
    "equal_vertices_not_merged": PlantedBug(
        ("test",),
        (
            (
                "                    first[row] = held - 1;\n",
                "                    first[row] = static_cast<std::uint32_t>(row);\n",
            ),
        ),
        "src/triangulate.cpp",
    ),
    # CDT erases only its super triangle: the triangles outside the shell and inside holes are kept
    "cdt_keeps_outer_triangles": PlantedBug(
        ("test",),
        (("cdt.eraseOuterTrianglesAndHoles();", "cdt.eraseSuperTriangle();"),),
        "src/triangulate.cpp",
    ),
    # boxes that only touch don't overlap, and a 2D box, flat on z, overlaps nothing
    "touching_boxes_do_not_overlap": PlantedBug(
        ("test",),
        (
            (
                "if (b_lo[axis] > a_hi[axis] || b_hi[axis] < a_lo[axis]) {",
                "if (b_lo[axis] >= a_hi[axis] || b_hi[axis] <= a_lo[axis]) {",
            ),
        ),
        "src/bvh.cpp",
    ),
    # nearest boxes ordered by distance alone: of two as near, the walk's first wins, not the lower index
    "nearest_tie_ignores_index": PlantedBug(
        ("test",),
        (
            (
                "return distance < other.distance || (distance == other.distance && box < other.box);",
                "return distance < other.distance;",
            ),
        ),
        "src/bvh.cpp",
    ),
    # a crossing taken for the last one found again by its face alone: the same face of the next
    # instance is skipped. Not the test dropped: a triangle found again would then be counted for ever
    "crossing_found_again_by_face_alone": PlantedBug(
        ("test",),
        (
            ("    std::int64_t last_instance = -1;\n", ""),
            ("if (instance == last_instance && face == last_face) {", "if (face == last_face) {"),
            ("        last_instance = instance;\n", ""),
        ),
        "src/bvh.cpp",
    ),
    # a transform's determinant never looked at: a singular one is placed, with its adjugate as inverse
    "place_skips_determinant": PlantedBug(
        ("test",),
        (
            (
                "bool invertible = trimkit::determinant_of(transform) != 0.0;",
                "bool invertible = true;",
            ),
        ),
        "src/bvh.cpp",
    ),
    # a distance written in coal's order, not the pair's: for a convex solid against a mesh the faces
    # and points trade places and the normal points back
    "witness_swap_ignored": PlantedBug(
        ("test",),
        (
            (
                "write_witness(swapped, result.b1, result.b2, result.nearest_points,",
                "write_witness(false, result.b1, result.b2, result.nearest_points,",
            ),
        ),
        "src/collision.cpp",
    ),
    # pairs culled by their boxes' gap alone: one within the margin whose boxes are apart never touches
    "may_touch_ignores_margin": PlantedBug(
        ("test",),
        (
            (
                "box_gap(a.box, b.box) > request.security_margin",
                "box_gap(a.box, b.box) > 0.0",
            ),
        ),
        "src/collision.cpp",
    ),
    # maximum_error handed to meshoptimizer in the caller's units, not the scaled ones it works in
    "simplify_error_not_scaled": PlantedBug(
        ("test",),
        (
            (
                "const double scaled_error = maximum_error * scale;",
                "const double scaled_error = maximum_error;",
            ),
        ),
        "src/meshoptimizer.cpp",
    ),
    # V-HACD's progress updates never look at the Stop: Ctrl-C waits for the whole decomposition
    "decompose_never_polls": PlantedBug(
        ("test",),
        (
            (
                "        if (trimkit::stopped(stop_)) {",
                "        if (stop_.poll == nullptr && trimkit::stopped(stop_)) {",
            ),
        ),
        "src/vhacd.cpp",
    ),
    # a device's log, grown for a query with many crossings, kept after it
    "device_log_kept": PlantedBug(
        ("test",),
        (("        settle();\n    } catch (...) {", "    } catch (...) {"),),
        "src/vulkan/vulkan.hpp",
    ),
    # xatlas left to print its warnings to the caller's terminal
    "xatlas_prints": PlantedBug(
        ("test",),
        (("        xatlas::SetPrint(nullptr, false);\n", ""),),
        "src/xatlas.cpp",
    ),
    # the broadphase's boxes unpadded: rounding in placing them culls pairs that touch
    "collision_cull_unpadded": PlantedBug(
        ("test",),
        (("    return trimkit::BOX_SLACK * farthest;", "    return 0.0 * farthest;"),),
        "src/collision.cpp",
    ),
    # a point's value read wherever the buffer maps it, past the values its attribute holds
    "draco_mapped_value_unchecked": PlantedBug(
        ("test",),
        (
            (
                "            if (at.value() >= held) {\n"
                "                malformed = true;\n"
                "                break;\n"
                "            }\n",
                "",
            ),
        ),
        "src/draco.cpp",
    ),
}


def code_mutants(checker: str) -> dict[str, tuple[str, str]]:
    # each planted bug `checker` must catch, by name: (its path, that file
    # with the bug in it). Every bug is applied, so none goes stale unseen
    mutated = {}
    for name, bug in CODE_MUTANTS.items():
        source = (ROOT / bug.path).read_text()
        for old, new in bug.edits:
            if source.count(old) != 1:
                sys.exit(f"CODE_MUTANTS: {name} no longer applies to {bug.path}")
            source = source.replace(old, new)
        unknown = set(bug.caught_by) - {"genmc", "trace", "fuzz", "test"}
        if unknown or (bug.path != "src/parallel.hpp" and bug.caught_by != ("test",)):
            sys.exit(f"CODE_MUTANTS: {name} can't be caught by {bug.caught_by}")
        if ("fuzz" in bug.caught_by) != bool(bug.fuzz_verdict):
            sys.exit(f"CODE_MUTANTS: {name} must name its fuzz_verdict exactly when the fuzzer catches it")
        if checker in bug.caught_by:
            mutated[name] = (bug.path, source)
    return mutated


def _tlc(spec: Path, name: str, constants: dict, mutation: str) -> tuple[int, str, dict[str, int]]:
    # TLC on one configuration: the distinct states it found, the property it
    # found broken ("" when none was), and how often each step's action was
    # taken (from TLC's coverage of each arm of Step)
    values = {**MODEL_BASE, **constants, "Mutation": mutation}
    lines = [f"    {key} = {str(value).upper() if isinstance(value, bool) else value}"
             for key, value in values.items() if key != "Mutation"]  # fmt: skip
    # each in its own directory, as TLC writes files beside the spec when it
    # finds something broken
    here = spec / re.sub(r"[^a-z0-9]+", "_", name)
    here.mkdir()
    for module in ("Scheduler.tla", "SlotWordTable.tla"):
        shutil.copy(spec / module, here)
    config = here / "run.cfg"
    config.write_text(
        "SPECIFICATION Spec\nCONSTANTS\n" + "\n".join(lines) + f'\n    Mutation = "{mutation}"\n'
        f"INVARIANTS {' '.join(MODEL_PROPERTIES)}\nPROPERTIES L1 P3\n"
    )
    # liveness checked once, at the end: TLC otherwise checks it on a clock
    # as it explores, so on a loaded machine a mutation breaking both an
    # invariant and L1 was reported as either, as the clock fell
    try:
        done = subprocess.run(
            [tool("java"), *tlc_java(here), "-cp", "/opt/tla2tools.jar", "tlc2.TLC", "-lncheck", "final",
             "-deadlock", "-workers", "1", "-coverage", "100000", "-metadir", str(here / "states"),
             "-config", config.name, "Scheduler.tla"],
            cwd=here, capture_output=True, text=True, check=False, timeout=MODEL_SECONDS,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return 0, f"TLC failed: no answer in {MODEL_SECONDS} s", {}
    finally:
        shutil.rmtree(here / "states", ignore_errors=True)  # gigabytes, for the biggest
    (here / "tlc.log").write_text(done.stdout + done.stderr)
    found = re.findall(r"([\d,]+) distinct states found", done.stdout)  # the last: the whole run's
    states = int(found[-1].replace(",", "")) if found else 0
    taken = {}
    for number, text in enumerate((spec / "Scheduler.tla").read_text().splitlines(), 1):
        arm = re.search(r"-> ((?:Caller|Worker)\w+)\(t, f\)", text)
        if arm:
            column = arm.end(1) + 2  # TLC counts an arm by its first argument, `t`
            at = rf"line {number}, col {column} to line \d+, col \d+ of module Scheduler: (\d+)"
            counts = re.findall(at, done.stdout)
            taken[arm.group(1)] = max((int(c) for c in counts), default=0)
    broken = re.search(
        r"Invariant (\w+) is violated|Action property (\w+) is violated|Temporal properties were (violated)",
        done.stdout,
    )
    if broken:
        return states, broken.group(1) or broken.group(2) or "L1", taken
    # a step outside a transition's precondition: outside SlotWordTable's domain
    if "Attempted to apply function" in done.stdout:
        return states, "SlotWord", taken
    if done.returncode != 0 or "No error has been found" not in done.stdout:
        error = [line for line in done.stdout.splitlines() if "rror" in line]
        why = error[0] if error else done.stdout[-300:]
        return states, f"TLC failed ({here / 'tlc.log'}): {why}", taken
    return states, "", taken


def _traces(spec: Path, src: Path, name: str) -> list[tuple[Path, dict]]:
    # the traced driver built on the header in `src`, then TRACE_ROUNDS logs
    # of it on TRACE_CORES cores, each with the constants the model replays
    # it with
    here = spec / f"trace_{name}"
    here.mkdir()
    driver = here / "trace"
    run(tool("clang++"), "-std=c++20", "-O1", "-pthread", f"-I{src}", CHECK / "verify" / "tla" / "trace.cpp",
        "-o", driver)  # fmt: skip
    cores = sorted(os.sched_getaffinity(0))[:TRACE_CORES]
    if len(cores) < TRACE_CORES:
        print(
            f"  traces on {len(cores)} cores, not {TRACE_CORES}: fewer workers join, leave and wake at once"
        )
    logs = []
    for round_ in range(TRACE_ROUNDS):
        log = here / f"round{round_}" / "TraceLog.tla"
        log.parent.mkdir()
        printed = subprocess.run(
            [str(driver), str(TRACE_CALLERS), str(TRACE_CALLS), str(TRACE_ITEMS), str(round_), str(log)],
            capture_output=True, text=True, check=True, preexec_fn=lambda: os.sched_setaffinity(0, cores),
        ).stdout  # fmt: skip
        constants = {key: int(value) for key, value in re.findall(r"^(\w+) = (\d+)$", printed, re.M)}
        logs.append((log, constants))
    return logs


def _replay(spec: Path, log: Path, constants: dict) -> tuple[bool, int, int]:
    # TLC walking the model along one log: whether it reached the end, how
    # many records it matched, and how many there were. TLC failing any other
    # way (a log outside the generated tables, a spec that doesn't parse) is
    # no verdict on the code, and stops the stage
    here = log.parent
    for module in ("Scheduler.tla", "SchedulerTrace.tla", "SlotWordTable.tla"):
        shutil.copy(spec / module, here)
    values = {**MODEL_BASE, "IdleExit": True, **{k: v for k, v in constants.items() if k != "records"}}
    values["Interrupt"] = bool(values["Interrupt"])
    lines = [
        f"    {key} = {str(value).upper() if isinstance(value, bool) else value}"
        for key, value in values.items()
    ]
    (here / "replay.cfg").write_text(
        "SPECIFICATION TraceSpec\nCONSTANTS\n"
        + "\n".join(lines)
        + '\n    Mutation = "none"\nINVARIANTS Unmatched\n'
    )
    done = subprocess.run(
        [tool("java"), *tlc_java(here), "-cp", "/opt/tla2tools.jar", "tlc2.TLC", "-deadlock",
         "-workers", "1", "-metadir", str(here / "states"), "-config", "replay.cfg", "SchedulerTrace.tla"],
        cwd=here, capture_output=True, text=True, check=False,
    )  # fmt: skip
    (here / "tlc.log").write_text(done.stdout + done.stderr)
    accepted = "Invariant Unmatched is violated" in done.stdout
    if not accepted and "Model checking completed. No error has been found" not in done.stdout:
        sys.exit(f"TLC couldn't replay {log}: see {here / 'tlc.log'}")
    matched = [int(n) for n in re.findall(r'<<"matched", (\d+)>>', done.stdout)]
    return accepted, max(matched, default=0), constants.get("records", 0)


def cmd_model(args) -> None:
    """
    TLC on check/verify/tla/Scheduler.tla, its slot word arithmetic generated from
    the C++ first (check/verify/tla/slot_word_table.cpp): every scenario must pass
    and reach its floor of states, every mutation of the model must break the
    property it names, every traced run of the code must be one the model
    allows, and every bug planted in the code for the trace to catch
    (CODE_MUTANTS) must make some run one it doesn't.
    """
    spec = args.root / "model"
    shutil.rmtree(spec, ignore_errors=True)
    spec.mkdir(parents=True)
    shutil.copy(CHECK / "verify" / "tla" / "Scheduler.tla", spec)
    shutil.copy(CHECK / "verify" / "tla" / "SchedulerTrace.tla", spec)
    table = spec / "slot_word_table"
    run(tool("clang++"), "-std=c++20", "-O1", CHECK / "verify" / "tla" / "slot_word_table.cpp", "-o", table)
    (spec / "SlotWordTable.tla").write_text(run(table, capture=True))
    failed = []
    traced = _traces(spec, ROOT / "src", "code")
    mutants = {}
    for name, (_, source) in code_mutants("trace").items():
        mutated = spec / f"src_{name}"
        mutated.mkdir()
        for header in (ROOT / "src").glob("*.hpp"):
            shutil.copy(header, mutated)
        (mutated / "parallel.hpp").write_text(source)
        mutants[name] = mutated
    with concurrent.futures.ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        replays = {log: pool.submit(_replay, spec, log, constants) for log, constants in traced}
        mutant_replays = {
            name: [pool.submit(_replay, spec, log, constants) for log, constants in _traces(spec, src, name)]
            for name, src in mutants.items()
        }
        scenarios = {
            name: pool.submit(_tlc, spec, name, constants, "none")
            for name, (constants, _) in MODEL_SCENARIOS.items()
        }
        mutations = {
            name: pool.submit(_tlc, spec, f"mutation {name}", constants, name)
            for name, (_, constants) in MODEL_MUTATIONS.items()
        }
        taken: dict[str, int] = {}
        for name, (_, floor) in MODEL_SCENARIOS.items():
            states, broken, counts = scenarios[name].result()
            for action, count in counts.items():
                taken[action] = taken.get(action, 0) + count
            verdict = "pass" if not broken and states >= floor else "FAIL"
            print(f"  {verdict}  {name:<38} {states:>9} states (floor {floor}) {broken}", flush=True)
            if broken:
                failed.append(f"{name}: {broken}")
            elif states < floor:
                failed.append(f"{name}: {states} states, under its floor of {floor}")
        actions: dict[str, int] = {}
        for log, replay in replays.items():
            accepted, matched, records = replay.result()
            for action in re.findall(r'action \|-> "(\w+)"', log.read_text()):
                actions[action] = actions.get(action, 0) + 1
            verdict = "pass" if accepted else "FAIL"
            print(
                f"  {verdict}  the code's trace {log.parent.name:<26} {records:>6} steps, {matched} matched"
            )
            if not accepted:
                failed.append(f"the code took a step the model doesn't allow: {log}, record {matched + 1}")
        print("  steps traced: " + ", ".join(f"{a} {n}" for a, n in sorted(actions.items())), flush=True)
        for name, runs in mutant_replays.items():
            outcomes = [run.result() for run in runs]
            rejected = sum(1 for accepted, _, _ in outcomes if not accepted)
            verdict = "pass" if rejected else "FAIL"
            print(f"  {verdict}  planted bug {name:<27} rejected in {rejected} of {len(outcomes)} rounds")
            if not rejected:
                failed.append(f"planted bug {name}: every round's log was one the model allows")
        # a step no scenario takes is one no scenario checks
        never = sorted(a for a, count in taken.items() if count == 0 and a not in MUTATION_ONLY_STEPS)
        if not taken or never:
            failed.append("no scenario takes " + (", ".join(never) or "any step: coverage unread"))
        for name, (expected, _) in MODEL_MUTATIONS.items():
            states, broken, _ = mutations[name].result()
            caught = bool(re.fullmatch(expected, broken))
            verdict = "pass" if caught else "FAIL"
            print(f"  {verdict}  mutation {name:<29} broke {broken or 'nothing'}", flush=True)
            if not caught:
                failed.append(f"mutation {name}: expected {expected} broken, found {broken or 'nothing'}")
    if failed:
        sys.exit("the model found:\n  " + "\n  ".join(failed))


def cmd_static(args) -> None:
    # the checks that need a build: lint's are `lint`
    for command in (
        # the sdist builds; building and testing from it alone is `sdist`,
        # which release CI runs, as it's a whole build and suite again
        lambda args: sdist(args.root),
        cmd_stubs,
        cmd_audit,
        cmd_tidy,
        cmd_threads,
    ):
        command(args)


def cmd_wheel(args) -> None:
    # cibuildwheel by hash from uv.lock, configured by pyproject.toml, on the
    # tracked files alone (its containers copy whatever directory they build)
    # plus the requirements its hooks install by hash with uv pip
    source = builder.export_tracked(args.root / "wheel-source")
    # what the build and its tests install inside each image, and what the
    # audit installs on the host: apart, as the audit tools need wheels (such
    # as cryptography's) that not every image's platform has
    export(source / "requirements.txt", "build", "test")
    export(source / "audit.txt", "audit")
    # where [toolchain] fetches the compiler (Windows, which cibuildwheel
    # builds on this machine), that LLVM first on PATH, once per hash
    here = builder.Platform.current()
    env = None
    if pinned := MANIFEST.fetch.get(f"{here.os}-{here.machine}"):
        llvm = args.root / "toolchain" / pinned["sha256"]
        if not llvm.exists():
            partial = llvm.with_name(llvm.name + ".partial")
            fetch(pinned["url"], pinned["sha256"], partial)
            partial.rename(llvm)
        (tree,) = llvm.iterdir()
        env = {"PATH": os.pathsep.join([str(tree / "bin"), os.environ["PATH"]])}
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
        env=env,
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
        machine = args.machine or distro.host_machine()
        # under qemu every instruction costs ten: half the machine, as `genmc` gets
        cores = VERIFY_CORES if distro.emulated(machine) else CORES
        distro.run(
            args.root / "distro",
            [tool("docker"), "run", "--rm", *docker_cores(cores)],
            f"{os.getuid()}:{os.getgid()}",
            machine,
        )


def cmd_fetch(args) -> None:
    # re-derive the named upstreams (default all) from their pins, or --latest
    held = vendor.fetch(args.names, args.latest)
    if held:
        sys.exit("patches that no longer apply, their upstreams left as they were:\n  " + "\n  ".join(held))


def cmd_patch(args) -> None:
    # vendor/patches/ from the `//// TRIMKIT PATCH` blocks in vendor/ (default
    # every upstream), each fetched back byte for byte (scripts/vendor.py)
    try:
        vendor.patch(args.names)
    except ValueError as error:
        sys.exit(str(error))


def cmd_outdated(args) -> None:
    vendor.outdated()


def cmd_lock(args) -> None:
    # every dependency to its newest release past the lock's one-week cooldown
    run(tool("uv"), "lock", "--upgrade")


def cmd_update(args) -> None:
    # every upstream to its ref's newest commit, every dependency, action and
    # the image's base to its newest release a week old, the stubs that quote
    # those commits, then what moved, for `check` to run on. An upstream whose
    # patch no longer applies is left exactly as it was, and named at the end
    held = vendor.fetch([], latest=True)
    vendor.update_actions()
    vendor.update_base_image()
    cmd_lock(args)
    stubs(args.root, write=True)
    pins = ["trimkit.toml", "uv.lock", ".github", "scripts/docker"]
    run(tool("git"), "-C", ROOT, "diff", "--stat", "--", *pins)
    # an upstream whose patch no longer applies stays at its pin, for a person
    # to rebase the patch or drop it: everything else has moved
    for reason in held:
        print(f"held: {reason}", flush=True)


SHADERS = ROOT / "src" / "vulkan"
SPIRV = SHADERS / "spirv.hpp"


def shaders() -> list[Path]:
    # every shader: the kernel, trace.comp, and layout.comp, which lays its answers out
    return sorted(p for p in builder.tracked_under(SHADERS) if p.suffix == ".comp")


def shader(write: bool) -> None:
    """
    Compile every src/vulkan/*.comp to SPIR-V with the image's glslang and write them
    as the words src/vulkan/vulkan.hpp embeds (so no build or wheel needs a shader
    compiler), or fail if the checked-in words drifted from the sources.
    """
    sources = shaders()
    arrays = []
    with tempfile.TemporaryDirectory() as fresh:
        for source in sources:
            compiled = Path(fresh) / f"{source.stem}.spv"
            binary = Path(fresh) / f"{source.stem}.stripped.spv"
            run(tool("glslang"), "-V", "--target-env", "vulkan1.0", "-o", compiled, source, capture=True)
            # without the names glslang keeps for debuggers: a sixth smaller, the
            # same code (spirv-opt's -O was measured 4-6% slower on NVIDIA: not used)
            run(tool("spirv-opt"), "--strip-debug", compiled, "-o", binary, capture=True)
            # a malformed module fails here, not in a driver
            run(tool("spirv-val"), "--target-env", "vulkan1.0", binary, capture=True)
            words = memoryview(binary.read_bytes()).cast("I")
            rows = [", ".join(f"0x{w:08x}" for w in words[i : i + 8]) for i in range(0, len(words), 8)]
            name = f"{source.stem.upper()}_SPIRV"
            # the numbers a shader defines, as the host must agree with them
            defines = re.findall(r"^#define (\w+) (\d+)u?$", source.read_text(), flags=re.MULTILINE)
            for define, value in defines:
                arrays.append(f"inline constexpr std::uint32_t {source.stem.upper()}_{define} = {value};\n")
            words = ",\n    ".join(rows)
            arrays.append(f"inline constexpr std::uint32_t {name}[] = {{\n    {words},\n}};\n")
    digests = [(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]
    named = "\n".join(f"// src/vulkan/{name} (sha256 {digest})" for name, digest in digests)
    text = (
        "// The shaders as SPIR-V, written by `uv run scripts generate`: what\n"
        "// src/vulkan/vulkan.hpp hands the device. Never edited by hand.\n"
        f"{named}\n"
        "// clang-format off\n"
        "#pragma once\n\n#include <cstdint>\n\n"
        "namespace trimkit::vulkan {\n\n" + "\n".join(arrays) + "\n}  // namespace trimkit::vulkan\n"
    )
    if write:
        SPIRV.write_text(text)
    elif not SPIRV.exists() or SPIRV.read_text() != text:
        sys.exit("src/vulkan/spirv.hpp drifted from src/vulkan/*.comp; run `uv run scripts generate`")


def cmd_shader(args) -> None:
    shader(write=False)


def cmd_generate(args) -> None:
    # everything derived from the sources: the shaders' SPIR-V, the stubs from
    # the module the dev build makes, the README's API summary from the stubs,
    # and the golden answers every platform is checked against
    shader(write=True)
    stubs(args.root, write=True)
    run(dev(args.root).python, CHECK / "tests" / "test_golden.py", "--write", cwd=CHECK / "tests")


# what `check` runs, in order: every stage runs even after one fails
# the gate, which should take minutes after a small change: the distribution
# packages (`distro --run`, two more builds and suites) are checked when the
# packaging changes, and the sdist's own build (`sdist`) in release CI
STAGES = ["lint", "test", "static", "asan", "tsan", "fuzz", "bench", "genmc", "model"]


def cmd_check(args) -> None:
    # a gate that can't show every planted bug says so, rather than passing
    # without them
    if (os.cpu_count() or 1) < PLANTED_TEST_CORES:
        sys.exit(
            f"check shows every planted bug on {PLANTED_TEST_CORES} cores, of which this machine has fewer"
        )
    results = []
    for stage in STAGES:
        command = [sys.executable, str(ROOT / "scripts"), "--root", str(args.root), stage]
        if stage == "fuzz":
            command += ["--seconds", str(args.fuzz_seconds)]
        if stage == "test":
            # every planted bug shows (PLANTED_TEST_CORES), on any machine with that many
            cores = min(os.cpu_count() or 1, max(VERIFY_CORES, PLANTED_TEST_CORES))
            command += ["--cores", str(cores)]
        if stage == "bench":
            command += ["--quick"]  # the short calls, in seconds: a pool that works but stalls fails here
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
# the dev image with the checkers built on it (GenMC and TLC), for the
# `genmc` and `model` stages alone
VERIFY_IMAGE = "trimkit-verify"
VERIFIES = {"genmc", "model"}
# every container of ours gets this many cores, the last ones, so a check
# never takes over a shared machine: by affinity rather than a quota, so the
# thread pool (which counts the cores it may run on) sizes itself to them
CORES = 2
# but the `genmc` and `model` stages, which run dozens of single-threaded
# checkers and nothing that measures time (the traced driver pins itself to
# TRACE_CORES): half the machine, or they are most of an hour on two cores.
# And `fuzz`, which measures no time either, and whose pool harness shows an
# idle respawn (CODE_MUTANTS) only with more than one worker to go idle: at
# least PLANTED_TEST_CORES, as the suite's planted bugs need, so a 4-core
# runner checks everything a 32-core machine does
VERIFY_CORES = min(os.cpu_count() or CORES, max(PLANTED_TEST_CORES, (os.cpu_count() or CORES) // 2))
HALF_MACHINE = VERIFIES | {"fuzz"}


def core_count(text: str) -> int:
    # `--cores`: how many of the last cores, or `all` of them
    return (os.cpu_count() or 1) if text == "all" else int(text)


def docker_cores(cores: int) -> list[str]:
    total = os.cpu_count() or 1
    if cores > total:
        sys.exit(f"{cores} cores asked for, of the {total} this machine has")
    first = total - cores
    return ["--cpuset-cpus", f"{first}-{total - 1}"]


# on the host: `check` dispatches stages that each enter the image, `lock`
# needs only uv, `wheel` and `distro` run their own containers, `image`
# builds it, and `outdated` only asks the upstreams
ON_HOST = {"check", "lock", "wheel", "image", "outdated", "distro"}
SANITIZED = {"asan", "tsan", "fuzz"}
WRITES_SOURCES = {"generate", "fetch", "patch", "update", "format"}


def cmd_image(args) -> None:
    # docker's layer cache makes this quick unless the Dockerfile or lock changed
    run(tool("docker"), "build", "-q", "-f", DOCKER / "Dockerfile", "-t", IMAGE, ROOT, capture=True)
    if getattr(args, "command", None) in VERIFIES:
        verify = DOCKER / "Dockerfile.verify"
        run(tool("docker"), "build", "-q", "-f", verify, "-t", VERIFY_IMAGE, DOCKER, capture=True)


def in_image(args) -> None:
    """
    Run this same command inside the image, as the caller's uid so files it
    writes under build/ stay theirs, with the sources read-only.
    """
    cmd_image(args)
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
        *docker_cores(
            args.cores
            if args.command in ("bench", "test")
            else VERIFY_CORES
            if args.command in HALF_MACHINE
            else CORES
        ),
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
    # a git worktree's .git is a file naming the main repository's directory:
    # mounted read-only at its own path, so `git ls-files` resolves inside too
    dotgit = ROOT / ".git"
    if dotgit.is_file():
        main_git = Path(dotgit.read_text().split("gitdir:", 1)[1].strip()).parents[1]
        docker += ["-v", f"{main_git}:{main_git}:ro"]
    inside = ["python3", "/src/scripts", "--inside", "--root", "/src/build/docker"]
    if args.command == "test":
        # and every planted bug's build, where the suite will see them: on
        # cores enough to show them, the one place that is decided
        planted = ["--planted"] if args.cores >= PLANTED_TEST_CORES and "--planted" not in args.given else []
        args.given += planted
        run(*docker, IMAGE, *inside, "build", *planted)
        # no swap to page out to, as the limit is what it's testing
        docker += ["--memory", "2g", "--memory-swap", "2g"]
    # the command as it was given, every option of it
    run(*docker, VERIFY_IMAGE if args.command in VERIFIES else IMAGE, *inside, *args.given)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT / "build", help="where builds and venvs go")
    # set when already running inside the image
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    # only the documented commands show in the usage and help
    documented = {
        "check": "the gate: lint, test, static, ASan, TSan, fuzz, bench --quick, genmc and model",
        "lint": "the fast deterministic checks, nothing compiled",
        "test": "the strict build and the tests in 2 GB, on the last 2 cores (--cores N or all)",
        "static": "the checks that need a build: clang-tidy, the sdist, stubs and the wheel audit",
        "bench": "the pool against hand-written pools, 144 cells (--quick: the short calls)",
        "distro": "Debian and Fedora packaging (--run: build, test and lint them; --machine i686|s390x)",
        "genmc": "model check src/parallel.hpp itself with GenMC, and its weakened memory orders",
        "model": "check the pool's protocol in check/verify/tla with TLC, and that each mutation is caught",
        "fuzz": "every fuzzer, native then Atheris, --seconds each",
        "generate": "regenerate the shaders' SPIR-V, the stubs, the README API summary and golden.json",
        "patch": "write vendor/patches/ from the //// TRIMKIT PATCH blocks in vendor/",
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
            if name == "cmd_fuzz":
                command.add_argument("--seconds", type=int, default=60)
            if name in ("cmd_fetch", "cmd_patch"):
                command.add_argument("names", nargs="*")
            if name == "cmd_build":
                command.add_argument(
                    "--planted", action="store_true", help="and a wheel for each bug planted for `test`"
                )
            if name == "cmd_test":
                # set by `in_image` alone: whether the cores asked for show the planted bugs
                command.add_argument("--planted", action="store_true", help=argparse.SUPPRESS)
            if name in ("cmd_bench", "cmd_test"):
                command.add_argument(
                    "--cores",
                    type=core_count,
                    default=BENCH_CORES if name == "cmd_bench" else CORES,
                    help="the last this many cores, or all",
                )
            if name == "cmd_bench":
                command.add_argument(
                    "--quick",
                    action="store_true",
                    help="only cells of at most 20 ms serial: seconds, not 30 minutes",
                )
            if name == "cmd_fetch":
                command.add_argument("--latest", action="store_true", help="move each pin to its ref's HEAD")
            if name == "cmd_distro":
                command.add_argument(
                    "--run", action="store_true", help="build, test and lint them in containers"
                )
                command.add_argument(
                    "--machine",
                    choices=sorted(distro.PLATFORMS),
                    help="in that machine's images (default this one's): i686 natively, others under qemu",
                )
            if name == "cmd_check":
                command.add_argument("--fuzz-seconds", type=int, default=10, help="seconds per fuzzer")
    # `uv run scripts -- static` reads the same as `uv run scripts static`
    argv = sys.argv[1:]
    argv = argv[1:] if argv[:1] == ["--"] else argv
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    # the command and its own options, as given, for `in_image` to pass on
    args.given = argv[argv.index(args.command) :]
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
