"""
The build system: trimkit.toml in, one ninja file out.

A wheel here is one shared library compiled from a fixed list of sources by
one compiler, so there is nothing to detect, generate or configure: a build
is a list of compile lines and one link line. That list is what this file
writes, and ninja does the hard part (dependencies from the compiler's own
depfiles, parallelism, incremental rebuilds) on every platform:

  load()    trimkit.toml as data: every upstream, part and accepted toolchain
  Target    what one build is for: platform, Python, options, checked tools
  tables    every flag, with the reason for it, by platform
  plan()    the two as ninja's text: same inputs, same bytes
  write()   that text into the build directory, the only IO

To see what a build does, read its `build.ninja`. Every build is clang, so one
dialect of flags serves Linux, macOS and Windows, and `check` plans every
platform from Linux, so a Windows-only mistake fails here first.

Standard library only: this runs inside the build backend (scripts/backend.py).
"""

import dataclasses
import functools
import platform as host
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOML = ROOT / "trimkit.toml"
# the one symbol the Linux extension exports, its init function
EXPORTS = ROOT / "src" / "extension.map"
# nanobind's headers, the one source of its runtime and its list of the C API
NANOBIND = ROOT / "vendor" / "nanobind"
# the one extension: the package `trimkit` itself, every part compiled into it
EXTENSION = "trimkit"


# -- the repository -------------------------------------------------------------


@functools.cache
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def macos_deployment_target() -> str:
    # the oldest macOS the wheels support: cibuildwheel sets it from
    # pyproject.toml, and delocate checks the wheels against it
    return pyproject()["tool"]["cibuildwheel"]["macos"]["environment"]["MACOSX_DEPLOYMENT_TARGET"]


@functools.cache
def tracked() -> frozenset[Path]:
    """
    Every file git tracks (staged ones included), as it is on disk: the only
    files any command looks at, so untracked scratch files change nothing.
    """
    git = shutil.which("git")
    if git is None:
        sys.exit("`git` is not installed: every command reads the files git tracks")
    listing = subprocess.run(
        [git, "-C", str(ROOT), "ls-files", "-z"], check=True, stdout=subprocess.PIPE
    ).stdout
    return frozenset(p for p in (ROOT / name for name in listing.decode().split("\0") if name) if p.is_file())


def tracked_under(directory: Path) -> list[Path]:
    return sorted(p for p in tracked() if p.is_relative_to(directory))


def export_tracked(destination: Path) -> Path:
    # every tracked file, modification times kept so ninja rebuilds only what
    # changed: what every build and sdist starts from
    shutil.rmtree(destination, ignore_errors=True)
    for path in tracked():
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return destination


# -- the manifest ---------------------------------------------------------------


@dataclass(frozen=True)
class Upstream:
    """
    One `[upstream.NAME]`: trimkit.toml's header describes each key.
    """

    name: str
    spdx: str
    repo: str = ""
    host: str = "github.com"
    ref: str = ""
    commit: str = ""
    tree_sha256: str = ""
    copyright: str = ""
    keep: tuple[str, ...] = ()
    license_from: str = ""
    patches: dict[str, str] = field(default_factory=dict)
    include: tuple[str, ...] = ()
    system: str = ""
    system_directory: str = ""
    distro: dict[str, str] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    defines: dict[str, str] = field(default_factory=dict)
    no_sanitize: dict[str, str] = field(default_factory=dict)

    @property
    def compiled(self) -> list[Path]:
        # the files it compiles, sorted: on disk, as a build sees them (it
        # runs on an export or an sdist, without git)
        return sorted({p for pattern in self.sources for p in ROOT.glob(pattern) if p.is_file()})

    @property
    def described(self) -> str:
        return f"{self.repo} {self.commit[:7]}" + (" +patches" if self.patches else "")


@dataclass(frozen=True)
class Part:
    """
    One `[part.NAME]`: src/NAME.cpp, whose `bind_NAME` adds to the module.
    """

    name: str
    upstreams: tuple[Upstream, ...]
    fuzz: tuple[str, ...] = ()
    seeds: str = ""

    @property
    def source(self) -> Path:
        return ROOT / "src" / f"{self.name}.cpp"


@dataclass(frozen=True)
class Manifest:
    upstreams: dict[str, Upstream]
    parts: dict[str, Part]
    # [extension]: the most bytes it may be stripped in a wheel, and the
    # upstreams every part compiles against
    budget: int
    extension_upstreams: tuple[Upstream, ...]
    # [toolchain]: each an inclusive [oldest, newest] of major versions
    llvm: tuple[int, int]
    apple_clang: tuple[int, int]
    # [distribution]: what `uv run scripts distro` writes packaging for, by name
    distributions: dict[str, dict]

    @property
    def harnesses(self) -> list[str]:
        return [h for part in self.parts.values() for h in part.fuzz]

    @property
    def sources(self) -> list[Path]:
        # our C++ that compiles, held to our warnings
        return sorted([ROOT / "src" / f"{EXTENSION}.cpp", *(part.source for part in self.parts.values())])

    @property
    def compiled_in(self) -> list[Upstream]:
        # every upstream the extension compiles in, in trimkit.toml's order
        used = {u.name for part in self.parts.values() for u in part.upstreams}
        used |= {u.name for u in self.extension_upstreams}
        return [u for u in self.upstreams.values() if u.name in used]

    @property
    def directories(self) -> list[Path]:
        # where our C++ is, headers included
        return [ROOT / "src"]


def _tuples(table: dict) -> dict:
    # TOML arrays are lists; the dataclasses hold tuples, frozen like them
    return {k: tuple(v) if isinstance(v, list) else v for k, v in table.items()}


@functools.cache
def load() -> Manifest:
    """
    trimkit.toml as data. A key no dataclass has is an error, so a build quirk
    can only be added as data with a place here; the rest is the rules a
    build relies on, which a typo in the manifest could otherwise break.
    """
    raw = tomllib.loads(TOML.read_text())
    upstreams, errors = {}, []
    for name, table in raw["upstream"].items():
        try:
            upstreams[name] = Upstream(name=name, **_tuples(table))
        except TypeError as error:
            errors.append(f"[upstream.{name}]: {error}")
            continue
        # a system's copy of an upstream can't carry our patches
        if table.get("system") and table.get("patches"):
            errors.append(f"[upstream.{name}]: a patched upstream can't come from the system")
    parts = {}
    for name, table in raw["part"].items():
        wanted = table["upstreams"]
        errors += [f"[part.{name}]: no [upstream.{u}]" for u in wanted if u not in upstreams]
        # what an upstream compiles with, a part compiling it must list too
        errors += [
            f"[part.{name}]: compiles {u}, which requires {r}, so it must list it too"
            for u in wanted
            for r in (upstreams[u].requires if u in upstreams else ())
            if r not in wanted
        ]
        if not errors:
            values = {**_tuples(table), "upstreams": tuple(upstreams[u] for u in wanted)}
            parts[name] = Part(name=name, **values)
    if errors:
        sys.exit("trimkit.toml:\n  " + "\n  ".join(errors))
    toolchain = raw["toolchain"]
    return Manifest(
        upstreams,
        parts,
        raw["extension"]["budget"],
        tuple(upstreams[u] for u in raw["extension"]["upstreams"]),
        tuple(toolchain["llvm"]),
        tuple(toolchain["apple_clang"]),
        raw["distribution"],
    )


# -- the target -----------------------------------------------------------------


@dataclass(frozen=True)
class Platform:
    os: str  # macos, windows, or linux for every other unix: ELF with lld
    machine: str  # x86_64, aarch64, i686, or whatever else the kernel calls itself
    musl: bool = False  # Linux on musl (musllinux) rather than glibc

    @classmethod
    def current(cls) -> "Platform":
        # no list of machines to be refused by: only x86 and macOS's and
        # Windows' linkers take flags of their own, and every other machine is
        # one clang can target, so a distribution's riscv64 or ppc64le just builds
        machine = host.machine().lower()
        machine = {"amd64": "x86_64", "arm64": "aarch64", "i386": "i686"}.get(machine, machine)
        # a 32-bit interpreter on a 64-bit x86 kernel builds for 32-bit x86
        if machine == "x86_64" and sys.maxsize < 2**32:
            machine = "i686"
        # the interpreter's own build triple names its C library
        musl = "musl" in (sysconfig.get_config_var("HOST_GNU_TYPE") or "")
        return cls({"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux"), machine, musl)

    def quote(self, argument: str) -> str:
        # ninja runs commands through sh, or on Windows straight through
        # CreateProcess, which splits a command line by the MSVC rules
        return subprocess.list2cmdline([argument]) if self.os == "windows" else shlex.quote(argument)

    def object(self, name: str) -> str:
        return f"{name}.obj" if self.os == "windows" else f"{name}.o"

    def archive(self, name: str) -> str:
        return f"{name}.lib" if self.os == "windows" else f"lib{name}.a"

    def executable(self, name: str) -> str:
        return f"{name}.exe" if self.os == "windows" else name


@dataclass(frozen=True)
class Python:
    include: Path
    extension_suffix: str
    free_threaded: bool
    # what a Windows extension links the C API from; elsewhere it resolves at load
    library: Path | None

    @classmethod
    def current(cls) -> "Python":
        """
        One abi3 wheel serves every CPython from 3.12. A free-threaded
        interpreter gets a wheel of its own: nanobind's runtime can't target
        3.15's abi3t without its split mode, a second package to depend on.
        """
        free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        library = None
        if sys.platform == "win32":
            version = "3" if not free_threaded else f"{sys.version_info[0]}{sys.version_info[1]}t"
            library = Path(sys.base_prefix) / "libs" / f"python{version}.lib"
        if not free_threaded:
            suffix = ".pyd" if sys.platform == "win32" else ".abi3" + Path(suffix).suffix
        return cls(Path(sysconfig.get_path("include")), suffix, free_threaded, library)

    @classmethod
    def example(cls, platform: Platform, free_threaded: bool = False) -> "Python":
        # what `check` plans every platform with, from any machine
        windows = platform.os == "windows"
        return cls(
            Path("/python/include"),
            ".pyd" if windows else (".cpython-315t.so" if free_threaded else ".abi3.so"),
            free_threaded,
            Path("/python/python3.lib") if windows else None,
        )


def major_version(output: str) -> int | None:
    # the first dotted number a tool's --version prints: "Debian clang version
    # 19.1.7 (3+b1)", "LLD 21.1.8 (compatible with GNU linkers)", "Apple clang
    # version 17.0.0 (clang-1700.0.13.5)"
    for word in output.split():
        parts = word.split(".")
        if len(parts) > 1 and all(part.isdigit() for part in parts):
            return int(parts[0])
    return None


def _output(*args: str | Path) -> str:
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True, check=False)
    return result.stdout + result.stderr


def _accept(compiler: Path, vendor: str, accepted: tuple[int, int]) -> int:
    # its major version if trimkit.toml accepts it: Apple's clang only on macOS
    version = _output(compiler, "--version")
    first = version.splitlines()[0] if version else "nothing"
    major = major_version(version)
    oldest, newest = accepted
    apple = version.startswith("Apple clang")
    if apple != (vendor == "Apple clang") or major is None or not oldest <= major <= newest:
        upper = "or newer" if newest == 10**6 else f"to {newest}"
        sys.exit(f"{compiler} is {first!r}: trimkit.toml accepts {vendor} {oldest} {upper}")
    return major


def include_path(compiler: Path) -> tuple[Path, ...]:
    # where `compiler` finds <headers> by itself, as it lists them preprocessing
    # nothing: where a system upstream is found
    listing = subprocess.run(
        [str(compiler), "-xc++", "-E", "-v", "-"], input="", capture_output=True, text=True, check=True
    ).stderr
    if "#include <...> search starts here:" not in listing:
        sys.exit(f"{compiler} -v lists no include search path")
    searched = listing.split("#include <...> search starts here:", 1)[1].split("End of search list.", 1)[0]
    # macOS marks framework directories, which hold no plain headers
    return tuple(
        Path(line.strip()) for line in searched.splitlines() if line.strip() and "(framework" not in line
    )


@dataclass(frozen=True)
class Tools:
    """
    The toolchain one build runs, by absolute path, so ninja runs exactly the
    tools checked against [toolchain] and no others. ThinLTO's bitcode and
    PGO's profiles are only read by the LLVM that wrote them, so every tool
    must be of clang's own LLVM.
    """

    compiler: Path  # clang++, which also drives every link
    archiver: Path  # llvm-ar, Xcode's libtool or llvm-lib
    profiler: Path  # llvm-profdata, which merges PGO's profiles
    include_path: tuple[Path, ...] = ()  # where clang++ finds <headers> by itself

    @classmethod
    def current(cls, platform: Platform, manifest: Manifest, strict: bool = True) -> "Tools":
        # our own builds (`strict`) use only compilers the gate has passed on;
        # anyone else building the sdist is held only to the oldest, as a new
        # clang on a rolling distribution is no reason to refuse them
        def accepted(pair: tuple[int, int]) -> tuple[int, int]:
            return pair if strict else (pair[0], 10**6)

        found = shutil.which("clang++")
        if found is None:
            sys.exit("no clang++ on PATH: every build is clang")
        compiler = Path(found)
        if platform.os == "macos":
            # /usr/bin's clang++ runs the selected Xcode's, with its SDK and
            # linker; libtool and llvm-profdata are that same Xcode's
            _accept(compiler, "Apple clang", accepted(manifest.apple_clang))
            paths = {
                name: Path(_output("xcrun", "--find", name).strip()) for name in ("libtool", "llvm-profdata")
            }
            if not all(path.is_file() for path in paths.values()):
                sys.exit(
                    f"xcrun finds no Xcode libtool and llvm-profdata: {[str(p) for p in paths.values()]}"
                )
            return cls(compiler, paths["libtool"], paths["llvm-profdata"], include_path(compiler))
        major = _accept(compiler, "clang", accepted(manifest.llvm))
        # each tool as clang itself finds it, which for the linker is the one
        # -fuse-ld=lld runs, and each of clang's own LLVM
        linker, archiver, profiler = {
            "linux": ("ld.lld", "llvm-ar", "llvm-profdata"),
            "windows": ("lld-link", "llvm-lib", "llvm-profdata"),
        }[platform.os]
        paths = {}
        for name in (linker, archiver, profiler):
            found = shutil.which(_output(compiler, f"-print-prog-name={name}").strip())
            if found is None:
                sys.exit(f"{compiler} finds no {name}: it comes with clang's LLVM")
            paths[name] = Path(found)
            # llvm-lib reports no version, so it must be the one beside clang
            if name == "llvm-lib" and paths[name].resolve().parent != compiler.resolve().parent:
                sys.exit(f"{paths[name]} isn't beside {compiler}, so may not be its LLVM's")
            if name != "llvm-lib" and major_version(_output(paths[name], "--version")) != major:
                sys.exit(f"{paths[name]} isn't LLVM {major}, as {compiler} is")
        return cls(compiler, paths[archiver], paths[profiler], include_path(compiler))

    @classmethod
    def example(cls, platform: Platform) -> "Tools":
        # what `check` plans every platform with, from any machine
        names = {
            "linux": ("clang++", "llvm-ar", "llvm-profdata"),
            "macos": ("clang++", "libtool", "llvm-profdata"),
            "windows": ("clang++.exe", "llvm-lib.exe", "llvm-profdata.exe"),
        }[platform.os]
        return cls(*(Path("/llvm/bin") / name for name in names))


@dataclass(frozen=True)
class Config:
    """
    One build's options: the backend's config settings, each the field name
    with `-` for `_` (`-Cbuild-dir=...`, `-Cstrict=true`).
    """

    build_dir: Path  # absolute: objects, and PGO's profiles
    strict: bool = False  # warnings in our sources are errors
    sanitize: str = ""  # clang's -fsanitize= list, for every object
    pgo: str = ""  # profile-guided: "generate" instruments, "use" reads profiles/merged.profdata
    fuzz: bool = False  # also the libFuzzer harnesses in fuzz/, sanitized
    # for distributions, never our wheels: debug information, unstripped,
    # optimized all the same; and the upstreams, by name, comma-separated, to
    # build against the system's copy rather than vendor/'s
    debug: bool = False
    system: str = ""

    def __post_init__(self):
        if self.pgo not in ("", "generate", "use"):
            sys.exit(f"pgo must be generate, use or empty, not {self.pgo!r}")
        if self.fuzz and self.sanitize:
            sys.exit("fuzz builds choose their own sanitizers")
        if not self.build_dir.is_absolute():
            sys.exit(f"build-dir must be absolute, not {self.build_dir}")

    @classmethod
    def from_settings(cls, settings: dict | None, build_dir: Path) -> "Config":
        values = {"build_dir": build_dir}
        fields = {f.name.replace("_", "-"): f for f in dataclasses.fields(cls)}
        for key, value in (settings or {}).items():
            if key not in fields or not isinstance(value, str):
                sys.exit(f"unknown config setting {key}={value!r}: expected one of {sorted(fields)}")
            if fields[key].type is bool:
                if value not in ("true", "false"):
                    sys.exit(f"{key} must be true or false, not {value!r}")
                values[fields[key].name] = value == "true"
            else:
                values[fields[key].name] = Path(value).resolve() if key == "build-dir" else value
        return cls(**values)

    @property
    def sanitizers(self) -> str:
        return "fuzzer-no-link,address,undefined" if self.fuzz else self.sanitize

    @property
    def system_upstreams(self) -> tuple[str, ...]:
        return tuple(name for name in self.system.split(",") if name)

    @property
    def profiles(self) -> Path:
        return self.build_dir / "profiles"

    @property
    def profile(self) -> Path:
        return self.profiles / "merged.profdata"


@dataclass(frozen=True)
class Target:
    platform: Platform
    python: Python
    tools: Tools
    config: Config
    # each system upstream's header directory to add, or None for one the
    # compiler already searches
    system: dict[str, Path | None] = field(default_factory=dict)

    @classmethod
    def current(cls, config: Config) -> "Target":
        platform = Platform.current()
        manifest = load()
        tools = Tools.current(platform, manifest, strict=config.strict)
        return cls(platform, Python.current(), tools, config, find_system(manifest, config, tools))


def find_system(manifest: Manifest, config: Config, tools: Tools) -> dict[str, Path | None]:
    """
    Where each upstream `config.system` names is on this system: only an
    upstream trimkit.toml names a `system` header for (never a patched one),
    and only a copy the compiler finds by itself, outside vendor/.
    """
    found = {}
    for name in config.system_upstreams:
        upstream = manifest.upstreams.get(name)
        if upstream is None or not upstream.system:
            sys.exit(
                f"{name} can't build against a system copy: trimkit.toml names no `system` header for it"
            )
        header = Path(upstream.system_directory) / upstream.system
        hits = [d for d in tools.include_path if (d / header).is_file()]
        if not hits:
            searched = ", ".join(str(d) for d in tools.include_path)
            sys.exit(f"{name}: no {header} where {tools.compiler} looks ({searched})")
        directory = hits[0].resolve()
        if directory.is_relative_to(ROOT.resolve()):
            sys.exit(f"{name}: {hits[0] / header} is this tree's own copy, not the system's")
        found[name] = directory / upstream.system_directory if upstream.system_directory else None
    return found


# -- flags ----------------------------------------------------------------------


@dataclass(frozen=True)
class Flags:
    compile: tuple[str, ...] = ()
    link: tuple[str, ...] = ()

    def __add__(self, other: "Flags") -> "Flags":
        return Flags(self.compile + other.compile, self.link + other.link)


NONE = Flags()


# Every object: C++17, and on ELF and Mach-O position independent with every
# symbol hidden unless marked, and on macOS for the oldest version the wheels
# support. A COFF DLL exports only what's marked anyway, and uses the DLL C
# runtime, as CPython does.
def language(platform: Platform) -> Flags:
    # every floating-point operation rounded on its own, never fused into a
    # multiply-add: arm64 has FMA and x86_64's baseline doesn't, so fusing
    # would give the two different answers to the same input
    exact = ("-std=c++17", "-ffp-contract=off")
    hidden = (*exact, "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden")
    if platform.os == "macos":
        oldest = f"-mmacosx-version-min={macos_deployment_target()}"
        return Flags((*hidden, oldest), (oldest,))
    if platform.os == "windows":
        # clang here targets 64-bit unless told; 32-bit Python needs the other
        target = ("--target=i686-pc-windows-msvc",) if platform.machine == "i686" else ()
        # the C and C++ runtimes linked in: a clean Windows has no msvcp140.dll,
        # and nothing C++ crosses the extension's boundary, as on musl. WIN32 is
        # what CMake defines on Windows and clang doesn't: coal's alloca.h tests it
        runtime = "-fms-runtime-lib=static"
        return Flags((*exact, *target, runtime, "-DWIN32"), (*target, runtime))
    if platform.machine == "i686":
        # doubles in SSE2 registers, IEEE binary64 as on every other machine,
        # not the x87's 80-bit registers, whose extra precision changes answers
        return Flags((*hidden, "-msse2", "-mfpmath=sse"))
    return Flags(hidden)


# release: -O3; sanitizer and fuzz builds: what a report needs. On the link
# too, which optimizes again under link-time optimization, and whose level
# the driver otherwise leaves at the linker's default -O2
OPTIMIZE = {False: Flags(("-O3", "-DNDEBUG"), ("-O3",)), True: Flags(("-O2", "-g", "-DNDEBUG"), ("-O2",))}
# a distribution's debug build: the release's optimization, with debug
# information to split into its debug package
DEBUG_INFO = Flags(("-g",))

# Link-time optimization across each library's objects, and identical-code
# folding, everywhere (measured on Linux with PGO: 15% smaller, as fast).
# ThinLTO objects are LLVM bitcode, which only an LLVM linker reads: lld on
# Linux and Windows, and on macOS Apple's own, which links bitcode through
# Xcode's libLTO, matching its clang. Folding is lld's "safe" mode, which
# merges only functions whose address is never taken, so pointer comparisons
# keep working: nanobind tells its own modules apart by comparing a function's
# address. lld-link folds everything unless told otherwise, so it's told
LINK_TIME = {
    "linux": Flags(("-flto=thin",), ("-flto=thin", "-fuse-ld=lld", "-Wl,--icf=safe")),
    "macos": Flags(("-flto=thin",), ("-flto=thin",)),
    "windows": Flags(("-flto=thin",), ("-flto=thin", "-fuse-ld=lld", "-Wl,/OPT:SAFEICF")),
}

# Hardening and dead-code stripping for everything we ship: per-function
# sections the linker drops when unused, stack protection, and zeroing any
# stack variable read before it is written (measured as free).
SECTIONS = (
    "-ffunction-sections",
    "-fdata-sections",
    "-fstack-protector-strong",
    "-ftrivial-auto-var-init=zero",
)
# some toolchains predefine a fortify level, so reset it rather than redefine
FORTIFY = ("-U_FORTIFY_SOURCE", "-D_FORTIFY_SOURCE=3")
HARDENING = {
    "linux": Flags(
        (*SECTIONS, *FORTIFY, "-fstack-clash-protection"),
        ("-Wl,--gc-sections", "-Wl,-z,relro,-z,now,-z,noexecstack", "-Wl,-z,separate-code"),
    ),
    "macos": Flags((*SECTIONS, *FORTIFY), ("-Wl,-dead_strip",)),
    # control flow guard, and the linker's dead-code stripping. clang's driver
    # takes -mguard=cf only for MinGW and /guard:cf only as clang-cl, so with an
    # msvc target this names what both become; ThinLTO carries it in the bitcode
    "windows": Flags((*SECTIONS, "-Xclang", "-cfguard"), ("-Wl,/guard:cf", "-Wl,/OPT:REF")),
}
# x86's control-flow enforcement: branch targets, and Windows' shadow stacks
HARDENING_X86 = {
    "linux": Flags(("-fcf-protection",)),
    "macos": Flags(("-fcf-protection",)),
    "windows": Flags(link=("-Wl,/CETCOMPAT",)),
}

# what ships carries no symbol table, unless a build wants its reports readable
STRIP = {"linux": Flags(link=("-Wl,-s",)), "macos": Flags(link=("-Wl,-x", "-Wl,-S")), "windows": NONE}

# musl has no libstdc++ of its own to rely on, so auditwheel would bundle
# Alpine's (3.9 MB, doubling the wheel): link the C++ runtime in instead,
# which the version script keeps private, as nothing C++ crosses the
# extension's boundary. glibc's libstdc++ is every manylinux system's
MUSL_RUNTIME = Flags(link=("-static-libstdc++", "-static-libgcc"))

# every warning that is reasonable for modern C++, for our sources only
WARNINGS = Flags(
    (
        "-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wsign-conversion", "-Wshadow",
        "-Wold-style-cast", "-Wcast-align", "-Wnull-dereference", "-Wdouble-promotion",
        "-Wformat=2", "-Wimplicit-fallthrough", "-Wundef", "-Wnon-virtual-dtor",
        "-Woverloaded-virtual", "-Wmissing-declarations",
    )
)  # fmt: skip
STRICT = Flags(("-Werror",))
# vendored code: we don't edit it, so its warnings aren't actionable
QUIET = Flags(("-w",))
# under `undefined`, our sources alone also take clang's stricter groups
# (unsigned overflow, implicit narrowing and sign changes, nullability, local
# bounds, float division by zero): vendored code is not ours to fix, ours must
# be clean. Each upstream's [no_sanitize] table turns checks off for it instead
OURS_CHECKS = ("integer", "nullability", "implicit-conversion", "local-bounds", "float-divide-by-zero")
OURS_SANITIZED = Flags((f"-fsanitize={','.join(OURS_CHECKS)}",))
# what clang's -fsanitize-ignorelist= reads, written beside build.ninja
IGNORELIST = "sanitize_ignorelist.txt"


def ignorelist(manifest: Manifest) -> str:
    """
    OURS_CHECKS reach the vendored headers our sources compile in (tinybvh,
    earcut, VHACD, fast_float, dragonbox, Eigen, Vulkan, nanobind), whose
    intentional wraps, shifts and casts aren't ours to fix: every vendored
    directory is left out of exactly those checks, keeping today's set
    """
    directories = ["vendor/nanobind/include"]
    for upstream in manifest.upstreams.values():
        directories += upstream.include
    # a section per check (its name is a pattern, so one can't name several),
    # each over every vendored directory, however the build names the tree
    lines = []
    for check in OURS_CHECKS:
        lines += [f"[{check}]", *(f"src:*/{d}/*" for d in directories)]
    # std::make_shared's cast to its not yet constructed storage: libstdc++ 14
    # moved it from _Sp_counted_ptr_inplace, which clang's own cfi_ignorelist
    # names, into __aligned_buffer::_M_ptr
    lines += ["[cfi-unrelated-cast]", "fun:_ZN9__gnu_cxx16__aligned_buffer*"]
    return "\n".join(lines) + "\n"


# The extension exports its init function and nothing else, so none of its
# vendored code's symbols leak into the process to clash with another copy.
# A DLL exports only what's marked, which is just that function.
def exports(platform: Platform, module: str) -> Flags:
    return {
        "linux": Flags(link=(f"-Wl,--version-script={EXPORTS}",)),
        "macos": Flags(link=(f"-Wl,-exported_symbol,_PyInit_{module}",)),
        "windows": NONE,
    }[platform.os]


# How nanobind builds its runtime, from nanobind-config.cmake (nanobind v3.1.0:
# nanobind_build_library and nanobind_add_module) for the options we use,
# STABLE_ABI, PROTECT_STACK and NOMINSIZE, and its static library, compiled
# into the one extension: re-derive these on every nanobind bump, which
# `uv run scripts outdated` names.
# - the runtime itself: NB_BUILD, compact assertion messages in optimized
#   builds, and no strict aliasing for its type punning of Python objects
NANOBIND_RUNTIME = Flags(("-DNB_BUILD", "-DNB_COMPACT_ASSERTIONS", "-fno-strict-aliasing"))
# - and on Windows, no warnings about vsnprintf
NANOBIND_RUNTIME_WINDOWS = Flags(("-D_CRT_SECURE_NO_WARNINGS",))
# - the faster TLSDESC model for its thread-locals, on x86_64 Linux
NANOBIND_RUNTIME_X86_LINUX = Flags(("-mtls-dialect=gnu2",))
# - everything compiled against it, the runtime included: the stable ABI, and
#   its own domain, so no other nanobind extension in the process shares our
#   type registry, and none of its bugs or leak reports can become ours
STABLE_ABI = Flags(("-DPy_LIMITED_API=0x030C0000", "-DNB_DOMAIN=trimkit"))
# only Windows' pyconfig.h leaves Py_GIL_DISABLED to the compiler's command line
FREE_THREADED = Flags(("-DPy_GIL_DISABLED=1", "-DNB_FREE_THREADED", "-DNB_DOMAIN=trimkit"))


def python_link(target: Target) -> Flags:
    """
    How the extension and the runtime reach the C API: at load on Linux, through
    nanobind's list of it on macOS, through python3.lib on Windows.
    """
    if target.platform.os == "macos":
        return Flags(link=(f"-Wl,@{NANOBIND / 'cmake' / 'darwin-ld-cpython.sym'}",))
    if target.platform.os == "windows":
        # named, and searchable: pyconfig.h asks for it again by bare name
        library = target.python.library
        return Flags(link=(str(library), f"-Wl,/LIBPATH:{library.parent}"))
    return NONE


def instrumentation(config: Config) -> Flags:
    """
    Sanitizers and profiles: on every object, vendored included.
    """
    flags = NONE
    if config.sanitizers:
        flags += Flags(
            (
                # every check reports and aborts: CFI alone would otherwise trap silently
                f"-fsanitize={config.sanitizers}", "-fno-sanitize-recover=all", "-fno-sanitize-trap=all",
                "-fno-omit-frame-pointer",
                # bounds-checked standard containers in every library: too slow
                # for release (V-HACD is 3-5% slower) but what analysis wants
                "-D_GLIBCXX_ASSERTIONS", "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE",
            ),
            (f"-fsanitize={config.sanitizers}",),
        )  # fmt: skip
    if "undefined" in config.sanitizers:
        flags += Flags((f"-fsanitize-ignorelist={config.build_dir / IGNORELIST}",))
    if "cfi" in config.sanitizers:
        # not indirect calls: their targets include libpython and numpy (type
        # slots from PyType_GetSlot, DLPack deleters), compiled without CFI
        # metadata, which every such call would fail without cross-DSO CFI.
        # Virtual calls and casts between polymorphic types stay checked
        flags += Flags(("-fno-sanitize=cfi-icall",), ("-fno-sanitize=cfi-icall",))
    if config.pgo == "generate":
        # counters bumped plainly, clang's default: threads racing in training
        # lose a few counts, which only steers the optimizer a little less.
        # Atomic updates are miscompiled for 32-bit x86 by clang 20 (an
        # absolute relocation to the counters, which a shared object refuses)
        generate = f"-fprofile-generate={config.profiles}"
        flags += Flags((generate,), (generate,))
    if config.pgo == "use":
        # code the training never ran stays optimized normally, and a stale
        # profile only steers less well: neither may fail a strict build
        flags += Flags(
            (
                f"-fprofile-use={config.profile}",
                "-Wno-profile-instr-unprofiled",
                "-Wno-profile-instr-out-of-date",
            )
        )
    return flags


# -- the ninja file -------------------------------------------------------------


def rules(platform: Platform, tools: Tools) -> list[str]:
    """
    The three things a build ever does, one ninja rule each. The compile's
    depfile is how ninja learns every header, and archives and link lines go
    through response files, as Windows command lines are short.
    """
    compiler, archiver = (platform.quote(str(tool)) for tool in (tools.compiler, tools.archiver))
    archive_command, members = {
        "linux": (f"rm -f $out && {archiver} crs $out @$out.rsp", "$in"),
        "macos": (f"{archiver} -static -no_warning_for_no_symbols -o $out -filelist $out.rsp", "$in_newline"),
        "windows": (f"{archiver} /nologo /out:$out @$out.rsp", "$in"),
    }[platform.os]
    return [
        "rule compile\n"
        f"  command = {compiler} $flags -MD -MF $out.d -c $in -o $out\n"
        "  description = compile $in\n"
        "  depfile = $out.d\n"
        "  deps = gcc",
        "rule archive\n"
        f"  command = {archive_command}\n"
        "  description = archive $out\n"
        "  rspfile = $out.rsp\n"
        f"  rspfile_content = {members}",
        "rule link\n"
        f"  command = {compiler} @$out.rsp $flags -o $out\n"
        "  description = link $out\n"
        "  rspfile = $out.rsp\n"
        "  rspfile_content = $in",
    ]


def escape_path(path: str) -> str:
    # a path in a `build` line, where `$`, space and `:` are syntax
    return path.replace("$", "$$").replace(" ", "$ ").replace(":", "$:")


def plan(manifest: Manifest, target: Target) -> str:
    """
    The whole build for one target, as a ninja file: an archive per upstream
    with sources, the one extension linking them, the nanobind runtime and
    every part, and in fuzz builds the harnesses and seed writers.
    """
    platform, python, config = target.platform, target.python, target.config
    x86 = platform.machine == "x86_64"
    common = (
        language(platform)
        + OPTIMIZE[bool(config.sanitizers)]
        + (DEBUG_INFO if config.debug and not config.sanitizers else NONE)
        + LINK_TIME[platform.os]
        + HARDENING[platform.os]
        + (HARDENING_X86[platform.os] if x86 else NONE)
        + instrumentation(config)
    )
    # sanitized builds are for reports, which want lines and symbols
    stripped = NONE if config.debug or config.sanitizers else STRIP[platform.os]
    abi = FREE_THREADED if python.free_threaded else STABLE_ABI
    ours = WARNINGS + (STRICT if config.strict else NONE)
    if "undefined" in config.sanitizers:
        ours += OURS_SANITIZED
    edges: list[str] = []

    def edge(rule: str, output: str, inputs, flags=None) -> str:
        # one `build` statement; flags are shell-quoted here, so only ninja's
        # own `$` is left to escape
        line = f"build {escape_path(output)}: {rule} {' '.join(map(escape_path, inputs))}"
        if flags is not None:
            quoted = " ".join(platform.quote(str(f)) for f in flags)
            line += "\n  flags = " + quoted.replace("$", "$$")
        edges.append(line)
        return output

    def includes(upstreams) -> list[str]:
        # vendored headers are system headers, so -Werror judges only us; a
        # system upstream's are the system's, where the compiler finds them
        words = []
        for upstream in upstreams:
            if upstream.name in target.system:
                directory = target.system[upstream.name]
                words += [] if directory is None else ["-isystem", str(directory)]
                continue
            for directory in upstream.include:
                words += ["-isystem", str(ROOT / directory)]
        return words

    def defines(upstreams) -> list[str]:
        return [f"-D{name}" for upstream in upstreams for name in upstream.defines]

    def object_for(source: Path) -> str:
        return platform.object(f"objects/{source.relative_to(ROOT).as_posix()}")

    # one quiet archive per upstream with sources; its includes and defines
    # serve whatever links it too
    archives = {}
    for upstream in manifest.upstreams.values():
        if not upstream.sources:
            continue
        unsanitized = [
            f"-fno-sanitize={check}" for check in upstream.no_sanitize if "undefined" in config.sanitizers
        ]
        # with the headers and defines of what it requires, as its users see them
        built_with = [upstream, *(manifest.upstreams[r] for r in upstream.requires)]
        values = [*common.compile, *QUIET.compile, *includes(built_with), *defines(built_with), *unsanitized]
        objects = [edge("compile", object_for(s), [str(s)], values) for s in upstream.compiled]
        archives[upstream.name] = edge("archive", platform.archive(f"vendored_{upstream.name}"), objects)

    # the nanobind runtime, compiled in, quiet like vendored code
    system_python = ["-isystem", str(python.include), "-isystem", str(NANOBIND / "include")]
    runtime_flags = NANOBIND_RUNTIME + abi
    if platform.os == "windows":
        runtime_flags += NANOBIND_RUNTIME_WINDOWS
    if platform.os == "linux" and x86:
        runtime_flags += NANOBIND_RUNTIME_X86_LINUX
    runtime_object = edge(
        "compile",
        platform.object("objects/nanobind/nb_combined"),
        [str(NANOBIND / "src" / "nb_combined.cpp")],
        [
            *common.compile,
            *QUIET.compile,
            *runtime_flags.compile,
            *system_python,
            *includes(manifest.extension_upstreams),
        ],
    )

    # the one extension: every part, each with its own upstreams' headers and
    # defines, and the module calling each part's bind_, linking them all;
    # its docstring names what it was built from, a system upstream as the system's
    described = ", ".join(
        f"{u.name} (system)" if u.name in target.system else u.described for u in manifest.compiled_in
    )
    ours_values = [*common.compile, *ours.compile, *abi.compile, *system_python]
    module = ROOT / "src" / f"{EXTENSION}.cpp"
    objects = [
        edge(
            "compile",
            object_for(module),
            [str(module)],
            [
                *ours_values,
                f'-DTRIMKIT_UPSTREAM="{described}"',
                f'-DTRIMKIT_VERSION="{pyproject()["project"]["version"]}"',
            ],
        )
    ]
    for part in manifest.parts.values():
        values = [*ours_values, *includes(part.upstreams), *defines(part.upstreams)]
        objects.append(edge("compile", object_for(part.source), [str(part.source)], values))
    linked = [archives[u.name] for u in manifest.compiled_in if u.name in archives]
    link = [
        "-shared",
        *common.link,
        *exports(platform, EXTENSION).link,
        *python_link(target).link,
        *stripped.link,
        *(MUSL_RUNTIME.link if platform.musl else ()),
    ]
    built = [edge("link", extension_file(python), [*objects, runtime_object, *linked], link)]

    # fuzz builds: the libFuzzer harnesses and the seed writers, plain programs
    if config.fuzz:
        for part in manifest.parts.values():
            programs = [(f"fuzz_{h}", h, ("-fsanitize=fuzzer",)) for h in part.fuzz]
            if part.seeds:
                programs.append((part.seeds, part.seeds, ()))
            linked = [archives[u.name] for u in part.upstreams if u.name in archives]
            values = [*common.compile, *ours.compile, *includes(part.upstreams), *defines(part.upstreams)]
            for program, source_name, extra in programs:
                source = ROOT / "fuzz" / f"{source_name}.cpp"
                obj = edge("compile", object_for(source), [str(source)], values)
                built.append(
                    edge("link", platform.executable(program), [obj, *linked], [*common.link, *extra])
                )

    head = [
        "# Written by scripts/builder.py from trimkit.toml for one build: edit those.",
        "ninja_required_version = 1.10",
    ]
    tail = ["default " + " ".join(map(escape_path, built))]
    return "\n\n".join([*head, *rules(platform, target.tools), *edges, *tail]) + "\n"


def extension_file(python: Python) -> str:
    # the package is the extension: what a wheel ships as trimkit/__init__...
    return f"__init__{python.extension_suffix}"


def write(target: Target) -> None:
    # the plan into its build directory, as build.ninja: what ninja then runs
    config = target.config
    if config.pgo == "use" and not config.profile.is_file():
        sys.exit(f"pgo=use but no {config.profile}: run scripts/pgo.py first")
    config.build_dir.mkdir(parents=True, exist_ok=True)
    manifest = load()
    (config.build_dir / IGNORELIST).write_text(ignorelist(manifest))
    (config.build_dir / "build.ninja").write_text(plan(manifest, target))


# -- checks ---------------------------------------------------------------------

# what `check` plans every platform with: each option on its own
EXAMPLE_CONFIGS = [
    {},
    {"strict": True},
    {"sanitize": "address,undefined,cfi"},
    {"pgo": "generate"},
    {"pgo": "use"},
    {"fuzz": True},
    {"debug": True},
]


def check() -> None:
    """
    What a build can't catch for itself, on this machine, in a second.
    """
    manifest = load()
    errors = []
    # every C++ file in fuzz/ is a harness or seed writer some part names,
    # so none can sit there unbuilt
    named = set(manifest.harnesses) | {m.seeds for m in manifest.parts.values() if m.seeds}
    present = {p.stem for p in tracked_under(ROOT / "fuzz") if p.suffix == ".cpp"}
    if present != named:
        errors.append(
            f"fuzz/*.cpp {sorted(present)} don't match trimkit.toml's fuzz and seeds {sorted(named)}"
        )
    # what a build compiles is exactly what git tracks
    for upstream in manifest.upstreams.values():
        if upstream.sources and not upstream.compiled:
            errors.append(f"[upstream.{upstream.name}]: its sources match no files")
        errors += [
            f"{p.relative_to(ROOT)} would be compiled but isn't tracked"
            for p in upstream.compiled
            if p not in tracked()
        ]
    errors += [
        f"[part.{m.name}]: no src/{m.name}.cpp" for m in manifest.parts.values() if m.source not in tracked()
    ]
    # every platform and option plans, against every upstream that may come
    # from the system too, with none of its vendored copy left on a command line
    system = {name: Path("/usr/include") for name, u in manifest.upstreams.items() if u.system}
    count = 0
    for os in ("linux", "macos", "windows"):
        for machine in (
            ("x86_64", "aarch64", "i686", "s390x") if os == "linux" else ("x86_64", "aarch64", "i686")
        ):
            for musl in (False, True) if os == "linux" else (False,):
                platform = Platform(os, machine, musl)
                tools = Tools.example(platform)
                for options in EXAMPLE_CONFIGS:
                    for python in (Python.example(platform), Python.example(platform, free_threaded=True)):
                        plan(
                            manifest,
                            Target(platform, python, tools, Config(Path("/build").resolve(), **options)),
                        )
                        count += 1
                config = Config(Path("/build").resolve(), debug=True, system=",".join(system))
                planned = plan(manifest, Target(platform, Python.example(platform), tools, config, system))
                errors += [
                    f"{platform}: vendor/{name} is still included"
                    for name in system
                    if f"vendor/{name}" in planned
                ]
                count += 1
    # a system upstream's header is the one its vendored copy provides, as
    # the sources include it
    for upstream in manifest.upstreams.values():
        if upstream.system and not any((ROOT / d / upstream.system).is_file() for d in upstream.include):
            errors.append(f"{upstream.name}: its system header {upstream.system} isn't under its `include`")
    if errors:
        sys.exit("\n".join(errors))
    print(f"ok: fuzz/ and sources match trimkit.toml, and {count} plans hold together")
