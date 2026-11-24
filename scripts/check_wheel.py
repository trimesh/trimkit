"""
Check a built wheel:
- its metadata renders (twine)
- one stable-ABI extension, the package itself, within its size budget, and
  no other shared library
- nothing in it beyond the stable ABI (abi3audit)
- every license and the SBOM shipped
- the extension's hardening, as builder.py's flags promise it, and what it
  imports from the system, against the lists below (ELF and PE; Mach-O is
  read by neither)

  python scripts/check_wheel.py dist/trimkit-*.whl

`uv run scripts static` runs it on the dev build's wheel, and cibuildwheel's
audit on every release wheel, with twine and abi3audit installed by hash.
"""

import re
import stat
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

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


# Each machine's control-flow marking: the instruction a function starts
# with (endbr64, paciasp), which the C runtime's few objects bring a couple
# dozen of and our code thousands, and its GNU property's feature word
EM_X86_64, EM_AARCH64 = 62, 183
MARKS = {EM_X86_64: (b"\xf3\x0f\x1e\xfa", 0xC0000002), EM_AARCH64: (b"\x3f\x23\x03\xd5", 0xC0000000)}
MARKED = 1000
# The property is the AND of every object linked, and only manylinux_2_28's
# x86_64 glibc marks its crt objects: Debian's, manylinux's aarch64 and
# musl's don't, which drops it there however we build (signing still works)
NOTED = {EM_X86_64: 0b11}  # IBT | SHSTK

# What the extension imports from the C library, past the stable ABI
# (abi3audit's) and the C++ runtime: every name today's Linux wheels import,
# glibc's and musl's, so a new one fails here until it is added with its
# reason. There is no socket, no fopen, no exec, no mmap. This bounds what
# the code we ship calls, not what code that isn't ours can do once it runs:
# dlsym and syscall reach everything, and nothing here is a sandbox. Nor is a
# system call written inline seen, on any platform. These reach past the
# process, each for one user (`cmd_threads` holds open, dlopen, dlsym and
# syscall to their one file in the source):
REACHING = {
    "open __open_2 read __read_chk close": "src/cgroup.hpp reads limits from /proc and /sys",
    "dlopen dlsym": "src/vulkan/vulkan.hpp opens the Vulkan loader, by name",
    "syscall": "src/platform.hpp's futex waits and wakes",
    "secure_getenv": "musl's libstdc++, linked in, reads the environment",
    # the process's own streams, as nothing imported opens another: vendored
    # logging, and musl's libstdc++
    "stdin stdout stderr printf fprintf vfprintf __vfprintf_chk puts fputs fputc putc putwc fwrite"
    " write fflush fread getc getwc ungetc ungetwc fseek ftell": "stdin, stdout and stderr",
}
# and these stay inside it
IN_PROCESS = """
    abort exit __stack_chk_fail __stack_chk_guard __errno_location __tls_get_addr __libc_single_threaded
    __gmon_start__ _ITM_deregisterTMCloneTable _ITM_registerTMCloneTable dl_iterate_phdr __getauxval
    malloc calloc realloc free posix_memalign malloc_usable_size malloc_trim
    memcpy __memcpy_chk memmove memset __memset_chk memcmp bcmp memchr qsort
    strlen strcmp strncmp strcpy strncpy strchr strrchr strstr strcspn strdup strcoll strxfrm strerror_r
    snprintf vsnprintf __vsnprintf_chk strftime setlocale __ctype_get_mb_cur_max
    atof atoi strtod strtof strtold strtol strtoul strtoull __isoc23_strtol __isoc23_strtoul __isoc23_strtoull
    isalnum isalpha isblank iscntrl islower isprint ispunct isspace isupper isxdigit tolower toupper
    btowc wctob mbrtowc wcrtomb iswctype wctype towlower towupper wcscoll wcsftime wcslen wcsxfrm
    wmemchr wmemcpy wmemmove wmemset
    acosf atan2 atan2f ceil ceilf cos cosf exp2 floor floorf fmod frexp hypot ldexp log log2
    nearbyint nextafter nextafterf pow remainder round sin sinf sqrt sqrtf
    pthread_create pthread_attr_init pthread_attr_destroy pthread_attr_setdetachstate
    pthread_attr_setstacksize
    pthread_self pthread_setname_np pthread_sigmask sigfillset sigdelset pthread_atfork __register_atfork
    __pthread_key_create
    pthread_once pthread_mutex_lock pthread_mutex_unlock pthread_cond_wait pthread_cond_broadcast
    sched_getaffinity __sched_cpualloc __sched_cpucount __sched_cpufree sysconf clock_gettime
"""
C_IMPORTS = {*IN_PROCESS.split(), *" ".join(REACHING).split()}
# the C++ runtime's, but for what reaches files, devices or threads
CXX_RUNTIME = re.compile(r"_Z|__cxa_|_Unwind_|__gxx_personality|__dynamic_cast")
CXX_REACHING = re.compile(r"filesystem|fstream|filebuf|random_device|_ZNSt6thread")

# Windows: the DLLs today's wheels import, and no networking or shell one
# (ws2_32, winhttp, wininet, urlmon, shell32). Not kernel32's names: they
# are mostly the static C runtime's, which moves with the runner's MSVC, and
# include files and processes, so on Windows this says less. Ours are CreateThread
# and the address waits (the pool), GlobalMemoryStatusEx,
# GetLogicalProcessorInformationEx, and LoadLibraryA with GetProcAddress (the
# Vulkan loader)
PE_IMPORTS = {"python3.dll", "kernel32.dll", "api-ms-win-core-synch-l1-2-0.dll"}


def elf_audit(data: bytes, wheel_name: str) -> list[str]:
    """
    Bound immediately and read-only after relocation, a stack that can't
    execute, PyInit_* the only export, the control-flow marking, and nothing
    imported beyond the lists above. 64-bit only: no 32-bit Linux wheel ships.
    """
    if data[4] != 2:
        return ["a 32-bit ELF, which no wheel is"]
    # every field below is read little-endian: a big-endian ELF would read as nonsense
    if data[5] != 1:
        return ["a big-endian ELF, which this reads as little-endian: read it with its own byte order first"]
    machine = struct.unpack_from("<H", data, 0x12)[0]
    phoff, shoff = struct.unpack_from("<QQ", data, 0x20)
    phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", data, 0x36)
    segments = []  # (type, flags, offset, size)
    for i in range(phnum):
        kind, flags, offset, _, _, size = struct.unpack_from("<IIQQQQ", data, phoff + i * phentsize)
        segments.append((kind, flags, offset, size))
    kinds = {kind: (flags, offset, size) for kind, flags, offset, size in segments}
    errors = []
    if 0x6474E551 not in kinds or kinds[0x6474E551][0] & 1:  # PT_GNU_STACK, PF_X
        errors.append("its stack is executable")
    if 0x6474E552 not in kinds:  # PT_GNU_RELRO
        errors.append("it has no RELRO segment")
    _, offset, size = kinds[2]  # PT_DYNAMIC
    dynamic = dict(struct.iter_unpack("<qQ", data[offset : offset + size]))
    if not (dynamic.get(30, 0) & 8 or dynamic.get(0x6FFFFFFB, 0) & 1):  # DF_BIND_NOW, DF_1_NOW
        errors.append("it binds lazily: no BIND_NOW")

    # the exports, from .dynsym's section, which stripping keeps
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    dynsym = next(s for s in sections if s[1] == 11)  # SHT_DYNSYM
    names = sections[dynsym[6]]  # its sh_link: the string table
    exported, imported = [], set()
    for at in range(dynsym[4], dynsym[4] + dynsym[5], dynsym[9]):
        name, info, _, index = struct.unpack_from("<IBBH", data, at)
        start = names[4] + name
        if info >> 4 in (1, 2, 10) and name:  # GLOBAL, WEAK or GNU_UNIQUE
            text = data[start : data.index(b"\0", start)].decode()
            if index != 0:
                exported.append(text)
            else:
                imported.add(text)
    if not exported or any(not n.startswith("PyInit_") for n in exported):
        errors.append(f"it exports {sorted(exported)}, not PyInit_* alone")
    python = {n for n in imported if n.startswith(("Py", "_Py"))}
    cxx = {n for n in imported if CXX_RUNTIME.match(n)}
    unknown = sorted(imported - python - cxx - C_IMPORTS) + sorted(n for n in cxx if CXX_REACHING.search(n))
    if unknown:
        errors.append(f"it imports {unknown}: add each to check_wheel.py's lists with what uses it, or don't")

    if machine in MARKS:
        instruction, feature = MARKS[machine]
        code = [data[o : o + s] for kind, flags, o, s in segments if kind == 1 and flags & 1]  # PT_LOAD, PF_X
        found = sum(c.count(instruction) for c in code)
        if found < MARKED:
            errors.append(f"{found} functions start with {instruction.hex()}: the control-flow flag is lost")
        if machine in NOTED and "manylinux" in wheel_name:
            words = []
            if 0x6474E553 in kinds:  # PT_GNU_PROPERTY: one NT_GNU_PROPERTY_TYPE_0 note
                _, offset, _ = kinds[0x6474E553]
                namesz, descsz = struct.unpack_from("<II", data, offset)
                at = offset + 12 + ((namesz + 3) & ~3)
                end = at + descsz
                while at < end:
                    kind, length = struct.unpack_from("<II", data, at)
                    if kind == feature:
                        words.append(struct.unpack_from("<I", data, at + 8)[0])
                    at += 8 + ((length + 7) & ~7)
            if not words or words[0] & NOTED[machine] != NOTED[machine]:
                errors.append("its GNU property doesn't mark it for IBT and SHSTK: an object linked in isn't")
    return errors


def pe_audit(data: bytes) -> list[str]:
    """
    Relocatable anywhere, data that can't execute, control flow guard with
    the load config that makes it work (the CRT's), on x64 shadow-stack
    compatible (/CETCOMPAT, a debug directory entry), and no DLL imported
    but PE_IMPORTS, none of it delay-loaded.
    """
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    machine, count = struct.unpack_from("<HH", data, pe + 4)
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    wide = struct.unpack_from("<H", data, optional)[0] == 0x20B
    characteristics = struct.unpack_from("<H", data, optional + 70)[0]
    wanted = {0x40: "DYNAMICBASE", 0x100: "NX_COMPAT", 0x4000: "GUARD_CF"}
    if wide:
        wanted[0x20] = "HIGH_ENTROPY_VA"
    errors = [f"it isn't {name}" for bit, name in wanted.items() if not characteristics & bit]
    # each section's (address, raw size, raw offset)
    table = optional + optional_size
    sections = [struct.unpack_from("<IIII", data, table + i * 40 + 8)[1:] for i in range(count)]

    def at(rva: int) -> int | None:
        # where an address's bytes are in the file
        offsets = [raw + rva - start for start, length, raw in sections if start <= rva < start + length]
        return offsets[0] if offsets else None

    def directory(index: int) -> tuple[int | None, int]:
        # a data directory's file offset (None if absent) and size
        rva, size = struct.unpack_from("<II", data, optional + (112 if wide else 96) + index * 8)
        return (at(rva) if size else None), size

    def text(rva: int) -> str:
        start = at(rva)
        return data[start : data.index(b"\0", start)].decode()

    # the import directory: a descriptor per DLL, each naming its imports
    # through a table of hint-and-name addresses, or ordinals
    entry, _ = directory(1)
    width, ordinal = (8, 1 << 63) if wide else (4, 1 << 31)
    while entry is not None:
        lookup, _, _, name, first = struct.unpack_from("<5I", data, entry)
        if name == 0:
            break
        dll = text(name).lower()
        names = set()
        thunk = at(lookup or first)
        while value := struct.unpack_from("<Q" if wide else "<I", data, thunk)[0]:
            names.add(f"#{value & 0xFFFF}" if value & ordinal else text(value + 2))
            thunk += width
        if dll not in PE_IMPORTS:
            errors.append(f"it imports {sorted(names)} from {dll}: add it to PE_IMPORTS with why, or don't")
        entry += 20
    if directory(13)[1]:
        errors.append("it delay-loads imports, which PE_IMPORTS can't see")

    config, _ = directory(10)
    guard = struct.unpack_from("<I", data, config + (0x90 if wide else 0x58))[0] if config else 0
    if not guard & 0x100:  # GuardFlags: IMAGE_GUARD_CF_INSTRUMENTED
        errors.append("its load config doesn't say control flow guard is instrumented")
    if machine == 0x8664:
        debug, size = directory(6)
        # each entry's (type, size, raw offset)
        starts = range(debug, debug + size, 28) if debug else ()
        entries = [struct.unpack_from("<12xII4xI", data, at) for at in starts]
        flags = [struct.unpack_from("<I", data, raw)[0] for kind, _, raw in entries if kind == 20]
        if not flags or not flags[0] & 1:  # IMAGE_DEBUG_TYPE_EX_DLLCHARACTERISTICS, CET_COMPAT
            errors.append("it isn't CETCOMPAT")
    return errors


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
    binary = wheel.read(found[0])
    if binary[:4] == b"\x7fELF":
        # its platform tag, from the file's name alone
        errors += [f"{found[0]}: {e}" for e in elf_audit(binary, Path(path).name)]
    elif binary[:2] == b"MZ":
        errors += [f"{found[0]}: {e}" for e in pe_audit(binary)]
    # the extension uses the stable ABI alone
    return errors + tool("abi3audit", "--strict", "--assume-minimum-abi3", "3.12", "--report", path)


if __name__ == "__main__":
    failed = False
    for wheel in sys.argv[1:]:
        for error in check(wheel):
            failed = True
            print(f"{wheel}: {error}")
    sys.exit(int(failed))
