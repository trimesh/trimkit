"""
vendor/ is what a build needs and nothing else, committed, so a build never
touches the network and this keeps building when the upstreams are gone:

  uv run scripts fetch [name ...]           re-derive it from the pinned commits:
                                            `git status vendor/` then shows nothing
  uv run scripts fetch --latest [name ...]  the same, moving each pin to its ref's HEAD
  uv run scripts outdated                   list refs that moved since
  uv run scripts patch [name ...]           write vendor/patches/ from what vendor/ holds

Our changes to an upstream are edited where they live, in vendor/NAME/, each
between two marker lines, `fix` (a bug or UB upstream) or `perf`, then a name:

  //// TRIMKIT PATCH: fix.lost-wakeup
  ...what we changed it to...
  //// TRIMKIT END PATCH

`patch` writes each name's unified diff, vendor/patches/NAME-lost-wakeup.patch,
as the difference between the upstream at its pin and vendor/, keeping the
text above the diff (`kind:` and why) as it was, then fetches the upstream
again from those patches and checks that vendor/ came back byte for byte.
Running it again changes nothing, and `fetch` puts every block back. Anything
changed outside a block is refused, naming where.

One rule picks the files. Every upstream downloads whole, its patches apply,
and then only these stay under vendor/NAME/, at the upstream's own paths:

  - what `keep` names, its licenses at least
  - whatever a dependency scan of the C++ compiling it reaches, on every wheel target

and `check` (`uv run scripts static`) rescans, so it can't drift. Standard
library only.
"""

import concurrent.futures
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from builder import CHECK, DOCKER, ROOT, TOML, Upstream, load, tracked_under

VENDOR = ROOT / "vendor"
# our changes to upstream code, as unified diffs: `vendor/patches/NAME.patch`,
# applied in the order `[upstream.X.patches]` lists them, each with its sha256
PATCHES = VENDOR / "patches"
HUNK = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _get(url: str, headers: dict | None = None) -> bytes:
    # unauthenticated on purpose: nothing here reads the environment, and a
    # fetch makes one API call per pin, far below the anonymous rate limit
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https url: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "trimkit-vendor", **(headers or {})})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as r:  # noqa: S310
        return r.read()


def _resolve(pin: Upstream) -> str:
    if pin.host == "gitlab.com":
        project = urllib.parse.quote(pin.repo, safe="")
        return json.loads(_get(f"https://gitlab.com/api/v4/projects/{project}/repository/commits/{pin.ref}"))[
            "id"
        ]
    return json.loads(_get(f"https://api.github.com/repos/{pin.repo}/commits/{pin.ref}"))["sha"]


def _tarball_url(pin: Upstream, commit: str) -> str:
    # either host's archive holds one top directory, `<name>-<commit>/`
    if pin.host == "gitlab.com":
        return f"https://gitlab.com/{pin.repo}/-/archive/{commit}/{pin.repo.split('/')[-1]}-{commit}.tar.gz"
    return f"https://codeload.github.com/{pin.repo}/tar.gz/{commit}"


# an upstream at a commit is the same bytes forever: `patch` and `fetch` read
# each one's archive from here after the first download
TARBALLS = ROOT / "build" / "upstreams"


def _tarball(pin: Upstream, commit: str) -> bytes:
    cached = TARBALLS / f"{pin.name}-{commit}.tar.gz"
    if cached.is_file():
        return cached.read_bytes()
    data = _get(_tarball_url(pin, commit))
    TARBALLS.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data


def _upstream_files(pin: Upstream, commit: str) -> dict[str, list[bytes]]:
    # every file of the upstream at `commit`, by its path in the repository
    files = {}
    with tarfile.open(fileobj=io.BytesIO(_tarball(pin, commit))) as tar:
        for member in tar.getmembers():
            if member.isfile():
                files[member.name.split("/", 1)[-1]] = (
                    tar.extractfile(member).read().splitlines(keepends=True)
                )
    return files


# -- closure: vendor only what the build reaches ------------------------------

# one triple of each kind a wheel is built for: Eigen's `Core` picks its SIMD
# headers by the target's predefined macros, so the closure is the union of them
CLOSURE_TARGETS = ("x86_64-linux-gnu", "aarch64-linux-gnu", "arm64-apple-macos11", "x86_64-pc-windows-msvc")
SOURCE_SUFFIXES = (".c", ".cc", ".cpp")


def _scan(files: list[Path], flags: list[str]) -> tuple[set[Path], set[str]]:
    """
    Every file the preprocessor's dependency scan of `files` reaches on any
    wheel target, and every include it couldn't find (`-MG`). No sysroot:
    standard headers are simply missing, which is all `-MM` needs.
    """
    clang = shutil.which("clang++")
    if clang is None:
        sys.exit("`clang++` is not installed: closure upstreams are scanned with it")

    def one(target: str) -> str:
        command = [
            clang,
            f"--target={target}",
            "-std=c++17",
            "-nostdinc++",
            "-nostdlibinc",
            # fast_float #errors without this, which standard headers would define
            "-DFLT_EVAL_METHOD=0",
            *flags,
            "-MM",
            "-MG",
        ]
        result = subprocess.run([*command, *map(str, files)], capture_output=True, text=True, check=False)
        if result.returncode:
            sys.exit(f"dependency scan for {target} failed:\n{result.stderr}")
        return result.stdout

    found, missing = set(), set()
    with concurrent.futures.ThreadPoolExecutor(len(CLOSURE_TARGETS)) as pool:
        for output in pool.map(one, CLOSURE_TARGETS):
            # make rules: `target.o: dependency dependency \` continued
            for word in output.replace("\\\n", " ").split():
                if word.endswith(":"):
                    continue
                path = Path(word)
                if path.is_file():
                    found.add(path.resolve())
                else:
                    missing.add(word)
    return found, missing


def _closure_inputs(pin: Upstream) -> tuple[list[Path], list[str]]:
    """
    What the scan starts from: the source files `keep` names and the sources of
    ours it compiles (src/tinybvh.cpp), and for every part compiling it in,
    that part's binding, fuzz harnesses and seed writer plus the sources of its
    other upstreams. The flags are those parts' include directories (`-I`,
    which `-MM` follows, where the build uses `-isystem`) and defines.
    """
    manifest = load()
    dest = VENDOR / pin.name
    roots = [dest / k for k in pin.keep if (dest / k).suffix in SOURCE_SUFFIXES]
    roots += [p for p in pin.compiled if not p.is_relative_to(dest)]
    flags = [f"-I{ROOT / 'src'}"]
    for part in manifest.parts.values():
        if pin.name not in {u.name for u in part.upstreams}:
            continue
        roots += [part.source, *(CHECK / "fuzz" / f"{h}.cpp" for h in part.fuzz)]
        roots += [CHECK / "fuzz" / f"{part.seeds}.cpp"] if part.seeds else []
        for upstream in part.upstreams:
            if upstream.name != pin.name:
                roots += upstream.compiled
            flags += [f"-I{ROOT / d}" for d in upstream.include] + [f"-D{d}" for d in upstream.defines]
    return sorted({r for r in roots if r.is_file()}), list(dict.fromkeys(flags))


def closure(pin: Upstream) -> tuple[set[Path], list[str]]:
    """
    The files under vendor/NAME/ the build reaches, and the includes it needs
    from there that aren't vendored. A fixpoint: scan the roots; each reached
    header brings the same-named files of `sources` (the implementation an
    include scan can't see); scan those; and repeat until nothing new is reached.
    """
    dest = (VENDOR / pin.name).resolve()
    roots, flags = _closure_inputs(pin)
    sources: dict[str, list[Path]] = {}
    for path in pin.compiled:
        sources.setdefault(path.stem, []).append(path.resolve())
    reached = {r.resolve() for r in roots if r.resolve().is_relative_to(dest)}
    pending, missing = list(roots), set()
    while pending:
        found, unfound = _scan(pending, flags)
        missing |= unfound
        pending = []
        for path in sorted(found - reached):
            if not path.is_relative_to(dest):
                continue
            reached.add(path)
            for source in sources.get(path.stem, []):
                if source not in reached:
                    reached.add(source)
                    pending.append(source)
    # an include that names a directory this upstream provides, but no file
    # in it: the closure needs a file `check` will find isn't vendored
    provided = {
        p.name for d in pin.include if (ROOT / d).is_dir() for p in (ROOT / d).iterdir() if p.is_dir()
    }
    needed = sorted(
        word for word in missing if not Path(word).is_absolute() and word.split("/")[0] in provided
    )
    return reached, needed


def _kept(pin: Upstream) -> set[Path]:
    # what `keep` names, each directory whole, and the LICENSE `license_from` writes
    dest = VENDOR / pin.name
    kept = {(dest / "LICENSE").resolve()} if pin.license_from else set()
    for name in pin.keep:
        path = dest / name
        kept |= {p.resolve() for p in path.rglob("*") if p.is_file()} if path.is_dir() else {path.resolve()}
    return kept


def _prune(pin: Upstream) -> None:
    # delete from a freshly fetched vendor/NAME/ everything the rule doesn't keep
    dest = VENDOR / pin.name
    reached, needed = closure(pin)
    if needed:
        sys.exit(f"{pin.name}: the build includes files the upstream doesn't have: {', '.join(needed)}")
    keep = reached | _kept(pin)
    for path in sorted(dest.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
        elif not path.is_dir() and path.resolve() not in keep:
            path.unlink()


def _record(table: str, key: str, value: str) -> None:
    """
    Rewrite one `key = "..."` line of `[table]` in trimkit.toml: the lines
    `fetch` owns, each upstream's `commit` and `tree_sha256` and the sha256
    of each of its patches. `key` is a regular expression.
    """
    text = TOML.read_text()
    section = re.compile(rf'(\[{re.escape(table)}\][^\[]*?{key} = ")[0-9a-f]*(")', re.S)
    if not section.search(text):
        # a new upstream's first fetch: its tree hash goes right after its commit
        section = re.compile(rf'(\[{re.escape(table)}\][^\[]*?commit = "[0-9a-f]+"\n)()', re.S)
        value = f'{key} = "{value}"\n'
    TOML.write_text(section.sub(lambda m: m[1] + value + m[2], text, count=1))


@dataclass(frozen=True)
class PatchHunk:
    # one hunk of a unified diff: where it starts on each side (1-based, or the
    # line it follows when that side is empty) and that side's lines
    path: str
    old_start: int
    old: tuple
    new_start: int
    new: tuple


def parse_patch(patch: bytes, name: str) -> tuple[bytes, list[PatchHunk]]:
    """
    A patch file's header, the text above its first `--- a/` (starting
    `kind: fix` or `kind: perf`, then why), and its hunks, strictly: every
    hunk exactly as long as its header counts. Bytes throughout, so line
    endings are never altered.
    """
    lines = patch.splitlines(keepends=True)
    if not lines or lines[0].strip() not in (b"kind: fix", b"kind: perf"):
        raise ValueError(f"{name}: must start with `kind: fix` or `kind: perf`, then why")
    index = next((i for i, line in enumerate(lines) if line.startswith(b"--- a/")), len(lines))
    if index == len(lines):
        raise ValueError(f"{name}: changes no file")
    header, hunks = b"".join(lines[:index]), []
    while index < len(lines):
        if not (lines[index].startswith(b"--- a/") and lines[index + 1].startswith(b"+++ b/")):
            raise ValueError(f"{name}: malformed at line {index + 1}")
        path = lines[index + 1][6:].strip().decode()
        index += 2
        while index < len(lines) and lines[index].startswith(b"@@"):
            match = HUNK.match(lines[index])
            if match is None:
                raise ValueError(f"{name}: malformed hunk at line {index + 1}")
            count, new_count = int(match[2] or b"1"), int(match[4] or b"1")
            index += 1
            old, new = [], []
            # exactly the lines the header counts: the next file's `--- a/`
            # header would otherwise read as one more removed line
            while index < len(lines):
                kind, text = lines[index][:1], lines[index][1:]
                if len(old) >= count and len(new) >= new_count and kind != b"\\":
                    break
                if kind not in (b" ", b"-", b"+", b"\\"):
                    raise ValueError(f"{name}: hunk at line {index + 1} ends before its header's line count")
                if kind in (b" ", b"-"):
                    old.append(text)
                if kind in (b" ", b"+"):
                    new.append(text)
                index += 1
            if len(old) != count or len(new) != new_count:
                raise ValueError(f"{name}: hunk ending at line {index} doesn't match its header's count")
            hunks.append(PatchHunk(path, int(match[1]), tuple(old), int(match[3]), tuple(new)))
    return header, hunks


def apply_hunks(source: list[bytes], hunks: list[PatchHunk], name: str, reverse: bool = False) -> list[bytes]:
    # one file's hunks, each matching exactly where it says, with no fuzz, or
    # nothing is guessed and it fails; `reverse` takes a patched file back
    result, cursor = [], 0
    for hunk in hunks:
        start, old, new = (
            (hunk.new_start, hunk.new, hunk.old) if reverse else (hunk.old_start, hunk.old, hunk.new)
        )
        start -= 1 if old else 0  # an empty side names the line it follows
        if start < cursor or source[start : start + len(old)] != list(old):
            raise ValueError(f"{name}: no longer applies to {hunk.path} at line {start + 1}")
        result += source[cursor:start] + list(new)
        cursor = start + len(old)
    return result + source[cursor:]


def apply_patch(directory: Path, patch: bytes, name: str) -> None:
    # a patch file applied to the files under `directory` (`fetch`)
    _, hunks = parse_patch(patch, name)
    for path in dict.fromkeys(h.path for h in hunks):
        target = directory / path
        lines = target.read_bytes().splitlines(keepends=True)
        target.write_bytes(b"".join(apply_hunks(lines, [h for h in hunks if h.path == path], name)))


# -- patches from vendor/: the marker blocks ------------------------------------

PATCH_BEGIN = re.compile(rb"^[ \t]*//// TRIMKIT PATCH: (fix|perf)\.([a-z0-9][a-z0-9-]*)[ \t]*\r?\n$")
PATCH_END = re.compile(rb"^[ \t]*//// TRIMKIT END PATCH[ \t]*\r?\n$")
PATCH_ANY = re.compile(rb"//// TRIMKIT")
CONTEXT = 3  # lines of context around each change, as `diff -u`
PLACEHOLDER = b"why: say why, and what covers it, then run `uv run scripts patch` again"


@dataclass(frozen=True)
class PatchBlock:
    # one marked block of a vendored file: the vendored lines [first, last]
    # (both markers), the kind and name its first marker gives, and the
    # upstream lines [old_first, old_end) it replaced
    kind: str
    name: str
    first: int
    last: int
    old_first: int = 0
    old_end: int = 0


def marked_blocks(path: str, lines: list[bytes]) -> list[PatchBlock]:
    # every block of a file, in order: never nested, never unclosed, never
    # touching the next (one unchanged line between them, or one block)
    blocks, opened = [], None
    for number, line in enumerate(lines):
        begin, end = PATCH_BEGIN.match(line), PATCH_END.match(line)
        if begin is None and end is None and PATCH_ANY.search(line):
            raise ValueError(
                f"{path}:{number + 1}: not a marker: "
                "`//// TRIMKIT PATCH: fix.name` or `//// TRIMKIT END PATCH`"
            )
        if begin is not None:
            if opened is not None:
                raise ValueError(
                    f"{path}:{number + 1}: a block opens inside the block at line {opened[0] + 1}"
                )
            opened = (number, begin[1].decode(), begin[2].decode())
        elif end is not None:
            if opened is None:
                raise ValueError(f"{path}:{number + 1}: a block ends that never opened")
            if blocks and blocks[-1].last + 1 == opened[0]:
                raise ValueError(
                    f"{path}:{opened[0] + 1}: touches the block before it: "
                    "merge them, or leave a line between"
                )
            blocks.append(PatchBlock(opened[1], opened[2], opened[0], number))
            opened = None
    if opened is not None:
        raise ValueError(f"{path}:{opened[0] + 1}: the block never ends: `//// TRIMKIT END PATCH`")
    return blocks


def _find(haystack: list[bytes], needle: list[bytes], start: int) -> int:
    # the first place at or after `start` that `needle` appears whole, or -1
    for at in range(start, len(haystack) - len(needle) + 1):
        if haystack[at : at + len(needle)] == needle:
            return at
    return -1


def replaced(path: str, upstream: list[bytes], vendored: list[bytes]) -> list[PatchBlock]:
    """
    Each block of a vendored file with the upstream lines it replaced: the
    text outside the blocks is the upstream's, found verbatim and in order,
    from the start to the end, so the upstream lines between two of those
    stretches are what the block between them replaced. Anything else
    changed is outside a block, and refused, naming the first line that
    differs.
    """
    blocks = marked_blocks(path, vendored)
    stretches = [(0, blocks[0].first if blocks else len(vendored))]
    stretches += [
        (b.last + 1, (blocks[i + 1].first if i + 1 < len(blocks) else len(vendored)))
        for i, b in enumerate(blocks)
    ]

    def outside(start: int, end: int, cursor: int) -> ValueError:
        # the stretch [start, end) isn't in the upstream past `cursor`: name
        # the first line where its best match there stops matching
        stretch, best = vendored[start:end], 0
        for at in range(cursor, len(upstream)):
            length = 0
            while (
                length < len(stretch)
                and at + length < len(upstream)
                and upstream[at + length] == stretch[length]
            ):
                length += 1
            best = max(best, length)
        return ValueError(f"{path}:{start + best + 1}: changed outside a `//// TRIMKIT PATCH` block")

    first = vendored[stretches[0][0] : stretches[0][1]]
    if upstream[: len(first)] != first:
        raise outside(0, len(first), 0)
    cursor, found = len(first), []
    for index, block in enumerate(blocks):
        start, end = stretches[index + 1]
        stretch = vendored[start:end]
        if index + 1 == len(blocks):
            at = len(upstream) - len(stretch)
            if at < cursor or upstream[at:] != stretch:
                raise outside(start, end, cursor)
        else:
            at = _find(upstream, stretch, cursor)
            if at < 0:
                raise outside(start, end, cursor)
        found.append(PatchBlock(block.kind, block.name, block.first, block.last, cursor, at))
        cursor = at + len(stretch)
    if not blocks and upstream != vendored:
        raise (
            outside(0, len(vendored), 0)
            if upstream[: len(vendored)] != vendored
            else ValueError(f"{path}:{len(vendored) + 1}: changed outside a `//// TRIMKIT PATCH` block")
        )
    return found


def _header(patch: bytes) -> bytes:
    # a patch file's text above its first `--- a/`, or all of it when it has no diff yet
    if patch.startswith(b"--- a/"):
        return b""
    at = patch.find(b"\n--- a/")
    return patch if at < 0 else patch[: at + 1]


def patch_file_name(upstream: str, name: str) -> str:
    return f"{upstream}-{name}.patch"


def _hunks_text(path: str, before: list[bytes], changes: list[tuple[int, int, list[bytes]]]) -> bytes:
    # one file's changes, each (first, end, new lines) of `before`, as a
    # unified diff with CONTEXT lines: changes whose context meets share a hunk
    groups: list[list[tuple[int, int, list[bytes]]]] = []
    for change in sorted(changes):
        if groups and change[0] - groups[-1][-1][1] <= 2 * CONTEXT:
            groups[-1].append(change)
        else:
            groups.append([change])
    text, shift = [f"--- a/{path}\n+++ b/{path}\n".encode()], 0
    for group in groups:
        old_first, old_end = max(0, group[0][0] - CONTEXT), min(len(before), group[-1][1] + CONTEXT)
        body, new_count, cursor = [], 0, old_first
        for first, end, new in group:
            body += [b" " + line for line in before[cursor:first]]
            body += [b"-" + line for line in before[first:end]]
            body += [b"+" + line for line in new]
            new_count += (first - cursor) + len(new)
            cursor = end
        body += [b" " + line for line in before[cursor:old_end]]
        new_count += old_end - cursor
        old_count = old_end - old_first
        if any(not line.endswith(b"\n") for line in body):
            raise ValueError(f"{path}: a change reaches its last line, which has no newline")
        old_at = old_first + (1 if old_count else 0)
        new_at = old_first + shift + (1 if new_count else 0)
        text.append(f"@@ -{old_at},{old_count} +{new_at},{new_count} @@\n".encode())
        text += body
        shift += new_count - old_count
    return b"".join(text)


def derive_patches(
    upstream: str, pristine: dict[str, list[bytes]], vendored: dict[str, list[bytes]]
) -> dict[str, tuple[str, bytes]]:
    """
    Every patch the blocks in an upstream's vendored files make, as
    {file name: (kind, diff)}, in the order `fetch` applies them (by name):
    each against the upstream with the patches before it applied, so they
    apply in that order exactly, and the last leaves every file as vendored.
    """
    blocks = {
        path: replaced(path, pristine[path], lines) for path, lines in vendored.items() if path in pristine
    }
    kinds: dict[str, str] = {}
    for path, found in blocks.items():
        for block in found:
            if kinds.setdefault(block.name, block.kind) != block.kind:
                raise ValueError(
                    f"{path}:{block.first + 1}: `{block.name}` is a {kinds[block.name]} elsewhere"
                )
    order = sorted(kinds)
    patches = {}
    for index, name in enumerate(order):
        text = []
        for path in sorted(blocks):
            # the file with the patches before this one applied
            before, changes = [], []
            cursor = 0
            for block in blocks[path]:
                before += pristine[path][cursor : block.old_first]
                block_lines = vendored[path][block.first : block.last + 1]
                if block.name == name:
                    changes.append((len(before), len(before) + block.old_end - block.old_first, block_lines))
                if order.index(block.name) < index:
                    before += block_lines
                else:
                    before += pristine[path][block.old_first : block.old_end]
                cursor = block.old_end
            before += pristine[path][cursor:]
            if changes:
                text.append(_hunks_text(path, before, changes))
        patches[patch_file_name(upstream, name)] = (kinds[name], b"".join(text))
    return patches


def _vendored_files(name: str) -> dict[str, list[bytes]]:
    dest = VENDOR / name
    return {
        p.relative_to(dest).as_posix(): p.read_bytes().splitlines(keepends=True)
        for p in sorted(dest.rglob("*"))
        if p.is_file()
    }


def _write_patches_table(name: str, digests: dict[str, str]) -> None:
    # `[upstream.NAME.patches]` as `patch` found the blocks: one line a patch,
    # in the order they apply, or no table for an upstream without any
    text = TOML.read_text()
    table = re.compile(rf"\n\[upstream\.{re.escape(name)}\.patches\]\n(?:\"[^\"\n]+\" = \"[0-9a-f]*\"\n)*")
    lines = "".join(f'"{patch}" = "{digest}"\n' for patch, digest in digests.items())
    body = f"\n[upstream.{name}.patches]\n{lines}" if digests else "\n"
    if table.search(text):
        text = table.sub(lambda _: body, text, count=1)
    elif digests:
        # after the upstream's own table, before whatever table comes next
        own = re.compile(rf"(\[upstream\.{re.escape(name)}\]\n(?:[^\[\n][^\n]*\n|\n)*?)(?=\n\[|\Z)", re.S)
        text = own.sub(lambda m: m[1] + body.rstrip("\n") + "\n", text, count=1)
    TOML.write_text(text)


def patch(names: list[str]) -> None:
    """
    vendor/patches/ from the blocks in vendor/: for each upstream named (all
    by default), its patch files written from its blocks (each keeping the
    text above its diff, a new one starting with a placeholder to replace),
    the ones no block names any more removed, and the manifest's table and
    hashes rewritten. Then it is fetched again, from those patches and the
    cached upstream, and must come back byte for byte; if it doesn't, or
    anything fails, vendor/NAME is put back as it was and nothing is kept.
    """
    pins = load().upstreams
    placeholders = []
    for name in names or list(pins):
        pin = pins[name]
        vendored = _vendored_files(name)
        upstream_files = _upstream_files(pin, pin.commit)
        # what `fetch` writes itself (a LICENSE from `license_from`) is no upstream file
        written = {"LICENSE"} if pin.license_from else set()
        compared = {p: lines for p, lines in vendored.items() if p not in written}
        derived = derive_patches(
            name, {p: upstream_files[p] for p in compared if p in upstream_files}, compared
        )
        if derived and pin.system:
            raise ValueError(f"{name}: a system copy can stand in for it, so it can't be patched")
        snapshot = {p: b"".join(lines) for p, lines in vendored.items()}
        manifest_before = TOML.read_text()
        written_before = {p: p.read_bytes() for p in PATCHES.glob(f"{name}-*.patch")}
        try:
            digests = {}
            for patch_name, (kind, diff) in derived.items():
                path = PATCHES / patch_name
                # what is above its diff, as written: a header alone starts a patch too
                header = (
                    _header(path.read_bytes())
                    if path.exists()
                    else (f"kind: {kind}\n\n".encode() + PLACEHOLDER + b"\n\n")
                )
                if header.splitlines()[0].strip() != f"kind: {kind}".encode():
                    raise ValueError(
                        f"{patch_name}: its header says {header.splitlines()[0].decode().strip()}, "
                        f"its blocks `{kind}`"
                    )
                if PLACEHOLDER in header:
                    placeholders.append(f"vendor/patches/{patch_name}")
                path.write_bytes(header + diff)
                digests[patch_name] = hashlib.sha256(header + diff).hexdigest()
            for stale in set(written_before) - {PATCHES / p for p in derived}:
                stale.unlink()
            _write_patches_table(name, digests)
            # the manifest as just written: `load` caches the one read before
            load.cache_clear()
            _fetch_into(load().upstreams[name], latest=False)
            after = {p: b"".join(lines) for p, lines in _vendored_files(name).items()}
            if after != snapshot:
                differ = sorted(
                    set(after) ^ set(snapshot) | {p for p in after if snapshot.get(p) != after[p]}
                )
                raise ValueError(
                    f"{name}: fetching from the patches doesn't give back vendor/{name}: "
                    f"{', '.join(differ[:5])}"
                )
        except BaseException:
            dest = VENDOR / name
            shutil.rmtree(dest, ignore_errors=True)
            for path, data in snapshot.items():
                (dest / path).parent.mkdir(parents=True, exist_ok=True)
                (dest / path).write_bytes(data)
            for path in PATCHES.glob(f"{name}-*.patch"):
                path.unlink()
            for path, data in written_before.items():
                path.write_bytes(data)
            TOML.write_text(manifest_before)
            load.cache_clear()
            raise
        print(f"{name}: {len(derived)} patches from vendor/{name}, which fetches back byte for byte")
    if placeholders:
        sys.exit(
            "say why in each new patch, above its diff, then run `uv run scripts patch` again:\n  "
            + "\n  ".join(placeholders)
        )


def check_patches(name: str, pin: Upstream) -> list[str]:
    """
    Offline: an upstream's patches are exactly what its blocks say. Taken
    back off vendor/NAME, last first, they must leave no marker behind, and
    from what that leaves, the blocks must derive the very same patch files.
    """
    vendored = _vendored_files(name)
    rerun = f"run `uv run scripts patch {name}`"
    pristine = {p: list(lines) for p, lines in vendored.items()}
    headers = {}
    try:
        for patch_name in reversed(list(pin.patches)):
            header, hunks = parse_patch((PATCHES / patch_name).read_bytes(), patch_name)
            headers[patch_name] = header
            for path in dict.fromkeys(h.path for h in hunks):
                pristine[path] = apply_hunks(
                    pristine[path], [h for h in hunks if h.path == path], patch_name, reverse=True
                )
    except (ValueError, KeyError) as error:
        return [f"vendor/{name} changed since its patches were written ({error}): {rerun}"]
    errors = [
        f"vendor/{name}/{path}:{number + 1}: a marker no patch adds: {rerun}"
        for path, lines in pristine.items()
        for number, line in enumerate(lines)
        if PATCH_ANY.search(line)
    ]
    if errors:
        return errors
    try:
        derived = derive_patches(name, pristine, vendored)
    except ValueError as error:
        return [str(error)]
    if list(derived) != list(pin.patches):
        return [
            f"vendor/{name}: its blocks make {sorted(derived) or 'no patches'}, "
            f"trimkit.toml lists {list(pin.patches) or 'none'}: {rerun}"
        ]
    errors = [
        f"vendor/patches/{p}: isn't what the blocks in vendor/{name} make: {rerun}"
        for p, (_, diff) in derived.items()
        if (PATCHES / p).read_bytes() != headers[p] + diff
    ]
    errors += [
        f"vendor/patches/{p}: still says `{PLACEHOLDER.decode()}`"
        for p in derived
        if PLACEHOLDER in headers[p]
    ]
    return errors


def tree_sha256(name: str, paths: list[Path]) -> str:
    """
    One hash of everything vendored for an upstream, patches applied: every
    file's path and contents, in path order. `git diff vendor/` names what
    changed when it doesn't match.
    """
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(VENDOR / name).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _git(*arguments: str, check: bool) -> None:
    # git in the repository, quietly: fetch keeps vendor/ as git has it
    subprocess.run(
        [shutil.which("git") or "git", "-C", str(ROOT), *arguments], check=check, capture_output=True
    )


def _fetch_one(pin: Upstream, latest: bool) -> None:
    # all or nothing: whatever stops it partway (a patch that no longer
    # applies, the network) leaves vendor/NAME and its pins as git has them
    try:
        _fetch_into(pin, latest)
    except BaseException:
        directory = f"vendor/{pin.name}"
        _git("checkout", "--", directory, check=False)
        # -x as well: an upstream's own .gitignore must not hide what it left
        _git("clean", "-fdqx", "--", directory, check=False)
        raise


def _fetch_into(pin: Upstream, latest: bool) -> None:
    name = pin.name
    commit = _resolve(pin) if latest else pin.commit

    dest = VENDOR / name
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)

    with tarfile.open(fileobj=io.BytesIO(_tarball(pin, commit))) as tar:
        for member in tar.getmembers():
            # strip the `<repo>-<commit>/` prefix both hosts add
            path = member.name.split("/", 1)[-1]
            if not member.isfile():
                continue
            # a member named `src/../../x` must still land inside `dest`
            target = (dest / path).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError(f"{name}: tarball member escapes vendor/: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    absent = [k for k in pin.keep if not (dest / k).exists()]
    if absent:
        sys.exit(f"{name}: `keep` names what the upstream doesn't have: {', '.join(absent)}")

    if pin.license_from:
        # keep every leading comment block: that is where the license lives
        source = (dest / pin.license_from).read_text()
        blocks = re.match(r"(\s*/\*.*?\*/)+", source, re.S)
        if blocks is None:
            raise ValueError(f"{name}: no license comment in {pin.license_from}")
        (dest / "LICENSE").write_text(blocks.group(0).strip() + "\n")

    # our changes, in order, each recorded by hash
    for patch_name in pin.patches:
        patch = (PATCHES / patch_name).read_bytes()
        apply_patch(dest, patch, patch_name)
        _record(f"upstream.{name}.patches", f'"{re.escape(patch_name)}"', hashlib.sha256(patch).hexdigest())

    # patched first: a patch can change what is included
    _prune(pin)

    # its pins once it is whole: the commit, so the manifest always names exact
    # bytes, and the tree of everything just written
    _record(f"upstream.{name}", "commit", commit)
    _record(f"upstream.{name}", "tree_sha256", tree_sha256(name, [p for p in dest.rglob("*") if p.is_file()]))
    # a file the upstream added is intended for the index, as every build
    # reads only what git tracks
    _git("add", "--intent-to-add", "--", f"vendor/{name}", check=True)
    print(f"{name}: {pin.ref} @ {commit[:12]}")


def fetch(names: list[str], latest: bool = False) -> list[str]:
    # every named upstream (all of them by default); returns why any whose
    # patch no longer applies stayed as it was, for the caller to report
    pins = load().upstreams
    held = []
    for name in names or list(pins):
        try:
            _fetch_one(pins[name], latest)
        except ValueError as error:
            held.append(f"{name}: {error}")
    return held


def check() -> None:
    # vendor/ holds exactly the upstreams and their patches, every patch is
    # the one its hash names, and every fetched upstream is the tree its
    # `tree_sha256` names
    pins = load().upstreams
    files = tracked_under(VENDOR)
    present = {p.relative_to(VENDOR).parts[0] for p in files} - {PATCHES.name}
    extra, missing = sorted(present - set(pins)), sorted(set(pins) - present)
    errors = [f"vendor/{name} isn't an upstream in trimkit.toml" for name in extra]
    errors += [f"vendor/{name} is missing: run `uv run scripts fetch {name}`" for name in missing]
    fetched = [name for name in sorted(pins) if name in present]
    errors += [
        f"vendor/{name} doesn't match its tree_sha256: see `git status vendor/{name}` and `git diff`"
        for name in fetched
        if tree_sha256(name, [p for p in files if p.is_relative_to(VENDOR / name)]) != pins[name].tree_sha256
    ]
    # vendor/NAME holds exactly what the rule keeps: a new include needs a
    # re-fetch, and a file nothing reaches any more is stale
    for name in fetched:
        pin = pins[name]
        reached, needed = closure(pin)
        dest = VENDOR / name
        kept = _kept(pin)
        vendored = {p.resolve() for p in files if p.is_relative_to(dest)}
        refetch = f"re-run `uv run scripts fetch {name}`"
        errors += [
            f"vendor/{name}: the build includes {word}, which isn't vendored: {refetch}" for word in needed
        ]
        errors += [
            f"vendor/{name}: nothing the build compiles reaches {p.relative_to(ROOT.resolve())}: {refetch}"
            for p in sorted(vendored - reached - kept)
        ]
        errors += [
            f"{p.relative_to(ROOT.resolve())} is reached but not tracked: `git add` it"
            for p in sorted(reached - vendored)
        ]
    listed = {patch: digest for pin in pins.values() for patch, digest in pin.patches.items()}
    on_disk = {p.name for p in files if p.parent == PATCHES}
    errors += [
        f"vendor/patches/{patch} isn't listed in trimkit.toml" for patch in sorted(on_disk - set(listed))
    ]
    errors += [
        f"vendor/patches/{patch} doesn't match its sha256 in trimkit.toml: re-fetch its upstream"
        for patch, digest in sorted(listed.items())
        if patch not in on_disk or hashlib.sha256((PATCHES / patch).read_bytes()).hexdigest() != digest
    ]
    # and every patch is exactly what the blocks in its upstream say
    for name in fetched:
        errors += check_patches(name, pins[name])
    if errors:
        sys.exit("\n".join(errors))
    print(
        f"ok: {len(fetched)} upstreams match their tree_sha256, and {len(listed)} patches their sha256"
        " and the blocks in vendor/"
    )


def outdated() -> None:
    stale = 0
    for name, pin in load().upstreams.items():
        latest = _resolve(pin)
        if latest != pin.commit:
            stale += 1
            print(f"{name}: pinned {pin.commit[:12]}, {pin.ref} is at {latest[:12]}")
    if stale:
        sys.exit(f"{stale} upstreams moved")


# The pins outside vendor/ that nothing else moves: the GitHub Actions the
# workflows use, by commit with the release in a comment, and the image's
# base, by digest. Each moves to its newest release a week old, the lock's
# own cooldown (`exclude-newer`), so a hijacked release has time to be pulled
COOLDOWN = timedelta(days=7)
WORKFLOWS = ROOT / ".github" / "workflows"
DOCKERFILE = DOCKER / "Dockerfile"
USES = re.compile(r"(uses: )([\w.-]+/[\w.-]+)@([0-9a-f]{40}) # (v[\w.-]+)")
RELEASE = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
BASE = re.compile(r"^FROM ([\w.-]+):([\w.-]+)@(sha256:[0-9a-f]{64})$", re.MULTILINE)


def _ripe(stamp: str) -> bool:
    # an ISO 8601 time at least the cooldown ago
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")) <= datetime.now(UTC) - COOLDOWN


def _newest_action(repo: str) -> tuple[str, str]:
    # its highest vX.Y.Z release past the cooldown, by version rather than by
    # date (a maintenance release of an old line often comes out last), and
    # the commit that release's tag names
    releases = json.loads(_get(f"https://api.github.com/repos/{repo}/releases?per_page=100"))
    ripe = [
        r["tag_name"]
        for r in releases
        if not (r["prerelease"] or r["draft"])
        and RELEASE.fullmatch(r["tag_name"])
        and _ripe(r["published_at"])
    ]
    tag = max(ripe, key=lambda t: tuple(int(n) for n in RELEASE.fullmatch(t).groups()))
    target = json.loads(_get(f"https://api.github.com/repos/{repo}/git/ref/tags/{tag}"))["object"]
    # an annotated tag names a tag object, which names the commit
    while target["type"] == "tag":
        target = json.loads(_get(target["url"]))["object"]
    return tag, target["sha"]


def _pinned_use(match: re.Match, newest: dict) -> str:
    # one `uses:` line moved to its action's newest release: the commit, then
    # the release in a comment
    tag, commit = newest[match[2]]
    return f"{match[1]}{match[2]}@{commit} # {tag}"


def update_actions() -> None:
    newest = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text()
        for repo in sorted({m.group(2) for m in USES.finditer(text)} - set(newest)):
            newest[repo] = _newest_action(repo)
        moved = USES.sub(lambda m: _pinned_use(m, newest), text)
        if moved != text:
            path.write_text(moved)
    for repo, (tag, commit) in newest.items():
        print(f"{repo}: {tag} ({commit[:12]})")


def update_base_image() -> None:
    # the base's tag as Docker Hub serves it now, once its linux/amd64 image
    # was built at least the cooldown ago: that digest, and with it the
    # Debian snapshot the image records, which every package is installed from
    text = DOCKERFILE.read_text()
    match = BASE.search(text)
    name, tag, pinned = match.groups()
    repository = name if "/" in name else f"library/{name}"
    token = json.loads(
        _get(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repository}:pull")
    )["token"]
    registry = f"https://registry-1.docker.io/v2/{repository}"
    accept = (
        "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, "
        "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
    )
    headers = {"Authorization": f"Bearer {token}", "Accept": accept}
    index_bytes = _get(f"{registry}/manifests/{tag}", headers)
    digest = "sha256:" + hashlib.sha256(index_bytes).hexdigest()
    index = json.loads(index_bytes)
    amd64 = next(
        m["digest"]
        for m in index["manifests"]
        if m.get("platform", {}).get("os") == "linux" and m.get("platform", {}).get("architecture") == "amd64"
    )
    config = json.loads(_get(f"{registry}/manifests/{amd64}", headers))["config"]["digest"]
    created = json.loads(_get(f"{registry}/blobs/{config}", headers))["created"]
    if digest != pinned and _ripe(created):
        DOCKERFILE.write_text(text.replace(f"{name}:{tag}@{pinned}", f"{name}:{tag}@{digest}"))
    print(f"{name}:{tag}: {digest if _ripe(created) else pinned} (newest built {created[:10]})")
