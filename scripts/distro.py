"""
Distribution packages, generated from trimkit.toml and pyproject.toml rather
than kept by hand, built from the sdist and linted the way each distribution
does: a Debian source and binary package (dpkg-buildpackage, then lintian)
and a Fedora spec, SRPM and RPMs (rpmbuild, then rpmlint), each building
against the system's copies of every upstream trimkit.toml allows and
running the test suite in its own build. build/distro/debian/ and
build/distro/fedora/ hold exactly what a distribution's maintainer commits;
sources/ the tarballs they build from, and built/ what `--run` makes. None of it
is committed here.

The images are pinned by digest; their package repositories are not (apt
and dnf fetch the distributions' current packages), which is the point: this
is the build a distribution would run today.

Standard library only.
"""

import datetime
import email.utils
import platform
import re
import shutil
import subprocess
import tarfile
import textwrap
from pathlib import Path

import builder
import licenses


def distribution(name: str) -> dict:
    # its [distribution.NAME] in trimkit.toml: the image to build in, and the packages it needs
    return builder.load().distributions[name]


def project() -> dict:
    return builder.pyproject()["project"]


def system_upstreams() -> list[str]:
    # every upstream trimkit.toml lets come from the system
    return [name for name, u in builder.load().upstreams.items() if u.system]


def bundled() -> list[builder.Upstream]:
    # every vendored upstream a distribution build still compiles in
    return [u for u in builder.load().compiled_in if not u.own and not u.system]


def requirements(name: str) -> list[str]:
    # what building and testing need on a distribution, the upstreams it provides included
    system = [builder.load().upstreams[n].distro[name] for n in system_upstreams()]
    return [*distribution(name)["requires"], *system]


def config_settings() -> list[str]:
    return ["debug=true", f"system={','.join(system_upstreams())}"]


def description() -> tuple[str, str]:
    # pyproject.toml's description: what comes before its colon is the one-line summary
    long = project()["description"]
    return long.split(":")[0], long


def debian_license(expression: str) -> str:
    # SPDX's `A OR B` and `A WITH X-exception` as Debian's copyright format
    # writes them: `A or B`, `A with X exception`
    return re.sub(r" WITH (\S+)-exception\b", r" with \1 exception", expression).replace(" OR ", " or ")


def debian_license_text(text: str) -> str:
    # Debian policy: the Apache License is a pointer to its own copy, any
    # exception appended to it kept, and lintian errors on the full text
    if "Apache License" not in text or "TERMS AND CONDITIONS" not in text:
        return text
    pointer = (
        "On Debian systems, the full text of the Apache License, Version 2.0,"
        " is in /usr/share/common-licenses/Apache-2.0."
    )
    exception = text.find("--- LLVM Exceptions")
    return pointer + ("\n\n" + text[exception:] if exception >= 0 else "")


def released() -> datetime.datetime:
    # a changelog entry's date: when the packaging was generated, to the second
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0)


def debian(out: Path) -> Path:
    """
    The debian/ directory, written into `out`.
    """
    meta = project()
    author = meta["authors"][0]
    maintainer = f"{author['name']} <{author['email']}>"
    summary, long = description()
    debian_dir = out / "debian"
    shutil.rmtree(debian_dir, ignore_errors=True)
    (debian_dir / "source").mkdir(parents=True)
    (debian_dir / "tests").mkdir()
    depends = requirements("debian")
    build_depends = ",\n ".join(["debhelper-compat (= 13)", "dh-sequence-python3", *depends])
    long_lines = "\n".join(" " + line for line in textwrap.wrap(long, 72))
    (debian_dir / "control").write_text(
        f"Source: {meta['name']}\n"
        "Section: python\n"
        "Priority: optional\n"
        f"Maintainer: {maintainer}\n"
        f"Build-Depends:\n {build_depends},\n"
        "Standards-Version: 4.7.2\n"
        f"Homepage: {meta['urls']['homepage']}\n"
        f"Vcs-Browser: {meta['urls']['homepage']}\n"
        f"Vcs-Git: {meta['urls']['homepage']}.git\n"
        "Rules-Requires-Root: no\n"
        "\n"
        f"Package: python3-{meta['name']}\n"
        "Architecture: any\n"
        "Depends: ${python3:Depends}, ${shlibs:Depends}, ${misc:Depends}\n"
        f"Description: {summary}\n{long_lines}\n"
    )
    settings = " ".join(f"--config-setting={s}" for s in config_settings())
    (debian_dir / "rules").write_text(
        "#!/usr/bin/make -f\n"
        f"export PYBUILD_NAME={meta['name']}\n"
        f"export PYBUILD_BUILD_ARGS={settings}\n"
        "# the tests, against the built package, without the wall-clock bounds a\n"
        "# loaded build machine can miss\n"
        "export PYBUILD_BEFORE_TEST=cp -r {dir}/tests {build_dir}\n"
        'export PYBUILD_TEST_ARGS=-m "not timing" {build_dir}/tests\n'
        "export PYBUILD_AFTER_TEST=rm -r {build_dir}/tests\n"
        "\n"
        "%:\n"
        "\tdh $@ --buildsystem=pybuild\n"
        "\n"
        "# trixie's dwz can't read the DWARF 5 clang writes (.debug_str_offsets),\n"
        "# so the debug information goes to the dbgsym package as it is\n"
        "override_dh_dwz:\n"
    )
    (debian_dir / "rules").chmod(0o755)
    (debian_dir / "source" / "format").write_text("3.0 (quilt)\n")
    (debian_dir / "changelog").write_text(
        f"{meta['name']} ({meta['version']}-1) unstable; urgency=medium\n\n"
        f"  * Generated from trimkit.toml by scripts/distro.py.\n\n"
        f" -- {maintainer}  {email.utils.format_datetime(released())}\n"
    )
    (debian_dir / "tests" / "control").write_text(
        'Test-Command: cp -r tests "$AUTOPKGTEST_TMP" && cd "$AUTOPKGTEST_TMP" && '
        'python3 -m pytest -m "not timing" tests\n'
        "Depends: @, python3-pytest, python3-hypothesis, python3-lark, python3-svg.path\n"
        "Restrictions: allow-stderr\n"
    )
    (debian_dir / "copyright").write_text(debian_copyright(maintainer))
    return debian_dir


def debian_copyright(maintainer: str) -> str:
    """
    Debian copyright format 1.0: ours, then each vendored upstream with its own
    license text, as its LICENSE file has it.
    """
    meta = project()
    paragraphs = [
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n"
        f"Upstream-Name: {meta['name']}\n"
        f"Upstream-Contact: {maintainer}\n"
        f"Source: {meta['urls']['homepage']}\n",
        "Files: *\n"
        f"Copyright: 2026 {licenses.OWNER}\n"
        f"License: {licenses.OWN}\n" + indented((builder.ROOT / "LICENSE").read_text()),
    ]
    for name, pin in builder.load().upstreams.items():
        if pin.own or pin.system:
            continue
        texts = [debian_license_text(p.read_text()) for p in licenses.license_files_of(name)]
        paragraphs.append(
            f"Files: vendor/{name}/*\n"
            f"Copyright: {licenses.copyright_of(pin).removeprefix('Copyright ').removeprefix('(c) ')}\n"
            f"License: {debian_license(pin.spdx)}\n" + indented("\n\n".join(texts))
        )
    return "\n".join(paragraphs)


def indented(text: str) -> str:
    # a license text as a copyright file's continuation lines, blank ones as `.`
    return "".join(f" {line}\n" if line.strip() else " .\n" for line in text.strip().splitlines())


def fedora(out: Path) -> Path:
    """
    The Fedora spec and its rpmlintrc, written into `out`.
    """
    out.mkdir()
    meta = project()
    author = meta["authors"][0]
    summary, long = description()
    manifest = builder.load()
    spdx = sorted({u.spdx for u in manifest.upstreams.values() if not u.system} | {licenses.OWN})
    license_expression = " AND ".join(f"({e})" if " OR " in e else e for e in spdx)
    requires = requirements("fedora")
    removed = " ".join(f"vendor/{n}" for n in system_upstreams())
    settings = " ".join(f"-C{s}" for s in config_settings())
    spec = out / f"python-{meta['name']}.spec"
    name = meta["name"]
    lines = [
        "# the build compiles everything with clang, LTO included, so it is",
        "# the toolchain for every flag Fedora adds",
        "%global toolchain clang",
        "",
        f"Name:           python-{name}",
        f"Version:        {meta['version']}",
        "Release:        1%{?dist}",
        f"Summary:        {summary[0].upper() + summary[1:]}",
        f"License:        {license_expression}",
        f"URL:            {meta['urls']['homepage']}",
        "Source0:        %{pypi_source " + name + "}",
        *(f"BuildRequires:  {r}" for r in requires),
        "",
        "%global _description %{expand:",
        *textwrap.wrap(long, 79),
        "}",
        "",
        "%description %_description",
        "",
        f"%package -n python3-{name}",
        "Summary:        %{summary}",
        *(f"Provides:       bundled({u.name}) = 0^git{u.commit[:12]}" for u in bundled()),
        "",
        f"%description -n python3-{name} %_description",
        "",
        "%prep",
        f"%autosetup -n {name}-%{{version}}",
        "# the upstreams built against Fedora's own copies, as bundling them is not",
        f"rm -r {removed}",
        "",
        "%build",
        f"%pyproject_wheel {settings}",
        "",
        "%install",
        "%pyproject_install",
        f"%pyproject_save_files -l {name}",
        "",
        "%check",
        "# without the wall-clock bounds a loaded build machine can miss",
        '%pytest -m "not timing" tests',
        "",
        f"%files -n python3-{name} -f %{{pyproject_files}}",
        "%doc README.md",
        "",
        "%changelog",
        f"* {released().strftime('%a %b %d %Y')} {author['name']} <{author['email']}> - {meta['version']}-1",
        "- Generated from trimkit.toml by scripts/distro.py",
    ]
    spec.write_text("\n".join(lines) + "\n")
    # names like numpy aren't misspellings
    (out / f"python-{name}.rpmlintrc").write_text('addFilter("spelling-error")\n')
    return spec


def debian_script(name: str, version: str) -> str:
    # inside the Debian image: install the build dependencies, build the
    # source and binary packages from the orig tarball, then lint them
    return textwrap.dedent(
        f"""\
        set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get -qq update
        tools="build-essential dpkg-dev devscripts equivs lintian"
        apt-get -qq install -y --no-install-recommends $tools >/dev/null
        mkdir -p /work && cd /work
        cp /out/sources/{name}_{version}.orig.tar.gz .
        tar -xzf {name}_{version}.orig.tar.gz
        cp -r /out/debian {name}-{version}/debian
        cd {name}-{version}
        mk-build-deps --install --remove --tool "apt-get -y --no-install-recommends" debian/control >/dev/null
        # the whole log kept, and its end shown when the build fails
        mkdir -p /out/built
        log=/out/built/debian-build.log
        dpkg-buildpackage -us -uc > $log 2>&1 || {{ tail -60 $log; exit 1; }}
        grep -E "passed|failed" $log | tail -1
        cd /work
        cp *.deb *.dsc *.debian.tar.* *.changes *.buildinfo /out/built/
        # generated for maintainers to start from: the upload that brings it into
        # Debian is the one that closes its ITP bug
        ignored=initial-upload-closes-no-bugs
        lintian --fail-on error,warning --display-info --suppress-tags $ignored {name}_{version}-1_*.changes
        """
    )


def fedora_script(name: str, version: str) -> str:
    # inside the Fedora image: install the build dependencies, build the SRPM
    # and RPMs, then lint the spec and everything built
    return textwrap.dedent(
        f"""\
        set -e
        # rpmlint spell-checks in en_US, so its locale too
        tools="rpm-build rpmlint pyproject-rpm-macros dnf-command(builddep) glibc-langpack-en"
        dnf -q -y install $tools >/dev/null
        mkdir -p ~/rpmbuild/SOURCES ~/rpmbuild/SPECS
        cp /out/sources/{name}-{version}.tar.gz ~/rpmbuild/SOURCES/
        cp /out/fedora/python-{name}.spec ~/rpmbuild/SPECS/
        dnf -q -y builddep ~/rpmbuild/SPECS/python-{name}.spec >/dev/null
        # the whole log kept, and its end shown when the build fails
        mkdir -p /out/built
        spec=~/rpmbuild/SPECS/python-{name}.spec log=/out/built/fedora-build.log
        rpmbuild -ba $spec > $log 2>&1 || {{ tail -60 $log; exit 1; }}
        grep -E "passed|failed" $log | tail -1
        cp ~/rpmbuild/SRPMS/*.rpm ~/rpmbuild/RPMS/*/*.rpm /out/built/
        # warnings fail too, all but the filters the rpmlintrc states
        filters=/out/fedora/python-{name}.rpmlintrc
        rpmlint --strict -r $filters $spec ~/rpmbuild/SRPMS/*.rpm ~/rpmbuild/RPMS/*/*.rpm
        """
    )


def write(sdist: Path, out: Path) -> None:
    """
    Both packages' files, from the sdist, into `out`: debian/ and fedora/, what a
    maintainer commits, and sources/, each one's tarball.
    """
    meta = project()
    name, version = meta["name"], meta["version"]
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with tarfile.open(sdist) as archive:
        top = {m.name.split("/", 1)[0] for m in archive.getmembers()}
    if top != {f"{name}-{version}"}:
        raise SystemExit(f"{sdist} doesn't unpack to {name}-{version}/, as both packagers expect")
    (out / "sources").mkdir()
    shutil.copy(sdist, out / "sources" / f"{name}_{version}.orig.tar.gz")
    shutil.copy(sdist, out / "sources" / sdist.name)
    debian(out)
    fedora(out / "fedora")
    print(f"wrote {out / 'debian'} and {out / 'fedora'}, what a maintainer commits")


def run(out: Path, docker_run: list[str], owner: str) -> None:
    """
    Build and lint both packages `write` made, each in its distribution's
    image on this machine's own architecture: any failure, any failing test,
    or any lint error or warning stops it.
    """
    meta = project()
    name, version = meta["name"], meta["version"]
    machine = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}[
        platform.machine().lower()
    ]
    for which, script in (
        ("debian", debian_script(name, version)),
        ("fedora", fedora_script(name, version)),
    ):
        image = distribution(which)["images"][machine]
        print(f"+ {which} ({machine}): build, test and lint", flush=True)
        # as root, as the package managers need, then the results handed back,
        # pass or fail, so a failed run leaves nothing only root can delete
        command = f"({script}); status=$?; chown -R {owner} /out; exit $status"
        command_line = [*docker_run, "-v", f"{out}:/out", image, "sh", "-c", command]
        subprocess.run(command_line, check=True)
    print(f"ok: Debian and Fedora packages built, tested and linted in {out}")
