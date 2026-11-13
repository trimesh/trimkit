"""
License and SBOM checks, all derived from trimkit.toml's upstreams.
`sbom_text` is the wheel's PEP 770 SBOM, which the backend writes into each
wheel, and `check` (`uv run scripts static`) verifies that:

- every license is permissive (the ALLOWED list): the package stays ~MIT
- every vendored upstream ships its license file in the wheel

Standard library only, and nothing read from the environment.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

from builder import ROOT, Upstream, load, pyproject

VENDOR = ROOT / "vendor"
# the SBOM's creation time: fixed, as every wheel entry's time is, so the
# same manifest gives the same bytes
CREATED = "1980-01-01T00:00:00Z"
OWNER = "Michael Dawson-Haggerty"
# the license of our own code, the repository's LICENSE
OWN = "MIT"
LICENSE_PATTERNS = ("LICENSE*", "COPYING*")

# permissive licenses only: everything here is compatible with shipping the
# whole wheel under MIT terms plus the upstream notices
ALLOWED = {"MIT", "ISC", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "BSL-1.0", "Zlib"}
# file-level copyleft, allowed only unmodified: MPL-2.0 binds the files
# themselves, which the sdist ships as source, and nothing else in the wheel
# (its "Larger Work"). A patch would make a modified MPL file of ours
UNMODIFIED_ONLY = {"MPL-2.0"}


def upstreams() -> dict[str, Upstream]:
    return load().upstreams


def expression() -> str:
    """
    The license of what a wheel contains, for pyproject.toml's `license`: our
    MIT, and every upstream compiled in, each one's choices in parentheses.
    """
    others = sorted({u.spdx for u in upstreams().values()} - {OWN})
    return " AND ".join([OWN, *(f"({e})" if " OR " in e else e for e in others)])


def license_files_of(name: str) -> list[Path]:
    # an upstream's license texts, as they are named on disk
    return sorted(p for pattern in LICENSE_PATTERNS for p in (VENDOR / name).glob(pattern))


def spdx_ids(expression: str) -> list[str]:
    # the expressions trimkit.toml uses are single ids or `A OR B OR C`
    return [part.strip() for part in expression.split(" OR ")]


def copyright_of(pin: Upstream) -> str:
    if pin.copyright:
        return pin.copyright
    # from disk, as the backend builds from an sdist without git
    for path in license_files_of(pin.name):
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if re.match(r"Copyright (\(c\)|\d)", line, re.IGNORECASE):
                return line.removesuffix(", All rights reserved.").removesuffix(".")
    raise ValueError(f"{pin.name}: no copyright line in its LICENSE: set `copyright` in trimkit.toml")


def sbom(created: str) -> dict:
    meta = pyproject()["project"]
    pins = upstreams()
    root_id = "SPDXRef-Package-trimkit"
    packages = [
        {
            "SPDXID": root_id,
            "name": meta["name"],
            "versionInfo": meta["version"],
            "downloadLocation": f"https://pypi.org/project/{meta['name']}/{meta['version']}/",
            "licenseConcluded": meta["license"],
            "licenseDeclared": meta["license"],
            "copyrightText": f"Copyright (c) 2026 {OWNER}",
            "filesAnalyzed": False,
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{meta['name']}@{meta['version']}",
                }
            ],
        }
    ]
    relationships = [
        {"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES", "relatedSpdxElement": root_id}
    ]
    for name, pin in pins.items():
        package = {
            "SPDXID": f"SPDXRef-Package-{name.replace('_', '-')}",
            "name": name,
            "licenseConcluded": pin.spdx,
            "licenseDeclared": pin.spdx,
            "copyrightText": copyright_of(pin),
            "filesAnalyzed": False,
        }
        package["versionInfo"] = pin.commit
        package["downloadLocation"] = f"git+https://{pin.host}/{pin.repo}@{pin.commit}"
        package["externalRefs"] = [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f"pkg:{pin.host.split('.')[0]}/{pin.repo.lower()}@{pin.commit}",
            }
        ]
        packages.append(package)
        # every vendored library is compiled statically into the extension
        contains = {"spdxElementId": root_id, "relationshipType": "CONTAINS"}
        relationships.append({**contains, "relatedSpdxElement": package["SPDXID"]})
    # the namespace is unique per distinct content, as SPDX asks
    digest = hashlib.sha256(json.dumps([packages, relationships], sort_keys=True).encode()).hexdigest()[:16]
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{meta['name']}-{meta['version']}",
        "documentNamespace": f"https://github.com/trimesh/trimkit/spdx/{meta['version']}-{digest}",
        "creationInfo": {"created": created, "creators": ["Tool: scripts/licenses.py", f"Person: {OWNER}"]},
        "packages": packages,
        "relationships": relationships,
    }


def sbom_text() -> str:
    return json.dumps(sbom(CREATED), indent=2) + "\n"


def check() -> None:
    errors = []
    meta = pyproject()
    if meta["project"]["license"] != expression():
        errors.append(f"pyproject.toml's license must be what a wheel contains: {expression()}")
    for name, pin in upstreams().items():
        ids = spdx_ids(pin.spdx)
        bad = [i for i in ids if i not in ALLOWED | UNMODIFIED_ONLY]
        # an OR expression needs only one permissive choice, a single id must be one
        if len(bad) == len(ids):
            errors.append(f"{name}: {pin.spdx!r} is not one of the permissive {sorted(ALLOWED)}")
        elif not any(i in ALLOWED for i in ids) and pin.patches:
            errors.append(f"{name}: {pin.spdx} is allowed only unmodified, and it has patches")
        if not license_files_of(name):
            errors.append(f"{name}: no license file in vendor/{name} to ship in the wheel")
    if errors:
        sys.exit("\n".join(errors))
    sbom_text()  # every upstream has a copyright line for it
    print(f"ok: {len(upstreams())} components, all permissive")
