#!/usr/bin/env python3
"""Writes a compact SPDX 2.3 SBOM for preview and dependency wheels."""

from email.parser import BytesParser
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import zipfile
from datetime import datetime, timezone


def package_for_wheel(wheel: Path) -> dict:
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    name = metadata["Name"]
    version = metadata["Version"]
    spdx_id = "SPDXRef-Package-" + re.sub(r"[^A-Za-z0-9.-]", "-", name)
    return {
        "SPDXID": spdx_id,
        "name": name,
        "versionInfo": version,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "checksums": [
            {
                "algorithm": "SHA256",
                "checksumValue": sha256(wheel.read_bytes()).hexdigest(),
            }
        ],
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": metadata.get("License-Expression") or "NOASSERTION",
        "copyrightText": "NOASSERTION",
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": (
                    f"pkg:pypi/{name.lower().replace('_', '-')}@{version}"
                ),
            }
        ],
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} WHEELHOUSE VERSION")
    wheelhouse = Path(sys.argv[1]).resolve()
    version = sys.argv[2]
    wheels = sorted(wheelhouse.glob("*.whl")) + sorted(
        (wheelhouse / "dependencies").glob("*.whl")
    )
    packages = [package_for_wheel(wheel) for wheel in wheels]
    source_date_epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    created = datetime.fromtimestamp(source_date_epoch, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    document_id = "SPDXRef-DOCUMENT"
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": document_id,
        "name": f"iree-metal-preview-{version}",
        "documentNamespace": (
            "https://github.com/niklio/iree-metal/releases/tag/"
            f"iree-metal-v{version}/sbom"
        ),
        "creationInfo": {
            "created": created,
            "creators": ["Tool: iree-metal-write-preview-sbom/1"],
        },
        "packages": packages,
        "relationships": [
            {
                "spdxElementId": document_id,
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": package["SPDXID"],
            }
            for package in packages
        ],
    }
    output = wheelhouse / f"iree-metal-preview-{version}.spdx.json"
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
