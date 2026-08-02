#!/usr/bin/env python3
"""Creates the deterministic offline developer-preview release bundle."""

from argparse import ArgumentParser
import gzip
from hashlib import sha256
import os
from pathlib import Path
import tarfile


REQUIRED_PATTERNS = (
    "iree_base_compiler_iree_metal-*.whl",
    "iree_pjrt_plugin_metal_iree_metal-*.whl",
    "SHA256SUMS",
    "THIRD_PARTY_LICENSES.txt",
    "INSTALL.md",
    "RELEASE_NOTES.md",
    "requirements-macos-arm64-py312.txt",
    "*.manifest.txt",
    "*.spdx.json",
)


def one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected one {pattern!r} in {root}, found {len(matches)}")
    return matches[0]


def add_file(archive: tarfile.TarFile, source: Path, arcname: str, epoch: int) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = source.stat().st_size
    info.mtime = epoch
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    with source.open("rb") as stream:
        archive.addfile(info, stream)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("version")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-date-epoch", type=int)
    args = parser.parse_args()

    wheelhouse = args.wheelhouse.resolve()
    output_dir = (args.output_dir or wheelhouse).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch = args.source_date_epoch
    if epoch is None:
        epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))

    files = [one_match(wheelhouse, pattern) for pattern in REQUIRED_PATTERNS]
    dependencies = sorted((wheelhouse / "dependencies").glob("*.whl"))
    if len(dependencies) != 6:
        raise SystemExit(f"expected six dependency wheels, found {len(dependencies)}")
    files.extend(dependencies)
    files.extend(sorted(wheelhouse.glob("*verifier-evidence*.tar.gz")))

    bundle_root = f"iree-metal-preview-{args.version}"
    output = output_dir / f"{bundle_root}-macos-arm64.tar.gz"
    with output.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", filename="", mtime=epoch) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
                root_info = tarfile.TarInfo(bundle_root)
                root_info.type = tarfile.DIRTYPE
                root_info.mode = 0o755
                root_info.mtime = epoch
                root_info.uid = root_info.gid = 0
                root_info.uname = root_info.gname = "root"
                archive.addfile(root_info)
                for source in sorted(files, key=lambda path: path.name):
                    if source.parent.name == "dependencies":
                        relative = Path("dependencies") / source.name
                    else:
                        relative = Path(source.name)
                    add_file(archive, source, f"{bundle_root}/{relative}", epoch)

    digest = sha256(output.read_bytes()).hexdigest()
    sidecar = output.with_name(output.name + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n")
    print(output)
    print(sidecar)


if __name__ == "__main__":
    main()
