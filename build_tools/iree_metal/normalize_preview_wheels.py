#!/usr/bin/env python3
"""Normalizes wheel member order and timestamps for reproducible archives."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import os
import tempfile
import zipfile


def normalize(wheel: Path, source_date_epoch: int) -> None:
    timestamp = datetime.fromtimestamp(source_date_epoch, timezone.utc)
    # ZIP cannot represent dates before 1980.
    zip_date = (
        max(timestamp.year, 1980),
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )
    with zipfile.ZipFile(wheel, "r") as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{wheel.name}.", suffix=".tmp", dir=wheel.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as destination:
            for old_info, payload in sorted(entries, key=lambda item: item[0].filename):
                info = zipfile.ZipInfo(old_info.filename, zip_date)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.comment = old_info.comment
                info.extra = b""
                info.create_system = old_info.create_system
                info.external_attr = old_info.external_attr
                destination.writestr(
                    info,
                    payload,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary, wheel)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--source-date-epoch", required=True, type=int)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    for wheel in args.wheels:
        normalize(wheel, args.source_date_epoch)
        print(f"normalized {wheel}")


if __name__ == "__main__":
    main()
