#!/usr/bin/env python3
"""Validates and packages sanitized independent-verifier release evidence."""

from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import tarfile
import tempfile
from typing import Any


EVIDENCE_FILES = (
    "cases.jsonl",
    "evidence.sqlite3",
    "manifest.json",
    "report.md",
    "summary.json",
)
PRIVATE_PATTERNS = (
    re.compile(rb"/Users/[^/\x00]+/"),
    re.compile(rb"/home/[^/\x00]+/"),
    re.compile(rb"/private/tmp/[^/\x00]+/"),
)
SECRET_PATTERNS = (
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return value


def wheel_records(wheelhouse: Path) -> dict[str, str]:
    wheels = sorted(wheelhouse.glob("*.whl")) + sorted(
        (wheelhouse / "dependencies").glob("*.whl")
    )
    if len(wheels) != 8:
        raise SystemExit(f"expected two project and six dependency wheels, found {len(wheels)}")
    return {wheel.name: file_sha256(wheel) for wheel in wheels}


def validate_summary(summary: dict[str, Any], version: str) -> None:
    expected_candidate = f"iree-metal-preview-{version}"
    expected = {
        "candidate": expected_candidate,
        "suite": "mvp",
        "total_cases": 166,
        "cases_completed": 166,
        "passed": 166,
        "pass_rate": 1.0,
        "wrong_results": 0,
        "execution_failures": 0,
        "timeouts": 0,
        "harness_errors": 0,
        "performance_cases": 166,
        "jax_version": "0.6.1",
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"verifier summary does not pass the release gate: {mismatches}")
    if summary.get("status_counts") != {"PASS": 166}:
        raise SystemExit(f"unexpected verifier statuses: {summary.get('status_counts')}")
    if not isinstance(summary.get("perf_vs_jaxmetal"), (float, int)) or summary[
        "perf_vs_jaxmetal"
    ] <= 0:
        raise SystemExit("verifier summary is missing a positive jax-metal comparison")
    if summary.get("hardware", {}).get("model") != "Apple M4":
        raise SystemExit("the first preview must be physically verified on Apple M4")


def validate_manifest(
    manifest: dict[str, Any], summary: dict[str, Any], wheels: dict[str, str]
) -> None:
    if manifest.get("run_id") != summary.get("run_id"):
        raise SystemExit("summary and manifest run identities differ")
    candidate = manifest.get("candidate", {})
    if candidate.get("id") != summary.get("candidate"):
        raise SystemExit("summary and candidate identities differ")
    if candidate.get("platform") != "iree_metal":
        raise SystemExit("candidate did not select the iree_metal platform")
    if candidate.get("environment"):
        raise SystemExit(
            "candidate verifier manifest contains runtime overrides; release evidence "
            "must exercise packaged defaults"
        )

    artifacts = manifest.get("artifacts", [])
    observed: dict[str, str] = {}
    for artifact in artifacts:
        name = Path(artifact.get("path", "")).name
        digest = artifact.get("sha256")
        if not name or not digest or name in observed:
            raise SystemExit(f"malformed or duplicate verifier artifact: {artifact}")
        observed[name] = digest
    if observed != wheels:
        missing = sorted(set(wheels) - set(observed))
        extra = sorted(set(observed) - set(wheels))
        changed = sorted(
            name for name in set(wheels) & set(observed) if wheels[name] != observed[name]
        )
        raise SystemExit(
            "verifier artifacts are not the exact release wheelhouse: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )

    install_names = {Path(path).name for path in candidate.get("install_artifacts", [])}
    if install_names != set(wheels):
        raise SystemExit("candidate disposable environment did not install the full wheelhouse")


def validate_database(path: Path, run_id: str) -> None:
    uri = f"file:{path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as database:
        integrity = database.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise SystemExit(f"verifier database integrity check failed: {integrity}")
        tables = {
            row[0]
            for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "cases" not in tables:
            raise SystemExit("verifier database is missing its cases evidence table")
        count, passed = database.execute(
            "SELECT COUNT(*), SUM(status = 'PASS') FROM cases WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if count != 166 or passed != 166:
            raise SystemExit(
                "verifier database does not contain 166 passing records: "
                f"count={count}, passed={passed}"
            )


def validate_case_records(path: Path, run_id: str) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != 166:
        raise SystemExit(f"cases.jsonl contains {len(records)} records, expected 166")
    if any(record.get("run_id") != run_id for record in records):
        raise SystemExit("cases.jsonl contains a record from another verifier run")
    if any(record.get("status") != "PASS" for record in records):
        raise SystemExit("cases.jsonl contains a non-passing verifier record")


def sanitize_backend(backend: dict[str, Any] | None, label: str) -> None:
    if not backend:
        return
    backend["python"] = f"<sanitized:{label}-python>"
    for key in ("artifacts", "install_artifacts"):
        if key in backend:
            backend[key] = [Path(path).name for path in backend[key]]


def sanitized_documents(
    manifest: dict[str, Any], summary: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_manifest = deepcopy(manifest)
    clean_summary = deepcopy(summary)
    for artifact in clean_manifest.get("artifacts", []):
        artifact["path"] = Path(artifact["path"]).name
    sanitize_backend(clean_manifest.get("candidate"), "candidate")
    sanitize_backend(clean_manifest.get("oracle"), "oracle")
    sanitize_backend(clean_manifest.get("performance_reference"), "reference")
    clean_manifest.get("hardware", {}).pop("hostname", None)
    clean_summary.get("hardware", {}).pop("hostname", None)
    return clean_manifest, clean_summary


def audit_payload(path: Path) -> None:
    payload = path.read_bytes()
    for pattern in (*PRIVATE_PATTERNS, *SECRET_PATTERNS):
        if pattern.search(payload):
            raise SystemExit(f"private path or credential pattern remains in {path.name}")


def add_file(archive: tarfile.TarFile, source: Path, name: str, epoch: int) -> None:
    info = tarfile.TarInfo(name)
    info.size = source.stat().st_size
    info.mtime = epoch
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    with source.open("rb") as stream:
        archive.addfile(info, stream)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("version")
    parser.add_argument("--source-date-epoch", type=int)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    wheelhouse = args.wheelhouse.resolve()
    missing = [name for name in EVIDENCE_FILES if not (run_dir / name).is_file()]
    if missing:
        raise SystemExit(f"verifier run is missing evidence files: {missing}")

    summary = load_json(run_dir / "summary.json")
    manifest = load_json(run_dir / "manifest.json")
    wheels = wheel_records(wheelhouse)
    validate_summary(summary, args.version)
    validate_manifest(manifest, summary, wheels)
    validate_database(run_dir / "evidence.sqlite3", summary["run_id"])
    validate_case_records(run_dir / "cases.jsonl", summary["run_id"])
    clean_manifest, clean_summary = sanitized_documents(manifest, summary)

    epoch = args.source_date_epoch
    if epoch is None:
        epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    root_name = f"iree-metal-preview-{args.version}-verifier-evidence"
    output = wheelhouse / f"{root_name}-macos-arm64.tar.gz"
    sidecar = output.with_name(output.name + ".sha256")

    with tempfile.TemporaryDirectory(prefix="iree-metal-evidence-") as temporary:
        staging = Path(temporary)
        (staging / "manifest.json").write_text(
            json.dumps(clean_manifest, indent=2, sort_keys=True) + "\n"
        )
        (staging / "summary.json").write_text(
            json.dumps(clean_summary, indent=2, sort_keys=True) + "\n"
        )
        for name in ("cases.jsonl", "evidence.sqlite3", "report.md"):
            (staging / name).write_bytes((run_dir / name).read_bytes())
        readme = (
            "iree-metal independent verifier evidence\n\n"
            f"Run: {summary['run_id']}\n"
            f"Result: {summary['passed']}/{summary['cases_completed']} PASS\n"
            "The candidate was installed from the eight hashed wheels listed in "
            "manifest.json and used no candidate environment overrides.\n"
            "cases.jsonl contains case-level correctness and synchronized timing "
            "records; evidence.sqlite3 is the verifier's resumable evidence store.\n"
        )
        (staging / "README.txt").write_text(readme)
        payloads = sorted(staging.iterdir(), key=lambda path: path.name)
        for path in payloads:
            audit_payload(path)
        checksums = "".join(
            f"{file_sha256(path)}  {path.name}\n" for path in payloads
        )
        (staging / "SHA256SUMS").write_text(checksums)
        payloads.append(staging / "SHA256SUMS")

        with output.open("wb") as raw_stream:
            with gzip.GzipFile(fileobj=raw_stream, mode="wb", filename="", mtime=epoch) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    root = tarfile.TarInfo(root_name)
                    root.type = tarfile.DIRTYPE
                    root.mode = 0o755
                    root.mtime = epoch
                    root.uid = root.gid = 0
                    root.uname = root.gname = "root"
                    archive.addfile(root)
                    for path in sorted(payloads, key=lambda item: item.name):
                        add_file(archive, path, f"{root_name}/{path.name}", epoch)

    sidecar.write_text(f"{file_sha256(output)}  {output.name}\n")
    print(output)
    print(sidecar)


if __name__ == "__main__":
    main()
