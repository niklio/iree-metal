#!/usr/bin/env python3
"""Validate and package sanitized iree-metal-verifier release evidence."""

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


COMBINED_FILES = ("manifest.json", "report.md", "summary.json")
RUN_FILES = (
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


def require_files(root: Path, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"{label} is missing evidence files: {missing}")


def wheel_records(wheelhouse: Path) -> dict[str, str]:
    wheels = sorted(wheelhouse.glob("*.whl")) + sorted(
        (wheelhouse / "dependencies").glob("*.whl")
    )
    if len(wheels) != 10:
        raise SystemExit(
            f"expected two project and eight dependency wheels, found {len(wheels)}"
        )
    return {wheel.name: file_sha256(wheel) for wheel in wheels}


def require_values(
    document: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    mismatches = {
        key: (document.get(key), value)
        for key, value in expected.items()
        if document.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"{label} does not pass the release gate: {mismatches}")


def require_parity(summary: dict[str, Any], label: str) -> None:
    parity = summary.get("perf_vs_jaxmetal")
    if isinstance(parity, bool) or not isinstance(parity, (float, int)) or parity <= 1.0:
        raise SystemExit(
            f"{label} geometric-mean parity must be strictly above 1.00; got {parity!r}"
        )


def validate_hardware(summary: dict[str, Any], label: str) -> None:
    if summary.get("hardware", {}).get("model") != "Apple M4":
        raise SystemExit(f"{label} must be physically verified on Apple M4")
    if summary.get("jax_version") != "0.6.1":
        raise SystemExit(f"{label} must use JAX 0.6.1")


def validate_combined_summary(summary: dict[str, Any], version: str) -> None:
    require_values(
        summary,
        {
            "candidate": f"iree-metal-preview-{version}",
            "suite": "candidate",
            "total_cases": 231,
            "cases_completed": 231,
            "passed": 231,
            "pass_rate": 1.0,
            "wrong_results": 0,
            "execution_failures": 0,
            "resource_failures": 0,
            "nondeterministic_failures": 0,
            "critical_failures": 0,
            "timeouts": 0,
            "performance_cases": 10,
            "models_passed": 10,
            "models_total": 10,
            "semantic_gate_passed": True,
            "application_gate_passed": True,
            "gate_passed": True,
            "release_ready": True,
        },
        "combined verifier summary",
    )
    validate_hardware(summary, "combined verifier summary")
    require_parity(summary, "combined verifier summary")
    if summary.get("model_mean_vs_jaxmetal") != summary.get("perf_vs_jaxmetal"):
        raise SystemExit("combined verifier summary contains inconsistent model parity")


def validate_semantic_summary(
    summary: dict[str, Any], candidate: str, run_id: str
) -> None:
    require_values(
        summary,
        {
            "run_id": run_id,
            "candidate": candidate,
            "suite": "release",
            "total_cases": 221,
            "cases_completed": 221,
            "passed": 221,
            "pass_rate": 1.0,
            "wrong_results": 0,
            "compile_failures": 0,
            "runtime_failures": 0,
            "execution_failures": 0,
            "resource_failures": 0,
            "nondeterministic_failures": 0,
            "timeouts": 0,
            "harness_errors": 0,
            "critical_failures": 0,
            "performance_cases": 0,
            "gate_passed": True,
            "release_ready": True,
            "status_counts": {"PASS": 221},
            "tiers": {
                "A": {"cases": 186, "passed": 186, "pass_rate": 1.0},
                "B": {"cases": 35, "passed": 35, "pass_rate": 1.0},
            },
        },
        "semantic verifier summary",
    )
    validate_hardware(summary, "semantic verifier summary")


def validate_model_summary(
    summary: dict[str, Any], candidate: str, run_id: str
) -> None:
    require_values(
        summary,
        {
            "run_id": run_id,
            "candidate": candidate,
            "suite": "models",
            "total_cases": 10,
            "cases_completed": 10,
            "passed": 10,
            "pass_rate": 1.0,
            "wrong_results": 0,
            "compile_failures": 0,
            "runtime_failures": 0,
            "execution_failures": 0,
            "resource_failures": 0,
            "nondeterministic_failures": 0,
            "timeouts": 0,
            "harness_errors": 0,
            "critical_failures": 0,
            "performance_cases": 10,
            "models_passed": 10,
            "models_total": 10,
            "gate_passed": True,
            "status_counts": {"PASS": 10},
        },
        "model verifier summary",
    )
    validate_hardware(summary, "model verifier summary")
    require_parity(summary, "model verifier summary")


def artifact_records(manifest: dict[str, Any], label: str) -> dict[str, str]:
    observed: dict[str, str] = {}
    for artifact in manifest.get("artifacts", []):
        name = Path(artifact.get("path", "")).name
        digest = artifact.get("sha256")
        if not name or not digest or name in observed:
            raise SystemExit(f"malformed or duplicate {label} artifact: {artifact}")
        observed[name] = digest
    return observed


def validate_candidate(
    manifest: dict[str, Any], candidate_id: str, wheels: dict[str, str], label: str
) -> None:
    candidate = manifest.get("candidate", {})
    if candidate.get("id") != candidate_id:
        raise SystemExit(f"{label} candidate identity differs from the combined run")
    if candidate.get("platform") != "iree_metal":
        raise SystemExit(f"{label} did not select the iree_metal platform")
    if candidate.get("environment"):
        raise SystemExit(
            f"{label} contains runtime overrides; release evidence must exercise "
            "packaged defaults"
        )
    install_names = {Path(path).name for path in candidate.get("install_artifacts", [])}
    if install_names != set(wheels):
        raise SystemExit(f"{label} did not install the complete release wheelhouse")


def validate_run_manifest(
    manifest: dict[str, Any], summary: dict[str, Any], wheels: dict[str, str], label: str
) -> None:
    if manifest.get("run_id") != summary.get("run_id"):
        raise SystemExit(f"{label} manifest and summary identities differ")
    if manifest.get("suite") != summary.get("suite"):
        raise SystemExit(f"{label} manifest and summary suites differ")
    validate_candidate(manifest, summary["candidate"], wheels, label)
    observed = artifact_records(manifest, label)
    if observed != wheels:
        missing = sorted(set(wheels) - set(observed))
        extra = sorted(set(observed) - set(wheels))
        changed = sorted(
            name for name in set(wheels) & set(observed) if wheels[name] != observed[name]
        )
        raise SystemExit(
            f"{label} artifacts are not the exact release wheelhouse: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def validate_model_protocol(manifest: dict[str, Any]) -> None:
    require_values(
        manifest,
        {
            "steps": 16,
            "warmups": 3,
            "suite_settle_seconds": 300,
        },
        "model verifier protocol",
    )
    verifier = manifest.get("verifier")
    if not isinstance(verifier, list) or not verifier:
        raise SystemExit("model verifier manifest does not identify the verifier source")
    for artifact in verifier:
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise SystemExit("model verifier manifest contains malformed verifier source")
        if not artifact["path"] or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
            raise SystemExit("model verifier manifest contains malformed verifier source")


def validate_database(path: Path, run_id: str, expected: int, label: str) -> None:
    uri = f"file:{path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as database:
        integrity = database.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise SystemExit(f"{label} database integrity check failed: {integrity}")
        tables = {
            row[0]
            for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "cases" not in tables:
            raise SystemExit(f"{label} database is missing its cases table")
        count, passed = database.execute(
            "SELECT COUNT(*), SUM(status = 'PASS') FROM cases WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if count != expected or passed != expected:
            raise SystemExit(
                f"{label} database does not contain {expected} passing records: "
                f"count={count}, passed={passed}"
            )


def validate_case_records(
    path: Path, run_id: str, expected: int, label: str
) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != expected:
        raise SystemExit(f"{label} cases.jsonl has {len(records)} records, expected {expected}")
    if any(record.get("run_id") != run_id for record in records):
        raise SystemExit(f"{label} cases.jsonl contains records from another run")
    if any(record.get("status") != "PASS" for record in records):
        raise SystemExit(f"{label} cases.jsonl contains a non-passing record")


def validate_model_record_protocol(path: Path) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    mismatched = [
        record.get("model", record.get("case_id", "<unknown>"))
        for record in records
        if record.get("metadata", {}).get("steps") != 16
        or record.get("metadata", {}).get("warmups") != 3
    ]
    if mismatched:
        raise SystemExit(
            "model verifier records do not use the release timing protocol: "
            f"{mismatched}"
        )


def resolve_evidence_dir(raw: Any, combined_dir: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SystemExit(f"combined manifest is missing {label} evidence")
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        combined_dir.parent.parent / path,
        combined_dir / path,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise SystemExit(f"could not resolve {label} evidence directory: {raw}")


def sanitize_backend(backend: dict[str, Any] | None, label: str) -> None:
    if not backend:
        return
    backend["python"] = f"<sanitized:{label}-python>"
    for key in ("artifacts", "install_artifacts"):
        if key in backend:
            backend[key] = [Path(path).name for path in backend[key]]


def sanitized_manifest(manifest: dict[str, Any], kind: str) -> dict[str, Any]:
    clean = deepcopy(manifest)
    for artifact in clean.get("artifacts", []):
        artifact["path"] = Path(artifact["path"]).name
    sanitize_backend(clean.get("candidate"), "candidate")
    sanitize_backend(clean.get("oracle"), "oracle")
    sanitize_backend(clean.get("performance_reference"), "reference")
    clean.get("hardware", {}).pop("hostname", None)
    for key in ("baseline", "reference"):
        if isinstance(clean.get(key), dict) and clean[key].get("path"):
            clean[key]["path"] = Path(clean[key]["path"]).name
    if kind == "combined":
        clean["semantic_evidence"] = "semantic"
        clean["application_evidence"] = "models"
    return clean


def sanitized_summary(summary: dict[str, Any]) -> dict[str, Any]:
    clean = deepcopy(summary)
    clean.get("hardware", {}).pop("hostname", None)
    return clean


def audit_payload(path: Path) -> None:
    payload = path.read_bytes()
    for pattern in (*PRIVATE_PATTERNS, *SECRET_PATTERNS):
        if pattern.search(payload):
            raise SystemExit(f"private path or credential pattern remains in {path}")


def add_file(archive: tarfile.TarFile, source: Path, name: str, epoch: int) -> None:
    info = tarfile.TarInfo(name)
    info.size = source.stat().st_size
    info.mtime = epoch
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    with source.open("rb") as stream:
        archive.addfile(info, stream)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="combined candidate evidence directory")
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("version")
    parser.add_argument("--source-date-epoch", type=int)
    args = parser.parse_args()

    combined_dir = args.run_dir.resolve()
    wheelhouse = args.wheelhouse.resolve()
    require_files(combined_dir, COMBINED_FILES, "combined verifier run")
    combined_manifest = load_json(combined_dir / "manifest.json")
    combined_summary = load_json(combined_dir / "summary.json")
    validate_combined_summary(combined_summary, args.version)

    semantic_dir = resolve_evidence_dir(
        combined_manifest.get("semantic_evidence"), combined_dir, "semantic"
    )
    model_dir = resolve_evidence_dir(
        combined_manifest.get("application_evidence"), combined_dir, "application"
    )
    require_files(semantic_dir, RUN_FILES, "semantic verifier run")
    require_files(model_dir, RUN_FILES, "model verifier run")

    semantic_manifest = load_json(semantic_dir / "manifest.json")
    semantic_summary = load_json(semantic_dir / "summary.json")
    model_manifest = load_json(model_dir / "manifest.json")
    model_summary = load_json(model_dir / "summary.json")
    candidate_id = combined_summary["candidate"]
    validate_semantic_summary(
        semantic_summary, candidate_id, combined_summary["semantic_run_id"]
    )
    validate_model_summary(
        model_summary, candidate_id, combined_summary["application_run_id"]
    )

    if combined_manifest.get("run_id") != combined_summary.get("run_id"):
        raise SystemExit("combined manifest and summary identities differ")
    if combined_manifest.get("suite") != "candidate":
        raise SystemExit("combined manifest is not a candidate-suite run")
    wheels = wheel_records(wheelhouse)
    validate_candidate(combined_manifest, candidate_id, wheels, "combined run")
    validate_run_manifest(
        semantic_manifest, semantic_summary, wheels, "semantic run"
    )
    validate_run_manifest(model_manifest, model_summary, wheels, "model run")
    validate_model_protocol(model_manifest)
    validate_database(
        semantic_dir / "evidence.sqlite3", semantic_summary["run_id"], 221, "semantic"
    )
    validate_database(
        model_dir / "evidence.sqlite3", model_summary["run_id"], 10, "model"
    )
    validate_case_records(
        semantic_dir / "cases.jsonl", semantic_summary["run_id"], 221, "semantic"
    )
    validate_case_records(
        model_dir / "cases.jsonl", model_summary["run_id"], 10, "model"
    )
    validate_model_record_protocol(model_dir / "cases.jsonl")

    epoch = args.source_date_epoch
    if epoch is None:
        epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    root_name = f"iree-metal-preview-{args.version}-verifier-evidence"
    output = wheelhouse / f"{root_name}-macos-arm64.tar.gz"
    sidecar = output.with_name(output.name + ".sha256")

    with tempfile.TemporaryDirectory(prefix="iree-metal-evidence-") as temporary:
        staging = Path(temporary)
        for name in ("combined", "semantic", "models"):
            (staging / name).mkdir()

        write_json(
            staging / "combined" / "manifest.json",
            sanitized_manifest(combined_manifest, "combined"),
        )
        write_json(
            staging / "combined" / "summary.json",
            sanitized_summary(combined_summary),
        )
        (staging / "combined" / "report.md").write_text(
            "# Combined candidate verification report\n\n"
            f"- Candidate: `{candidate_id}`\n"
            "- Semantic gate: **PASS** (221/221)\n"
            "- Application gate: **PASS** (10/10)\n"
            f"- Model geometric mean: **{combined_summary['perf_vs_jaxmetal']:.6f}x "
            "vs jax-metal**\n"
            "- Release ready: **YES**\n"
            "- Semantic evidence: `../semantic`\n"
            "- Application evidence: `../models`\n"
        )

        for name, source, manifest, summary in (
            ("semantic", semantic_dir, semantic_manifest, semantic_summary),
            ("models", model_dir, model_manifest, model_summary),
        ):
            write_json(
                staging / name / "manifest.json",
                sanitized_manifest(manifest, name),
            )
            write_json(
                staging / name / "summary.json", sanitized_summary(summary)
            )
            for filename in ("cases.jsonl", "evidence.sqlite3", "report.md"):
                (staging / name / filename).write_bytes((source / filename).read_bytes())

        (staging / "README.txt").write_text(
            "iree-metal independent verifier evidence\n\n"
            f"Combined run: {combined_summary['run_id']}\n"
            "Semantic result: 221/221 PASS\n"
            "Application result: 10/10 PASS\n"
            f"Model geometric mean: {combined_summary['perf_vs_jaxmetal']:.6f}x "
            "versus the frozen jax-metal baseline\n"
            "The candidate was installed from the ten hashed wheels listed in each "
            "manifest and used no candidate environment overrides. The semantic and "
            "models directories contain case-level JSONL records, reports, summaries, "
            "and integrity-checked SQLite evidence stores.\n"
        )

        payloads = sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: str(path.relative_to(staging)),
        )
        for path in payloads:
            audit_payload(path)
        (staging / "SHA256SUMS").write_text(
            "".join(
                f"{file_sha256(path)}  {path.relative_to(staging)}\n"
                for path in payloads
            )
        )
        payloads.append(staging / "SHA256SUMS")

        with output.open("wb") as raw_stream:
            with gzip.GzipFile(
                fileobj=raw_stream, mode="wb", filename="", mtime=epoch
            ) as gz:
                with tarfile.open(
                    fileobj=gz, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    root = tarfile.TarInfo(root_name)
                    root.type = tarfile.DIRTYPE
                    root.mode = 0o755
                    root.mtime = epoch
                    root.uid = root.gid = 0
                    root.uname = root.gname = "root"
                    archive.addfile(root)
                    for path in sorted(
                        payloads, key=lambda item: str(item.relative_to(staging))
                    ):
                        add_file(
                            archive,
                            path,
                            f"{root_name}/{path.relative_to(staging)}",
                            epoch,
                        )

    sidecar.write_text(f"{file_sha256(output)}  {output.name}\n")
    print(output)
    print(sidecar)


if __name__ == "__main__":
    main()
