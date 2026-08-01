#!/bin/bash
# Performs structural checks that do not require loading the plugin.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 WHEELHOUSE VERSION" >&2
  exit 2
fi

wheelhouse="$(cd "$1" && pwd)"
preview_version="$2"
python_bin="${IREE_METAL_PYTHON:-python3}"

"${python_bin}" - "${wheelhouse}" "${preview_version}" <<'PY'
from email.parser import Parser
from pathlib import Path
import re
import sys
import zipfile

wheelhouse = Path(sys.argv[1])
version = sys.argv[2]
wheels = sorted(wheelhouse.glob("*.whl"))
if len(wheels) != 2:
    raise SystemExit(f"expected two wheels, found {len(wheels)}")

expected = {
    "iree_base_compiler_iree_metal": "compiler",
    "iree_pjrt_plugin_metal": "plugin",
}
seen = set()
for wheel in wheels:
    kind = next((v for k, v in expected.items() if wheel.name.startswith(k + "-")), None)
    if not kind:
        raise SystemExit(f"unexpected wheel name: {wheel.name}")
    if "arm64" not in wheel.name:
        raise SystemExit(f"wheel is not tagged for arm64: {wheel.name}")
    if kind == "compiler" and "cp312-abi3" not in wheel.name:
        raise SystemExit(f"compiler wheel is not cp312-abi3: {wheel.name}")
    if kind == "plugin" and "py3-none" not in wheel.name:
        raise SystemExit(f"plugin wheel is not py3-none: {wheel.name}")
    seen.add(kind)

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        allowed_prefixes = (
            ("iree/",) if kind == "compiler" else ("iree/", "jax_plugins/")
        )
        unexpected = [
            name
            for name in names
            if ".dist-info/" not in name
            and not name.startswith(allowed_prefixes)
        ]
        if unexpected:
            raise SystemExit(
                f"unexpected top-level wheel contents in {wheel.name}: {unexpected[:10]}"
            )

        secret_patterns = (
            re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
            re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
            re.compile(rb"AKIA[0-9A-Z]{16}"),
            re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        )
        for name in names:
            payload = archive.read(name)
            if any(pattern.search(payload) for pattern in secret_patterns):
                raise SystemExit(f"possible credential material in {wheel.name}:{name}")

        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise SystemExit(f"expected one METADATA file in {wheel.name}")
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode())
        if metadata["Version"] != version:
            raise SystemExit(
                f"version mismatch in {wheel.name}: {metadata['Version']} != {version}"
            )

        if kind == "compiler":
            version_modules = [n for n in names if n == "iree/compiler/version.py"]
            if len(version_modules) != 1:
                raise SystemExit("compiler wheel is missing iree/compiler/version.py")
            version_module = archive.read(version_modules[0]).decode()
            if 'PACKAGE_SUFFIX = "-iree-metal"' not in version_module:
                raise SystemExit("compiler wheel is missing the iree-metal fork marker")

        if kind == "plugin":
            requirements = metadata.get_all("Requires-Dist", [])
            required_fragments = (
                f"iree-base-compiler-iree-metal=={version}",
                "jax==0.6.1",
                "jaxlib==0.6.1",
            )
            normalized = [r.replace(" ", "") for r in requirements]
            for fragment in required_fragments:
                if not any(fragment in r for r in normalized):
                    raise SystemExit(
                        f"plugin metadata is missing exact dependency {fragment}: {requirements}"
                    )
            native = [n for n in names if "pjrt_plugin_iree_metal" in n]
            if len(native) != 1:
                raise SystemExit(
                    f"expected one Metal PJRT native library, found {native}"
                )

if seen != {"compiler", "plugin"}:
    raise SystemExit(f"incomplete wheel set: {seen}")
print("wheel structure and metadata validation passed")
PY

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/iree-metal-wheel-audit.XXXXXX")"
cleanup() {
  rm -rf "${temporary_dir}"
}
trap cleanup EXIT

for wheel in "${wheelhouse}"/*.whl; do
  "${python_bin}" -m zipfile -e "${wheel}" "${temporary_dir}/$(basename "${wheel}")"
done

native_count=0
dependency_file="${temporary_dir}/otool-dependencies.txt"
while IFS= read -r native_library; do
  native_count=$((native_count + 1))
  echo "Inspecting native dependencies: ${native_library}"
  if ! otool -L "${native_library}" > "${dependency_file}"; then
    echo "error: otool could not inspect ${native_library}" >&2
    exit 2
  fi
  while IFS= read -r dependency; do
    case "${dependency}" in
      /System/Library/*|/usr/lib/*|@rpath/*|@loader_path/*|@executable_path/*) ;;
      *)
        echo "error: unexpected native dependency: ${dependency}" >&2
        exit 2
        ;;
    esac
  done < <(tail -n +2 "${dependency_file}" | awk '{print $1}')
done < <(find "${temporary_dir}" -type f \( -name '*.dylib' -o -name '*.so' \) -print)

if [[ "${native_count}" -eq 0 ]]; then
  echo "error: no native libraries found in preview wheels" >&2
  exit 2
fi
