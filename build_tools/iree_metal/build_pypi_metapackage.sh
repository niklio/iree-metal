#!/bin/bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
version="${1:?usage: $0 VERSION RELEASE_WHEEL_DIR OUTPUT_DIR}"
release_wheel_dir="$(cd "${2:?usage: $0 VERSION RELEASE_WHEEL_DIR OUTPUT_DIR}" && pwd)"
output_dir="${3:?usage: $0 VERSION RELEASE_WHEEL_DIR OUTPUT_DIR}"
python_bin="${IREE_METAL_PYTHON:-python3.12}"
package_dir="${repo_root}/integrations/pjrt/python_packages/iree_metal_metapackage"

if ! "${python_bin}" -c 'import re, sys; raise SystemExit(not re.fullmatch(r"3\.11\.0\.dev[0-9]{8}(?:[0-9]{2})?", sys.argv[1]))' "${version}"; then
  echo "error: invalid iree-metal preview version: ${version}" >&2
  exit 2
fi

compiler="${release_wheel_dir}/iree_base_compiler_iree_metal-${version}-cp312-abi3-macosx_13_0_arm64.whl"
plugin="${release_wheel_dir}/iree_pjrt_plugin_metal_iree_metal-${version}-py3-none-macosx_13_0_arm64.whl"
for wheel in "${compiler}" "${plugin}"; do
  if [[ ! -f "${wheel}" ]]; then
    echo "error: missing exact release wheel: ${wheel}" >&2
    exit 2
  fi
done

mkdir -p "${output_dir}"
if compgen -G "${output_dir}/*" >/dev/null; then
  echo "error: output directory must be empty: ${output_dir}" >&2
  exit 2
fi

cp "${compiler}" "${plugin}" "${output_dir}/"
IREE_METAL_VERSION="${version}" "${python_bin}" -m pip wheel \
  --disable-pip-version-check --no-deps --wheel-dir "${output_dir}" "${package_dir}"

IREE_METAL_PYPI_DIR="${output_dir}" IREE_METAL_VERSION="${version}" \
  "${python_bin}" - <<'PY'
import email
import os
import pathlib
import zipfile

root = pathlib.Path(os.environ["IREE_METAL_PYPI_DIR"])
version = os.environ["IREE_METAL_VERSION"]
wheels = sorted(root.glob("*.whl"))
if len(wheels) != 3:
    raise SystemExit(f"expected exactly three PyPI wheels, found {len(wheels)}")

metadata = {}
for wheel in wheels:
    with zipfile.ZipFile(wheel) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        message = email.message_from_bytes(archive.read(member))
    metadata[message["Name"]] = (message, wheel)

expected = {
    "iree-base-compiler-iree-metal",
    "iree-pjrt-plugin-metal-iree-metal",
    "iree-metal",
}
if set(metadata) != expected:
    raise SystemExit(f"unexpected distributions: {sorted(metadata)}")
for name, (message, wheel) in metadata.items():
    if message["Version"] != version:
        raise SystemExit(f"{name} version mismatch: {message['Version']} != {version}")
    if "macosx_13_0_arm64" not in wheel.name:
        raise SystemExit(f"wheel is not platform constrained: {wheel.name}")

requirements = set(metadata["iree-metal"][0].get_all("Requires-Dist", []))
for dependency in (
    f"iree-base-compiler-iree-metal=={version}",
    f"iree-pjrt-plugin-metal-iree-metal=={version}",
):
    if dependency not in requirements:
        raise SystemExit(f"iree-metal is missing exact dependency: {dependency}")

description = metadata["iree-metal"][0].get_payload()
if "not an official release" not in description:
    raise SystemExit("iree-metal README is missing the unofficial-release disclosure")
print("validated PyPI distributions:", ", ".join(sorted(metadata)))
PY
