#!/bin/bash
# Collects source license texts into a release-sidecar inventory.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 WHEELHOUSE" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
wheelhouse="$(cd "$1" && pwd)"
output="${wheelhouse}/THIRD_PARTY_LICENSES.txt"
paths_file="$(mktemp "${TMPDIR:-/tmp}/iree-metal-licenses.XXXXXX")"

cleanup() {
  rm -f "${paths_file}"
}
trap cleanup EXIT

find "${repo_root}" \
  -type f \
  \( -iname 'LICENSE' -o -iname 'LICENSE.*' -o -iname 'COPYING' \
     -o -iname 'COPYING.*' -o -iname 'NOTICE' -o -iname 'NOTICE.*' \) \
  -not -path '*/.git/*' \
  -not -path '*/build/*' \
  -not -path '*/wheelhouse/*' \
  -print | LC_ALL=C sort > "${paths_file}"

license_count="$(wc -l < "${paths_file}" | tr -d ' ')"
if [[ "${license_count}" -lt 5 ]]; then
  echo "error: only ${license_count} license files found; are submodules initialized?" >&2
  exit 2
fi

{
  echo "iree-metal developer preview license inventory"
  echo "Generated from source revision $(git -C "${repo_root}" rev-parse HEAD)"
  echo
  while IFS= read -r license_file; do
    relative_path="${license_file#${repo_root}/}"
    echo "============================================================================="
    echo "FILE: ${relative_path}"
    echo "============================================================================="
    command cat "${license_file}"
    echo
  done < "${paths_file}"
} > "${output}"

echo "Collected ${license_count} license files into ${output}"
