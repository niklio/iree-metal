#!/bin/bash
# Records enough provenance to identify and reproduce a wheel bundle.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 WHEELHOUSE VERSION" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
wheelhouse="$1"
preview_version="$2"
python_bin="${IREE_METAL_PYTHON:-python3}"
python_version="$("${python_bin}" --version 2>&1)"
manifest="${wheelhouse}/iree-metal-preview-${preview_version}.manifest.txt"
checksums="${wheelhouse}/SHA256SUMS"

shopt -s nullglob
wheels=("${wheelhouse}"/*.whl)
if [[ ${#wheels[@]} -ne 2 ]]; then
  echo "error: expected exactly two wheels in ${wheelhouse}, found ${#wheels[@]}" >&2
  exit 2
fi

{
  echo "format-version: 1"
  echo "version: ${preview_version}"
  echo "iree-revision: $(git -C "${repo_root}" rev-parse HEAD)"
  git -C "${repo_root}" submodule status --recursive | sed 's/^/submodule: /'
  echo "built-at-utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -s) $(uname -r) $(uname -m)"
  echo "product: $(sw_vers -productName) $(sw_vers -productVersion)"
  echo "xcode: $(xcodebuild -version | tr '\n' ' ')"
  echo "python: ${python_version}"
  echo "deployment-target: ${MACOSX_DEPLOYMENT_TARGET:-13.0}"
  echo "jax-version: 0.6.1"
} > "${manifest}"

(
  cd "${wheelhouse}"
  shasum -a 256 ./*.whl ./*.manifest.txt ./THIRD_PARTY_LICENSES.txt > "${checksums}"
)
