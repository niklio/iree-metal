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
  echo "format-version: 2"
  echo "version: ${preview_version}"
  echo "tag: iree-metal-v${preview_version}"
  echo "iree-revision: $(git -C "${repo_root}" rev-parse HEAD)"
  git -C "${repo_root}" submodule status --recursive | sed 's/^/submodule: /'
  while IFS= read -r overlay; do
    [[ "${overlay}" == \#* || -z "${overlay}" ]] && continue
    echo "source-overlay: ${overlay}"
  done < "${repo_root}/submodule-patches/LOCK"
  echo "built-at-utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -s) $(uname -r) $(uname -m)"
  echo "product: $(sw_vers -productName) $(sw_vers -productVersion)"
  echo "xcode: $(xcodebuild -version | tr '\n' ' ')"
  echo "python: ${python_version}"
  echo "deployment-target: ${MACOSX_DEPLOYMENT_TARGET:-13.0}"
  echo "validated-host: Apple M4; product version above; CPython 3.12"
  echo "jax-version: 0.6.1"
  echo "default-profile: preview-20260802"
  echo "rollback-profile: baseline"
  echo "compiler-options: --iree-metal-compile-to-metallib=false --iree-dispatch-creation-fuse-multi-use=false --iree-dispatch-creation-enable-aggressive-fusion=true"
  echo "numeric-contract: optimized causal attention assumes finite model inputs; use IREE_METAL_PROFILE=baseline for strict non-finite propagation diagnostics"
  echo "dependency-lock: requirements-macos-arm64-py312.txt"
  echo "sbom: iree-metal-preview-${preview_version}.spdx.json"
} > "${manifest}"

(
  cd "${wheelhouse}"
  find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort | \
    while IFS= read -r asset; do shasum -a 256 "${asset}"; done > "${checksums}"
)
