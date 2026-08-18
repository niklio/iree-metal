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
if [[ "${preview_version}" =~ \.dev([0-9]{8})([0-9]{2})?$ ]]; then
  preview_profile="preview-${BASH_REMATCH[1]}"
else
  echo "error: version must end in .devYYYYMMDD or .devYYYYMMDDNN" >&2
  exit 2
fi
python_bin="${IREE_METAL_PYTHON:-python3}"
jax_version="${IREE_METAL_JAX_VERSION:-0.11.1}"
python_version="$("${python_bin}" --version 2>&1)"
manifest="${wheelhouse}/iree-metal-preview-${preview_version}.manifest.txt"
checksums="${wheelhouse}/SHA256SUMS"

write_submodule_revisions() {
  local superproject="$1"
  local path_prefix="$2"
  local modules_file="${superproject}/.gitmodules"
  [[ -f "${modules_file}" ]] || return 0
  git -C "${superproject}" config --file "${modules_file}" --get-regexp path |
    while read -r _ relative_path; do
      local revision
      revision="$(git -C "${superproject}" ls-tree HEAD -- "${relative_path}" | awk '{print $3}')"
      [[ -n "${revision}" ]] || {
        echo "error: cannot resolve submodule revision: ${path_prefix}${relative_path}" >&2
        return 1
      }
      echo "submodule: ${revision} ${path_prefix}${relative_path}"
      write_submodule_revisions "${superproject}/${relative_path}" \
        "${path_prefix}${relative_path}/"
    done
}

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
  write_submodule_revisions "${repo_root}" ""
  while IFS= read -r overlay; do
    [[ "${overlay}" == \#* || -z "${overlay}" ]] && continue
    echo "source-overlay: ${overlay}"
  done < "${repo_root}/submodule-patches/LOCK"
  echo "built-at-utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -s) $(uname -r) $(uname -m)"
  echo "product: $(sw_vers -productName) $(sw_vers -productVersion)"
  echo "xcode: $(xcodebuild -version | tr '\n' ' ')"
  echo "cmake: $(cmake --version | head -n 1)"
  echo "ninja: $(ninja --version)"
  echo "python: ${python_version}"
  echo "deployment-target: ${MACOSX_DEPLOYMENT_TARGET:-13.0}"
  echo "validated-host: Apple M4; product version above; CPython 3.12"
  echo "jax-version: ${jax_version}"
  echo "jax-compatibility-range: >=0.10.2,<0.12"
  echo "default-profile: ${preview_profile}"
  echo "rollback-profile: baseline"
  echo "compiler-options: --iree-metal-compile-to-metallib=false --iree-dispatch-creation-fuse-multi-use=true --iree-dispatch-creation-enable-aggressive-fusion=true --iree-dispatch-creation-enable-split-reduction=true"
  echo "numeric-contract: optimized causal attention assumes finite model inputs; use IREE_METAL_PROFILE=baseline for strict non-finite propagation diagnostics"
  echo "dependency-lock: requirements-macos-arm64-py312.txt"
  echo "build-dependency-lock: requirements-build-macos-arm64-py312.txt"
  echo "sbom: iree-metal-preview-${preview_version}.spdx.json"
} > "${manifest}"

(
  cd "${wheelhouse}"
  find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort | \
    while IFS= read -r asset; do shasum -a 256 "${asset}"; done > "${checksums}"
)
