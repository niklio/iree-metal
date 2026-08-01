#!/bin/bash
# Builds the two macOS arm64 wheels needed by the iree-metal developer preview.

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
wheelhouse="${IREE_METAL_WHEELHOUSE:-${repo_root}/wheelhouse}"
python_bin="${IREE_METAL_PYTHON:-python3}"
preview_version="${IREE_METAL_VERSION:-}"
jax_version="0.6.1"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "error: preview wheels must be built natively on macOS arm64" >&2
  exit 2
fi

if [[ -z "${preview_version}" ]]; then
  echo "error: set IREE_METAL_VERSION to a PEP 440 version (for example 3.11.0.dev20260801)" >&2
  exit 2
fi

if ! "${python_bin}" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo "error: the first preview is built and tested with Python 3.12 exactly" >&2
  exit 2
fi

if ! "${python_bin}" -c 'import sys; from packaging.version import Version, InvalidVersion; Version(sys.argv[1])' "${preview_version}"; then
  echo "error: IREE_METAL_VERSION is not a valid PEP 440 version" >&2
  exit 2
fi

if git -C "${repo_root}" submodule status --recursive | grep -q '^[+-U]'; then
  echo "error: submodules are uninitialized or do not match the recorded revisions" >&2
  exit 2
fi

if [[ "${IREE_METAL_ALLOW_DIRTY:-0}" != "1" ]] && \
   [[ -n "$(git -C "${repo_root}" status --porcelain --untracked-files=no)" ]]; then
  echo "error: tracked source changes are present; release builds require a clean checkout" >&2
  exit 2
fi

compiler_version_file="${repo_root}/compiler/version_local.json"
pjrt_version_file="${repo_root}/integrations/pjrt/version_info.json"

if [[ -e "${compiler_version_file}" || -e "${pjrt_version_file}" ]]; then
  echo "error: local version metadata already exists; refusing to overwrite it" >&2
  exit 2
fi

cleanup() {
  rm -f "${compiler_version_file}" "${pjrt_version_file}"
}
trap cleanup EXIT

mkdir -p "${wheelhouse}"
if compgen -G "${wheelhouse}/*.whl" > /dev/null; then
  echo "error: ${wheelhouse} already contains wheels; use an empty output directory" >&2
  exit 2
fi

printf '{\n  "package-version": "%s",\n  "package-suffix": "-iree-metal"\n}\n' \
  "${preview_version}" > "${compiler_version_file}"
printf '{\n  "package-version": "%s",\n  "compiler-package-name": "iree-base-compiler-iree-metal",\n  "compiler-package-version": "%s",\n  "jax-version": "%s",\n  "jaxlib-version": "%s",\n  "python-requires": ">=3.12,<3.13",\n  "project-url": "https://github.com/niklio/iree-metal"\n}\n' \
  "${preview_version}" "${preview_version}" "${jax_version}" "${jax_version}" \
  > "${pjrt_version_file}"

export CMAKE_OSX_ARCHITECTURES=arm64
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
export IREE_CMAKE_BUILD_TYPE=Release
export LLVM_PARALLEL_LINK_JOBS="${LLVM_PARALLEL_LINK_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-3}"

echo "Building iree-base-compiler-iree-metal ${preview_version}"
IREE_COMPILER_CUSTOM_DESCRIPTION="IREE compiler for the iree-metal developer preview" \
IREE_COMPILER_CUSTOM_HOMEPAGE_URL="https://github.com/niklio/iree-metal" \
IREE_COMPILER_CUSTOM_REPOSITORY_URL="https://github.com/niklio/iree-metal" \
  "${python_bin}" -m pip wheel --no-deps -v \
  --wheel-dir "${wheelhouse}" "${repo_root}/compiler"

echo "Building iree-pjrt-plugin-metal ${preview_version}"
"${python_bin}" -m pip wheel --no-deps -v \
  --wheel-dir "${wheelhouse}" \
  "${repo_root}/integrations/pjrt/python_packages/iree_metal_plugin"

IREE_METAL_PYTHON="${python_bin}" \
  "${script_dir}/validate_preview_wheels.sh" "${wheelhouse}" "${preview_version}"
"${script_dir}/collect_preview_licenses.sh" "${wheelhouse}"
IREE_METAL_PYTHON="${python_bin}" \
  "${script_dir}/write_preview_manifest.sh" "${wheelhouse}" "${preview_version}"

echo "Preview artifacts written to ${wheelhouse}"
