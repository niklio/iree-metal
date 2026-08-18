#!/bin/bash
# Builds the two macOS arm64 wheels needed by the iree-metal developer preview.

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
wheelhouse="${IREE_METAL_WHEELHOUSE:-${repo_root}/wheelhouse}"
python_bin="${IREE_METAL_PYTHON:-python3}"
preview_version="${IREE_METAL_VERSION:-}"
jax_version="0.11.1"
jax_min_version="0.10.2"
jax_max_version="0.12"
build_root="${IREE_METAL_BUILD_ROOT:-${repo_root}/.iree-metal-build}"
requirements_file="${script_dir}/requirements-preview-macos-arm64-py312.txt"
build_requirements_file="${script_dir}/requirements-build-macos-arm64-py312.txt"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "error: preview wheels must be built natively on macOS arm64" >&2
  exit 2
fi

if [[ -z "${preview_version}" ]]; then
  echo "error: set IREE_METAL_VERSION to a PEP 440 version (for example 3.11.0.dev20260801)" >&2
  exit 2
fi

if ! "${python_bin}" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo "error: the preview is built and tested with Python 3.12 exactly" >&2
  exit 2
fi

if ! "${python_bin}" -c 'import re, sys; raise SystemExit(not re.fullmatch(r"3\.11\.0\.dev[0-9]{8}(?:[0-9]{2})?", sys.argv[1]))' "${preview_version}"; then
  echo "error: IREE_METAL_VERSION must match 3.11.0.devYYYYMMDD or 3.11.0.devYYYYMMDDNN" >&2
  exit 2
fi

verify_submodules() {
  local superproject="$1"
  local modules_file="${superproject}/.gitmodules"
  [[ -f "${modules_file}" ]] || return 0
  git -C "${superproject}" config --file "${modules_file}" --get-regexp path |
    while read -r _ relative_path; do
      local submodule="${superproject}/${relative_path}"
      if [[ ! -e "${submodule}/.git" ]]; then
        echo "error: submodule is not initialized: ${submodule}" >&2
        return 1
      fi
      local expected_revision
      expected_revision="$(git -C "${superproject}" ls-tree HEAD -- "${relative_path}" | awk '{print $3}')"
      local actual_revision
      actual_revision="$(git -C "${submodule}" rev-parse HEAD)"
      if [[ -z "${expected_revision}" || "${actual_revision}" != "${expected_revision}" ]]; then
        echo "error: submodule does not match the recorded revision: ${submodule}" >&2
        return 1
      fi
      verify_submodules "${submodule}" || return 1
    done
}

if ! verify_submodules "${repo_root}"; then
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
  if [[ "${patches_applied:-0}" == "1" ]]; then
    "${repo_root}/submodule-patches/unapply.sh"
  fi
}
trap cleanup EXIT

mkdir -p "${wheelhouse}"
if compgen -G "${wheelhouse}/*.whl" > /dev/null; then
  echo "error: ${wheelhouse} already contains wheels; use an empty output directory" >&2
  exit 2
fi

if [[ ! -f "${requirements_file}" || ! -f "${build_requirements_file}" ]]; then
  echo "error: locked runtime or build dependencies are missing" >&2
  exit 2
fi

mkdir -p "${build_root}"
build_venv="${build_root}/venv"
if [[ ! -x "${build_venv}/bin/python" ]]; then
  "${python_bin}" -m venv "${build_venv}"
fi
build_python="${build_venv}/bin/python"
"${build_python}" -m pip install \
  --disable-pip-version-check \
  --require-hashes \
  -r "${build_requirements_file}"
export PATH="${build_venv}/bin:${PATH}"
export IREE_COMPILER_API_CMAKE_BUILD_DIR="${build_root}/compiler"
export IREE_PJRT_CMAKE_BUILD_DIR="${build_root}/pjrt-metal"

# Make source references independent of the maintainer's checkout path and
# normalize diagnostics. SOURCE_DATE_EPOCH also gives wheel metadata and
# archives a stable timestamp. Modern Apple ld derives required Mach-O UUIDs
# from linked content; omitting LC_UUID makes binaries unloadable on macOS.
prefix_maps="-ffile-prefix-map=${repo_root}=iree-metal -fdebug-prefix-map=${repo_root}=iree-metal -fmacro-prefix-map=${repo_root}=iree-metal"
export CFLAGS="${CFLAGS:-} ${prefix_maps}"
export CXXFLAGS="${CXXFLAGS:-} ${prefix_maps}"
export OBJCFLAGS="${OBJCFLAGS:-} ${prefix_maps}"
export OBJCXXFLAGS="${OBJCXXFLAGS:-} ${prefix_maps}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "${repo_root}" show -s --format=%ct HEAD)}"
export ZERO_AR_DATE=1
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export CMAKE_OSX_ARCHITECTURES=arm64
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
export ARCHFLAGS="-arch arm64"
# GitHub's arm64 Python distribution can report itself as universal2. Force
# setuptools to describe the arm64-only native payload truthfully.
export _PYTHON_HOST_PLATFORM="macosx-${MACOSX_DEPLOYMENT_TARGET}-arm64"
wheel_platform="$("${build_python}" -c \
  'from setuptools.command.bdist_wheel import get_platform; print(get_platform(None).lower().replace("-", "_").replace(".", "_").replace(" ", "_"))')"
if [[ "${wheel_platform}" != "macosx_13_0_arm64" ]]; then
  echo "error: wheel platform resolved to ${wheel_platform}, expected macosx_13_0_arm64" >&2
  exit 2
fi

patches_applied=0
"${repo_root}/submodule-patches/apply.sh"
patches_applied=1

printf '{\n  "package-version": "%s",\n  "package-suffix": "-iree-metal"\n}\n' \
  "${preview_version}" > "${compiler_version_file}"
printf '{\n  "package-version": "%s",\n  "package-suffix": "-iree-metal",\n  "compiler-package-name": "iree-base-compiler-iree-metal",\n  "compiler-package-version": "%s",\n  "jax-requires": ">=%s,<%s",\n  "jaxlib-requires": ">=%s,<%s",\n  "python-requires": ">=3.12,<3.13",\n  "project-url": "https://github.com/niklio/iree-metal"\n}\n' \
  "${preview_version}" "${preview_version}" \
  "${jax_min_version}" "${jax_max_version}" \
  "${jax_min_version}" "${jax_max_version}" \
  > "${pjrt_version_file}"

export IREE_CMAKE_BUILD_TYPE=Release
export LLVM_PARALLEL_LINK_JOBS="${LLVM_PARALLEL_LINK_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-3}"

echo "Building iree-base-compiler-iree-metal ${preview_version}"
IREE_COMPILER_CUSTOM_DESCRIPTION="IREE compiler for the iree-metal developer preview" \
IREE_COMPILER_CUSTOM_HOMEPAGE_URL="https://github.com/niklio/iree-metal" \
IREE_COMPILER_CUSTOM_REPOSITORY_URL="https://github.com/niklio/iree-metal" \
  "${build_python}" -m pip wheel --no-build-isolation --no-deps -v \
  --wheel-dir "${wheelhouse}" "${repo_root}/compiler"

echo "Building iree-pjrt-plugin-metal-iree-metal ${preview_version}"
# setuptools builds local projects in place. Remove this package's generated
# staging tree so an earlier developer build can never leak a stale native
# library or bytecode into the release wheel. Reconfigure a reused CMake build
# so release-only compiler flags such as the source-prefix maps cannot be
# silently ignored by an older cache. The dependency and object caches live in
# ${build_root} and are intentionally preserved; Ninja rebuilds only commands
# whose configuration changed.
plugin_package_dir="${repo_root}/integrations/pjrt/python_packages/iree_metal_plugin"
rm -rf \
  "${plugin_package_dir}/build" \
  "${plugin_package_dir}/iree_pjrt_plugin_metal_iree_metal.egg-info"
rm -f "${IREE_PJRT_CMAKE_BUILD_DIR}/CMakeCache.txt"
"${build_python}" -m pip wheel --no-build-isolation --no-deps -v \
  --wheel-dir "${wheelhouse}" \
  "${plugin_package_dir}"

"${build_python}" "${script_dir}/normalize_preview_wheels.py" \
  --source-date-epoch "${SOURCE_DATE_EPOCH}" "${wheelhouse}"/*.whl

mkdir -p "${wheelhouse}/dependencies"
cp "${requirements_file}" \
  "${wheelhouse}/requirements-macos-arm64-py312.txt"
cp "${build_requirements_file}" \
  "${wheelhouse}/requirements-build-macos-arm64-py312.txt"
"${build_python}" -m pip download \
  --dest "${wheelhouse}/dependencies" \
  --require-hashes \
  --only-binary=:all: \
  --platform macosx_13_0_arm64 \
  --python-version 3.12 \
  --implementation cp \
  --abi cp312 \
  -r "${requirements_file}"

IREE_METAL_PYTHON="${build_python}" \
  "${script_dir}/validate_preview_wheels.sh" "${wheelhouse}" "${preview_version}"
"${script_dir}/collect_preview_licenses.sh" "${wheelhouse}"
cp "${repo_root}/docs/metal/developer-preview.md" "${wheelhouse}/INSTALL.md"
cp "${repo_root}/docs/metal/releases/${preview_version}.md" \
  "${wheelhouse}/RELEASE_NOTES.md"
"${build_python}" "${script_dir}/write_preview_sbom.py" \
  "${wheelhouse}" "${preview_version}"
IREE_METAL_PYTHON="${build_python}" \
  IREE_METAL_JAX_VERSION="${jax_version}" \
  "${script_dir}/write_preview_manifest.sh" "${wheelhouse}" "${preview_version}"

echo "Preview artifacts written to ${wheelhouse}"
