#!/usr/bin/env bash
# Regenerates one committed overlay and prints its new lock values.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SUBMODULE_NAME COMMITTED_BRANCH" >&2
  exit 2
fi

patch_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${patch_dir}/.." && pwd)"
submodule_name="$1"
branch="$2"

case "${submodule_name}" in
  llvm-project|spirv_cross|stablehlo) ;;
  *)
    echo "error: unsupported overlay: ${submodule_name}" >&2
    exit 2
    ;;
esac

submodule_path="third_party/${submodule_name}"
submodule_dir="${repo_root}/${submodule_path}"
base_revision="$(git -C "${repo_root}" ls-tree HEAD "${submodule_path}" | awk '{print $3}')"
patched_revision="$(git -C "${submodule_dir}" rev-parse "${branch}^{commit}")"
patch_file="${patch_dir}/${submodule_name}.patch"
temporary_file="$(mktemp "${patch_file}.tmp.XXXXXX")"

git -C "${submodule_dir}" diff --binary \
  "${base_revision}..${patched_revision}" > "${temporary_file}"
mv "${temporary_file}" "${patch_file}"

patched_tree="$(git -C "${submodule_dir}" rev-parse "${patched_revision}^{tree}")"
patch_sha="$(shasum -a 256 "${patch_file}" | awk '{print $1}')"
printf '%s\t%s\t%s\t%s\t%s\n' \
  "${submodule_path}" "${base_revision}" "${patched_tree}" \
  "${submodule_name}.patch" "${patch_sha}"
echo "Replace the matching line in ${patch_dir}/LOCK after reviewing the patch."
