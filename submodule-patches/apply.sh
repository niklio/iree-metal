#!/usr/bin/env bash
# Applies and verifies all in-repository third-party source overlays.

set -euo pipefail

patch_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${patch_dir}/.." && pwd)"
lock_file="${patch_dir}/LOCK"
applied_file="$(mktemp "${TMPDIR:-/tmp}/iree-metal-overlays.XXXXXX")"

rollback_partial_application() {
  while IFS=$'\t' read -r applied_submodule applied_patch; do
    [[ -z "${applied_submodule}" ]] && continue
    git -C "${repo_root}/${applied_submodule}" apply \
      --index --reverse --whitespace=nowarn "${patch_dir}/${applied_patch}" || true
  done < "${applied_file}"
  rm -f "${applied_file}"
}
trap rollback_partial_application ERR

# Complete a non-mutating preflight for every overlay before applying any one
# of them. This makes a stale lock or dirty sibling fail before partial state is
# introduced.
while IFS=$'\t' read -r submodule base_revision expected_tree patch_name patch_sha; do
  [[ -z "${submodule}" || "${submodule}" == \#* ]] && continue
  submodule_dir="${repo_root}/${submodule}"
  patch_file="${patch_dir}/${patch_name}"

  if [[ ! -d "${submodule_dir}/.git" && ! -f "${submodule_dir}/.git" ]]; then
    echo "error: submodule is not initialized: ${submodule}" >&2
    exit 2
  fi
  if [[ "$(git -C "${submodule_dir}" rev-parse HEAD)" != "${base_revision}" ]]; then
    echo "error: ${submodule} is not at locked base ${base_revision}" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${submodule_dir}" status --porcelain)" ]]; then
    echo "error: ${submodule} is dirty before overlay application" >&2
    exit 2
  fi
  if [[ "$(shasum -a 256 "${patch_file}" | awk '{print $1}')" != "${patch_sha}" ]]; then
    echo "error: patch checksum mismatch: ${patch_name}" >&2
    exit 2
  fi

  git -C "${submodule_dir}" apply \
    --check --index --whitespace=nowarn "${patch_file}"
done < "${lock_file}"

while IFS=$'\t' read -r submodule base_revision expected_tree patch_name patch_sha; do
  [[ -z "${submodule}" || "${submodule}" == \#* ]] && continue
  submodule_dir="${repo_root}/${submodule}"
  patch_file="${patch_dir}/${patch_name}"

  git -C "${submodule_dir}" apply --index --whitespace=nowarn "${patch_file}"
  printf '%s\t%s\n' "${submodule}" "${patch_name}" >> "${applied_file}"
  actual_tree="$(git -C "${submodule_dir}" write-tree)"
  if [[ "${actual_tree}" != "${expected_tree}" ]]; then
    echo "error: patched tree mismatch for ${submodule}: ${actual_tree}" >&2
    exit 2
  fi
  echo "applied ${patch_name}: ${actual_tree}"
done < "${lock_file}"

trap - ERR
rm -f "${applied_file}"
