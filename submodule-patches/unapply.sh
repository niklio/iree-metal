#!/usr/bin/env bash
# Safely reverses overlays only when their locked trees are still intact.

set -euo pipefail

patch_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${patch_dir}/.." && pwd)"
lock_file="${patch_dir}/LOCK"

while IFS=$'\t' read -r submodule base_revision expected_tree patch_name patch_sha; do
  [[ -z "${submodule}" || "${submodule}" == \#* ]] && continue
  submodule_dir="${repo_root}/${submodule}"
  patch_file="${patch_dir}/${patch_name}"

  if [[ "$(git -C "${submodule_dir}" rev-parse HEAD)" != "${base_revision}" ]]; then
    echo "error: refusing to alter ${submodule}; HEAD moved during the build" >&2
    exit 2
  fi
  if ! git -C "${submodule_dir}" diff --quiet; then
    echo "error: refusing to alter ${submodule}; worktree changed during the build" >&2
    exit 2
  fi
  if [[ "$(git -C "${submodule_dir}" write-tree)" != "${expected_tree}" ]]; then
    echo "error: refusing to alter ${submodule}; index no longer matches LOCK" >&2
    exit 2
  fi

  git -C "${submodule_dir}" apply --index --reverse --whitespace=nowarn "${patch_file}"
  if [[ -n "$(git -C "${submodule_dir}" status --porcelain)" ]]; then
    echo "error: ${submodule} did not return to a clean base tree" >&2
    exit 2
  fi
  echo "reversed ${patch_name}: ${base_revision}"
done < "${lock_file}"
