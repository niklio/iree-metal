#!/usr/bin/env bash
# Refresh the modified LLVM and SPIRV-Cross snapshots, including untracked
# files. StableHLO is intentionally excluded because its patch is maintained
# independently from the Metal attention work.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"

snapshot() {
  local submodule="$1"
  local destination="$root/submodule-patches/$submodule.patch"
  local temporary
  temporary="$(mktemp "$destination.tmp.XXXXXX")"

  (
    cd "$root/third_party/$submodule"
    git diff --binary
    while IFS= read -r -d '' file; do
      git diff --no-index --binary -- /dev/null "$file" || {
        status=$?
        if [[ "$status" -ne 1 ]]; then
          exit "$status"
        fi
      }
    done < <(git ls-files --others --exclude-standard -z)
  ) >"$temporary"

  mv "$temporary" "$destination"
  echo "updated $destination"
}

snapshot llvm-project
snapshot spirv_cross
