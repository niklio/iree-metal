#!/usr/bin/env bash
# Apply the nlearn submodule patches. Run from the repo root after
# `git submodule update --init third_party/{llvm-project,spirv_cross,stablehlo}`.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
for sub in llvm-project spirv_cross stablehlo; do
  echo "== applying $sub.patch =="
  git -C "third_party/$sub" apply "$here/$sub.patch"
done
echo "done — all nlearn submodule patches applied"
