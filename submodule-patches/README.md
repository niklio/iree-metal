# In-repository third-party source overlays

iree-metal keeps its LLVM/MLIR, SPIRV-Cross, and StableHLO modifications in
this repository. The upstream projects remain ordinary submodules pinned to
public base commits; the files in this directory are the complete binary-safe
diffs applied on top of those bases.

`LOCK` records five values for every overlay:

1. submodule path;
2. public upstream base revision;
3. expected Git tree after applying the patch;
4. patch path; and
5. patch SHA-256.

This makes the release source self-contained without requiring separately
maintained GitHub forks. A clean recursive clone plus this superproject commit
contains every source modification used to build the wheels.

## Release build behavior

`build_tools/iree_metal/build_preview_wheels.sh` starts by requiring a clean,
base-pinned recursive checkout. It calls `apply.sh`, which verifies patch
checksums, applies each overlay to the submodule index, and compares the exact
resulting Git tree with `LOCK`. The build cleanup trap calls `unapply.sh`, which
will reverse an overlay only when both the index tree and worktree still match
the locked state. Unexpected developer changes are therefore left in place
instead of being overwritten.

The same lock and patch hashes are copied into the release provenance manifest.

## Maintainer workflow

To update an overlay, make and commit the third-party change on a local
submodule branch, then run `refresh.sh SUBMODULE BRANCH`. Review the regenerated
patch and lock entry in the superproject, run `apply.sh` and `unapply.sh`, and
commit both files together. Release tags must never rely on untracked changes
inside a submodule.
