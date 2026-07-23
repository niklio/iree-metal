# IREE-Metal submodule patches

This fork (`iree-metal`) carries changes to three vendored submodules that can't be
committed into this superproject directly (they live in their own git repos). They are
preserved here as diffs so **all** code changes are captured in one place.

| submodule | pinned commit | what the patch adds |
|---|---|---|
| `third_party/llvm-project` | `66395ad94` | MLIR `gpu.mma_matrix` + SPIR-V coop-matrix **bf16** support (GPUBase.td / GPUOps.td / GPUDialect.cpp) |
| `third_party/spirv_cross` | `7affe74` | SPIR-V→MSL **cooperative-matrix → `simdgroup_matrix`** emission (Load/Store/MulAdd, bf16, coop stride) in `spirv_msl.cpp` |
| `third_party/stablehlo` | `46af9d3` | StableHLO preprocessing tweak |

## Apply

```bash
./iree-metal-submodule-patches/apply.sh      # from the repo root, after `git submodule update --init`
```

The submodule pointers in this superproject are left at their upstream commits; applying
these patches reproduces the exact tree that produced the benchmarked numbers.
