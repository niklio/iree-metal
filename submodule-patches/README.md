# IREE-Metal submodule patches

This fork (`iree-metal`) carries changes to three vendored submodules that can't be
committed into this superproject directly (they live in their own git repos). They are
preserved here as diffs so **all** code changes are captured in one place.

| submodule | pinned commit | what the patch adds |
|---|---|---|
| `third_party/llvm-project` | `66395ad94` | First-class f16/bf16 subgroup MMA types, GPU→SPIR-V cooperative-matrix conversion, capability inference, and fail-closed NVVM handling |
| `third_party/spirv_cross` | `7affe74` | Hardened SPIR-V→MSL cooperative-matrix lowering to native Apple `simdgroup_matrix`, including f16/bf16 8×8 and logical 16×16 fixtures |
| `third_party/stablehlo` | `46af9d3` | StableHLO preprocessing tweak |

## Apply

```bash
./submodule-patches/apply.sh      # from the repo root, after `git submodule update --init`
```

The submodule pointers in this superproject are left at their upstream commits; applying
these patches reproduces the exact tree that produced the benchmarked numbers.

`./submodule-patches/refresh.sh` regenerates the LLVM and SPIRV-Cross snapshots,
including untracked new test fixtures. It deliberately leaves the independently
maintained StableHLO snapshot untouched.
