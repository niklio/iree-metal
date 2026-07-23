# Metal-SPIRV Attention Cooperative-Matrix Port — Implementation Plan

**Status:** diagnosis complete (2026-07-23); implementation is a multi-week, multi-pass codegen port.
**Goal:** make `iree_linalg_ext.attention` emit cooperative-matrix (simdgroup) MSL on the metal-spirv
target so a *fused flash* attention (no materialized T×T scores) also uses the matrix units — the last
lever to close the jax-metal gap (backward = 70% of it; causal models lose an extra ~14pts to jax-metal
skipping the masked upper triangle).

## Why this is the sole remaining lever (all angles converge here)
Every config/flag/primitive lever is falsified with full-suite data (67.4% of jax-metal is the
config-level ceiling — see `~/iree-fork-bench/CAMPAIGN.md` ledger). The four gap angles all reduce to
"metal-spirv attention codegen is scalar":
1. Fused-kernel quality — the backward coop kernels + ~22 glue dispatches vs MPSGraph's few.
2. Dispatch-count — folding transposes (`COOP_NO_SPLIT_TRANSPOSE`) reduces count but is **−20%** (slow
   fused codegen); count-reduction ≠ speedup.
3. Causal ~2× waste — jax-metal skips the masked upper triangle (causal gpt2 3.27 > bidir bert 2.88);
   the fork computes full T×T then masks. Mask is a clean recoverable `select(iota_i >= iota_j, s, -inf)`
   pattern at stablehlo (recognizable), but exploiting it needs a non-scalar attention kernel first.
4. The `iree_linalg_ext.attention` `$mask` operand tile-skips causal on CUDA/ROCm — but metal-spirv
   codegen for the op is scalar.

## Current state: the fork today
HF models do NOT raise to `iree_linalg_ext.attention`; they emit **separate coop matmul dispatches**
(qk, pv) which DO use coop (verified: coop=64/128 in real bf16 dumps), plus a materialized-scores +
mask dispatch. That's the 67.4% path. Raising to the attention op is currently STRICTLY WORSE (scalar).

## The scalar chain — layers found (each a real blocker)
Repro: `IREE_METAL_COOP_ATTENTION_WIP=1 IREE_METAL_COOP_ATTN_DECOMP=1 IREE_METAL_COOP_ATTN_M=16 \
iree-compile /tmp/attn_op.mlir --iree-hal-target-backends=metal-spirv --iree-metal-compile-to-metallib=false`
(seed MLIR is a plain `iree_linalg_ext.attention`, 12x512x64 f16). Result: 166–882 KB kernel, **0 coop**.

1. **Config/predicate are FINE.** `setAttentionOpConfig` (KernelConfig.cpp:1214, gated
   `IREE_METAL_COOP_ATTENTION_WIP`) + the `COOP_ATTN_DECOMP` decomposition_config attach coop
   `lowering_config`s (`[.,.,.,[1,16,16,16]]`) to the decomposed qk/pv matmuls. Both pass
   `isMatmulOrBatchMatmul` (3 parallel loops, contraction interface, 2 non-unit parallel dims).
2. **Single-rootOp coop pass.** `SPIRVTileAndVectorizeToCooperativeOps.cpp:391-407` walks for the
   FIRST matmul-with-config as a single `rootOp`, sets ONE per-function coop shape + subgroup count.
   Built for single-matmul dispatches; attention has TWO matmuls (qk `batch_matmul_transpose_b` +
   pv) nested inside the flash `scf.for`. (Both share 16×16×16, so the single shape isn't wrong per se,
   but the tiling/distribution is not structured for two loop-nested contractions.)
3. **The SPIR-V is FULLY SCALAR — no coop compute is ever generated.** (Corrected 2026-07-23: an
   earlier read of "spirv.coopmatrix ops present" was 1 unused *type declaration*.) The tiny seq=16
   attention SPIR-V has **0 `CooperativeMatrixMulAdd`, 0 `CooperativeMatrixLoad`, and 3128 scalar
   `spirv.FMul/FAdd`**. So the coop conversion never happens at the MLIR level: after
   `SPIRVVectorizeToCooperativeOps` the flash-nested `vector.contract`s are gone but lowered to
   **scalar** vector ops (generic vectorization), not coop — so nothing coop reaches SPIR-V, and
   spirv-cross survival (a `spvCoopMat` question) is MOOT until MLIR-level coop conversion works.
4. **Thread-distribution path is also broken.** `IREE_METAL_COOP_ATTN_THREAD` tiles to `scf.forall`
   with `#gpu.thread<linear_dim_0>` which "fails to legalize" — SPIRV has no scf.forall→thread
   distribution for that mapping (`createGPUDistributePass` is in the pipeline but doesn't handle it).

## Falsified WIP knobs (do not re-try as-is)
`COOP_ATTN_M` (1/16 scalar, 32 hangs), `COOP_ATTN_DECOMP` (0 coop), `COOP_ATTN_THREAD` (scf.forall
legalization failure), `COOP_ATTN_K2/SMEM/COOPSMEM`. All present-but-non-functional scaffolding.

## Phased implementation plan (re-ordered 2026-07-23: SPIR-V is fully scalar → MLIR coop conversion first)
- **P1 — MLIR-level coop conversion (THE first blocker; layer 2).** The flash-nested qk/pv
  `vector.contract`s go to scalar generic vectorization instead of coop. Generalize
  `SPIRVTileAndVectorizeToCooperativeOps` (single-`rootOp`, .cpp:391-407) to convert BOTH matmuls
  inside the flash `scf.for`, OR restructure `DecomposeAttention` so each contraction is a separately
  coop-tileable region. Success metric on the seed repro: SPIR-V shows `CooperativeMatrixMulAdd` > 0
  (currently 0 / 3128 scalar FMul). Everything else is downstream of this.
- **P2 — spirv-cross → MSL coop survival (layer 3).** Only relevant AFTER P1 emits coop SPIR-V:
  verify our `spvCoopMat` simdgroup emission (spirv_msl.cpp) lowers the attention path's
  `CooperativeMatrixMulAdd` to `simdgroup_multiply` in MSL (it handles the FFN path; confirm the
  attention shapes/flavor too).
- **P3 — thread/subgroup distribution (layer 4).** Add scf.forall→thread legalization for the
  `linear_dim_0` mapping so M-block>1 runs at full occupancy without unroll explosion.
- **P4 — causal tile-skip.** Recognize the `select(iota_i>=iota_j, scores, -inf)` mask (clean at
  stablehlo) → mark attention causal → skip upper-triangle K2 tiles in the flash loop. ~2× on the 4
  causal models.

## Validation harness
`make ab` (correctness-gated, full 10-model), NOT a subset A/B (subset traps: `COOP_BMM_BOOST` looked
+1.5% on 5 models but was full-suite-neutral). Backup + `make ship` only on a full-suite win.
