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

## P1 IMPLEMENTATION STATUS (2026-07-23): 5 fixes landed; blocked at the bufferization stage
Committed, gated `IREE_METAL_COOP_ATTN_DECOMP` (all in SPIRV/Passes.cpp): (1) `SpecializeAttnMatmulPass`
raises the decomposed qk/pv generics to named `linalg.matmul` (+ copies the coop lowering_config);
(2) config-rank **trim** (specialize drops the unit batch dim → getTileSizes read `[1,16,16]` → native
size wrong → 8192 `vector<1x1>`; trim → L3 `[16,16,16]`); (3) `GenericVectorization` wired into the
DECOMP branch (SPIRVVectorizeToCoop unrolls an existing contract, doesn't vectorize linalg); (4)
`HoistScaleFromContractPass` hoists the softmax scale out of the qk LHS (`contract(mulf(A,s),B,0) ==
mulf(contract(A,B,0),s)`) so the operand is a plain transfer_read; (5) `SPIRVVectorToGPUSubgroupMMA`
wired in. Result: the qk/pv now become **2 clean coop-sized 16×16 `vector.contract` with plain
transfer_read operands** (was 3128 scalar FMul).
**HARD BLOCKER (architectural):** `convertVectorToMMAOps` needs **memrefs** —
`transferReadSupportsMMAMatrixType`→`getStaticallyKnownRowStride`→`dyn_cast<MemRefType>`. The DECOMP
coop path runs at **tensor** level (pre-bufferization), so every transfer_read is on a tensor → not
MMA-supported → 0 mma ops → the graceful fallback unrolls to scalar. The FFN coop pipeline runs its
coop/MMA passes POST-bufferization. **P1's remaining core = restructure the attention pipeline to
bufferize the flash scores/accumulators BEFORE the coop/MMA passes** (the code's own comment: grafting
promote/bufferize into the pre-bufferize tensor path "SEGFAULTS" — needs a proper restructure, the
multi-week heart of the port). Metric `CooperativeMatrixMulAdd` still 0, blocked here.

## Phased implementation plan (re-ordered 2026-07-23: SPIR-V is fully scalar → MLIR coop conversion first)
- **P1 — MLIR-level coop conversion (THE first blocker; layer 2).** Pinned to the exact pass:
  `GenericVectorization` (SPIRV/Passes.cpp:705, attention pipeline) vectorizes the decomposed qk/pv
  matmul generics to **scalar `vector.fma`, not `vector.contract`** — so `SPIRVVectorizeToCooperativeOps`
  never sees a contract to convert. RULED OUT: it is NOT the `generateContract` option (that defaults
  true and the attention pipeline uses the default; `generateContract=false` only in
  `addSPIRVSubgroupReducePassPipeline` for reductions). The real cause is op FORM: the decomposed op
  is a non-canonical `linalg.generic` (batch_matmul_transpose_b, reduction-in-middle iterator
  `[par,par,reduction,par]`) that the upstream linalg vectorizer's contraction path doesn't match.
  **FIX (validated at IR level 2026-07-23):** run `linalg-specialize-generic-ops` on the decomposed
  ops BEFORE GenericVectorization — it raises the qk/pv generics to named `linalg.matmul` (verified:
  11 generics → 9 + 2 matmul on the seed pre-GV IR). GOTCHA: `specializeGenericOp` DROPS the coop
  `lowering_config` (raised ops have 0 config), and `SPIRVTileToCooperativeOps` gates on
  matmul-*with-config* (.cpp:392). So the fix is a small pass that specializes the qk/pv generics AND
  copies their `lowering_config` to the new named op (or emit named ops with config directly in
  `DecomposeAttention`). Add it after `SPIRVTileToCooperativeOps`/before `GenericVectorization`
  (Passes.cpp:~704). Success metric on the seed repro: `CooperativeMatrixMulAdd` > 0 in SPIR-V
  (currently 0 / 3128 FMul).
  (Verify `SPIRVTileAndVectorizeToCooperativeOps` single-`rootOp` at .cpp:391 then also handles both
  matmuls; may need generalizing to multiple roots once contracts exist.)
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
