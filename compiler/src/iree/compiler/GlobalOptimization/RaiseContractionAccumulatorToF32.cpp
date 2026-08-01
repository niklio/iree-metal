// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Raises the ACCUMULATOR (init/result) of a low-precision (bf16/f16) linalg
// contraction to f32, then truncates the result back. This is the metal-spirv
// analogue of the CPU/VMVX `skipIntermediateRoundings` behaviour: without it, a
// bf16-in/bf16-out matmul rounds the accumulator to bf16 on EVERY multiply-add
// (slow + inaccurate). Accumulating in f32 and rounding once matches XLA /
// jax-metal default mixed-precision semantics and is strictly more accurate.
// (This IREE build targets metal-spirv only, so the transform runs globally.)
//===----------------------------------------------------------------------===//

#include <cstdlib>

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_RAISECONTRACTIONACCUMULATORTOF32PASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

// Cast a ranked tensor elementwise to `dstElemType` via a linalg.generic
// (extf when widening, truncf when narrowing).
static Value castTensorElementwise(PatternRewriter &rewriter, Location loc,
                                   Value src, Type dstElemType, bool widen) {
  auto srcType = cast<RankedTensorType>(src.getType());
  auto dstType = RankedTensorType::get(srcType.getShape(), dstElemType,
                                       srcType.getEncoding());
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, src);
  Value empty = tensor::EmptyOp::create(rewriter, loc, mixedSizes, dstElemType);
  SmallVector<AffineMap> maps(
      2, rewriter.getMultiDimIdentityMap(srcType.getRank()));
  SmallVector<utils::IteratorType> iters(srcType.getRank(),
                                         utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             rewriter, loc, TypeRange{dstType}, ValueRange{src},
             ValueRange{empty}, maps, iters,
             [&](OpBuilder &b, Location loc, ValueRange args) {
               Value r = widen ? Value(arith::ExtFOp::create(b, loc, dstElemType,
                                                             args[0]))
                               : Value(arith::TruncFOp::create(
                                     b, loc, dstElemType, args[0]));
               linalg::YieldOp::create(b, loc, r);
             })
      ->getResults()[0];
}

struct RaiseAccumulatorPattern : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = linalgOp.getOperation();
    if (!isa<linalg::ContractionOpInterface>(op)) {
      return failure();
    }
    // Only handle single-init, single-result contractions.
    if (linalgOp.getNumDpsInits() != 1 || op->getNumResults() != 1) {
      return failure();
    }
    Value init = linalgOp.getDpsInits()[0];
    auto initType = dyn_cast<RankedTensorType>(init.getType());
    if (!initType) {
      return failure();
    }
    Type elemType = initType.getElementType();
    // Match a low-precision float accumulator (bf16 or f16). f32/f64 already ok.
    if (!(elemType.isBF16() || elemType.isF16())) {
      return failure();
    }
    // Only PURE 2D matmul (FFN/projections + transposed backward forms).
    // Batch matmuls (attention q@kᵀ / a@v) CONVERT to coop standalone (verified:
    // scf.forall tiles the batch to 1), but in real attention they feed a
    // scale->mask->softmax epilogue chain that blocks coop conversion and leaves
    // illegal 2D vectors. Isolating them (barrier) would MATERIALIZE the [T,T]
    // scores ([24,512,512], 25MB) — a perf loss, the opposite of flash attention.
    // So attention-on-coop needs real flash-attention fusion (iree_linalg_ext
    // attention op), not batch promotion. Left 2D-only; batch stays scalar.
    bool isMatmul = IREE::LinalgExt::isPureMatmul(op);
    // Promote attention batch matmuls (q@kᵀ / a@v) to f32 so they can hit coop —
    // paired with the reduction-epilogue barrier split below that isolates them
    // from the softmax into their own clean coop dispatch on the matrix units.
    //
    // iree-metal (cont408): DEFAULT-ON after a full 10-model correctness-gated A/B
    // (all 10 correct vs deployed; mean +3.6% throughput, gpt2 +10.1%, deit +4.9%,
    // the four 512-seq encoders +3.4%; vit-base neutral+correct via the pad-skip
    // gate below). Two gates below keep it safe: (1) skip PADDED bmms (odd-seq vit
    // corrupts under f32-promote), (2) only promote when mn<=MAXMN so the isolation
    // gate also fires (large-T gpt2 T=1024 stayed on the untouched path pre-gate,
    // -34% → now neutral/positive). Opt out entirely with IREE_METAL_COOP_NO_ATTN_SPLIT.
    //   IREE_METAL_COOP_ATTN_F32ACC = force promotion-only (no barrier) for experiments
    //     (bmm stays scalar-fused with softmax but accumulates in f32).
    bool allowBatch = getenv("IREE_METAL_COOP_NO_ATTN_SPLIT") == nullptr ||
                      getenv("IREE_METAL_COOP_ATTN_F32ACC") != nullptr;
    bool isBatchMatmul = allowBatch && IREE::LinalgExt::isPureBatchMatmul(op);
    if (!isMatmul && !isBatchMatmul) {
      return failure();
    }
    // Size-gate: 2D matmuls to LARGE (>=128, FFN/projections). Batch matmuls gate
    // the last-3 contraction dims to mult-16/>=16 (coop-native), batch dim free.
    {
      SmallVector<int64_t> ranges = linalgOp.getStaticLoopRanges();
      if (isBatchMatmul) {
        if (ranges.size() < 3)
          return failure();
        // iree-metal (cont407): do NOT f32-promote a PADDED attention batch-matmul.
        // PadBatchMatmulToCoopPattern pads an odd-seq bmm (e.g. vit 197->208) to
        // reach coop; f32-promoting the *padded* op corrupts the result (vit-base:
        // 141x drift / blowup — the fake padded score columns interact badly with
        // the f32 accumulator + slice-back). Isolated by probe: pad-off=correct,
        // promote+pad=FAIL, isolate-off+pad=FAIL. Skip promotion when any input is
        // a tensor.pad so odd-seq models stay on their correct (non-promoted) path.
        // (Future work: mask the padded scores so pad+promote is numerically exact
        // and vit attention can also hit the matrix units.)
        for (Value in : linalgOp.getDpsInputs())
          if (in.getDefiningOp<tensor::PadOp>())
            return failure();
        for (int64_t d : ArrayRef<int64_t>(ranges).take_back(3))
          if (ShapedType::isDynamic(d) || d < 16 || (d % 16) != 0)
            return failure();
        // iree-metal (cont405): ONLY promote an attention batch-matmul to f32 when the
        // isolation gate below (IsolateBatchMatmulPattern, mn<=MAXMN) will ALSO
        // fire. Promoting without isolating leaves an f32 bmm FUSED into the
        // softmax epilogue: still scalar (reduction blocks coop) but now
        // materializing f32 scores instead of bf16 => 2x score memory for zero
        // coop benefit (measured: gpt2 T=1024 -33.9%). Gating promotion to the
        // same mn<=MAXMN keeps promote+isolate consistent so large-T stays on the
        // untouched bf16-fused-scalar path. M,N are the last-3-but-K dims.
        int64_t M = ranges[ranges.size() - 3];
        int64_t N = ranges[ranges.size() - 2];
        int64_t maxMN = 600000;
        if (const char *e = getenv("IREE_METAL_COOP_ATTN_ISOLATE_MAXMN"))
          maxMN = std::strtoll(e, nullptr, 10);
        if (M * N > maxMN)
          return failure();
      } else if (ranges.empty() || llvm::any_of(ranges, [](int64_t d) {
                   return ShapedType::isDynamic(d) || d < 128;
                 })) {
        return failure();
      }
    }
    Type f32 = rewriter.getF32Type();
    Location loc = linalgOp.getLoc();

    // Build the f32 accumulator init. If the original init is a zero linalg.fill
    // (the common fresh-matmul case), create a CLEAN f32 zero fill — this matches
    // the jax f32-accum form (dot preferred_element_type=f32) whose store-downcast
    // fuses correctly, instead of an extf(bf16-fill) that needs the barrier
    // workaround. Otherwise extf to preserve accumulate-into semantics.
    Value f32Init;
    auto fillOp = init.getDefiningOp<linalg::FillOp>();
    if (fillOp) {
      auto f32InitTy = RankedTensorType::get(initType.getShape(), f32,
                                             initType.getEncoding());
      Value empty = tensor::EmptyOp::create(rewriter, loc, f32InitTy.getShape(),
                                            f32);
      Value zero = arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getF32FloatAttr(0.0));
      f32Init = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                       ValueRange{empty})
                    .getResult(0);
    } else {
      f32Init = castTensorElementwise(rewriter, loc, init, f32, /*widen=*/true);
    }
    auto f32ResultType =
        RankedTensorType::get(initType.getShape(), f32, initType.getEncoding());
    SmallVector<Value> inputs = linalgOp.getDpsInputs();

    Operation *newOp =
        isBatchMatmul
            ? linalg::BatchMatmulOp::create(rewriter, loc,
                                            TypeRange{f32ResultType}, inputs,
                                            ValueRange{f32Init})
                  .getOperation()
            : linalg::MatmulOp::create(rewriter, loc, TypeRange{f32ResultType},
                                       inputs, ValueRange{f32Init})
                  .getOperation();
    // Keep the f32 matmul and the narrowing truncf in SEPARATE dispatches: a
    // FUSED f32-C -> bf16-store miscompiles the coop pipeline, but a standalone
    // f32-output coop matmul + a separate bf16 cast is numerically correct
    // (validated: bf16-in/f32-out coop matmul rel_err 2e-3). The barrier breaks
    // the producer-consumer chain so dispatch formation won't fuse the truncf in.
    // In real models the narrowing truncf has a consumer, so early-trunc-fusion
    // forward-fuses it (keeping the matmul f32-output = correct) — measured 3.49
    // TFLOP/s (vs 2.47 with a barrier that isolates the truncf). So NO barrier by
    // default. Only the degenerate "matmul result returned directly with a bf16
    // cast, no consumer" case fuses the truncf BACKWARD into the matmul and
    // miscompiles the store-downcast — opt into the barrier there via env.
    Value f32Res = newOp->getResult(0);
    if (getenv("IREE_METAL_COOP_BARRIER"))
      f32Res = IREE::Util::OptimizationBarrierOp::create(rewriter, loc, f32Res)
                   ->getResult(0);
    // Narrow the f32 result back to the original low-precision type.
    Value narrowed =
        castTensorElementwise(rewriter, loc, f32Res, elemType, /*widen=*/false);
    rewriter.replaceOp(op, narrowed);
    return success();
  }
};

// Returns true if `v` is (transitively, through identity-map elementwise
// generics like a truncf) produced by a large linalg.matmul.
static bool isFedByLargeMatmul(Value v, int depth = 3) {
  Operation *def = v.getDefiningOp();
  if (!def || depth < 0)
    return false;
  if (auto mm = dyn_cast<linalg::MatmulOp>(def)) {
    SmallVector<int64_t> ranges =
        cast<linalg::LinalgOp>(mm.getOperation()).getStaticLoopRanges();
    return !ranges.empty() && llvm::all_of(ranges, [](int64_t d) {
      return !ShapedType::isDynamic(d) && d >= 128;
    });
  }
  // Walk back through a single-input elementwise generic whose input map is the
  // identity (e.g. the f32->bf16 truncf epilogue): its input still carries the
  // matmul's untransposed layout.
  if (auto g = dyn_cast<linalg::GenericOp>(def)) {
    if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1)
      return false;
    OpOperand *in = g.getDpsInputOperands()[0];
    AffineMap inMap = g.getMatchingIndexingMap(in);
    AffineMap outMap = g.getMatchingIndexingMap(g.getDpsInitOperand(0));
    if (inMap != outMap || !inMap.isIdentity())
      return false;
    return isFedByLargeMatmul(in->get(), depth - 1);
  }
  return false;
}

// Isolates a TRANSPOSE that consumes a matmul (directly or through its truncf
// epilogue) into its own dispatch by inserting a util.optimization_barrier on the
// transpose's input. Backward weight-gradients (dW = xᵀ·dY) get stored transposed
// as `matmul(f32) -> truncf(bf16) -> transpose`; if the transpose fuses back into
// the matmul dispatch, the metal-spirv coop store codegen can't emit the
// mma_matrix result through the transposed layout, so the whole matmul falls off
// the matrix units to scalar. Breaking the edge just BEFORE the transpose keeps
// matmul+truncf together (a clean f32-out matmul with an identity truncf epilogue,
// which DOES lower to matrix units — the same shape as the inference path) and
// makes the transpose a cheap separate memory-bound dispatch. This is what puts
// the backward (2/3 of training FLOPs) onto the matrix units.
struct SplitTransposeEpiloguePattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Must be a pure single-input permutation (transpose): in-map != out-map.
    if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
      return failure();
    OpOperand *in = genericOp.getDpsInputOperands()[0];
    AffineMap inMap = genericOp.getMatchingIndexingMap(in);
    AffineMap outMap =
        genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
    if (inMap == outMap || !inMap.isPermutation() || !outMap.isPermutation())
      return failure();
    Value src = in->get();
    // Already isolated behind a barrier?
    if (src.getDefiningOp<IREE::Util::OptimizationBarrierOp>())
      return failure();
    if (!isFedByLargeMatmul(src))
      return failure();
    rewriter.setInsertionPoint(genericOp);
    auto barrier =
        IREE::Util::OptimizationBarrierOp::create(rewriter, genericOp.getLoc(), src);
    rewriter.modifyOpInPlace(
        genericOp, [&]() { in->set(barrier->getResult(0)); });
    return success();
  }
};

// IREE_METAL_COOP_ATTN_SPLIT experiment: isolate every (promoted, f32-output) batch
// matmul into its OWN dispatch by putting a barrier on its result. Attention
// q@kᵀ / a@v are the only batch matmuls; isolating them from their scale/softmax/
// transpose epilogue makes each a clean f32-output coop dispatch (materializing
// the [B*H,T,T] scores). Tests by MEASUREMENT whether materialized-coop attention
// beats the current scalar-fused attention on this fork.
struct IsolateBatchMatmulPattern : OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp bmm,
                                PatternRewriter &rewriter) const override {
    Value result = bmm.getResult(0);
    // Normally only isolate the f32-output (promoted) form. IREE_METAL_COOP_ATTN_
    // ISOLATE_BF16 also isolates bf16 bmms (un-fuse the attention bmm from its
    // softmax/scale/transpose backward epilogue WITHOUT f32 promotion) — tests
    // whether un-fusing alone fixes the bf16 attention-backward NaN while keeping
    // bf16 scores (no f32 materialization / no large-T regression).
    // DEFAULT-ON (Nik-approved cont190): un-fusing the bf16 attention bmm from its
    // backward epilogue fixes a real correctness bug (NaN on 6/10 std bf16
    // transformers) + is faster on 8/10, zero regressions. Opt out with
    // IREE_METAL_COOP_NO_ATTN_ISOLATE_BF16.
    static const bool isolateBf16 =
        ::getenv("IREE_METAL_COOP_NO_ATTN_ISOLATE_BF16") == nullptr;
    Type et = cast<ShapedType>(result.getType()).getElementType();
    if (!et.isF32() && !(isolateBf16 && et.isBF16()))
      return failure();
    if (result.hasOneUse() &&
        isa<IREE::Util::OptimizationBarrierOp>(*result.getUsers().begin()))
      return failure();
    // Size-gate: isolating un-fuses the bmm from its epilogue, materializing the
    // [B,M,N] result. For attention that's the [B*H,T,T] scores; at large T
    // (e.g. T=1024) that extra materialization outweighs the benefit (+17% on
    // gpt2-1024, which doesn't need the NaN fix anyway). Skip isolation when the
    // per-batch output M*N exceeds a threshold (default 600k ≈ T≤775 for square
    // scores) so the T=512 NaN models + vit still isolate but large-T stays fused.
    // Override via IREE_METAL_COOP_ATTN_ISOLATE_MAXMN.
    {
      auto rTy = cast<ShapedType>(result.getType());
      int64_t rank = rTy.getRank();
      if (rank >= 2 && !rTy.isDynamicDim(rank - 1) &&
          !rTy.isDynamicDim(rank - 2)) {
        int64_t mn = rTy.getDimSize(rank - 1) * rTy.getDimSize(rank - 2);
        int64_t maxMN = 600000;
        if (const char *e = ::getenv("IREE_METAL_COOP_ATTN_ISOLATE_MAXMN"))
          maxMN = std::strtoll(e, nullptr, 10);
        if (mn > maxMN)
          return failure();
      }
    }
    rewriter.setInsertionPointAfter(bmm);
    auto barrier =
        IREE::Util::OptimizationBarrierOp::create(rewriter, bmm.getLoc(), result);
    rewriter.replaceAllUsesExcept(result, barrier->getResult(0), barrier);
    return success();
  }
};

// Pads a non-mult-16 2D matmul's M/N/K up to a multiple of 16 with zeros, runs
// the matmul on the padded operands, and extract_slices the result back to the
// original shape. Zero padding is algebraically equivalent for matmul (padded K
// rows are 0 → contribute nothing; padded M/N rows/cols are computed then sliced
// off), though changing the tile shape can change floating-point reduction
// association. The metal-spirv coop path REQUIRES mult-16 M/N/K (mustBeAligned)
// and has no unaligned support, so odd-sized models (e.g. vit: M=B*T=4616)
// otherwise fall to scalar (0.31 vs jax-metal 2.51). Padding makes them aligned
// → they hit the matrix units. The padded matmul is then f32-promoted by
// RaiseAccumulatorPattern (dims now mult-16 and, for vit, >=128). Env-gated
// IREE_METAL_COOP_PAD.
struct PadMatmulToCoopPattern : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = linalgOp.getOperation();
    if (!IREE::LinalgExt::isPureMatmul(op) || linalgOp.hasDynamicShape()) {
      return failure();
    }
    SmallVector<int64_t> ranges = linalgOp.getStaticLoopRanges();
    if (ranges.size() != 3) // M, N, K
      return failure();
    // Only pad if a dim is non-mult-16 (else nothing to do — avoids re-firing on
    // the already-padded matmul, which terminates the greedy rewrite).
    if (llvm::all_of(ranges, [](int64_t d) { return d % 16 == 0; }))
      return failure();
    // Skip tiny matmuls (not worth the coop path / padding overhead).
    if (llvm::any_of(ranges, [](int64_t d) { return d < 16; }))
      return failure();

    // The ViT FFN's flattened row extent M=8*577=4616 is first padded to 4624
    // by the general multiple-of-16 path below. Although that is legal for the
    // cooperative matrix intrinsic, it makes Metal's schedule heuristic fall
    // back to smaller workgroup tiles. Padding the exact BF16 FFN shape family
    // to the next multiple of 64 (4672) selects a larger aligned schedule
    // family for all six forward/backward contractions. Keep this deliberately
    // exact and opt-in while it is correctness/performance gated on the
    // complete model.
    bool padViTFFNTo64 = false;
    if (const char *value = getenv("IREE_METAL_FFN_PAD_M64")) {
      SmallVector<int64_t> sortedRanges = ranges;
      llvm::sort(sortedRanges);
      padViTFFNTo64 =
          StringRef(value) == "1" &&
          sortedRanges == SmallVector<int64_t>({768, 3072, 4616}) &&
          llvm::all_of(linalgOp.getDpsInputs(), [](Value input) {
            return cast<ShapedType>(input.getType()).getElementType().isBF16();
          });
    }

    linalg::LinalgPaddingOptions options;
    SmallVector<Attribute> padValues;
    for (Value operand : op->getOperands()) {
      Type et = cast<ShapedType>(operand.getType()).getElementType();
      padValues.push_back(rewriter.getZeroAttr(et));
    }
    options.setPaddingValues(padValues);
    options.setPaddingDimensions({0, 1, 2});
    options.setPadToMultipleOf(padViTFFNTo64
                                   ? SmallVector<int64_t>({64, 64, 64})
                                   : SmallVector<int64_t>({16, 16, 16}));
    // Return the unpadded result via extract_slice; don't materialize a copy back
    // into the (differently-shaped) original destination.
    options.setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);

    linalg::LinalgOp paddedOp;
    SmallVector<Value> replacements;
    SmallVector<tensor::PadOp> padOps;
    if (failed(linalg::rewriteAsPaddedOp(rewriter, linalgOp, options, paddedOp,
                                         replacements, padOps))) {
      return failure();
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

// Pad attention batch-matmuls (QK^T, AV; rank-4 loops: batch,M,N,K) to mult-16
// so odd-sequence models (e.g. vit seq=577) reach the matrix units instead of
// running scalar. Per-op slice-back (CopyBackOp::None) returns the original
// [.,M,N] shape, so the fake padded rows/cols are dropped before the softmax
// consumer -> numerically exact IF the slice survives downstream fusion
// (validated on vit attention vs a numpy ref before trusting). Opt-in
// (experimental): IREE_METAL_COOP_PAD_ATTN.
struct PadBatchMatmulToCoopPattern
    : OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp bmm,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = cast<linalg::LinalgOp>(bmm.getOperation());
    if (linalgOp.hasDynamicShape())
      return failure();
    SmallVector<int64_t> ranges = linalgOp.getStaticLoopRanges();
    if (ranges.size() != 4) // batch, M, N, K
      return failure();
    // Only the M(1)/N(2)/K(3) loops need mult-16 for coop; the batch loop (0)
    // is left alone. No-op once aligned (so the greedy rewrite terminates).
    if (ranges[1] % 16 == 0 && ranges[2] % 16 == 0 && ranges[3] % 16 == 0)
      return failure();
    if (ranges[1] < 16 || ranges[2] < 16 || ranges[3] < 16)
      return failure();

    linalg::LinalgPaddingOptions options;
    SmallVector<Attribute> padValues;
    for (Value operand : bmm->getOperands()) {
      Type et = cast<ShapedType>(operand.getType()).getElementType();
      padValues.push_back(rewriter.getZeroAttr(et));
    }
    options.setPaddingValues(padValues);
    options.setPaddingDimensions({1, 2, 3});
    options.setPadToMultipleOf({16, 16, 16});
    options.setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);

    linalg::LinalgOp paddedOp;
    SmallVector<Value> replacements;
    SmallVector<tensor::PadOp> padOps;
    if (failed(linalg::rewriteAsPaddedOp(rewriter, linalgOp, options, paddedOp,
                                         replacements, padOps))) {
      return failure();
    }
    rewriter.replaceOp(bmm, replacements);
    return success();
  }
};

class RaiseContractionAccumulatorToF32Pass
    : public impl::RaiseContractionAccumulatorToF32PassBase<
          RaiseContractionAccumulatorToF32Pass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<RaiseAccumulatorPattern>(ctx);
    // IREE_METAL_COOP_PAD (default ON): pad non-mult-16 matmuls to mult-16 so
    // odd-sized models (vit M=B*T=4616: 0.31->0.66, 2.1×) reach the matrix units
    // instead of scalar. Validated exact (identical rel_err with/without padding)
    // and zero-regression on aligned models (gpt2 3.49, bert 2.94 unchanged — pad
    // no-ops when all dims are already mult-16). Opt out with IREE_METAL_COOP_NO_PAD.
    if (!getenv("IREE_METAL_COOP_NO_PAD")) {
      patterns.add<PadMatmulToCoopPattern>(ctx);
    }
    // IREE_METAL_COOP_PAD_ATTN (default ON): also coop-pad attention batch-matmuls
    // so odd-seq models (vit seq=577) leave the scalar path (0.21->0.73 TFLOP/s,
    // 3.5x; loss rel 5.7e-4 vs scalar ref — per-op slice-back is exact, no mask).
    // No-op / zero-regression on mult-16 models (10-model sweep: 8/8 aligned
    // models byte-identical flag on==off). Opt out with IREE_METAL_COOP_NO_PAD_ATTN.
    if (!getenv("IREE_METAL_COOP_NO_PAD_ATTN")) {
      patterns.add<PadBatchMatmulToCoopPattern>(ctx);
    }
    // IREE_METAL_COOP_SPLIT_TRANSPOSE (default ON when this pass runs): isolate
    // transpose-epilogue matmuls so the backward weight-grads reach the matrix
    // units. Opt out with IREE_METAL_COOP_NO_SPLIT_TRANSPOSE.
    if (!getenv("IREE_METAL_COOP_NO_SPLIT_TRANSPOSE")) {
      patterns.add<SplitTransposeEpiloguePattern>(ctx);
    }
    // IsolateBatchMatmulPattern isolates attention batch-matmuls from their
    // epilogue. bf16 isolation is DEFAULT-ON (the NaN correctness fix; opt out
    // per-pattern via IREE_METAL_COOP_NO_ATTN_ISOLATE_BF16). f32 isolation only has
    // an effect under IREE_METAL_COOP_ATTN_SPLIT (which produces f32 bmms). Opt the
    // whole pattern out with IREE_METAL_COOP_NO_ATTN_ISOLATE.
    if (!getenv("IREE_METAL_COOP_NO_ATTN_ISOLATE")) {
      patterns.add<IsolateBatchMatmulPattern>(ctx);
    }
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
