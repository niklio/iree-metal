// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-tiling-level"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYTILINGLEVELPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUApplyTilingLevelPass final
    : impl::GPUApplyTilingLevelPassBase<GPUApplyTilingLevelPass> {
  using GPUApplyTilingLevelPassBase::GPUApplyTilingLevelPassBase;
  void runOnOperation() override;
};
} // namespace

static constexpr StringLiteral kAppleAttentionBackwardRole =
    "iree_codegen.apple_attention_backward_role";
static constexpr StringLiteral kAppleAttentionBackwardCausal =
    "iree_codegen.apple_attention_backward_causal";
static constexpr StringLiteral kAppleAttentionBackwardCausalScoreAlignment =
    "iree_codegen.apple_attention_backward_causal_score_alignment";
static constexpr StringLiteral
    kAppleAttentionBackwardCausalScoreWorkgroupAligned =
        "iree_codegen.apple_attention_backward_causal_score_workgroup_aligned";
static constexpr StringLiteral kAppleAttentionCausal =
    "iree_codegen.apple_attention_causal";

// Returns the workgroup/reduction-tiled slice of the score tensor consumed by
// a native attention-backward contraction. Apple operand promotion inserts a
// linalg.copy between that slice and the contraction.
static tensor::ExtractSliceOp
getAttentionBackwardScoreSlice(linalg::LinalgOp op) {
  if (op.getNumDpsInputs() == 0) {
    return {};
  }
  Value score = op.getDpsInputOperand(0)->get();
  if (auto copy = score.getDefiningOp<linalg::CopyOp>()) {
    score = copy.getDpsInputOperand(0)->get();
  }
  return score.getDefiningOp<tensor::ExtractSliceOp>();
}

// With finite inputs, causal score entries are nonzero only when query >= key.
// Preserve the existing full-size reduction tiles, but skip tiles that are
// wholly outside that triangle:
//   dQ rows [m0,m1): keys [0, ceil(m1/R)*R)
//   dK/dV keys [k0,k1): queries [floor(k0/R)*R, T)
// Outward alignment retains the diagonal/boundary tile, keeps every promoted
// operand shape static, and makes the bound uniform across the workgroup.
//
// This is intentionally opt-in: strict IEEE behavior differs for masked
// zero multiplied by NaN/Inf. The Apple model benchmark enables the pass only
// under its finite-value contract.
static LogicalResult applyCausalAttentionBackwardReductionShortening(
    FunctionOpInterface funcOp, IRRewriter &rewriter) {
  const char *triangularGridValue =
      std::getenv("IREE_METAL_CAUSAL_TRIANGULAR_GRID");
  bool triangularGridEnabled =
      triangularGridValue && StringRef(triangularGridValue) == "1";
  DominanceInfo dominance(funcOp);
  SmallVector<linalg::LinalgOp> causalContractions;
  funcOp->walk([&](linalg::LinalgOp op) {
    if (op->hasAttr(kAppleAttentionBackwardCausal) &&
        op->hasAttr(kAppleAttentionBackwardRole)) {
      causalContractions.push_back(op);
    }
  });

  for (linalg::LinalgOp op : causalContractions) {
    auto role = op->getAttrOfType<StringAttr>(kAppleAttentionBackwardRole);
    bool isDQ = role && role.getValue() == "dq_attrs";
    bool isKeyOwned =
        role && (role.getValue() == "dk_attrs" ||
                 role.getValue() == "dv_attrs");
    if (!isDQ && !isKeyOwned) {
      return op.emitOpError(
          "causal attention-backward marker requires a dQ/dK/dV role");
    }

    scf::ForOp reductionLoop = op->getParentOfType<scf::ForOp>();
    tensor::ExtractSliceOp scoreSlice =
        getAttentionBackwardScoreSlice(op);
    if (!reductionLoop || !scoreSlice) {
      return op.emitOpError(
          "failed to find the tiled causal score reduction");
    }
    auto scoreType =
        dyn_cast<RankedTensorType>(scoreSlice.getSource().getType());
    if (!scoreType || scoreType.getRank() < 2) {
      return op.emitOpError("expected a ranked causal score tensor");
    }

    FailureOr<linalg::ContractionDimensions> contractionDims =
        linalg::inferContractionDims(op);
    if (failed(contractionDims) || contractionDims->m.size() != 1 ||
        contractionDims->k.size() != 1) {
      return op.emitOpError(
          "expected one M and one K causal contraction dimension");
    }
    AffineMap scoreMap =
        op.getMatchingIndexingMap(op.getDpsInputOperand(0));
    MLIRContext *context = op.getContext();
    std::optional<unsigned> outputSequenceDim = scoreMap.getResultPosition(
        getAffineDimExpr(contractionDims->m.front(), context));
    std::optional<unsigned> reductionSequenceDim = scoreMap.getResultPosition(
        getAffineDimExpr(contractionDims->k.front(), context));
    SmallVector<OpFoldResult> offsets = scoreSlice.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = scoreSlice.getMixedSizes();
    SmallVector<OpFoldResult> strides = scoreSlice.getMixedStrides();
    if (!outputSequenceDim || !reductionSequenceDim ||
        scoreMap.getNumResults() != offsets.size() ||
        offsets.size() != sizes.size() || offsets.size() != strides.size() ||
        getConstantIntValue(strides[*outputSequenceDim]) != 1 ||
        getConstantIntValue(strides[*reductionSequenceDim]) != 1) {
      return op.emitOpError(
          "expected a non-rank-reduced unit-stride causal score slice");
    }
    auto reductionOffset =
        dyn_cast<Value>(offsets[*reductionSequenceDim]);
    if (!reductionOffset ||
        reductionOffset != reductionLoop.getInductionVar()) {
      return op.emitOpError(
          "causal score slice is not driven by the reduction loop");
    }
    std::optional<int64_t> reductionTile =
        getConstantIntValue(reductionLoop.getStep());
    if (!reductionTile || *reductionTile <= 0) {
      return op.emitOpError(
          "causal reduction requires a positive static tile size");
    }
    auto scoreAlignment = op->getAttrOfType<IntegerAttr>(
        kAppleAttentionBackwardCausalScoreAlignment);
    bool requireScoreAlignment =
        triangularGridEnabled || scoreAlignment ||
        op->hasAttr(kAppleAttentionBackwardCausalScoreWorkgroupAligned);
    if (requireScoreAlignment && !scoreAlignment) {
      return op.emitOpError(
          "compact causal score grid requires a score alignment contract");
    }
    if (requireScoreAlignment &&
        !op->hasAttr(kAppleAttentionBackwardCausalScoreWorkgroupAligned)) {
      return op.emitOpError(
          "compact causal score grid requires aligned workgroup provenance");
    }
    if (scoreAlignment &&
        (scoreAlignment.getInt() <= 0 ||
         scoreAlignment.getInt() != *reductionTile)) {
      return op.emitOpError(
          "causal score alignment contract does not match the reduction tile");
    }
    if (requireScoreAlignment) {
      int64_t outputExtent = scoreType.getDimSize(*outputSequenceDim);
      int64_t reductionExtent =
          scoreType.getDimSize(*reductionSequenceDim);
      std::optional<int64_t> outputTile =
          getConstantIntValue(sizes[*outputSequenceDim]);
      if (ShapedType::isDynamic(outputExtent) ||
          ShapedType::isDynamic(reductionExtent) ||
          outputExtent != reductionExtent ||
          outputExtent % scoreAlignment.getInt() != 0 || !outputTile ||
          *outputTile <= 0 ||
          scoreAlignment.getInt() % *outputTile != 0) {
        return op.emitOpError(
            "compact causal consumer requires a square aligned score domain "
            "and an output tile dividing the score alignment");
      }
    }
    auto dominatesReductionLoop = [&](OpFoldResult value) {
      auto dynamicValue = dyn_cast<Value>(value);
      return !dynamicValue ||
             dominance.dominates(dynamicValue, reductionLoop.getOperation());
    };
    if (!dominatesReductionLoop(offsets[*outputSequenceDim]) ||
        !dominatesReductionLoop(sizes[*outputSequenceDim])) {
      return op.emitOpError(
          "causal output tile offset and size must dominate the reduction loop");
    }

    rewriter.setInsertionPoint(reductionLoop);
    Location loc = reductionLoop.getLoc();
    Value tile = reductionLoop.getStep();
    Value outputOffset = getValueOrCreateConstantIndexOp(
        rewriter, loc, offsets[*outputSequenceDim]);
    Value oldLowerBound = reductionLoop.getLowerBound();
    if (requireScoreAlignment) {
      std::optional<int64_t> lowerBound =
          getConstantIntValue(oldLowerBound);
      std::optional<int64_t> upperBound =
          getConstantIntValue(reductionLoop.getUpperBound());
      if (!lowerBound || *lowerBound != 0 || !upperBound ||
          *upperBound <= 0 ||
          *upperBound % scoreAlignment.getInt() != 0) {
        return op.emitOpError(
            "compact causal reduction requires a zero-based static domain "
            "aligned to the score contract");
      }
    }
    if (isDQ) {
      Value outputSize = getValueOrCreateConstantIndexOp(
          rewriter, loc, sizes[*outputSequenceDim]);
      Value outputEnd =
          arith::AddIOp::create(rewriter, loc, outputOffset, outputSize);
      Value clampedEnd = arith::MaxUIOp::create(
          rewriter, loc, outputEnd, oldLowerBound);
      Value relativeEnd = arith::SubIOp::create(
          rewriter, loc, clampedEnd, oldLowerBound);
      Value reductionTiles =
          arith::CeilDivUIOp::create(rewriter, loc, relativeEnd, tile);
      Value roundedRelativeEnd =
          arith::MulIOp::create(rewriter, loc, reductionTiles, tile);
      Value roundedEnd = arith::AddIOp::create(
          rewriter, loc, oldLowerBound, roundedRelativeEnd);
      Value newUpperBound = arith::MinUIOp::create(
          rewriter, loc, roundedEnd, reductionLoop.getUpperBound());
      reductionLoop.setUpperBound(newUpperBound);
    } else {
      Value clampedOffset = arith::MaxUIOp::create(
          rewriter, loc, outputOffset, oldLowerBound);
      Value relativeOffset = arith::SubIOp::create(
          rewriter, loc, clampedOffset, oldLowerBound);
      Value reductionTiles =
          arith::DivUIOp::create(rewriter, loc, relativeOffset, tile);
      Value roundedRelativeBegin =
          arith::MulIOp::create(rewriter, loc, reductionTiles, tile);
      Value newLowerBound = arith::AddIOp::create(
          rewriter, loc, oldLowerBound, roundedRelativeBegin);
      reductionLoop.setLowerBound(newLowerBound);
    }
  }
  return success();
}

// Collect the extract_slice ownership chain for an attention operand. The
// Apple attention pipeline may place a linalg.copy around an owner slice when
// promoting the operand; only that transparent spelling is peeled here.
static SmallVector<tensor::ExtractSliceOp>
getAttentionOperandSlices(Value operand) {
  SmallVector<tensor::ExtractSliceOp> slices;
  while (Operation *definingOp = operand.getDefiningOp()) {
    if (auto copy = dyn_cast<linalg::CopyOp>(definingOp)) {
      operand = copy.getDpsInputOperand(0)->get();
      continue;
    }
    auto slice = dyn_cast<tensor::ExtractSliceOp>(definingOp);
    if (!slice) {
      break;
    }
    slices.push_back(slice);
    operand = slice.getSource();
  }
  return slices;
}

// With finite inputs, the exact lower-triangular mask proved by the native
// attention raise makes keys at k > m contribute zero. Keep the existing
// owner/query grid and full-size K2 reduction tiles, but shorten each owner's
// loop to:
//
//   keys [old_lb, min(old_ub,
//       old_lb + ceil((query_end - old_lb) / R) * R))
//
// Clamping query_end to old_lb before subtracting avoids an unsigned underflow.
// Outward alignment preserves the diagonal/boundary tile and all static tile
// shapes. This is intentionally opt-in: skipping masked arithmetic can differ
// from multiplying a zero weight by NaN/Inf, so the benchmark enables it only
// under its finite-input semantic contract.
static LogicalResult applyCausalAttentionForwardReductionShortening(
    FunctionOpInterface funcOp, IRRewriter &rewriter) {
  DominanceInfo dominance(funcOp);
  SmallVector<IREE::LinalgExt::OnlineAttentionOp> causalAttentionOps;
  funcOp->walk([&](IREE::LinalgExt::OnlineAttentionOp op) {
    DictionaryAttr config = op.getDecompositionConfigAttr();
    if (config && config.getAs<UnitAttr>(kAppleAttentionCausal)) {
      causalAttentionOps.push_back(op);
    }
  });

  llvm::SmallDenseSet<Operation *> shortenedLoops;
  for (IREE::LinalgExt::OnlineAttentionOp op : causalAttentionOps) {
    auto opInfo = IREE::LinalgExt::AttentionOpDetail::get(
        op.getQueryMap(), op.getKeyMap(), op.getValueMap(), op.getOutputMap());
    if (failed(opInfo) || opInfo->getMDims().size() != 1 ||
        opInfo->getK2Dims().size() != 1) {
      return op.emitOpError(
          "causal forward shortening requires exactly one M and one K2 "
          "dimension");
    }

    scf::ForOp reductionLoop = op->getParentOfType<scf::ForOp>();
    if (!reductionLoop) {
      return op.emitOpError("failed to find the tiled causal K2 reduction");
    }
    if (!shortenedLoops.insert(reductionLoop.getOperation()).second) {
      return op.emitOpError(
          "expected exactly one marked online attention per K2 loop");
    }

    MLIRContext *context = op.getContext();
    std::optional<unsigned> querySequenceDim =
        op.getQueryMap().getResultPosition(
            getAffineDimExpr(opInfo->getMDims().front(), context));
    std::optional<unsigned> keySequenceDim = op.getKeyMap().getResultPosition(
        getAffineDimExpr(opInfo->getK2Dims().front(), context));
    if (!querySequenceDim || !keySequenceDim) {
      return op.emitOpError(
          "failed to map the causal M and K2 dimensions to Q and K");
    }

    SmallVector<tensor::ExtractSliceOp> querySlices =
        getAttentionOperandSlices(op.getQuery());
    SmallVector<tensor::ExtractSliceOp> keySlices =
        getAttentionOperandSlices(op.getKey());
    if (querySlices.empty() || keySlices.empty()) {
      return op.emitOpError(
          "causal Q and K operands must retain extract_slice ownership");
    }

    // Reduction tiling may add an identity Q slice inside the pre-existing
    // workgroup owner slice. The outermost slice carries the owner's M offset.
    tensor::ExtractSliceOp queryOwnerSlice = querySlices.back();
    SmallVector<OpFoldResult> queryOffsets = queryOwnerSlice.getMixedOffsets();
    SmallVector<OpFoldResult> querySizes = queryOwnerSlice.getMixedSizes();
    SmallVector<OpFoldResult> queryStrides = queryOwnerSlice.getMixedStrides();
    if (queryOffsets.size() != querySizes.size() ||
        queryOffsets.size() != queryStrides.size() ||
        op.getQueryMap().getNumResults() != queryOffsets.size() ||
        *querySequenceDim >= queryOffsets.size() ||
        getConstantIntValue(queryStrides[*querySequenceDim]) != 1) {
      return op.emitOpError(
          "expected a non-rank-reduced unit-stride causal Q owner slice");
    }

    // Find the K slice introduced by this reduction loop. This also proves
    // that the selected scf.for owns K2 rather than another reduction.
    tensor::ExtractSliceOp keyReductionSlice;
    for (tensor::ExtractSliceOp slice : keySlices) {
      SmallVector<OpFoldResult> offsets = slice.getMixedOffsets();
      if (*keySequenceDim >= offsets.size()) {
        continue;
      }
      auto offset = dyn_cast<Value>(offsets[*keySequenceDim]);
      if (offset && offset == reductionLoop.getInductionVar()) {
        keyReductionSlice = slice;
        break;
      }
    }
    if (!keyReductionSlice) {
      return op.emitOpError(
          "causal K slice is not driven by the K2 reduction loop");
    }
    SmallVector<OpFoldResult> keyOffsets = keyReductionSlice.getMixedOffsets();
    SmallVector<OpFoldResult> keySizes = keyReductionSlice.getMixedSizes();
    SmallVector<OpFoldResult> keyStrides = keyReductionSlice.getMixedStrides();
    if (keyOffsets.size() != keySizes.size() ||
        keyOffsets.size() != keyStrides.size() ||
        op.getKeyMap().getNumResults() != keyOffsets.size() ||
        *keySequenceDim >= keyOffsets.size() ||
        getConstantIntValue(keyStrides[*keySequenceDim]) != 1) {
      return op.emitOpError(
          "expected a non-rank-reduced unit-stride causal K2 slice");
    }

    auto querySourceType =
        dyn_cast<RankedTensorType>(queryOwnerSlice.getSource().getType());
    auto keySourceType =
        dyn_cast<RankedTensorType>(keyReductionSlice.getSource().getType());
    if (!querySourceType || !keySourceType ||
        *querySequenceDim >= static_cast<unsigned>(querySourceType.getRank()) ||
        *keySequenceDim >= static_cast<unsigned>(keySourceType.getRank())) {
      return op.emitOpError("expected ranked causal Q and K owner tensors");
    }
    int64_t querySequence = querySourceType.getDimSize(*querySequenceDim);
    int64_t keySequence = keySourceType.getDimSize(*keySequenceDim);
    if (ShapedType::isDynamic(querySequence) || querySequence <= 0 ||
        querySequence != keySequence) {
      return op.emitOpError(
          "causal Q and K owner slices must share a positive static sequence");
    }

    std::optional<int64_t> reductionTile =
        getConstantIntValue(reductionLoop.getStep());
    if (!reductionTile || *reductionTile <= 0) {
      return op.emitOpError(
          "causal K2 reduction requires a positive static tile size");
    }
    auto dominatesReductionLoop = [&](OpFoldResult value) {
      auto dynamicValue = dyn_cast<Value>(value);
      return !dynamicValue ||
             dominance.dominates(dynamicValue, reductionLoop.getOperation());
    };
    OpFoldResult queryOffset = queryOffsets[*querySequenceDim];
    OpFoldResult querySize = querySizes[*querySequenceDim];
    if (!dominatesReductionLoop(queryOffset) ||
        !dominatesReductionLoop(querySize)) {
      return op.emitOpError(
          "causal Q owner offset and size must dominate the K2 loop");
    }

    rewriter.setInsertionPoint(reductionLoop);
    Location loc = reductionLoop.getLoc();
    Value oldLowerBound = reductionLoop.getLowerBound();
    Value oldUpperBound = reductionLoop.getUpperBound();
    Value tile = reductionLoop.getStep();
    Value ownerOffset =
        getValueOrCreateConstantIndexOp(rewriter, loc, queryOffset);
    Value ownerSize = getValueOrCreateConstantIndexOp(rewriter, loc, querySize);
    Value ownerEnd =
        arith::AddIOp::create(rewriter, loc, ownerOffset, ownerSize);
    Value clampedEnd =
        arith::MaxUIOp::create(rewriter, loc, ownerEnd, oldLowerBound);
    Value relativeEnd =
        arith::SubIOp::create(rewriter, loc, clampedEnd, oldLowerBound);
    Value reductionTiles =
        arith::CeilDivUIOp::create(rewriter, loc, relativeEnd, tile);
    Value roundedRelativeEnd =
        arith::MulIOp::create(rewriter, loc, reductionTiles, tile);
    Value roundedEnd =
        arith::AddIOp::create(rewriter, loc, oldLowerBound, roundedRelativeEnd);
    Value newUpperBound =
        arith::MinUIOp::create(rewriter, loc, roundedEnd, oldUpperBound);
    reductionLoop.setUpperBound(newUpperBound);
  }
  return success();
}

static llvm::SmallDenseSet<TilingInterface>
getTiledOps(Operation *funcOp, IREE::GPU::TilingLevel tilingLevel) {
  llvm::SmallDenseSet<TilingInterface> targets;
  unsigned opaqueLevel = llvm::to_underlying(tilingLevel);
  funcOp->walk([&](TilingInterface target) {
    // TODO: This would probably be easier with a lowering config interface
    // method that checks whether a particular level is tiled.
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(target)) {
      if (loweringConfig.hasTilingLevel(opaqueLevel)) {
        targets.insert(target);
      }
    }
  });
  return targets;
}

void GPUApplyTilingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  if (!llvm::is_contained({IREE::GPU::TilingLevel::Reduction,
                           IREE::GPU::TilingLevel::Thread,
                           IREE::GPU::TilingLevel::Subgroup,
                           IREE::GPU::TilingLevel::PartialReduction,
                           IREE::GPU::TilingLevel::Serial},
                          tilingLevel)) {
    funcOp.emitError() << "unsupported tiling level: "
                       << IREE::GPU::stringifyEnum(tilingLevel) << "\n";
    return signalPassFailure();
  }

  llvm::SmallDenseSet<TilingInterface> targetOps =
      getTiledOps(funcOp, tilingLevel);

  IRRewriter rewriter(funcOp);
  if (failed(applyTileAndFuseToEachRoot(
          rewriter, targetOps, tilingLevel, allowZeroSlices,
          /*targetTileMap=*/std::nullopt, fuseConsumers))) {
    funcOp.emitError() << "tiling of level "
                       << IREE::GPU::stringifyEnum(tilingLevel) << " failed\n";
    return signalPassFailure();
  }
  if (tilingLevel == IREE::GPU::TilingLevel::Reduction &&
      shortenCausalAttentionBackwardReductions &&
      failed(applyCausalAttentionBackwardReductionShortening(funcOp,
                                                             rewriter))) {
    return signalPassFailure();
  }
  if (tilingLevel == IREE::GPU::TilingLevel::Reduction &&
      shortenCausalAttentionForwardReductions &&
      failed(applyCausalAttentionForwardReductionShortening(funcOp,
                                                            rewriter))) {
    return signalPassFailure();
  }

  MLIRContext *context = &getContext();

  // Swap `collapse_shape` with `extract_slice` to enable more loop fusion
  // opportunity. Currently this is only needed for convolution IGEMM path.
  // TODO(vivian): Move the pattern to `GPUFuseAndHoistParallelLoopsPass`.
  if (normalizeLoops) {
    funcOp->walk(
        [&](scf::ForOp forOp) { (void)normalizeLoopBounds(rewriter, forOp); });
    funcOp->walk([&](scf::ForallOp forallOp) {
      (void)normalizeLoopBounds(rewriter, forallOp);
    });

    RewritePatternSet patterns(context);
    populateSwapExtractWithCollapsePattern(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // Apply cleanup patterns.
  {
    RewritePatternSet patterns(context);
    IREE::GPU::populateFoldSwizzleHintOpPatterns(patterns);
    // Merge consecutive insert/extract slice ops to simplify later loop
    // hoisting patterns.
    tensor::populateFoldTensorEmptyPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "tiling cleanup failed\n";
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
