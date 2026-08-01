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
