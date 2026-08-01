// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileAndFuseUtils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstdlib>

#define DEBUG_TYPE "tile-and-distribute-to-workgroups-using-forall-op"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TILEANDDISTRIBUTETOWORKGROUPSUSINGFORALLOPPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct TileAndDistributeToWorkgroupsUsingForallOpPass final
    : impl::TileAndDistributeToWorkgroupsUsingForallOpPassBase<
          TileAndDistributeToWorkgroupsUsingForallOpPass> {
  explicit TileAndDistributeToWorkgroupsUsingForallOpPass(
      bool transposeWorkgroup) {
    this->transposeWorkgroup = transposeWorkgroup;
  }
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

/// Find the lowering config to use for getting the tile sizes.
// TODO: For now this is taking the "last op" in the dispatch, but
// ideally this should take the "root op" that gets tiled and everything
// gets fused with it. For now to keep consistent with the legacy
// tile-and-distribute it is still looking for the "last compute operation".
struct TilingInfo {
  Operation *tilableOp;
  SmallVector<OpFoldResult> tileSizes;
  SmallVector<int64_t> interchange;
};

static FailureOr<TilingInfo>
getTiledAndDistributionInfo(RewriterBase &rewriter,
                            ArrayRef<Operation *> computeOps) {
  // TODO: It is expected that at most one compute op has a workgroup tiling
  // level. Currently, it selects the last compute op that has workgroup tiling
  // level.
  Operation *tilableOp = nullptr;
  for (Operation *op : llvm::reverse(computeOps)) {
    if (getLoweringConfig(op)) {
      if (!getLoweringConfig(op).hasWorkgroupTilingLevel()) {
        continue;
      }
      tilableOp = op;
      break;
    }
  }
  if (!tilableOp) {
    // There is no lowering config. Return `null`.
    return TilingInfo{nullptr, {}, {}};
  }

  IREE::Codegen::LoweringConfigAttrInterface tilableOpConfig =
      getLoweringConfig(tilableOp);
  if (!tilableOpConfig) {
    return tilableOp->emitOpError("unable to find configuration of root op to "
                                  "define workgroup count region");
  }
  auto tileSizes = llvm::map_to_vector(
      tilableOpConfig.getWorkgroupTileSizes(),
      [&](int64_t t) -> OpFoldResult { return rewriter.getIndexAttr(t); });
  SmallVector<int64_t> interchange = tilableOpConfig.getWorkgroupInterchange();

  // Avoid distributing unit-trip count loops.

  // Set tile sizes for non-partitioned loops to zero.
  if (auto partitionableLoopsInterface =
          dyn_cast<PartitionableLoopsInterface>(tilableOp)) {
    SmallVector<unsigned> partitionableLoops =
        partitionableLoopsInterface.getPartitionableLoops(std::nullopt);
    llvm::SmallDenseSet<unsigned> partitionableLoopsSet(
        partitionableLoops.begin(), partitionableLoops.end());
    OpFoldResult zero = rewriter.getIndexAttr(0);
    for (auto loopId : llvm::seq<unsigned>(0, tileSizes.size())) {
      if (partitionableLoopsSet.count(loopId)) {
        continue;
      }
      tileSizes[loopId] = zero;
    }
  }

  // Set tile sizes for full tiles to zero. This prevents single trip loops from
  // being created, which can sometimes block certain cleanup patterns from
  // applying during producer fusion.
  if (auto tilingInterfaceOp = dyn_cast<TilingInterface>(tilableOp)) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(tilingInterfaceOp);
    SmallVector<Range> bounds = tilingInterfaceOp.getIterationDomain(rewriter);
    SmallVector<int64_t> staticLoopSizes;
    SmallVector<Value> d;
    for (Range bound : bounds) {
      dispatchIndexOpFoldResult(bound.size, d, staticLoopSizes);
    }
    OpFoldResult zero = rewriter.getIndexAttr(0);
    SmallVector<int64_t> tileSizesInt = tilableOpConfig.getWorkgroupTileSizes();
    int numNonZero = tileSizesInt.size() - llvm::count(tileSizesInt, 0);
    for (auto loopId :
         llvm::reverse(llvm::seq<unsigned>(0, tileSizesInt.size()))) {
      // Do not set all sizes to 0, or else the distribution loop will not be
      // created.
      if (numNonZero <= 1) {
        break;
      }
      if (loopId < staticLoopSizes.size() &&
          staticLoopSizes[loopId] == tileSizesInt[loopId]) {
        tileSizes[loopId] = zero;
        --numNonZero;
      }
    }
  }

  return TilingInfo{tilableOp, tileSizes, interchange};
}

/// Helper function to return the mapping attribute to use given the tile sizes.
static SmallVector<Attribute> getMapping(MLIRContext *context,
                                         ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<Attribute> mapping;
  mapping.reserve(tileSizes.size());
  for (auto tileSize : llvm::reverse(tileSizes)) {
    if (isZeroInteger(tileSize)) {
      continue;
    }
    uint64_t currSize = mapping.size();
    switch (currSize) {
    case 0:
    case 1:
    case 2:
      mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
          context, IREE::Codegen::symbolizeWorkgroupId(currSize).value()));
      break;
    default:
      mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
          context, IREE::Codegen::WorkgroupId::IdZ, currSize - 2));
    }
  }
  return llvm::to_vector(llvm::reverse(mapping));
}

static constexpr StringLiteral kAppleAttentionBackwardRole =
    "iree_codegen.apple_attention_backward_role";
static constexpr StringLiteral kAppleAttentionBackwardCausalScore =
    "iree_codegen.apple_attention_backward_causal_score";
static constexpr StringLiteral kAppleAttentionBackwardCausalScoreAlignment =
    "iree_codegen.apple_attention_backward_causal_score_alignment";

static bool useCausalAttentionTriangularGrid(Operation *op) {
  const char *value = std::getenv("IREE_METAL_CAUSAL_TRIANGULAR_GRID");
  if (!value || StringRef(value) != "1" ||
      !op->hasAttr(kAppleAttentionBackwardCausalScore)) {
    return false;
  }
  auto role = op->getAttrOfType<StringAttr>(kAppleAttentionBackwardRole);
  return role &&
         (role.getValue() == "qk_attrs" || role.getValue() == "dp_attrs");
}

// Creates a genuinely compact launch grid for a static lower-triangular score
// contraction. The original square row/column workgroup dimensions are
// replaced by one X-mapped triangular ordinal. A unit Y-mapped dimension keeps
// the workgroup mapping well-formed, while all pre-existing batch dimensions
// retain their Z delinearization.
//
// The score grid is compacted in square macrotiles whose size matches the
// outward-aligned reduction tiles consumed by dQ/dK/dV. Every producer tile in
// a diagonal macrotile is retained, including its fine-grained upper triangle:
// those values are read by the consumers even though they are mathematically
// masked. For a macro-grid ordinal `u`, the mapping is:
//
//   macro_row = max r where r * (r + 1) / 2 <= u
//   macro_col = u - macro_row * (macro_row + 1) / 2
//   row = macro_row * rows_per_macro + intra_macro_row
//   col = macro_col * cols_per_macro + intra_macro_col
//
// Static select thresholds avoid a floating-point inverse-triangle operation
// in the shader and make the bijection straightforward to inspect.
static FailureOr<scf::SCFTilingOptions::CustomLoopHeaderInfo>
generateCausalAttentionTriangularGridHeader(
    RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> givenTileSizes, ValueRange outerDestinationTensors,
    unsigned rowDim, unsigned colDim, Operation *rootOp) {
  if (loopRanges.size() != givenTileSizes.size() ||
      rowDim >= loopRanges.size() || colDim >= loopRanges.size() ||
      rowDim == colDim) {
    return rootOp->emitOpError(
        "invalid loop domain for causal triangular workgroup grid");
  }

  auto requireStaticUnitRange =
      [&](unsigned dim) -> FailureOr<std::tuple<int64_t, int64_t>> {
    std::optional<int64_t> offset = getConstantIntValue(loopRanges[dim].offset);
    std::optional<int64_t> size = getConstantIntValue(loopRanges[dim].size);
    std::optional<int64_t> stride = getConstantIntValue(loopRanges[dim].stride);
    std::optional<int64_t> tile = getConstantIntValue(givenTileSizes[dim]);
    if (!offset || *offset != 0 || !size || *size <= 0 || !stride ||
        *stride != 1 || !tile || *tile <= 0 || *size % *tile != 0) {
      return rootOp->emitOpError(
          "causal triangular workgroup grid requires a static, zero-based, "
          "unit-stride, exactly tiled domain");
    }
    return std::tuple<int64_t, int64_t>{*size, *tile};
  };

  FailureOr<std::tuple<int64_t, int64_t>> rowShape =
      requireStaticUnitRange(rowDim);
  FailureOr<std::tuple<int64_t, int64_t>> colShape =
      requireStaticUnitRange(colDim);
  if (failed(rowShape) || failed(colShape)) {
    return failure();
  }
  auto [rowSize, rowTile] = *rowShape;
  auto [colSize, colTile] = *colShape;
  auto scoreAlignmentAttr = rootOp->getAttrOfType<IntegerAttr>(
      kAppleAttentionBackwardCausalScoreAlignment);
  if (!scoreAlignmentAttr || scoreAlignmentAttr.getInt() <= 0) {
    return rootOp->emitOpError(
        "causal triangular workgroup grid requires a positive score "
        "alignment contract");
  }
  int64_t scoreAlignment = scoreAlignmentAttr.getInt();
  if (rowSize != colSize || rowSize % scoreAlignment != 0 ||
      scoreAlignment % rowTile != 0 || scoreAlignment % colTile != 0) {
    return rootOp->emitOpError(
        "causal triangular workgroup grid requires a square score domain "
        "exactly tiled by the contracted score alignment");
  }
  int64_t rowsPerMacro = scoreAlignment / rowTile;
  int64_t colsPerMacro = scoreAlignment / colTile;
  int64_t coarseTileCount = colSize / scoreAlignment;
  int64_t tilesPerMacro;
  int64_t coarseTileCountPlusOne;
  int64_t triangularCount;
  if (llvm::MulOverflow(rowsPerMacro, colsPerMacro, tilesPerMacro) ||
      llvm::AddOverflow(coarseTileCount, int64_t{1},
                        coarseTileCountPlusOne) ||
      llvm::MulOverflow(tilesPerMacro, coarseTileCount, triangularCount) ||
      llvm::MulOverflow(triangularCount, coarseTileCountPlusOne,
                        triangularCount)) {
    return rootOp->emitOpError(
        "causal triangular workgroup grid count overflows i64");
  }
  triangularCount /= 2;

  SmallVector<unsigned> preservedDims;
  for (unsigned dim = 0; dim < loopRanges.size(); ++dim) {
    if (dim == rowDim || dim == colDim || isZeroInteger(givenTileSizes[dim])) {
      continue;
    }
    if (failed(requireStaticUnitRange(dim))) {
      return failure();
    }
    preservedDims.push_back(dim);
  }

  SmallVector<OpFoldResult> lowerBounds;
  SmallVector<OpFoldResult> upperBounds;
  SmallVector<OpFoldResult> steps;
  SmallVector<Attribute> mapping;
  lowerBounds.reserve(preservedDims.size() + 2);
  upperBounds.reserve(preservedDims.size() + 2);
  steps.reserve(preservedDims.size() + 2);
  mapping.reserve(preservedDims.size() + 2);

  for (auto [index, dim] : llvm::enumerate(preservedDims)) {
    lowerBounds.push_back(loopRanges[dim].offset);
    upperBounds.push_back(loopRanges[dim].size);
    steps.push_back(givenTileSizes[dim]);
    int64_t delinearizedDim = preservedDims.size() - index - 1;
    mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
        rewriter.getContext(), IREE::Codegen::WorkgroupId::IdZ,
        delinearizedDim));
  }
  OpFoldResult zero = rewriter.getIndexAttr(0);
  OpFoldResult one = rewriter.getIndexAttr(1);
  lowerBounds.append({zero, zero});
  upperBounds.append({one, rewriter.getIndexAttr(triangularCount)});
  steps.append({one, one});
  mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
      rewriter.getContext(), IREE::Codegen::WorkgroupId::IdY));
  mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
      rewriter.getContext(), IREE::Codegen::WorkgroupId::IdX));
  if (failed(IREE::Codegen::WorkgroupMappingAttr::verifyAttrList(
          rewriter.getContext(), loc, mapping))) {
    return failure();
  }

  auto forallOp = scf::ForallOp::create(rewriter, loc, lowerBounds, upperBounds,
                                        steps, outerDestinationTensors,
                                        rewriter.getArrayAttr(mapping));
  rewriter.setInsertionPoint(forallOp.getTerminator());
  SmallVector<Value> inductionVars = forallOp.getInductionVars();
  Value ordinal = inductionVars.back();

  Value coarseOrdinal = ordinal;
  Value intraMacroOrdinal = arith::ConstantIndexOp::create(rewriter, loc, 0);
  if (tilesPerMacro != 1) {
    Value ratio =
        arith::ConstantIndexOp::create(rewriter, loc, tilesPerMacro);
    coarseOrdinal =
        arith::DivUIOp::create(rewriter, loc, ordinal, ratio).getResult();
    intraMacroOrdinal =
        arith::RemUIOp::create(rewriter, loc, ordinal, ratio).getResult();
  }
  Value coarseRow = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value coarseRowBase = coarseRow;
  for (int64_t candidateRow = 1; candidateRow < coarseTileCount;
       ++candidateRow) {
    int64_t candidateBase = candidateRow * (candidateRow + 1) / 2;
    Value base = arith::ConstantIndexOp::create(rewriter, loc, candidateBase);
    Value isAtOrPastRow = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::uge, coarseOrdinal, base);
    Value candidate =
        arith::ConstantIndexOp::create(rewriter, loc, candidateRow);
    coarseRow = arith::SelectOp::create(rewriter, loc, isAtOrPastRow, candidate,
                                        coarseRow);
    coarseRowBase = arith::SelectOp::create(rewriter, loc, isAtOrPastRow, base,
                                            coarseRowBase);
  }
  Value coarseCol =
      arith::SubIOp::create(rewriter, loc, coarseOrdinal, coarseRowBase);

  Value intraMacroRow = intraMacroOrdinal;
  Value intraMacroCol = arith::ConstantIndexOp::create(rewriter, loc, 0);
  if (colsPerMacro != 1) {
    Value columnCount =
        arith::ConstantIndexOp::create(rewriter, loc, colsPerMacro);
    intraMacroRow =
        arith::DivUIOp::create(rewriter, loc, intraMacroOrdinal, columnCount);
    intraMacroCol =
        arith::RemUIOp::create(rewriter, loc, intraMacroOrdinal, columnCount);
  }
  auto addIntraMacroIndex = [&](Value coarseIndex, int64_t tilesPerDimension,
                                Value intraIndex) {
    if (tilesPerDimension == 1) {
      return coarseIndex;
    }
    Value scale = arith::ConstantIndexOp::create(rewriter, loc,
                                                 tilesPerDimension);
    Value coarseStart =
        arith::MulIOp::create(rewriter, loc, coarseIndex, scale);
    return arith::AddIOp::create(rewriter, loc, coarseStart, intraIndex)
        .getResult();
  };
  Value row = addIntraMacroIndex(coarseRow, rowsPerMacro, intraMacroRow);
  Value col = addIntraMacroIndex(coarseCol, colsPerMacro, intraMacroCol);

  SmallVector<Value> preservedInductionVars(loopRanges.size());
  for (auto [index, dim] : llvm::enumerate(preservedDims)) {
    preservedInductionVars[dim] = inductionVars[index];
  }
  SmallVector<OpFoldResult> offsets(loopRanges.size());
  SmallVector<OpFoldResult> sizes(loopRanges.size());
  for (unsigned dim = 0; dim < loopRanges.size(); ++dim) {
    if (isZeroInteger(givenTileSizes[dim])) {
      offsets[dim] = loopRanges[dim].offset;
      sizes[dim] = loopRanges[dim].size;
      continue;
    }
    Value tile =
        getValueOrCreateConstantIndexOp(rewriter, loc, givenTileSizes[dim]);
    if (dim == rowDim || dim == colDim) {
      Value tileIndex = dim == rowDim ? row : col;
      offsets[dim] =
          arith::MulIOp::create(rewriter, loc, tileIndex, tile).getResult();
    } else {
      offsets[dim] = preservedInductionVars[dim];
    }
    sizes[dim] = givenTileSizes[dim];
  }

  SmallVector<Value> regionOutArgs;
  for (auto argument : forallOp.getRegionOutArgs()) {
    regionOutArgs.push_back(argument);
  }
  return scf::SCFTilingOptions::CustomLoopHeaderInfo{
      {cast<LoopLikeOpInterface>(forallOp.getOperation())},
      offsets,
      sizes,
      regionOutArgs};
}

static LogicalResult generateCausalAttentionTriangularGridTerminator(
    RewriterBase &rewriter, Location loc, ArrayRef<LoopLikeOpInterface> loops,
    ValueRange tiledResults, ArrayRef<SmallVector<OpFoldResult>> resultOffsets,
    ArrayRef<SmallVector<OpFoldResult>> resultSizes,
    ValueRange destinationTensors) {
  if (loops.size() != 1) {
    return emitError(loc) << "expected one causal triangular workgroup loop";
  }
  LoopLikeOpInterface loop = loops.front();
  scf::ForallOp *forallOp = dyn_cast<scf::ForallOp>(&loop);
  if (!forallOp) {
    return emitError(loc)
           << "expected an scf.forall causal triangular workgroup loop";
  }
  rewriter.setInsertionPointToEnd(forallOp->getTerminator().getBody());
  for (auto [tiledValue, destinationTensor, resultOffset, resultSize] :
       llvm::zip_equal(tiledResults, destinationTensors, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> resultStrides(resultOffset.size(),
                                            rewriter.getIndexAttr(1));
    tensor::ParallelInsertSliceOp::create(rewriter, loc, tiledValue,
                                          destinationTensor, resultOffset,
                                          resultSize, resultStrides);
  }
  return success();
}

/// Checks whether we have static dimension for all the loop bounds and steps.
/// This is a requirement if the reordering strategy is set to `transpose`.
static bool areAllStaticLoopBounds(scf::ForallOp forallOp) {

  for (auto [lb, ub, step] : llvm::zip_equal(forallOp.getMixedLowerBound(),
                                             forallOp.getMixedUpperBound(),
                                             forallOp.getMixedStep())) {

    std::optional<int64_t> lbVal = getConstantIntValue(lb);
    std::optional<int64_t> ubVal = getConstantIntValue(ub);
    std::optional<int64_t> stepVal = getConstantIntValue(step);

    if (!(lbVal && ubVal && stepVal)) {
      return false;
    }
  }
  return true;
}

/// Returns true if it is allowed to leave the `op` outside distribution loops,
/// i.e., scf.forall loops. The consumer fusion could fail even the `op`
/// implements TilingInterface. E.g., linalg.pack op can only be fused as a
/// consumer in perfect tiling scenario.
static bool isAllowedToFailOnCunsumerFusion(Operation *op) {
  return isa<linalg::PackOp>(op);
}

/// Returns true if all the compute ops are within scf.forall distribution
/// loops, except the ops that are allowed to stay outside.
static bool verifyComputeOpsAfterDistribution(FunctionOpInterface funcOp) {
  WalkResult res = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<scf::ForallOp>(op) || !isComputeOp(op)) {
      return WalkResult::skip();
    }
    if (!isAllowedToFailOnCunsumerFusion(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !res.wasInterrupted();
}

//===---------------------------------------------------------------------===//
// Pass implementation.
//===---------------------------------------------------------------------===//

/// Returns true if any value produced by `producer` is used as an init value
/// for the DPS `user`. Returns false if the user is not in DPS.
static bool isUsedAsInit(Operation *producer, Operation *user) {
  auto dpsIface = dyn_cast<DestinationStyleOpInterface>(user);
  if (!dpsIface) {
    return false;
  }
  ValueRange results = producer->getResults();
  return llvm::any_of(dpsIface.getDpsInits(), [&](Value operand) {
    return llvm::is_contained(results, operand);
  });
}

void TileAndDistributeToWorkgroupsUsingForallOpPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  auto *context = &getContext();
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);

  IRRewriter rewriter(context);
  FailureOr<TilingInfo> tilingInfo =
      getTiledAndDistributionInfo(rewriter, computeOps);
  if (failed(tilingInfo)) {
    return signalPassFailure();
  }
  auto tilableOp = dyn_cast_if_present<TilingInterface>(tilingInfo->tilableOp);
  if (!tilableOp) {
    // Did not find a tileable op. So do nothing.
    return;
  }
  mlir::DominanceInfo dominanceInfo(tilableOp);
  llvm::SmallDenseSet<Operation *> tiledAndFusedOps;
  collectTiledAndFusedOps(tilableOp, tiledAndFusedOps);

  llvm::DenseSet<Operation *> yieldReplacementsFor;
  for (auto op : tiledAndFusedOps) {
    // Require replacement for values that are used after the main tilable op or
    // by ops that will definitely not be fused. Note that if a value is used as
    // an init of a DPS op, the user currently cannot be fused. Having a
    // replacement for it would attempt fusion and fail, so avoid such cases.
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          if (isUsedAsInit(op, user)) {
            return false;
          }
          return dominanceInfo.properlyDominates(tilableOp, user) ||
                 !tiledAndFusedOps.contains(user);
        })) {
      yieldReplacementsFor.insert(op);
    }
  }
  SmallVector<Attribute> deviceMappingAttribute =
      getMapping(context, tilingInfo->tileSizes);
  if (failed(IREE::Codegen::WorkgroupMappingAttr::verifyAttrList(
          context, funcOp.getLoc(), deviceMappingAttribute))) {
    return signalPassFailure();
  }
  std::optional<std::pair<unsigned, unsigned>> causalScoreDims;
  if (useCausalAttentionTriangularGrid(tilingInfo->tilableOp)) {
    if (llvm::any_of(llvm::enumerate(tilingInfo->interchange),
                     [](auto indexedDim) {
                       return indexedDim.index() != indexedDim.value();
                     })) {
      tilingInfo->tilableOp->emitOpError(
          "causal triangular workgroup grid requires identity interchange");
      return signalPassFailure();
    }
    IREE::Codegen::TranslationInfoAttr translationInfo =
        getTranslationInfo(funcOp);
    if (!translationInfo ||
        translationInfo.getDispatchLoweringPassPipeline() !=
            IREE::Codegen::DispatchLoweringPassPipeline::
                SPIRVAppleVectorDistributeAttention) {
      tilingInfo->tilableOp->emitOpError(
          "causal triangular workgroup grid requires the native Apple "
          "attention pipeline");
      return signalPassFailure();
    }
    const char *causalBoundsValue = std::getenv("IREE_METAL_CAUSAL_BWD_BOUNDS");
    if (!causalBoundsValue || StringRef(causalBoundsValue) != "1") {
      tilingInfo->tilableOp->emitOpError(
          "causal triangular workgroup grid requires "
          "IREE_METAL_CAUSAL_BWD_BOUNDS=1");
      return signalPassFailure();
    }
    auto contraction = dyn_cast<linalg::LinalgOp>(tilingInfo->tilableOp);
    if (!contraction) {
      tilingInfo->tilableOp->emitOpError(
          "causal triangular workgroup grid requires a linalg contraction");
      return signalPassFailure();
    }
    FailureOr<linalg::ContractionDimensions> contractionDims =
        linalg::inferContractionDims(contraction);
    if (failed(contractionDims) || contractionDims->m.size() != 1 ||
        contractionDims->n.size() != 1) {
      tilingInfo->tilableOp->emitOpError(
          "causal triangular workgroup grid requires one M and one N "
          "contraction dimension");
      return signalPassFailure();
    }
    causalScoreDims = std::pair<unsigned, unsigned>{contractionDims->m.front(),
                                                    contractionDims->n.front()};
  }
  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tilingInfo->tileSizes);
  tilingOptions.setInterchange(tilingInfo->interchange);
  tilingOptions.setMapping(deviceMappingAttribute);

  IREE::Codegen::WorkgroupReorderingAttrInterface workgroupReorderingStrategy =
      getLoweringConfig(tilingInfo->tilableOp).getWorkgroupReorderingStrategy();
  if (causalScoreDims) {
    unsigned rowDim = causalScoreDims->first;
    unsigned colDim = causalScoreDims->second;
    Operation *rootOp = tilingInfo->tilableOp;
    scf::SCFTilingOptions::GenerateLoopHeaderFn loopHeaderFn =
        [rowDim, colDim, rootOp](RewriterBase &rewriter, Location loc,
                                 ArrayRef<Range> loopRanges,
                                 ArrayRef<OpFoldResult> givenTileSizes,
                                 ValueRange outerDestinationTensors)
        -> FailureOr<scf::SCFTilingOptions::CustomLoopHeaderInfo> {
      return generateCausalAttentionTriangularGridHeader(
          rewriter, loc, loopRanges, givenTileSizes, outerDestinationTensors,
          rowDim, colDim, rootOp);
    };
    scf::SCFTilingOptions::GenerateLoopTerminatorFn terminatorFn =
        generateCausalAttentionTriangularGridTerminator;
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::CustomOp);
    tilingOptions.setCustomLoopGenerationFns(loopHeaderFn, terminatorFn);
  } else if (workgroupReorderingStrategy) {
    scf::SCFTilingOptions::GenerateLoopHeaderFn loopHeaderFn =
        [&workgroupReorderingStrategy](RewriterBase &rewriter, Location loc,
                                       ArrayRef<Range> loopRanges,
                                       ArrayRef<OpFoldResult> givenTileSizes,
                                       ValueRange outerDestinationTensors)
        -> FailureOr<scf::SCFTilingOptions::CustomLoopHeaderInfo> {
      return workgroupReorderingStrategy.generateLoopHeaderFn(
          rewriter, loc, loopRanges, givenTileSizes, outerDestinationTensors);
    };
    scf::SCFTilingOptions::GenerateLoopTerminatorFn terminatorFn =
        [&workgroupReorderingStrategy](
            RewriterBase &rewriter, Location loc,
            ArrayRef<LoopLikeOpInterface> loops, ValueRange tiledResults,
            ArrayRef<SmallVector<OpFoldResult>> resultOffsets,
            ArrayRef<SmallVector<OpFoldResult>> resultSizes,
            ValueRange destinationTensors) -> LogicalResult {
      return workgroupReorderingStrategy.generateLoopTerminatorFn(
          rewriter, loc, loops, tiledResults, resultOffsets, resultSizes,
          destinationTensors);
    };
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::CustomOp);
    tilingOptions.setCustomLoopGenerationFns(loopHeaderFn, terminatorFn);
  } else {
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  }

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(tilingOptions);
  RewritePatternSet cleanupPatterns(context);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(cleanupPatterns, context);
  tensor::DimOp::getCanonicalizationPatterns(cleanupPatterns, context);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(cleanupPatterns);
  // TODO(Max191): Replace populateSwapExtractWithExpandPattern with upstream
  // MLIR version once it is available (llvm-project/pull/126898).
  populateSwapExtractWithExpandPattern(cleanupPatterns);
  populateFoldExtractSliceOfBroadcastPattern(cleanupPatterns);
  // When fusing pads we do not want to generate zeroSliceGuards when doing
  // workgroup tiling. In `GPUApplyTilingLevelPass` we do have an option called
  // `allowZeroSlices` that can control this but we do not want these
  // generated if workgroup tiling is happening first.
  cleanupPatterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
      context, [](tensor::ExtractSliceOp) { return /*zeroSliceGuard=*/false; });
  tileAndFuseOptions.cleanupPatterns =
      FrozenRewritePatternSet(std::move(cleanupPatterns));

  // The control function that determines whether a tiled producer should yield
  // its replacement.
  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    Operation *owner = originalProducer.getOwner();
    if (isa<tensor::PadOp>(owner)) {
      return std::nullopt;
    }
    bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
    return scf::SCFTileAndFuseOptions::ControlFnResult{
        yieldProducerReplacement};
    return std::nullopt;
  };
  tileAndFuseOptions.setFusionControlFn(controlFn);
  rewriter.setInsertionPoint(tilableOp);

  // If the `tilableOp` is a `memref` op, then just tile the operation.
  SmallVector<LoopLikeOpInterface> tilingLoops;
  if (tilableOp->getNumResults() == 0) {
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilableOp, tilingOptions);
    if (failed(tilingResult)) {
      funcOp.emitOpError("tiling failed");
      return signalPassFailure();
    }
    rewriter.eraseOp(tilableOp);
    std::swap(tilingResult->loops, tilingLoops);
  } else {
    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilableOp,
                                                  tileAndFuseOptions);
    if (failed(tileAndFuseResult)) {
      funcOp.emitOpError("tile and fuse greedily failed");
      return signalPassFailure();
    }
    for (auto [origValue, replacement] : tileAndFuseResult->replacements) {
      Value replacementCopy = replacement;
      rewriter.replaceUsesWithIf(origValue, replacement, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return !isa<tensor::DimOp>(user) &&
               dominanceInfo.dominates(replacementCopy, user);
      });
    }
    std::swap(tileAndFuseResult->loops, tilingLoops);

    FailureOr<std::queue<Operation *>> newFusionOpportunities =
        fuseConsumersIntoForall(
            rewriter, tileAndFuseResult->tiledAndFusedOps.getArrayRef(),
            tilingLoops, [&tiledAndFusedOps](Operation *op) {
              return tiledAndFusedOps.contains(op);
            });
    if (failed(newFusionOpportunities)) {
      // Continue the work if the failure is allowed.
      if (!verifyComputeOpsAfterDistribution(funcOp)) {
        tileAndFuseResult->tiledAndFusedOps.front()->emitOpError(
            "failed to fuse consumers");
        return signalPassFailure();
      }
    } else {
      // Because we restrict to at most a single tilable consumer for yielding
      // a replacement, no new fusion opportunities will yield a replacement,
      // meaning there is no need to run consumer fusion again afterwards.
      // TODO: run producer and consumer fusion in one worklist.
      fuseProducersOfSlices(rewriter, *newFusionOpportunities,
                            tileAndFuseOptions, tilingLoops);
    }
  }
  if (!tilingLoops.empty()) {
    if (tilingLoops.size() != 1 || !isa<scf::ForallOp>(tilingLoops[0])) {
      funcOp.emitOpError(
          "expected tiling to produce a single `scf.forall` loop");
      return signalPassFailure();
    }

    // Reorder the workgroups if the strategy is set to `transpose`.
    // This just transposes the first two dimensions of the workgroup i.e., the
    // #iree.codegen.workgroup_id_x and #iree.codegen.workgroup_id_y.
    // Only reorders if the loop bounds are static.
    auto forallOp = cast<scf::ForallOp>(tilingLoops[0]);
    if (transposeWorkgroup && !causalScoreDims) {
      SmallVector<Attribute> mappingAttrs(forallOp.getMappingAttr().getValue());
      int64_t mappingSize = mappingAttrs.size();
      if (areAllStaticLoopBounds(forallOp) && mappingSize >= 2) {
        std::swap(mappingAttrs[mappingSize - 1], mappingAttrs[mappingSize - 2]);
        forallOp.setMappingAttr(ArrayAttr::get(context, mappingAttrs));
      }
    }
  }

  // Cleanup patterns for tile and distribute
  {
    RewritePatternSet patterns(context);
    populateSwapExtractWithCollapsePattern(patterns);
    populateFoldExtractSliceOfBroadcastPattern(patterns);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    context->getOrLoadDialect<tensor::TensorDialect>()
        ->getCanonicalizationPatterns(patterns);
    context->getOrLoadDialect<IREE::LinalgExt::IREELinalgExtDialect>()
        ->getCanonicalizationPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    scf::ForallOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("tiling canonicalization failed");
      return signalPassFailure();
    }
  }

  return;
}
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTileAndDistributeToWorkgroupsWithReordering(bool transposeWorkgroup) {
  return std::make_unique<TileAndDistributeToWorkgroupsUsingForallOpPass>(
      transposeWorkgroup);
}
} // namespace mlir::iree_compiler
