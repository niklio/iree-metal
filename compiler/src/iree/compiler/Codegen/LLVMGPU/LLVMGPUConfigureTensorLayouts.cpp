// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "iree/compiler/Codegen/Utils/Utils.h"

#define DEBUG_TYPE "iree-llvmgpu-configure-vector-layouts"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUCONFIGURETENSORLAYOUTSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

using IREE::GPU::MMASingleSubgroupLayout;
using IREE::VectorExt::NestedLayoutAttr;
using IREE::VectorExt::ToLayoutOp;
using IREE::VectorExt::VectorLayoutInterface;

namespace {

static SmallVector<bool> getPromotedOperands(Operation *op) {
  SmallVector<bool> promotedOperands(op->getNumOperands(), false);

  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  if (!config) {
    return promotedOperands;
  }

  std::optional<SmallVector<int64_t>> promoteConfig =
      getPromotedOperandList(config);
  if (!promoteConfig) {
    return promotedOperands;
  }

  for (int64_t operand : promoteConfig.value()) {
    promotedOperands[operand] = true;
  }

  return promotedOperands;
}

static IREE::Codegen::InnerTileDescAttrInterface getIntrinsic(Operation *op) {
  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  assert(config && "Cannot find intrinsic from unconfigured op.");

  IREE::Codegen::InnerTileDescAttrInterface mmaIntrinsic = getMmaKind(config);
  assert(mmaIntrinsic && "Cannot find intrinsic in lowering config.");
  return mmaIntrinsic;
}

/// Given two arrays bounds and tile, compute bounds = ceil(bounds / tile).
///
/// If "tile" contains 0, or is smaller than bounds, divide bounds by 1
/// for those values.
///
/// Returns the actual divisor (without zeros or out of bounds) used to compute
/// bounds /= divisor.
FailureOr<SmallVector<int64_t>> divideTile(SmallVector<int64_t> &bounds,
                                           ArrayRef<int64_t> tile) {
  assert(bounds.size() >= tile.size() &&
         "cannot divide bounds with a different rank");

  SmallVector<int64_t> divisor(bounds.size(), 1);
  for (auto [div, size] : llvm::zip(divisor, tile)) {
    if (size == 0) {
      continue;
    }
    div = size;
  }

  for (auto [bound, div] : llvm::zip_equal(bounds, divisor)) {
    bound = llvm::divideCeil(bound, div);
  }

  return divisor;
}

SmallVector<int64_t> applyProjectedPermutation(ArrayRef<int64_t> input,
                                               ArrayRef<int64_t> perm) {
  SmallVector<int64_t> result;
  result.reserve(perm.size());
  for (int64_t dim : perm) {
    result.push_back(input[dim]);
  }
  return result;
}

SmallVector<int64_t> getStridesFromBasis(ArrayRef<int64_t> basis) {
  SmallVector<int64_t> strides(basis.size());
  int64_t currStride = 1;
  for (auto [stride, size] : llvm::reverse(llvm::zip_equal(strides, basis))) {
    stride = currStride;
    currStride *= size;
  }
  return strides;
}

static LogicalResult distributeTilingSizes(Operation *candidate,
                                           IREE::GPU::LoweringConfigAttr config,
                                           IREE::GPU::TilingLevel level,
                                           SmallVector<int64_t> &bounds,
                                           SmallVector<int64_t> &sizes,
                                           SmallVector<int64_t> &strides) {
  if (ShapedType::isDynamicShape(bounds)) {
    candidate->emitError()
        << "Cannot set layouts on a dynamically shaped iteration space";
    return failure();
  }

  FailureOr<IREE::GPU::Basis> basis = IREE::GPU::getBasis(config, level);
  if (failed(basis)) {
    candidate->emitError()
        << "Could not find a subgroup basis from lowering config";
    return failure();
  }

  sizes = applyProjectedPermutation(basis->counts, basis->mapping);
  strides = applyProjectedPermutation(getStridesFromBasis(basis->counts),
                                      basis->mapping);

  if (failed(divideTile(bounds, sizes))) {
    candidate->emitError()
        << "Could not divide bounds over given basis for level: "
        << IREE::GPU::stringifyTilingLevel(level);
    return failure();
  }

  return success();
}

struct ContractionLayout {
  VectorLayoutInterface lhs;
  VectorLayoutInterface rhs;
  VectorLayoutInterface acc;
};

static bool
usesApplePhysicalFragmentLayout(IREE::GPU::MmaInterfaceAttr intrinsic) {
  auto mma = dyn_cast<IREE::GPU::MMAAttr>(intrinsic);
  return mma && mma.getApplePhysicalFragmentLayout();
}

static SmallVector<int64_t>
getApplePhysicalFragmentFactors(IREE::GPU::MmaInterfaceAttr intrinsic) {
  auto [m, n, k] = intrinsic.getMNKShape();
  assert(m == n && n == k && "expected square Apple MMA intrinsic");
  if (m == 8) {
    return {2, 4};
  }
  assert(m == 16 && "unexpected Apple MMA shape");
  return {2, 2, 4};
}

static FailureOr<SmallVector<int64_t>>
getPhysicalFragmentDims(ArrayRef<unsigned> contractionDims,
                        ArrayRef<int64_t> bounds, ArrayRef<int64_t> factors) {
  if (contractionDims.size() < factors.size()) {
    return failure();
  }
  SmallVector<int64_t> result;
  for (auto [dim, factor] :
       llvm::zip_equal(contractionDims.take_back(factors.size()), factors)) {
    if (bounds[dim] != factor) {
      return failure();
    }
    result.push_back(dim);
  }
  return result;
}

// Get the layouts to use for the contraction given the intrinsic to use and
// number of subgroups on the M and N dimension.
//
// The contraction is expected to have 3 operands: lhs, rhs and acc of the
// contraction and a single accumulator.
static FailureOr<ContractionLayout>
getContractionLayout(Operation *candidate, ArrayRef<int64_t> bounds,
                     ArrayRef<AffineMap> contractIndexingMaps) {
  auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(candidate);
  if (!config) {
    return failure();
  }

  auto intrinsic =
      dyn_cast_if_present<IREE::GPU::MmaInterfaceAttr>(getIntrinsic(candidate));
  if (!intrinsic) {
    return failure();
  }

  int64_t rank = bounds.size();
  FailureOr<VectorContractOpInfo> maybeOpInfo =
      VectorContractOpInfo::inferFromIndexingMaps(contractIndexingMaps);
  if (failed(maybeOpInfo)) {
    return failure();
  }
  VectorContractOpInfo opInfo = maybeOpInfo.value();
  // Get the inner dimensions.
  int64_t innerMDim = opInfo.getMDims().back();
  int64_t innerNDim = opInfo.getNDims().back();
  int64_t innerKDim = opInfo.getKDims().back();

  bool physicalAppleLayout = usesApplePhysicalFragmentLayout(intrinsic);
  SmallVector<int64_t> physicalFactors;
  SmallVector<int64_t> physicalMDims, physicalNDims, physicalKDims;
  if (physicalAppleLayout) {
    physicalFactors = getApplePhysicalFragmentFactors(intrinsic);
    FailureOr<SmallVector<int64_t>> maybeMDims =
        getPhysicalFragmentDims(opInfo.getMDims(), bounds, physicalFactors);
    FailureOr<SmallVector<int64_t>> maybeNDims =
        getPhysicalFragmentDims(opInfo.getNDims(), bounds, physicalFactors);
    FailureOr<SmallVector<int64_t>> maybeKDims =
        getPhysicalFragmentDims(opInfo.getKDims(), bounds, physicalFactors);
    if (failed(maybeMDims) || failed(maybeNDims) || failed(maybeKDims)) {
      return candidate->emitError(
          "marked Apple physical fragment lacks its expanded M/N/K factors");
    }
    physicalMDims = std::move(*maybeMDims);
    physicalNDims = std::move(*maybeNDims);
    physicalKDims = std::move(*maybeKDims);
  }

  SmallVector<int64_t> batchCounts(bounds);

  // Subgroup distribution layouts.
  SmallVector<int64_t> subgroupCounts, subgroupStrides;
  if (failed(distributeTilingSizes(
          candidate, config, IREE::GPU::TilingLevel::Subgroup, batchCounts,
          subgroupCounts, subgroupStrides))) {
    return failure();
  }

  // Since these MMA intrinsics have a given tile size for each subgroup, we can
  // calculate the batch dimensions without looking at the subgroup layout.
  SmallVector<int64_t> subgroupSize(rank, 1);
  auto [mSize, nSize, kSize] = intrinsic.getMNKShape();
  if (physicalAppleLayout) {
    for (ArrayRef<int64_t> dims :
         {ArrayRef<int64_t>(physicalMDims), ArrayRef<int64_t>(physicalNDims),
          ArrayRef<int64_t>(physicalKDims)}) {
      for (auto [dim, factor] : llvm::zip_equal(dims, physicalFactors)) {
        // Schedule-level subgroup distribution belongs to the dynamic outer
        // factor inserted ahead of these fixed fragment factors.
        if (subgroupCounts[dim] != 1) {
          return candidate->emitError(
              "Apple physical fragment factor is subgroup-distributed");
        }
        subgroupSize[dim] = factor;
      }
    }
  } else {
    subgroupSize[innerMDim] = mSize;
    subgroupSize[innerNDim] = nSize;
    subgroupSize[innerKDim] = kSize;
  }

  for (auto i : llvm::seq<int64_t>(rank)) {
    batchCounts[i] = llvm::divideCeil(batchCounts[i], subgroupSize[i]);
  }

  // MMA intrinsics can be weird and usually don't have a single subgroup
  // iteration space, so we need to find their value subgroup iteration space
  // individually.
  auto getFragmentLayout = [&](int operandIndex, int64_t outerDim,
                               int64_t innerDim, ArrayRef<int64_t> rowDims,
                               ArrayRef<int64_t> colDims,
                               AffineMap map) -> VectorLayoutInterface {
    // Note that the struct MMASingleSubgroupLayout contains the partial layout
    // for the canonical (M, K) x (K, N) -> (M, N) matmul form. We treat the
    // concrete nested layout as the layout for the innermost M, N, K
    // dimensions.
    SmallVector<int64_t> outerCounts(rank, 1);
    SmallVector<int64_t> elementCounts(rank, 1);
    SmallVector<int64_t> threadCounts(rank, 1);
    SmallVector<int64_t> threadStrides(rank, 0);

    if (physicalAppleLayout) {
      auto setPhysicalAxis = [&](ArrayRef<int64_t> dims, bool row) {
        size_t offset = 0;
        if (physicalFactors.size() == 3) {
          outerCounts[dims[0]] = 2;
          offset = 1;
        }
        // Physical 8x8 mapping:
        // row: lane factors 2@stride16, 4@stride2
        // col: lane factors 2@stride8, 2@stride1, element factor 2.
        threadCounts[dims[offset]] = 2;
        threadStrides[dims[offset]] = row ? 16 : 8;
        if (row) {
          threadCounts[dims[offset + 1]] = 4;
          threadStrides[dims[offset + 1]] = 2;
        } else {
          threadCounts[dims[offset + 1]] = 2;
          threadStrides[dims[offset + 1]] = 1;
          elementCounts[dims[offset + 1]] = 2;
        }
      };
      setPhysicalAxis(rowDims, /*row=*/true);
      setPhysicalAxis(colDims, /*row=*/false);
    } else {
      MMASingleSubgroupLayout subgroupLayout =
          IREE::GPU::getSingleSubgroupLayout(intrinsic, operandIndex);
      outerCounts[outerDim] = subgroupLayout.outer[0];
      outerCounts[innerDim] = subgroupLayout.outer[1];
      threadCounts[outerDim] = subgroupLayout.thread[0];
      threadCounts[innerDim] = subgroupLayout.thread[1];
      threadStrides[outerDim] = subgroupLayout.tstrides[0];
      threadStrides[innerDim] = subgroupLayout.tstrides[1];
      elementCounts[outerDim] = subgroupLayout.element[0];
      elementCounts[innerDim] = subgroupLayout.element[1];
    }
    // Get the fragment layout for the entire iteration space and then project
    // it. This is significantly easier than trying to create a layout for each
    // fragment itself.
    auto fragmentSpaceLayout = NestedLayoutAttr::get(
        map.getContext(), subgroupCounts, batchCounts, outerCounts,
        threadCounts, elementCounts, subgroupStrides, threadStrides);
    return fragmentSpaceLayout.apply(map);
  };

  VectorLayoutInterface lhs =
      getFragmentLayout(IREE::GPU::kMMAOperandLhs, innerMDim, innerKDim,
                        physicalMDims, physicalKDims, contractIndexingMaps[0]);
  VectorLayoutInterface rhs =
      getFragmentLayout(IREE::GPU::kMMAOperandRhs, innerKDim, innerNDim,
                        physicalKDims, physicalNDims, contractIndexingMaps[1]);
  VectorLayoutInterface acc =
      getFragmentLayout(IREE::GPU::kMMAOperandAcc, innerMDim, innerNDim,
                        physicalMDims, physicalNDims, contractIndexingMaps[2]);

  return ContractionLayout{lhs, rhs, acc};
}

SmallVector<int64_t> getIterationSpaceBounds(linalg::LinalgOp linalgOp) {
  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
  std::optional<VectorizationTileSizes> sizes =
      inferSizesFromIR(linalgOp, std::nullopt);
  // Even though the opShape could be dynamic, we could potentially
  // infer the vector shape
  if (sizes.has_value()) {
    bounds = sizes.value().vectorSizes;
  }
  return bounds;
}

// Preserve an output layout across an immediate pointwise DPS update. This is
// more than a convenience for layout propagation: canonicalization can replace
// the pointwise result with a constant and erase the contraction that supplied
// the original anchor. Anchoring the tied result keeps later reductions and
// loops distributable in that case.
static void anchorPointwiseDpsResults(ToLayoutOp anchor,
                                      RewriterBase &rewriter) {
  Value anchoredValue = anchor.getResult();
  VectorLayoutInterface layout = anchor.getLayout();
  SmallVector<OpResult> resultsToAnchor;

  for (OpOperand &use : anchoredValue.getUses()) {
    auto genericOp = dyn_cast<linalg::GenericOp>(use.getOwner());
    if (!genericOp || genericOp.getNumReductionLoops() != 0) {
      continue;
    }
    for (OpResult result : genericOp->getResults()) {
      OpOperand *initOperand =
          genericOp.getDpsInitOperand(result.getResultNumber());
      if (initOperand != &use || result.getType() != anchoredValue.getType()) {
        continue;
      }
      if (genericOp.getMatchingIndexingMap(initOperand) !=
          genericOp.getIndexingMapMatchingResult(result)) {
        continue;
      }
      resultsToAnchor.push_back(result);
    }
  }

  for (OpResult result : resultsToAnchor) {
    rewriter.setInsertionPointAfter(result.getOwner());
    auto resultAnchor =
        ToLayoutOp::create(rewriter, result.getLoc(), result, layout);
    rewriter.replaceAllUsesExcept(result, resultAnchor.getResult(),
                                  resultAnchor);
  }
}

static LogicalResult
setContractionAnchor(IREE::Codegen::InnerTileDescAttrInterface intrinsic,
                     SmallVector<bool> promotedOperands, RewriterBase &rewriter,
                     linalg::LinalgOp contract) {
  // This function should have only be called on a contraction op.
  assert(linalg::isaContractionOpInterface(contract) &&
         "cannot set contraction anchor on non contraction op");

  SmallVector<int64_t> bounds = getIterationSpaceBounds(contract);
  auto layouts =
      getContractionLayout(contract, bounds, contract.getIndexingMapsArray());
  if (failed(layouts)) {
    return contract->emitError("cannot get concrete layout for contraction");
  }

  auto [aLayout, bLayout, cLayout] = *layouts;
  Location loc = contract.getLoc();

  Value lhs = contract->getOperand(0);
  Value rhs = contract->getOperand(1);
  Value acc = contract->getOperand(2);

  // Set layouts for lhs, rhs and acc.
  rewriter.setInsertionPoint(contract);
  auto layoutedLhs = ToLayoutOp::create(rewriter, loc, lhs, aLayout);
  auto layoutedRhs = ToLayoutOp::create(rewriter, loc, rhs, bLayout);
  auto layoutedAcc = ToLayoutOp::create(rewriter, loc, acc, cLayout);

  // Promote matmul lhs and rhs.
  // TODO: This is a hack until layout analysis is improved. The layout analysis
  // should decide where to put these shared memory conversions.
  if (promotedOperands[0]) {
    layoutedLhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[1]) {
    layoutedRhs.setSharedMemoryConversion(true);
  }

  if (promotedOperands[2]) {
    layoutedAcc.setSharedMemoryConversion(true);
  }

  contract->setOperand(0, layoutedLhs.getResult());
  contract->setOperand(1, layoutedRhs.getResult());
  contract->setOperand(2, layoutedAcc.getResult());

  // Set layout for result.
  rewriter.setInsertionPointAfter(contract);
  auto toLayout =
      ToLayoutOp::create(rewriter, loc, contract->getResult(0), cLayout);
  rewriter.replaceAllUsesExcept(contract->getResult(0), toLayout.getResult(),
                                toLayout);
  anchorPointwiseDpsResults(toLayout, rewriter);

  return success();
}

static LogicalResult setDerivedThreadConfigLayout(
    IREE::GPU::DerivedThreadConfigAttr config, linalg::LinalgOp linalgOp,
    ArrayRef<int64_t> workgroupSize, RewriterBase &rewriter) {

  int64_t opRank = linalgOp.getNumLoops();

  SmallVector<int64_t> elementTile = config.getStaticTilingLevelSizes(
      static_cast<unsigned>(IREE::GPU::TilingLevel::Thread), linalgOp);

  SmallVector<int64_t> opShape = linalgOp.getStaticLoopRanges();
  std::optional<VectorizationTileSizes> sizes =
      inferSizesFromIR(linalgOp, std::nullopt);
  // Even though the opShape could be dynamic, we could potentially
  // infer the vector shape
  if (sizes.has_value()) {
    opShape = sizes.value().vectorSizes;
  }

  for (auto [index, size, element] : llvm::enumerate(opShape, elementTile)) {
    if (ShapedType::isDynamic(size)) {
      linalgOp->emitError() << "opShape could not be inferred";
      return failure();
    }

    if (size % element != 0) {
      linalgOp->emitError()
          << "Operation with unsupported number of elements. "
             "Chosen vector tile sizes for operation are not "
             "divisible by operation loop ranges at dim: "
          << index << ", size=" << size << ", vector size = " << element;
      return failure();
    }

    size /= element;
  }

  SmallVector<int64_t> threadTile(opRank, 1);
  SmallVector<int64_t> threadStrides(opRank, 0);

  int64_t residualThreads = ShapedType::getNumElements(workgroupSize);
  int64_t currStride = 1;

  for (auto [tile, stride, size] :
       llvm::reverse(llvm::zip(threadTile, threadStrides, opShape))) {
    int64_t threadBlock;
    if (residualThreads % size == 0) {
      threadBlock = size;
    } else if (size % residualThreads == 0) {
      threadBlock = residualThreads;
    } else {
      linalgOp->emitError() << "Operation with unsupported number of elements.";
      return failure();
    }

    tile = threadBlock;
    stride = currStride;
    size /= threadBlock;

    currStride *= threadBlock;
    residualThreads /= threadBlock;
  }

  SmallVector<int64_t> subgroupTile(opRank, 1);
  SmallVector<int64_t> subgroupStrides(opRank, 0);
  SmallVector<int64_t> outerTile(opRank, 1);

  MLIRContext *context = rewriter.getContext();
  auto layout = IREE::VectorExt::NestedLayoutAttr::get(
      context, subgroupTile, opShape, outerTile, threadTile, elementTile,
      subgroupStrides, threadStrides);

  Location loc = linalgOp.getLoc();

  rewriter.setInsertionPointAfter(linalgOp);
  for (OpResult result : linalgOp->getResults()) {
    VectorLayoutInterface resultLayout =
        layout.apply(linalgOp.getIndexingMapMatchingResult(result));
    auto toLayout = ToLayoutOp::create(rewriter, loc, result, resultLayout);
    rewriter.replaceAllUsesExcept(result, toLayout, toLayout);
  }

  return success();
}

static LogicalResult setIntrinsicLoweringConfigLayout(
    IREE::GPU::LoweringConfigAttr config, linalg::LinalgOp candidate,
    ArrayRef<int64_t> workgroupSize, RewriterBase &rewriter) {

  SmallVector<bool> promotedOperands = getPromotedOperands(candidate);
  IREE::Codegen::InnerTileDescAttrInterface intrinsic = getIntrinsic(candidate);

  if (linalg::isaContractionOpInterface(candidate)) {
    if (succeeded(setContractionAnchor(intrinsic, promotedOperands, rewriter,
                                       candidate))) {
      return success();
    }
  }

  candidate->emitError() << "Unable to set intrinsic layouts on operation "
                            "based on given lowering config: "
                         << config;
  return failure();
}

static LogicalResult setGPULoweringConfigLayout(
    IREE::GPU::LoweringConfigAttr config, linalg::LinalgOp candidate,
    ArrayRef<int64_t> workgroupSize, RewriterBase &rewriter) {
  MLIRContext *context = config.getContext();
  Location loc = candidate.getLoc();

  SmallVector<int64_t> bounds = getIterationSpaceBounds(candidate);

  // Subgroup distribution layouts.
  SmallVector<int64_t> subgroupSizes, subgroupStrides;
  if (failed(distributeTilingSizes(candidate, config,
                                   IREE::GPU::TilingLevel::Subgroup, bounds,
                                   subgroupSizes, subgroupStrides))) {
    return failure();
  }

  // Thread distribution layouts.
  SmallVector<int64_t> threadSizes, threadStrides;
  if (failed(distributeTilingSizes(candidate, config,
                                   IREE::GPU::TilingLevel::Thread, bounds,
                                   threadSizes, threadStrides))) {
    return failure();
  }

  // Use thread tile sizes as the vector width for each thread.
  SmallVector<int64_t> threadTileSizes = config.getStaticTilingLevelSizes(
      llvm::to_underlying(IREE::GPU::TilingLevel::Thread), candidate);
  FailureOr<SmallVector<int64_t>> elementTile =
      divideTile(bounds, threadTileSizes);
  if (failed(elementTile)) {
    candidate->emitError() << "Could not divide bounds over given thread tile";
  }
  // The remaining bounds become batch sizes. We could also use subgroup tile
  // sizes, as a way of specifying batch size, but since it is a derived
  // property, we choose to compute it.
  ArrayRef<int64_t> batchTile = bounds;
  SmallVector<int64_t> outerTile(bounds.size(), 1);

  auto layout = IREE::VectorExt::NestedLayoutAttr::get(
      context, subgroupSizes, batchTile, outerTile, threadSizes,
      elementTile.value(), subgroupStrides, threadStrides);

  SmallVector<bool> promotedOperands = getPromotedOperands(candidate);

  rewriter.setInsertionPoint(candidate);
  for (OpOperand &operand : candidate->getOpOperands()) {
    VectorLayoutInterface operandLayout =
        layout.apply(candidate.getMatchingIndexingMap(&operand));
    auto toLayout =
        ToLayoutOp::create(rewriter, loc, operand.get(), operandLayout);
    // Set shared memory promotion if requested.
    toLayout.setSharedMemoryConversion(
        promotedOperands[operand.getOperandNumber()]);
    operand.set(toLayout);
  }

  rewriter.setInsertionPointAfter(candidate);
  for (OpResult result : candidate->getResults()) {
    VectorLayoutInterface resultLayout =
        layout.apply(candidate.getIndexingMapMatchingResult(result));
    auto toLayout = ToLayoutOp::create(rewriter, loc, result, resultLayout);
    rewriter.replaceAllUsesExcept(result, toLayout, toLayout);
  }

  return success();
}

struct LLVMGPUConfigureTensorLayoutsPass final
    : impl::LLVMGPUConfigureTensorLayoutsPassBase<
          LLVMGPUConfigureTensorLayoutsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp);

    std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
        getWorkgroupSize(funcOp);
    if (!maybeWorkgroupSize) {
      funcOp->emitOpError()
          << "unable to query workgroup_size information from entry point";
      return signalPassFailure();
    }

    if (failed(setLayoutsFromLoweringConfig(funcOp, maybeWorkgroupSize.value(),
                                            rewriter))) {
      return signalPassFailure();
    }
  }

  LogicalResult setLayoutsFromLoweringConfig(FunctionOpInterface funcOp,
                                             ArrayRef<int64_t> workgroupSize,
                                             RewriterBase &rewriter) {
    SmallVector<linalg::LinalgOp> candidates;
    funcOp->walk([&](linalg::LinalgOp op) {
      if (getLoweringConfig(op)) {
        candidates.push_back(op);
      }
    });

    for (linalg::LinalgOp candidate : candidates) {
      auto result =
          TypeSwitch<IREE::Codegen::LoweringConfigAttrInterface, LogicalResult>(
              getLoweringConfig(candidate))
              .Case([&](IREE::GPU::DerivedThreadConfigAttr config) {
                return setDerivedThreadConfigLayout(config, candidate,
                                                    workgroupSize, rewriter);
              })
              .Case([&](IREE::GPU::LoweringConfigAttr config) {
                if (getMmaKind(config)) {
                  return setIntrinsicLoweringConfigLayout(
                      config, candidate, workgroupSize, rewriter);
                }
                return setGPULoweringConfigLayout(config, candidate,
                                                  workgroupSize, rewriter);
              })
              .Default(failure());

      if (failed(result)) {
        return failure();
      }
    }

    return success();
  }
};
} // namespace

} // namespace mlir::iree_compiler
