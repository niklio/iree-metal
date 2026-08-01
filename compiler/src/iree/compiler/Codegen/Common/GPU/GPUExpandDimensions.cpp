// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-expand-dimensions"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUEXPANDDIMENSIONSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct GPUExpandDimensionsPass final
    : impl::GPUExpandDimensionsPassBase<GPUExpandDimensionsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static constexpr StringLiteral kDimensionExpansionName = "expand_dims";
static constexpr StringLiteral kApplePhysicalFragmentExpansionMaterializedName =
    "apple_physical_fragment_expansion_materialized";

static std::optional<IREE::GPU::MMAAttr>
getApplePhysicalFragmentMma(IREE::GPU::LoweringConfigAttr config) {
  auto mma =
      dyn_cast_if_present<IREE::GPU::MMAAttr>(IREE::GPU::getMmaKind(config));
  if (!mma || !mma.getApplePhysicalFragmentLayout()) {
    return std::nullopt;
  }
  switch (mma.getIntrinsic()) {
  case IREE::GPU::MMAIntrinsic::APPLE_SIMDGROUP_F32_8x8x8_F16:
  case IREE::GPU::MMAIntrinsic::APPLE_SIMDGROUP_F32_8x8x8_BF16:
  case IREE::GPU::MMAIntrinsic::APPLE_SIMDGROUP_F32_16x16x16_F16:
  case IREE::GPU::MMAIntrinsic::APPLE_SIMDGROUP_F32_16x16x16_BF16:
    return mma;
  default:
    return std::nullopt;
  }
}

static Operation *getDefiningOpSkippingTensorCasts(Value value) {
  Operation *producer = value.getDefiningOp();
  while (producer) {
    if (auto castOp = dyn_cast<tensor::CastOp>(producer)) {
      producer = castOp.getSource().getDefiningOp();
      continue;
    }
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(producer)) {
      producer = collapseOp.getSrc().getDefiningOp();
      continue;
    }
    break;
  }
  return producer;
}

// The ordinary expand_dims contract is consumed by this pass, but a physical
// Apple fragment marker must survive the cleanup rewrite so later layout,
// packing, and terminal lowering can prove it. Track only that marker here to
// leave the existing ordinary-expand behavior unchanged.
class ApplePhysicalConfigTrackingListener final
    : public RewriterBase::Listener {
public:
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override {
    auto config = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
    if (!config || !getApplePhysicalFragmentMma(config) ||
        replacement.empty()) {
      return;
    }
    Operation *producer = getDefiningOpSkippingTensorCasts(replacement.front());
    if (!producer || producer->getName() != op->getName()) {
      return;
    }
    if (llvm::any_of(replacement.drop_front(), [&](Value value) {
          return getDefiningOpSkippingTensorCasts(value) != producer;
        })) {
      return;
    }
    if (!getLoweringConfig(producer)) {
      setLoweringConfig(producer, config);
    }
  }
};

// Preserve the existing config-tracking behavior for ordinary expansion while
// additionally looking through the result collapse_shape used by expansion
// patterns to retain an Apple physical-fragment contract.
class ExpansionConfigTrackingListener final : public RewriterBase::Listener {
public:
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override {
    defaultListener.notifyOperationReplaced(op, replacement);
    physicalListener.notifyOperationReplaced(op, replacement);
  }

private:
  ConfigTrackingListener defaultListener;
  ApplePhysicalConfigTrackingListener physicalListener;
};

static SmallVector<int64_t>
getApplePhysicalFragmentFactors(IREE::GPU::MMAAttr mma) {
  auto [m, n, k] = mma.getMNKShape();
  assert(m == n && n == k && "expected square Apple MMA intrinsic");
  if (m == 8) {
    return {2, 4};
  }
  assert(m == 16 && "unexpected Apple MMA shape");
  return {2, 2, 4};
}

static bool hasTrailingShape(ArrayRef<unsigned> dims, ArrayRef<int64_t> bounds,
                             ArrayRef<int64_t> expectedShape) {
  if (dims.size() < expectedShape.size()) {
    return false;
  }
  return llvm::equal(
      expectedShape,
      llvm::map_range(dims.take_back(expectedShape.size()),
                      [&](unsigned dim) { return bounds[dim]; }));
}

static LogicalResult validateApplePhysicalFragmentExpansionMaterialized(
    linalg::LinalgOp op, IREE::GPU::LoweringConfigAttr config,
    IREE::GPU::MMAAttr mma) {
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(op);
  if (failed(dims)) {
    return op.emitOpError(
        "materialized Apple physical fragment must be a contraction");
  }
  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  SmallVector<int64_t> factors = getApplePhysicalFragmentFactors(mma);
  if (!hasTrailingShape(dims->m, bounds, factors) ||
      !hasTrailingShape(dims->n, bounds, factors) ||
      !hasTrailingShape(dims->k, bounds, factors)) {
    return op.emitOpError(
        "materialized Apple physical fragment lacks exact trailing M/N/K "
        "factors");
  }

  int64_t rank = op.getNumLoops();
  for (IREE::GPU::TilingLevel level :
       {IREE::GPU::TilingLevel::Workgroup,
        IREE::GPU::TilingLevel::PartialReduction,
        IREE::GPU::TilingLevel::Reduction, IREE::GPU::TilingLevel::Serial,
        IREE::GPU::TilingLevel::Thread, IREE::GPU::TilingLevel::Subgroup,
        IREE::GPU::TilingLevel::Lane}) {
    StringRef name = IREE::GPU::getTilingLevelName(level);
    Attribute rawAttr = config.getAttributes().get(name);
    if (!rawAttr) {
      continue;
    }
    auto tiles = dyn_cast<ArrayAttr>(rawAttr);
    if (!tiles || tiles.size() != static_cast<size_t>(rank) ||
        !llvm::all_of(tiles, llvm::IsaPred<IntegerAttr>)) {
      return op.emitOpError("malformed materialized tiling level '")
             << name << "'";
    }
    if (llvm::any_of(tiles, [](Attribute tile) {
          return cast<IntegerAttr>(tile).getInt() < 0;
        })) {
      return op.emitOpError("negative size in materialized tiling level '")
             << name << "'";
    }
  }

  auto validateBasis = [&](IREE::GPU::TilingLevel level,
                           StringRef name) -> LogicalResult {
    if (!config.getAttributes().get(name)) {
      return success();
    }
    FailureOr<IREE::GPU::Basis> maybeBasis = IREE::GPU::getBasis(config, level);
    if (failed(maybeBasis) ||
        maybeBasis->mapping.size() != static_cast<size_t>(rank)) {
      return op.emitOpError("malformed materialized ") << name;
    }
    if (llvm::any_of(maybeBasis->counts,
                     [](int64_t count) { return count <= 0; })) {
      return op.emitOpError("non-positive count in materialized ") << name;
    }
    if (llvm::any_of(maybeBasis->mapping, [&](int64_t mappedDim) {
          return mappedDim < 0 ||
                 mappedDim >= static_cast<int64_t>(maybeBasis->counts.size());
        })) {
      return op.emitOpError("out-of-range mapping in materialized ") << name;
    }
    llvm::SmallDenseSet<int64_t> seenMapping;
    if (llvm::any_of(maybeBasis->mapping, [&](int64_t mappedDim) {
          return !seenMapping.insert(mappedDim).second;
        })) {
      return op.emitOpError("duplicate mapping in materialized ") << name;
    }
    return success();
  };
  if (failed(
          validateBasis(IREE::GPU::TilingLevel::Subgroup, "subgroup_basis")) ||
      failed(validateBasis(IREE::GPU::TilingLevel::Thread, "lane_basis"))) {
    return failure();
  }
  return success();
}

struct ApplePhysicalFragmentExpansion {
  IREE::GPU::DimensionExpansionAttr dimensions;
  IREE::GPU::LoweringConfigAttr loweringConfig;
};

static FailureOr<ApplePhysicalFragmentExpansion>
getApplePhysicalFragmentExpansion(linalg::LinalgOp op,
                                  IREE::GPU::LoweringConfigAttr config,
                                  IREE::GPU::MMAAttr mma) {
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(op);
  if (failed(dims) || dims->m.empty() || dims->n.empty() || dims->k.empty()) {
    return op.emitOpError(
        "Apple physical fragments require a contraction with M/N/K dims");
  }
  if (IREE::GPU::getDimensionExpansion(config)) {
    return op.emitOpError(
        "Apple physical fragments cannot be combined with expand_dims");
  }

  int64_t rank = op.getNumLoops();
  llvm::SmallDenseSet<int64_t> splitDims = {
      static_cast<int64_t>(dims->m.back()),
      static_cast<int64_t>(dims->n.back()),
      static_cast<int64_t>(dims->k.back())};
  SmallVector<int64_t> factors = getApplePhysicalFragmentFactors(mma);
  int64_t intrinsicSize = llvm::product_of(factors);
  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  for (unsigned dim : {dims->m.back(), dims->n.back(), dims->k.back()}) {
    int64_t bound = bounds[dim];
    if (ShapedType::isDynamic(bound) || bound <= 0 ||
        bound % intrinsicSize != 0) {
      return op.emitOpError(
                 "Apple physical fragment dimension must be a positive "
                 "static multiple of the intrinsic size; dimension ")
             << dim << " has bound " << bound;
    }
  }

  SmallVector<ReassociationIndices> reassociations;
  SmallVector<int64_t> outputShape;
  int64_t nextOutputDim = 0;
  for (int64_t dim = 0; dim < rank; ++dim) {
    ReassociationIndices &group = reassociations.emplace_back();
    group.push_back(nextOutputDim++);
    if (!splitDims.contains(dim)) {
      outputShape.push_back(ShapedType::kDynamic);
      continue;
    }
    // Keep all schedule-level batching/subgroup tiling in an outer factor;
    // the trailing fixed factors describe exactly one native Apple fragment.
    outputShape.push_back(ShapedType::kDynamic);
    for (int64_t factor : factors) {
      group.push_back(nextOutputDim++);
      outputShape.push_back(factor);
    }
  }
  auto expansion = IREE::GPU::DimensionExpansionAttr::get(
      op.getContext(), reassociations, outputShape);

  NamedAttrList expandedAttrs(config.getAttributes());
  Builder b(op.getContext());
  for (IREE::GPU::TilingLevel level :
       {IREE::GPU::TilingLevel::Workgroup,
        IREE::GPU::TilingLevel::PartialReduction,
        IREE::GPU::TilingLevel::Reduction, IREE::GPU::TilingLevel::Serial,
        IREE::GPU::TilingLevel::Thread, IREE::GPU::TilingLevel::Subgroup,
        IREE::GPU::TilingLevel::Lane}) {
    StringRef name = IREE::GPU::getTilingLevelName(level);
    Attribute rawAttr = config.getAttributes().get(name);
    if (!rawAttr) {
      continue;
    }
    auto oldAttr = dyn_cast<ArrayAttr>(rawAttr);
    if (!oldAttr) {
      return op.emitOpError("malformed tiling level '") << name << "'";
    }
    if (oldAttr.size() != static_cast<size_t>(rank) ||
        !llvm::all_of(oldAttr, llvm::IsaPred<IntegerAttr>)) {
      return op.emitOpError("malformed tiling level '") << name << "'";
    }
    SmallVector<int64_t> expandedTiles;
    for (auto [dim, tileAttr] : llvm::enumerate(oldAttr)) {
      int64_t tile = cast<IntegerAttr>(tileAttr).getInt();
      if (tile < 0) {
        return op.emitOpError("negative size in tiling level '") << name << "'";
      }
      if (!splitDims.contains(dim)) {
        expandedTiles.push_back(tile);
        continue;
      }
      if (tile == 0) {
        expandedTiles.append(factors.size() + 1, 0);
        continue;
      }
      if (tile < intrinsicSize || tile % intrinsicSize != 0) {
        return op.emitOpError("tiling level '")
               << name << "' size " << tile
               << " does not preserve an Apple fragment of size "
               << intrinsicSize;
      }
      expandedTiles.push_back(tile / intrinsicSize);
      llvm::append_range(expandedTiles, factors);
    }
    expandedAttrs.set(name, b.getI64ArrayAttr(expandedTiles));
  }

  auto expandBasis = [&](IREE::GPU::TilingLevel level,
                         StringRef name) -> LogicalResult {
    Attribute rawBasis = config.getAttributes().get(name);
    if (!rawBasis) {
      return success();
    }
    FailureOr<IREE::GPU::Basis> maybeBasis = IREE::GPU::getBasis(config, level);
    if (failed(maybeBasis)) {
      return op.emitOpError("malformed ") << name;
    }
    IREE::GPU::Basis basis = *maybeBasis;
    if (basis.mapping.size() != static_cast<size_t>(rank)) {
      return op.emitOpError("malformed ") << name;
    }
    if (llvm::any_of(basis.counts, [](int64_t count) { return count <= 0; })) {
      return op.emitOpError("non-positive count in ") << name;
    }
    if (llvm::any_of(basis.mapping, [&](int64_t mappedDim) {
          return mappedDim < 0 ||
                 mappedDim >= static_cast<int64_t>(basis.counts.size());
        })) {
      return op.emitOpError("out-of-range mapping in ") << name;
    }
    llvm::SmallDenseSet<int64_t> seenMapping;
    if (llvm::any_of(basis.mapping, [&](int64_t mappedDim) {
          return !seenMapping.insert(mappedDim).second;
        })) {
      return op.emitOpError("duplicate mapping in ") << name;
    }
    SmallVector<int64_t> expandedMapping;
    int64_t nextBasisDim = basis.counts.size();
    for (auto [dim, mappedDim] : llvm::enumerate(basis.mapping)) {
      expandedMapping.push_back(mappedDim);
      if (!splitDims.contains(dim)) {
        continue;
      }
      for (size_t i = 0; i < factors.size(); ++i) {
        basis.counts.push_back(1);
        expandedMapping.push_back(nextBasisDim++);
      }
    }
    basis.mapping = std::move(expandedMapping);
    ArrayAttr basisAttr = b.getArrayAttr(
        {b.getI64ArrayAttr(basis.counts), b.getI64ArrayAttr(basis.mapping)});
    expandedAttrs.set(name, basisAttr);
    return success();
  };
  if (failed(expandBasis(IREE::GPU::TilingLevel::Subgroup, "subgroup_basis")) ||
      failed(expandBasis(IREE::GPU::TilingLevel::Thread, "lane_basis"))) {
    return failure();
  }
  expandedAttrs.set(kApplePhysicalFragmentExpansionMaterializedName,
                    b.getUnitAttr());

  auto expandedConfig = IREE::GPU::LoweringConfigAttr::get(
      op.getContext(), expandedAttrs.getDictionary(op.getContext()));
  return ApplePhysicalFragmentExpansion{expansion, expandedConfig};
}

// Compute the expanded shape for a reassociation group. Requires the original
// dimension to be static and evenly divisible by the product of static factors
// in the target shape.
static FailureOr<SmallVector<OpFoldResult>> computeExpandedGroupShape(
    RewriterBase &rewriter, Location loc, OpFoldResult origDimSize,
    ArrayRef<int64_t> groupTargetShape, unsigned iteratorDim) {
  if (groupTargetShape.size() == 1) {
    return SmallVector<OpFoldResult>{origDimSize};
  }

  std::optional<int64_t> staticOrigDim = getConstantIntValue(origDimSize);
  if (!staticOrigDim) {
    return rewriter.notifyMatchFailure(
        loc, "dimension " + Twine(iteratorDim) +
                 " is dynamic, but expand_dims requires static dimensions");
  }

  int64_t staticFactor = llvm::product_of(
      llvm::make_filter_range(groupTargetShape, ShapedType::isStatic));

  if (staticFactor < 1) {
    return rewriter.notifyMatchFailure(
        loc, "invalid expansion factor " + Twine(staticFactor) +
                 " for iterator dimension " + Twine(iteratorDim));
  }

  if (staticOrigDim.value() % staticFactor != 0) {
    return rewriter.notifyMatchFailure(
        loc, "dimension " + Twine(iteratorDim) +
                 " (size=" + Twine(staticOrigDim.value()) +
                 ") not divisible by expansion factor " + Twine(staticFactor));
  }

  return llvm::map_to_vector(
      groupTargetShape, [&](int64_t size) -> OpFoldResult {
        if (ShapedType::isStatic(size)) {
          return rewriter.getIndexAttr(size);
        }
        AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
        return affine::makeComposedFoldedAffineApply(
            rewriter, loc, s0.floorDiv(staticFactor), {origDimSize});
      });
}

// For an operation annotated with the `expand_dims` attribute, replace relevant
// operands with tensor.expand_shape/tensor.collapse_shape pair to materialize
// dimension expansion according to the reassociation and output_shape defined
// in the attribute.
//
// Example:
//
// ```mlir
// %0 = <some_op>(..., %0, ...) {
//   lowering_config = #iree_gpu.lowering_config<{
//     expand_dims = #iree_gpu.expand_dims
//       [[0], [1, 2]], output_shape = [?, ?, 8]>
//   }>
// } : ... -> tensor<4x128xf32>
// ```
//
// becomes:
//
// ```mlir
// %expanded = tensor.expand_shape %0 [[0], [1, 2]]
//     : tensor<4x128xf32> into tensor<4x16x8xf32>
// %barrier = util.optimization_barrier %expanded
// %collapsed = tensor.collapse_shape %barrier [[0], [1, 2]]
//     : tensor<4x16x8xf32> into tensor<4x128xf32>
// %1 = <some_op>(..., %collapsed, ...) : ... -> tensor<4x128xf32>
// ```
static std::optional<ReshapeOps>
createDimensionExpansionOps(RewriterBase &rewriter,
                            IREE::GPU::DimensionExpansionAttr config, Value v,
                            AffineMap indexingMap, linalg::LinalgOp op) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    return std::nullopt;
  }

  Location loc = v.getLoc();
  MLIRContext *ctx = op.getContext();
  int64_t tensorRank = tensorType.getRank();
  ArrayRef<int64_t> outputShape = config.getOutputShape().asArrayRef();
  SmallVector<OpFoldResult> origShape = tensor::getMixedSizes(rewriter, loc, v);

  // Map each tensor dimension to its expanded shape components.
  SmallVector<SmallVector<OpFoldResult>> expandedShapes(tensorRank);
  for (auto [iterDim, reassocIndices] :
       llvm::enumerate(config.getReassociationIndices())) {
    std::optional<unsigned> tensorDim =
        indexingMap.getResultPosition(getAffineDimExpr(iterDim, ctx));
    if (!tensorDim.has_value()) {
      continue;
    }

    auto groupOutputShape = llvm::map_to_vector(
        reassocIndices, [&](int64_t i) { return outputShape[i]; });

    FailureOr<SmallVector<OpFoldResult>> groupShape = computeExpandedGroupShape(
        rewriter, loc, origShape[tensorDim.value()], groupOutputShape, iterDim);
    if (failed(groupShape)) {
      return std::nullopt;
    }

    expandedShapes[tensorDim.value()] = std::move(groupShape.value());
  }

  // Build reassociation indices and expanded shape in tensor dimension order.
  SmallVector<ReassociationIndices> reassociation;
  SmallVector<OpFoldResult> expandedShape;
  for (auto [tensorDim, expanded] : llvm::enumerate(expandedShapes)) {
    ReassociationIndices &indices = reassociation.emplace_back();
    auto addDim = [&](OpFoldResult dim) {
      indices.push_back(expandedShape.size());
      expandedShape.push_back(dim);
    };
    if (expanded.empty()) {
      addDim(origShape[tensorDim]);
    } else {
      llvm::for_each(expanded, addDim);
    }
  }

  // If no expansion is needed, return early.
  if (llvm::equal(origShape, expandedShape)) {
    return std::nullopt;
  }

  auto staticShape = llvm::map_to_vector(expandedShape, [](OpFoldResult ofr) {
    return getConstantIntValue(ofr).value_or(ShapedType::kDynamic);
  });

  auto expandedType = RankedTensorType::get(
      staticShape, tensorType.getElementType(), tensorType.getEncoding());

  auto expandOp = tensor::ExpandShapeOp::create(rewriter, loc, expandedType, v,
                                                reassociation, expandedShape);
  Value barrier = IREE::Util::OptimizationBarrierOp::create(
                      rewriter, loc, expandOp.getResult())
                      .getResult(0);
  auto collapseOp = tensor::CollapseShapeOp::create(rewriter, loc, tensorType,
                                                    barrier, reassociation);

  return ReshapeOps{expandOp, collapseOp};
}

static LogicalResult
expandIterationSpace(RewriterBase &rewriter, linalg::LinalgOp op,
                     IREE::GPU::DimensionExpansionAttr config) {
  LDBG() << "Expanding dimensions for op: " << *op;

  for (OpOperand &operand : op->getOpOperands()) {
    AffineMap indexingMap = op.getMatchingIndexingMap(&operand);
    std::optional<ReshapeOps> reshapes = createDimensionExpansionOps(
        rewriter, config, operand.get(), indexingMap, op);
    if (reshapes.has_value()) {
      rewriter.modifyOpInPlace(
          op, [&]() { operand.set(reshapes.value().collapseShapeOp); });
    }
  }

  return success();
}

void GPUExpandDimensionsPass::runOnOperation() {
  Operation *operation = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<linalg::LinalgOp> worklist;
  operation->walk([&](linalg::LinalgOp op) {
    if (auto cfg = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op)) {
      auto physicalMma = getApplePhysicalFragmentMma(cfg);
      if ((expandApplePhysicalFragments && physicalMma) ||
          cfg.getAttributes().get(kDimensionExpansionName)) {
        worklist.push_back(op);
      }
    }
  });

  for (linalg::LinalgOp op : worklist) {
    rewriter.setInsertionPoint(op);
    auto loweringConfig = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
    assert(loweringConfig && "worklist op must have a lowering config");
    Attribute rawExpansion =
        loweringConfig.getAttributes().get(kDimensionExpansionName);
    auto expansion =
        dyn_cast_if_present<IREE::GPU::DimensionExpansionAttr>(rawExpansion);
    if (rawExpansion && !expansion) {
      op.emitOpError("malformed expand_dims lowering config");
      return signalPassFailure();
    }
    Attribute physicalExpansionMarker = loweringConfig.getAttributes().get(
        kApplePhysicalFragmentExpansionMaterializedName);
    if (physicalExpansionMarker && !isa<UnitAttr>(physicalExpansionMarker)) {
      op.emitOpError("malformed Apple physical fragment expansion marker");
      return signalPassFailure();
    }
    std::optional<IREE::GPU::MMAAttr> configuredPhysicalMma =
        getApplePhysicalFragmentMma(loweringConfig);
    if (configuredPhysicalMma && rawExpansion) {
      op.emitOpError(
          "Apple physical fragments cannot be combined with expand_dims");
      return signalPassFailure();
    }
    if (std::optional<IREE::GPU::MMAAttr> physicalMma =
            expandApplePhysicalFragments ? configuredPhysicalMma
                                         : std::nullopt) {
      if (!physicalExpansionMarker) {
        FailureOr<ApplePhysicalFragmentExpansion> physicalExpansion =
            getApplePhysicalFragmentExpansion(op, loweringConfig, *physicalMma);
        if (failed(physicalExpansion)) {
          return signalPassFailure();
        }
        expansion = physicalExpansion->dimensions;
        // Reshape propagation copies the source operation's attributes to the
        // expanded replacement. Install the rank-adjusted tiling/basis config
        // and materialization marker before running those patterns.
        setLoweringConfig(op, physicalExpansion->loweringConfig);
      } else {
        if (failed(validateApplePhysicalFragmentExpansionMaterialized(
                op, loweringConfig, *physicalMma))) {
          return signalPassFailure();
        }
        continue;
      }
    } else if (expansion) {
      if (expansion.getReassociationIndices().size() !=
          static_cast<size_t>(op.getNumLoops())) {
        op.emitOpError(
            "expand_dims reassociation count must match operation loop rank");
        return signalPassFailure();
      }
      // expand_dims is a one-shot transformation contract. Consume it before
      // propagation so a later invocation is explicitly idempotent instead of
      // guessing from a rank mismatch.
      NamedAttrList consumedAttrs(loweringConfig.getAttributes());
      consumedAttrs.erase(kDimensionExpansionName);
      setLoweringConfig(op, IREE::GPU::LoweringConfigAttr::get(
                                context, consumedAttrs.getDictionary(context)));
    }
    if (!expansion || failed(expandIterationSpace(rewriter, op, expansion))) {
      return signalPassFailure();
    }
  }

  LDBG() << "After expanding dimensions: " << *operation;

  ExpansionConfigTrackingListener listener;
  GreedyRewriteConfig config;
  config.setListener(&listener);
  linalg::ControlFusionFn controlFn = [](OpOperand *opOperand) {
    return !isa_and_nonnull<linalg::FillOp, tensor::EmptyOp>(
        opOperand->get().getDefiningOp());
  };

  {
    RewritePatternSet bubbleExpandShapePatterns(context);
    linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                      controlFn);
    IREE::LinalgExt::populateFoldReshapeOpsByExpansionPatterns(
        bubbleExpandShapePatterns, controlFn);
    IREE::Codegen::populateFoldReshapeOpsByExpansionPatterns(
        bubbleExpandShapePatterns, controlFn);
    populateReshapeToInterfaceTensorPatterns(bubbleExpandShapePatterns);
    populateFoldTensorReshapeIntoBufferPatterns(bubbleExpandShapePatterns);
    populateHoistReshapesFromLoopsPatterns(bubbleExpandShapePatterns);
    tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
    tensor::populateBubbleUpExpandShapePatterns(bubbleExpandShapePatterns);
    linalg::FillOp::getCanonicalizationPatterns(
        bubbleExpandShapePatterns, bubbleExpandShapePatterns.getContext());
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        bubbleExpandShapePatterns);
    if (failed(applyPatternsGreedily(
            operation, std::move(bubbleExpandShapePatterns), config))) {
      operation->emitOpError(
          "failed in application of bubble up expand shape patterns");
      return signalPassFailure();
    }
  }

  LDBG() << "After reshape propagation: " << *operation;

  {
    RewritePatternSet removeBarrierOpsPatterns(context);
    populateRemoveOptimizationBarrierPatterns(removeBarrierOpsPatterns);
    linalg::populateFoldReshapeOpsByExpansionPatterns(removeBarrierOpsPatterns,
                                                      controlFn);
    IREE::LinalgExt::populateFoldReshapeOpsByExpansionPatterns(
        removeBarrierOpsPatterns, controlFn);
    IREE::Codegen::populateFoldReshapeOpsByExpansionPatterns(
        removeBarrierOpsPatterns, controlFn);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(removeBarrierOpsPatterns,
                                                       context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(
        removeBarrierOpsPatterns, context);
    populateReshapeToInterfaceTensorPatterns(removeBarrierOpsPatterns);
    populateFoldTensorReshapeIntoBufferPatterns(removeBarrierOpsPatterns);
    populateHoistReshapesFromLoopsPatterns(removeBarrierOpsPatterns);
    tensor::populateFoldTensorEmptyPatterns(removeBarrierOpsPatterns);
    linalg::FillOp::getCanonicalizationPatterns(removeBarrierOpsPatterns,
                                                context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        removeBarrierOpsPatterns);
    ApplePhysicalConfigTrackingListener physicalConfigListener;
    GreedyRewriteConfig cleanupConfig;
    cleanupConfig.setListener(&physicalConfigListener);
    if (failed(applyPatternsGreedily(
            operation, std::move(removeBarrierOpsPatterns), cleanupConfig))) {
      operation->emitOpError("failed in cleanup patterns");
      return signalPassFailure();
    }
  }

  return;
}

} // namespace mlir::iree_compiler
