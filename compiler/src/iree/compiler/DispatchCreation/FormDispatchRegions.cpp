// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib> // iree-metal: getenv for aggressive-fusion bypass bisect guards

#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/LoopMappingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-dispatch-creation-form-dispatch-regions"

namespace mlir::iree_compiler::DispatchCreation {

using IREE::LinalgExt::getOuterParallelLoops;
using IREE::LinalgExt::getRootParallelLoopToOpMap;

#define GEN_PASS_DEF_FORMDISPATCHREGIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Root and fusion group handling
//===----------------------------------------------------------------------===//

namespace {
// `FusionGroup` is used to track operations that are to be fused with a given
// `rootOp`.
//
// This class contains an AffineMap for each operation to be fused. This map
// represents a mapping from the root op's outer parallel dims to this op's
// iteration space. `0` is used to represent when the iteration dimension has no
// mapping to the root op's outer parallel dimensions.
//
// For example:
//   affine_map<(d0, d1) -> (d0, 0, d1)>
//
// The root op has 2 outer parallel loops (`d0` and `d1`) and the example op
// has 3 dimensions where the first and last map `d0` and `d1` and the middle
// has no mapping to the root's outer parallel dimensions.
class FusionGroup {
public:
  FusionGroup(Operation *op) : rootOp(op) {
    llvm::SmallBitVector loops = getOuterParallelLoops(op);
    auto map = AffineMap::getFilteredIdentityMap(
        op->getContext(), loops.size(), [&](AffineDimExpr dimExpr) {
          return loops.test(dimExpr.getPosition());
        });
    map = inverseAndBroadcastProjectedPermutation(map);
    loopMaps.insert({op, map});
  };

  SmallVector<Operation *> getFusedOperations() const {
    return llvm::map_to_vector(
        loopMaps.getArrayRef(),
        [](std::pair<Operation *, AffineMap> pair) { return pair.first; });
  }

  Operation *getRoot() const { return rootOp; }

  // Get the mapping from `rootOp`'s outer parallel loops to `op`. This assumes
  // that the dependency chain from `rootOp` to `op` has already been inserted
  // into the group.
  //
  // Returns failure when there is no mapping or more than one mapping exists.
  FailureOr<AffineMap> getRootParallelLoopToOpMap(Operation *op) const;

  bool isFusable(Operation *op) const {
    // We only handle fusion across operation's operands. Don't fuse if the
    // operation is using values in the fusion group in it's body.
    bool hasUseFromAbove = false;
    mlir::visitUsedValuesDefinedAbove(
        op->getRegions(), [&](OpOperand *operand) {
          if (loopMaps.contains(operand->get().getDefiningOp())) {
            hasUseFromAbove = true;
          }
        });
    if (hasUseFromAbove) {
      return false;
    }

    FailureOr<AffineMap> maybeMap = getRootParallelLoopToOpMap(op);
    if (failed(maybeMap)) {
      return false;
    }

    // If the candidate is not all parallel, then its loop configuration should
    // be the same as the root.
    auto candidateOuterParallelLoop = getOuterParallelLoops(op);
    if (candidateOuterParallelLoop.size() !=
        candidateOuterParallelLoop.count()) {
      return loopMaps.lookup(rootOp) == maybeMap.value();
    }
    return true;
  }

  bool contains(Operation *op) const { return loopMaps.contains(op); }

  // Insert `op` into the fusion group.
  void insert(Operation *op);

  /// Returns true if `consumerOp` has a transitive dependency on the fusion
  /// group. This means that some transitive dependency of `consumerOp` (not in
  /// the fusion group) itself uses an operation in the fusion group. This is
  /// required for fusion because it must be legal to take a program slice that
  /// contains only the ops in the fusion group.
  bool
  hasTransitiveDependencyOnFusionGroup(Operation *consumerOp,
                                       DominanceInfo const &dominance) const {
    BackwardSliceOptions options;
    options.inclusive = true;
    options.omitUsesFromAbove = false;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *sliceBoundaryOp) {
      return !llvm::all_of(
          loopMaps.getArrayRef(), [&](std::pair<Operation *, AffineMap> pair) {
            return dominance.properlyDominates(sliceBoundaryOp, pair.first);
          });
    };

    llvm::SetVector<Operation *> slice;
    auto populateSlice = [&](OpOperand *operand) {
      // It's okay if the consumer directly uses an operation in the fusion
      // group.
      if (loopMaps.contains(operand->get().getDefiningOp())) {
        return;
      }
      LogicalResult result = getBackwardSlice(operand->get(), &slice, options);
      assert(result.succeeded() && "expected a backward slice");
      (void)result;
    };

    // Search all of the operands op `consumerOp` as well as all the values used
    // in its regions.
    mlir::visitUsedValuesDefinedAbove(consumerOp->getRegions(), populateSlice);
    for (OpOperand &operand : consumerOp->getOpOperands()) {
      populateSlice(&operand);
    }

    return llvm::any_of(loopMaps.getArrayRef(),
                        [&](std::pair<Operation *, AffineMap> pair) {
                          return slice.contains(pair.first);
                        });
  }

  // Check if adding `op` would exceed the operand limit.
  bool wouldExceedOperandLimit(Operation *op) const;

private:
  Operation *rootOp;
  // All operations to be fused with the root op. This does not include
  // `rootOp`.
  llvm::MapVector<Operation *, AffineMap> loopMaps;
};
} // namespace

void FusionGroup::insert(Operation *op) {
  assert(!contains(op) && "op already fused");
  FailureOr<AffineMap> map = getRootParallelLoopToOpMap(op);
  if (succeeded(map)) {
    loopMaps.insert({op, map.value()});
  } else {
    // TODO(IanWood1): some ops can be fused but don't implement
    // `LinalgFusionOpInterface` e.g. `tensor.insert_slice` or `linalg.unpack`.
    // `getRootParallelLoopToOpMap` fails when `op` is trying to fuse with one
    // of these ops. So, give `op` a root map.
    llvm::SmallBitVector loops = getOuterParallelLoops(op);
    auto map = AffineMap::getFilteredIdentityMap(
        op->getContext(), loops.size(), [&](AffineDimExpr dimExpr) {
          return loops.test(dimExpr.getPosition());
        });
    map = inverseAndBroadcastProjectedPermutation(map);
    loopMaps.insert({op, map});
  }
}

bool FusionGroup::wouldExceedOperandLimit(Operation *newOp) const {
  llvm::SmallSetVector<Operation *, kIreeMaxOperandCount> dispatchOperands;
  int64_t numResults = 0;

  auto visitOp = [&](Operation *op) {
    auto visitOperand = [&](OpOperand *operand) {
      if (!isa<RankedTensorType>(operand->get().getType())) {
        return;
      }
      Operation *definingOp = operand->get().getDefiningOp();
      if (llvm::isa_and_nonnull<linalg::FillOp, tensor::EmptyOp>(definingOp)) {
        return;
      }
      if (definingOp && definingOp != newOp && !loopMaps.contains(definingOp)) {
        dispatchOperands.insert(definingOp);
      }
    };
    visitUsedValuesDefinedAbove(op->getRegions(), visitOperand);
    llvm::for_each(llvm::make_pointer_range(op->getOpOperands()), visitOperand);

    for (OpResult result : op->getResults()) {
      if (llvm::any_of(result.getUsers(), [&](Operation *user) {
            return user != newOp && !loopMaps.contains(user);
          })) {
        ++numResults;
      }
    }
  };

  visitOp(newOp);
  for (auto [op, map] : this->loopMaps) {
    visitOp(op);
  }
  return (dispatchOperands.size() + numResults) > kIreeMaxOperandCount;
}

FailureOr<AffineMap>
FusionGroup::getRootParallelLoopToOpMap(Operation *op) const {
  assert(!contains(op) && "op cannot already be in group");
  return IREE::LinalgExt::getRootParallelLoopToOpMap(op, loopMaps);
}

namespace {

/// Tracks all the FusionGroups for the program.
class FusionTracker {
public:
  /// Create a new fusion group with `op` as the root.
  FusionGroup &createFusionGroup(MLIRContext *ctx, Operation *op) {
    fusionGroups.push_back(std::make_unique<FusionGroup>(op));
    opToGroup[op] = fusionGroups.back().get();
    return *fusionGroups.back();
  }

  // Get the fusion group that contains `op`.
  const FusionGroup &getFusionGroup(Operation *op) const {
    return *opToGroup.at(op);
  }

  // Get the fusion group that contains `op`.
  FusionGroup &getFusionGroup(Operation *op) { return *opToGroup.at(op); }

  const SmallVector<std::unique_ptr<FusionGroup>> &getFusionGroups() const {
    return fusionGroups;
  }

  void appendToFusionGroup(Operation *op, FusionGroup &fusionGroup) {
    assert(!isFusedOp(op) && "op already in a group");
    fusionGroup.insert(op);
    opToGroup[op] = &fusionGroup;
  }

  // Returns if `op` has been added to a FusionGroup in the tracker.
  bool isFusedOp(Operation *op) const { return opToGroup.contains(op); }

  // Returns if `op` is the root of a FusionGroup.
  bool isRootOp(Operation *op) const {
    return isFusedOp(op) && op == getFusionGroup(op).getRoot();
  }

private:
  SmallVector<std::unique_ptr<FusionGroup>> fusionGroups;
  DenseMap<Operation *, FusionGroup *> opToGroup;
};
} // namespace

//===----------------------------------------------------------------------===//
// Op property charecterizations
//===----------------------------------------------------------------------===//

/// Returns true if the reduced dimensions in the linalgOp of the unpack result
/// are not unpacked by the producer linalg::UnPackOp. This means the reduced
/// dimensions of the unpack result are not part of the inner_dims_pos.
static bool hasNoPackedReductionDimensions(linalg::LinalgOp linalgOp,
                                           Operation *producer) {
  auto unpack = dyn_cast<linalg::UnPackOp>(producer);
  if (!unpack) {
    return false;
  }
  AffineMap map;
  for (auto &use : producer->getResult(0).getUses()) {
    if (use.getOwner() == linalgOp) {
      map = linalgOp.getMatchingIndexingMap(&use);
      break;
    }
  }
  if (!map) {
    return false;
  }
  auto iterators = linalgOp.getIteratorTypesArray();
  auto reduction = utils::IteratorType::reduction;
  for (auto expr : llvm::enumerate(map.getResults())) {
    auto dim = dyn_cast<AffineDimExpr>(expr.value());
    if (!dim) {
      return false;
    }
    unsigned pos = dim.getPosition();
    if (iterators[pos] == reduction &&
        llvm::any_of(unpack.getInnerDimsPos(),
                     [expr](int64_t idp) { return expr.index() == idp; })) {
      return false;
    }
  }
  return true;
}

/// Returns true if the linalgOp is fusable with an unpack producer
static bool hasFusableUnpackProducer(linalg::LinalgOp linalgOp) {
  return llvm::any_of(linalgOp->getOperands(), [&](Value operand) {
    auto producer = operand.getDefiningOp<linalg::UnPackOp>();
    return producer && hasNoPackedReductionDimensions(linalgOp, producer);
  });
}

/// Operations that are treated as root operations for dispatch region
/// formation.
static bool isRootLikeOp(Operation *op) {
  if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
    return false;
  }
  // Dequantization-like ops get cloned into dispatches later.
  if (IREE::LinalgExt::isBitExtendOp(op)) {
    return false;
  }
  // Any Linalg named op or generic op with reduction iterator types is a root
  // op.
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (isa<linalg::GenericOp>(op)) {
      return linalgOp.getNumReductionLoops() != 0 &&
             !hasFusableUnpackProducer(linalgOp);
    }
    return !isa<linalg::FillOp>(op);
  }
  if (isa<TilingInterface>(op)) {
    return !isa<IREE::LinalgExt::GatherOp, tensor::PadOp, tensor::ConcatOp,
                linalg::PackOp>(op);
  }
  return isa<linalg::UnPackOp>(op);
}

/// Returns true if the operation is a `pack` op or a `set_encoding` op that
/// has pack semantics.
// TODO(ravishankarm): This seems like a use case for an interface.
static bool isPackLikeOp(Operation *op) {
  return isa<IREE::Encoding::SetEncodingOp, linalg::PackOp>(op);
}

/// Returns true if the operation is an `unpack` op or an `unset_encoding` op.
static bool isUnpackLikeOp(Operation *op) {
  return isa<IREE::Encoding::UnsetEncodingOp, linalg::UnPackOp>(op);
}

//===----------------------------------------------------------------------===//
// Heuristics for fusing dispatchble ops with root ops using tile + fuse.
//===----------------------------------------------------------------------===//

// Returns true for the canonical JAX one-hot expansion: an integer row label
// is broadcast across the final output dimension and compared with
// linalg.index. Fusing this cheap predicate into its earliest consumer avoids
// a standalone full-vocabulary dispatch; the fused root can still return the
// value when a later sibling needs it.
static bool isSparseOneHotLike(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getNumDpsInits() != 1 ||
      genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
    return false;
  }
  auto inputType = dyn_cast<RankedTensorType>(
      genericOp.getDpsInputOperand(0)->get().getType());
  auto outputType = dyn_cast<RankedTensorType>(
      genericOp.getDpsInitOperand(0)->get().getType());
  if (!inputType || !outputType || inputType.getRank() + 1 != outputType.getRank() ||
      !inputType.getElementType().isIntOrIndex() ||
      !isa<FloatType>(outputType.getElementType())) {
    return false;
  }
  AffineMap inputMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap outputMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (!outputMap.isIdentity() ||
      inputMap.getNumResults() != static_cast<unsigned>(inputType.getRank())) {
    return false;
  }
  for (auto [position, result] : llvm::enumerate(inputMap.getResults())) {
    auto dim = dyn_cast<AffineDimExpr>(result);
    if (!dim || dim.getPosition() != position) {
      return false;
    }
  }
  Block *body = genericOp.getBlock();
  if (std::distance(body->begin(), body->end()) != 5) {
    return false;
  }
  auto it = body->begin();
  return isa<linalg::IndexOp>(*it++) && isa<arith::IndexCastOp>(*it++) &&
         isa<arith::CmpIOp>(*it++) && isa<arith::UIToFPOp>(*it++) &&
         isa<linalg::YieldOp>(*it);
}

// Returns true only for a shape-preserving floating-point extension. Folding
// this producer into a reduction is lossless and cannot change the reduction's
// arithmetic topology; it only avoids materializing the widened tensor.
static bool isLosslessWideningCastLike(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getNumDpsInits() != 1 ||
      genericOp.getNumLoops() != genericOp.getNumParallelLoops() ||
      !genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0))
           .isIdentity() ||
      !genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0))
           .isIdentity()) {
    return false;
  }
  Block *body = genericOp.getBlock();
  if (std::distance(body->begin(), body->end()) != 2) {
    return false;
  }
  auto extension = dyn_cast<arith::ExtFOp>(&body->front());
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  return extension && yield &&
         extension.getIn() == body->getArgument(0) &&
         yield.getValues().size() == 1 &&
         yield.getValues().front() == extension.getResult();
}

// Replaces selection reductions over one_hot(labels) with a sparse load of the
// selected logits. JAX can spell the selection as either
// `logits * one_hot` or `select(one_hot != 0, logits, 0)`. Besides avoiding a
// full-vocabulary read in the forward loss, removing this use leaves the cheap
// one-hot predicate single-use so it can fuse into the elementwise gradient.
// This also keeps linalg.index out of a fused reduction: producer indices are
// local to a reduction tile and cannot represent the global vocabulary column.
struct RewriteSparseOneHotSelection final
    : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp reduction,
                                PatternRewriter &rewriter) const override {
    if (reduction.getNumDpsInputs() != 2 ||
        reduction.getNumDpsInits() != 1 || reduction.getNumLoops() != 2 ||
        reduction.getNumParallelLoops() != 1 ||
        reduction.getNumReductionLoops() != 1) {
      return failure();
    }

    OpOperand *firstOperand = reduction.getDpsInputOperand(0);
    OpOperand *secondOperand = reduction.getDpsInputOperand(1);
    auto firstOneHot = firstOperand->get().getDefiningOp<linalg::GenericOp>();
    auto secondOneHot = secondOperand->get().getDefiningOp<linalg::GenericOp>();
    bool firstIsOneHot = firstOneHot && isSparseOneHotLike(firstOneHot);
    bool secondIsOneHot = secondOneHot && isSparseOneHotLike(secondOneHot);
    if (firstIsOneHot == secondIsOneHot) {
      return failure();
    }
    OpOperand *oneHotOperand = firstIsOneHot ? firstOperand : secondOperand;
    OpOperand *logitsOperand = firstIsOneHot ? secondOperand : firstOperand;
    linalg::GenericOp oneHotOp = firstIsOneHot ? firstOneHot : secondOneHot;
    auto logitsType =
        dyn_cast<RankedTensorType>(logitsOperand->get().getType());
    auto resultType =
        dyn_cast<RankedTensorType>(reduction.getResult(0).getType());
    if (!logitsType || logitsType.getRank() != 2 || !resultType ||
        resultType.getRank() != 1 ||
        logitsType.getElementType() != resultType.getElementType()) {
      return failure();
    }

    AffineMap identity2 =
        AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
    AffineMap rowMap = AffineMap::get(
        2, 0, rewriter.getAffineDimExpr(0), rewriter.getContext());
    if (reduction.getMatchingIndexingMap(logitsOperand) != identity2 ||
        reduction.getMatchingIndexingMap(oneHotOperand) != identity2 ||
        reduction.getMatchingIndexingMap(reduction.getDpsInitOperand(0)) !=
            rowMap) {
      return failure();
    }

    Block *body = reduction.getBlock();
    auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!yield || yield.getValues().size() != 1) {
      return failure();
    }
    Value firstArgument = body->getArgument(0);
    Value secondArgument = body->getArgument(1);
    Value oneHotArgument = firstIsOneHot ? firstArgument : secondArgument;
    Value logitsArgument = firstIsOneHot ? secondArgument : firstArgument;
    Value accumulatorArgument = body->getArgument(2);
    auto isAccumulatingAdd = [&](arith::AddFOp add, Value selected) {
      return add &&
             ((add.getLhs() == selected &&
               add.getRhs() == accumulatorArgument) ||
              (add.getRhs() == selected &&
               add.getLhs() == accumulatorArgument)) &&
             yield.getValues().front() == add.getResult();
    };

    bool matchesSelection = false;
    if (std::distance(body->begin(), body->end()) == 3) {
      auto multiply = dyn_cast<arith::MulFOp>(&body->front());
      auto add = dyn_cast<arith::AddFOp>(body->front().getNextNode());
      matchesSelection =
          multiply &&
          ((multiply.getLhs() == logitsArgument &&
            multiply.getRhs() == oneHotArgument) ||
           (multiply.getLhs() == oneHotArgument &&
            multiply.getRhs() == logitsArgument)) &&
          isAccumulatingAdd(add, multiply.getResult());
    } else if (std::distance(body->begin(), body->end()) == 4) {
      auto compare = dyn_cast<arith::CmpFOp>(&body->front());
      auto select =
          dyn_cast<arith::SelectOp>(body->front().getNextNode());
      auto add = dyn_cast<arith::AddFOp>(
          body->front().getNextNode()->getNextNode());
      bool comparesOneHotToZero =
          compare && compare.getPredicate() == arith::CmpFPredicate::UNE &&
          ((compare.getLhs() == oneHotArgument &&
            matchPattern(compare.getRhs(), m_AnyZeroFloat())) ||
           (compare.getRhs() == oneHotArgument &&
            matchPattern(compare.getLhs(), m_AnyZeroFloat())));
      matchesSelection =
          comparesOneHotToZero && select &&
          select.getCondition() == compare.getResult() &&
          select.getTrueValue() == logitsArgument &&
          matchPattern(select.getFalseValue(), m_AnyZeroFloat()) &&
          isAccumulatingAdd(add, select.getResult());
    }
    if (!matchesSelection) {
      return failure();
    }

    auto fill =
        reduction.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>();
    if (!fill || !matchPattern(fill.getInputs().front(), m_AnyZeroFloat())) {
      return failure();
    }

    Value labels = oneHotOp.getDpsInputOperand(0)->get();
    auto labelsType = dyn_cast<RankedTensorType>(labels.getType());
    if (!labelsType || labelsType.getRank() != 1 ||
        !labelsType.getElementType().isIntOrIndex()) {
      return failure();
    }

    // Retained mixed-precision logits are commonly widened from BF16 before
    // the selection. Read the retained tensor directly and widen only the one
    // selected scalar per row. Capturing the source in a rank-1 parallel
    // generic is intentional: unlike map_load, it cannot ask the tiler for a
    // rectangular tile of an indirectly-indexed vocabulary dimension.
    Value sparseSource = logitsOperand->get();
    if (auto extension = sparseSource.getDefiningOp<linalg::GenericOp>()) {
      Block *extensionBody = extension.getBlock();
      if (extension.getNumDpsInputs() == 1 &&
          extension.getNumDpsInits() == 1 &&
          extension.getNumLoops() == extension.getNumParallelLoops() &&
          extension.getMatchingIndexingMap(extension.getDpsInputOperand(0))
                  .isIdentity() &&
          extension.getMatchingIndexingMap(extension.getDpsInitOperand(0))
                  .isIdentity() &&
          std::distance(extensionBody->begin(), extensionBody->end()) == 2) {
        auto ext = dyn_cast<arith::ExtFOp>(&extensionBody->front());
        auto extensionYield =
            dyn_cast<linalg::YieldOp>(extensionBody->getTerminator());
        auto sourceType = dyn_cast<RankedTensorType>(
            extension.getDpsInputOperand(0)->get().getType());
        if (ext && extensionYield && sourceType &&
            ext.getIn() == extensionBody->getArgument(0) &&
            extensionYield.getValues().size() == 1 &&
            extensionYield.getValues().front() == ext.getResult() &&
            sourceType.getShape() == logitsType.getShape() &&
            isa<FloatType>(sourceType.getElementType())) {
          sparseSource = extension.getDpsInputOperand(0)->get();
        }
      }
    }

    Location loc = reduction.getLoc();
    Value empty = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType());
    SmallVector<AffineMap> maps(
        2, rewriter.getMultiDimIdentityMap(resultType.getRank()));
    SmallVector<utils::IteratorType> iterators(
        resultType.getRank(), utils::IteratorType::parallel);
    auto sparseSelection = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{labels},
        ValueRange{empty}, maps, iterators,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
          Value row = linalg::IndexOp::create(builder, bodyLoc, 0);
          Value label = args[0];
          if (!label.getType().isIndex()) {
            label = arith::IndexCastOp::create(
                builder, bodyLoc, builder.getIndexType(), label);
          }
          Value selected = tensor::ExtractOp::create(
              builder, bodyLoc, sparseSource, ValueRange{row, label});
          if (selected.getType() != resultType.getElementType()) {
            selected = arith::ExtFOp::create(
                builder, bodyLoc, resultType.getElementType(), selected);
          }
          linalg::YieldOp::create(builder, bodyLoc, selected);
        });
    rewriter.replaceOp(reduction, sparseSelection.getResult(0));
    return success();
  }
};

// After the sparse forward selection has been rewritten, the remaining
// full-vocabulary one-hot is only used as a boolean predicate by the dense
// backward update. Replace that tensor operand with the rank-1 labels and a
// rank-1 column iota, then form the predicate directly in the consumer. The
// small materialized iota avoids an integer divide/remainder per dense element
// that a tiled linalg.index would otherwise introduce, while still avoiding
// the 4096xV mask.
struct FuseSparseOneHotPredicate final
    : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    linalg::GenericOp oneHot;
    unsigned oneHotOperandNumber = 0;
    for (OpOperand *operand : consumer.getDpsInputOperands()) {
      auto candidate = operand->get().getDefiningOp<linalg::GenericOp>();
      if (!candidate || !isSparseOneHotLike(candidate)) {
        continue;
      }
      if (oneHot) {
        return failure();
      }
      oneHot = candidate;
      oneHotOperandNumber = operand->getOperandNumber();
    }
    if (!oneHot || !oneHot->hasOneUse()) {
      return failure();
    }

    Block *body = consumer.getBlock();
    BlockArgument oneHotArgument = body->getArgument(oneHotOperandNumber);
    if (!oneHotArgument.hasOneUse()) {
      return failure();
    }
    auto compare = dyn_cast<arith::CmpFOp>(*oneHotArgument.user_begin());
    if (!compare || compare.getPredicate() != arith::CmpFPredicate::UNE ||
        !((compare.getLhs() == oneHotArgument &&
           matchPattern(compare.getRhs(), m_AnyZeroFloat())) ||
          (compare.getRhs() == oneHotArgument &&
           matchPattern(compare.getLhs(), m_AnyZeroFloat())))) {
      return failure();
    }

    Value labels = oneHot.getDpsInputOperand(0)->get();
    auto labelsType = cast<RankedTensorType>(labels.getType());
    auto oneHotType = cast<RankedTensorType>(oneHot.getResult(0).getType());
    auto columnsType = RankedTensorType::get(
        {oneHotType.getDimSize(1)}, labelsType.getElementType());
    Value columnsEmpty = tensor::EmptyOp::create(
        rewriter, oneHot.getLoc(), columnsType.getShape(),
        columnsType.getElementType());
    auto columns = linalg::GenericOp::create(
        rewriter, oneHot.getLoc(), TypeRange{columnsType}, ValueRange{},
        ValueRange{columnsEmpty},
        ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(1)},
        ArrayRef<utils::IteratorType>{utils::IteratorType::parallel},
        [&](OpBuilder &builder, Location loc, ValueRange) {
          Value column = linalg::IndexOp::create(builder, loc, 0);
          if (!labelsType.getElementType().isIndex()) {
            column = arith::IndexCastOp::create(
                builder, loc, labelsType.getElementType(), column);
          }
          linalg::YieldOp::create(builder, loc, column);
        });
    columns->setAttr("iree-metal.sparse-one-hot-columns",
                     UnitAttr::get(rewriter.getContext()));

    unsigned oldInputCount = consumer.getNumDpsInputs();
    SmallVector<Value> inputs(consumer.getDpsInputs());
    inputs[oneHotOperandNumber] = labels;
    inputs.push_back(columns.getResult(0));
    SmallVector<AffineMap> maps = consumer.getIndexingMapsArray();
    maps[oneHotOperandNumber] = AffineMap::get(
        consumer.getNumLoops(), 0, rewriter.getAffineDimExpr(0),
        rewriter.getContext());
    maps.insert(maps.begin() + oldInputCount,
                AffineMap::get(consumer.getNumLoops(), 0,
                               rewriter.getAffineDimExpr(1),
                               rewriter.getContext()));
    auto replacement = linalg::GenericOp::create(
        rewriter, consumer.getLoc(), consumer->getResultTypes(), inputs,
        consumer.getDpsInits(), maps,
        consumer.getIteratorTypesArray(),
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          IRMapping mapping;
          for (auto [index, oldArgument] :
               llvm::enumerate(body->getArguments())) {
            if (index == oneHotOperandNumber) {
              continue;
            }
            unsigned newIndex = index < oldInputCount ? index : index + 1;
            mapping.map(oldArgument, args[newIndex]);
          }
          Value label = args[oneHotOperandNumber];
          Value column = args[oldInputCount];
          Value predicate = arith::CmpIOp::create(
              builder, loc, arith::CmpIPredicate::eq, label, column);
          mapping.map(compare.getResult(), predicate);
          for (Operation &operation : body->getOperations()) {
            if (&operation == compare.getOperation()) {
              continue;
            }
            builder.clone(operation, mapping);
          }
        });
    rewriter.replaceOp(consumer, replacement.getResults());
    return success();
  }
};

/// For all uses of an operation, return the uses that could be fused.
/// The returned vector contains the uses in dominance order.
static SmallVector<OpOperand *>
getFusableUses(MLIRContext *context, Operation *op,
               DominanceInfo const &dominanceInfo, bool aggressiveFusion) {
  // iree-metal (SHIPPED 2026-07-17): multi-use fusion under aggressive fusion double-counts gradients
  // when a fan-out (multi-use) producer is cloned into a REDUCTION consumer — the residual
  // (x = x + sublayer(x)) makes a LayerNorm/bias-backward reduction multi-use, and aggressive
  // fusion recomputes the producer inside the reduction's tiling with a wrong range -> +16% (bert
  // 34x) grad over-count. FIX: keep the single-use restriction even under aggressive fusion
  // (blanket). Validated bit-exact grads vs non-aggressive on gpt2-small/bert/distilbert/roberta/vit
  // and +1.5-2.6% fwd+bwd. A consumer-side narrow (restrict only reduction consumers) was tried and
  // REJECTED: it's correct but the partial-fusion graph makes metal-spirv compile blow up (>7min/
  // model). Escape hatch IREE_METAL_ALLOW_MULTIUSE_RED = original (faster but training-INCORRECT) multi-
  // use fusion, for A/B only. Fully closing the fusion gap needs correct fused reduction+matmul
  // codegen (deep, in progress) — then this restriction can lift.
  bool restrictSingleUse = !aggressiveFusion || !getenv("IREE_METAL_ALLOW_MULTIUSE_RED");
  if (restrictSingleUse &&
      llvm::count_if(op->getUses(), [](OpOperand &use) {
        return !isa<tensor::DimOp>(use.getOwner());
      }) != 1) {
    return {};
  }

  // Collect all fusable user candidates.
  SetVector<OpOperand *> fusableUses;
  for (OpOperand &use : op->getUses()) {
    Operation *user = use.getOwner();
    if (isa<tensor::DimOp>(user)) {
      continue;
    }
    if (op->getBlock() != user->getBlock()) {
      continue;
    }
    fusableUses.insert(&use);
  }

  SmallVector<OpOperand *> usesVec = fusableUses.takeVector();
  llvm::sort(usesVec, [&](OpOperand *lhsUse, OpOperand *rhsUse) {
    return dominanceInfo.properlyDominates(lhsUse->getOwner(),
                                           rhsUse->getOwner());
  });

  return usesVec;
}

/// For the fusion of root op -> elementwise operation to be bufferized
/// in-place without use of extra memory, the result of the root operation
/// must be able to reuse the buffer for the result of the elementwise
/// operation. Check if that is possible for the input/init operand pair.
static bool canUseInOperandAsInitOperand(OpOperand *inOperand,
                                         OpOperand *initOperand) {
  assert(inOperand->getOwner() == initOperand->getOwner() &&
         "expected in-operand and init-operand to be owned by same operation");

  // Check that the owner is a `generic` op.
  auto genericOp = dyn_cast<linalg::GenericOp>(inOperand->getOwner());
  if (!genericOp) {
    return false;
  }

  // All loops to be parallel.
  if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
    return false;
  }

  /// The input operand cannot be an init operand already.
  if (genericOp.isDpsInit(inOperand)) {
    return false;
  }

  // If the init operand value is used it cannot be reused for the input
  // operand.
  if (genericOp.payloadUsesValueFromOperand(initOperand)) {
    return false;
  }

  // Indexing map used to access the input and init have to match.
  if (genericOp.getMatchingIndexingMap(inOperand) !=
      genericOp.getMatchingIndexingMap(initOperand)) {
    return false;
  }

  // Types have to match for the input operand to reuse the buffer from the init
  // operand
  if (inOperand->get().getType() != initOperand->get().getType()) {
    return false;
  }

  return true;
}

/// Returns true if this is a fusable use, while fusing a root with its
/// consumer.
static bool
isFusableWithConsumer(OpOperand &fusedOperand, const FusionTracker &tracker,
                      FormDispatchRegionsPassOptions const &options) {
  Operation *producer = fusedOperand.get().getDefiningOp();
  Operation *consumer = fusedOperand.getOwner();

  // (iree-metal 2026-07-21: a transpose-split guard here was a DEAD-END — the transpose co-locates with the
  // reduction via a non-consumer-fusion path; reverted. See CAMPAIGN.md ledger.)

  // If consumer is a dequant operation, dont fuse it. These get cloned
  // into their consumers.
  IREE::Flow::CloneableIntoDispatchOptions cloneableOptions;
  cloneableOptions.aggressive = options.aggressiveFusion;
  if (IREE::Flow::isCloneableIntoDispatchOp(consumer, cloneableOptions)) {
    return false;
  }

  // Fuse unset_encoding operations with `tensor.extract_slice` and elementwise
  // generic ops.
  if (isUnpackLikeOp(producer)) {
    // Fuse `unset_encoding/unpack` -> elementwise operations. Fuse unpack with
    // non-overlapping reductions (i.e., the reduction dimension is not packed).
    if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumer)) {
      if (hasNoPackedReductionDimensions(consumerLinalgOp, producer)) {
        return true;
      }
      return linalg::isElementwise(consumerLinalgOp) &&
             consumerLinalgOp.getNumLoops() ==
                 cast<RankedTensorType>(producer->getResult(0).getType())
                     .getRank();
    }
    return false;
  }

  if (isPackLikeOp(consumer)) {
    return TypeSwitch<Operation *, bool>(producer)
        .Case([&](tensor::PadOp padOp) { return true; })
        .Case([&](linalg::LinalgOp linalgOp) {
          AffineMap producerIndexingMap = linalgOp.getIndexingMapMatchingResult(
              cast<OpResult>(fusedOperand.get()));
          // Make sure the producer op has an identity result indexing map. As
          // CPU backend currently can't handle transpose between fused ops.
          return producerIndexingMap.isIdentity();
        })
        .Default(false);
  }

  // By default, padding should be fused with producers. It is hard to square
  // this with fusion of pad with consumer. So for now split the difference.
  // Either fuse pad with producer or with consumer.
  if (auto padOp = dyn_cast<tensor::PadOp>(consumer)) {
    if (options.fusePadWithProducers) {
      return isa<linalg::LinalgOp>(producer);
    }
    return false;
  }

  // Insert slice ops should always be fused with their producers.
  if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(consumer)) {
    // TODO: Enable multi-use slice source fusion.
    Value source = insertSliceOp.getSource();
    if (!source.hasOneUse() || source.getDefiningOp() != producer) {
      return false;
    }
    // Fuse in `insert_slice` consumer operations if destination is a fill.
    // TODO: This can be generalized, but destination cannot be a
    // `arith.constant` or other constant-like objects. `linalg.fill` captures a
    // common case of pad generalization.
    return insertSliceOp.getDest().getDefiningOp<linalg::FillOp>();
  }

  // TODO(#16025): Enable mmt4d fusion. It is disabled because the backends
  // can not set multi lowering_config properly. See the issue for more details.
  if (isa<linalg::Mmt4DOp, linalg::BatchMmt4DOp>(producer)) {
    return false;
  }

  auto producerFusionOp =
      dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(producer);
  auto consumerFusionOp =
      dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(consumer);
  if (!producerFusionOp || !consumerFusionOp) {
    return false;
  }

  // Check that the consumer is all parallel.
  if (consumerFusionOp.getNumLoops() !=
      consumerFusionOp.getNumParallelLoops()) {
    return false;
  }

  if (!tracker.getFusionGroup(producer).isFusable(consumer)) {
    return false;
  }

  // Check operand limit before allowing fusion
  if (tracker.getFusionGroup(producer).wouldExceedOperandLimit(consumer)) {
    return false;
  }

  // Check if the iteration spaces of the producer and consumer are same.
  // TODO(#12664): This is unnecessary requirement, but we need a better config
  // to tile the consumer with a larger iteration space.
  // iree-metal bisect: IREE_METAL_ITERSPACE_GUARD restores the producer<consumer iteration-space
  // block under aggressive fusion (isolates a small reduction fused into a larger consumer).
  if (!options.aggressiveFusion || getenv("IREE_METAL_ITERSPACE_GUARD")) {
    // FIXME: Implement getStaticLoopRanges for LinalgExt::CustomOp.
    if (isa<IREE::LinalgExt::CustomOp>(producer)) {
      return false;
    }

    SmallVector<int64_t> producerIterationSpace =
        producerFusionOp.getStaticLoopRanges();
    SmallVector<int64_t> consumerIterationSpace =
        consumerFusionOp.getStaticLoopRanges();
    if (producerIterationSpace.size() < consumerIterationSpace.size()) {
      return false;
    }
  }

  // Block fusion if the consumer has more non-unit loops than the producer's
  // fusion group root. This prevents fusing cases where a small reduction
  // result is broadcast to a much larger consumer (e.g., batchn.
  // patterns). Unit dimensions are ignored..
  Operation *rootOp = tracker.getFusionGroup(producer).getRoot();
  if (auto rootFusionOp =
          dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(rootOp);
      rootFusionOp && !isa<IREE::LinalgExt::CustomOp>(rootOp)) {
    SmallVector<int64_t> rootLoopRanges = rootFusionOp.getStaticLoopRanges();
    SmallVector<int64_t> consumerLoopRanges =
        consumerFusionOp.getStaticLoopRanges();
    auto countNonUnitDims = [](ArrayRef<int64_t> ranges) {
      return llvm::count_if(ranges, [](int64_t size) { return size != 1; });
    };
    if (countNonUnitDims(consumerLoopRanges) >
        countNonUnitDims(rootLoopRanges)) {
      return false;
    }
  }

  // Under aggressive fusion assume that the dispatches are vectorized. In which
  // case we dont need to account for the subsequent stack allocation condition.
  // iree-metal bisect: IREE_METAL_STACKALLOC_GUARD makes aggressive fusion still run the
  // stack-allocation bufferization check below (isolates that bypass).
  if (options.aggressiveFusion && !getenv("IREE_METAL_STACKALLOC_GUARD")) {
    return true;
  }

  // While fusing with consumer, the result of the root might not be the final
  // result of the dispatch. To avoid a stack allocation we have to ensure that
  // all operations can bufferize without needing additional memory.
  auto consumerDstOp =
      dyn_cast<DestinationStyleOpInterface>(consumerFusionOp.getOperation());
  if (!consumerDstOp) {
    return true;
  }

  for (OpOperand *inputOperand : consumerDstOp.getDpsInputOperands()) {
    if (inputOperand->get().getDefiningOp() != producer) {
      continue;
    }
    if (isa<linalg::ConvolutionOpInterface>(producer) &&
        llvm::none_of(
            consumerDstOp.getDpsInitsMutable(), [&](OpOperand &initOperand) {
              return canUseInOperandAsInitOperand(inputOperand, &initOperand);
            })) {
      return false;
    }
  }

  return true;
}

/// Fuses roots with its consumers. If a root is fused with its consumer, it is
/// no more tagged as a root to aid with the dispatch region formation.
static void
fuseRootsWithConsumers(MLIRContext *context, ArrayRef<Operation *> roots,
                       DominanceInfo const &dominanceInfo,
                       FormDispatchRegionsPassOptions const &options,
                       FusionTracker &tracker) {
  // Fuse with consumers where possible.
  for (Operation *root : roots) {
    SmallVector<Operation *> workList;
    FusionGroup &fusionGroup = tracker.getFusionGroup(root);
    workList.push_back(root);
    while (!workList.empty()) {
      Operation *currRoot = workList.pop_back_val();

      SmallVector<OpOperand *> fusableUses =
          getFusableUses(context, currRoot, dominanceInfo,
                         /*aggressiveFusion=*/options.aggressiveFusion);
      if (fusableUses.empty()) {
        continue;
      }

      // Analyse the use to see if it is fusable.
      for (OpOperand *fusableUse : fusableUses) {
        Operation *consumerOp = fusableUse->getOwner();
        if (tracker.isRootOp(consumerOp) || tracker.isFusedOp(consumerOp)) {
          continue;
        }

        // Ensure that fusing the consumer would not cause use-def violations.
        if (tracker.getFusionGroup(currRoot)
                .hasTransitiveDependencyOnFusionGroup(fusableUse->getOwner(),
                                                      dominanceInfo)) {
          continue;
        }

        if (isFusableWithConsumer(*fusableUse, tracker, options)) {
          tracker.appendToFusionGroup(consumerOp, fusionGroup);
          workList.push_back(consumerOp);
        }
      }
    }
  }
}

/// Method to check if the consumer of a use can be fused with its producer.
static bool isFusableWithProducer(OpOperand &operand,
                                  const FusionTracker &tracker,
                                  FormDispatchRegionsPassOptions const &options,
                                  bool fuseWithTruncate) {
  Operation *producer = operand.get().getDefiningOp();
  Operation *consumer = operand.getOwner();

  if (!fuseWithTruncate && IREE::LinalgExt::isBitTruncateOp(producer)) {
    return false;
  }

  if (auto padOp = dyn_cast<tensor::PadOp>(consumer)) {
    if (options.fusePadWithProducers) {
      return isa<linalg::LinalgOp>(producer);
    }
    return false;
  }

  auto linalgConsumer = dyn_cast<linalg::LinalgOp>(consumer);
  if (options.fusePadWithConsumers && isa<tensor::PadOp>(producer) &&
      linalgConsumer && linalg::isaConvolutionOpInterface(linalgConsumer)) {
    return true;
  }

  if (auto attentionOp = dyn_cast<IREE::LinalgExt::AttentionOp>(consumer)) {
    // Disable all other producer fusion. TODO: Enable some producer fusions.
    return false;
  }

  if (isPackLikeOp(consumer)) {
    return TypeSwitch<Operation *, bool>(producer)
        .Case([&](tensor::PadOp padOp) { return true; })
        .Case([&](linalg::LinalgOp linalgOp) {
          if (auto packOp = dyn_cast<linalg::PackOp>(consumer)) {
            // TODO(#12746): fusion of pack with dynamic inner tile size
            // causes an error in backend. Disable for now.
            if (!packOp.getInnerTiles().empty()) {
              return false;
            }
          }
          AffineMap producerIndexingMap = linalgOp.getIndexingMapMatchingResult(
              cast<OpResult>(operand.get()));
          // Make sure the producer op has an identity result indexing map. As
          // CPU backend currently can't handle transpose between fused ops.
          return producerIndexingMap.isIdentity();
        })
        .Default(false);
  }

  if (!isa<IREE::LinalgExt::LinalgFusionOpInterface>(consumer) ||
      !isa<IREE::LinalgExt::LinalgFusionOpInterface>(producer)) {
    return false;
  }

  // Keep a sparse one-hot producer materialized for its remaining dense
  // backward consumer. Recomputing linalg.index plus the label comparison for
  // every vocabulary-gradient element is measurably slower on Apple than one
  // coalesced mask read. The forward target selection is already rewritten to
  // a rank-1 sparse gather, so this boundary does not retain its reduction use.
  if (isSparseOneHotLike(producer)) {
    return false;
  }
  if (producer->hasAttr("iree-metal.sparse-one-hot-columns")) {
    return false;
  }
  // Non-unique scatter batch loops are reductions and remain untiled when the
  // trailing update window is distributed across a Metal workgroup. Fusing a
  // producer into a large update operand can therefore materialize the entire
  // batch for each 32-wide window tile in workgroup memory. Keep that producer
  // in a separate dispatch when even this lower-bound estimate exceeds Metal's
  // 32 KiB threadgroup-memory limit. Dynamic batch dimensions are unbounded and
  // must retain the dispatch boundary as well.
  bool oversizedNonUniqueScatterUpdate = false;
  if (auto scatterOp = dyn_cast<IREE::LinalgExt::ScatterOp>(consumer);
      scatterOp && !scatterOp.getUniqueIndices() &&
      operand.getOperandNumber() ==
          scatterOp.getUpdatesMutable().getOperandNumber()) {
    constexpr int64_t kMetalWorkgroupWidth = 32;
    constexpr int64_t kMetalWorkgroupMemoryLimitBits = 32 * 1024 * 8;
    int64_t bitsPerBatchElement =
        kMetalWorkgroupWidth *
            scatterOp.getUpdateType().getElementTypeBitWidth() +
        scatterOp.getIndexDepth() *
            scatterOp.getIndicesType().getElementTypeBitWidth();
    int64_t maxBatchElements =
        kMetalWorkgroupMemoryLimitBits / bitsPerBatchElement;
    int64_t batchElementCount = 1;
    for (int64_t dim : scatterOp.getBatchShape()) {
      if (ShapedType::isDynamic(dim)) {
        oversizedNonUniqueScatterUpdate = true;
        break;
      }
      if (dim != 0 && batchElementCount > maxBatchElements / dim) {
        oversizedNonUniqueScatterUpdate = true;
        break;
      }
      batchElementCount *= dim;
    }
  }

  // A reduction consumer cannot safely tile a fused non-init producer using
  // the reduction's mapping. Under aggressive fusion this changed the forward
  // summation topology of differentiated logsumexp while leaving its gradient
  // unchanged. Preserve this producer boundary; init-operand fusion and
  // elementwise aggressive fusion remain enabled.
  bool reductionNonInitOperand =
      linalgConsumer && linalgConsumer.getNumReductionLoops() != 0 &&
      !isLosslessWideningCastLike(producer);

  // IREE_METAL_NONINIT_GUARD remains available as a broad diagnostic control.
  if (!options.aggressiveFusion || oversizedNonUniqueScatterUpdate ||
      reductionNonInitOperand ||
      getenv("IREE_METAL_NONINIT_GUARD")) {
    auto consumerFusionOp = dyn_cast<DestinationStyleOpInterface>(consumer);
    if (consumerFusionOp && !consumerFusionOp.isDpsInit(&operand)) {
      return false;
    }
  }

  if (!tracker.getFusionGroup(consumer).isFusable(producer)) {
    return false;
  }

  // Check operand limit before allowing fusion
  if (tracker.getFusionGroup(consumer).wouldExceedOperandLimit(producer)) {
    return false;
  }

  return true;
}

/// Check if moving the producer into the dispatch at the root's position would
/// break any existing uses. Returns true if there are uses between the producer
/// and the root that are not in the fusion group, which would cause a dominance
/// violation.
static bool hasUsesBetweenProducerAndRoot(Operation *producer, Operation *root,
                                          const FusionGroup &fusionGroup,
                                          const DominanceInfo &dominanceInfo) {
  for (OpOperand &use : producer->getUses()) {
    Operation *user = use.getOwner();
    // Walk up to the parent in the same block as the producer/root.
    while (user && user->getBlock() != root->getBlock()) {
      user = user->getParentOp();
    }
    // If the user is in the fusion group, it will be moved too.
    if (!user || fusionGroup.contains(user)) {
      continue;
    }
    // If the user does not come after the root, then moving the producer
    // into the dispatch at root's position would break dominance.
    if (!dominanceInfo.properlyDominates(root, user)) {
      return true;
    }
  }
  return false;
}

/// Starting from the `root` op, traverse the operand use-def chain
/// in reverse to fuse with producers.
static void
fuseRootsWithProducers(MLIRContext *context, Operation *root,
                       FusionGroup &fusionGroup,
                       DominanceInfo const &dominanceInfo,
                       FormDispatchRegionsPassOptions const &options,
                       FusionTracker &tracker, bool fuseWithTruncate) {
  SmallVector<Operation *> worklist;
  worklist.push_back(root);
  IREE::Flow::CloneableIntoDispatchOptions cloneableOptions;
  cloneableOptions.aggressive = options.aggressiveFusion;
  while (!worklist.empty()) {
    Operation *candidate = worklist.pop_back_val();
    for (OpOperand &operand : candidate->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer) {
        continue;
      }
      if (IREE::Flow::isCloneableIntoDispatchOp(producer, cloneableOptions) ||
          tracker.isFusedOp(producer) || tracker.isRootOp(producer)) {
        continue;
      }

      if (!isFusableWithProducer(operand, tracker, options, fuseWithTruncate)) {
        continue;
      }

      if (hasUsesBetweenProducerAndRoot(producer, root, fusionGroup,
                                        dominanceInfo)) {
        continue;
      }

      SmallVector<OpOperand *> fusableUses =
          getFusableUses(context, producer, dominanceInfo,
                         /*aggressiveFusion=*/options.aggressiveFusion);
      if (fusableUses.empty() || fusableUses.front()->getOwner() != candidate) {
        continue;
      }

      tracker.appendToFusionGroup(producer, fusionGroup);
      worklist.push_back(producer);
    }
  }
}

/// Some heuristic is needed to fuse a dispatchable op with root operations
/// using tile + fuse.
static void
decideFusableLinalgOps(Region &region, DominanceInfo const &dominanceInfo,
                       FormDispatchRegionsPassOptions const &options,
                       FusionTracker &tracker, unsigned numRootOps = 0) {
  MLIRContext *context = region.getContext();
  OpBuilder builder(context);
  IREE::Flow::CloneableIntoDispatchOptions cloneableOptions;
  cloneableOptions.aggressive = options.aggressiveFusion;
  for (Block &block : region) {
    // Dispatch region formation works by first cloning the root into
    // the dispatch region and then pulling operations in.
    // So procedure here is to
    // - First find the roots
    // - To fuse with consumers make the consumer the root.
    SmallVector<Operation *> roots;
    for (Operation &op : llvm::reverse(block)) {
      if (isa<scf::SCFDialect>(op.getDialect())) {
        for (auto &region : op.getRegions()) {
          decideFusableLinalgOps(region, dominanceInfo, options, tracker,
                                 numRootOps);
        }
        continue;
      }

      // Start with a root operation and fuse its producers.
      if (tracker.isFusedOp(&op) || !isRootLikeOp(&op)) {
        continue;
      }
      FusionGroup &newGroup = tracker.createFusionGroup(context, &op);
      fuseRootsWithProducers(context, &op, newGroup, dominanceInfo, options,
                             tracker,
                             /*fuseWithTruncate=*/false);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo, options, tracker);
    for (Operation *root : roots) {
      FusionGroup &fusionGroup = tracker.getFusionGroup(root);
      fuseRootsWithProducers(context, root, fusionGroup, dominanceInfo, options,
                             tracker,
                             /*fuseWithTruncate=*/true);
    }
  }

  // Once all root linalg ops have been tagged, put all remaining generic ops
  // into their own dispatches.
  for (Block &block : region) {
    SmallVector<Operation *> roots;
    for (Operation &op : llvm::reverse(block)) {
      // If it is part of a fusion group or root op, ignore it.
      if (tracker.isFusedOp(&op) || tracker.isRootOp(&op)) {
        continue;
      }
      // Only look for Linalg ops here. Avoid moving `linalg.fill` that aren't
      // fused with anything else into their own dispatches since it is better
      // to convert them to splats. Also avoid moving dequantization-like ops
      // into their own dispatch since it is better to clone these ops and avoid
      // materializing large tensors between dispatches.
      if (!isa<linalg::LinalgOp, tensor::PadOp, linalg::PackOp>(op) ||
          IREE::Flow::isCloneableIntoDispatchOp(&op, cloneableOptions)) {
        continue;
      }

      // For now check if this is a rope computation that is to be fused with
      // attention.
      // TODO: Ideally this is just regular gather fusion which will be covered
      // by the `isCloneableIntoDispatchOp` call above, but for now this is done
      // as a point fix.
      if (IREE::LinalgExt::isGatherlikeOp(&op) &&
          llvm::all_of(op.getUsers(),
                       llvm::IsaPred<IREE::LinalgExt::AttentionOp>)) {
        continue;
      }

      FusionGroup &newGroup = tracker.createFusionGroup(context, &op);
      fuseRootsWithProducers(context, &op, newGroup, dominanceInfo, options,
                             tracker,
                             /*fuseWithTruncate=*/false);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo, options, tracker);
    for (Operation *root : roots) {
      FusionGroup &fusionGroup = tracker.getFusionGroup(root);
      fuseRootsWithProducers(context, root, fusionGroup, dominanceInfo, options,
                             tracker,
                             /*fuseWithTruncate=*/true);
    }
  }
}

//===----------------------------------------------------------------------===//
// Dispatch region formation
//===----------------------------------------------------------------------===//

/// Create IREE::Flow::DispatchGroupsOps based on a fusion heuristic.
static LogicalResult
createFusionGroups(TensorDimTrackingRewriter &rewriter,
                   mlir::FunctionOpInterface funcOp,
                   DominanceInfo &dominanceInfo,
                   FormDispatchRegionsPassOptions const &options) {
  // Step 1: Decide fusion groups (heuristic).
  FusionTracker tracker;
  decideFusableLinalgOps(funcOp.getFunctionBody(), dominanceInfo, options,
                         tracker);

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After deciding fusion groups ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Step 2. Create a DispatchRegionOp for every fusion group.
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<IREE::Flow::DispatchRegionOp> regionOps;
  for (const auto &fusionGroup : tracker.getFusionGroups()) {
    Operation *root = fusionGroup->getRoot();
    // Sort producers and consumers topologically. All fused ops must be in the
    // same block as the root.
    SmallVector<Operation *> currFusedOperations =
        fusionGroup->getFusedOperations();
    bool sortResult = mlir::computeTopologicalSorting(currFusedOperations);
    (void)sortResult;
    assert(sortResult && "could not compute topological sorting");

    int rootPos = 0;
    for (auto [index, fusedOperation] : llvm::enumerate(currFusedOperations)) {
      if (fusedOperation == root) {
        rootPos = index;
        break;
      }
    }
    SmallVector<Operation *> producers, consumers;
    if (rootPos > 0) {
      producers = llvm::to_vector(
          ArrayRef<Operation *>(currFusedOperations).take_front(rootPos));
    }
    if (rootPos < currFusedOperations.size() - 1) {
      consumers = llvm::to_vector(
          ArrayRef<Operation *>(currFusedOperations).drop_front(rootPos + 1));
    }

    // Simplify tensor::DimOps.
    {
      SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
      if (failed(IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
        return failure();
      }
    }

    // Create fusion group.
    IREE::Flow::DispatchRegionOp regionOp;
    auto maybeRegionOp = IREE::Flow::wrapOpInDispatchRegion(rewriter, root);
    if (failed(maybeRegionOp)) {
      return root->emitOpError("failed to move root into dispatch");
    }
    regionOp = *maybeRegionOp;

    // Move ops into the region.
    for (Operation *producer : llvm::reverse(producers)) {
      // Simplify tensor::DimOps.
      {
        SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
        if (failed(IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
          return failure();
        }
      }

      auto newRegionOp =
          movePrecedingOpsIntoDispatchRegion(rewriter, producer, regionOp);
      if (failed(newRegionOp)) {
        producer->emitWarning("failed to move producer into region");
        continue;
      }
      regionOp = *newRegionOp;
    }

    for (Operation *consumer : consumers) {
      // Simplify tensor::DimOps.
      {
        SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
        if (failed(IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
          return failure();
        }
      }

      auto newRegionOp = IREE::Flow::moveFollowingOpIntoDispatchRegion(
          rewriter, consumer, regionOp);
      if (failed(newRegionOp)) {
        consumer->emitWarning("failed to move consumer into region");
        continue;
      }
      regionOp = *newRegionOp;
    }
    // Simplify tensor::DimOps.
    {
      SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
      if (failed(IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
        return failure();
      }
    }
    regionOps.push_back(regionOp);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After creating flow.dispatch.region ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  return success();
}

namespace {
/// Pass declaration.
struct FormDispatchRegionsPass final
    : impl::FormDispatchRegionsPassBase<FormDispatchRegionsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

/// Create dispatch.region Ops based on a fusion heuristic.
void FormDispatchRegionsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet preprocessingPatterns(context);
  preprocessingPatterns.add<RewriteSparseOneHotSelection>(context,
                                                           PatternBenefit(2));
  preprocessingPatterns.add<FuseSparseOneHotPredicate>(
      context, PatternBenefit(1));
  if (failed(applyPatternsGreedily(funcOp, std::move(preprocessingPatterns)))) {
    funcOp.emitOpError("failed in sparse selection preprocessing");
    return signalPassFailure();
  }
  DominanceInfo &dominanceInfo = getAnalysis<DominanceInfo>();
  TensorDimTrackingRewriter rewriter(funcOp);
  FormDispatchRegionsPassOptions options{aggressiveFusion, fusePadWithConsumers,
                                         fusePadWithProducers};
  if (failed(createFusionGroups(rewriter, funcOp, dominanceInfo, options))) {
    funcOp->emitOpError("failed to create fusion groups");
    return signalPassFailure();
  }

  // Canonicalize all the dispatch regions to remove unused operands.
  RewritePatternSet patterns(context);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  IREE::Flow::DispatchRegionOp::getCanonicalizationPatterns(patterns, context);
  GreedyRewriteConfig config;
  config.setMaxIterations(GreedyRewriteConfig::kNoLimit).enableFolding(true);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
    funcOp.emitOpError("failed in cleanup patterns");
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
