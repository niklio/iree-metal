// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/IR/IRMapping.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "iree/compiler/Utils/RegionOpUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-fuse-horizontal-contractions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FUSEHORIZONTALCONTRACTIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

static bool operator==(const linalg::ContractionDimensions &lhs,
                       const linalg::ContractionDimensions &rhs) {
  return lhs.batch == rhs.batch && lhs.m == rhs.m && lhs.n == rhs.n &&
         lhs.k == rhs.k;
}
static bool operator!=(const linalg::ContractionDimensions &lhs,
                       const linalg::ContractionDimensions &rhs) {
  return !(lhs == rhs);
}

struct FuseHorizontalContractionsPass final
    : impl::FuseHorizontalContractionsPassBase<FuseHorizontalContractionsPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

/// For indexing maps of Linalg ops passed in as `indexingMaps`, permute them
/// such that `seedLhsIndexingMap` is same as `indexingMaps[0]`. Returns the
/// permutation of the iteration space of the RHS. Returns failure if the
/// permutation of `iteratorTypes` results in a change of `iteratorTypes`. This
/// is so that permutation doesnt change the position of reduction iterator
/// type.
static std::optional<SmallVector<int64_t>>
permuteIndexingMapsToMatchSeedLhs(MLIRContext *context,
                                  AffineMap seedLhsIndexingMap,
                                  ArrayRef<utils::IteratorType> iteratorTypes,
                                  SmallVector<AffineMap> &indexingMaps) {
  if (indexingMaps.empty()) {
    return std::nullopt;
  }
  AffineMap lhsIndexingMap = indexingMaps[0];
  if (seedLhsIndexingMap == lhsIndexingMap) {
    return llvm::to_vector(llvm::seq<int64_t>(0, lhsIndexingMap.getNumDims()));
  }

  assert(lhsIndexingMap.getNumDims() == seedLhsIndexingMap.getNumDims());
  if (!lhsIndexingMap.isProjectedPermutation() ||
      !seedLhsIndexingMap.isProjectedPermutation() ||
      lhsIndexingMap.getNumResults() != seedLhsIndexingMap.getNumResults()) {
    return std::nullopt;
  }

  auto getResultDimsRange = [](ArrayRef<AffineExpr> exprs) {
    return llvm::map_range(exprs, [](AffineExpr expr) {
      return cast<AffineDimExpr>(expr).getPosition();
    });
  };
  auto seedLhsResultDimsRange =
      getResultDimsRange(seedLhsIndexingMap.getResults());
  auto lhsResultDimsRange = getResultDimsRange(lhsIndexingMap.getResults());

  // Start with an identity permutations. For now try to only swap dimensions
  // which is not a general solution.
  SmallVector<int64_t> interchangeVector =
      llvm::to_vector(llvm::seq<int64_t>(0, lhsIndexingMap.getNumDims()));
  for (auto [seedDimPos, lhsDimPos] :
       llvm::zip_equal(seedLhsResultDimsRange, lhsResultDimsRange)) {
    if (seedDimPos == lhsDimPos) {
      continue;
    }
    // If the current positions are what we started with, swap the positions.
    if (interchangeVector[lhsDimPos] == lhsDimPos &&
        interchangeVector[seedDimPos] == seedDimPos) {
      std::swap(interchangeVector[lhsDimPos], interchangeVector[seedDimPos]);
      continue;
    }
    // If this was a changed dimension, check that it is consistent.
    if (interchangeVector[lhsDimPos] != seedDimPos ||
        interchangeVector[seedDimPos] != lhsDimPos) {
      return std::nullopt;
    }
  }

  // Check that the iterator types remain the same
  SmallVector<utils::IteratorType> permutedIteratorTypes =
      llvm::to_vector(iteratorTypes);
  applyPermutationToVector(permutedIteratorTypes, interchangeVector);
  if (permutedIteratorTypes != iteratorTypes) {
    return std::nullopt;
  }

  AffineMap interchangeMap =
      AffineMap::getPermutationMap(interchangeVector, context);
  for (auto &map : indexingMaps) {
    if (!map.isEmpty()) {
      map = map.compose(interchangeMap);
    }
  }
  return interchangeVector;
}

/// Helper method to check operations equivalence
static bool checkContractionOpEquivalence(MLIRContext *context, Operation *aOp,
                                          Operation *bOp) {
  auto aLinalgOp = dyn_cast<linalg::LinalgOp>(aOp);
  auto bLinalgOp = dyn_cast<linalg::LinalgOp>(bOp);

  if (!aLinalgOp || !bLinalgOp) {
    return false;
  }
  // Contraction ops verifies that there are two operands and one result.
  assert(linalg::isaContractionOpInterface(aLinalgOp) &&
         linalg::isaContractionOpInterface(bLinalgOp) &&
         "expected lhs and rhs to be contraction ops");

  // Check that the LHS operand is the same.
  if (aLinalgOp.getDpsInputOperand(0)->get() !=
      bLinalgOp.getDpsInputOperand(0)->get()) {
    return false;
  }

  // Check that the n-dimensions are the same
  SmallVector<AffineMap> aIndexingMaps = aLinalgOp.getIndexingMapsArray();
  SmallVector<AffineMap> bIndexingMaps = bLinalgOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> aIteratorTypes =
      aLinalgOp.getIteratorTypesArray();
  SmallVector<utils::IteratorType> bIteratorTypes =
      bLinalgOp.getIteratorTypesArray();
  std::optional<SmallVector<int64_t>> bPermutationVector;
  if (aIndexingMaps[0] != bIndexingMaps[0]) {
    bPermutationVector = permuteIndexingMapsToMatchSeedLhs(
        context, aIndexingMaps[0], bIteratorTypes, bIndexingMaps);
    if (!bPermutationVector) {
      return false;
    }
  }

  FailureOr<linalg::ContractionDimensions> aContractionDims =
      linalg::inferContractionDims(aIndexingMaps);
  FailureOr<linalg::ContractionDimensions> bContractionDims =
      linalg::inferContractionDims(bIndexingMaps);
  if (failed(aContractionDims) || failed(bContractionDims)) {
    return false;
  }
  if (aContractionDims.value() != bContractionDims.value()) {
    return false;
  }

  SmallVector<int64_t> aStaticDims = aLinalgOp.getStaticLoopRanges();
  SmallVector<int64_t> bStaticDims = bLinalgOp.getStaticLoopRanges();
  if (bPermutationVector) {
    applyPermutationToVector(bStaticDims, bPermutationVector.value());
  }
  for (auto nDim : aContractionDims->n) {
    if (aStaticDims[nDim] != bStaticDims[nDim] ||
        ShapedType::isDynamic(aStaticDims[nDim])) {
      return false;
    }
  }

  // TODO(#20116): hack to prevent codegen failure for small horizontally fused
  // matmuls that go down LLVMGPUDistribute.
  unsigned mDimsSize = 1;
  for (unsigned dim : aContractionDims.value().m) {
    mDimsSize *= aStaticDims[dim];
  }
  if (mDimsSize < 16) {
    return false;
  }

  auto checkSameRankAndElementType = [](Value aVal, Value bVal) {
    auto aType = dyn_cast<ShapedType>(aVal.getType());
    auto bType = dyn_cast<ShapedType>(bVal.getType());
    return aType && bType && aType.getRank() == bType.getRank() &&
           aType.getElementType() == bType.getElementType();
  };
  // Check that the RHS rank and element type are the same. We dont check the
  // type cause we allow RHS to be transposes.
  if (!checkSameRankAndElementType(aLinalgOp.getDpsInputOperand(1)->get(),
                                   bLinalgOp.getDpsInputOperand(1)->get())) {
    return false;
  }

  // Check that the output rank and element type are the same. We dont check the
  // type cause we allow output to be transposes.
  if (!checkSameRankAndElementType(aLinalgOp.getDpsInitOperand(0)->get(),
                                   bLinalgOp.getDpsInitOperand(0)->get())) {
    return false;
  }

  // Check that the iterator types are the same.
  if (aLinalgOp.getIteratorTypesArray() != bLinalgOp.getIteratorTypesArray()) {
    return false;
  }

  // Check region equivalence.
  if (!OperationEquivalence::isRegionEquivalentTo(
          &aLinalgOp->getRegion(0), &bLinalgOp->getRegion(0),
          OperationEquivalence::IgnoreLocations)) {
    return false;
  }

  return true;
}

// nlearn (2026-07-22): HORIZONTAL REDUCTION FUSION (env NLEARN_FUSE_HORIZ_REDUCTIONS).
// LayerNorm/softmax emit several reductions (mean/var fwd, dgamma/dbeta/dx bwd) that share
// the reduced input and iteration space, but each is a dispatch root -> LN fwd+bwd = 14
// dispatches vs ~2 on MPS (the glue-fusion gap). Fuse same-iteration-space reduction generics
// that share input operand 0 into ONE multi-result generic (validated: multi-result reduction
// codegens to a single dispatch). Reuses fuseContractionsHorizontally's generic multi-result
// generation. Unlike the contraction check, we do NOT require region equivalence (mean=sum,
// var=sum-of-squares have different bodies) — the generator merges distinct bodies.
static bool nlearnFuseHorizReductions() {
  static const bool on = ::getenv("NLEARN_FUSE_HORIZ_REDUCTIONS") != nullptr;
  return on;
}

// nlearn (2026-07-22): VARIANCE-REWRITE (env NLEARN_LN_VAR_REWRITE). LayerNorm's variance
// reduction sum((x-mean)^2) reduces a DERIVED value (x-mean), so it can't horizontally fuse
// with the mean's sum(x). Rewrite it to sum(x^2) - N*mean^2: sum(x^2) reduces x DIRECTLY
// (squares per-element in the reduction body), so it shares operand-0 with sum(x) and the
// two collapse via horizontal reduction fusion (LN fwd 4->2 dispatches; numerically bit-exact
// on bf16 LN, validated rel-err 0.0). Enable both flags together.
//
// Returns the reduced input value if `op` is a single-input reduction generic whose body is a
// plain sum: yield(addf(out, in)). Returns the input value if a sum-of-squares:
// yield(addf(out, mulf(in,in))). `wantSquare` selects which.
static Value matchSumReductionInput(linalg::LinalgOp op, bool wantSquare) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getNumDpsInits() != 1 || genericOp->getNumResults() != 1 ||
      op.getNumReductionLoops() == 0)
    return nullptr;
  Block *body = genericOp.getBlock();
  if (body->getNumArguments() != 2)
    return nullptr;
  Value inArg = body->getArgument(0), outArg = body->getArgument(1);
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  auto add = yieldOp->getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!add)
    return nullptr;
  // One addf operand must be the accumulator (outArg); the other is the summand.
  Value summand;
  if (add.getLhs() == outArg)
    summand = add.getRhs();
  else if (add.getRhs() == outArg)
    summand = add.getLhs();
  else
    return nullptr;
  if (wantSquare) {
    auto mul = summand.getDefiningOp<arith::MulFOp>();
    if (!mul || mul.getLhs() != inArg || mul.getRhs() != inArg)
      return nullptr;
  } else if (summand != inArg) {
    return nullptr;
  }
  return genericOp.getDpsInputOperand(0)->get();
}

// Rewrite var = sum((x-mean)^2) into sum(x^2) - N*mean^2. Aggressive fusion emits the variance
// as ONE reduction generic ins(x, mean) with body addf(out, mulf(subf(x,mean), subf(x,mean))).
// We build a sum(x^2) reduction reading the MEAN's input (so both reductions share operand-0 and
// horizontally fuse — the model duplicates the bf16->f32 extf, so the var op's own x is a
// different SSA value), then subtract N*mean^2.
struct RewriteVarianceToSumSq final : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp varGen,
                                PatternRewriter &rewriter) const override {
    if (!::getenv("NLEARN_LN_VAR_REWRITE"))
      return failure();
    auto varOp = cast<linalg::LinalgOp>(varGen.getOperation());
    if (varOp.getNumReductionLoops() == 0 || varGen.getNumDpsInputs() != 2 ||
        varGen.getNumDpsInits() != 1 || varGen->getNumResults() != 1)
      return failure();
    Block *b = varGen.getBlock();
    if (b->getNumArguments() != 3)
      return failure();
    Value a0 = b->getArgument(0), a1 = b->getArgument(1), out = b->getArgument(2);
    auto yieldOp = cast<linalg::YieldOp>(b->getTerminator());
    auto add = yieldOp->getOperand(0).getDefiningOp<arith::AddFOp>();
    if (!add)
      return failure();
    Value summand =
        add.getLhs() == out ? add.getRhs() : (add.getRhs() == out ? add.getLhs() : Value());
    if (!summand)
      return failure();
    auto mul = summand.getDefiningOp<arith::MulFOp>();
    if (!mul || mul.getLhs() != mul.getRhs())
      return failure();
    auto sub = mul.getLhs().getDefiningOp<arith::SubFOp>();
    if (!sub || sub.getLhs() != a0 || sub.getRhs() != a1)
      return failure();  // body != addf(out, (x-mean)^2)

    Value mean = varGen.getDpsInputOperand(1)->get();
    // mean = divf(sum(x), N)
    auto meanDiv = mean.getDefiningOp<linalg::GenericOp>();
    if (!meanDiv || meanDiv.getNumDpsInputs() < 1)
      return failure();
    auto divYield = cast<linalg::YieldOp>(meanDiv.getBlock()->getTerminator());
    if (!divYield->getOperand(0).getDefiningOp<arith::DivFOp>())
      return failure();
    auto meanSum =
        meanDiv.getDpsInputOperand(0)->get().getDefiningOp<linalg::GenericOp>();
    if (!meanSum)
      return failure();
    Value sumInput = matchSumReductionInput(
        cast<linalg::LinalgOp>(meanSum.getOperation()), /*wantSquare=*/false);
    if (!sumInput)
      return failure();
    // sumInput must have the same type as the var op's x input (same iteration space).
    if (sumInput.getType() != varGen.getDpsInputOperand(0)->get().getType())
      return failure();

    Location loc = varGen.getLoc();
    SmallVector<int64_t> vbounds = varOp.getStaticLoopRanges();
    int64_t N = 1;
    for (unsigned d : llvm::seq<unsigned>(0, varOp.getNumLoops()))
      if (linalg::isReductionIterator(varOp.getIteratorTypesArray()[d]))
        N *= vbounds[d];
    if (ShapedType::isDynamic(N) || N <= 0)
      return failure();

    // Build sum(x^2) reading sumInput (shares operand-0 with meanSum -> fuses).
    AffineMap inMap = varGen.getMatchingIndexingMap(varGen.getDpsInputOperand(0));
    AffineMap outMap = varGen.getMatchingIndexingMap(varGen.getDpsInitOperand(0));
    Value initVal = varGen.getDpsInitOperand(0)->get();
    auto resType = cast<ShapedType>(varGen->getResult(0).getType());
    auto sumSq = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resType}, ValueRange{sumInput},
        ValueRange{initVal}, ArrayRef<AffineMap>{inMap, outMap},
        varOp.getIteratorTypesArray(),
        [&](OpBuilder &ob, Location ol, ValueRange args) {
          Value sq = arith::MulFOp::create(ob, ol, args[0], args[0]);
          Value acc = arith::AddFOp::create(ob, ol, args[1], sq);
          linalg::YieldOp::create(ob, ol, acc);
        });

    // Correction: result = sum(x^2) - N * mean^2 (elementwise over the reduced result).
    Type et = resType.getElementType();
    Value nConst = arith::ConstantOp::create(rewriter, loc, et,
                                             rewriter.getFloatAttr(et, (double)N));
    Value empty = tensor::EmptyOp::create(rewriter, loc, resType.getShape(), et);
    SmallVector<AffineMap> maps(3,
                                rewriter.getMultiDimIdentityMap(resType.getRank()));
    SmallVector<utils::IteratorType> iters(resType.getRank(),
                                           utils::IteratorType::parallel);
    auto corrected = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resType}, ValueRange{sumSq.getResult(0), mean},
        ValueRange{empty}, maps, iters,
        [&](OpBuilder &ob, Location ol, ValueRange args) {
          Value msq = arith::MulFOp::create(ob, ol, args[1], args[1]);
          Value nmsq = arith::MulFOp::create(ob, ol, nConst, msq);
          Value r = arith::SubFOp::create(ob, ol, args[0], nmsq);
          linalg::YieldOp::create(ob, ol, r);
        });
    rewriter.replaceOp(varGen, corrected.getResult(0));
    return success();
  }
};
static bool isHorizFusableReduction(linalg::LinalgOp op) {
  return isa<linalg::GenericOp>(op.getOperation()) &&
         op.getNumReductionLoops() != 0 && op.getNumDpsInputs() >= 1 &&
         op->getNumResults() == 1 && !op.hasDynamicShape();
}
static bool checkReductionOpEquivalence(MLIRContext *context, Operation *aOp,
                                        Operation *bOp) {
  auto aLinalgOp = dyn_cast<linalg::LinalgOp>(aOp);
  auto bLinalgOp = dyn_cast<linalg::LinalgOp>(bOp);
  if (!aLinalgOp || !bLinalgOp)
    return false;
  if (!isHorizFusableReduction(aLinalgOp) || !isHorizFusableReduction(bLinalgOp))
    return false;
  // Must reduce the SAME input value (operand 0) — the shared operand horizontal
  // fusion keys on.
  if (aLinalgOp.getDpsInputOperand(0)->get() !=
      bLinalgOp.getDpsInputOperand(0)->get())
    return false;
  // Same iterator types (same reduction dims) and same static loop ranges (same
  // iteration space) — required for a single co-tiled multi-result reduction.
  if (aLinalgOp.getIteratorTypesArray() != bLinalgOp.getIteratorTypesArray())
    return false;
  if (aLinalgOp.getStaticLoopRanges() != bLinalgOp.getStaticLoopRanges())
    return false;
  if (aLinalgOp.hasDynamicShape())
    return false;
  // Same input-0 indexing map (or permutable to match the seed's).
  SmallVector<AffineMap> aIndexingMaps = aLinalgOp.getIndexingMapsArray();
  SmallVector<AffineMap> bIndexingMaps = bLinalgOp.getIndexingMapsArray();
  if (aIndexingMaps[0] != bIndexingMaps[0]) {
    SmallVector<utils::IteratorType> bIteratorTypes =
        bLinalgOp.getIteratorTypesArray();
    if (!permuteIndexingMapsToMatchSeedLhs(context, aIndexingMaps[0],
                                           bIteratorTypes, bIndexingMaps))
      return false;
  }
  return true;
}

/// Check that a given operation is "horizontal" to the group. The operation
/// is horizontal if the `slice` of the operation does not contain any op
/// from the group.
static bool isHorizontalToGroup(Operation *op,
                                const llvm::SetVector<Operation *> &currGroup,
                                const DominanceInfo &dominanceInfo,
                                Operation *seedOp) {
  BackwardSliceOptions options;
  options.inclusive = true;
  // Limit the slice to the seed to make sure the slice is small.
  options.filter = [&](Operation *op) {
    return !dominanceInfo.properlyDominates(op, seedOp);
  };
  llvm::SetVector<Operation *> slice;
  [[maybe_unused]] LogicalResult result = getBackwardSlice(op, &slice, options);
  assert(result.succeeded());
  return llvm::none_of(currGroup, [&](Operation *groupedOp) {
    return slice.contains(groupedOp);
  });
}

/// Find all candidates that can be used for horizontal fusion. For example
/// ```
/// %0 = linalg.matmul ins(%arg0, %arg1)
/// %1 = linalg.matmul ins(%arg0, %arg2)
/// %2 = linalg.matmul ins(%arg0, %arg3)
/// ```
///
/// where all matmul share an operand can be combined into
///
/// ```
/// %4 = linalg.matmul ins(%arg0, concat(%arg1, %arg2, %arg3))
/// ```
///
/// Note: The actual operation generated does not concat the RHS.
static std::optional<SmallVector<Operation *>> getHorizontalFusionGroupMembers(
    MLIRContext *context, linalg::LinalgOp seedOp,
    const llvm::SmallDenseSet<Operation *> &groupedOperations,
    const DominanceInfo &dominanceInfo, int fusionLimit) {

  Value lhs = seedOp->getOperand(0);

  SetVector<Operation *> allOps;
  SmallVector<Operation *> contractionOps = {seedOp};
  allOps.insert(seedOp);

  auto canBeGrouped = [&](linalg::LinalgOp linalgOp) -> bool {
    if (linalgOp->getParentOp() != seedOp->getParentOp()) {
      return false;
    }

    // Constraints of the operation itself. When the seed is a reduction generic
    // (NLEARN_FUSE_HORIZ_REDUCTIONS), use the reduction-equivalence predicate;
    // otherwise the contraction one.
    if (nlearnFuseHorizReductions() && isHorizFusableReduction(seedOp)) {
      if (!isHorizFusableReduction(linalgOp) ||
          !checkReductionOpEquivalence(context, linalgOp, seedOp)) {
        return false;
      }
    } else if (!linalg::isaContractionOpInterface(linalgOp) ||
               !checkContractionOpEquivalence(context, linalgOp, seedOp)) {
      return false;
    }
    if (groupedOperations.contains(linalgOp)) {
      return false;
    }

    // Structural constraints related to being able to fuse the operations.
    if (!dominanceInfo.properlyDominates(seedOp, linalgOp)) {
      return false;
    }
    return true;
  };

  // Iterate over users of LHS to find ops that can be grouped with the seed.
  SmallVector<Operation *> lhsUsers;
  for (Operation *lhsUser : lhs.getUsers()) {
    if (lhsUser->getBlock() != seedOp->getBlock() || lhsUser == seedOp) {
      continue;
    }

    auto linalgUser = dyn_cast<linalg::LinalgOp>(lhsUser);
    if (!linalgUser || !canBeGrouped(linalgUser)) {
      continue;
    }
    lhsUsers.push_back(lhsUser);
  }

  // Sort the users so that the order is deterministic
  llvm::sort(lhsUsers, [&](Operation *a, Operation *b) {
    return dominanceInfo.properlyDominates(a, b);
  });

  // Collect all contraction op users of lhs.
  for (Operation *lhsUser : lhsUsers) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(lhsUser);
    if (!linalgUser) {
      continue;
    }

    if (!isHorizontalToGroup(linalgUser, allOps, dominanceInfo, seedOp)) {
      continue;
    }

    contractionOps.push_back(linalgUser);
    allOps.insert(linalgUser);
    if (contractionOps.size() >= fusionLimit) {
      break;
    }
  }

  if (contractionOps.size() == 1) {
    return std::nullopt;
  }

  return contractionOps;
}

/// Generate the horizontally fused operation as an operation with multiple
/// results, corresponding to the results of the fused operations. It is assumed
/// that the LHS of the contraction operations fused horizontally is the same
/// and have the same indexing map for all the operations. The RHS/outputs of
/// the operations can be different, but share the same iteration space.
/// Returns the generated fused op, or `std::nullopt` when the fused op
/// could not be generated.
static std::optional<linalg::GenericOp>
fuseContractionsHorizontally(RewriterBase &rewriter, Location loc,
                             MutableArrayRef<Operation *> linalgOps) {
  if (linalgOps.empty()) {
    return std::nullopt;
  }

  SmallVector<Value> fusedIns;
  SmallVector<Value> fusedOuts;
  SmallVector<Type> fusedResultTypes;
  SmallVector<AffineMap> fusedInsIndexingMaps;
  SmallVector<AffineMap> fusedOutsIndexingMaps;

  auto seedOp = cast<linalg::LinalgOp>(linalgOps.front());
  SmallVector<utils::IteratorType> fusedIteratorTypes =
      seedOp.getIteratorTypesArray();

  OpOperand *seedOpLhs = seedOp.getDpsInputOperand(0);
  AffineMap seedOpLhsIndexingMap = seedOp.getMatchingIndexingMap(seedOpLhs);
  fusedIns.push_back(seedOpLhs->get());
  fusedInsIndexingMaps.push_back(seedOpLhsIndexingMap);

  llvm::SmallDenseSet<Operation *> droppedOps;
  for (auto op : linalgOps) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp ||
        linalgOp.getDpsInputOperand(0)->get() != seedOpLhs->get()) {
      droppedOps.insert(op);
      continue;
    }

    SmallVector<AffineMap> opIndexingMaps = linalgOp.getIndexingMapsArray();
    if (!permuteIndexingMapsToMatchSeedLhs(
            rewriter.getContext(), seedOpLhsIndexingMap, fusedIteratorTypes,
            opIndexingMaps)) {
      droppedOps.insert(op);
      continue;
    }

    // Append the RHS operands.
    SmallVector<OpOperand *> ins = linalgOp.getDpsInputOperands();
    llvm::append_range(
        fusedIns,
        llvm::map_range(ArrayRef<OpOperand *>(ins).drop_front(),
                        [](OpOperand *operand) { return operand->get(); }));

    // Append the Outs operands.
    llvm::append_range(fusedOuts, llvm::map_range(linalgOp.getDpsInitsMutable(),
                                                  [](OpOperand &operand) {
                                                    return operand.get();
                                                  }));

    // Append the result types.
    fusedResultTypes.append(linalgOp->result_type_begin(),
                            linalgOp->result_type_end());

    // Append the rhs indexing maps.
    llvm::append_range(fusedInsIndexingMaps,
                       ArrayRef<AffineMap>(opIndexingMaps)
                           .slice(1, linalgOp.getNumDpsInputs() - 1));

    // Append the outs indexing maps.
    llvm::append_range(fusedOutsIndexingMaps,
                       ArrayRef<AffineMap>(opIndexingMaps)
                           .drop_front(linalgOp.getNumDpsInputs()));
  }

  SmallVector<AffineMap> fusedIndexingMaps = std::move(fusedInsIndexingMaps);
  fusedIndexingMaps.append(fusedOutsIndexingMaps);
  auto fusedOp = linalg::GenericOp::create(
      rewriter, loc, fusedResultTypes, fusedIns, fusedOuts, fusedIndexingMaps,
      fusedIteratorTypes, [](OpBuilder &, Location, ValueRange) {});

  Block *fusedBody = fusedOp.getBlock();
  int64_t rhsIndex = 0;
  int64_t outsIndex = fusedOp.getNumDpsInputs();
  SmallVector<Value> yieldVals;
  for (auto op : linalgOps) {
    if (droppedOps.contains(op)) {
      continue;
    }
    auto linalgOp = cast<linalg::LinalgOp>(op);
    Block *body = linalgOp.getBlock();
    SmallVector<Value> replacements = {fusedBody->getArgument(0)};
    llvm::append_range(
        replacements,
        llvm::map_range(fusedBody->getArguments().slice(
                            rhsIndex + 1, linalgOp.getNumDpsInputs() - 1),
                        [](BlockArgument arg) -> Value { return arg; }));

    llvm::append_range(
        replacements,
        llvm::map_range(fusedBody->getArguments().slice(
                            outsIndex, linalgOp.getNumDpsInits()),
                        [](BlockArgument arg) -> Value { return arg; }));

    rewriter.mergeBlocks(body, fusedBody, replacements);
    rhsIndex += linalgOp.getNumDpsInputs() - 1;
    outsIndex += linalgOp.getNumDpsInits();

    auto yieldOp = cast<linalg::YieldOp>(fusedBody->getTerminator());
    yieldVals.append(yieldOp->operand_begin(), yieldOp->operand_end());
    rewriter.eraseOp(yieldOp);
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToEnd(fusedBody);
  linalg::YieldOp::create(rewriter, loc, yieldVals);

  unsigned resultsIndex = 0;
  for (auto linalgOp : linalgOps) {
    unsigned numResults = linalgOp->getNumResults();
    rewriter.replaceOp(linalgOp,
                       fusedOp->getResults().slice(resultsIndex, numResults));
    resultsIndex += numResults;
  }

  return fusedOp;
}

static void fuseGroup(RewriterBase &rewriter,
                      MutableArrayRef<Operation *> fusionGroup,
                      DominanceInfo &dominanceInfo) {
  if (!llvm::all_of(fusionGroup, [](Operation *op) {
        return isa_and_nonnull<linalg::LinalgOp>(op);
      })) {
    return;
  }
  auto baseContractOp = cast<linalg::LinalgOp>(fusionGroup.front());
  Location loc = baseContractOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(baseContractOp);

  if (failed(moveOperandDefs(rewriter, fusionGroup, baseContractOp,
                             dominanceInfo))) {
    return;
  }

  std::optional<linalg::GenericOp> fusedOp =
      fuseContractionsHorizontally(rewriter, loc, fusionGroup);
  (void)fusedOp;
}

void FuseHorizontalContractionsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  // nlearn: rewrite LayerNorm variance sum((x-mean)^2) -> sum(x^2) - N*mean^2 so the
  // mean/var reductions read x and horizontally fuse below (env NLEARN_LN_VAR_REWRITE).
  if (::getenv("NLEARN_LN_VAR_REWRITE")) {
    RewritePatternSet varPatterns(context);
    varPatterns.add<RewriteVarianceToSumSq>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(varPatterns))))
      return signalPassFailure();
  }

  DominanceInfo dominanceInfo(getOperation());

  SmallVector<SmallVector<Operation *>> horizontalFusionGroups;
  llvm::SmallDenseSet<Operation *> groupedOperations;

  // nlearn: when the pass is triggered ONLY by the reduction/variance env knobs (not the
  // cl::opt), seed reductions only — horizontally fusing the matmuls regresses them on Metal
  // ("runs but slower"), which would swamp the reduction-fusion signal. NLEARN_HORIZ_CONTRACTIONS
  // re-enables contraction seeding alongside.
  bool reductionOnlyMode =
      (nlearnFuseHorizReductions() || ::getenv("NLEARN_LN_VAR_REWRITE")) &&
      !::getenv("NLEARN_HORIZ_CONTRACTIONS");
  getOperation()->walk([&](linalg::LinalgOp linalgOp) {
    // Seed from contractions, or (env-gated) from fusable reduction generics.
    bool isSeed = (!reductionOnlyMode && linalg::isaContractionOpInterface(linalgOp)) ||
                  (nlearnFuseHorizReductions() &&
                   isHorizFusableReduction(linalgOp));
    if (!isSeed) {
      return;
    }
    // Avoid already grouped operations;
    if (groupedOperations.contains(linalgOp)) {
      return;
    }

    std::optional<SmallVector<Operation *>> fusionGroup =
        getHorizontalFusionGroupMembers(context, linalgOp, groupedOperations,
                                        dominanceInfo, fusionLimit);

    if (!fusionGroup) {
      return;
    }

    // Update statistics.
    numFusionGroups++;
    switch (fusionGroup->size()) {
    case 2:
      numSize2FusionGroups++;
      break;
    case 3:
      numSize3FusionGroups++;
      break;
    default:
      break;
    }

    groupedOperations.insert(fusionGroup->begin(), fusionGroup->end());
    horizontalFusionGroups.emplace_back(std::move(fusionGroup.value()));
  });

  if (horizontalFusionGroups.empty()) {
    return;
  }

  IRRewriter rewriter(context);
  for (auto &fusionGroup : horizontalFusionGroups) {
    fuseGroup(rewriter, fusionGroup, dominanceInfo);
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
