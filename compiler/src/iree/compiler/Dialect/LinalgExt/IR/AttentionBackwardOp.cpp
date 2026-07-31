// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"

#include <array>
#include <initializer_list>
#include <tuple>
#include <utility>

namespace mlir::iree_compiler::IREE::LinalgExt {
namespace {

static SmallVector<utils::IteratorType> getIteratorTypes(AffineMap outputMap) {
  SmallVector<utils::IteratorType> iteratorTypes(
      outputMap.getNumDims(), utils::IteratorType::reduction);
  for (AffineExpr expr : outputMap.getResults()) {
    iteratorTypes[cast<AffineDimExpr>(expr).getPosition()] =
        utils::IteratorType::parallel;
  }
  return iteratorTypes;
}

// Renumber the iteration domain so result dimensions appear in result-tensor
// order, followed by reduction dimensions. Apple vector distribution relies on
// this canonical order when mapping accumulator fragments back to a tensor.
static void canonicalizeLoopOrderForOutput(SmallVectorImpl<AffineMap> &maps) {
  AffineMap outputMap = maps.back();
  int64_t numDims = outputMap.getNumDims();
  SmallVector<int64_t> oldDimsInNewOrder;
  SmallVector<bool> seen(numDims, false);
  for (AffineExpr expr : outputMap.getResults()) {
    int64_t oldDim = cast<AffineDimExpr>(expr).getPosition();
    if (!seen[oldDim]) {
      seen[oldDim] = true;
      oldDimsInNewOrder.push_back(oldDim);
    }
  }
  for (int64_t oldDim = 0; oldDim < numDims; ++oldDim) {
    if (!seen[oldDim]) {
      oldDimsInNewOrder.push_back(oldDim);
    }
  }

  SmallVector<AffineExpr> replacements(numDims);
  MLIRContext *context = outputMap.getContext();
  for (auto [newDim, oldDim] : llvm::enumerate(oldDimsInNewOrder)) {
    replacements[oldDim] = getAffineDimExpr(newDim, context);
  }
  for (AffineMap &map : maps) {
    map = map.replaceDimsAndSymbols(
        replacements, /*symReplacements=*/ArrayRef<AffineExpr>{}, numDims,
        /*numResultSyms=*/0);
  }
}

static Value createZeroTensor(OpBuilder &builder, Location loc, AffineMap map,
                              ArrayRef<OpFoldResult> domainSizes,
                              Type elementType) {
  SmallVector<OpFoldResult> shape;
  shape.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    shape.push_back(domainSizes[cast<AffineDimExpr>(expr).getPosition()]);
  }
  Value empty = tensor::EmptyOp::create(builder, loc, shape, elementType);
  Value zero =
      arith::ConstantOp::create(builder, loc, builder.getZeroAttr(elementType));
  return linalg::FillOp::create(builder, loc, zero, empty).getResult(0);
}

static Value computeMatmul(OpBuilder &builder, Location loc, AffineMap lhsMap,
                           AffineMap rhsMap, AffineMap outputMap, Value lhs,
                           Value rhs, Value output, DictionaryAttr attrs = {},
                           bool canonicalizeLoopOrder = false) {
  SmallVector<AffineMap> maps =
      compressUnusedDims(SmallVector<AffineMap>{lhsMap, rhsMap, outputMap});
  if (canonicalizeLoopOrder) {
    canonicalizeLoopOrderForOutput(maps);
  }
  auto genericOp = linalg::GenericOp::create(
      builder, loc, output.getType(), ValueRange{lhs, rhs}, output, maps,
      getIteratorTypes(maps.back()),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Type accumulatorType = args[2].getType();
        Value convertedLhs =
            convertScalarToDtype(nestedBuilder, nestedLoc, args[0],
                                 accumulatorType, /*isUnsignedCast=*/false);
        Value convertedRhs =
            convertScalarToDtype(nestedBuilder, nestedLoc, args[1],
                                 accumulatorType, /*isUnsignedCast=*/false);
        Value product = arith::MulFOp::create(nestedBuilder, nestedLoc,
                                              convertedLhs, convertedRhs);
        Value sum =
            arith::AddFOp::create(nestedBuilder, nestedLoc, product, args[2]);
        linalg::YieldOp::create(nestedBuilder, nestedLoc, sum);
      });
  if (attrs) {
    genericOp->setDiscardableAttrs(attrs);
  }
  return genericOp.getResult(0);
}

template <typename OpTy>
static Value computePointwise(OpBuilder &builder, Location loc,
                              AffineMap lhsMap, AffineMap rhsMap,
                              AffineMap outputMap, Value lhs, Value rhs,
                              Value output) {
  SmallVector<AffineMap> maps =
      compressUnusedDims(SmallVector<AffineMap>{lhsMap, rhsMap, outputMap});
  SmallVector<utils::IteratorType> iteratorTypes(maps.back().getNumDims(),
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      builder, loc, output.getType(), ValueRange{lhs, rhs}, output, maps,
      iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Type outputType = args[2].getType();
        Value convertedLhs =
            convertScalarToDtype(nestedBuilder, nestedLoc, args[0], outputType,
                                 /*isUnsignedCast=*/false);
        Value convertedRhs =
            convertScalarToDtype(nestedBuilder, nestedLoc, args[1], outputType,
                                 /*isUnsignedCast=*/false);
        Value result =
            OpTy::create(nestedBuilder, nestedLoc, convertedLhs, convertedRhs);
        linalg::YieldOp::create(nestedBuilder, nestedLoc, result);
      });
  return genericOp.getResult(0);
}

// Match AttentionOp's score construction exactly: scale Q in Q's element type
// before the f32 QK contraction. Using the tensor as the destination preserves
// the same rounding point without mutating the SSA input.
static Value scaleTensorInPlace(OpBuilder &builder, Location loc,
                                AffineMap valueMap, AffineMap scaleMap,
                                Value value, Value scale) {
  SmallVector<AffineMap> maps =
      compressUnusedDims(SmallVector<AffineMap>{scaleMap, valueMap});
  SmallVector<utils::IteratorType> iteratorTypes(maps.back().getNumDims(),
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      builder, loc, value.getType(), ValueRange{scale}, value, maps,
      iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value convertedScale = convertScalarToDtype(nestedBuilder, nestedLoc,
                                                    args[0], args[1].getType(),
                                                    /*isUnsignedCast=*/false);
        Value result = arith::MulFOp::create(nestedBuilder, nestedLoc,
                                             convertedScale, args[1]);
        linalg::YieldOp::create(nestedBuilder, nestedLoc, result);
      });
  return genericOp.getResult(0);
}

static Value convertTensor(OpBuilder &builder, Location loc, AffineMap inputMap,
                           AffineMap outputMap, Value input, Value output) {
  SmallVector<AffineMap> maps =
      compressUnusedDims(SmallVector<AffineMap>{inputMap, outputMap});
  SmallVector<utils::IteratorType> iteratorTypes(maps.back().getNumDims(),
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      builder, loc, output.getType(), ValueRange{input}, output, maps,
      iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value result =
            convertScalarToDtype(nestedBuilder, nestedLoc, args[0],
                                 args[1].getType(), /*isUnsignedCast=*/false);
        linalg::YieldOp::create(nestedBuilder, nestedLoc, result);
      });
  return genericOp.getResult(0);
}

// Native backward contractions consume score intermediates in the same
// low-precision type as their primal operand while accumulating in f32.
static Value castContractionLhsToRhsType(OpBuilder &builder, Location loc,
                                         AffineMap lhsMap, Value lhs,
                                         Value rhs) {
  Type lhsElementType = getElementTypeOrSelf(lhs.getType());
  Type rhsElementType = getElementTypeOrSelf(rhs.getType());
  if (lhsElementType == rhsElementType ||
      (!rhsElementType.isF16() && !rhsElementType.isBF16())) {
    return lhs;
  }
  Value converted = tensor::EmptyOp::create(
      builder, loc, tensor::getMixedSizes(builder, loc, lhs), rhsElementType);
  return convertTensor(builder, loc, lhsMap, lhsMap, lhs, converted);
}

static bool hasAppleAttentionBackwardRole(DictionaryAttr attrs) {
  return attrs &&
         static_cast<bool>(attrs.getAs<StringAttr>(
             "iree_codegen.apple_attention_backward_role"));
}

static Value getIntegerMaskCondition(OpBuilder &builder, Location loc,
                                     Value mask) {
  auto maskType = cast<IntegerType>(mask.getType());
  if (maskType.getWidth() == 1) {
    return mask;
  }
  // Match the forward attention operation's legacy i8 predicate semantics.
  return arith::TruncIOp::create(builder, loc, builder.getI1Type(), mask);
}

static bool hasIntegerMask(Value mask) {
  return mask && isa<IntegerType>(getElementTypeOrSelf(mask.getType()));
}

static Value applyMaskToScores(OpBuilder &builder, Location loc,
                               AffineMap scoreMap, AffineMap maskMap,
                               Value scores, Value mask, bool useExp2) {
  SmallVector<AffineMap> maps =
      compressUnusedDims(SmallVector<AffineMap>{maskMap, scoreMap});
  SmallVector<utils::IteratorType> iteratorTypes(maps.back().getNumDims(),
                                                 utils::IteratorType::parallel);
  auto scoreElementType =
      cast<FloatType>(getElementTypeOrSelf(scores.getType()));
  Value maskedOut = arith::ConstantOp::create(
      builder, loc,
      builder.getFloatAttr(
          scoreElementType,
          APFloat::getLargest(scoreElementType.getFloatSemantics(),
                              /*Negative=*/true)));
  auto genericOp = linalg::GenericOp::create(
      builder, loc, scores.getType(), ValueRange{mask}, scores, maps,
      iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value result;
        if (isa<IntegerType>(args[0].getType())) {
          Value condition =
              getIntegerMaskCondition(nestedBuilder, nestedLoc, args[0]);
          result = arith::SelectOp::create(nestedBuilder, nestedLoc, condition,
                                           args[1], maskedOut);
        } else {
          Value convertedMask = convertScalarToDtype(nestedBuilder, nestedLoc,
                                                     args[0], args[1].getType(),
                                                     /*isUnsignedCast=*/false);
          if (useExp2) {
            Value log2e = arith::ConstantOp::create(
                nestedBuilder, nestedLoc,
                nestedBuilder.getFloatAttr(args[1].getType(), M_LOG2E));
            convertedMask = arith::MulFOp::create(nestedBuilder, nestedLoc,
                                                  convertedMask, log2e);
          }
          result = arith::AddFOp::create(nestedBuilder, nestedLoc, args[1],
                                         convertedMask);
        }
        linalg::YieldOp::create(nestedBuilder, nestedLoc, result);
      });
  return genericOp.getResult(0);
}

static Value computeProbabilities(OpBuilder &builder, Location loc,
                                  AffineMap scoreMap, AffineMap logsumexpMap,
                                  Value scores, Value logsumexp, Value mask,
                                  std::optional<AffineMap> maskMap,
                                  Type probabilityElementType, bool useExp2) {
  bool useIntegerMask = hasIntegerMask(mask);
  SmallVector<AffineMap> maps{logsumexpMap};
  SmallVector<Value> inputs{logsumexp};
  if (useIntegerMask) {
    maps.push_back(*maskMap);
    inputs.push_back(mask);
  }
  maps.push_back(scoreMap);
  inputs.push_back(scores);
  maps.push_back(scoreMap);
  maps = compressUnusedDims(maps);
  SmallVector<utils::IteratorType> iteratorTypes(maps.back().getNumDims(),
                                                 utils::IteratorType::parallel);
  SmallVector<OpFoldResult> probabilitySizes =
      tensor::getMixedSizes(builder, loc, scores);
  Value probabilityInit = tensor::EmptyOp::create(
      builder, loc, probabilitySizes, probabilityElementType);
  unsigned scoreOperandIndex = inputs.size() - 1;
  auto genericOp = linalg::GenericOp::create(
      builder, loc, probabilityInit.getType(), inputs, probabilityInit, maps,
      iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value score = args[scoreOperandIndex];
        Value lse = convertScalarToDtype(nestedBuilder, nestedLoc, args[0],
                                         score.getType(),
                                         /*isUnsignedCast=*/false);
        if (useExp2) {
          // AttentionOp returns natural-log LSE even when its scores and
          // exponentiation are in base-2 units.
          Value log2e = arith::ConstantOp::create(
              nestedBuilder, nestedLoc,
              nestedBuilder.getFloatAttr(lse.getType(), M_LOG2E));
          lse = arith::MulFOp::create(nestedBuilder, nestedLoc, lse, log2e);
        }
        Value condition;
        Value zero;
        if (useIntegerMask) {
          condition =
              getIntegerMaskCondition(nestedBuilder, nestedLoc, args[1]);
          zero = arith::ConstantOp::create(
              nestedBuilder, nestedLoc,
              nestedBuilder.getZeroAttr(score.getType()));
          // Sanitize both operands before subtraction. For an all-false row
          // LSE is -inf, while the finite masked score is the lowest finite
          // value. Selecting zeros for both operands prevents even transient
          // infinities or NaNs in lanes that will ultimately be masked out.
          score = arith::SelectOp::create(nestedBuilder, nestedLoc, condition,
                                          score, zero);
          lse = arith::SelectOp::create(nestedBuilder, nestedLoc, condition,
                                        lse, zero);
        }
        Value shifted =
            arith::SubFOp::create(nestedBuilder, nestedLoc, score, lse);
        Value probability =
            useExp2 ? math::Exp2Op::create(nestedBuilder, nestedLoc, shifted)
                          .getResult()
                    : math::ExpOp::create(nestedBuilder, nestedLoc, shifted)
                          .getResult();
        if (useIntegerMask) {
          probability = arith::SelectOp::create(nestedBuilder, nestedLoc,
                                                condition, probability, zero);
        }
        probability = convertScalarToDtype(
            nestedBuilder, nestedLoc, probability, args.back().getType(),
            /*isUnsignedCast=*/false);
        linalg::YieldOp::create(nestedBuilder, nestedLoc, probability);
      });
  return genericOp.getResult(0);
}

using DimSet = llvm::SmallDenseSet<int64_t>;

static DimSet getDimSet(AffineMap map) {
  DimSet dims;
  for (AffineExpr expr : map.getResults()) {
    dims.insert(cast<AffineDimExpr>(expr).getPosition());
  }
  return dims;
}

static DimSet getDimSet(ArrayRef<int64_t> dims) {
  return DimSet(dims.begin(), dims.end());
}

static bool equalSets(const DimSet &lhs, const DimSet &rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(lhs, [&](int64_t dim) { return rhs.contains(dim); });
}

static void unionInto(DimSet &result, const DimSet &other) {
  result.insert(other.begin(), other.end());
}

static bool intersects(const DimSet &lhs, const DimSet &rhs) {
  return llvm::any_of(lhs, [&](int64_t dim) { return rhs.contains(dim); });
}

static LogicalResult verifyMapShape(Operation *op, StringRef operandName,
                                    RankedTensorType type, AffineMap map,
                                    SmallVectorImpl<int64_t> &domainShape,
                                    SmallVectorImpl<bool> &knownDims) {
  if (map.getNumResults() != type.getRank()) {
    return op->emitOpError("rank mismatch for ")
           << operandName << ": map has " << map.getNumResults()
           << " results but tensor has rank " << type.getRank();
  }
  for (auto [operandDim, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      return op->emitOpError("expected projected-permutation map for ")
             << operandName;
    }
    unsigned domainDim = dimExpr.getPosition();
    int64_t size = type.getDimSize(operandDim);
    if (ShapedType::isDynamic(size)) {
      continue;
    }
    if (!knownDims[domainDim]) {
      knownDims[domainDim] = true;
      domainShape[domainDim] = size;
      continue;
    }
    if (domainShape[domainDim] != size) {
      return op->emitOpError("shape mismatch for ")
             << operandName << " dimension " << operandDim << ": expected "
             << domainShape[domainDim] << " but got " << size;
    }
  }
  return success();
}

} // namespace

LogicalResult AttentionBackwardOp::verify() {
  if (DictionaryAttr config = getDecompositionConfigAttr()) {
    if (Attribute useExp2 = config.get("use_exp2");
        useExp2 && !isa<BoolAttr>(useExp2)) {
      return emitOpError(
          "expected decomposition_config entry 'use_exp2' to be a boolean");
    }
    for (StringRef attrName :
         {getQKAttrStr(), getDPAttrStr(), getDQAttrStr(), getDKAttrStr(),
          getDVAttrStr()}) {
      if (Attribute attrs = config.get(attrName);
          attrs && !isa<DictionaryAttr>(attrs)) {
        return emitOpError() << "expected decomposition_config entry '"
                             << attrName << "' to be a dictionary";
      }
    }
  }

  SmallVector<AffineMap> maps = getIndexingMapsArray();
  if (maps.size() != getOperation()->getNumOperands()) {
    return emitOpError("expected one indexing map for each operand");
  }

  int64_t domainRank = getQueryMap().getNumDims();
  for (auto [index, map] : llvm::enumerate(maps)) {
    if (map.getNumDims() != domainRank || map.getNumSymbols() != 0) {
      return emitOpError("all indexing maps must have the same symbol-free "
                         "domain; map ")
             << index << " does not";
    }
    if (!map.isProjectedPermutation()) {
      return emitOpError("all indexing maps must be projected permutations; "
                         "map ")
             << index << " is not";
    }
  }

  // AttentionOpDetail assumes projected-permutation maps and asserts when that
  // precondition is violated, so all structural map checks must precede it.
  FailureOr<AttentionOpDetail> maybeOpInfo = AttentionOpDetail::get(
      getQueryMap(), getKeyMap(), getValueMap(), getOutputMap());
  if (failed(maybeOpInfo)) {
    return emitOpError("failed to verify attention indexing maps");
  }
  AttentionOpDetail opInfo = *maybeOpInfo;

  DimSet batchDims = getDimSet(opInfo.getBatchDims());
  DimSet mDims = getDimSet(opInfo.getMDims());
  DimSet k1Dims = getDimSet(opInfo.getK1Dims());
  DimSet k2Dims = getDimSet(opInfo.getK2Dims());
  DimSet nDims = getDimSet(opInfo.getNDims());
  std::array<const DimSet *, 5> categories = {&batchDims, &mDims, &k1Dims,
                                              &k2Dims, &nDims};
  for (auto [i, lhs] : llvm::enumerate(categories)) {
    for (const DimSet *rhs : llvm::drop_begin(categories, i + 1)) {
      if (intersects(*lhs, *rhs)) {
        return emitOpError(
            "attention batch/M/K1/K2/N dimension categories must be "
            "pairwise disjoint");
      }
    }
  }
  DimSet allDims;
  for (const DimSet *category : categories) {
    unionInto(allDims, *category);
  }
  if (allDims.size() != domainRank) {
    return emitOpError(
        "attention batch/M/K1/K2/N dimensions must cover the map domain");
  }
  auto checkMapCategories =
      [&](StringRef name, AffineMap actual,
          std::initializer_list<const DimSet *> expectedCategories)
      -> LogicalResult {
    DimSet expected;
    for (const DimSet *category : expectedCategories) {
      unionInto(expected, *category);
    }
    if (!equalSets(getDimSet(actual), expected)) {
      return emitOpError() << name
                           << " map has dimensions inconsistent with attention";
    }
    return success();
  };
  if (failed(checkMapCategories("query", getQueryMap(),
                                {&batchDims, &mDims, &k1Dims})) ||
      failed(checkMapCategories("key", getKeyMap(),
                                {&batchDims, &k2Dims, &k1Dims})) ||
      failed(checkMapCategories("value", getValueMap(),
                                {&batchDims, &k2Dims, &nDims})) ||
      failed(checkMapCategories("output", getOutputMap(),
                                {&batchDims, &mDims, &nDims}))) {
    return failure();
  }

  if (getOutputGradMap() != getOutputMap()) {
    return emitOpError("output gradient map must equal output map");
  }
  if (getQueryGradMap() != getQueryMap() || getKeyGradMap() != getKeyMap() ||
      getValueGradMap() != getValueMap()) {
    return emitOpError("gradient maps must equal their primal operand maps");
  }
  if (getScaleMap().getNumResults() != 0) {
    return emitOpError("scale indexing map must have no results");
  }

  SmallVector<AffineExpr> expectedLseExprs;
  for (int64_t dim :
       llvm::concat<const int64_t>(opInfo.getBatchDims(), opInfo.getMDims())) {
    expectedLseExprs.push_back(getAffineDimExpr(dim, getContext()));
  }
  AffineMap expectedLseMap =
      AffineMap::get(opInfo.getDomainRank(), /*symbolCount=*/0,
                     expectedLseExprs, getContext());
  if (getLogsumexpMap() != expectedLseMap) {
    return emitOpError(
        "logsumexp map must contain the attention batch and query-row "
        "dimensions in canonical order");
  }

  if (std::optional<AffineMap> maskMap = getMaskMap()) {
    // Keep the score map alive while constructing the set. Calling getSMap()
    // independently for begin() and end() creates two temporary maps whose
    // result ranges do not form a valid iterator pair.
    AffineMap scoreMap = opInfo.getSMap();
    llvm::SmallDenseSet<AffineExpr> scoreDims(scoreMap.getResults().begin(),
                                              scoreMap.getResults().end());
    llvm::SmallDenseSet<AffineExpr> seen;
    for (AffineExpr expr : maskMap->getResults()) {
      if (!isa<AffineDimExpr>(expr) || !scoreDims.contains(expr) ||
          !seen.insert(expr).second) {
        return emitOpError(
            "mask map must be a projected permutation of score dimensions");
      }
    }
  }

  auto queryType = cast<RankedTensorType>(getQuery().getType());
  auto keyType = cast<RankedTensorType>(getKey().getType());
  auto valueType = cast<RankedTensorType>(getValue().getType());
  auto outputType = cast<RankedTensorType>(getOutput().getType());
  auto outputGradType = cast<RankedTensorType>(getOutputGrad().getType());
  auto logsumexpType = cast<RankedTensorType>(getLogsumexp().getType());
  auto queryGradType = cast<RankedTensorType>(getQueryGrad().getType());
  auto keyGradType = cast<RankedTensorType>(getKeyGrad().getType());
  auto valueGradType = cast<RankedTensorType>(getValueGrad().getType());

  // Do not zip ArrayRefs built from initializer-list temporaries: the backing
  // arrays have already expired when the range-for begins.
  const std::array<std::pair<StringRef, RankedTensorType>, 5> floatOperands = {
      std::pair<StringRef, RankedTensorType>{"query", queryType},
      {"key", keyType},
      {"value", valueType},
      {"output", outputType},
      {"output gradient", outputGradType},
  };
  for (auto [name, type] : floatOperands) {
    Type elementType = type.getElementType();
    if (!elementType.isF16() && !elementType.isBF16() && !elementType.isF32()) {
      return emitOpError() << name
                           << " must have f16, bf16, or f32 element type";
    }
  }
  Type scaleType = getScale().getType();
  if (!scaleType.isF16() && !scaleType.isBF16() && !scaleType.isF32()) {
    return emitOpError("scale must have f16, bf16, or f32 type");
  }
  if (!logsumexpType.getElementType().isF32()) {
    return emitOpError("logsumexp must have f32 element type");
  }
  if (queryGradType != queryType || keyGradType != keyType ||
      valueGradType != valueType) {
    return emitOpError(
        "gradient init/result types must exactly match query, key, and value");
  }
  if (outputGradType != outputType) {
    return emitOpError("output gradient type must exactly match output type");
  }
  if (Value mask = getMask()) {
    Type maskElementType = getElementTypeOrSelf(mask.getType());
    if (!isa<FloatType, IntegerType>(maskElementType)) {
      return emitOpError(
          "mask must have floating-point or integer element type");
    }
    if (isa<FloatType>(maskElementType) && !maskElementType.isF16() &&
        !maskElementType.isBF16() && !maskElementType.isF32()) {
      return emitOpError(
          "floating-point mask must have f16, bf16, or f32 element type");
    }
  }

  SmallVector<int64_t> domainShape(domainRank);
  SmallVector<bool> knownDims(domainRank, false);
  SmallVector<std::tuple<StringRef, RankedTensorType, AffineMap>> shaped = {
      {"query", queryType, getQueryMap()},
      {"key", keyType, getKeyMap()},
      {"value", valueType, getValueMap()},
      {"output", outputType, getOutputMap()},
      {"output gradient", outputGradType, getOutputGradMap()},
      {"logsumexp", logsumexpType, getLogsumexpMap()},
      {"query gradient", queryGradType, getQueryGradMap()},
      {"key gradient", keyGradType, getKeyGradMap()},
      {"value gradient", valueGradType, getValueGradMap()},
  };
  if (Value mask = getMask()) {
    shaped.emplace_back("mask", cast<RankedTensorType>(mask.getType()),
                        *getMaskMap());
  }
  for (auto [name, type, map] : shaped) {
    if (failed(verifyMapShape(getOperation(), name, type, map, domainShape,
                              knownDims))) {
      return failure();
    }
  }
  return success();
}

MutableOperandRange AttentionBackwardOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, getOperation()->getNumOperands() - 3,
                             /*length=*/3);
}

LogicalResult AttentionBackwardOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(builder, reifiedReturnShapes);
}

SmallVector<AffineMap> AttentionBackwardOp::getIndexingMapsArray() {
  return SmallVector<AffineMap>(
      getIndexingMaps().getAsValueRange<AffineMapAttr>());
}

FailureOr<SmallVector<Value>>
AttentionBackwardOp::decomposeOperation(OpBuilder &builder) {
  Location loc = getLoc();
  FailureOr<AttentionOpDetail> maybeOpInfo = AttentionOpDetail::get(
      getQueryMap(), getKeyMap(), getValueMap(), getOutputMap());
  if (failed(maybeOpInfo)) {
    return failure();
  }
  AttentionOpDetail opInfo = *maybeOpInfo;
  AffineMap scoreMap = opInfo.getSMap();
  AffineMap scalarMap = AffineMap::get(getQueryMap().getNumDims(),
                                       /*symbolCount=*/0, getContext());

  SmallVector<OpFoldResult> domainSizes(getQueryMap().getNumDims());
  SmallVector<bool> foundDims(getQueryMap().getNumDims(), false);
  auto inferSizes = [&](Value value, AffineMap map) {
    for (auto [operandDim, expr] : llvm::enumerate(map.getResults())) {
      unsigned domainDim = cast<AffineDimExpr>(expr).getPosition();
      if (foundDims[domainDim]) {
        continue;
      }
      foundDims[domainDim] = true;
      domainSizes[domainDim] =
          tensor::getMixedSize(builder, loc, value, operandDim);
    }
  };
  inferSizes(getQuery(), getQueryMap());
  inferSizes(getKey(), getKeyMap());
  inferSizes(getValue(), getValueMap());
  if (llvm::is_contained(foundDims, false)) {
    return failure();
  }

  Type f32Type = builder.getF32Type();
  DictionaryAttr config = getDecompositionConfigAttr();
  DictionaryAttr qkAttrs, dpAttrs, dqAttrs, dkAttrs, dvAttrs;
  bool useExp2 = true;
  if (config) {
    qkAttrs = config.getAs<DictionaryAttr>(getQKAttrStr());
    dpAttrs = config.getAs<DictionaryAttr>(getDPAttrStr());
    dqAttrs = config.getAs<DictionaryAttr>(getDQAttrStr());
    dkAttrs = config.getAs<DictionaryAttr>(getDKAttrStr());
    dvAttrs = config.getAs<DictionaryAttr>(getDVAttrStr());
    if (auto useExp2Attr = config.getAs<BoolAttr>(getUseExp2AttrStr())) {
      useExp2 = useExp2Attr.getValue();
    }
  }
  bool nativeQK = hasAppleAttentionBackwardRole(qkAttrs);
  bool nativeDP = hasAppleAttentionBackwardRole(dpAttrs);
  bool nativeDQ = hasAppleAttentionBackwardRole(dqAttrs);
  bool nativeDK = hasAppleAttentionBackwardRole(dkAttrs);
  bool nativeDV = hasAppleAttentionBackwardRole(dvAttrs);

  // Recompute scores and normalized probabilities without materializing any
  // forward intermediates. LSE is supplied by the paired forward operation.
  Value scoreScale = getScale();
  if (useExp2) {
    Value log2e = arith::ConstantOp::create(
        builder, loc, builder.getFloatAttr(scoreScale.getType(), M_LOG2E));
    scoreScale = arith::MulFOp::create(builder, loc, scoreScale, log2e);
  }
  Value scaledQuery = scaleTensorInPlace(builder, loc, getQueryMap(), scalarMap,
                                         getQuery(), scoreScale);
  Value scores = createZeroTensor(builder, loc, scoreMap, domainSizes, f32Type);
  scores = computeMatmul(builder, loc, getQueryMap(), getKeyMap(), scoreMap,
                         scaledQuery, getKey(), scores, qkAttrs);
  Value probabilityMask = getMask();
  if (probabilityMask && nativeQK) {
    // Keep causal-mask construction out of the vector-distributed QK shader.
    // A 512x512 iota/compare producer otherwise expands dramatically during
    // SPIR-V vector lowering even though its i1 materialization is tiny.
    probabilityMask = IREE::Util::OptimizationBarrierOp::create(
                          builder, loc, ValueRange{probabilityMask})
                          .getResult(0);
  }
  // Integer masks are applied directly in computeProbabilities. Reapplying
  // them to an intermediate score tensor would leave a dead, score-sized
  // second result in the fused native QK epilogue.
  bool applyMaskToIntermediate =
      probabilityMask && (!nativeQK || !hasIntegerMask(probabilityMask));
  if (applyMaskToIntermediate) {
    // CPU contraction-root codegen cannot yet distribute a fused score
    // epilogue. The native Metal path deliberately keeps scale, mask, and
    // probability normalization fused with its isolated QK contraction.
    if (!nativeQK) {
      auto scoreMatmulBarrier = IREE::Util::OptimizationBarrierOp::create(
          builder, loc, ValueRange{scores});
      scores = scoreMatmulBarrier.getResult(0);
    }
    scores = applyMaskToScores(builder, loc, scoreMap, *getMaskMap(), scores,
                               probabilityMask, useExp2);
  }
  // The portable path retains its explicit score boundary. On native Metal,
  // materialize only probabilities: this removes one score-sized f32 tensor
  // without adding another logical MMA to the QK shader.
  if (!nativeQK) {
    auto scoreBarrier = IREE::Util::OptimizationBarrierOp::create(
        builder, loc, ValueRange{scores});
    scores = scoreBarrier.getResult(0);
  }
  Type probabilityElementType =
      nativeQK ? getElementTypeOrSelf(getQuery().getType()) : f32Type;
  Value probabilities = computeProbabilities(
      builder, loc, scoreMap, getLogsumexpMap(), scores, getLogsumexp(),
      probabilityMask, getMaskMap(), probabilityElementType, useExp2);
  if (nativeQK) {
    probabilities = IREE::Util::OptimizationBarrierOp::create(
                        builder, loc, ValueRange{probabilities})
                        .getResult(0);
  }

  // D = rowsum(dO * O), expressed as a contraction so it can share the same
  // f32 accumulation path as the other backward matmuls.
  Value rowDot =
      createZeroTensor(builder, loc, getLogsumexpMap(), domainSizes, f32Type);
  rowDot =
      computeMatmul(builder, loc, getOutputGradMap(), getOutputMap(),
                    getLogsumexpMap(), getOutputGrad(), getOutput(), rowDot);
  if (nativeDP) {
    // Materialize the score-row reduction, which is tiny compared to dP. If
    // left fusible, dispatch formation groups the dS epilogue with rowDot and
    // forces the score-sized f32 dP tensor across a dispatch boundary. Keeping
    // rowDot separate instead lets dP own and fuse the epilogue, so only the
    // final low-precision dS tensor is materialized.
    rowDot = IREE::Util::OptimizationBarrierOp::create(
                 builder, loc, ValueRange{rowDot})
                 .getResult(0);
  }

  // dP = dO @ V^T.
  Value probabilityGrad =
      createZeroTensor(builder, loc, scoreMap, domainSizes, f32Type);
  probabilityGrad =
      computeMatmul(builder, loc, getOutputGradMap(), getValueMap(), scoreMap,
                    getOutputGrad(), getValue(), probabilityGrad, dpAttrs);

  // dS = P * (dP - D) * scale.
  // Do not reuse dP as the destination. A fresh destination makes dP a
  // single-use producer so dispatch formation can fuse this pointwise
  // epilogue and materialize only the final low-precision dS tensor.
  Value scoreGradInit = tensor::EmptyOp::create(
      builder, loc, tensor::getMixedSizes(builder, loc, probabilityGrad),
      f32Type);
  Value scoreGrad = computePointwise<arith::SubFOp>(
      builder, loc, scoreMap, getLogsumexpMap(), scoreMap, probabilityGrad,
      rowDot, scoreGradInit);
  scoreGrad = computePointwise<arith::MulFOp>(builder, loc, scoreMap, scoreMap,
                                              scoreMap, probabilities,
                                              scoreGrad, scoreGrad);
  scoreGrad = computePointwise<arith::MulFOp>(builder, loc, scoreMap, scalarMap,
                                              scoreMap, scoreGrad, getScale(),
                                              scoreGrad);

  Value queryGradF32 =
      createZeroTensor(builder, loc, getQueryMap(), domainSizes, f32Type);
  Value queryScoreGrad =
      nativeDQ ? castContractionLhsToRhsType(builder, loc, scoreMap, scoreGrad,
                                             getKey())
               : scoreGrad;
  if (nativeDQ) {
    queryScoreGrad = IREE::Util::OptimizationBarrierOp::create(
                         builder, loc, ValueRange{queryScoreGrad})
                         .getResult(0);
  }
  queryGradF32 =
      computeMatmul(builder, loc, scoreMap, getKeyMap(), getQueryMap(),
                    queryScoreGrad, getKey(), queryGradF32, dqAttrs);

  Value keyGradF32 =
      createZeroTensor(builder, loc, getKeyMap(), domainSizes, f32Type);
  Value keyScoreGrad = nativeDK
                           ? castContractionLhsToRhsType(builder, loc, scoreMap,
                                                         scoreGrad, getQuery())
                           : scoreGrad;
  if (nativeDK) {
    keyScoreGrad = IREE::Util::OptimizationBarrierOp::create(
                       builder, loc, ValueRange{keyScoreGrad})
                       .getResult(0);
  }
  keyGradF32 = computeMatmul(builder, loc, scoreMap, getQueryMap(), getKeyMap(),
                             keyScoreGrad, getQuery(), keyGradF32, dkAttrs,
                             /*canonicalizeLoopOrder=*/nativeDK);

  Value valueGradF32 =
      createZeroTensor(builder, loc, getValueMap(), domainSizes, f32Type);
  Value valueProbabilities =
      nativeDV ? castContractionLhsToRhsType(builder, loc, scoreMap,
                                             probabilities, getOutputGrad())
               : probabilities;
  if (nativeDV) {
    valueProbabilities = IREE::Util::OptimizationBarrierOp::create(
                             builder, loc, ValueRange{valueProbabilities})
                             .getResult(0);
  }
  valueGradF32 =
      computeMatmul(builder, loc, scoreMap, getOutputGradMap(), getValueMap(),
                    valueProbabilities, getOutputGrad(), valueGradF32, dvAttrs);

  Value queryGrad =
      convertTensor(builder, loc, getQueryMap(), getQueryGradMap(),
                    queryGradF32, getQueryGrad());
  Value keyGrad = convertTensor(builder, loc, getKeyMap(), getKeyGradMap(),
                                keyGradF32, getKeyGrad());
  Value valueGrad =
      convertTensor(builder, loc, getValueMap(), getValueGradMap(),
                    valueGradF32, getValueGrad());
  return SmallVector<Value>{queryGrad, keyGrad, valueGrad};
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
