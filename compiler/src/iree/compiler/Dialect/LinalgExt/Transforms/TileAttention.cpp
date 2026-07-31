// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_CONVERTATTENTIONTOONLINEATTENTIONPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct ConvertAttentionToOnlineAttentionPass final
    : impl::ConvertAttentionToOnlineAttentionPassBase<
          ConvertAttentionToOnlineAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect, arith::ArithDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

} // namespace

static Value createLogsumexp(RewriterBase &rewriter, Location loc,
                             AffineMap maxMap, AffineMap sumMap,
                             AffineMap logsumexpMap, Value max, Value sum,
                             Value logsumexpInit, bool useExp2,
                             bool negativeInfForZeroSum) {
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{maxMap, sumMap, logsumexpMap});
  maxMap = compressedMaps[0];
  sumMap = compressedMaps[1];
  logsumexpMap = compressedMaps[2];

  SmallVector<utils::IteratorType> iteratorTypes(maxMap.getNumDims(),
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, logsumexpInit.getType(), ValueRange{max, sum},
      logsumexpInit, SmallVector<AffineMap>{maxMap, sumMap, logsumexpMap},
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        auto resultType = cast<FloatType>(args[2].getType());
        Value max = convertScalarToDtype(b, loc, args[0], resultType,
                                         /*isUnsignedCast=*/false);
        Value sum = convertScalarToDtype(b, loc, args[1], resultType,
                                         /*isUnsignedCast=*/false);
        Value logSum = useExp2 ? math::Log2Op::create(b, loc, sum).getResult()
                               : math::LogOp::create(b, loc, sum).getResult();
        Value result = arith::AddFOp::create(b, loc, max, logSum);
        if (useExp2) {
          Value ln2 = arith::ConstantOp::create(
              b, loc, b.getFloatAttr(resultType, 0.6931471805599453));
          result = arith::MulFOp::create(b, loc, result, ln2);
        }
        if (negativeInfForZeroSum) {
          Value zero = arith::ConstantOp::create(
              b, loc, b.getFloatAttr(resultType, 0.0));
          Value sumIsZero = arith::CmpFOp::create(
              b, loc, arith::CmpFPredicate::OEQ, sum, zero);
          Value negativeInf = arith::ConstantOp::create(
              b, loc,
              b.getFloatAttr(resultType,
                             APFloat::getInf(resultType.getFloatSemantics(),
                                             /*Negative=*/true)));
          result =
              arith::SelectOp::create(b, loc, sumIsZero, negativeInf, result);
        }
        linalg::YieldOp::create(b, loc, result);
      });
  return genericOp.getResult(0);
}

void convertToOnlineAttention(IREE::LinalgExt::AttentionOp attnOp,
                              SmallVectorImpl<Operation *> &ops,
                              RewriterBase &rewriter) {
  rewriter.setInsertionPoint(attnOp);

  Location loc = attnOp.getLoc();
  MLIRContext *ctx = attnOp.getContext();

  FailureOr<AttentionOpDetail> maybeOpInfo =
      AttentionOpDetail::get(attnOp.getQueryMap(), attnOp.getKeyMap(),
                             attnOp.getValueMap(), attnOp.getOutputMap());
  assert(succeeded(maybeOpInfo) && "Invalid attention indexing maps");
  AttentionOpDetail opInfo = maybeOpInfo.value();

  // Create standard maps for max and sum: (batch, m)
  int64_t rank = opInfo.getDomainRank();
  AffineMap maxMap = AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, ctx);
  for (auto dim :
       llvm::concat<const int64_t>(opInfo.getBatchDims(), opInfo.getMDims())) {
    maxMap = maxMap.insertResult(rewriter.getAffineDimExpr(dim),
                                 maxMap.getNumResults());
  }
  AffineMap sumMap = maxMap;

  AffineMap accMap = attnOp.getOutputMap();
  bool useExp2 = true;
  if (DictionaryAttr config = attnOp.getDecompositionConfigAttr()) {
    if (auto useExp2Attr =
            config.getAs<BoolAttr>(AttentionOp::getUseExp2AttrStr())) {
      useExp2 = useExp2Attr.getValue();
    }
  }

  SmallVector<Range> domain = attnOp.getIterationDomain(rewriter);

  // Create fill for acc, max and sum.
  // TODO: Acc should not need a fill. The attention op should get a filled
  // input instead of an empty input.

  SmallVector<OpFoldResult> sizes =
      llvm::map_to_vector(domain, [](Range x) { return x.size; });
  SmallVector<OpFoldResult> accSize =
      applyPermutationMap<OpFoldResult>(accMap, sizes);
  SmallVector<OpFoldResult> rowRedSize =
      applyPermutationMap<OpFoldResult>(maxMap, sizes);

  Type f32Type = rewriter.getF32Type();
  Value acc = tensor::EmptyOp::create(rewriter, loc, accSize, f32Type);
  Value rowRedEmpty =
      tensor::EmptyOp::create(rewriter, loc, rowRedSize, f32Type);

  Value accInit =
      arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type, rewriter,
                              loc, /*useOnlyFiniteValue=*/true);
  Value maxInit =
      arith::getIdentityValue(arith::AtomicRMWKind::maximumf, f32Type, rewriter,
                              loc, /*useOnlyFiniteValue=*/true);
  Value sumInit = arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type,
                                          rewriter, loc);

  Value accFill =
      linalg::FillOp::create(rewriter, loc, ValueRange{accInit}, acc)
          .getResult(0);
  Value maxFill =
      linalg::FillOp::create(rewriter, loc, ValueRange{maxInit}, rowRedEmpty)
          .getResult(0);
  Value sumFill =
      linalg::FillOp::create(rewriter, loc, ValueRange{sumInit}, rowRedEmpty)
          .getResult(0);

  // Create online attention op.
  SmallVector<AffineMap> indexingMaps = attnOp.getIndexingMapsForOperands();
  indexingMaps.push_back(accMap);
  indexingMaps.push_back(maxMap);
  indexingMaps.push_back(sumMap);

  Value mask = attnOp.getMask() ? attnOp.getMask() : Value();
  bool hasIntegerMask =
      mask && isa<IntegerType>(getElementTypeOrSelf(mask.getType()));

  OnlineAttentionOp onlineAttn = OnlineAttentionOp::create(
      rewriter, loc,
      TypeRange{accFill.getType(), maxFill.getType(), sumFill.getType()},
      attnOp.getQuery(), attnOp.getKey(), attnOp.getValue(), attnOp.getScale(),
      mask, accFill, maxFill, sumFill,
      rewriter.getAffineMapArrayAttr(indexingMaps),
      attnOp.getDecompositionConfigAttr());

  rewriter.cloneRegionBefore(attnOp.getRegion(), onlineAttn.getRegion(),
                             onlineAttn.getRegion().begin());
  onlineAttn->setDiscardableAttrs(attnOp->getDiscardableAttrDictionary());
  ops.push_back(onlineAttn);

  Value x = onlineAttn.getResult(0);
  Value max = onlineAttn.getResult(1);
  Value sum = onlineAttn.getResult(2);

  // Merge the outputs of online attention:
  //  x = (1 / sum) * x

  // Compress the indexing maps.
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{sumMap, accMap, accMap});

  SmallVector<utils::IteratorType> iteratorTypes(compressedMaps[0].getNumDims(),
                                                 utils::IteratorType::parallel);

  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, attnOp.getOutput().getType(), ValueRange{sum, x},
      attnOp.getOutput(), compressedMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value sum = args[0];
        Value sumIsZero;
        Value zero;
        Value one = arith::ConstantOp::create(
            b, loc, b.getFloatAttr(sum.getType(), 1.0));
        if (hasIntegerMask) {
          // An all-false mask produces the neutral online state (x = 0,
          // sum = 0). Avoid forming 1 / 0 and define its final output as zero.
          zero = arith::ConstantOp::create(b, loc,
                                           b.getFloatAttr(sum.getType(), 0.0));
          sumIsZero = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ,
                                            sum, zero);
          sum = arith::SelectOp::create(b, loc, sumIsZero, one, sum);
        }
        Value reciprocal = arith::DivFOp::create(b, loc, one, sum);
        // Both sum and x are in fp32, as created earlier, so we only need
        // to cast after the mul.
        Value result = arith::MulFOp::create(b, loc, reciprocal, args[1]);
        if (hasIntegerMask) {
          result = arith::SelectOp::create(b, loc, sumIsZero, zero, result);
        }
        // Cast result to the required type by attention output.
        result = convertScalarToDtype(b, loc, result, args[2].getType(),
                                      /*isUnsignedCast=*/false);
        linalg::YieldOp::create(b, loc, result);
      });
  ops.push_back(genericOp);

  SmallVector<Value> replacements{genericOp.getResult(0)};
  if (Value logsumexpInit = attnOp.getLogsumexp()) {
    Value logsumexp = createLogsumexp(rewriter, loc, maxMap, sumMap,
                                      *attnOp.getLogsumexpMap(), max, sum,
                                      logsumexpInit, useExp2,
                                      /*negativeInfForZeroSum=*/hasIntegerMask);
    ops.push_back(logsumexp.getDefiningOp());
    replacements.push_back(logsumexp);
  }
  rewriter.replaceOp(attnOp, replacements);
}

void ConvertAttentionToOnlineAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  getOperation()->walk([&](AttentionOp attnOp) {
    SmallVector<Operation *> ops;
    convertToOnlineAttention(attnOp, ops, rewriter);
  });
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
