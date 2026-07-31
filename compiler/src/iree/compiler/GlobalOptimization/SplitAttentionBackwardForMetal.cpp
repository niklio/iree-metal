// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_SPLITATTENTIONBACKWARDFORMETALPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

using IREE::LinalgExt::AttentionBackwardOp;

static bool isAppleAttentionBackwardEligible(AttentionBackwardOp op) {
  FailureOr<IREE::LinalgExt::AttentionOpDetail> maybeOpInfo =
      IREE::LinalgExt::AttentionOpDetail::get(op.getQueryMap(), op.getKeyMap(),
                                              op.getValueMap(),
                                              op.getOutputMap());
  if (failed(maybeOpInfo)) {
    return false;
  }
  IREE::LinalgExt::AttentionOpDetail opInfo = *maybeOpInfo;

  auto getElementType = [](Value value) {
    return cast<RankedTensorType>(value.getType()).getElementType();
  };
  auto isNativeInputType = [](Type type) {
    return type.isF16() || type.isBF16();
  };
  Type queryType = getElementType(op.getQuery());
  Type keyType = getElementType(op.getKey());
  Type valueType = getElementType(op.getValue());
  Type outputGradType = getElementType(op.getOutputGrad());
  if (!isNativeInputType(queryType) || queryType != keyType ||
      queryType != valueType || valueType != outputGradType) {
    return false;
  }

  SmallVector<int64_t> domainSizes(opInfo.getDomainRank(),
                                   ShapedType::kDynamic);
  auto inferStaticSizes = [&](Value value, AffineMap map) {
    auto type = cast<RankedTensorType>(value.getType());
    for (auto [operandDim, expr] : llvm::enumerate(map.getResults())) {
      unsigned domainDim = cast<AffineDimExpr>(expr).getPosition();
      int64_t size = type.getDimSize(operandDim);
      if (ShapedType::isDynamic(size)) {
        continue;
      }
      int64_t &knownSize = domainSizes[domainDim];
      if (!ShapedType::isDynamic(knownSize) && knownSize != size) {
        return false;
      }
      knownSize = size;
    }
    return true;
  };
  if (!inferStaticSizes(op.getQuery(), op.getQueryMap()) ||
      !inferStaticSizes(op.getKey(), op.getKeyMap()) ||
      !inferStaticSizes(op.getValue(), op.getValueMap())) {
    return false;
  }

  auto hasEligibleInnerDim = [&](ArrayRef<int64_t> dims) {
    if (dims.empty()) {
      return false;
    }
    int64_t size = domainSizes[dims.back()];
    return size >= 8 && size % 8 == 0;
  };
  return hasEligibleInnerDim(opInfo.getMDims()) &&
         hasEligibleInnerDim(opInfo.getK1Dims()) &&
         hasEligibleInnerDim(opInfo.getK2Dims()) &&
         hasEligibleInnerDim(opInfo.getNDims());
}

static void configureMaterializedAttentionBackward(AttentionBackwardOp op) {
  Builder builder(op.getContext());
  NamedAttrList config(op.getDecompositionConfigAttr());
  for (StringRef role : {AttentionBackwardOp::getQKAttrStr(),
                         AttentionBackwardOp::getDPAttrStr(),
                         AttentionBackwardOp::getDQAttrStr(),
                         AttentionBackwardOp::getDKAttrStr(),
                         AttentionBackwardOp::getDVAttrStr()}) {
    auto existingAttrs = dyn_cast_or_null<DictionaryAttr>(config.get(role));
    NamedAttrList roleAttrs(existingAttrs);
    roleAttrs.set("iree_codegen.apple_attention_backward_role",
                  builder.getStringAttr(role));
    config.set(role, roleAttrs.getDictionary(op.getContext()));
  }
  op.setDecompositionConfigAttr(config.getDictionary(op.getContext()));
}

class SplitAttentionBackwardForMetalPass final
    : public impl::SplitAttentionBackwardForMetalPassBase<
          SplitAttentionBackwardForMetalPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<AttentionBackwardOp> candidates;
    moduleOp.walk([&](AttentionBackwardOp op) { candidates.push_back(op); });
    if (candidates.empty()) {
      return;
    }

    IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
    if (failed(affinityAnalysis.run())) {
      return signalPassFailure();
    }
    IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    for (AttentionBackwardOp op : candidates) {
      SmallVector<IREE::Stream::AffinityAttr> affinities;
      if (!affinityAnalysis.tryLookupExecutionAffinity(op, affinities) ||
          affinities.empty()) {
        continue;
      }
      bool allAffinitiesAreMetal = true;
      for (IREE::Stream::AffinityAttr affinity : affinities) {
        SetVector<IREE::HAL::ExecutableTargetAttr> executableTargets;
        deviceAnalysis.gatherRequiredExecutableTargets(affinity, op,
                                                       executableTargets);
        if (executableTargets.empty() ||
            llvm::any_of(executableTargets, [](auto target) {
              return target.getBackend().getValue() != "metal-spirv";
            })) {
          allAffinitiesAreMetal = false;
          break;
        }
      }
      if (allAffinitiesAreMetal && isAppleAttentionBackwardEligible(op)) {
        configureMaterializedAttentionBackward(op);
      }
    }
  }
};

}  // namespace
}  // namespace mlir::iree_compiler::GlobalOptimization
