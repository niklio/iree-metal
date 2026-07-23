// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVFinalVectorLowering.cpp ---------------------------------------===//
//
// This pass hosts final steps towards lowering vectors ops to meet SPIR-V
// requirements--it applies vector lowering patterns to convert vector ops
// to more basic forms.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-final-vector-lowering"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVFINALVECTORLOWERINGPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {

void debugPrint(mlir::FunctionOpInterface funcOp, const char *message) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- " << message << " ---//\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

class SPIRVFinalVectorLoweringPass final
    : public impl::SPIRVFinalVectorLoweringPassBase<
          SPIRVFinalVectorLoweringPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // vector.gather lowering patterns target scf ops.
    registry.insert<scf::SCFDialect, vector::VectorDialect, ub::UBDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::FunctionOpInterface funcOp = getOperation();

    // Lower vector transfer permutation map.
    {
      RewritePatternSet patterns(context);
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(patterns,
                                                                 context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after lowering transfer ops");

    // Lower vector broadcast/transpose and contraction.
    {
      RewritePatternSet patterns(context);
      auto options = vector::VectorTransformsOptions()
                         .setVectorTransformsOptions(
                             vector::VectorContractLowering::OuterProduct)
                         .setVectorTransposeLowering(
                             vector::VectorTransposeLowering::EltWise);
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      // nlearn (cont81): in the coop-flash pipeline, DON'T lower vector.contract to
      // outerproduct/fma here — that scalarizes the qk/pv coop matmuls before
      // ConvertToSPIRV can emit spirv.KHR.CooperativeMatrix from them (the standard
      // coop matmul pipeline never runs this pass; attention needs it only for the
      // softmax reductions, which carry no contracts). Leaving the coop contracts
      // intact lets them survive to the coop conversion. Gated so normal codegen is
      // unchanged.
      if (!getenv("NLEARN_COOP_ATTN_KEEPCONTRACT"))
        vector::populateVectorContractLoweringPatterns(
            patterns, options.vectorContractLowering);
      vector::populateVectorMultiReductionReorderPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);
      vector::populateVectorMultiReductionFlatteningPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);
      vector::populateVectorMultiReductionUnrollingPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);
      vector::populateVectorTransposeLoweringPatterns(
          patterns, options.vectorTransposeLowering);
      vector::populateVectorGatherToConditionalLoadPatterns(patterns);
      vector::populateVectorGatherLoweringPatterns(patterns);
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      // Lower vector.step to a constant/arith sequence. The SPIR-V conversion
      // has no pattern for vector.step (unlike the NVVM/ROCDL/CPU paths, which
      // call this in their ConvertToLLVM passes), so without this any iota /
      // jnp.arange that survives into a vectorized kernel fails to legalize.
      vector::populateVectorStepLoweringPatterns(patterns);
      vector::CreateMaskOp::getCanonicalizationPatterns(patterns, context);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after lowering various vector ops");

    // Run all sorts of canonicalization patterns to clean up again.
    {
      RewritePatternSet patterns(context);
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::InsertOp::getCanonicalizationPatterns(patterns, context);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
      populateVectorTransferTensorSliceTransforms(patterns);
      vector::ReductionOp::getCanonicalizationPatterns(patterns, context);
      scf::IfOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
