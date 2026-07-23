// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVVECTORTOGPUSUBGROUPMMAPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {
struct SPIRVVectorToGPUSubgroupMMAPass final
    : impl::SPIRVVectorToGPUSubgroupMMAPassBase<
          SPIRVVectorToGPUSubgroupMMAPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    RewritePatternSet flatternpatterns(funcOp.getContext());
    populateVectorTransferToGPUMMAPreparationPatterns(flatternpatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(flatternpatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(funcOp.getContext());
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    populatePrepareVectorToMMAPatterns(patterns, /*useNvGpu=*/false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }

    IRRewriter rewriter(&getContext());
    if (failed(convertVectorToMMAOps(rewriter, funcOp))) {
      // nlearn: convertVectorToMMAOps HARD-FAILING used to kill the WHOLE module
      // compile — but it fails for whole classes of matmuls that simply can't hit
      // the coop path in this context (unaligned/non-mult-16 shapes like vit
      // M=B*T=4616, attention batch-matmuls, some transposed backward forms). It
      // leaves the original vector.contract ops in place (no partial mutation), so
      // instead of failing, fall through: the "no subgroup_mma ops" branch below
      // scalarizes the leftover contracts (unroll -> SPIRVBreakDownLargeVector) so
      // the dispatch degrades GRACEFULLY to scalar. Dispatches that DID convert are
      // unaffected. Opt back into the hard error with NLEARN_COOP_STRICT_MMA.
      if (getenv("NLEARN_COOP_STRICT_MMA")) {
        funcOp->emitError("failed conversion to GPU subgroup MMA ops");
        return signalPassFailure();
      }
    }

    // nlearn: convertVectorToMMAOps can leave the ORIGINAL vector.contract ops
    // behind as DEAD code alongside the new gpu.subgroup_mma_compute ops (observed
    // in the fwd+bwd training graph: a K-loop ends up carrying BOTH a
    // !gpu.mma_matrix accumulator (used by the store) AND a parallel
    // vector<16x16xf32> contract chain whose result is never yielded/used). Those
    // dead coop-native-shaped vectors then fail SPIR-V legalization ("failed to
    // legalize arith.constant vector<16xf32>"). Run canonicalization (incl. scf.for
    // dead-iter-arg elimination) + greedy DCE to strip the dead contract chain so
    // only the real matrix-unit path remains. This is what lets TRAINING compile
    // onto the matrix units instead of dying on the leftover vectors.
    {
      RewritePatternSet cleanup(&getContext());
      scf::ForOp::getCanonicalizationPatterns(cleanup, &getContext());
      vector::ContractionOp::getCanonicalizationPatterns(cleanup, &getContext());
      (void)applyPatternsGreedily(funcOp, std::move(cleanup));
    }

    // nlearn: if the conversion produced NO subgroup mma ops (some matmul shapes,
    // esp. transposed backward forms, don't vectorize to coop), DON'T hard-fail —
    // the leftover unrolled vector.contract ops lower to scalar SPIR-V downstream
    // (addSPIRVVectorLoweringPasses). This makes the coop pipeline degrade
    // GRACEFULLY to scalar for unsupported shapes instead of killing the whole
    // compile — critical for training (backward matmuls) to compile at all.
    WalkResult result = funcOp.walk([](Operation *op) {
      return isa<gpu::SubgroupMmaComputeOp>(op) ? WalkResult::interrupt()
                                                : WalkResult::advance();
    });
    if (!result.wasInterrupted()) {
      if (getenv("NLEARN_COOP_STRICT_MMA")) {
        funcOp->emitError("no GPU subgroup mma compute ops generated");
        return signalPassFailure();
      }
      // Fallback: the contract didn't convert to coop mma. It's still unrolled to
      // the coop-native size (e.g. vector<16xf32>) which scalar SPIR-V can't
      // legalize. Fully unroll leftover vector.contract ops to 1x1x1 so they
      // lower as scalar. Lets the coop pipeline degrade gracefully to scalar for
      // context-broken matmuls (e.g. training backward) instead of failing.
      RewritePatternSet unrollPatterns(&getContext());
      vector::UnrollVectorOptions opts;
      // Unroll the leftover coop-native ops to 1-wide so scalar SPIR-V can
      // legalize them. Cover contracts AND the transfer read/writes (which carry
      // the coop-native vector<16x16>/<16xf32> shapes) — unrolling only the
      // contract leaves 2D transfer vectors that SPIRVBreakDownLargeVector can't
      // legalize. All-1 native shape => fully scalar.
      opts.setNativeShapeFn(
          [](Operation *op) -> std::optional<SmallVector<int64_t>> {
            if (auto c = dyn_cast<vector::ContractionOp>(op))
              return SmallVector<int64_t>(c.getIteratorTypes().size(), 1);
            if (auto r = dyn_cast<vector::TransferReadOp>(op))
              return SmallVector<int64_t>(r.getVectorType().getRank(), 1);
            if (auto w = dyn_cast<vector::TransferWriteOp>(op))
              return SmallVector<int64_t>(w.getVectorType().getRank(), 1);
            return std::nullopt;
          });
      vector::populateVectorUnrollPatterns(unrollPatterns, opts);
      vector::populateCastAwayVectorLeadingOneDimPatterns(unrollPatterns);
      (void)applyPatternsGreedily(funcOp, std::move(unrollPatterns));
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createSPIRVVectorToGPUSubgroupMMAOpsPass() {
  return std::make_unique<SPIRVVectorToGPUSubgroupMMAPass>();
}

} // namespace mlir::iree_compiler
