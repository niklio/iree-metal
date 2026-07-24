// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-metal: On the Metal/SPIR-V backend a linalg.generic that reads one of its
// inputs through a NON-identity (permuted/transposed) indexing map lowers to
// strided, uncoalesced loads. When that generic has a large parallel iteration
// space (e.g. the 96x512x512 softmax-backward epilogue) the uncoalesced read
// dominates -- measured ~32ms of a 56ms dispatch, vs a standalone coalesced
// transpose at ~2ms. IREE normally *propagates* transposes INTO such consumers
// (PropagateLinalgTranspose + elementwise fusion + dispatch cloning, via several
// robust paths), which is the wrong tradeoff here.
//
// This pass HOISTS such a permuted read back out: it materializes the transposed
// operand as a standalone linalg.transpose and inserts a util.optimization_barrier
// on it so downstream fusion/propagation cannot re-absorb it, then rewrites the
// consumer to read the (now identity-indexed) materialized tensor. The transpose
// lowers as a fast coalesced dispatch and the consumer's reads become contiguous.
// Validated 2x (56.9ms -> 27.9ms) on the gpt2 softmax-backward repro.

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_ISOLATETRANSPOSEDREADSPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

namespace {

// Only hoist when the consumer iteration space is at least this many elements;
// small permuted reads are cheap and materializing them just adds a dispatch.
static constexpr int64_t kMinIterationElements = 1 << 20; // ~1M

// AND only when the permuted operand itself is at least this large. The
// uncoalesced-read penalty scales with the operand's size; hoisting small
// transposed reads just adds a transpose dispatch + barrier for no benefit
// (measured: firing on every transposed read regressed the full model -4.5%,
// while restricting to large operands keeps only the wins like the 96x512x512
// softmax-backward read).
static constexpr int64_t kMinOperandElements = 1 << 22; // ~4M

struct IsolateTransposedReadsPass
    : iree_compiler::Preprocessing::impl::IsolateTransposedReadsPassBase<
          IsolateTransposedReadsPass> {
  using iree_compiler::Preprocessing::impl::IsolateTransposedReadsPassBase<
      IsolateTransposedReadsPass>::IsolateTransposedReadsPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<linalg::GenericOp> candidates;
    funcOp.walk([&](linalg::GenericOp genericOp) {
      if (!IREE::Flow::isNonNullAndOutsideDispatch(genericOp))
        return;
      // Elementwise-ish: all parallel loops (the uncoalesced-read pathology is
      // for the large parallel epilogue; reductions are handled elsewhere).
      if (genericOp.getNumReductionLoops() != 0)
        return;
      // Large iteration space only.
      int64_t iters = 1;
      for (int64_t r : genericOp.getStaticLoopRanges()) {
        if (ShapedType::isDynamic(r)) {
          iters = -1;
          break;
        }
        iters *= r;
      }
      if (iters < kMinIterationElements)
        return;
      candidates.push_back(genericOp);
    });

    for (linalg::GenericOp genericOp : candidates) {
      isolateOperands(rewriter, genericOp);
    }
  }

  // For each input operand read through a non-identity permutation, materialize a
  // coalesced transpose + barrier and rewrite the operand to an identity read.
  void isolateOperands(IRRewriter &rewriter, linalg::GenericOp genericOp) {
    Location loc = genericOp.getLoc();
    unsigned numLoops = genericOp.getNumLoops();
    AffineMap ident = AffineMap::getMultiDimIdentityMap(numLoops, &getContext());

    for (OpOperand *in : genericOp.getDpsInputOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(in->get().getType());
      if (!tensorType)
        continue;
      AffineMap map = genericOp.getMatchingIndexingMap(in);
      // Only a full-rank permutation that is a genuine transpose (not identity,
      // not a projection/broadcast). Projections (e.g. reductions of rank) are
      // left alone.
      if (map.getNumResults() != numLoops || !map.isPermutation() ||
          map.isIdentity())
        continue;

      // Only worth a standalone transpose dispatch if the operand is large.
      if (!tensorType.hasStaticShape() ||
          tensorType.getNumElements() < kMinOperandElements)
        continue;

      // Build the permutation that reorders the operand's dims into iteration
      // (identity) order. map maps iter dims -> operand result positions; the
      // i-th operand dim is iteration dim map.getDimPosition(i). Transposing the
      // operand by that permutation yields an identity-indexed tensor.
      SmallVector<int64_t> perm;
      perm.reserve(numLoops);
      for (unsigned i = 0; i < numLoops; ++i)
        perm.push_back(map.getDimPosition(i));

      rewriter.setInsertionPoint(genericOp);
      SmallVector<int64_t> resultShape(numLoops);
      ArrayRef<int64_t> srcShape = tensorType.getShape();
      for (unsigned i = 0; i < numLoops; ++i)
        resultShape[perm[i]] = srcShape[i];
      Value empty = rewriter.create<tensor::EmptyOp>(
          loc, resultShape, tensorType.getElementType());
      auto transposed = rewriter.create<linalg::TransposeOp>(
          loc, in->get(), empty, perm);
      Value barrier = rewriter.create<IREE::Util::OptimizationBarrierOp>(
          loc, transposed.getResult()[0]).getResult(0);

      // Rewrite the consumer: this operand now reads identity.
      SmallVector<AffineMap> maps = genericOp.getIndexingMapsArray();
      maps[in->getOperandNumber()] = ident;
      rewriter.modifyOpInPlace(genericOp, [&]() {
        genericOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
        genericOp.setOperand(in->getOperandNumber(), barrier);
      });
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::Preprocessing
