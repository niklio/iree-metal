// Copyright 2022 The IREE Authors
#include <cstdlib>
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileAndPromote.cpp --------------------------------------------===//
//
// This pass tiles promote Linalg ops with buffer semantics to use workgroup
// memory and then tiles to invocations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-tile-and-promote"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVTILEANDPROMOTEPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

//====---------------------------------------------------------------------===//
// Reduction tiling patterns
//====---------------------------------------------------------------------===//

static LogicalResult
tileReductionLoops(mlir::FunctionOpInterface funcOp,
                   LinalgTransformationFilter filter,
                   const scf::SCFTileSizeComputationFunction &computeFn) {
  auto options =
      scf::SCFTilingOptions().setTileSizeComputationFunction(computeFn);
  return tileLinalgOpsWithFilter(funcOp, options, filter);
}

//===----------------------------------------------------------------------===//
// Invocation tiling patterns
//===----------------------------------------------------------------------===//

static LogicalResult
tileToInvocation(mlir::FunctionOpInterface funcOp,
                 LinalgTransformationFilter filter,
                 const linalg::TileSizeComputationFunction &computeFn) {
  auto getThreadProcInfoFn = [](OpBuilder &builder, Location loc,
                                ArrayRef<Range> parallelLoopRanges) {
    return getGPUProcessorIdsAndCounts<gpu::ThreadIdOp, gpu::BlockDimOp>(
        builder, loc, parallelLoopRanges.size());
  };
  linalg::LinalgLoopDistributionOptions distributionOptions;
  distributionOptions.procInfo = getThreadProcInfoFn;

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(computeFn)
                           .setDistributionOptions(distributionOptions);

  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Promotion patterns
//===----------------------------------------------------------------------===//

static const char promoteBothMarker[] = "promote_lhs_and_rhs";

// iree-metal: targeted A/B threadgroup-staging for the coop path. Compute the bytes a matmul's A+B tiles
// would occupy in threadgroup memory (operands are already workgroup+reduction tiled at this point).
// Small tiles (all real matmul shapes incl. large-K backward weight-grads) get staged -> K-loop reuse
// -> ~2x on large-K; big/fused tiles that would over-allocate past Metal's 32KB cap stay device-loaded
// (the HAL-wedge fix). Returns -1 if shapes are dynamic (be conservative -> skip staging).
static int64_t coopPromoteABBytes(linalg::LinalgOp op) {
  auto aTy = dyn_cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
  auto bTy = dyn_cast<ShapedType>(op.getDpsInputOperand(1)->get().getType());
  if (!aTy || !bTy || !aTy.hasStaticShape() || !bTy.hasStaticShape())
    return -1;
  int64_t aElems = 1, bElems = 1;
  for (int64_t d : aTy.getShape()) aElems *= d;
  for (int64_t d : bTy.getShape()) bElems *= d;
  return (aElems * aTy.getElementTypeBitWidth() +
          bElems * bTy.getElementTypeBitWidth()) / 8;
}
// A+B staging budget (bytes). Metal threadgroup cap is 32KB; leave headroom for the f32 C accumulator
// staging (promoteCMatrix) + margin. Known-good staged FFN A/B ~5KB; the wedging fused shapes were ~50KB.
// iree-metal (task#26, 2026-07-23): env-tunable A/B stage cap. Raising it lets BIGGER A/B tiles stay
// STAGED in threadgroup memory (more MAC:Load reuse) instead of device-loaded — probing the FFN's
// 1:1-reuse residual (3.26 vs jax-metal 3.78). Metal cap is 32768; leave headroom for C+epilogue.
static int64_t coopStageABByteCap() {
  static const int64_t cap = []() -> int64_t {
    if (const char *e = ::getenv("IREE_METAL_COOP_STAGE_CAP"))
      return (int64_t)atoi(e);
    return 20480;
  }();
  return cap;
}

static void populatePromotionPatterns(RewritePatternSet &patterns,
                                      StringAttr replaceMarker) {
  MLIRContext *context = patterns.getContext();
  auto baseOptions =
      linalg::LinalgPromotionOptions()
          .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                        deallocateWorkgroupMemory)
          .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory)
          .setUseFullTileBuffers({false, false});
  auto promoteBothOptions = baseOptions.setOperandsToPromote({0, 1});

  LinalgTransformationFilter promoteBothFilter(
      {StringAttr::get(context, promoteBothMarker)}, replaceMarker);

  patterns.insert<LinalgPromotionPattern<linalg::MatmulOp>,
                  LinalgPromotionPattern<linalg::BatchMatmulOp>,
                  LinalgPromotionPattern<linalg::GenericOp>>(
      context, promoteBothOptions, promoteBothFilter);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {

class SPIRVTileAndPromotePass final
    : public impl::SPIRVTileAndPromotePassBase<SPIRVTileAndPromotePass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override;

private:
  /// Promotes C matrix to shared memory when necessary and returns success if
  /// no error happens.
  LogicalResult doPromoteCMatrix(mlir::FunctionOpInterface funcOp) const;
};

} // namespace

void SPIRVTileAndPromotePass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  auto threadTileComputeFn = getSPIRVTileSizeComputeFn(funcOp, 1);
  if (failed(threadTileComputeFn)) {
    return signalPassFailure();
  }
  auto reductionTileComputeFn = getSPIRVScfTileSizeComputeFn(funcOp, 2);
  if (failed(reductionTileComputeFn)) {
    return signalPassFailure();
  }

  // Promote C matrix and propagate the potential fill producer into the
  // allocation. This needs to be done before reduction tiling.
  if (failed(doPromoteCMatrix(funcOp))) {
    return signalPassFailure();
  }

  StringLiteral markerAttrName = LinalgTransforms::kLinalgTransformMarker;
  auto workgroupMarker = StringAttr::get(context, getWorkgroupMemoryMarker());
  auto kTiledMarker = StringAttr::get(context, getWorkgroupKTiledMarker());

  { // Tile reduction dimensions.
    RewritePatternSet patterns(context);
    LinalgTransformationFilter filter(
        // Going through C matrix promotion we will have the marker..
        {workgroupMarker}, kTiledMarker);
    // Not going through C matrix promotion we will have no marker..
    filter.setMatchByDefault();
    if (failed(tileReductionLoops(funcOp, filter, *reductionTileComputeFn))) {
      funcOp.emitOpError() << "failed tiling reduction";
      return signalPassFailure();
    }
  }
  {
    RewritePatternSet patterns(context);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After tiling reduction dimensions ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
      getWorkgroupSize(funcOp);
  if (!maybeWorkgroupSize) {
    funcOp.emitOpError(
        "failed to get workgroup size for tile and promote pass");
    return signalPassFailure();
  }

  SmallVector<int64_t> &workgroupSize = maybeWorkgroupSize.value();
  int64_t totalThreads = workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  std::optional<int> subgroupSize = getGPUSubgroupSize(funcOp);
  if (!subgroupSize) {
    funcOp.emitError("failed to query subgroup size");
    return signalPassFailure();
  }

  // Only promote to workgroup size if there are multiple warps.
  // iree-metal: for the cooperative-matrix path (skipOperandPromotion), A/B staging is TARGETED — stage a
  // matmul's A/B in threadgroup memory only when the tiles fit under coopStageABByteCap() (restores the
  // K-loop reuse that device-only simdgroup_load loses -> ~2x on large-K backward weight-grads), and
  // device-load the big/fused shapes that would over-allocate past Metal's 32KB cap (the HAL-wedge fix).
  if (totalThreads > *subgroupSize) {
    // Attach markers to contract ops to drive promotion.
    funcOp.walk([&](linalg::LinalgOp op) {
      if (isMatmulOrBatchMatmul(op)) {
        if (skipOperandPromotion) {
          int64_t abBytes = coopPromoteABBytes(op);
          if (getenv("IREE_METAL_STAGE_DEBUG")) {
            auto aTy = dyn_cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
            auto bTy = dyn_cast<ShapedType>(op.getDpsInputOperand(1)->get().getType());
            llvm::errs() << "[stage] abBytes=" << abBytes << " cap=" << coopStageABByteCap()
                         << " -> " << (abBytes < 0 || abBytes > coopStageABByteCap() ? "DEVICE" : "STAGE")
                         << "  A=" << (aTy ? aTy : Type()) << " B=" << (bTy ? bTy : Type()) << "\n";
          }
          if (abBytes < 0 || abBytes > coopStageABByteCap())
            return; // too big / dynamic -> device-load (no wedge)
        }
        auto promoteMarker = StringAttr::get(context, promoteBothMarker);
        op->setAttr(markerAttrName, promoteMarker);
      }
    });

    RewritePatternSet promotionPatterns(context);
    populatePromotionPatterns(promotionPatterns, workgroupMarker);
    if (failed(applyPatternsGreedily(funcOp, std::move(promotionPatterns)))) {
      return signalPassFailure();
    }

    // Insert barriers before and after copies to workgroup memory.
    insertBarriersAroundSharedMemoryCopy(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After inserting barriers ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // If we fail to promote (e.g., for cases where we just have one tile so
    // that there are no subview ops), clear markers to enable following steps.
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      auto marker = linalgOp->getAttrOfType<StringAttr>(markerAttrName);
      if (!marker) {
        return WalkResult::advance();
      }
      if (marker.getValue() == promoteBothMarker) {
        linalgOp->removeAttr(markerAttrName);
      }
      return WalkResult::advance();
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After promotion ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Attach markers to ops without them to drive tiling next.
  funcOp.walk([&](linalg::LinalgOp op) {
    auto marker = op->getAttrOfType<StringAttr>(markerAttrName);
    if (!marker || marker.getValue() != getCopyToWorkgroupMemoryMarker()) {
      op->setAttr(markerAttrName, workgroupMarker);
    }
  });

  if (!skipThreadLevel) { // Tile and distribute to invocations.
    LinalgTransformationFilter filter({workgroupMarker}, std::nullopt);
    if (failed(tileToInvocation(funcOp, filter, *threadTileComputeFn))) {
      funcOp.emitOpError() << "failed tiling and distributing to invocations";
      return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    SmallVector<int64_t> numWorkgroups = getStaticNumWorkgroups(funcOp);
    populateFoldAffineMinInDistributedLoopsPatterns(patterns, numWorkgroups);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      // TODO(#4759): This does not converge after the max number of iterations.
      // It indicates that some pattern upstream is generating ops even when the
      // pattern failed to match. Not related to correctness, but would be good
      // to figure out and fix.
      // return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to invocations ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

LogicalResult SPIRVTileAndPromotePass::doPromoteCMatrix(
    mlir::FunctionOpInterface funcOp) const {
  MLIRContext *context = funcOp.getContext();
  if (!promoteCMatrix) {
    return success();
  }

  // iree-metal (cont85): the single-matmul detection below (finds ONE matmulOutBuf,
  // errors on >2 linalg ops, bf16-only) cannot reason about the coop attention
  // flash (TWO matmuls qk+pv + the softmax C-region, f16). Both qk and pv store an
  // f32 accumulator into f16 buffers -> invalid f16-accumulator coop store unless C
  // is staged in f32 (cont81e-84). Bypass the detection and directly run the C-
  // promotion pattern on ALL contractions (it promotes each matmul's f32 C to a
  // shared-memory buffer + a separate copy-out, which is exactly what both attention
  // matmuls need). Gated to the coop-flash pipeline so normal codegen is unchanged.
  if (getenv("IREE_METAL_COOP_ATTN_COOPSMEM")) {
    RewritePatternSet patterns(context);
    populateContractPromotionPatterns(patterns, {2});
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return failure();
    }
    propagateSharedMemoryCopy(funcOp);
    return success();
  }

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  // Find the contraction (the matmul) so we can tell a PROLOGUE f32->bf16 cast
  // (output feeds the matmul input) from an EPILOGUE one (input is the matmul
  // output = the store-downcast truncf we must NOT skip).
  Value matmulOutBuf;
  for (Operation *op : computeOps) {
    if (auto l = dyn_cast<linalg::LinalgOp>(op))
      if (l.getNumReductionLoops() > 0)
        matmulOutBuf = l.getDpsInitOperand(0)->get();
  }
  auto baseBuffer = [](Value v) -> Value {
    while (auto sv = v.getDefiningOp<memref::SubViewOp>())
      v = sv.getViewSource();
    return v;
  };
  SmallVector<Operation *> linalgOps;
  for (Operation *op : computeOps) {
    if (isa<linalg::FillOp>(op)) {
      continue; // Don't care
    }
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Bufferization may express a constant accumulator initialization as a
      // zero-input linalg.generic instead of linalg.fill. It is still only a
      // fill and must not be counted as a third matmul/epilogue operation.
      if (linalgOp.getNumDpsInputs() == 0 &&
          linalgOp.getNumReductionLoops() == 0) {
        continue;
      }
      // iree-metal: skip the f32->bf16 input-trunc PROLOGUE producers from the IREE_METAL_COOP_BF16CAST
      // downcast (elementwise, single input, f32->bf16). On memref semantics they feed the matmul via
      // memory (not SSA), so identify them by signature. They fuse into the matmul tiles; doPromoteCMatrix
      // only reasons about the contraction + its epilogue.
      // BUT do NOT skip the EPILOGUE store-downcast truncf (same f32->bf16 signature but its INPUT is the
      // matmul OUTPUT): skipping it left linalgOps=[matmul] -> no C-promotion -> an invalid bf16 coop
      // store of the f32 accumulator -> GARBAGE (the GQA k-proj N=128 NaN). Keep it so C gets promoted.
      if (linalgOp.getNumReductionLoops() == 0 && linalgOp.getNumDpsInputs() == 1 &&
          cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType()).getElementType().isF32() &&
          cast<ShapedType>(linalgOp.getDpsInitOperand(0)->get().getType()).getElementType().isBF16()) {
        bool isEpilogue =
            matmulOutBuf &&
            baseBuffer(linalgOp.getDpsInputOperand(0)->get()) ==
                baseBuffer(matmulOutBuf);
        // Only force-promote (keep the epilogue) for SMALL-N matmuls — the GQA KV
        // projections (N = KV*head_dim, e.g. 128) that hit the invalid bf16 coop
        // store. Large-N matmuls (FFN/projections, N>=512) forward-fuse their
        // truncf and don't need promotion; force-promoting them just adds staging
        // (measured: ~10% training regression). Threshold 256 splits GQA KV heads
        // from real projections. Opt out with IREE_METAL_COOP_NO_PROMOTE_DOWNCAST.
        int64_t outN = 0;
        if (auto mt = dyn_cast<MemRefType>(linalgOp.getDpsInitOperand(0)
                                               ->get()
                                               .getType()))
          if (mt.getRank() >= 1)
            outN = mt.getShape().back();
        if (!isEpilogue || outN <= 0 || outN > 256)
          continue;
      }
      linalgOps.push_back(linalgOp);
    } else {
      return funcOp.emitError("unknown compute op ") << *op;
    }
  }

  if (linalgOps.size() > 2) {
    return funcOp.emitError("unhandled multiple matmul/generic cases");
  }

  // If there are no fused elementwise ops, we can avoid promoting C matrix.
  if (linalgOps.size() <= 1) {
    return success();
  }

  auto matmulOp = cast<linalg::LinalgOp>(linalgOps.front());
  auto genericOp = cast<linalg::GenericOp>(*linalgOps.back());

  auto matmulType =
      cast<MemRefType>(matmulOp.getDpsInitOperand(0)->get().getType());
  if (hasSharedMemoryAddressSpace(matmulType)) {
    // The matmul output is already in shared memory. This can happen when
    // bufferization decides an allocation is needed, e.g., matmul + arith.extf,
    // where the output have different element types. For such cases, don't need
    // to promote and propagate shared memory copy anymore. Just mark the
    // following generic op for distribution accordingly.
    setMarker(genericOp, getCopyToWorkgroupMemoryMarker());
    return success();
  }

  // iree-metal: a NARROWING store-downcast (f32->bf16 truncf) fused into the coop
  // matmul MUST force C-promotion. Apple's coop store is f32-accumulate only and
  // Metal's simdgroup_store requires matching matrix/pointer element types with NO
  // element-conversion primitive — so an un-promoted (coop-fused) narrowing store
  // emits a bf16 coop store of the f32 accumulator matrix, which mis-stores f32
  // data into the bf16 buffer => GARBAGE (root cause of the GQA k-proj N=128 NaN;
  // proven un-fixable at the spirv_cross/Metal level). Promoting C stages the f32
  // accumulator to shared memory + does the truncf as a separate copy-out (the
  // path the working q-proj already takes). Opt out with IREE_METAL_COOP_NO_PROMOTE_DOWNCAST.
  bool isNarrowingCast = false;
  if (!getenv("IREE_METAL_COOP_NO_PROMOTE_DOWNCAST") &&
      genericOp.getNumDpsInputs() == 1 && genericOp.getNumDpsInits() == 1) {
    auto inTy = dyn_cast<ShapedType>(genericOp.getDpsInputs()[0].getType());
    auto outTy =
        dyn_cast<ShapedType>(genericOp.getDpsInitOperand(0)->get().getType());
    if (inTy && outTy && inTy.getElementType().isIntOrFloat() &&
        outTy.getElementType().isIntOrFloat() &&
        inTy.getElementType().getIntOrFloatBitWidth() >
            outTy.getElementType().getIntOrFloatBitWidth())
      isNarrowingCast = true;
  }

  // If the fused elementwise ops are allowed to use cooperative types, we can
  // also avoid promoting C matrix.
  if (!isNarrowingCast && isCooperativeMatrixFusable(genericOp)) {
    return success();
  }

  // Finally do promote C matrix.
  RewritePatternSet patterns(context);
  populateContractPromotionPatterns(patterns, {2});
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "--- After promoting C matrix ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  propagateSharedMemoryCopy(funcOp);
  LLVM_DEBUG({
    llvm::dbgs() << "--- After propagating shared memory copy ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
  return success();
}

} // namespace mlir::iree_compiler
