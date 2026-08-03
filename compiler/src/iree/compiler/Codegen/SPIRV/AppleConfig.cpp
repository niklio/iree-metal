// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AppleConfig.h - Apple CodeGen Configurations -----------------------===//
//
// This file contains CodeGen configurations for Apple GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cstdlib>

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler::detail {

// iree-metal: emit an in-dispatch f32->bf16 truncate of `src` (a ranked f32 tensor) as a linalg.generic.
// Since config assignment runs AFTER dispatch-region formation, this op lands INSIDE the matmul's
// dispatch and the SPIRV tiler fuses the elementwise producer into the matmul's tiled loops -> the
// downcast happens IN-REGISTER (no separate cast dispatch). Lets an f32-input matmul use the bf16
// matrix units (2.8 TFLOP/s) directly, absorbing the cast the custom kernel does internally.
static Value truncTensorToBF16(OpBuilder &b, Location loc, Value src) {
  auto ty = cast<RankedTensorType>(src.getType());
  Type bf16 = b.getBF16Type();
  Value empty = b.create<tensor::EmptyOp>(loc, ty.getShape(), bf16);
  auto idMap = b.getMultiDimIdentityMap(ty.getRank());
  SmallVector<AffineMap> maps = {idMap, idMap};
  SmallVector<utils::IteratorType> iters(ty.getRank(), utils::IteratorType::parallel);
  auto generic = b.create<linalg::GenericOp>(
      loc, empty.getType(), ValueRange{src}, ValueRange{empty}, maps, iters,
      [&](OpBuilder &nb, Location nloc, ValueRange args) {
        Value t = nb.create<arith::TruncFOp>(nloc, bf16, args[0]);
        nb.create<linalg::YieldOp>(nloc, t);
      });
  return generic.getResult(0);
}

static LogicalResult setAppleMatmulConfig(linalg::LinalgOp op,
                                          IREE::GPU::TargetAttr target) {
  // iree-metal EXPERIMENT (IREE_METAL_COOP_BF16CAST): if the matmul has f32 inputs, truncate them to bf16
  // in-dispatch and REBUILD it as a bf16 matmul so it uses the bf16 matrix units (faster than f32 coop)
  // with no separate cast dispatch. Must build a NEW named op (its region block args are typed to the
  // operands) — can't just setOperand on the f32 matmul.
  if (getenv("IREE_METAL_COOP_BF16CAST")) {
    // IREE generalizes named matmuls to linalg.generic before config, so handle the generic
    // contraction: rebuild with bf16 inputs + a mixed-precision body (extf bf16->f32, mulf, addf)
    // keeping the f32 accumulator. The in-dispatch truncs fuse into the matmul tiles (in-register).
    auto gen = dyn_cast<linalg::GenericOp>(op.getOperation());
    auto aTy = gen && gen.getNumDpsInputs() == 2
                   ? dyn_cast<RankedTensorType>(op.getDpsInputOperand(0)->get().getType())
                   : nullptr;
    auto bTy = gen ? dyn_cast<RankedTensorType>(op.getDpsInputOperand(1)->get().getType()) : nullptr;
    if (aTy && bTy && aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
        cast<ShapedType>(op.getDpsInitOperand(0)->get().getType()).getElementType().isF32()) {
      OpBuilder b(op);
      Value a2 = truncTensorToBF16(b, op.getLoc(), op.getDpsInputOperand(0)->get());
      Value b2 = truncTensorToBF16(b, op.getLoc(), op.getDpsInputOperand(1)->get());
      Value init = op.getDpsInitOperand(0)->get();
      Type f32 = b.getF32Type();
      auto newGen = b.create<linalg::GenericOp>(
          op.getLoc(), TypeRange{op->getResult(0).getType()}, ValueRange{a2, b2}, ValueRange{init},
          gen.getIndexingMapsArray(), gen.getIteratorTypesArray(),
          [&](OpBuilder &nb, Location nloc, ValueRange args) {
            Value af = nb.create<arith::ExtFOp>(nloc, f32, args[0]);
            Value bf = nb.create<arith::ExtFOp>(nloc, f32, args[1]);
            Value m = nb.create<arith::MulFOp>(nloc, af, bf);
            Value s = nb.create<arith::AddFOp>(nloc, args[2], m);
            nb.create<linalg::YieldOp>(nloc, s);
          });
      op->getResult(0).replaceAllUsesWith(newGen.getResult(0));
      op->erase();
      op = cast<linalg::LinalgOp>(newGen.getOperation());
    }
  }
  // iree-metal: first try Apple's matrix units via the cooperative-matrix pipeline (SPIR-V coop-matrix →
  // MSL simdgroup_matrix). Falls back to the scalar/vector config below if no MMA schedule fits.
  // The (4,4) copied from NVIDIA is not tuned for Apple's 32-wide simdgroups / 8x8 simdgroup_matrix;
  // sweep the schedule via env (IREE_METAL_COOP_{SG,MNT,PD,SS}) to find Apple-optimal tiling.
  auto envU = [](const char *n, unsigned d) -> unsigned {
    if (const char *v = getenv(n)) return (unsigned)atoi(v);
    return d;
  };
  // iree-metal (cont313): Apple's 32-wide simdgroups favor MORE subgroups + SMALLER MN
  // tiles + software-pipelining for LARGE 2D (non-batch) matmuls (FFN/proj). Sweep-
  // validated bit-identical (config-only) and up to +39% (M=8192 wide-N), +9% wide-N,
  // +2-3% deep-K vs the NVIDIA-copied (4,4,PD0) default. RESTRICT to rank-2 outputs
  // with large M: batch matmuls (attention, output rank>=3) REGRESS with (8,2) — vit
  // 0.89->0.64 — and small M shows no gain. So (8,2,PD1) only for rank-2 M>=2048,
  // N>=512; everything else keeps (4,4,PD0). Opt out: IREE_METAL_COOP_NO_APPLE_TUNE.
  unsigned sgD = 4, mntD = 4, pdD = 0, ktD = 2;
  // DEFAULT-ON (opt-out IREE_METAL_COOP_NO_APPLE_TUNE): Apple's 32-wide simdgroups favor
  // MORE subgroups + SMALLER MN tiles + software-pipelining than the NVIDIA-copied
  // (4,4,PD0) default. Sweep-validated bit-identical (config-only) and up to +39%
  // (M=8192 wide-N), +54% ([4736x768x768]), +9% wide-N. Restrict the tuned
  // schedule to rank-2 large 2D matmuls whose every dimension (M, N, and the
  // contraction K on the inputs) is a multiple of 128. The 8-subgroup schedule
  // forms 128-element macro-tiles; on smaller alignment its 256-thread
  // workgroups can trigger a sustained frequency drop in long Apple workloads.
  // GPT/BERT (M=4096), DeiT (M=4608), and the validated M=4736 shape remain
  // eligible, while ViT's padded M=4672 keeps the lower-power default schedule.
  if (!getenv("IREE_METAL_COOP_NO_APPLE_TUNE")) {
    auto outTy = dyn_cast<ShapedType>(op.getDpsInitOperand(0)->get().getType());
    bool ok = outTy && outTy.getRank() == 2 && outTy.getDimSize(0) >= 2048 &&
              outTy.getDimSize(1) >= 512 && outTy.getDimSize(0) % 128 == 0 &&
              outTy.getDimSize(1) % 128 == 0;
    if (ok) {
      for (OpOperand *in : op.getDpsInputOperands()) {
        auto t = dyn_cast<ShapedType>(in->get().getType());
        if (!t) { ok = false; break; }
        for (int64_t d : t.getShape())
          if (d % 128 != 0) { ok = false; break; }
        if (!ok) break;
      }
    }
    // iree-metal (2026-07-23): MNT=4 (was 2) — direct isolated matmul measurement showed the fork's
    // coop FFN matmul was only 80% of jax-metal (3.02 vs 3.78 TFLOP/s); an (SG=8,MNT=4,KT=4)
    // sweep recovered it to 3.27 and gave +2.0-2.8% end-to-end on aligned models (correct).
    // ViT's padded 4672 dimension is excluded; naturally 128-aligned transformer
    // and DeiT shapes retain the eight-subgroup win.
    // KT=4 (not the default 2) is REQUIRED with MNT=4: MNT=4 alone overflows the 32768-byte
    // threadgroup cap (57344 B on 4096x768x768); KT=4 bounds the staged A/B tiles so it fits and
    // hits 3.27 TFLOP/s (vs 3.02 default). Both together = the measured +2-2.8% end-to-end.
    if (ok) {
      sgD = 8;
      mntD = 4;
      pdD = 1;
      ktD = 4;
    }

    // Narrow 384-wide rank-2 contractions benefit from smaller per-subgroup MN
    // tiles even when the surrounding large-shape tune is active. This covers
    // forward, projection, and weight-gradient forms without changing batched
    // attention. Balanced and reversed 16-step controls measured MNT=1 about
    // 1.0% faster than MNT=2 and about 2.4% faster end-to-end than MNT=4; the
    // resulting configuration change is limited to 13 rank-2 dispatches in the
    // representative training graph.
    bool isNarrow384Matmul = outTy && outTy.getRank() == 2;
    bool has384Dimension = false;
    int64_t largestDimension = 0;
    auto inspectNarrow384Type = [&](ShapedType type) {
      if (!type || type.getRank() != 2) {
        isNarrow384Matmul = false;
        return;
      }
      for (int64_t d : type.getShape()) {
        if (d <= 0) {
          isNarrow384Matmul = false;
          return;
        }
        has384Dimension |= d == 384;
        if (d > largestDimension) largestDimension = d;
      }
    };
    inspectNarrow384Type(outTy);
    for (OpOperand *in : op.getDpsInputOperands()) {
      inspectNarrow384Type(dyn_cast<ShapedType>(in->get().getType()));
    }
    if (isNarrow384Matmul && has384Dimension && largestDimension >= 1536) {
      mntD = 1;
    }
  }
  if (succeeded(setCooperativeMatrixConfig(target, op,
                                           /*numSubgroupsPerWorkgroup=*/envU("IREE_METAL_COOP_SG", sgD),
                                           /*numMNTilesPerSubgroup=*/envU("IREE_METAL_COOP_MNT", mntD),
                                           /*softwarePipelineDepth=*/envU("IREE_METAL_COOP_PD", pdD),
                                           /*softwarePipelineStoreStage=*/envU("IREE_METAL_COOP_SS", 0),
                                           /*numKTiles=*/envU("IREE_METAL_COOP_KT", ktD)))) {
    return success();
  }
  const std::array<int64_t, 2> workgroupXY = {256, 1};
  std::array<int64_t, 3> threadMNK;
  auto inputType = cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
  if (IREE::Util::getTypeBitWidth(inputType.getElementType()) == 16) {
    threadMNK = {4, 8, 8};
  } else {
    threadMNK = {4, 4, 4};
  }
  return setMatmulOpConfig(target, op, workgroupXY, threadMNK);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setAppleCodeGenConfig(IREE::GPU::TargetAttr target,
                                    Operation *rootOp) {
  int subgroupSize = target.getPreferredSubgroupSize();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    if (isMatmulOrBatchMatmul(linalgOp)) {
      return setAppleMatmulConfig(linalgOp, target);
    }
  }

  if (auto convOp = dyn_cast<linalg::ConvolutionOpInterface>(rootOp)) {
    // Use the result type in case of larger bitwidth for accumulators.
    auto type = cast<ShapedType>(convOp->getResult(0).getType());
    const int bitwidth = type.getElementTypeBitWidth();
    if (bitwidth > 32) {
      return failure();
    }
    const int multiplier = 32 / bitwidth;
    const int bestTilingFactor = 16 * multiplier;
    return setConvOpConfig(cast<linalg::LinalgOp>(rootOp), subgroupSize,
                           bestTilingFactor);
  }

  return failure();
}

} // namespace mlir::iree_compiler::detail
