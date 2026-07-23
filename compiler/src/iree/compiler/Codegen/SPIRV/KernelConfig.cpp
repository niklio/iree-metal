// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-spirv-kernel-config"

using llvm::divideCeil;
using llvm::APIntOps::GreatestCommonDivisor;

// The default number of tiles along K dimension to use per subgroup/workgroup.
constexpr unsigned numKTilesPerSubgroup = 2;

constexpr int kMaxVectorNumBits = 128;

namespace mlir::iree_compiler {

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

// Check if the given linalg op is fused with another op that may result
// in too much shared memory usage.
static bool fusedOpMayUseExtraSharedMemory(linalg::LinalgOp matmul) {
  if (matmul->getNumResults() != 1) {
    return true;
  }

  auto entryPoint = matmul->getParentOfType<mlir::FunctionOpInterface>();

  auto getResultBits = [](linalg::LinalgOp linalgOp) {
    auto shapedType = cast<ShapedType>(linalgOp->getResult(0).getType());
    return IREE::Util::getTypeBitWidth(shapedType.getElementType());
  };
  auto matmulResultBits = getResultBits(matmul);

  bool fusedWithOp = false;
  entryPoint.walk([&](linalg::LinalgOp linalgOp) {
    if (linalgOp == matmul || isMatmulOrBatchMatmul(linalgOp) ||
        isa<linalg::FillOp>(linalgOp)) {
      return WalkResult::advance();
    }

    if (linalgOp->getNumResults() != 1 ||
        getResultBits(linalgOp) != matmulResultBits) {
      fusedWithOp = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return fusedWithOp;
}

//===----------------------------------------------------------------------===//
// Convolution Default Configuration
//===----------------------------------------------------------------------===//

/// Decides the tiling and distribution parameters for one convolution
/// dimension. Returns true if we can successfully deduce.
///
/// - `inputDim` is the size of the dimension to be distributed.
/// - `residualThreads` is the remaining threads we can distribute.
/// - `residualTilingFactor` indicates the remaining tiling scale factor.
/// - `wgDimSize` will be updated with the decided workgroup dimension size.
/// - `wgTileSize` will be updated with the decided workgroup tile size.
static bool tileConvOneDim(const int64_t inputDim, const bool isInnerMostDim,
                           int vectorSize, int64_t &residualThreads,
                           int64_t &residualTilingFactor, int64_t &wgDimSize,
                           int64_t &wgTileSize) {
  const int64_t lb = isInnerMostDim ? 2 : 1;
  for (int64_t dim = residualThreads; dim >= lb; dim >>= 1) {
    int64_t chosenTileSize = 0;
    if (isInnerMostDim) {
      // Handle `vectorSize` elements per thread for the innermost dimension.
      // We need this for the best utilization of memory.
      chosenTileSize = vectorSize;
      if (inputDim % (dim * chosenTileSize) != 0) {
        continue;
      }
    } else {
      for (int64_t t = residualTilingFactor; t >= 1; t >>= 1) {
        if (inputDim % (dim * t) == 0) {
          chosenTileSize = t;
          break;
        }
      }
    }
    if (chosenTileSize) {
      wgDimSize = dim;
      wgTileSize = dim * chosenTileSize;
      residualThreads /= dim;
      residualTilingFactor /= chosenTileSize;
      return true;
    }
  }
  return false;
};

/// Decides the tiling and distribution parameters for two convolution window
/// dimensions to two workgroup dimensions as a square. Returns true if we can
/// successfully deduce.
static bool tileConvSquare(const int64_t oh, const int64_t ow,
                           int64_t &residualThreads,
                           int64_t &residualTilingFactor,
                           MutableArrayRef<int64_t> wgDimSizes,
                           MutableArrayRef<int64_t> wgTileSizes) {
  assert(wgDimSizes.size() == 2 && wgTileSizes.size() == 2);

  const unsigned log2Threads = llvm::Log2_64(residualThreads);
  if (oh == ow && residualThreads != 1 && log2Threads % 2 == 0) {
    const int64_t yz = 1ll << (log2Threads / 2);

    int64_t chosenTileSize = 1ll << (llvm::Log2_64(residualTilingFactor) / 2);
    while (chosenTileSize >= 1 && ow % (yz * chosenTileSize) != 0) {
      chosenTileSize >>= 1;
    }

    if (chosenTileSize != 0) {
      wgDimSizes.front() = wgDimSizes.back() = yz;
      wgTileSizes.front() = wgTileSizes.back() = yz * chosenTileSize;
      return true;
    }
  }
  return false;
}

namespace detail {

LogicalResult setConvOpConfig(linalg::LinalgOp linalgOp,
                              const int64_t subgroupSize,
                              const int64_t bestTilingFactor) {
  assert(isa<linalg::ConvolutionOpInterface>(*linalgOp));
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as convolution...\n");

  Type inputType = linalgOp.getDpsInputOperand(0)->get().getType();
  ArrayRef<int64_t> inputShape = cast<ShapedType>(inputType).getShape();
  Type outputType = linalgOp.getDpsInitOperand(0)->get().getType();
  ArrayRef<int64_t> outputShape = cast<ShapedType>(outputType).getShape();
  // Restrict to pure 4-D input/output shapes for now. This excludes convolution
  // ops with 1- or 3-D window sizes. It also excludes 2-D-window convolution
  // ops like `linalg.depthwise_conv_2d_nhwc_hwcm`.
  if (inputShape.size() != 4 || outputShape.size() != 4) {
    return failure();
  }

  auto convDimsOrFailure = linalg::inferConvolutionDims(linalgOp);
  if (failed(convDimsOrFailure)) {
    return failure();
  }
  const mlir::linalg::ConvolutionDimensions &convDims = *convDimsOrFailure;
  LLVM_DEBUG(llvm::dbgs() << "conv: " << linalgOp << "\n"
                          << "conv batch dim: "
                          << llvm::interleaved(convDims.batch) << "\n"
                          << "conv output window dims: "
                          << llvm::interleaved(convDims.outputImage) << "\n"
                          << "conv output channel dim: "
                          << llvm::interleaved(convDims.outputChannel) << "\n"
                          << "conv filter window dims: "
                          << llvm::interleaved(convDims.filterLoop) << "\n"
                          << "conv input channel dims: "
                          << llvm::interleaved(convDims.inputChannel) << "\n"
                          << "conv depth multiplier: "
                          << llvm::interleaved(convDims.depth) << "\n");
  assert(convDims.outputImage.size() == 2);
  assert(convDims.filterLoop.size() == 2);

  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();

  const int ohIndex = convDims.outputImage.front();
  const int64_t oh = loopRanges[ohIndex];
  const int64_t ow = loopRanges[convDims.outputImage.back()];
  int ocIndex;
  if (!convDims.outputChannel.empty()) {
    assert(convDims.outputChannel.size() == 1);
    ocIndex = convDims.outputChannel.front();
  } else if (!convDims.depth.empty()) {
    // For depthwise convolution ops with multiplier 1, we have the same
    // input/filter/output channel size, which is being categorized as the
    // multiplier.
    assert(convDims.depth.size() == 1);
    ocIndex = convDims.depth.front();
  } else {
    // For pooling ops, the input/output channel size will be categorized
    // as the additional batch dimension.
    assert(convDims.batch.size() == 2);
    ocIndex = convDims.batch.back();
  }
  const int64_t oc = loopRanges[ocIndex];
  // We may not have an input channel dimension in the case of depthwise
  // convolution ops.
  std::optional<int64_t> ic = std::nullopt;
  if (!convDims.inputChannel.empty()) {
    assert(convDims.inputChannel.size() == 1);
    ic = loopRanges[convDims.inputChannel.front()];
  }

  if ((ic && ShapedType::isDynamic(*ic)) ||
      llvm::any_of(outputShape.drop_front(), ShapedType::isDynamic)) {
    return failure();
  }

  const int bitwidth = cast<ShapedType>(outputType).getElementTypeBitWidth();
  const int vectorSize = kMaxVectorNumBits / bitwidth;

  // We use `vectorSize` as the tile size along IC dimension. If smaller than
  // 4, it will be unrolled into size 1.
  if (ic && !(*ic % vectorSize == 0 || *ic < 4)) {
    return failure();
  }

  // The core idea is to distribute the convolution dimensions to the workgroup
  // Z/Y/X dimensions, with each thread in a workgroup handling multiple vector
  // elements. We try to 1) utilize all threads in a subgroup, and 2) handle an
  // optimal tile size along each dimension.

  int64_t residualThreads = subgroupSize;
  int64_t residualTilingFactor = bestTilingFactor;

  SmallVector<int64_t, 3> workgroupSize(3, 1); // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(4, 0);

  const bool isNCHW = ocIndex < ohIndex;
  if (isNCHW) {
    // OW -> x, OH -> y, OC -> z
    if (!tileConvOneDim(ow, /*isInnerMostDim=*/true, vectorSize,
                        residualThreads, residualTilingFactor, workgroupSize[0],
                        workgroupTileSizes[3]) ||
        !tileConvOneDim(oh, /*isInnerMostDim=*/false, vectorSize,
                        residualThreads, residualTilingFactor, workgroupSize[1],
                        workgroupTileSizes[2]) ||
        !tileConvOneDim(oc, /*isInnerMostDim=*/false, vectorSize,
                        residualThreads, residualTilingFactor, workgroupSize[2],
                        workgroupTileSizes[1])) {
      return failure();
    }
  } else {
    // OC -> x
    if (!tileConvOneDim(oc, /*isInnerMostDim=*/true, vectorSize,
                        residualThreads, residualTilingFactor, workgroupSize[0],
                        workgroupTileSizes[3])) {
      return failure();
    }

    // Deduce the configuration for the OW and OH dimension. Try to make them
    // even if possible given we typically have images with the same height
    // and width.
    const bool tileToSquare = tileConvSquare(
        oh, ow, residualThreads, residualTilingFactor,
        llvm::MutableArrayRef(workgroupSize).drop_front(),
        llvm::MutableArrayRef(workgroupTileSizes).drop_front().drop_back());

    // Otherwise treat OW and OH separately to allow them to have different
    // number of threads and tiling size.
    if (!tileToSquare) {
      if (!tileConvOneDim(ow, /*isInnerMostDim=*/false, vectorSize,
                          residualThreads, residualTilingFactor,
                          workgroupSize[1], workgroupTileSizes[2]) ||
          !tileConvOneDim(oh, /*isInnerMostDim=*/false, vectorSize,
                          residualThreads, residualTilingFactor,
                          workgroupSize[2], workgroupTileSizes[1])) {
        return failure();
      }
    }
  }

  SmallVector<int64_t> threadTileSizes(4, 0);
  threadTileSizes[0] = 1; // Tile along the N dimension with size 1
  for (int i = 1; i <= 3; ++i) {
    threadTileSizes[i] = workgroupTileSizes[i] / workgroupSize[3 - i];
  }

  auto pipeline = CodeGenPipeline::SPIRVBaseVectorize;
  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(threadTileSizes);

  // Tiling along reduction dimensions.
  SmallVector<int64_t> reductionTileSizes(loopRanges.size(), 0);
  // For filter window dimensions, use tile size 1 to reduce the contraction
  // problem to be similar to a normal matmul.
  reductionTileSizes[convDims.filterLoop.front()] = 1;
  reductionTileSizes[convDims.filterLoop.back()] = 1;
  if (ic) {
    // Tile input channel dimension with size 4 to avoid code bloat in
    // vectorization later.
    reductionTileSizes[convDims.inputChannel.front()] = vectorSize;
  }
  tileSizes.push_back(reductionTileSizes);

  // Tile along OH by size 1 to enable downsizing 2-D convolution to 1-D.
  SmallVector<int64_t> windowTileSizes(4, 0);
  windowTileSizes[ohIndex] = 1;
  tileSizes.push_back(windowTileSizes);

  auto funcOp = linalgOp->getParentOfType<mlir::FunctionOpInterface>();
  return setOpConfigAndEntryPointFnTranslation(funcOp, linalgOp, tileSizes,
                                               pipeline, workgroupSize);
}

} // namespace detail

//===----------------------------------------------------------------------===//
// Matmul Default Configuration
//===----------------------------------------------------------------------===//

/// Given the linalg `op` with `lhsShape` and `rhsShape`, tries to treat as a
/// (batch) matmul like op and deduce the index of the loop corresponding to
/// B/M/N/K dimension respectively. Returns -1 as the index if unable to deduce.
std::tuple<int, int, int, int> getMatmulBMNKIndex(linalg::LinalgOp op,
                                                  int *lastParallelDim) {
  OpOperand *lhs = op.getDpsInputOperand(0);
  OpOperand *rhs = op.getDpsInputOperand(1);
  auto lhsShape = cast<ShapedType>(lhs->get().getType()).getShape();
  auto rhsShape = cast<ShapedType>(rhs->get().getType()).getShape();

  auto lhsLoopIndices =
      llvm::map_to_vector(llvm::seq<int>(0, lhsShape.size()), [&](int i) {
        return op.getMatchingIndexingMap(lhs).getDimPosition(i);
      });
  auto rhsLoopIndices =
      llvm::map_to_vector(llvm::seq<int>(0, rhsShape.size()), [&](int i) {
        return op.getMatchingIndexingMap(rhs).getDimPosition(i);
      });

  // Figure out what dimension each loop corresponds to.
  int bIndex = -1, mIndex = -1, nIndex = -1, kIndex = -1;
  for (unsigned i = 0; i < op.getNumLoops(); ++i) {
    if (linalg::isReductionIterator(op.getIteratorTypesArray()[i])) {
      kIndex = i;
      continue;
    }

    const bool inLHS = llvm::is_contained(lhsLoopIndices, i);
    const bool inRHS = llvm::is_contained(rhsLoopIndices, i);
    if (inLHS && inRHS) {
      bIndex = i;
    } else if (inLHS) {
      // For cases where we have two parallel dimensions only accessed by
      // the LHS, treat the outer one of them as the batch dimension.
      if (mIndex >= 0 && bIndex < 0) {
        bIndex = mIndex;
      }
      mIndex = i;
    } else if (inRHS) {
      // For cases where we have two parallel dimensions only accessed by
      // the RHS, treat the outer one of them as the batch dimension.
      if (nIndex >= 0 && bIndex < 0) {
        bIndex = nIndex;
      }
      nIndex = i;
    }
    if (lastParallelDim) {
      *lastParallelDim = i;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "(B, M, N, K) indices = (" << bIndex << ", " << mIndex
                 << ", " << nIndex << ", " << kIndex << ")\n";
  });
  return {bIndex, mIndex, nIndex, kIndex};
}

/// Decides the tiling and distribution parameters for matmul's N dimension to
/// workgroup X dimension.
static bool tileMatmulNToWorkgroupX(const int64_t dimN,
                                    const int64_t bestThreadN,
                                    int64_t &residualThreads,
                                    const int64_t bestX,
                                    int64_t &residualTilingFactor,
                                    int64_t &wgDimSize, int64_t &wgTileSize) {
  // Deduce the configuration for the N dimension. Start with the best workgroup
  // X size, and reduce by a factor of two each time.
  for (int64_t x = bestX; x >= 2; x >>= 1) {
    // Handle 4 elements per thread for the innermost dimension. We need this
    // for vectorized load.
    int64_t chosenTileSize = bestThreadN;
    if (dimN % (x * chosenTileSize) == 0) {
      wgDimSize = x;
      wgTileSize = x * chosenTileSize;
      residualThreads /= x;
      assert(residualTilingFactor % chosenTileSize == 0);
      residualTilingFactor /= chosenTileSize;
      return true;
    }
  }
  return false;
}

/// Decides the tiling and distribution parameters for matmul's M dimension to
/// workgroup Y dimension.
static bool tileMatmulMToWorkgroupY(const int64_t dimM,
                                    const int64_t bestThreadM,
                                    int64_t &residualThreads,
                                    const int64_t bestY,
                                    int64_t &residualTilingFactor,
                                    int64_t &wgDimSize, int64_t &wgTileSize) {
  // Deduce the configuration for the M dimension. Start with the best workgroup
  // Y size, and reduce by a factor of two each time.
  for (int64_t y = residualThreads; y >= 1; y >>= 1) {
    int64_t chosenTileSize = 0;
    // Reduce the thread tiling size by one each time. We read one row each
    // time; so it's fine to not be some power of two here.
    for (int64_t t = bestThreadM; t >= 1; --t) {
      if (dimM % (y * t) == 0) {
        chosenTileSize = t;
        break;
      }
    }
    if (chosenTileSize) {
      wgDimSize = y;
      wgTileSize = y * chosenTileSize;
      assert(residualTilingFactor > chosenTileSize);
      residualTilingFactor -= chosenTileSize;
      return true;
    }
  }
  return false;
}

/// Decides the tiling parameters for matmul's K dimension.
static bool tileMatmulK(const int64_t dimK, const int64_t residualTilingFactor,
                        int64_t &tileSize) {
  // Deduce the configuration for the K dimension. We need some power of two
  // here so that we can do vector load.
  for (int64_t t = llvm::bit_floor<uint64_t>(residualTilingFactor); t >= 2;
       t >>= 1) {
    if (dimK % t == 0) {
      tileSize = t;
      return true;
    }
  }
  return false;
}

int64_t getTileBytes(int64_t mTileSize, int64_t nTileSize, int64_t kTileSize,
                     int64_t elementBits, bool promoteC, int64_t cElementBits) {
  // The promoted C matrix is the ACCUMULATOR (e.g. f32 for a bf16/f16 matmul),
  // which is wider than the A/B inputs. Using the input width for C underestimates
  // the threadgroup memory (by 2x for bf16->f32), letting the config emit a kernel
  // that over-allocates shared memory and silently HANGS Metal. Account for C's own
  // width. (iree-metal fix.)
  if (cElementBits == 0)
    cElementBits = elementBits;
  int64_t paddingBits = detail::bankConflictReductionPaddingBits / elementBits;
  int64_t bytes =
      (elementBits / 8) * ((mTileSize + nTileSize) * (kTileSize + paddingBits));
  if (promoteC) {
    int64_t cPaddingBits = detail::bankConflictReductionPaddingBits / cElementBits;
    bytes += (cElementBits / 8) * (mTileSize * (nTileSize + cPaddingBits));
  }
  return bytes;
}

int64_t getMultiBufferMemoryUsage(int64_t singleBufferBytes, unsigned depth,
                                  unsigned storeStage) {
  if (depth == 0) {
    return singleBufferBytes;
  }
  return singleBufferBytes * (storeStage == 1 ? depth : depth + 1);
};

/// Tries to adjust workgroup and tile sizes to enable vector load for both
/// matmul LHS and RHS. Returns false only when it's not beneficial to promote.
static bool adjustToVectorLoad(ArrayRef<int64_t> dimMNKSize, int64_t &mTileSize,
                               int64_t &nTileSize, int64_t &kTileSize,
                               SmallVectorImpl<int64_t> &wgSize,
                               const int64_t subgroupSize, int64_t vectorSize) {
  const int64_t totalThreads = wgSize[0] * wgSize[1] * wgSize[2];
  LLVM_DEBUG(llvm::dbgs() << "initial total thread = " << totalThreads << "\n");
  if (totalThreads <= subgroupSize) {
    return false;
  }

  const bool canVectorLoadLHS = canPerformVectorAccessUsingAllThreads(
      {mTileSize, kTileSize}, totalThreads, vectorSize);
  const bool canVectorLoadRHS = canPerformVectorAccessUsingAllThreads(
      {kTileSize, nTileSize}, totalThreads, vectorSize);
  LLVM_DEBUG(llvm::dbgs() << "LHS vector load: " << canVectorLoadLHS << "\n");
  LLVM_DEBUG(llvm::dbgs() << "RHS vector load: " << canVectorLoadRHS << "\n");

  // If we can perform vector load of neither, just don't use shared memory.
  if (!canVectorLoadLHS && !canVectorLoadRHS) {
    return false;
  }

  // If we can only perform vector load of one operands, adjust the tiling
  // scheme to see if we can make both work. Increase K to load more data for
  // the smaller tile; decrease M or N, for the larger tile.
  if (canVectorLoadLHS && !canVectorLoadRHS) {
    for (const int scale : {2, 4}) {
      const int64_t newKTileSize = kTileSize * scale;
      if (dimMNKSize[2] % newKTileSize != 0) {
        continue;
      }
      const int64_t newMTileSize = mTileSize / scale;
      const int64_t newWgMDim = wgSize[1] / scale;
      if (newMTileSize == 0 || newWgMDim == 0) {
        continue;
      }
      const int64_t newCount = wgSize[0] * newWgMDim * wgSize[2];
      if (newCount <= subgroupSize) {
        continue;
      }
      if (!canPerformVectorAccessUsingAllThreads({newMTileSize, newKTileSize},
                                                 newCount, vectorSize) ||
          !canPerformVectorAccessUsingAllThreads({newKTileSize, nTileSize},
                                                 newCount, vectorSize)) {
        continue;
      }
      LLVM_DEBUG({
        llvm::dbgs() << "initial [M, N, K] tile size = [" << mTileSize << ", "
                     << nTileSize << ", " << kTileSize << "]\n";
        llvm::dbgs() << "revised [M, N, K] tile size = [" << newMTileSize
                     << ", " << nTileSize << ", " << newKTileSize << "]\n";
      });
      mTileSize = newMTileSize;
      kTileSize = newKTileSize;
      wgSize[1] = newWgMDim;
      break;
    }
  }
  // TODO: improve (!canVectorLoadLHS && canVectorLoadRHS)

  return true;
}

/// Tries to adjust workgorup and tile sizes to promote matmul LHS and RHS and
/// returns true if it's beneficial to promote.
static bool adjustToPromote(ArrayRef<int64_t> dimMNKSize, int64_t &mTileSize,
                            int64_t &nTileSize, int64_t &kTileSize,
                            SmallVectorImpl<int64_t> &wgSize,
                            unsigned &pipelineDepth, unsigned &storeStage,
                            const int subgroupSize, const int maxBytes,
                            const int elementBits) {
  LLVM_DEBUG(llvm::dbgs() << "subgroup size = " << subgroupSize << "\n");
  const int vectorSize = kMaxVectorNumBits / elementBits;
  if (!adjustToVectorLoad(dimMNKSize, mTileSize, nTileSize, kTileSize, wgSize,
                          subgroupSize, vectorSize)) {
    return false;
  }

  // Don't do multibuffering if the inner reduction loop is folded out.
  if (dimMNKSize[2] == kTileSize) {
    pipelineDepth = 1;
    storeStage = 1;
  }

  auto usedBytes =
      getTileBytes(mTileSize, nTileSize, kTileSize, elementBits, false);

  LLVM_DEBUG(llvm::dbgs() << "initial multibuffering bytes = "
                          << getMultiBufferMemoryUsage(usedBytes, pipelineDepth,
                                                       storeStage)
                          << "\n");

  // First try to fit the given tile sizes with the largest pipelining depth
  // possible.
  do {
    if (getMultiBufferMemoryUsage(usedBytes, pipelineDepth, storeStage) <=
        maxBytes) {
      return true;
    }
  } while (pipelineDepth-- > 1);

  // If we can't fit in workgroup memory, don't multibuffer.
  pipelineDepth = 1;

  if (storeStage == 0) {
    storeStage = 1;
    if (getMultiBufferMemoryUsage(usedBytes, pipelineDepth, storeStage) <=
        maxBytes) {
      return true;
    }
  }

  // Using too much workgroup memory. Try to reduce the tile size for X/Y once
  // by a factor of two.
  int64_t &wgDimSize = wgSize[0] > wgSize[1] ? wgSize[0] : wgSize[1];
  int64_t &tileSize = wgSize[0] > wgSize[1] ? nTileSize : mTileSize;
  assert(wgDimSize % 2 == 0);
  wgDimSize /= 2;
  tileSize /= 2;

  int64_t totalThreads = wgSize[0] * wgSize[1] * wgSize[2];
  LLVM_DEBUG(llvm::dbgs() << "revised total thread = " << totalThreads << "\n");
  usedBytes = getTileBytes(mTileSize, nTileSize, kTileSize, elementBits, false);
  LLVM_DEBUG(llvm::dbgs() << "revised tile bytes = " << usedBytes << "\n");
  return totalThreads > subgroupSize && usedBytes <= maxBytes;
}

namespace detail {

LogicalResult setMatmulOpConfig(IREE::GPU::TargetAttr target,
                                linalg::LinalgOp op,
                                std::array<int64_t, 2> bestWorkgroupSizeXY,
                                std::array<int64_t, 3> bestThreadTileSizeMNK,
                                bool enablePromotion,
                                unsigned softwarePipelineDepth,
                                unsigned softwarePipelineStoreStage) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as matmul...\n");
  OpOperand *lhs = op.getDpsInputOperand(0);
  OpOperand *rhs = op.getDpsInputOperand(1);

  auto lhsType = cast<ShapedType>(lhs->get().getType());
  auto rhsType = cast<ShapedType>(rhs->get().getType());
  auto elementBits =
      static_cast<int>(IREE::Util::getTypeBitWidth(lhsType.getElementType()));
  if (!llvm::is_contained({8, 16, 32}, elementBits)) {
    return failure();
  }

  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  if (llvm::any_of(lhsShape, ShapedType::isDynamic)) {
    return failure();
  }
  if (llvm::any_of(rhsShape, ShapedType::isDynamic)) {
    return failure();
  }

  assert(llvm::is_contained({2u, 3u}, op.getNumParallelLoops()));

  int lastParallelDim = -1;
  const auto [bIndex, mIndex, nIndex, kIndex] =
      getMatmulBMNKIndex(op, &lastParallelDim);
  if (mIndex < 0 || nIndex < 0 || kIndex < 0) {
    return failure();
  }
  const bool isBM = bIndex >= 0;

  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  const unsigned numLoops = loopRanges.size();

  const int64_t dimM = loopRanges[mIndex];
  const int64_t dimK = loopRanges[kIndex];
  const int64_t dimN = loopRanges[nIndex];

  // The core idea is to distribute the matmul M/N dimension to the workgroup
  // Y/X dimension, with each thread in a workgroup handling multiple vector
  // elements. We start from the best (X, Y) and the tiling sizes for (M, N, K)
  // and try different configurations by scaling them down until we find a
  // configuration that can perfectly tile the input matmul.

  const int64_t bestThreadM = bestThreadTileSizeMNK[0],
                bestThreadN = bestThreadTileSizeMNK[1],
                bestThreadK = bestThreadTileSizeMNK[2];

  int64_t bestX = bestWorkgroupSizeXY[0], bestY = bestWorkgroupSizeXY[1];
  // We will deduce a configuration first for x and then y. But look at y here
  // to see if the problem size is too small; for such cases, "shift" the
  // parallelism to x.
  if (dimM < bestThreadM) {
    int64_t factor = llvm::PowerOf2Ceil(divideCeil(bestThreadM, dimM));
    bestX *= factor;
    bestY = divideCeil(bestY, factor);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "best thread tile size (M, N, K) = (" << bestThreadM << ", "
                 << bestThreadN << ", " << bestThreadK << ")\n";
    llvm::dbgs() << "best workgroup size (X, Y) = (" << bestX << ", " << bestY
                 << ")\n";
  });

  int64_t residualThreads = bestX * bestY;
  int64_t residualTilingFactor = (bestThreadM + bestThreadK) * bestThreadN;

  SmallVector<int64_t, 3> workgroupSize(3, 1); // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(numLoops, 0);
  SmallVector<int64_t> reductionTileSizes(numLoops, 0);

  if (isBM) {
    workgroupTileSizes[bIndex] = 1;
  }

  if (!tileMatmulNToWorkgroupX(dimN, bestThreadN, residualThreads, bestX,
                               residualTilingFactor, workgroupSize[0],
                               workgroupTileSizes[nIndex]) ||
      !tileMatmulMToWorkgroupY(dimM, bestThreadM, residualThreads, bestY,
                               residualTilingFactor, workgroupSize[1],
                               workgroupTileSizes[mIndex]) ||
      !tileMatmulK(dimK, residualTilingFactor, reductionTileSizes[kIndex])) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "workgroup tile size before promotion = "
                          << llvm::interleaved_array(workgroupTileSizes)
                          << "\nreduction tile size before promotion = "
                          << llvm::interleaved_array(reductionTileSizes)
                          << "\nworkgroup size before promotion = "
                          << llvm::interleaved_array(workgroupSize) << "\n");

  int subgroupSize = target.getPreferredSubgroupSize();
  const int maxBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  // We want a 2-stage pipeline without multi-buffering if the depth is 0 to
  // keep the default for compilation configs that don't specify a pipeline
  // depth.
  auto pipelineDepth = softwarePipelineDepth ? softwarePipelineDepth : 1;
  auto storeStage = softwarePipelineStoreStage;

  // TODO: Remove this check once either bufferization doesn't produce an extra
  // buffer when fused with something like elementwise extf, or the shared
  // memory calculation incorporates the fused op properly.
  if ((pipelineDepth != 1 || storeStage != 1) &&
      fusedOpMayUseExtraSharedMemory(op)) {
    pipelineDepth = 1;
    storeStage = 1;
  }

  // TODO: Enable multibuffering with leading elementwise.
  if (hasFusedLeadingOp(op)) {
    pipelineDepth = 0;
    storeStage = 1;
  }

  // Try to adjust tiling sizes to fit in shared memory.
  auto usePromotionPipeline =
      enablePromotion &&
      adjustToPromote({dimM, dimN, dimK}, workgroupTileSizes[mIndex],
                      workgroupTileSizes[nIndex], reductionTileSizes[kIndex],
                      workgroupSize, pipelineDepth, storeStage, subgroupSize,
                      maxBytes, elementBits);

  // Tile all additional reduction dimensions with size 1 to materialize loops.
  for (auto [i, it] : llvm::enumerate(op.getIteratorTypesArray())) {
    if (linalg::isReductionIterator(it) && reductionTileSizes[i] == 0) {
      reductionTileSizes[i] = 1;
    }
  }

  TileSizesListType tileSizes;

  // Only the promotion pipeline has multibuffering + pipelining.
  if (usePromotionPipeline) {
    // Merge reductionTileSizes into workgroupTileSizes--this is needed by the
    // pipeline passes shared between SPIR-V and LLVMGPU.
    for (auto [i, it] : llvm::enumerate(op.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(it)) {
        workgroupTileSizes[i] = reductionTileSizes[i];
      }
    }
    tileSizes.push_back(workgroupTileSizes);

    return setOpConfigAndEntryPointFnTranslation(
        op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
        CodeGenPipeline::SPIRVMatmulPromoteVectorize, workgroupSize,
        /*subgroupSize=*/std::nullopt,
        getSoftwarePipeliningAttrDict(op->getContext(), pipelineDepth,
                                      storeStage));
  }

  SmallVector<int64_t> threadTileSizes(numLoops, 0);
  if (isBM) {
    threadTileSizes[bIndex] = workgroupTileSizes[bIndex] / workgroupSize[2];
  }
  threadTileSizes[mIndex] = workgroupTileSizes[mIndex] / workgroupSize[1];
  threadTileSizes[nIndex] = workgroupTileSizes[nIndex] / workgroupSize[0];

  workgroupTileSizes.resize(lastParallelDim + 1);
  threadTileSizes.resize(lastParallelDim + 1);
  llvm::append_values(tileSizes, workgroupTileSizes, threadTileSizes,
                      reductionTileSizes);
  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
      CodeGenPipeline::SPIRVBaseVectorize, workgroupSize);
}

static LogicalResult setTilingAndMatmulOpConfig(linalg::LinalgOp op,
                                                IREE::GPU::TargetAttr target) {
  if (!isMatmulOrBatchMatmul(op)) {
    return failure();
  }
  // Try to tile and vectorize first. It's common to see 32 threads
  // per subgroup for GPUs.
  std::array<int64_t, 2> workgroupXY = {32, 2};
  std::array<int64_t, 3> threadMNK;
  auto inputType = cast<ShapedType>(op->getOperand(0).getType());
  if (IREE::Util::getTypeBitWidth(inputType.getElementType()) == 16) {
    threadMNK = {8, 8, 8};
  } else {
    threadMNK = {8, 8, 4};
  }
  return detail::setMatmulOpConfig(target, op, workgroupXY, threadMNK);
}

} // namespace detail

//===----------------------------------------------------------------------===//
// Cooperative Matrix Default Configuration
//===----------------------------------------------------------------------===//

bool isCooperativeMatrixFusable(linalg::GenericOp genericOp) {
  if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
    return false;
  }

  // Look at fused elementwise ops to make sure they are allowed by the
  // cooperative matrix spec.
  for (Operation &op : genericOp.getBlock()->without_terminator()) {
    if (!isa<
            // These ops are directly allowed to use cooperative matrix types.
            arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
            arith::DivFOp, arith::DivSIOp, arith::DivUIOp, arith::NegFOp,
            arith::TruncFOp, arith::TruncIOp, arith::ExtFOp, arith::ExtSIOp,
            arith::ExtUIOp, arith::FPToSIOp, arith::FPToUIOp, arith::SIToFPOp,
            arith::UIToFPOp,
            // Special cases of these ops are directly allowed to sue
            // cooperative matrix types. Other cases can use a loop.
            arith::MulFOp>(op)) {
      return false;
    }
  }

  // Look at operands to make sure we don't have inlined constants. Cooperative
  // matrix loads can only happen from StorageBuffer or Workgroup storage
  // classes.
  for (Value input : genericOp.getInputs()) {
    if (isa<TensorType>(input.getType())) {
      if (matchPattern(input, m_Constant())) {
        return false;
      }
      continue;
    }

    // For buffers we need to walk back the subview chain to see if it's
    // originally from a constant.
    while (auto subviewOp = input.getDefiningOp<memref::SubViewOp>()) {
      input = subviewOp.getViewSource();
    }
    if (auto toMemrefOp = input.getDefiningOp<bufferization::ToBufferOp>()) {
      if (matchPattern(toMemrefOp.getTensor(), m_Constant())) {
        return false;
      }
    }
  }

  return true;
}

bool needToPrmoteCForCooperativeMatrix(linalg::LinalgOp matmulOp) {
  assert(matmulOp.hasPureTensorSemantics());
  Value result = matmulOp.getOperation()->getResult(0);
  if (!result.hasOneUse()) {
    return true; // Be conservative.
  }
  Operation *user = *result.getUsers().begin();
  if (isa<IREE::TensorExt::DispatchTensorStoreOp>(user)) {
    return false;
  }
  if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
    return !isCooperativeMatrixFusable(genericOp);
  }
  return true; // Be conservative.
}

namespace detail {

LogicalResult
setCooperativeMatrixConfig(IREE::GPU::TargetAttr target, linalg::LinalgOp op,
                           const unsigned numSubgroupsPerWorkgroup,
                           const unsigned numMNTilesPerSubgroup,
                           unsigned softwarePipelineDepth,
                           unsigned softwarePipelineStoreStage,
                           unsigned numKTiles) {
  LLVM_DEBUG(llvm::dbgs() << "trying to matmul cooperative matrix config...\n");
  // This configuration is only for cooperative matrix.
  if (target.getWgp().getMma().empty()) {
    return failure();
  }

  if (op.hasDynamicShape()) {
    return failure();
  }

  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value init = op.getDpsInitOperand(0)->get();

  int lastParallelDim = -1;
  const auto [bIndex, mIndex, nIndex, kIndex] =
      getMatmulBMNKIndex(op, &lastParallelDim);
  if (mIndex < 0 || nIndex < 0 || kIndex < 0) {
    return failure();
  }
  const bool isBM = bIndex >= 0;

  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();

  const int64_t dimM = loopRanges[mIndex];
  const int64_t dimK = loopRanges[kIndex];
  const int64_t dimN = loopRanges[nIndex];

  // iree-metal (task#26, 2026-07-23): the coop path is SPECIFICALLY bad at small K — measured batched
  // bmm scales 0.57(K=64)/0.98/1.46/2.19(K=512) TFLOP/s: coop tile-load+setup overhead isn't
  // amortized over only K/16 k-tiles. Attention (K=head_dim=64) is the worst case (4× vs jax-metal's
  // 2.56). Reject small-K batched from coop so it falls through to the register-tiled vectorize path
  // (setMatmulOpConfig), which has no matrix-unit setup overhead and high per-thread reuse — probing
  // whether that beats coop for K<=64. Env-gated IREE_METAL_NO_SMALLK_COOP.
  if (isBM && dimK <= 64 && getenv("IREE_METAL_NO_SMALLK_COOP")) {
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "input matmul shape (B, M, N, K) = ("
                 << (bIndex >= 0 ? loopRanges[bIndex] : -1) << ", " << dimM
                 << ", " << dimN << ", " << dimK << ")\n";
  });

  // TODO: Cooperative matrix support is fairly restricted. We can only have
  // a curated list of fused element wise ops as defined in the extension
  // SPV_KHR_cooperative_matrix. Check that once we move bufferization after
  // vectorization.

  auto getElementType = [](Value v) {
    return cast<ShapedType>(v.getType()).getElementType();
  };

  Type lhsElem = getElementType(lhs);
  Type rhsElem = getElementType(rhs);
  Type initElem = getElementType(init);
  // TODO(Max191): Support multiple M/N/K dimension problems for MMASchedules
  // once the pipeline is able to support it. After adding multiple dimensions,
  // all instances of schedule->m/nSubgroupCounts[0],
  // schedule->m/n/kTileSizes[0] and schedule->m/n/kSizes[0] need to use the
  // full list of sizes instead of just the first element.
  GPUMatmulShapeType problem(dimM, dimN, dimK, lhsElem, rhsElem, initElem);

  SmallVector<GPUIntrinsicType> intrinsics;
  intrinsics.reserve(target.getWgp().getMma().size());
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType, mma);
  }

  // iree-metal: numKTilesPerSubgroup tunable (IREE_METAL_COOP_KT) to probe coop matmul
  // kernel quality — larger K-tiles do more reduction per workgroup load (fewer
  // barriers), closing toward jax-metal's hand-tuned MPS matmul.
  unsigned ktiles = numKTiles;
  if (const char *kt = getenv("IREE_METAL_COOP_KT"))
    ktiles = std::max(1, atoi(kt));
  GPUMMAHeuristicSeeds seeds{numSubgroupsPerWorkgroup, numMNTilesPerSubgroup,
                             ktiles};

  int64_t sharedMemoryLimitInBytes =
      target.getWgp().getMaxWorkgroupMemoryBytes();
  // iree-metal (cont330): under aggressive-fusion a matmul's fused epilogue stages extra
  // threadgroup memory ON TOP of the coop A/B/C tiles this budget bounds, overflowing
  // Apple's 32768 cap on the backward dW/dX matmuls (38-54KB) -> Metal mis-executes ->
  // 34x GNORM. IREE_METAL_COOP_SMEM_BUDGET lowers the schedule's budget so tiles shrink and
  // leave headroom for the fused epilogue (test the fix direction: does correctness
  // return + does the +7% FFN+LN fusion survive).
  if (const char *b = getenv("IREE_METAL_COOP_SMEM_BUDGET"))
    sharedMemoryLimitInBytes = std::min<int64_t>(sharedMemoryLimitInBytes, atoi(b));

  // AMD RDNA architectures supports both wave32 and wave64 modes. Prefer to use
  // wave32 mode for better performance.
  int64_t subgroupSize = target.getPreferredSubgroupSize();

  // Infer if lhs or rhs is transposed to help generate better schedule.
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  bool transposedLhs =
      kIndex != cast<AffineDimExpr>(maps[0].getResults().back()).getPosition();
  bool transposedRhs =
      nIndex != cast<AffineDimExpr>(maps[1].getResults().back()).getPosition();

  // iree-metal: transposed-operand matmuls (e.g. training gradients dW = xᵀ·dY and
  // dX = dY·Wᵀ) DO reach the matrix units — deduceMMASchedule handles the
  // transposed layout, and once the DCE cleanup (dead cloned contracts) +
  // SPIRVBreakDownLargeVector fallback are in place they vectorize+legalize
  // correctly. Allowing them roughly DOUBLES training throughput (1.0 -> 1.86
  // TFLOP/s on gpt2-small fwd+bwd) with the backward grads on coop, and does not
  // regress inference (still 3.49). Kept as default-on (matching upstream, which
  // never rejected transposed); opt OUT with IREE_METAL_COOP_NO_TRANSPOSED only if a
  // specific transposed shape misbehaves.
  if ((transposedLhs || transposedRhs) && getenv("IREE_METAL_COOP_NO_TRANSPOSED")) {
    return failure();
  }

  // iree-metal: reject coop if the matmul result feeds a LAYOUT-CHANGING or
  // REDUCING epilogue fused into the same dispatch. The coop store codegen writes
  // the mma_matrix result straight through the output layout with a plain
  // parallel-parallel store; if the consumer transposes (e.g. training backward
  // dW = xᵀ·dY, identity-in/transposed-out maps) OR reduces (e.g. attention
  // q@kᵀ feeding the softmax max/sum reduction), VectorToGPU converts only part
  // of the K-loop and leaves illegal coop-native vectors (vector<16x16xf32> /
  // vector<16xf32>) that fail SPIR-V legalization and kill the WHOLE module
  // compile. Routing these to the scalar pipeline compiles cleanly (it has the
  // full vector-breakdown passes). A CLEAN elementwise epilogue (truncf/gelu,
  // identity maps, all-parallel) is fine and stays on coop. Opt out with
  // IREE_METAL_COOP_ALLOW_TRANSPOSE_EPILOGUE.
  if (!getenv("IREE_METAL_COOP_ALLOW_TRANSPOSE_EPILOGUE")) {
    Value mmResult = op.getOperation()->getResult(0);
    for (Operation *user : mmResult.getUsers()) {
      auto genericOp = dyn_cast<linalg::GenericOp>(user);
      if (!genericOp || genericOp.getNumDpsInits() != 1)
        continue;
      // A consumer with any reduction iterator (softmax/norm) can't be a coop
      // store epilogue.
      if (genericOp.getNumReductionLoops() != 0)
        return failure();
      AffineMap outMap =
          genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
      for (OpOperand *in : genericOp.getDpsInputOperands()) {
        if (in->get() != mmResult)
          continue;
        AffineMap inMap = genericOp.getMatchingIndexingMap(in);
        // Different in/out maps on the matmul result => a transpose (or other
        // layout-changing) epilogue. Coop can't store through it.
        if (inMap != outMap)
          return failure();
      }
    }
  }

  FailureOr<GPUMMASchedule> schedule = deduceMMASchedule(
      problem, intrinsics, seeds, sharedMemoryLimitInBytes, subgroupSize,
      /*cuCount=*/std::nullopt, op.getLoc(), transposedLhs, transposedRhs,
      // iree-metal: canUpcastAcc lets a bf16-OUTPUT matmul match the f32-acc bf16 MMA.
      // But the RaiseContractionAccumulatorToF32 pass already makes the target
      // matmuls f32-OUTPUT (exact coop match, no upcast needed). canUpcastAcc only
      // affects the bf16-OUTPUT matmuls the pass DIDN'T promote (attention +
      // transposed backward forms), which FAIL coop vectorization. So keep it OFF
      // by default (separate env) — promoted matmuls still get matrix units via
      // exact match; the failing bf16-output ones cleanly fall to scalar.
      /*canUpcastAcc=*/getenv("IREE_METAL_COOP_UPCAST_ACC") != nullptr &&
          dimM >= 128 && dimN >= 128 && dimK >= 128,
      /*useDirectLoad=*/false, /*prefetchNumStages=*/0,
      // iree-metal: mustBeAligned=true (default) rejects non-mult-16 M/N/K from coop
      // (e.g. vit M=B*T=4616 -> scalar, 0.31 TFLOP/s vs jax-metal 2.51). Allow
      // UNALIGNED coop (the codegen pads/masks the boundary tiles) so odd-sized
      // models still hit the matrix units. Env-gated IREE_METAL_COOP_UNALIGNED until
      // validated numerically on the Metal path.
      /*mustBeAligned=*/getenv("IREE_METAL_COOP_UNALIGNED") == nullptr);
  if (getenv("IREE_METAL_COOP_DEBUG")) {
    llvm::errs() << "[coop] MxNxK=" << dimM << "x" << dimN << "x" << dimK
                 << " lhs=" << lhsElem << " " << (failed(schedule) ? "SCALAR" : "COOP")
                 << "\n";
  }
  if (failed(schedule)) {
    return failure();
  }
  assert(schedule->hasSingleDimensions() && "expected single M/N/K dimension");

  // iree-metal (task#26, 2026-07-23): batched small-K matmuls (attention q@kᵀ / a@v, K<=128) run at
  // ~25% of MPS because deduceMMASchedule (with batch tiled to 1) picks a small per-workgroup MN
  // tile, so each workgroup does too little coop work to hide A/B-load latency (measured
  // latency-bound: 17.5 vs peak ~200 GB/s). The MNT *seed* can't move this (the heuristic caps
  // it), so directly grow the per-subgroup MN coop tiles here. Env-gated IREE_METAL_COOP_BMM_BOOST=<f>;
  // the SMEM check below reduces pipeline depth / the schedule already bounds tiles, and any
  // over-large tile is rejected downstream. Correctness-gate + measure before default-on.
  if (isBM && dimK <= 128) {
    if (const char *bf = getenv("IREE_METAL_COOP_BMM_BOOST")) {
      unsigned factor = std::max(1, atoi(bf));
      // Only grow while the tile still divides the problem dim (keep mustBeAligned valid).
      while (factor > 1 &&
             (dimM % (schedule->mSubgroupCounts[0] * schedule->mTileSizes[0] * 2 *
                      schedule->mSizes[0]) == 0)) {
        schedule->mTileSizes[0] *= 2;
        if ((factor /= 2) <= 1)
          break;
      }
      factor = std::max(1u, (unsigned)(getenv("IREE_METAL_COOP_BMM_BOOST") ? atoi(getenv("IREE_METAL_COOP_BMM_BOOST")) : 1));
      while (factor > 1 &&
             (dimN % (schedule->nSubgroupCounts[0] * schedule->nTileSizes[0] * 2 *
                      schedule->nSizes[0]) == 0)) {
        schedule->nTileSizes[0] *= 2;
        if ((factor /= 2) <= 1)
          break;
      }
    }
  }

  auto pipeline = CodeGenPipeline::SPIRVCooperativeMatrixVectorize;

  std::array<int64_t, 3> workgroupSize{schedule->nSubgroupCounts[0] *
                                           subgroupSize,
                                       schedule->mSubgroupCounts[0], 1};

  SmallVector<int64_t> vectorSizes(kIndex + 1, 0);
  if (isBM) {
    vectorSizes[bIndex] = 1;
  }
  vectorSizes[mIndex] = schedule->mSizes[0];
  vectorSizes[nIndex] = schedule->nSizes[0];
  vectorSizes[kIndex] = schedule->kSizes[0];

  SmallVector<int64_t> subgroupTileSizes(lastParallelDim + 1, 0);
  if (isBM) {
    subgroupTileSizes[bIndex] = 1;
  }
  subgroupTileSizes[mIndex] = schedule->mTileSizes[0] * vectorSizes[mIndex];
  subgroupTileSizes[nIndex] = schedule->nTileSizes[0] * vectorSizes[nIndex];

  SmallVector<int64_t> workgroupTileSizes(lastParallelDim + 1, 0);
  if (isBM) {
    workgroupTileSizes[bIndex] = 1;
  }
  workgroupTileSizes[mIndex] =
      schedule->mSubgroupCounts[0] * subgroupTileSizes[mIndex];
  workgroupTileSizes[nIndex] =
      schedule->nSubgroupCounts[0] * subgroupTileSizes[nIndex];

  // Also create one level for reduction. This is needed because of
  // SPIRVTileAndPromotePass requires it.
  // TODO(#10499): Consolidate tiling configuration across different pipelines.
  SmallVector<int64_t> reductionTileSizes;
  reductionTileSizes.append(kIndex, 0);
  reductionTileSizes.push_back(schedule->kTileSizes[0] * schedule->kSizes[0]);

  TileSizesListType tileSizes = {workgroupTileSizes, subgroupTileSizes,
                                 reductionTileSizes, vectorSizes};

  // Don't do multibuffering if the inner reduction loop is folded out.
  auto pipelineDepth = softwarePipelineDepth;
  auto storeStage = softwarePipelineStoreStage;
  if (schedule->kTileSizes[0] <= 1) {
    pipelineDepth = 0;
    storeStage = 0;
  }

  // Check if the C matrix will be promoted for computing shared memory usage.
  bool promoteC = needToPrmoteCForCooperativeMatrix(op);

  // Decrease pipeline depth until it fits in shared memory.
  const int maxBytes = target.getWgp().getMaxWorkgroupMemoryBytes();
  auto usedBytes =
      getTileBytes(workgroupTileSizes[mIndex], workgroupTileSizes[nIndex],
                   reductionTileSizes[kIndex],
                   IREE::Util::getTypeBitWidth(getElementType(lhs)), promoteC,
                   /*cElementBits=*/IREE::Util::getTypeBitWidth(
                       getElementType(init)));

  while (pipelineDepth > 0 &&
         getMultiBufferMemoryUsage(usedBytes, pipelineDepth, storeStage) >
             maxBytes) {
    pipelineDepth--;
  }

  if (getenv("IREE_METAL_COOP_MEM_DEBUG")) {
    llvm::errs() << "[coop-mem] MxNxK=" << dimM << "x" << dimN << "x" << dimK
                 << " wgTile M/N=" << workgroupTileSizes[mIndex] << "/"
                 << workgroupTileSizes[nIndex]
                 << " redK=" << reductionTileSizes[kIndex]
                 << " promoteC=" << promoteC << " pd=" << pipelineDepth
                 << " usedBytes=" << usedBytes << " multiBuf="
                 << getMultiBufferMemoryUsage(usedBytes, pipelineDepth, storeStage)
                 << " max=" << maxBytes << "\n";
  }

  // iree-metal: even at minimum pipeline depth the tile may exceed the target's
  // threadgroup-memory limit (getTileBytes only reflects the base tiles; the
  // pipeline-depth loop above can't shrink the tile itself). Emitting such a
  // kernel makes Metal SILENTLY HANG at runtime (52KB requested vs 32KB limit —
  // the HAL wedge). Refuse the coop config here so this op falls back to the
  // scalar/vector pipeline instead of generating an over-allocating kernel.
  if (getMultiBufferMemoryUsage(usedBytes, pipelineDepth, storeStage) >
      maxBytes) {
    return failure();
  }

  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes, pipeline,
      workgroupSize, subgroupSize,
      getSoftwarePipeliningAttrDict(op->getContext(), pipelineDepth,
                                    storeStage));
}

} // namespace detail

//===----------------------------------------------------------------------===//
// FFT Default Configuration
//===----------------------------------------------------------------------===//

// iree-metal: flash-attention config. Routes iree_linalg_ext.attention to the
// SPIRVVectorDistributeAttention pipeline (decompose online-attention within
// codegen -> per-tile matmul+softmax -> vectorize). First increment: distribute
// the leading parallel dims (batch + query-M) one-per-workgroup and let the
// reduction (scores/K2) tile via the pipeline's GPUApplyTilingLevel. Tile sizes
// are intentionally simple; perf tuning (coop routing) is a follow-up.
static LogicalResult setAttentionOpConfig(IREE::GPU::TargetAttr target,
                                          IREE::LinalgExt::AttentionOp op) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as attention...\n");
  // iree-metal: the SPIRVVectorDistributeAttention pipeline + this config are DORMANT
  // scaffolding. Routing attention to it hangs codegen: the decomposed per-tile
  // matmuls need proper tile configs, and the only machinery that produces them
  // (LLVMGPU vector-distribute: ConfigureTensorLayouts + PackToIntrinsics + vector
  // distribution) is deeply LLVMGPU/LLVM-coupled with no SPIR-V/coop equivalent —
  // a major codegen port, not a wiring job. Until that's built, return failure so
  // attention ops don't route to the (non-functional) pipeline. Opt in for
  // development with IREE_METAL_COOP_ATTENTION_WIP.
  if (!getenv("IREE_METAL_COOP_ATTENTION_WIP")) {
    return failure();
  }
  int subgroupSize = target.getPreferredSubgroupSize();
  auto pipeline = CodeGenPipeline::SPIRVVectorDistributeAttention;

  FailureOr<SmallVector<int64_t>> maybeBounds = op.getStaticLoopRanges();
  if (failed(maybeBounds) || maybeBounds->size() < 2 ||
      llvm::any_of(*maybeBounds, ShapedType::isDynamic)) {
    return failure();
  }
  ArrayRef<int64_t> bounds = *maybeBounds;

  // iree-metal: classify the attention iteration dims (batch, M=query, K1=qk-dim,
  // K2=kv-seq, N=value-dim) so we can build a proper iree_gpu.lowering_config
  // with named tiling levels. The earlier scaffold set a generic
  // Codegen::LoweringConfig with only a flat workgroup list — but
  // GPUApplyTilingLevel(Reduction) reads hasTilingLevel(Reduction) off an
  // IREE::GPU::LoweringConfigAttr, so with the generic config it was a NO-OP:
  // the kv-reduction never tiled, each workgroup materialized the FULL [M,K2]
  // scores, and SPIRVVectorLowering HUNG unrolling 256-wide vectors. Here we
  // (a) workgroup-tile batch+query-M and (b) reduction-tile the kv-seq (K2), so
  // the flash scf.for loop forms and the decomposed per-tile matmul/softmax ops
  // stay small (first increment: COMPILE without hanging; coop routing of the
  // per-tile matmuls is the perf follow-up).
  auto opInfo = IREE::LinalgExt::AttentionOpDetail::get(
      op.getQueryMap(), op.getKeyMap(), op.getValueMap(), op.getOutputMap());
  if (failed(opInfo)) {
    return failure();
  }
  int64_t rank = opInfo->getDomainRank();
  if (opInfo->getMDims().empty() || opInfo->getK2Dims().empty()) {
    return failure();
  }
  SmallVector<int64_t> workgroupTileSizes(rank, 0);
  SmallVector<int64_t> reductionTileSizes(rank, 0);
  for (int64_t d : opInfo->getBatchDims())
    workgroupTileSizes[d] = 1;
  for (int64_t d : llvm::drop_end(opInfo->getMDims()))
    workgroupTileSizes[d] = 1;
  int64_t mDim = opInfo->getMDims().back();
  // iree-metal first-increment: tile query-M to a small block. Without thread-level
  // distribution (the LLVMGPU vector-distribute step this SPIRV pipeline lacks),
  // the per-workgroup tile is fully UNROLLED to vector<4> ops, so a large M-tile
  // (e.g. 32) explodes code size (~1.2M ops -> effectively hangs). Keep M small
  // until proper thread/subgroup distribution is added (perf follow-up).
  int64_t mBlock = std::min<int64_t>(bounds[mDim], getenv("IREE_METAL_COOP_ATTN_M")
                                                       ? atoi(getenv("IREE_METAL_COOP_ATTN_M"))
                                                       : 1);
  workgroupTileSizes[mDim] = mBlock;
  for (int64_t d : llvm::drop_end(opInfo->getK2Dims()))
    reductionTileSizes[d] = 1;
  int64_t k2Dim = opInfo->getK2Dims().back();
  reductionTileSizes[k2Dim] = std::min<int64_t>(
      bounds[k2Dim],
      getenv("IREE_METAL_COOP_ATTN_K2") ? atoi(getenv("IREE_METAL_COOP_ATTN_K2")) : 32);

  Builder b(op.getContext());
  SmallVector<NamedAttribute, 3> attrs = {
      b.getNamedAttr("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      b.getNamedAttr("reduction", b.getI64ArrayAttr(reductionTileSizes))};
  // iree-metal (IREE_METAL_COOP_ATTN_THREAD): distribute the query-M block across the
  // subgroup's threads (each thread does ONE query's full attention: scalar
  // qk/softmax/pv). This makes a workgroup process mBlock queries in PARALLEL
  // instead of 1 (fixing the M=1 32x-redundancy) and keeps per-thread code small
  // (fixing the M=32 unroll explosion) — WITHOUT the coop-layout/softmax
  // thread-distribution interface that walls the coop path. No matrix units, but
  // full occupancy.
  if (getenv("IREE_METAL_COOP_ATTN_THREAD")) {
    SmallVector<int64_t> threadTileSizes(rank, 0);
    threadTileSizes[mDim] = 1; // one query per thread
    attrs.push_back(
        b.getNamedAttr("thread", b.getI64ArrayAttr(threadTileSizes)));
  }
  auto configDict = DictionaryAttr::get(op.getContext(), attrs);
  auto loweringConfig =
      IREE::GPU::LoweringConfigAttr::get(op.getContext(), configDict);

  // iree-metal (IREE_METAL_COOP_ATTN_DECOMP): attach SPIR-V coop lowering_configs to the
  // decomposed qk/pv matmuls via decomposition_config, so the coop passes
  // (SPIRVTileToCooperativeOps + SPIRVVectorizeToCooperativeOps, wired into the
  // attention pipeline under the same env) route them onto the matrix units
  // instead of generic vectorization. qk loops = [batch, M, K1(reduction), K2]
  // (coop M=M, N=K2, K=K1); pv loops = [batch, M, K2(reduction), N] (coop M=M,
  // N=N, K=K2). Levels: L0 workgroup (already attention-tiled), L1 subgroup,
  // L2 reduction, L3 native/coop shape (16x16x16). Perf follow-up to the
  // correct-but-slow M=1 path; must run with IREE_METAL_COOP_ATTN_M>=16.
  if (getenv("IREE_METAL_COOP_ATTN_DECOMP") || getenv("IREE_METAL_COOP_ATTN_SMEM") ||
      getenv("IREE_METAL_COOP_ATTN_COOPSMEM")) {
    auto mk = [&](TileSizesListType t) {
      return IREE::Codegen::LoweringConfigAttr::get(op.getContext(), t);
    };
    // L0 workgroup + L1 subgroup must be NON-ZERO on the parallel dims or
    // deduceSubgroupCounts (SPIRVTileAndVectorizeToCooperativeOps.cpp:107) computes
    // workgroupTile/subgroupTile = 0/16 = 0 subgroups -> affine.delinearize_index
    // non-positive-basis error (cont78g). Set L0==L1 = the full per-workgroup op
    // size so subgroupCount=1 (matches the attention workgroupSize={subgroupSize,1,1});
    // the L3 native 16x16x16 tile makes the single subgroup loop coop tiles.
    // qk loops [batch, M, K1(reduction), K2]; pv loops [batch, M, K2(reduction), N].
    // GENERALIZED off the op's real dims (cont80): mBlock (query tile), k2Tile (kv
    // reduction tile), k1Size (qk reduction = head dim), nSize (pv N = value dim).
    int64_t k2Tile = reductionTileSizes[k2Dim];
    int64_t k1Size = opInfo->getK1Dims().empty()
                         ? 16
                         : bounds[opInfo->getK1Dims().back()];
    int64_t nSize = opInfo->getNDims().empty()
                        ? 16
                        : bounds[opInfo->getNDims().back()];
    auto redTile = [](int64_t d) { return d >= 16 ? int64_t(16) : d; };
    TileSizesListType qkT = {{0, mBlock, 0, k2Tile},
                             {0, mBlock, 0, k2Tile},
                             {0, 0, redTile(k1Size), 0},
                             {1, 16, 16, 16}};
    TileSizesListType pvT = {{0, mBlock, 0, nSize},
                             {0, mBlock, 0, nSize},
                             {0, 0, redTile(k2Tile), 0},
                             {1, 16, 16, 16}};
    auto qkAttrs = DictionaryAttr::get(
        op.getContext(), {b.getNamedAttr("lowering_config", mk(qkT))});
    auto pvAttrs = DictionaryAttr::get(
        op.getContext(), {b.getNamedAttr("lowering_config", mk(pvT))});
    auto decomp = DictionaryAttr::get(
        op.getContext(),
        {b.getNamedAttr(IREE::LinalgExt::AttentionOp::getQKAttrStr(), qkAttrs),
         b.getNamedAttr(IREE::LinalgExt::AttentionOp::getPVAttrStr(), pvAttrs)});
    op.setDecompositionConfigAttr(decomp);
  }

  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};
  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<mlir::FunctionOpInterface>(), op, loweringConfig,
      pipeline, workgroupSize, subgroupSize);
}

static LogicalResult setFftOpConfig(IREE::GPU::TargetAttr target,
                                    IREE::LinalgExt::FftOp op) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as fft...\n");
  int subgroupSize = target.getPreferredSubgroupSize();
  auto pipeline = CodeGenPipeline::SPIRVBaseDistribute;

  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};

  SmallVector<utils::IteratorType> loopIteratorTypes =
      op.getLoopIteratorTypes();
  unsigned loopDepth = loopIteratorTypes.size();
  SmallVector<int64_t> workgroupTileSize(loopDepth, 0);

  // Tiling along partitioned loops with size 1.
  for (auto [index, iteratorType] : llvm::enumerate(loopIteratorTypes)) {
    if (iteratorType == utils::IteratorType::parallel) {
      workgroupTileSize[index] = 1;
    }
  }
  auto rank = op.getOperandRank();
  if (workgroupTileSize.size() >= rank && workgroupTileSize[rank - 1] != 0) {
    APInt value;
    if (matchPattern(op.getStage(), m_ConstantInt(&value))) {
      workgroupTileSize[rank - 1] = 1ll << value.getSExtValue();
    } else {
      op.emitError("non-constant stage might not work for fft op");
      return failure();
    }
  }
  TileSizesListType tileSizes = {workgroupTileSize};
  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes, pipeline,
      workgroupSize);
}

//===----------------------------------------------------------------------===//
// Scatter Default Configuration
//===----------------------------------------------------------------------===//

// iree-metal: scatter config. By default a scatter whose leading (batch/index) loop
// is a reduction (non-unique indices, e.g. embedding-gradient scatter-add) gets
// NO partitionable loops -> SPIRVBaseDistribute with workgroup_size [1,1,1] =
// ONE thread does the whole scatter (measured 27x slower than jax-metal). But
// the update-vector (D) loops ARE parallel: distribute them to threads, keeping
// the reduction batch loop untiled/serial inside each thread. Each thread then
// owns distinct output COLUMNS, so scatter-add into overlapping rows has no
// cross-thread race (no atomics needed). Restores parallelism.
static LogicalResult setScatterOpConfig(IREE::GPU::TargetAttr target,
                                        IREE::LinalgExt::ScatterOp op) {
  int subgroupSize = target.getPreferredSubgroupSize();
  auto pipeline = CodeGenPipeline::SPIRVBaseDistribute;
  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};

  SmallVector<utils::IteratorType> its = op.getLoopIteratorTypes();
  unsigned depth = its.size();
  SmallVector<int64_t> wgTile(depth, 0), threadTile(depth, 0);
  int lastParallel = -1;
  // iree-metal (IREE_METAL_SCATTER_ATOMIC): a non-unique-index scatter marks its batch dims as REDUCTION, so
  // by default only the window dim is distributed and the (large) update batch runs serially in ~1
  // workgroup — the ~10x-slow embedding-backward. When the scatter-add is lowered to atomic_rmw (see
  // ScatterOp::generateScalarImplementation), duplicate indices are safe, so the batch IS parallel:
  // tile those reduction dims into a workgroup grid too.
  bool atomicParallel = getenv("IREE_METAL_SCATTER_ATOMIC") != nullptr;
  for (auto [i, it] : llvm::enumerate(its)) {
    if (it == utils::IteratorType::parallel) {
      wgTile[i] = 1;
      threadTile[i] = 1;
      lastParallel = i;
    } else if (atomicParallel) {
      wgTile[i] = 1; // atomic scatter-add -> batch dim is safe to distribute across workgroups
      threadTile[i] = 1;
    }
  }
  if (lastParallel < 0) { // no parallel loop -> keep serial [1,1,1] (old default)
    return setOpConfigAndEntryPointFnTranslation(
        op->getParentOfType<mlir::FunctionOpInterface>(), op, TileSizesListType{},
        pipeline, std::array<int64_t, 3>{1, 1, 1});
  }
  // Distribute the innermost parallel loop across the subgroup's threads.
  wgTile[lastParallel] = subgroupSize;
  threadTile[lastParallel] = 1;
  TileSizesListType tileSizes = {wgTile, threadTile};
  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes, pipeline,
      workgroupSize);
}

//===----------------------------------------------------------------------===//
// Winograd Default Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setWinogradOpConfig(IREE::GPU::TargetAttr target,
                                         IREE::LinalgExt::LinalgExtOp op) {
  // Tiling is already done by tile and decompose, so we only set pipeline and
  // workgroup size. The tile sizes below are placeholders and were obtained
  // by manual tuning on the AMD Navi2 GPU on a small set of convolution
  // sizes found in the StableDiffusion model.
  auto pipeline = CodeGenPipeline::SPIRVWinogradVectorize;
  std::array<int64_t, 3> workgroupSize = {32, 4, 4};
  TileSizesListType tileSizes = {{1, 0, 0, 32}, {1, 1, 1, 1}, {0, 0, 0, 0}};
  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes, pipeline,
      workgroupSize);
}

//===----------------------------------------------------------------------===//
// Reduction Default Configuration
//===----------------------------------------------------------------------===//

static bool canDistributeShape(ArrayRef<int64_t> shape, int64_t groupSize) {
  for (int64_t dim : shape) {
    if (dim % groupSize == 0) {
      return true;
    }
    if (groupSize % dim == 0) {
      groupSize /= dim;
      continue;
    }
    return false;
  }
  return groupSize == 1;
};

// Check if a reduction has consumers incompatible with warp distribution
// based on the reductions' attached lowering config. The reduction itself may
// be distributable, but since distribution patterns work bottom-up from the
// yield, if a consumer shape fails distribution, it stays inside the region,
// which keeps its operands (including the reduction) inside too. This forces
// fallback to single thread execution.
//
// Example: consumer broadcast shape incompatible with the proposed workgroup
// size. Here the heuristics select workgroup size = 512.
//
//   %red = linalg.reduce ins(%in) outs(%init) -> tensor<f32>
//   %consumer = linalg.generic {
//     indexing_maps = [affine_map<(d0, d1, d2) -> ()>,
//                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
//   } ins(%red) outs(%out : tensor<64x3x32xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       %add = arith.addf %in, %out : f32
//       linalg.yield %add : f32
//   }
//
// The scalar reduction result broadcasts across consumer dims [64, 3, 32]. When
// we try to distribute across this shape for elementwise addition using warp
// distribution expectations:
//   - d0=64: 512 % 64 = 0; so remaining is 512/64 = 8
//   - d1=3: 8 % 3 != 0 and 3 % 8 != 0 -> failure
// Since the consumer fails distribution, the reduction is also blocked and
// the entire chain of operations stays inside the region.
//
// Example: consumer indexing map permutes dims in a way that
// breaks producer-defined distribution.
//
//   %red = linalg.generic {
//     indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
//                      affine_map<(d0, d1, d2) -> (d0, d1)>],
//     iterator_types = ["parallel", "parallel", "reduction"]
//   } ins(%in : tensor<16x64x74xf32>)
//     outs(%init : tensor<16x74xf32>) { ... } -> tensor<16x74xf32>
//
//   %consumer = linalg.generic {
//     indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
//                      affine_map<(d0, d1, d2) -> (d0, d2)>,
//                      affine_map<(d0, d1, d2) -> (d0, d2, d1)>],
//     iterator_types = ["parallel", "parallel", "parallel"]
//   } ins(%in, %red : tensor<16x64x74xf32>, tensor<16x74xf32>)
//     outs(%out : tensor<16x74x64xf32>) { ... }
//
// Producer tiling is defined in the producer iteration space (par: d0=16,d1=74;
// red: d2=64). Consumer indexing maps can reassociate %red such that a
// producer reduction dim becomes a consumer indexing dim; those dims must
// satisfy the groupSize distribution constraint directly. Here that extent is
// 74, which fails for group size = 32, so we bail out of the reduction
// pipeline.
static bool isConsumerCompatible(linalg::LinalgOp consumerOp, Value result,
                                 ArrayRef<unsigned> reductionDims,
                                 int64_t groupSize) {
  // Collect dims used by operands referencing the reduction result.
  llvm::SmallDenseSet<unsigned> usedDims;
  for (OpOperand &operand : consumerOp->getOpOperands()) {
    if (operand.get() != result) {
      continue;
    }
    for (AffineExpr expr :
         consumerOp.getMatchingIndexingMap(&operand).getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        usedDims.insert(dimExpr.getPosition());
      }
    }
  }

  SmallVector<int64_t> usedShape, broadcastShape;
  for (auto [i, size] : llvm::enumerate(consumerOp.getStaticLoopRanges())) {
    // Broadcast dims are those not indexed by any use of the result.
    if (!usedDims.contains(i)) {
      broadcastShape.push_back(size);
      continue;
    }
    // Producer reduction dims that are used by the consumer are not covered
    // by workgroup tiling, so require them to be distributable for groupSize.
    if (!llvm::is_contained(reductionDims, i)) {
      continue;
    }
    usedShape.push_back(size);
  }

  return (usedShape.empty() || canDistributeShape(usedShape, groupSize)) &&
         (broadcastShape.empty() ||
          canDistributeShape(broadcastShape, groupSize));
}

static bool allConsumersCompatible(linalg::LinalgOp reductionOp,
                                   ArrayRef<unsigned> reductionDims,
                                   int64_t groupSize, int64_t candidate) {
  for (Value result : reductionOp->getResults()) {
    for (Operation *user : result.getUsers()) {
      auto consumerOp = dyn_cast<linalg::LinalgOp>(user);
      if (!consumerOp) {
        continue;
      }
      if (!isConsumerCompatible(consumerOp, result, reductionDims, candidate)) {
        return false;
      }
    }
  }
  return true;
}

// Try to find a workgroup size compatible with all consumers by
// halving from groupSize down to subgroupSize.
static FailureOr<int>
maybeFindConsumerCompatibleSize(linalg::LinalgOp reductionOp,
                                ArrayRef<unsigned> reductionDims,
                                int64_t groupSize, int64_t subgroupSize) {
  for (int64_t candidate = groupSize; candidate >= subgroupSize;
       candidate /= 2) {
    if (allConsumersCompatible(reductionOp, reductionDims, groupSize,
                               candidate)) {
      return candidate;
    }
  }
  return failure();
}

/// Set the configuration for reductions that can be mapped to warp reductions.
static LogicalResult setReductionConfig(IREE::GPU::TargetAttr target,
                                        linalg::LinalgOp op) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as reduction...\n");

  // This pipeline eventually generates non-uniform group shuffle ops, which
  // requires special capability.
  if (!target.supportsSubgroupShuffle()) {
    return failure();
  }

  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;
  op.getParallelDims(parallelDims);
  op.getReductionDims(reductionDims);

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  int64_t numParallelDims = op.getNumParallelLoops();

  // We should have reduction dimensions.
  if (reductionDims.empty()) {
    return failure();
  }

  // Make sure reduction dimensions are static and innermost ones.
  int64_t numDynamicReductionDims = 0;
  for (unsigned dim : reductionDims) {
    if (ShapedType::isDynamic(bounds[dim])) {
      numDynamicReductionDims++;
    }
    if (dim < numParallelDims) {
      LLVM_DEBUG(llvm::dbgs() << "failed: non-innermost reduction dims\n");
      return failure();
    }
  }

  // Distribution of multi-dim masked writes currently aren't fully supported.
  if (numDynamicReductionDims > 1) {
    return failure();
  }

  if (op.getRegionOutputArgs().size() != 1) {
    return failure();
  }

  // Only support projected permutation for now. This could be extended to
  // projected permutated with broadcast.
  if (llvm::any_of(op.getDpsInputOperands(), [&](OpOperand *input) {
        return !op.getMatchingIndexingMap(input).isProjectedPermutation();
      })) {
    return failure();
  }

  bool foundSingleReductionOutput = false;
  for (int64_t i = 0, e = op.getDpsInits().size(); i < e; i++) {
    // Only single combiner operations are supported for now.
    SmallVector<Operation *> combinerOps;
    if (matchReduction(op.getRegionOutputArgs(), i, combinerOps) &&
        combinerOps.size() == 1) {
      if (foundSingleReductionOutput) {
        return failure();
      }
      foundSingleReductionOutput = true;
      continue;
    }
    if (!op.getMatchingIndexingMap(op.getDpsInitOperand(i)).isIdentity()) {
      return failure();
    }
  }
  if (!foundSingleReductionOutput) {
    return failure();
  }

  int subgroupSize = target.getPreferredSubgroupSize();

  // Tile all the parallel dimension to 1.
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  llvm::SmallDenseSet<unsigned, 4> partitionedLoopsSet;
  partitionedLoopsSet.insert(partitionedLoops.begin(), partitionedLoops.end());
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);

  // Without any bounds on dynamic reduction dims, we need specialization to
  // get peak performance. For now, just use the subgroup size.
  if (numDynamicReductionDims) {
    SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
    reductionTileSizes[reductionDims[0]] = subgroupSize;
    TileSizesListType tileSizes;
    tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
    tileSizes.emplace_back(std::move(reductionTileSizes)); // Reduction level
    std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};
    if (failed(setOpConfigAndEntryPointFnTranslation(
            op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
            CodeGenPipeline::SPIRVSubgroupReduce, workgroupSize))) {
      return failure();
    }

    // Set lowering configuration to drive tiling for other Linalg ops too---the
    // pipeline expects it.
    op->getParentOfType<FunctionOpInterface>().walk([&](linalg::LinalgOp op) {
      setLoweringConfig(op, IREE::Codegen::LoweringConfigAttr::get(
                                op.getContext(), tileSizes));
    });
    return success();
  }

  int64_t reductionSize = 1;
  for (int64_t dim : reductionDims) {
    reductionSize *= bounds[dim];
  }
  if (reductionSize % subgroupSize != 0) {
    // IREE_METAL_COOP_REDUCE_NONMULT (default ON): a STATIC reduction whose size isn't
    // a multiple of the subgroup (e.g. softmax over seq=577) would otherwise bail to
    // the scalar default config — which is BOTH 16.6x slower AND (validated vs CPU)
    // computes WRONG gradients on the softmax backward (vit grad-norm was 43% off).
    // The subgroup-reduce pipeline already handles arbitrary sizes with a masked
    // remainder (the dynamic-reduction path above), so use it here too: correct +
    // fast. Opt out with IREE_METAL_COOP_NO_REDUCE_NONMULT.
    if (getenv("IREE_METAL_COOP_NO_REDUCE_NONMULT")) {
      return failure();
    }
    SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
    reductionTileSizes[reductionDims[0]] = subgroupSize;
    TileSizesListType tileSizes;
    tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
    tileSizes.emplace_back(std::move(reductionTileSizes)); // Reduction level
    std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};
    if (failed(setOpConfigAndEntryPointFnTranslation(
            op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
            CodeGenPipeline::SPIRVSubgroupReduce, workgroupSize))) {
      return failure();
    }
    op->getParentOfType<FunctionOpInterface>().walk([&](linalg::LinalgOp op2) {
      setLoweringConfig(op2, IREE::Codegen::LoweringConfigAttr::get(
                                 op2.getContext(), tileSizes));
    });
    return success();
  }

  const Type elementType =
      cast<ShapedType>(op.getDpsInits()[0].getType()).getElementType();
  if (!elementType.isIntOrFloat()) {
    return failure();
  }
  unsigned bitWidth = IREE::Util::getTypeBitWidth(elementType);
  // Reduction distribution only supports 8/16/32 bit types now.
  if (bitWidth != 32 && bitWidth != 16 && bitWidth != 8) {
    return failure();
  }

  // Let each thread handle `vectorSize` elements.
  unsigned vectorSize = kMaxVectorNumBits / bitWidth;
  while ((reductionSize / vectorSize) % subgroupSize != 0) {
    vectorSize /= 2;
  }

  // Deduce the workgroup size we should use for reduction. Currently a
  // workgroup processes all elements in reduction dimensions. Need to make sure
  // the workgroup size we use can divide the total reduction size, and it's
  // also within hardware limitations.
  const int64_t maxWorkgroupSize =
      target.getWgp().getMaxThreadCountPerWorkgroup();
  int64_t groupSize = reductionSize / vectorSize;
  if (groupSize > maxWorkgroupSize) {
    groupSize = GreatestCommonDivisor(APInt(64, uint64_t(groupSize)),
                                      APInt(64, uint64_t(maxWorkgroupSize)))
                    .getZExtValue();
  }

  // Then we need to strike a balance--
  // 1) parallel dimensions are distributed to workgroups. If there are many
  //    workgroups dispatched, we'd want to have each GPU core hosting multiple
  //    of them for occupancy.
  // 2) we want each thread to read quite a few 128-bit vectors for better
  //    memory cache behavior.
  // Both means we cannot use a too large workgroup size.

  int64_t parallelSize = 1;
  for (int64_t dim : parallelDims) {
    if (ShapedType::isStatic(bounds[dim])) {
      parallelSize *= bounds[dim];
    }
  }
  // Total parallel size that can fill the GPU with enough workgorups.
  // TODO: query from the target device; roughly 2x hardware compute unit.
  // iree-metal: env-tunable (IREE_METAL_RED_PT) to probe reduction occupancy on Apple —
  // the default 256 (NVIDIA-era) may under/over-subscribe Apple's ~10-20 cores.
  int parallelThreshold = 256;
  if (const char *v = getenv("IREE_METAL_RED_PT")) parallelThreshold = atoi(v);
  // How many 128-bit vectors each thread should at least read (IREE_METAL_RED_VC):
  // more work/thread => fewer subgroups => less reduction-tree overhead.
  int targetVectorCount = 8;
  if (const char *v = getenv("IREE_METAL_RED_VC")) targetVectorCount = atoi(v);
  while (parallelSize > parallelThreshold &&
         (groupSize / 2) % subgroupSize == 0 &&
         reductionSize / (groupSize * vectorSize) < targetVectorCount) {
    // Use less subgroups per workgroup..
    groupSize /= 2;
    // in order to host more workgroups per hardware compute unit.
    parallelSize /= 2;
  }

  // Current warp reduction pattern is a two step butterfly warp reduce.
  // First, do warp reductions along multiple subgroups.
  // Second, reduce results from multiple subgroups using single warp reduce.
  // The final warp reduce requires subgroup count <= subgroup size to work.
  if ((groupSize / subgroupSize) > subgroupSize) {
    return failure();
  }

  FailureOr<int64_t> maybeCompatibleGroupSize = maybeFindConsumerCompatibleSize(
      op, reductionDims, groupSize, subgroupSize);
  if (failed(maybeCompatibleGroupSize)) {
    LDBG() << "Reduction has incompatible consumer";
    return failure();
  }
  if (groupSize != maybeCompatibleGroupSize.value()) {
    LDBG() << "Reduction adjusted workgroup size from " << groupSize << " to "
           << maybeCompatibleGroupSize.value() << " for consumer compatibility";
    groupSize = maybeCompatibleGroupSize.value();
  }

  std::array<int64_t, 3> workgroupSize = {groupSize, 1, 1};

  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  int64_t remainingGroupSize = groupSize;
  for (int i = reductionDims.size() - 1; i >= 0; --i) {
    int64_t dim = reductionDims[i];
    int64_t bound = bounds[dim];
    if (i == reductionDims.size() - 1) {
      bound /= vectorSize;
    }
    APInt size = GreatestCommonDivisor(APInt(64, uint64_t(remainingGroupSize)),
                                       APInt(64, uint64_t(bound)));
    reductionTileSizes[dim] = size.getSExtValue();
    if (i == reductionDims.size() - 1) {
      reductionTileSizes[dim] *= vectorSize;
    }
    remainingGroupSize /= size.getSExtValue();
  }

  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(workgroupTileSizes)); // Workgroup level
  tileSizes.emplace_back(std::move(reductionTileSizes)); // reduction level
  if (failed(setOpConfigAndEntryPointFnTranslation(
          op->getParentOfType<mlir::FunctionOpInterface>(), op, tileSizes,
          CodeGenPipeline::SPIRVSubgroupReduce, workgroupSize))) {
    return failure();
  }

  // Set lowering configuration to drive tiling for other Linalg ops too---the
  // pipeline expects it.
  op->getParentOfType<FunctionOpInterface>().walk([&](linalg::LinalgOp op) {
    setLoweringConfig(
        op, IREE::Codegen::LoweringConfigAttr::get(op.getContext(), tileSizes));
  });
  return success();
}

//===----------------------------------------------------------------------===//
// Everything Default Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setDefaultOpConfig(IREE::GPU::TargetAttr target,
                                        Operation *op,
                                        bool allowVectorization = true) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce as default op...\n");
  auto funcOp = op->getParentOfType<mlir::FunctionOpInterface>();
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);

  // Special case for not tiled ops.
  if (partitionedLoops.empty()) {
    // No tiled loops means we cannot tile (and distribute) at all. Use just one
    // single thread to run everything.
    auto pipeline = CodeGenPipeline::SPIRVBaseDistribute;
    std::array<int64_t, 3> workgroupSize = {1, 1, 1};
    return setOpConfigAndEntryPointFnTranslation(
        funcOp, op, TileSizesListType{}, pipeline, workgroupSize);
  }

  int subgroupSize = target.getPreferredSubgroupSize();
  const unsigned loopDepth = partitionedLoops.back() + 1;

  // Configurations we need to decide.
  std::array<int64_t, 3> workgroupSize;
  SmallVector<int64_t> workgroupTileSizes;
  SmallVector<int64_t> threadTileSizes;

  // Initialize the configuration.
  auto initConfiguration = [&]() {
    workgroupSize = {subgroupSize, 1, 1};
    workgroupTileSizes.resize(loopDepth, 0);
    threadTileSizes.resize(loopDepth, 0);

    // Initialize tiling along all partitioned loops with size 1.
    for (int64_t loopIndex : partitionedLoops) {
      workgroupTileSizes[loopIndex] = threadTileSizes[loopIndex] = 1;
    }
    // Override the innermost dimension to distribute to threads in a subgroup.
    workgroupTileSizes.back() = subgroupSize;
    threadTileSizes.back() = 1;
  };

  // Special case for non-linalg ops.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp.getNumDpsInits() != 1) {
    auto pipeline = CodeGenPipeline::SPIRVBaseDistribute;

    initConfiguration();
    TileSizesListType tileSizes;
    tileSizes.push_back(workgroupTileSizes);
    tileSizes.push_back(threadTileSizes);

    return setOpConfigAndEntryPointFnTranslation(funcOp, op, tileSizes,
                                                 pipeline, workgroupSize);
  }

  // Common case for all linalg ops.

  // The core idea is to distribute the partitioned loops to the workgroup
  // dimensions. The goal is to fill up the GPU as much as possible, which means
  // 1) distributing to as many threads as possible, and 2) avoid assigning too
  // many threads to handle out-of-bound elements (thus idle).

  auto elementHasPowerOfTwoBitwidth = [](Value operand) {
    Type elementType = getElementTypeOrSelf(operand.getType());
    return isa<IntegerType, FloatType>(elementType) &&
           llvm::isPowerOf2_64(IREE::Util::getTypeBitWidth(elementType));
  };

  // Whether we can try to use the vectorization pipeline.
  SmallVector<int64_t> loopBounds = linalgOp.getStaticLoopRanges();
  bool vectorizable =
      allowVectorization &&
      // The vectorization pipeline assumes tensor semantics for tiling.
      linalgOp.hasPureTensorSemantics() && !linalgOp.hasIndexSemantics() &&
      // Require all affine maps to be projected permutation so that we can
      // generate vector transfer ops.
      llvm::all_of(
          linalgOp.getIndexingMapsArray(),
          [](AffineMap map) { return map.isProjectedPermutation(); }) &&
      llvm::all_of(linalgOp->getOperands(), elementHasPowerOfTwoBitwidth) &&
      llvm::none_of(loopBounds, ShapedType::isDynamic);

  const unsigned minBitwidth = getMinElementBitwidth(linalgOp);
  // Make sure we use a tile size that results in some integral number of bytes.
  const unsigned scaleToByte = minBitwidth < 8 ? 8 / minBitwidth : 1;

  // Distribute workload to the given `numThreads` by allowing a potential loss.
  auto distributeToThreads = [&](int64_t numThreads,
                                 std::optional<int64_t> lossFactor =
                                     std::nullopt) {
    LLVM_DEBUG(llvm::dbgs() << "\nLoss factor: " << lossFactor << "\n");
    initConfiguration();
    // If there are more than 3 parallel dim try to tile the extra higher level
    // dimensions to 1 for extra dimensions.
    if (isa<linalg::GenericOp>(linalgOp.getOperation())) {
      for (int64_t i = 0, e = workgroupTileSizes.size(); i < e; i++) {
        if (workgroupTileSizes[i] != 0) {
          break;
        }
        if (loopBounds[i] != 1) {
          workgroupTileSizes[i] = 1;
        }
      }
    }
    // Scan from the innermost shape dimension and try to deduce the
    // configuration for the corresponding GPU workgroup dimension.
    int64_t wgDim = 0;
    for (auto shapeDim : llvm::reverse(partitionedLoops)) {
      int64_t loopBound = loopBounds[shapeDim];
      // Skip dynamic dimensions.
      if (ShapedType::isDynamic(loopBound)) {
        continue;
      }

      // Try to find some power of two that can divide the current shape dim
      // size. This vector keeps the candidate tile sizes.
      SmallVector<int64_t, 8> candidates;

      // For the inner most workgroup dim, try to see if we can have 4
      // elements per thread. This enables vectorization.
      if (vectorizable && wgDim == 0 && !lossFactor) {
        candidates.push_back(4 * numThreads);
      }
      // Try all power of two numbers upto the subgroup size.
      for (unsigned i = numThreads; i >= 1; i >>= 1) {
        candidates.push_back(i);
      }
      LLVM_DEBUG(llvm::dbgs() << "Base candidate tile sizes: "
                              << llvm::interleaved_array(candidates) << "\n");

      for (int64_t candidate : candidates) {
        int64_t scaledTileSize = candidate * scaleToByte;
        if (loopBound % scaledTileSize != 0) {
          // IREE_METAL_COOP_VEC_NONMULT (opt-in): for a non-divisible INNERMOST dim
          // (e.g. transpose/elementwise over a prime seq=577), the divisibility
          // gate otherwise rejects every vectorizable tile and falls to tile-1
          // (scalar) — measured ~2.7x slower. Accept the vectorizable candidate
          // (4*numThreads, %4==0) anyway and let IREE tile-with-remainder +
          // vectorize the main body. Gated + correctness-validated before default.
          bool forceVec = !getenv("IREE_METAL_COOP_NO_VEC_NONMULT") && vectorizable &&
                          wgDim == 0 && !lossFactor && candidate % 4 == 0;
          if (!forceVec) {
            if (!lossFactor) {
              continue;
            }
            // Skip this candidate if it causes many threads to be idle.
            int64_t idleThreads = candidate - (loopBound % scaledTileSize);
            if (idleThreads > candidate / *lossFactor) {
              continue;
            }
          }
        }
        // If the workload is too small and we cannot distribute to more than 2
        // workgroups, try a smaller tile size to increase parallelism.
        if (partitionedLoops.size() == 1 && candidate > subgroupSize &&
            divideCeil(loopBound, scaledTileSize) <= 2) {
          continue;
        }

        // Found a suitable candidate. Try to let each thread handle 4
        // elements if this is the workgroup x dimension.
        workgroupTileSizes[shapeDim] = scaledTileSize;
        LLVM_DEBUG(llvm::dbgs()
                   << "Chosen workgroup tile size: " << scaledTileSize << "\n");
        if (vectorizable && wgDim == 0 && !lossFactor && candidate % 4 == 0) {
          // Use size-1 vectors to increase parallelism if larger ones causes
          // idle threads in the subgroup.
          bool hasIdleThreads =
              partitionedLoops.size() == 1 && candidate <= subgroupSize;
          int vectorSize = hasIdleThreads ? 1 : 4;
          LLVM_DEBUG(llvm::dbgs() << "Use vector size: " << vectorSize << "\n");
          threadTileSizes[shapeDim] = vectorSize * scaleToByte;
          workgroupSize[wgDim] = candidate / vectorSize;
          assert(numThreads % (candidate / vectorSize) == 0);
          numThreads /= candidate / vectorSize;
        } else {
          if (wgDim == 0) {
            vectorizable = false;
          }
          threadTileSizes[shapeDim] = scaleToByte;
          workgroupSize[wgDim] = candidate;
          assert(numThreads % candidate == 0);
          numThreads /= candidate;
        }
        assert(numThreads >= 1);
        break;
      }

      // Stop if we have distributed all threads.
      if (numThreads == 1) {
        break;
      }
      wgDim++;
    }
    return numThreads;
  };

  // First try to see if we can use up all threads without any loss.
  if (distributeToThreads(subgroupSize) != 1) {
    // Otherwise, allow larger and larger loss factor.

    // Threads for distribution. Use 32 at least.
    int64_t numThreads = std::max(subgroupSize, 32);
    // We can tolerate (1 / lossFactor) of threads in the workgroup to be idle.
    int64_t lossFactor = 32;

    for (; lossFactor >= 1; lossFactor >>= 1) {
      if (distributeToThreads(numThreads, lossFactor) == 1) {
        break;
      }
    }
  }

  auto pipeline = vectorizable ? CodeGenPipeline::SPIRVBaseVectorize
                               : CodeGenPipeline::SPIRVBaseDistribute;

  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(threadTileSizes);

  if (vectorizable) {
    // Try to tile all reductions by some small factor, preferably 4, when
    // possible. This gives us a chance to perform vector4 load if an input has
    // its innnermost dimension being reduction. It also avoids generating too
    // many instructions when unrolling vector later.
    SmallVector<int64_t> loopTileSizes(linalgOp.getNumLoops(), 0);
    for (const auto &[i, iter] :
         llvm::enumerate(linalgOp.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(iter) || i >= workgroupTileSizes.size() ||
          workgroupTileSizes[i] == 0) {
        loopTileSizes[i] = getReductionTilingFactor(loopBounds[i]);
      }
    }
    if (llvm::any_of(loopTileSizes, [](int64_t s) { return s != 0; })) {
      tileSizes.push_back(loopTileSizes);
    }
  }

  return setOpConfigAndEntryPointFnTranslation(funcOp, op, tileSizes, pipeline,
                                               workgroupSize);
}

//===----------------------------------------------------------------------===//
// Configuration Dispatcher
//===----------------------------------------------------------------------===//

/// Sets the CodeGen configuration as attributes to the given `rootOp` if it's a
/// known Linalg matmul/convolution op with good configurations.
// iree-metal: does the dispatch emit a TRANSPOSED parallel output — a linalg op whose init
// indexing map is a permutation but not the identity (e.g. a fused weight-grad store [N,M]
// alongside an [M,N] reduction)? The SPIRVSubgroupReduce pipeline mis-distributes such a
// transpose write under the reduction tiling, so a reduction dispatch that also transposes
// must NOT take the subgroup-reduce path (see the fused LayerNorm-backward over/under-count
// root-caused 2026-07-17). A projected-permutation reduction output like (d0,d1)->(d0) is NOT
// flagged (results < dims => not a full permutation), so ordinary reductions keep the fast path.
static bool dispatchHasTransposeOutput(mlir::FunctionOpInterface funcOp) {
  bool hasTranspose = false;
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    for (OpOperand &init : linalgOp.getDpsInitsMutable()) {
      AffineMap m = linalgOp.getMatchingIndexingMap(&init);
      if (m.isPermutation() && !m.isIdentity()) {
        hasTranspose = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return hasTranspose;
}

static LogicalResult setSPIRVOpConfig(IREE::GPU::TargetAttr target,
                                      mlir::FunctionOpInterface entryPointFn,
                                      Operation *rootOp) {
  // First try to find a proper CodeGen configuration to tile and vectorize for
  // the current target architecture.
  if (target.isAMD() &&
      succeeded(detail::setAMDCodeGenConfig(target, rootOp))) {
    return success();
  }
  if (target.isApple() &&
      succeeded(detail::setAppleCodeGenConfig(target, rootOp))) {
    return success();
  }
  if (target.isARM() &&
      succeeded(detail::setMaliCodeGenConfig(target, rootOp))) {
    return success();
  }
  if (target.isNVIDIA() &&
      succeeded(detail::setNVIDIACodeGenConfig(target, rootOp))) {
    return success();
  }
  if (target.isQualcomm() &&
      succeeded(detail::setAdrenoCodeGenConfig(target, rootOp))) {
    return success();
  }

  // Otherwise fallback to use a default configuration that tiles and
  // distributes/vectorizes.
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::MatmulOp, linalg::BatchMatmulOp>([](auto op) {
        // Assertion is better than returning failure here to
        // avoid unexpected configurations.
        assert(false && "named matmul not supported here, pass expects it to "
                        "generalized first");
        return op->emitOpError(
            "named matmul not supported, expected to be generalized first");
      })
      .Case([target](linalg::ConvolutionOpInterface op) {
        // Use the result type in case of larger bitwidth for accumulators.
        auto type = cast<ShapedType>(op->getResult(0).getType());
        const int bitwidth = type.getElementTypeBitWidth();
        if (bitwidth <= 32) {
          const int multiplier = 32 / bitwidth;
          const int bestTilingFactor = 32 * multiplier;
          const int subgroupSize = 32;
          auto result = detail::setConvOpConfig(cast<linalg::LinalgOp>(*op),
                                                subgroupSize, bestTilingFactor);
          if (succeeded(result)) {
            return success();
          }
        }
        // If unsuccessful, try to tile and distribute/vectorize.
        return setDefaultOpConfig(target, op);
      })
      .Case([&](linalg::GenericOp op) {
        LLVM_DEBUG(llvm::dbgs() << "configuring for generic op\n");
        if (succeeded(detail::setTilingAndMatmulOpConfig(op, target))) {
          return success();
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "failed to set matmul op config, trying reduction\n");

        // iree-metal: IREE_METAL_REDUCE_SKIP_TRANSPOSE = experiment (REJECTED, kept opt-in): skip the
        // subgroup-reduce pipeline for a dispatch that also emits a transposed parallel output.
        // The intent was to dodge the fused-LN-backward reduction+transpose mis-tile, but the
        // fallback (default distribute pipeline) overflows Metal stack on the 4096x768 reduction
        // ("Compute function exceeds available stack space"). Neither pipeline handles reduction
        // +transpose co-located -> the real fix is at dispatch FORMATION (don't co-locate them).
        if (!(getenv("IREE_METAL_REDUCE_SKIP_TRANSPOSE") &&
              dispatchHasTransposeOutput(entryPointFn)) &&
            succeeded(setReductionConfig(target, op))) {
          return success();
        }
        LLVM_DEBUG(llvm::dbgs() << "failed to set reduction op config");

        // If a generic op has reduction iterator types, it can be treated as a
        // root op for configuration as well. Use the default configuration,
        // which will mark it as a root.
        if (op.getNumLoops() != op.getNumParallelLoops()) {
          LLVM_DEBUG(llvm::dbgs() << "trying default config for generic");
          return setDefaultOpConfig(target, op);
        }

        LLVM_DEBUG(llvm::dbgs() << "failed to set config of generic");
        return failure();
      })
      .Case([target](IREE::LinalgExt::FftOp op) {
        return setFftOpConfig(target, op);
      })
      .Case([target](IREE::LinalgExt::ScatterOp op) {
        return setScatterOpConfig(target, op);
      })
      .Case([target](IREE::LinalgExt::AttentionOp op) {
        return setAttentionOpConfig(target, op);
      })
      .Case<IREE::LinalgExt::WinogradInputTransformOp,
            IREE::LinalgExt::WinogradOutputTransformOp>(
          [&](auto op) { return setWinogradOpConfig(target, op); })
      .Default(failure());
};

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

/// Find the root operation for the dispatch. The root is the op that will be
/// tiled and distributed to workgroups; all other ops fuse with it as producers
/// or consumers.
///
/// Priority (all passes iterate in reverse to prefer later ops):
///   1. Named ops (matmul, conv) or generics with reduction iterators.
///   2. Any generic op (elementwise).
///   3. Fill ops.
static Operation *getRootOperation(ArrayRef<Operation *> computeOps) {
  Operation *rootOperation = nullptr;

  // Pass 1: named ops or generics with reductions.
  for (Operation *op : llvm::reverse(computeOps)) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
        rootOperation = op;
        break;
      }
      continue;
    }
    if (!isa<linalg::FillOp>(op) && isa<TilingInterface>(op)) {
      rootOperation = op;
      break;
    }
  }

  // Pass 2: any generic op (elementwise).
  if (!rootOperation) {
    for (Operation *op : llvm::reverse(computeOps)) {
      if (isa<linalg::GenericOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  // Pass 3: fill ops.
  if (!rootOperation) {
    for (Operation *op : llvm::reverse(computeOps)) {
      if (isa<linalg::FillOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  return rootOperation;
}

static LogicalResult setConfigForKernel(IREE::GPU::TargetAttr target,
                                        mlir::FunctionOpInterface funcOp) {
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  if (computeOps.empty()) {
    // No compute operations found. Allow to pass through without a config.
    return success();
  }

  Operation *rootOp = getRootOperation(computeOps);
  if (!rootOp) {
    return computeOps.back()->emitOpError(
        "unable to find root operation in dispatch");
  }

  if (succeeded(setSPIRVOpConfig(target, funcOp, rootOp))) {
    return success();
  }

  if (succeeded(setDefaultOpConfig(target, rootOp))) {
    return success();
  }

  return rootOp->emitOpError(
      "without known roots, the last compute operation in the tiled "
      "loop body is expected to be set as root");
}

LogicalResult initSPIRVLaunchConfig(FunctionOpInterface funcOp) {
  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target) {
    return funcOp.emitError("missing GPU target in #hal.executable.target");
  }

  if (getTranslationInfo(funcOp)) {
    return success();
  }

  if (auto exportOp = getEntryPoint(funcOp)) {
    // If no translation info set, first check whether we already have workgroup
    // count set--it's a "contract" to indicate that we should bypass all tiling
    // and distribution to go down just the most basic lowering flow.
    if (Block *body = exportOp->getWorkgroupCountBody()) {
      auto retOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
      // For scalar dispatch cases--using just one thread of one workgroup.
      auto isOne = [](Value value) { return matchPattern(value, m_One()); };
      if (llvm::all_of(retOp.getOperands(), isOne)) {
        std::array<int64_t, 3> workgroupSize = {1, 1, 1};
        auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
            funcOp.getContext(), CodeGenPipeline::SPIRVBaseLowering,
            workgroupSize);
        return setTranslationInfo(funcOp, translationInfo);
      }
    }
  }

  if (failed(setConfigForKernel(target, funcOp))) {
    return failure();
  }

  return success();
}

} // namespace mlir::iree_compiler
