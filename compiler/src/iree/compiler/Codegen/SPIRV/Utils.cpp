// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/Utils.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {

bool usesSPIRVCodeGen(IREE::HAL::ExecutableVariantOp variantOp) {
  if (variantOp.getObjects().has_value()) {
    // Variants containing external executables do not go through CodeGen.
    return false;
  }

  DictionaryAttr configuration = variantOp.getTargetAttr().getConfiguration();
  // The spirv.target_env attribute is attached if going down SPIR-V CodeGen
  // pipelines. Later we turn spirv.target_env into iree.spirv.features after
  // materializing device queries.
  return configuration.contains(spirv::getTargetEnvAttrName()) ||
         configuration.contains("iree.spirv.features");
}

const char *getSPIRVDistributeAttrName() { return "iree.spirv.distribute_dim"; }

DictionaryAttr getTargetConfigAttr(Operation *op) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!targetAttr) {
    return nullptr;
  }
  return targetAttr.getConfiguration();
}

bool usesIndirectBindingsAttr(Operation *op) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  return targetAttr ? targetAttr.getFormat().getValue().ends_with("-ptr")
                    : false;
}

FailureOr<SmallVector<int64_t>>
getSPIRVTileSize(mlir::FunctionOpInterface funcOp, int tilingLevel) {
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<IREE::Codegen::LoweringConfigAttr> config =
      getFirstLoweringConfig<IREE::Codegen::LoweringConfigAttr>(computeOps);
  if (failed(config)) {
    return funcOp.emitOpError("failed to get lowering configuration");
  }

  return config->getTileSizeVals(tilingLevel);
}

FailureOr<linalg::TileSizeComputationFunction>
getSPIRVTileSizeComputeFn(mlir::FunctionOpInterface funcOp, int tilingLevel) {
  auto tileSizes = getSPIRVTileSize(funcOp, tilingLevel);
  if (failed(tileSizes)) {
    return failure();
  }

  // iree-metal (IREE_METAL_SCATTER_WINDOW_WORKGROUPS): non-unique scatters
  // keep their leading update loops serial, but their trailing update-window
  // loops can be safely distributed because they address disjoint output
  // slices. When such a scatter is initialized by a fused fill, tile only that
  // fill's innermost loop to one element per invocation. Applying the scatter's
  // thread tile sizes to every producer instead would also distribute index
  // normalization across reduction dimensions without a synchronization
  // barrier.
  SmallVector<Operation *> laneTiledScatterFills;
  if (tilingLevel == 1) {
    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    FailureOr<Operation *> configOp =
        getLoweringConfigCarryingOp<IREE::Codegen::LoweringConfigAttr>(
            computeOps);
    if (succeeded(configOp)) {
      auto scatterOp = dyn_cast<IREE::LinalgExt::ScatterOp>(*configOp);
      if (scatterOp && !scatterOp.getUniqueIndices() &&
          scatterOp->hasAttrOfType<UnitAttr>(
              "iree_codegen.apple_scatter_window_workgroups")) {
        if (auto fillOp =
                scatterOp.getOriginal().getDefiningOp<linalg::FillOp>()) {
          laneTiledScatterFills.push_back(fillOp);
        }
        for (Operation *candidate : computeOps) {
          auto fillOp = dyn_cast<linalg::FillOp>(candidate);
          if (fillOp &&
              fillOp.getOutputs().front() == scatterOp.getOriginal()) {
            laneTiledScatterFills.push_back(fillOp);
          }
        }
      }
    }
  }
  linalg::TileSizeComputationFunction computeFn =
      [tileSizes, laneTiledScatterFills](OpBuilder &builder, Operation *op) {
        if (llvm::is_contained(laneTiledScatterFills, op)) {
          int64_t numLoops =
              cast<TilingInterface>(op).getLoopIteratorTypes().size();
          SmallVector<Value> fillTileSizes;
          fillTileSizes.reserve(numLoops);
          for (int64_t i = 0; i < numLoops; ++i) {
            int64_t size = i == numLoops - 1 ? 1 : 0;
            fillTileSizes.push_back(arith::ConstantIndexOp::create(
                builder, op->getLoc(), size));
          }
          return fillTileSizes;
        }
        auto range = llvm::map_range(*tileSizes, [&](int64_t size) -> Value {
          return arith::ConstantIndexOp::create(builder, op->getLoc(), size);
        });
        return llvm::to_vector(range);
      };
  return computeFn;
}

FailureOr<scf::SCFTileSizeComputationFunction>
getSPIRVScfTileSizeComputeFn(mlir::FunctionOpInterface funcOp,
                             int tilingLevel) {
  FailureOr<SmallVector<int64_t>> tileSizes =
      getSPIRVTileSize(funcOp, tilingLevel);
  if (failed(tileSizes)) {
    return failure();
  }
  scf::SCFTileSizeComputationFunction computeFn =
      [tileSizes](OpBuilder &builder,
                  Operation *op) -> SmallVector<OpFoldResult> {
    auto tileSizesOfr = getAsIndexOpFoldResult(op->getContext(), *tileSizes);
    auto zeroAttr = builder.getIndexAttr(0);
    int numLoops = cast<TilingInterface>(op).getLoopIteratorTypes().size();
    tileSizesOfr.resize(numLoops, zeroAttr);
    return tileSizesOfr;
  };
  return computeFn;
}

template <typename GPUIdOp, typename GPUCountOp>
static linalg::ProcInfo
getGPUProcessorIdAndCountImpl(OpBuilder &builder, Location loc, unsigned dim) {
  assert(dim < kNumGPUDims && "processor index out of range!");

  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  Type indexType = builder.getIndexType();
  return {GPUIdOp::create(builder, loc, indexType, dimAttr[dim]),
          GPUCountOp::create(builder, loc, indexType, dimAttr[dim]),
          linalg::DistributionMethod::Cyclic};
}

template <typename GPUIdOp, typename GPUCountOp>
static SmallVector<linalg::ProcInfo, 2>
getGPUProcessorIdsAndCountsImpl(OpBuilder &builder, Location loc,
                                unsigned numDims) {
  SmallVector<linalg::ProcInfo, 2> procInfo(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    procInfo[numDims - 1 - i] =
        getGPUProcessorIdAndCountImpl<GPUIdOp, GPUCountOp>(builder, loc, i);
  }
  return procInfo;
}

template <typename GPUIdOp, typename GPUCountOp>
SmallVector<linalg::ProcInfo, 2> getGPUProcessorIdsAndCounts(OpBuilder &builder,
                                                             Location loc,
                                                             unsigned numDims) {
  return getGPUProcessorIdsAndCountsImpl<GPUIdOp, GPUCountOp>(builder, loc,
                                                              numDims);
}

/// Explicit instantiation of gpuGPUProcessorIdsAndCounts.
template SmallVector<linalg::ProcInfo, 2>
getGPUProcessorIdsAndCounts<gpu::ThreadIdOp, gpu::BlockDimOp>(
    OpBuilder &builder, Location loc, unsigned numDims);

} // namespace mlir::iree_compiler
