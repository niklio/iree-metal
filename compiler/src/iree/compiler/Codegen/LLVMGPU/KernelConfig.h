// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_KERNELCONFIG_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_KERNELCONFIG_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

LogicalResult initGPULaunchConfig(mlir::FunctionOpInterface funcOp);

// Configures a standalone contraction for intrinsic-based vector
// distribution. SPIR-V targets can override the pipeline, and Apple callers
// can restrict selection to first-class simdgroup intrinsics.
LogicalResult setMatmulVectorDistributionConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    linalg::LinalgOp op,
    IREE::Codegen::DispatchLoweringPassPipeline pipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorDistribute,
    bool appleSimdgroupOnly = false);

// iree-metal (attn vdist port): exposed so the metal-spirv attention pipeline
// can reuse the LLVMGPU intrinsic-based vector-distribute attention config. The
// pipeline param lets the caller route to SPIRVVectorDistributeAttention
// instead of the default LLVMGPU pipeline.
LogicalResult setAttentionIntrinsicBasedVectorDistributionConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    IREE::LinalgExt::AttentionOp op,
    IREE::Codegen::DispatchLoweringPassPipeline pipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorDistribute);

} // namespace mlir::iree_compiler
#endif // IREE_COMPILER_CODEGEN_LLVMGPU_KERNELCONFIG_H_
