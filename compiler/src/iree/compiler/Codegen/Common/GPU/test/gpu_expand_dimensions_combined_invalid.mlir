// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-opt %s --verify-diagnostics \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-expand-dimensions))"

#combined = #iree_gpu.lowering_config<{
  expand_dims = #iree_gpu.expand_dims<
      [[0], [1], [2, 3]], output_shape = [?, ?, ?, 4]>,
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>
}>

// The ordinary early invocation must reject this combination before it can
// consume expand_dims and erase the evidence seen by the late physical pass.
func.func @reject_physical_with_ordinary_expand_dims(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<32x48xf32>) -> tensor<32x48xf32> {
  // expected-error @+1 {{Apple physical fragments cannot be combined with expand_dims}}
  %result = linalg.matmul {lowering_config = #combined}
      ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}
