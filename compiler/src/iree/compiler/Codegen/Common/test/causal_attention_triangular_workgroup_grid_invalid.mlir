// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: not env IREE_METAL_CAUSAL_TRIANGULAR_GRID=1 IREE_METAL_CAUSAL_BWD_BOUNDS=1 iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op))" %s 2>&1 | FileCheck %s

// CHECK: error: 'linalg.matmul' op causal triangular workgroup grid requires a positive score alignment contract

func.func @missing_score_alignment(
    %lhs: tensor<512x64xbf16>, %rhs: tensor<64x512xbf16>,
    %init: tensor<512x512xf32>) -> tensor<512x512xf32>
    attributes {
      translation_info = #iree_codegen.translation_info<
          pipeline = SPIRVAppleVectorDistributeAttention>} {
  %result = linalg.matmul {
      iree_codegen.apple_attention_backward_causal_score,
      iree_codegen.apple_attention_backward_role = "qk_attrs",
      lowering_config =
          #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0]]>}
      ins(%lhs, %rhs : tensor<512x64xbf16>, tensor<64x512xbf16>)
      outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %result : tensor<512x512xf32>
}
