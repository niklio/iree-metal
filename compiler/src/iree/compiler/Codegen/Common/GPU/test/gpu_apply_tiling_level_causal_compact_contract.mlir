// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: env IREE_METAL_CAUSAL_TRIANGULAR_GRID=1 \
// RUN:   iree-opt --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level{shorten-causal-attention-backward-reductions=true}))" \
// RUN:   %s | FileCheck %s

#score = affine_map<(b, m, n, k) -> (b, m, k)>
#rhs = affine_map<(b, m, n, k) -> (b, k, n)>
#out = affine_map<(b, m, n, k) -> (b, m, n)>

func.func @compact_dq_bounds(
    %scores: tensor<1x512x512xbf16>, %rhs: tensor<1x512x64xbf16>,
    %init: tensor<1x64x64xf32>, %m0: index)
    -> tensor<1x64x64xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %result = scf.for %k = %c0 to %c512 step %c128
      iter_args(%acc = %init) -> tensor<1x64x64xf32> {
    %score_tile = tensor.extract_slice %scores[0, %m0, %k] [1, 64, 128]
        [1, 1, 1] : tensor<1x512x512xbf16> to tensor<1x64x128xbf16>
    %rhs_tile = tensor.extract_slice %rhs[0, %k, 0] [1, 128, 64]
        [1, 1, 1] : tensor<1x512x64xbf16> to tensor<1x128x64xbf16>
    %next = linalg.generic {
        indexing_maps = [#score, #rhs, #out],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"],
        iree_codegen.apple_attention_backward_causal,
        iree_codegen.apple_attention_backward_causal_score_alignment = 128 : i64,
        iree_codegen.apple_attention_backward_causal_score_workgroup_aligned,
        iree_codegen.apple_attention_backward_role = "dq_attrs"}
        ins(%score_tile, %rhs_tile :
            tensor<1x64x128xbf16>, tensor<1x128x64xbf16>)
        outs(%acc : tensor<1x64x64xf32>) {
      ^bb0(%lhs: bf16, %rhs_value: bf16, %old: f32):
        %lhs_f32 = arith.extf %lhs : bf16 to f32
        %rhs_f32 = arith.extf %rhs_value : bf16 to f32
        %product = arith.mulf %lhs_f32, %rhs_f32 : f32
        %sum = arith.addf %old, %product : f32
        linalg.yield %sum : f32
    } -> tensor<1x64x64xf32>
    scf.yield %next : tensor<1x64x64xf32>
  }
  return %result : tensor<1x64x64xf32>
}

// CHECK-LABEL: func.func @compact_dq_bounds(
// CHECK-SAME: %[[M0:[A-Za-z0-9_]+]]: index
// CHECK: %[[END:.+]] = arith.addi %[[M0]], %{{.*}} : index
// CHECK: %[[TILES:.+]] = arith.ceildivui %[[END]], %{{.*}} : index
// CHECK: %[[ROUNDED:.+]] = arith.muli %[[TILES]], %{{.*}} : index
// CHECK: %[[UB:.+]] = arith.minui %[[ROUNDED]], %{{.*}} : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[UB]] step %{{.*}}
