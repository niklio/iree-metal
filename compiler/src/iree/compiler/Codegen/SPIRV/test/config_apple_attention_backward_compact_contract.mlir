// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=ON
// RUN: not env IREE_METAL_DISABLE_NATIVE_ATTENTION=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLED

#score = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

func.func @compact_dq(
    %score: tensor<128x128xbf16>, %rhs: tensor<128x64xbf16>,
    %init: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %result = linalg.generic {
      indexing_maps = [#score, #rhs, #out],
      iterator_types = ["parallel", "parallel", "reduction"],
      iree_codegen.apple_attention_backward_causal,
      iree_codegen.apple_attention_backward_causal_score_alignment = 128 : i64,
      iree_codegen.apple_attention_backward_role = "dq_attrs"}
      ins(%score, %rhs : tensor<128x128xbf16>, tensor<128x64xbf16>)
      outs(%init : tensor<128x64xf32>) {
    ^bb0(%lhs: bf16, %rhs_value: bf16, %old: f32):
      %lhs_f32 = arith.extf %lhs : bf16 to f32
      %rhs_f32 = arith.extf %rhs_value : bf16 to f32
      %product = arith.mulf %lhs_f32, %rhs_f32 : f32
      %sum = arith.addf %old, %product : f32
      linalg.yield %sum : f32
  } -> tensor<128x64xf32>
  return %result : tensor<128x64xf32>
}

// ON-DAG: #[[APPLE:.+]] = #iree_codegen.translation_info<pipeline = SPIRVAppleVectorDistributeAttention
// ON-LABEL: func.func @compact_dq(
// ON-SAME: translation_info = #[[APPLE]]
// ON: iree_codegen.apple_attention_backward_causal_score_alignment = 128 : i64
// ON-SAME: iree_codegen.apple_attention_backward_causal_score_workgroup_aligned

// DISABLED: error: 'linalg.generic' op compact causal score grid requires the native Apple attention pipeline
