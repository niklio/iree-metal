// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: env -u IREE_METAL_COOP_SG -u IREE_METAL_COOP_MNT \
// RUN:   -u IREE_METAL_COOP_PD -u IREE_METAL_COOP_KT \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s

// CHECK-DAG: #[[TEXT_CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[\[8, 1024\], \[4, 8\], \[0, 0, 8\]\]}}>
// CHECK-DAG: #[[VISION_CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[\[64, 64\], \[32, 32\], \[0, 0, 32\], \[16, 16, 16\]\]}}>
// CHECK-DAG: #[[TEXT_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [128, 2, 1]>
// CHECK-DAG: #[[VISION_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVCooperativeMatrixVectorize workgroup_size = [64, 2, 1] subgroup_size = 32, {pipeline_depth = 0 : i64, store_stage = 0 : i64}>
// CHECK-LABEL: func.func @aligned_text_ffn
// CHECK-SAME: attributes {translation_info = #[[TEXT_TRANSLATION]]}
// CHECK: linalg.matmul {lowering_config = #[[TEXT_CONFIG]]}
func.func @aligned_text_ffn(
    %lhs: tensor<1024x768xbf16>, %rhs: tensor<768x3072xbf16>,
    %init: tensor<1024x3072xf32>) -> tensor<1024x3072xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<1024x768xbf16>, tensor<768x3072xbf16>)
      outs(%init : tensor<1024x3072xf32>) -> tensor<1024x3072xf32>
  return %result : tensor<1024x3072xf32>
}

// ViT's M64-padded odd-token FFN must retain the lower-power fallback.
// CHECK-LABEL: func.func @padded_vit_ffn
// CHECK-SAME: attributes {translation_info = #[[VISION_TRANSLATION]]}
// CHECK: linalg.matmul {lowering_config = #[[VISION_CONFIG]]}
func.func @padded_vit_ffn(
    %lhs: tensor<4672x768xbf16>, %rhs: tensor<768x3072xbf16>,
    %init: tensor<4672x3072xf32>) -> tensor<4672x3072xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<4672x768xbf16>, tensor<768x3072xbf16>)
      outs(%init : tensor<4672x3072xf32>) -> tensor<4672x3072xf32>
  return %result : tensor<4672x3072xf32>
}
