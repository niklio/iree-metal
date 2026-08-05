// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: env -u IREE_METAL_RED_NONMULT_WG -u IREE_METAL_COOP_MNT \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s

#row = affine_map<(d0, d1) -> (d0, d1)>
#row_result = affine_map<(d0, d1) -> (d0)>

// CHECK-DAG: #[[REDUCTION_CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[\[1\], \[0, 512\]\]}}>
// CHECK-DAG: #[[REDUCTION_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [512, 1, 1]>
// CHECK-DAG: #[[SPLIT_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize
// CHECK-DAG: #[[VOCAB_CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[\[64, 128\], \[4, 8\], \[0, 0, 8\]\]}}>
// CHECK-DAG: #[[VOCAB_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [16, 16, 1]>
// CHECK-LABEL: func.func @vocabulary_sum
// CHECK-SAME: attributes {translation_info = #[[REDUCTION_TRANSLATION]]}
// CHECK: linalg.generic {indexing_maps = {{.*}}lowering_config = #[[REDUCTION_CONFIG]]
func.func @vocabulary_sum(%input: tensor<4096x50257xf32>)
    -> tensor<4096xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4096xf32>
  %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<4096xf32>) -> tensor<4096xf32>
  %result = linalg.generic {
      indexing_maps = [#row, #row_result],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<4096x50257xf32>)
      outs(%init : tensor<4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
  } -> tensor<4096xf32>
  return %result : tensor<4096xf32>
}

// A partial-reduction matmul cannot use the cooperative-matrix pipeline: its
// additional partial-result dimension is not representable by VectorToGPU.
// CHECK-LABEL: func.func @split_vocabulary_matmul
// CHECK-SAME: attributes {translation_info = #[[SPLIT_TRANSLATION]]}
func.func @split_vocabulary_matmul(
    %lhs: tensor<128x6256xbf16>, %rhs: tensor<6256x768xbf16>,
    %init: tensor<128x768xf32>) -> tensor<128x768xf32> {
  %result = linalg.matmul {
      iree_linalg_ext.split_reduction = [6256 : index]}
      ins(%lhs, %rhs : tensor<128x6256xbf16>, tensor<6256x768xbf16>)
      outs(%init : tensor<128x768xf32>) -> tensor<128x768xf32>
  return %result : tensor<128x768xf32>
}

// The extreme forward-vocabulary orientation exceeds the cooperative staging
// budget and must retain the tuned 16x16 vector fallback rather than failing
// configuration or exceeding Apple threadgroup memory.
// CHECK-LABEL: func.func @full_vocabulary_matmul
// CHECK-SAME: attributes {translation_info = #[[VOCAB_TRANSLATION]]}
// CHECK: linalg.matmul {lowering_config = #[[VOCAB_CONFIG]]}
func.func @full_vocabulary_matmul(
    %lhs: tensor<4096x768xbf16>, %rhs: tensor<768x50304xbf16>,
    %init: tensor<4096x50304xf32>) -> tensor<4096x50304xf32> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<4096x768xbf16>, tensor<768x50304xbf16>)
      outs(%init : tensor<4096x50304xf32>) -> tensor<4096x50304xf32>
  return %result : tensor<4096x50304xf32>
}
