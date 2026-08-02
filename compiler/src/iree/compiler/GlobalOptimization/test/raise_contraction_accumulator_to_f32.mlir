// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: env -u IREE_METAL_COOP_NO_PAD -u IREE_METAL_COOP_NO_ATTN_ISOLATE -u IREE_METAL_COOP_NO_ATTN_ISOLATE_BF16 -u IREE_METAL_FFN_PAD_M64 iree-opt --iree-global-opt-raise-contraction-accumulator-to-f32 %s | FileCheck %s --check-prefix=DEFAULT
// RUN: env -u IREE_METAL_COOP_NO_PAD -u IREE_METAL_COOP_NO_ATTN_ISOLATE -u IREE_METAL_COOP_NO_ATTN_ISOLATE_BF16 IREE_METAL_FFN_PAD_M64=0 iree-opt --iree-global-opt-raise-contraction-accumulator-to-f32 %s | FileCheck %s --check-prefix=DEFAULT
// RUN: env -u IREE_METAL_COOP_NO_PAD -u IREE_METAL_COOP_NO_ATTN_ISOLATE -u IREE_METAL_COOP_NO_ATTN_ISOLATE_BF16 IREE_METAL_FFN_PAD_M64=invalid iree-opt --iree-global-opt-raise-contraction-accumulator-to-f32 %s | FileCheck %s --check-prefix=DEFAULT
// RUN: env -u IREE_METAL_COOP_NO_PAD -u IREE_METAL_COOP_NO_ATTN_ISOLATE -u IREE_METAL_COOP_NO_ATTN_ISOLATE_BF16 IREE_METAL_FFN_PAD_M64=1 iree-opt --iree-global-opt-raise-contraction-accumulator-to-f32 %s | FileCheck %s --check-prefix=PAD64
// RUN: env -u IREE_METAL_COOP_NO_PAD -u IREE_METAL_COOP_NO_ATTN_ISOLATE -u IREE_METAL_COOP_NO_ATTN_ISOLATE_BF16 IREE_METAL_FFN_PAD_M64=1 iree-opt --pass-pipeline="builtin.module(func.func(iree-global-opt-raise-contraction-accumulator-to-f32,iree-global-opt-raise-contraction-accumulator-to-f32))" %s | FileCheck %s --check-prefix=PAD64

// DEFAULT-LABEL: func.func @m_major(
// DEFAULT: tensor<4624x768xbf16>
// DEFAULT: tensor<4624x3072xf32>
// DEFAULT-NOT: tensor<4672
// DEFAULT: return

// PAD64-LABEL: func.func @m_major(
// PAD64: tensor<4672x768xbf16>
// PAD64: tensor<4672x3072xf32>
// PAD64: tensor.extract_slice
// PAD64-SAME: tensor<4672x3072xf32> to tensor<4616x3072xf32>
// PAD64: return
func.func @m_major(
    %lhs: tensor<4616x768xbf16>,
    %rhs: tensor<768x3072xbf16>) -> tensor<4616x3072xbf16> {
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<4616x3072xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<4616x3072xbf16>) -> tensor<4616x3072xbf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<4616x768xbf16>, tensor<768x3072xbf16>)
      outs(%init : tensor<4616x3072xbf16>) -> tensor<4616x3072xbf16>
  return %result : tensor<4616x3072xbf16>
}

// DEFAULT-LABEL: func.func @k_major(
// DEFAULT: tensor<768x4624xbf16>
// DEFAULT: tensor<4624x3072xbf16>
// DEFAULT-NOT: tensor<4672
// DEFAULT: return

// PAD64-LABEL: func.func @k_major(
// PAD64: tensor<768x4672xbf16>
// PAD64: tensor<4672x3072xbf16>
// PAD64: linalg.matmul
// PAD64-SAME: tensor<768x4672xbf16>, tensor<4672x3072xbf16>
// PAD64: return
func.func @k_major(
    %lhs: tensor<768x4616xbf16>,
    %rhs: tensor<4616x3072xbf16>) -> tensor<768x3072xbf16> {
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<768x3072xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<768x3072xbf16>) -> tensor<768x3072xbf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<768x4616xbf16>, tensor<4616x3072xbf16>)
      outs(%init : tensor<768x3072xbf16>) -> tensor<768x3072xbf16>
  return %result : tensor<768x3072xbf16>
}

// PAD64-LABEL: func.func @wrong_ffn_width(
// PAD64: tensor<4624x768xbf16>
// PAD64-NOT: tensor<4672
// PAD64: return
func.func @wrong_ffn_width(
    %lhs: tensor<4616x768xbf16>,
    %rhs: tensor<768x1536xbf16>) -> tensor<4616x1536xbf16> {
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<4616x1536xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<4616x1536xbf16>) -> tensor<4616x1536xbf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<4616x768xbf16>, tensor<768x1536xbf16>)
      outs(%init : tensor<4616x1536xbf16>) -> tensor<4616x1536xbf16>
  return %result : tensor<4616x1536xbf16>
}

// PAD64-LABEL: func.func @wrong_input_type(
// PAD64: tensor<4624x768xf16>
// PAD64-NOT: tensor<4672
// PAD64: return
func.func @wrong_input_type(
    %lhs: tensor<4616x768xf16>,
    %rhs: tensor<768x3072xf16>) -> tensor<4616x3072xf16> {
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<4616x3072xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<4616x3072xf16>) -> tensor<4616x3072xf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<4616x768xf16>, tensor<768x3072xf16>)
      outs(%init : tensor<4616x3072xf16>) -> tensor<4616x3072xf16>
  return %result : tensor<4616x3072xf16>
}

// PAD64-LABEL: func.func @nearby_row_count(
// PAD64: tensor<4624x768xbf16>
// PAD64-NOT: tensor<4672
// PAD64: return
func.func @nearby_row_count(
    %lhs: tensor<4615x768xbf16>,
    %rhs: tensor<768x3072xbf16>) -> tensor<4615x3072xbf16> {
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<4615x3072xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<4615x3072xbf16>) -> tensor<4615x3072xbf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<4615x768xbf16>, tensor<768x3072xbf16>)
      outs(%init : tensor<4615x3072xbf16>) -> tensor<4615x3072xbf16>
  return %result : tensor<4615x3072xbf16>
}

// PAD64-LABEL: func.func @already_aligned(
// PAD64-NOT: tensor.pad
// PAD64-NOT: tensor<4672
// PAD64: linalg.matmul
// PAD64-SAME: tensor<4608x768xbf16>, tensor<768x3072xbf16>
// PAD64: return
func.func @already_aligned(
    %lhs: tensor<4608x768xbf16>,
    %rhs: tensor<768x3072xbf16>) -> tensor<4608x3072xbf16> {
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<4608x3072xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<4608x3072xbf16>) -> tensor<4608x3072xbf16>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<4608x768xbf16>, tensor<768x3072xbf16>)
      outs(%init : tensor<4608x3072xbf16>) -> tensor<4608x3072xbf16>
  return %result : tensor<4608x3072xbf16>
}

// DEFAULT-LABEL: func.func @native_f32_batch_matmul(
// DEFAULT: %[[BMM:.+]] = linalg.batch_matmul
// DEFAULT-NOT: util.optimization_barrier
// DEFAULT: return %[[BMM]]
func.func @native_f32_batch_matmul(
    %lhs: tensor<3x64x64xf32>,
    %rhs: tensor<3x64x64xf32>) -> tensor<3x64x64xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<3x64x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<3x64x64xf32>) -> tensor<3x64x64xf32>
  %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<3x64x64xf32>, tensor<3x64x64xf32>)
      outs(%init : tensor<3x64x64xf32>) -> tensor<3x64x64xf32>
  return %result : tensor<3x64x64xf32>
}

// DEFAULT-LABEL: func.func @tiny_bf16_batch_matmul(
// DEFAULT: %[[TINY_BMM:.+]] = linalg.batch_matmul
// DEFAULT-NOT: util.optimization_barrier
// DEFAULT: return %[[TINY_BMM]]
func.func @tiny_bf16_batch_matmul(
    %lhs: tensor<3x8x16xbf16>,
    %rhs: tensor<3x16x17xbf16>) -> tensor<3x8x17xbf16> {
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<3x8x17xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<3x8x17xbf16>) -> tensor<3x8x17xbf16>
  %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<3x8x16xbf16>, tensor<3x16x17xbf16>)
      outs(%init : tensor<3x8x17xbf16>) -> tensor<3x8x17xbf16>
  return %result : tensor<3x8x17xbf16>
}
