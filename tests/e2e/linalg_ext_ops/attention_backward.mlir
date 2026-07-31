// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#lse = affine_map<(b, m, k1, k2, n) -> (b, m)>
#scalar = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (b, m, k2)>

func.func @attention_backward_f32() {
  %query = util.unfoldable_constant
      dense<[[[0.2, -0.4], [0.7, 0.1]]]> : tensor<1x2x2xf32>
  %key = util.unfoldable_constant
      dense<[[[0.3, 0.5], [-0.6, 0.2], [0.8, -0.1]]]>
      : tensor<1x3x2xf32>
  %value = util.unfoldable_constant
      dense<[[[1.0, -0.5], [0.2, 0.7], [-0.3, 0.4]]]>
      : tensor<1x3x2xf32>
  %output_grad = util.unfoldable_constant
      dense<[[[0.6, -0.2], [-0.1, 0.9]]]> : tensor<1x2x2xf32>
  %scale = arith.constant 0.75 : f32
  %output_init = tensor.empty() : tensor<1x2x2xf32>
  %logsumexp_init = tensor.empty() : tensor<1x2xf32>
  %forward:2 = iree_linalg_ext.attention {
      decomposition_config = {use_exp2 = false},
      indexing_maps = [#q, #k, #v, #scalar, #o, #lse]}
      ins(%query, %key, %value, %scale :
          tensor<1x2x2xf32>, tensor<1x3x2xf32>, tensor<1x3x2xf32>, f32)
      outs(%output_init, %logsumexp_init :
          tensor<1x2x2xf32>, tensor<1x2xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x2x2xf32>, tensor<1x2xf32>

  %query_grad_init = tensor.empty() : tensor<1x2x2xf32>
  %key_grad_init = tensor.empty() : tensor<1x3x2xf32>
  %value_grad_init = tensor.empty() : tensor<1x3x2xf32>
  %grads:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {use_exp2 = false},
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %forward#0, %output_grad, %forward#1,
          %scale :
          tensor<1x2x2xf32>, tensor<1x3x2xf32>, tensor<1x3x2xf32>,
          tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2xf32>, f32)
      outs(%query_grad_init, %key_grad_init, %value_grad_init :
          tensor<1x2x2xf32>, tensor<1x3x2xf32>, tensor<1x3x2xf32>)
      -> tensor<1x2x2xf32>, tensor<1x3x2xf32>, tensor<1x3x2xf32>

  check.expect_almost_eq_const(
      %forward#0,
      dense<[[[0.24767324, 0.21111703],
              [0.26258194, 0.14859961]]]> : tensor<1x2x2xf32>,
      atol 1.0e-5, rtol 1.0e-5) : tensor<1x2x2xf32>
  check.expect_almost_eq_const(
      %forward#1,
      dense<[[1.0725648, 1.2432086]]> : tensor<1x2xf32>,
      atol 1.0e-5, rtol 1.0e-5) : tensor<1x2xf32>
  check.expect_almost_eq_const(
      %grads#0,
      dense<[[[-0.029492715, 0.073909581],
              [-0.026320858, -0.079558849]]]> : tensor<1x2x2xf32>,
      atol 1.0e-5, rtol 1.0e-5) : tensor<1x2x2xf32>
  check.expect_almost_eq_const(
      %grads#1,
      dense<[[[-0.093579605, -0.072142176],
              [0.050794955, 0.019218633],
              [0.042784642, 0.052923545]]]> : tensor<1x3x2xf32>,
      atol 1.0e-5, rtol 1.0e-5) : tensor<1x3x2xf32>
  check.expect_almost_eq_const(
      %grads#2,
      dense<[[[0.14976025, 0.25390309],
              [0.15531489, 0.13343012],
              [0.19492489, 0.31266671]]]> : tensor<1x3x2xf32>,
      atol 1.0e-5, rtol 1.0e-5) : tensor<1x3x2xf32>
  return
}

func.func @attention_backward_causal_and_all_masked() {
  %query = util.unfoldable_constant
      dense<[[[0.2, -0.4], [0.7, 0.1], [-0.3, 0.9]]]>
      : tensor<1x3x2xf32>
  %key = util.unfoldable_constant
      dense<[[[0.3, 0.5], [-0.6, 0.2], [0.8, -0.1]]]>
      : tensor<1x3x2xf32>
  %value = util.unfoldable_constant
      dense<[[[1.0, -0.5], [0.2, 0.7], [-0.3, 0.4]]]>
      : tensor<1x3x2xf32>
  %output_grad = util.unfoldable_constant
      dense<[[[0.6, -0.2], [-0.1, 0.9], [0.4, -0.7]]]>
      : tensor<1x3x2xf32>
  %mask_value = util.unfoldable_constant
      dense<[[[true, false, false],
              [true, true, false],
              [false, false, false]]]> : tensor<1x3x3xi1>
  %scale = arith.constant 0.75 : f32
  %output_init = tensor.empty() : tensor<1x3x2xf32>
  %logsumexp_init = tensor.empty() : tensor<1x3xf32>
  %forward:2 = iree_linalg_ext.attention {
      decomposition_config = {use_exp2 = true},
      indexing_maps = [#q, #k, #v, #scalar, #mask, #o, #lse]}
      ins(%query, %key, %value, %scale, %mask_value :
          tensor<1x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x2xf32>, f32,
          tensor<1x3x3xi1>)
      outs(%output_init, %logsumexp_init :
          tensor<1x3x2xf32>, tensor<1x3xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x3x2xf32>, tensor<1x3xf32>

  %query_grad_init = tensor.empty() : tensor<1x3x2xf32>
  %key_grad_init = tensor.empty() : tensor<1x3x2xf32>
  %value_grad_init = tensor.empty() : tensor<1x3x2xf32>
  %grads:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {use_exp2 = true},
      indexing_maps = [
        #q, #k, #v, #o, #o, #lse, #scalar, #mask, #q, #k, #v]}
      ins(%query, %key, %value, %forward#0, %output_grad, %forward#1,
          %scale, %mask_value :
          tensor<1x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x2xf32>,
          tensor<1x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3xf32>, f32,
          tensor<1x3x3xi1>)
      outs(%query_grad_init, %key_grad_init, %value_grad_init :
          tensor<1x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x2xf32>)
      -> tensor<1x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x2xf32>

  check.expect_almost_eq_const(
      %forward#0,
      dense<[[[1.0, -0.5],
              [0.697026849, -0.045540333],
              [0.0, 0.0]]]> : tensor<1x3x2xf32>,
      atol 2.0e-5, rtol 2.0e-5) : tensor<1x3x2xf32>
  check.expect_almost_eq_const(
      %grads#0,
      dense<[[[0.0, 0.0],
              [-0.184232309, -0.061410766],
              [0.0, 0.0]]]> : tensor<1x3x2xf32>,
      atol 2.0e-5, rtol 2.0e-5) : tensor<1x3x2xf32>
  check.expect_almost_eq_const(
      %grads#1,
      dense<[[[-0.143291786, -0.020470256],
              [0.143291786, 0.020470256],
              [0.0, 0.0]]]> : tensor<1x3x2xf32>,
      atol 2.0e-5, rtol 2.0e-5) : tensor<1x3x2xf32>
  check.expect_almost_eq_const(
      %grads#2,
      dense<[[[0.53787166, 0.35915521],
              [-0.03787164, 0.34084472],
              [0.0, 0.0]]]> : tensor<1x3x2xf32>,
      atol 2.0e-5, rtol 2.0e-5) : tensor<1x3x2xf32>
  return
}

func.func @attention_backward_bf16() {
  %query = util.unfoldable_constant
      dense<[[[0.2, -0.4], [0.7, 0.1]]]> : tensor<1x2x2xbf16>
  %key = util.unfoldable_constant
      dense<[[[0.3, 0.5], [-0.6, 0.2], [0.8, -0.1]]]>
      : tensor<1x3x2xbf16>
  %value = util.unfoldable_constant
      dense<[[[1.0, -0.5], [0.2, 0.7], [-0.3, 0.4]]]>
      : tensor<1x3x2xbf16>
  %output_grad = util.unfoldable_constant
      dense<[[[0.6, -0.2], [-0.1, 0.9]]]> : tensor<1x2x2xbf16>
  %scale = arith.constant 0.75 : bf16
  %output_init = tensor.empty() : tensor<1x2x2xbf16>
  %logsumexp_init = tensor.empty() : tensor<1x2xf32>
  %forward:2 = iree_linalg_ext.attention {
      decomposition_config = {use_exp2 = true},
      indexing_maps = [#q, #k, #v, #scalar, #o, #lse]}
      ins(%query, %key, %value, %scale :
          tensor<1x2x2xbf16>, tensor<1x3x2xbf16>, tensor<1x3x2xbf16>, bf16)
      outs(%output_init, %logsumexp_init :
          tensor<1x2x2xbf16>, tensor<1x2xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x2x2xbf16>, tensor<1x2xf32>

  %query_grad_init = tensor.empty() : tensor<1x2x2xbf16>
  %key_grad_init = tensor.empty() : tensor<1x3x2xbf16>
  %value_grad_init = tensor.empty() : tensor<1x3x2xbf16>
  %grads:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {use_exp2 = true},
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %forward#0, %output_grad, %forward#1,
          %scale :
          tensor<1x2x2xbf16>, tensor<1x3x2xbf16>, tensor<1x3x2xbf16>,
          tensor<1x2x2xbf16>, tensor<1x2x2xbf16>, tensor<1x2xf32>, bf16)
      outs(%query_grad_init, %key_grad_init, %value_grad_init :
          tensor<1x2x2xbf16>, tensor<1x3x2xbf16>, tensor<1x3x2xbf16>)
      -> tensor<1x2x2xbf16>, tensor<1x3x2xbf16>, tensor<1x3x2xbf16>

  check.expect_almost_eq_const(
      %forward#0,
      dense<[[[0.248046875, 0.2109375],
              [0.263671875, 0.1484375]]]> : tensor<1x2x2xbf16>,
      atol 4.0e-3, rtol 1.0e-2) : tensor<1x2x2xbf16>
  check.expect_almost_eq_const(
      %grads#0,
      dense<[[[-0.029785156, 0.07421875],
              [-0.026245117, -0.079589844]]]> : tensor<1x2x2xbf16>,
      atol 2.0e-3, rtol 1.0e-2) : tensor<1x2x2xbf16>
  check.expect_almost_eq_const(
      %grads#1,
      dense<[[[-0.093261719, -0.072265625],
              [0.050537109, 0.019165039],
              [0.042724609, 0.053222656]]]> : tensor<1x3x2xbf16>,
      atol 2.0e-3, rtol 1.0e-2) : tensor<1x3x2xbf16>
  check.expect_almost_eq_const(
      %grads#2,
      dense<[[[0.150390625, 0.25390625],
              [0.155273438, 0.1328125],
              [0.1953125, 0.3125]]]> : tensor<1x3x2xbf16>,
      atol 2.0e-3, rtol 1.0e-2) : tensor<1x3x2xbf16>
  return
}

// This compact case is deliberately sensitive to where bf16 score scaling is
// rounded. It prevents backward from silently switching back to f32 post-QK
// scaling while consuming LSE from the forward pre-scaled-Q path.
func.func @attention_backward_bf16_score_replay() {
  %query = util.unfoldable_constant
      dense<[[[3.015625, -0.322265625]]]> : tensor<1x1x2xbf16>
  %key = util.unfoldable_constant
      dense<[[[4.3125, -1.390625], [3.96875, 5.34375]]]>
      : tensor<1x2x2xbf16>
  %value = util.unfoldable_constant
      dense<[[[1.59375], [-5.21875]]]> : tensor<1x2x1xbf16>
  %output_grad = util.unfoldable_constant
      dense<[[[3.4375]]]> : tensor<1x1x1xbf16>
  %scale = arith.constant 0.353516 : bf16
  %output_init = tensor.empty() : tensor<1x1x1xbf16>
  %logsumexp_init = tensor.empty() : tensor<1x1xf32>
  %forward:2 = iree_linalg_ext.attention {
      decomposition_config = {use_exp2 = true},
      indexing_maps = [#q, #k, #v, #scalar, #o, #lse]}
      ins(%query, %key, %value, %scale :
          tensor<1x1x2xbf16>, tensor<1x2x2xbf16>, tensor<1x2x1xbf16>, bf16)
      outs(%output_init, %logsumexp_init :
          tensor<1x1x1xbf16>, tensor<1x1xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x1x1xbf16>, tensor<1x1xf32>

  %query_grad_init = tensor.empty() : tensor<1x1x2xbf16>
  %key_grad_init = tensor.empty() : tensor<1x2x2xbf16>
  %value_grad_init = tensor.empty() : tensor<1x2x1xbf16>
  %grads:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {use_exp2 = true},
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %forward#0, %output_grad, %forward#1,
          %scale :
          tensor<1x1x2xbf16>, tensor<1x2x2xbf16>, tensor<1x2x1xbf16>,
          tensor<1x1x1xbf16>, tensor<1x1x1xbf16>, tensor<1x1xf32>, bf16)
      outs(%query_grad_init, %key_grad_init, %value_grad_init :
          tensor<1x1x2xbf16>, tensor<1x2x2xbf16>, tensor<1x2x1xbf16>)
      -> tensor<1x1x2xbf16>, tensor<1x2x2xbf16>, tensor<1x2x1xbf16>

  check.expect_almost_eq_const(
      %forward#0,
      dense<[[[-0.0546875]]]> : tensor<1x1x1xbf16>,
      atol 1.0e-3, rtol 1.0e-3) : tensor<1x1x1xbf16>
  check.expect_almost_eq_const(
      %grads#0,
      dense<[[[0.50390625, -10.25]]]> : tensor<1x1x2xbf16>,
      atol 1.0e-2, rtol 1.0e-3) : tensor<1x1x2xbf16>
  return
}

// Regression for distributed CPU codegen of backward QK followed by masking.
// The collapsed batch/head extent of four forces multiple workgroups for this
// shape; a fused DPS-init mask consumer used to leave a global vector.store
// outside the mapped scf.forall.
func.func @attention_backward_bf16_distributed_masked() {
  %query = util.unfoldable_constant dense<1.0> : tensor<4x4x8xbf16>
  %key = util.unfoldable_constant dense<1.0> : tensor<4x4x8xbf16>
  %value = util.unfoldable_constant dense<1.0> : tensor<4x4x8xbf16>
  %output_grad = util.unfoldable_constant dense<1.0> :
      tensor<4x4x8xbf16>
  %mask_value = util.unfoldable_constant dense<[
      [[true, false, false, false],
       [true, true, false, false],
       [true, true, true, false],
       [true, true, true, true]],
      [[true, false, false, false],
       [true, true, false, false],
       [true, true, true, false],
       [true, true, true, true]],
      [[true, false, false, false],
       [true, true, false, false],
       [true, true, true, false],
       [true, true, true, true]],
      [[true, false, false, false],
       [true, true, false, false],
       [true, true, true, false],
       [true, true, true, true]]
    ]> : tensor<4x4x4xi1>
  %scale = arith.constant 0.353516 : bf16
  %output_init = tensor.empty() : tensor<4x4x8xbf16>
  %logsumexp_init = tensor.empty() : tensor<4x4xf32>
  %forward:2 = iree_linalg_ext.attention {
      decomposition_config = {use_exp2 = false},
      indexing_maps = [#q, #k, #v, #scalar, #mask, #o, #lse]}
      ins(%query, %key, %value, %scale, %mask_value :
          tensor<4x4x8xbf16>, tensor<4x4x8xbf16>,
          tensor<4x4x8xbf16>, bf16, tensor<4x4x4xi1>)
      outs(%output_init, %logsumexp_init :
          tensor<4x4x8xbf16>, tensor<4x4xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<4x4x8xbf16>, tensor<4x4xf32>

  %query_grad_init = tensor.empty() : tensor<4x4x8xbf16>
  %key_grad_init = tensor.empty() : tensor<4x4x8xbf16>
  %value_grad_init = tensor.empty() : tensor<4x4x8xbf16>
  %grads:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {use_exp2 = false},
      indexing_maps = [
        #q, #k, #v, #o, #o, #lse, #scalar, #mask, #q, #k, #v]}
      ins(%query, %key, %value, %forward#0, %output_grad, %forward#1,
          %scale, %mask_value :
          tensor<4x4x8xbf16>, tensor<4x4x8xbf16>,
          tensor<4x4x8xbf16>, tensor<4x4x8xbf16>,
          tensor<4x4x8xbf16>, tensor<4x4xf32>, bf16,
          tensor<4x4x4xi1>)
      outs(%query_grad_init, %key_grad_init, %value_grad_init :
          tensor<4x4x8xbf16>, tensor<4x4x8xbf16>,
          tensor<4x4x8xbf16>)
      -> tensor<4x4x8xbf16>, tensor<4x4x8xbf16>,
         tensor<4x4x8xbf16>

  %expected_value_grad_rows = util.unfoldable_constant
      dense<[2.078125, 1.0859375, 0.58203125, 0.25]> : tensor<4xbf16>
  %expected_value_grad_init = tensor.empty() : tensor<4x4x8xbf16>
  %expected_value_grad = linalg.generic {
      indexing_maps = [
        affine_map<(b, k, n) -> (k)>,
        affine_map<(b, k, n) -> (b, k, n)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%expected_value_grad_rows : tensor<4xbf16>)
      outs(%expected_value_grad_init : tensor<4x4x8xbf16>) {
    ^bb0(%expected: bf16, %unused: bf16):
      linalg.yield %expected : bf16
  } -> tensor<4x4x8xbf16>

  check.expect_almost_eq_const(
      %forward#0, dense<1.0> : tensor<4x4x8xbf16>,
      atol 0.0, rtol 0.0) : tensor<4x4x8xbf16>
  check.expect_almost_eq_const(
      %grads#0, dense<0.0> : tensor<4x4x8xbf16>,
      atol 0.0, rtol 0.0) : tensor<4x4x8xbf16>
  check.expect_almost_eq_const(
      %grads#1, dense<0.0> : tensor<4x4x8xbf16>,
      atol 0.0, rtol 0.0) : tensor<4x4x8xbf16>
  check.expect_almost_eq(
      %grads#2, %expected_value_grad,
      atol 1.0e-2, rtol 1.0e-2) : tensor<4x4x8xbf16>
  return
}
