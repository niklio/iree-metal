// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: env -u IREE_METAL_LN_PAIRED iree-opt --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" %s | FileCheck %s --check-prefix=OFF
// RUN: env IREE_METAL_LN_PAIRED=0 iree-opt --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" %s | FileCheck %s --check-prefix=OFF
// RUN: env IREE_METAL_LN_PAIRED=1 iree-opt --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" %s | FileCheck %s --check-prefix=ON
// RUN: env IREE_METAL_LN_PAIRED=1 iree-opt --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse),iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" %s | FileCheck %s --check-prefix=ON
// RUN: sed 's/zero_f32 = stablehlo.constant dense<0.000000e+00>/zero_f32 = stablehlo.constant dense<1.000000e+00>/' %s | env IREE_METAL_LN_PAIRED=1 iree-opt --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" | FileCheck %s --check-prefix=NONZERO
// RUN: sed 's/centered = stablehlo.subtract \([^,]*\), \([^ ]*\)/centered = stablehlo.subtract \2, \1/' %s | env IREE_METAL_LN_PAIRED=1 iree-opt --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" | FileCheck %s --check-prefix=NONZERO
// RUN: sed 's/divided_variance_seed = stablehlo.divide \([^,]*\), \([^ ]*\)/divided_variance_seed = stablehlo.divide \2, \1/' %s | env IREE_METAL_LN_PAIRED=1 iree-opt --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" | FileCheck %s --check-prefix=NONZERO
// RUN: sed 's/variance_mean = stablehlo.divide \([^,]*\), \([^ ]*\)/variance_mean = stablehlo.divide \2, \1/' %s | env IREE_METAL_LN_PAIRED=1 iree-opt --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" | FileCheck %s --check-prefix=NONZERO
// RUN: sed 's/direct_mean = stablehlo.divide \([^,]*\), \([^ ]*\)/direct_mean = stablehlo.divide \2, \1/' %s | env IREE_METAL_LN_PAIRED=1 iree-opt --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" | FileCheck %s --check-prefix=NONZERO
// RUN: awk '$2 == "=" && $3 == "stablehlo.add" && $1 ~ /output$/ { type=$7; print "  " $1 " = scf.execute_region -> " type " {"; print "    " $1 "_nested = stablehlo.add " $4 " " $5 " : " type; print "    scf.yield " $1 "_nested : " type; print "  }"; next } { print }' %s | env IREE_METAL_LN_PAIRED=1 iree-opt --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" | FileCheck %s --check-prefix=NONZERO

// This is the exact canonical rank-3 BF16 LayerNorm forward/VJP spelling from
// JAX, reduced to B=2, S=3, D=384. The two redundant parameter-gradient
// reductions and expanded two-pass variance VJP are deliberately retained.
// The mutation RUNs independently change the forward reduction init, reverse
// the centered subtraction, and reverse each matched backward division. Each
// otherwise-positive fixture must retain its complete original VJP. The awk
// mutation nests the forward output in another block to cover the block guard.

// OFF-LABEL: func.func @paired_layer_norm(
// OFF: %[[DY:.+]] = stablehlo.negate
// OFF: %[[BETA_FIRST:.+]] = stablehlo.reduce(%[[DY]] init:
// OFF-SAME: across dimensions = [0, 1]
// OFF: %[[BETA_RESHAPE:.+]] = stablehlo.reshape %[[BETA_FIRST]]
// OFF: %[[BETA:.+]] = stablehlo.reduce(%[[BETA_RESHAPE]] init:
// OFF-SAME: across dimensions = [0, 1]
// OFF-NOT: :2 = stablehlo.reduce
// OFF: return

// OFF-LABEL: func.func @paired_layer_norm_broadcast_f32(
// OFF: %[[VISION_DY:.+]] = stablehlo.negate
// OFF: %[[VISION_BETA_FIRST:.+]] = stablehlo.reduce(%[[VISION_DY]] init:
// OFF-SAME: across dimensions = [0, 1]
// OFF: %[[VISION_BETA_RESHAPE:.+]] = stablehlo.reshape %[[VISION_BETA_FIRST]]
// OFF: stablehlo.reduce(%[[VISION_BETA_RESHAPE]] init:
// OFF-SAME: across dimensions = [0, 1]
// OFF: %[[VISION_VARIANCE_CONTRIBUTION:.+]] = stablehlo.convert {{.*}} : (tensor<2x3x384xf32>) -> tensor<2x3x384xbf16>
// OFF-NEXT: %[[VISION_NEGATIVE_DIRECT:.+]] = stablehlo.negate
// OFF-SAME: tensor<2x3x384xbf16>
// OFF-NEXT: %[[VISION_DIRECT_MEAN_SUM:.+]] = stablehlo.reduce(%[[VISION_NEGATIVE_DIRECT]] init:
// OFF: %[[VISION_DIRECT_MEAN_SUM_2:.+]] = stablehlo.reduce
// OFF-NEXT: %[[DIRECT_MEAN_BROADCAST_F32:.+]] = stablehlo.broadcast_in_dim %[[VISION_DIRECT_MEAN_SUM_2]]
// OFF-SAME: (tensor<2x3xf32>) -> tensor<2x3x384xf32>
// OFF-NEXT: stablehlo.convert %[[DIRECT_MEAN_BROADCAST_F32]]
// OFF-SAME: (tensor<2x3x384xf32>) -> tensor<2x3x384xbf16>
// OFF-NOT: :2 = stablehlo.reduce
// OFF: return

// NONZERO-LABEL: func.func @paired_layer_norm(
// NONZERO: %[[REJECTED_DY:.+]] = stablehlo.negate
// NONZERO-NEXT: %[[REJECTED_BETA_FIRST:.+]] = stablehlo.reduce(%[[REJECTED_DY]] init:
// NONZERO-SAME: across dimensions = [0, 1]
// NONZERO-NEXT: %[[REJECTED_BETA_RESHAPE:.+]] = stablehlo.reshape %[[REJECTED_BETA_FIRST]]
// NONZERO-NEXT: %[[REJECTED_BETA:.+]] = stablehlo.reduce(%[[REJECTED_BETA_RESHAPE]] init:
// NONZERO-SAME: across dimensions = [0, 1]
// NONZERO-NOT: :2 = stablehlo.reduce
// NONZERO: return

// NONZERO-LABEL: func.func @paired_layer_norm_broadcast_f32(
// NONZERO: %[[VISION_REJECTED_DY:.+]] = stablehlo.negate
// NONZERO-NEXT: %[[VISION_REJECTED_BETA_FIRST:.+]] = stablehlo.reduce(%[[VISION_REJECTED_DY]] init:
// NONZERO-SAME: across dimensions = [0, 1]
// NONZERO-NEXT: %[[VISION_REJECTED_BETA_RESHAPE:.+]] = stablehlo.reshape %[[VISION_REJECTED_BETA_FIRST]]
// NONZERO-NEXT: %[[VISION_REJECTED_BETA:.+]] = stablehlo.reduce(%[[VISION_REJECTED_BETA_RESHAPE]] init:
// NONZERO-SAME: across dimensions = [0, 1]
// NONZERO-NOT: :2 = stablehlo.reduce
// NONZERO: return

// ON-LABEL: func.func @paired_layer_norm(
// ON-NOT: :2 = stablehlo.reduce
// ON: %[[DY:.+]] = stablehlo.negate
// ON-NEXT: %[[SCALED_DY:.+]] = stablehlo.multiply %[[DY]],
// ON-NEXT: %[[SCALED_CENTERED:.+]] = stablehlo.multiply %[[SCALED_DY]],
// ON-NEXT: %[[GAMMA_CENTERED:.+]] = stablehlo.multiply %[[DY]],
// ON-NEXT: %[[GAMMA_INPUT:.+]] = stablehlo.multiply %[[GAMMA_CENTERED]],
// ON-NEXT: %[[GAMMA_GRAD:.+]] = stablehlo.reduce(%[[GAMMA_INPUT]] init:
// ON-SAME: across dimensions = [0, 1]
// ON-NEXT: %[[BETA_GRAD:.+]] = stablehlo.reduce(%[[DY]] init:
// ON-SAME: across dimensions = [0, 1]
// ON-NOT: :2 = stablehlo.reduce
// ON-NEXT: %[[SCALED_DY_F32:.+]] = stablehlo.convert %[[SCALED_DY]]
// ON-NEXT: %[[SCALED_CENTERED_F32:.+]] = stablehlo.convert %[[SCALED_CENTERED]]
// ON-NEXT: %[[SUM_GRAD:.+]] = stablehlo.reduce(%[[SCALED_DY_F32]] init:
// ON-SAME: across dimensions = [2]
// ON-NEXT: %[[SUM_CENTERED_GRAD:.+]] = stablehlo.reduce(%[[SCALED_CENTERED_F32]] init:
// ON-SAME: across dimensions = [2]
// ON-NOT: :2 = stablehlo.reduce
// ON: %[[CORRECTION:.+]] = stablehlo.divide
// ON: %[[CENTERED_GRAD:.+]] = stablehlo.subtract %[[SCALED_DY_F32]], %[[CORRECTION]]
// ON: %[[INPUT_GRAD_F32:.+]] = stablehlo.multiply
// ON: %[[INPUT_GRAD:.+]] = stablehlo.convert %[[INPUT_GRAD_F32]]
// ON-NOT: :2 = stablehlo.reduce
// ON: return {{.*}}%[[INPUT_GRAD]], %[[GAMMA_GRAD]], %[[BETA_GRAD]]

// ON-LABEL: func.func @paired_layer_norm_broadcast_f32(
// ON-NOT: :2 = stablehlo.reduce
// ON: %[[VISION_DY:.+]] = stablehlo.negate
// ON-NEXT: %[[VISION_SCALED_DY:.+]] = stablehlo.multiply %[[VISION_DY]],
// ON-NEXT: %[[VISION_SCALED_CENTERED:.+]] = stablehlo.multiply %[[VISION_SCALED_DY]],
// ON-NEXT: %[[VISION_GAMMA_CENTERED:.+]] = stablehlo.multiply %[[VISION_DY]],
// ON-NEXT: %[[VISION_GAMMA_INPUT:.+]] = stablehlo.multiply %[[VISION_GAMMA_CENTERED]],
// ON-NEXT: %[[VISION_GAMMA_GRAD:.+]] = stablehlo.reduce(%[[VISION_GAMMA_INPUT]] init:
// ON-SAME: across dimensions = [0, 1]
// ON-NEXT: %[[VISION_BETA_GRAD:.+]] = stablehlo.reduce(%[[VISION_DY]] init:
// ON-SAME: across dimensions = [0, 1]
// ON-NOT: :2 = stablehlo.reduce
// ON-NEXT: %[[VISION_SCALED_DY_F32:.+]] = stablehlo.convert %[[VISION_SCALED_DY]]
// ON-NEXT: %[[VISION_SCALED_CENTERED_F32:.+]] = stablehlo.convert %[[VISION_SCALED_CENTERED]]
// ON-NEXT: %[[VISION_SUM_GRAD:.+]] = stablehlo.reduce(%[[VISION_SCALED_DY_F32]] init:
// ON-SAME: across dimensions = [2]
// ON-NEXT: %[[VISION_SUM_CENTERED_GRAD:.+]] = stablehlo.reduce(%[[VISION_SCALED_CENTERED_F32]] init:
// ON-SAME: across dimensions = [2]
// ON-NOT: :2 = stablehlo.reduce
// ON: %[[VISION_CENTERED_GRAD:.+]] = stablehlo.subtract
// ON: %[[VISION_RSTD_BROADCAST:.+]] = stablehlo.broadcast_in_dim
// ON-NEXT: %[[VISION_INPUT_GRAD_F32:.+]] = stablehlo.multiply %[[VISION_RSTD_BROADCAST]], %[[VISION_CENTERED_GRAD]]
// ON: %[[VISION_INPUT_GRAD:.+]] = stablehlo.convert %[[VISION_INPUT_GRAD_F32]]
// ON-NOT: :2 = stablehlo.reduce
// ON: return {{.*}}%[[VISION_INPUT_GRAD]], %[[VISION_GAMMA_GRAD]], %[[VISION_BETA_GRAD]]

func.func @paired_layer_norm(
    %input: tensor<2x3x384xbf16>, %gamma: tensor<384xbf16>,
    %beta: tensor<384xbf16>, %incoming_grad: tensor<2x3x384xbf16>)
    -> (tensor<2x3x384xbf16>, tensor<2x3x384xbf16>,
        tensor<384xbf16>, tensor<384xbf16>) {
  %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %zero_small_bf16 = stablehlo.constant dense<0.000000e+00> : tensor<2x3x1xbf16>
  %two = stablehlo.constant dense<2.000000e+00> : tensor<2x3x384xf32>
  %negative_half = stablehlo.constant dense<-5.000000e-01> : tensor<2x3x1xbf16>
  %epsilon = stablehlo.constant dense<1.001360e-05> : tensor<2x3x1xbf16>
  %feature_count = stablehlo.constant dense<3.840000e+02> : tensor<2x3x1xf32>
  %feature_count_scalar = stablehlo.constant dense<3.840000e+02> : tensor<f32>
  %nan_f32 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
  %ddof = stablehlo.constant dense<0> : tensor<i32>
  %ddof_f32 = stablehlo.convert %ddof : (tensor<i32>) -> tensor<f32>
  %denominator = stablehlo.subtract %feature_count_scalar, %ddof_f32 : tensor<f32>
  %denominator_broadcast = stablehlo.broadcast_in_dim %denominator, dims = [] : (tensor<f32>) -> tensor<2x3x1xf32>
  %valid_variance = stablehlo.compare GT, %denominator, %zero_f32, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %nan_bf16 = stablehlo.convert %nan_f32 : (tensor<f32>) -> tensor<bf16>
  %nan = stablehlo.broadcast_in_dim %nan_bf16, dims = [] : (tensor<bf16>) -> tensor<2x3x1xbf16>

  %input_f32 = stablehlo.convert %input : (tensor<2x3x384xbf16>) -> tensor<2x3x384xf32>
  %mean_sum = stablehlo.reduce(%input_f32 init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x384xf32>, tensor<f32>) -> tensor<2x3xf32>
  %mean_reshape = stablehlo.reshape %mean_sum : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
  %mean = stablehlo.divide %mean_reshape, %feature_count : tensor<2x3x1xf32>
  %mean_bf16 = stablehlo.convert %mean : (tensor<2x3x1xf32>) -> tensor<2x3x1xbf16>
  %mean_broadcast_f32 = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2] : (tensor<2x3x1xf32>) -> tensor<2x3x384xf32>
  %centered_f32 = stablehlo.subtract %input_f32, %mean_broadcast_f32 : tensor<2x3x384xf32>
  %centered_squared = stablehlo.multiply %centered_f32, %centered_f32 : tensor<2x3x384xf32>
  %twice_centered = stablehlo.multiply %two, %centered_f32 : tensor<2x3x384xf32>
  %variance_sum = stablehlo.reduce(%centered_squared init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x384xf32>, tensor<f32>) -> tensor<2x3xf32>
  %variance_reshape = stablehlo.reshape %variance_sum : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
  %variance_f32 = stablehlo.divide %variance_reshape, %denominator_broadcast : tensor<2x3x1xf32>
  %variance_bf16 = stablehlo.convert %variance_f32 : (tensor<2x3x1xf32>) -> tensor<2x3x1xbf16>
  %variance = stablehlo.select %valid_variance, %variance_bf16, %nan : tensor<i1>, tensor<2x3x1xbf16>
  %mean_broadcast_bf16 = stablehlo.broadcast_in_dim %mean_bf16, dims = [0, 1, 2] : (tensor<2x3x1xbf16>) -> tensor<2x3x384xbf16>
  %centered = stablehlo.subtract %input, %mean_broadcast_bf16 : tensor<2x3x384xbf16>
  %variance_epsilon = stablehlo.add %variance, %epsilon : tensor<2x3x1xbf16>
  %rstd = stablehlo.rsqrt %variance_epsilon : tensor<2x3x1xbf16>
  %rstd_divided = stablehlo.divide %rstd, %variance_epsilon : tensor<2x3x1xbf16>
  %negative_half_rstd = stablehlo.multiply %negative_half, %rstd_divided : tensor<2x3x1xbf16>
  %rstd_broadcast = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<2x3x1xbf16>) -> tensor<2x3x384xbf16>
  %normalized = stablehlo.multiply %centered, %rstd_broadcast : tensor<2x3x384xbf16>
  %gamma_broadcast = stablehlo.broadcast_in_dim %gamma, dims = [2] : (tensor<384xbf16>) -> tensor<2x3x384xbf16>
  %scaled = stablehlo.multiply %normalized, %gamma_broadcast : tensor<2x3x384xbf16>
  %beta_broadcast = stablehlo.broadcast_in_dim %beta, dims = [2] : (tensor<384xbf16>) -> tensor<2x3x384xbf16>
  %output = stablehlo.add %scaled, %beta_broadcast : tensor<2x3x384xbf16>
  %output_grad = stablehlo.negate %incoming_grad : tensor<2x3x384xbf16>
  // Keep the backward init below dY to cover rewrite-point dominance.
  %zero_bf16 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>

  %beta_first = stablehlo.reduce(%output_grad init: %zero_bf16) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x3x384xbf16>, tensor<bf16>) -> tensor<384xbf16>
  %beta_reshape = stablehlo.reshape %beta_first : (tensor<384xbf16>) -> tensor<1x1x384xbf16>
  %beta_grad = stablehlo.reduce(%beta_reshape init: %zero_bf16) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x384xbf16>, tensor<bf16>) -> tensor<384xbf16>
  %dgamma_input = stablehlo.multiply %normalized, %output_grad : tensor<2x3x384xbf16>
  %gamma_first = stablehlo.reduce(%dgamma_input init: %zero_bf16) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x3x384xbf16>, tensor<bf16>) -> tensor<384xbf16>
  %gamma_reshape = stablehlo.reshape %gamma_first : (tensor<384xbf16>) -> tensor<1x1x384xbf16>
  %gamma_grad = stablehlo.reduce(%gamma_reshape init: %zero_bf16) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x384xbf16>, tensor<bf16>) -> tensor<384xbf16>

  %scaled_output_grad = stablehlo.multiply %output_grad, %gamma_broadcast : tensor<2x3x384xbf16>
  %centered_scaled_grad = stablehlo.multiply %centered, %scaled_output_grad : tensor<2x3x384xbf16>
  %centered_grad_sum = stablehlo.reduce(%centered_scaled_grad init: %zero_bf16) applies stablehlo.add across dimensions = [2] : (tensor<2x3x384xbf16>, tensor<bf16>) -> tensor<2x3xbf16>
  %centered_grad_reshape = stablehlo.reshape %centered_grad_sum : (tensor<2x3xbf16>) -> tensor<2x3x1xbf16>
  %direct_grad = stablehlo.multiply %scaled_output_grad, %rstd_broadcast : tensor<2x3x384xbf16>
  %variance_seed = stablehlo.multiply %centered_grad_reshape, %negative_half_rstd : tensor<2x3x1xbf16>
  %selected_variance_seed = stablehlo.select %valid_variance, %variance_seed, %zero_small_bf16 : tensor<i1>, tensor<2x3x1xbf16>
  %variance_seed_f32 = stablehlo.convert %selected_variance_seed : (tensor<2x3x1xbf16>) -> tensor<2x3x1xf32>
  %divided_variance_seed = stablehlo.divide %variance_seed_f32, %denominator_broadcast : tensor<2x3x1xf32>
  %variance_seed_sum = stablehlo.reduce(%divided_variance_seed init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x3xf32>
  %variance_seed_broadcast = stablehlo.broadcast_in_dim %variance_seed_sum, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x384xf32>
  %variance_product = stablehlo.multiply %variance_seed_broadcast, %twice_centered : tensor<2x3x384xf32>
  %negative_variance_product = stablehlo.negate %variance_product : tensor<2x3x384xf32>
  %variance_mean_sum = stablehlo.reduce(%negative_variance_product init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x384xf32>, tensor<f32>) -> tensor<2x3xf32>
  %variance_mean_reshape = stablehlo.reshape %variance_mean_sum : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
  %variance_mean = stablehlo.divide %variance_mean_reshape, %feature_count : tensor<2x3x1xf32>
  %variance_mean_sum_2 = stablehlo.reduce(%variance_mean init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x3xf32>
  %variance_mean_broadcast = stablehlo.broadcast_in_dim %variance_mean_sum_2, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x384xf32>
  %variance_add = stablehlo.add %variance_product, %variance_mean_broadcast : tensor<2x3x384xf32>
  %variance_contribution = stablehlo.convert %variance_add : (tensor<2x3x384xf32>) -> tensor<2x3x384xbf16>

  %negative_direct = stablehlo.negate %direct_grad : tensor<2x3x384xbf16>
  %direct_mean_sum = stablehlo.reduce(%negative_direct init: %zero_bf16) applies stablehlo.add across dimensions = [2] : (tensor<2x3x384xbf16>, tensor<bf16>) -> tensor<2x3xbf16>
  %direct_mean_f32 = stablehlo.convert %direct_mean_sum : (tensor<2x3xbf16>) -> tensor<2x3xf32>
  %direct_mean_reshape = stablehlo.reshape %direct_mean_f32 : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
  %direct_mean = stablehlo.divide %direct_mean_reshape, %feature_count : tensor<2x3x1xf32>
  %direct_mean_sum_2 = stablehlo.reduce(%direct_mean init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x3xf32>
  %direct_mean_bf16 = stablehlo.convert %direct_mean_sum_2 : (tensor<2x3xf32>) -> tensor<2x3xbf16>
  %direct_mean_broadcast = stablehlo.broadcast_in_dim %direct_mean_bf16, dims = [0, 1] : (tensor<2x3xbf16>) -> tensor<2x3x384xbf16>
  %direct_add = stablehlo.add %direct_grad, %variance_contribution : tensor<2x3x384xbf16>
  %input_grad = stablehlo.add %direct_add, %direct_mean_broadcast : tensor<2x3x384xbf16>
  return %output, %input_grad, %gamma_grad, %beta_grad : tensor<2x3x384xbf16>, tensor<2x3x384xbf16>, tensor<384xbf16>, tensor<384xbf16>
}

// Vision graphs use the same VJP with the direct-mean broadcast and convert
// interchanged. Keep a complete fixture for that distinct canonical spelling.
func.func @paired_layer_norm_broadcast_f32(
    %input: tensor<2x3x384xbf16>, %gamma: tensor<384xbf16>,
    %beta: tensor<384xbf16>, %incoming_grad: tensor<2x3x384xbf16>)
    -> (tensor<2x3x384xbf16>, tensor<2x3x384xbf16>,
        tensor<384xbf16>, tensor<384xbf16>) {
  %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %zero_bf16 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  %zero_small_bf16 = stablehlo.constant dense<0.000000e+00> : tensor<2x3x1xbf16>
  %two = stablehlo.constant dense<2.000000e+00> : tensor<2x3x384xf32>
  %negative_half = stablehlo.constant dense<-5.000000e-01> : tensor<2x3x1xbf16>
  %epsilon = stablehlo.constant dense<1.001360e-05> : tensor<2x3x1xbf16>
  %feature_count = stablehlo.constant dense<3.840000e+02> : tensor<2x3x1xf32>
  %feature_count_scalar = stablehlo.constant dense<3.840000e+02> : tensor<f32>
  %nan_f32 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
  %ddof = stablehlo.constant dense<0> : tensor<i32>
  %ddof_f32 = stablehlo.convert %ddof : (tensor<i32>) -> tensor<f32>
  %denominator = stablehlo.subtract %feature_count_scalar, %ddof_f32 : tensor<f32>
  %denominator_broadcast = stablehlo.broadcast_in_dim %denominator, dims = [] : (tensor<f32>) -> tensor<2x3x1xf32>
  %valid_variance = stablehlo.compare GT, %denominator, %zero_f32, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %nan_bf16 = stablehlo.convert %nan_f32 : (tensor<f32>) -> tensor<bf16>
  %nan = stablehlo.broadcast_in_dim %nan_bf16, dims = [] : (tensor<bf16>) -> tensor<2x3x1xbf16>

  %input_f32 = stablehlo.convert %input : (tensor<2x3x384xbf16>) -> tensor<2x3x384xf32>
  %mean_sum = stablehlo.reduce(%input_f32 init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x384xf32>, tensor<f32>) -> tensor<2x3xf32>
  %mean_reshape = stablehlo.reshape %mean_sum : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
  %mean = stablehlo.divide %mean_reshape, %feature_count : tensor<2x3x1xf32>
  %mean_bf16 = stablehlo.convert %mean : (tensor<2x3x1xf32>) -> tensor<2x3x1xbf16>
  %mean_broadcast_f32 = stablehlo.broadcast_in_dim %mean, dims = [0, 1, 2] : (tensor<2x3x1xf32>) -> tensor<2x3x384xf32>
  %centered_f32 = stablehlo.subtract %input_f32, %mean_broadcast_f32 : tensor<2x3x384xf32>
  %centered_squared = stablehlo.multiply %centered_f32, %centered_f32 : tensor<2x3x384xf32>
  %twice_centered = stablehlo.multiply %two, %centered_f32 : tensor<2x3x384xf32>
  %variance_sum = stablehlo.reduce(%centered_squared init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x384xf32>, tensor<f32>) -> tensor<2x3xf32>
  %variance_reshape = stablehlo.reshape %variance_sum : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
  %variance_f32 = stablehlo.divide %variance_reshape, %denominator_broadcast : tensor<2x3x1xf32>
  %variance_bf16 = stablehlo.convert %variance_f32 : (tensor<2x3x1xf32>) -> tensor<2x3x1xbf16>
  %variance = stablehlo.select %valid_variance, %variance_bf16, %nan : tensor<i1>, tensor<2x3x1xbf16>
  %mean_broadcast_bf16 = stablehlo.broadcast_in_dim %mean_bf16, dims = [0, 1, 2] : (tensor<2x3x1xbf16>) -> tensor<2x3x384xbf16>
  %centered = stablehlo.subtract %input, %mean_broadcast_bf16 : tensor<2x3x384xbf16>
  %variance_epsilon = stablehlo.add %variance, %epsilon : tensor<2x3x1xbf16>
  %rstd = stablehlo.rsqrt %variance_epsilon : tensor<2x3x1xbf16>
  %rstd_divided = stablehlo.divide %rstd, %variance_epsilon : tensor<2x3x1xbf16>
  %negative_half_rstd = stablehlo.multiply %negative_half, %rstd_divided : tensor<2x3x1xbf16>
  %rstd_broadcast = stablehlo.broadcast_in_dim %rstd, dims = [0, 1, 2] : (tensor<2x3x1xbf16>) -> tensor<2x3x384xbf16>
  %normalized = stablehlo.multiply %centered, %rstd_broadcast : tensor<2x3x384xbf16>
  %gamma_broadcast = stablehlo.broadcast_in_dim %gamma, dims = [2] : (tensor<384xbf16>) -> tensor<2x3x384xbf16>
  %scaled = stablehlo.multiply %normalized, %gamma_broadcast : tensor<2x3x384xbf16>
  %beta_broadcast = stablehlo.broadcast_in_dim %beta, dims = [2] : (tensor<384xbf16>) -> tensor<2x3x384xbf16>
  %output = stablehlo.add %scaled, %beta_broadcast : tensor<2x3x384xbf16>
  %output_grad = stablehlo.negate %incoming_grad : tensor<2x3x384xbf16>

  %beta_first = stablehlo.reduce(%output_grad init: %zero_bf16) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x3x384xbf16>, tensor<bf16>) -> tensor<384xbf16>
  %beta_reshape = stablehlo.reshape %beta_first : (tensor<384xbf16>) -> tensor<1x1x384xbf16>
  %beta_grad = stablehlo.reduce(%beta_reshape init: %zero_bf16) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x384xbf16>, tensor<bf16>) -> tensor<384xbf16>
  %dgamma_input = stablehlo.multiply %normalized, %output_grad : tensor<2x3x384xbf16>
  %gamma_first = stablehlo.reduce(%dgamma_input init: %zero_bf16) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x3x384xbf16>, tensor<bf16>) -> tensor<384xbf16>
  %gamma_reshape = stablehlo.reshape %gamma_first : (tensor<384xbf16>) -> tensor<1x1x384xbf16>
  %gamma_grad = stablehlo.reduce(%gamma_reshape init: %zero_bf16) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x1x384xbf16>, tensor<bf16>) -> tensor<384xbf16>

  %scaled_output_grad = stablehlo.multiply %output_grad, %gamma_broadcast : tensor<2x3x384xbf16>
  %centered_scaled_grad = stablehlo.multiply %centered, %scaled_output_grad : tensor<2x3x384xbf16>
  %centered_grad_sum = stablehlo.reduce(%centered_scaled_grad init: %zero_bf16) applies stablehlo.add across dimensions = [2] : (tensor<2x3x384xbf16>, tensor<bf16>) -> tensor<2x3xbf16>
  %centered_grad_reshape = stablehlo.reshape %centered_grad_sum : (tensor<2x3xbf16>) -> tensor<2x3x1xbf16>
  %direct_grad = stablehlo.multiply %scaled_output_grad, %rstd_broadcast : tensor<2x3x384xbf16>
  %variance_seed = stablehlo.multiply %centered_grad_reshape, %negative_half_rstd : tensor<2x3x1xbf16>
  %selected_variance_seed = stablehlo.select %valid_variance, %variance_seed, %zero_small_bf16 : tensor<i1>, tensor<2x3x1xbf16>
  %variance_seed_f32 = stablehlo.convert %selected_variance_seed : (tensor<2x3x1xbf16>) -> tensor<2x3x1xf32>
  %divided_variance_seed = stablehlo.divide %variance_seed_f32, %denominator_broadcast : tensor<2x3x1xf32>
  %variance_seed_sum = stablehlo.reduce(%divided_variance_seed init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x3xf32>
  %variance_seed_broadcast = stablehlo.broadcast_in_dim %variance_seed_sum, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x384xf32>
  %variance_product = stablehlo.multiply %variance_seed_broadcast, %twice_centered : tensor<2x3x384xf32>
  %negative_variance_product = stablehlo.negate %variance_product : tensor<2x3x384xf32>
  %variance_mean_sum = stablehlo.reduce(%negative_variance_product init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x384xf32>, tensor<f32>) -> tensor<2x3xf32>
  %variance_mean_reshape = stablehlo.reshape %variance_mean_sum : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
  %variance_mean = stablehlo.divide %variance_mean_reshape, %feature_count : tensor<2x3x1xf32>
  %variance_mean_sum_2 = stablehlo.reduce(%variance_mean init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x3xf32>
  %variance_mean_broadcast = stablehlo.broadcast_in_dim %variance_mean_sum_2, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x384xf32>
  %variance_add = stablehlo.add %variance_product, %variance_mean_broadcast : tensor<2x3x384xf32>
  %variance_contribution = stablehlo.convert %variance_add : (tensor<2x3x384xf32>) -> tensor<2x3x384xbf16>

  %negative_direct = stablehlo.negate %direct_grad : tensor<2x3x384xbf16>
  %direct_mean_sum = stablehlo.reduce(%negative_direct init: %zero_bf16) applies stablehlo.add across dimensions = [2] : (tensor<2x3x384xbf16>, tensor<bf16>) -> tensor<2x3xbf16>
  %direct_mean_f32 = stablehlo.convert %direct_mean_sum : (tensor<2x3xbf16>) -> tensor<2x3xf32>
  %direct_mean_reshape = stablehlo.reshape %direct_mean_f32 : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
  %direct_mean = stablehlo.divide %direct_mean_reshape, %feature_count : tensor<2x3x1xf32>
  %direct_mean_sum_2 = stablehlo.reduce(%direct_mean init: %zero_f32) applies stablehlo.add across dimensions = [2] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x3xf32>
  %direct_mean_broadcast_f32 = stablehlo.broadcast_in_dim %direct_mean_sum_2, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3x384xf32>
  %direct_mean_broadcast = stablehlo.convert %direct_mean_broadcast_f32 : (tensor<2x3x384xf32>) -> tensor<2x3x384xbf16>
  %direct_add = stablehlo.add %direct_grad, %variance_contribution : tensor<2x3x384xbf16>
  %input_grad = stablehlo.add %direct_add, %direct_mean_broadcast : tensor<2x3x384xbf16>
  return %output, %input_grad, %gamma_grad, %beta_grad : tensor<2x3x384xbf16>, tensor<2x3x384xbf16>, tensor<384xbf16>, tensor<384xbf16>
}
