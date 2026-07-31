// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(canonicalize,iree-stablehlo-canonicalize,cse))" %s | FileCheck %s --check-prefix=CURRENT
// RUN: env IREE_METAL_DISABLE_NATIVE_ATTENTION=1 iree-opt --split-input-file --pass-pipeline="builtin.module(inline,func.func(canonicalize,iree-stablehlo-canonicalize,cse),iree-stablehlo-convert-flash-attention-dispatch)" %s | FileCheck %s --check-prefix=OPTION-OFF
// RUN: env IREE_METAL_COOP_RAISE_FLASH=1 iree-opt --split-input-file --pass-pipeline="builtin.module(inline,func.func(canonicalize,iree-stablehlo-canonicalize,cse),iree-stablehlo-convert-flash-attention-dispatch{suppress-legacy-attention-raise=true})" %s | FileCheck %s --check-prefix=OPTION-OFF
// RUN: env IREE_METAL_DISABLE_NATIVE_ATTENTION=1 iree-opt --split-input-file --pass-pipeline="builtin.module(inline,func.func(canonicalize,iree-stablehlo-canonicalize,cse),iree-stablehlo-convert-flash-attention-dispatch{raise-native-attention=true},func.func(canonicalize,iree-stablehlo-canonicalize,cse))" %s | FileCheck %s --check-prefixes=RAISED,NEGATIVE
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s | FileCheck %s --check-prefix=PIPELINE-OFF
// RUN: env IREE_METAL_DISABLE_NATIVE_ATTENTION=1 iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline="enable-native-attention=true" %s | FileCheck %s --check-prefix=PIPELINE-OFF
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline="enable-native-attention=true" %s | FileCheck %s --check-prefixes=PIPELINE,PIPELINE-NEGATIVE
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION IREE_METAL_ATTN_PAD_SEQUENCE=16 iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline="enable-native-attention=true" %s | FileCheck %s --check-prefix=PADDED

// This is a compact JAX 0.6.1 lowering of the bf16 causal attention used by
// iree-fork-bench/bench.py, including its value_and_grad graph. B, H, T, and D
// are reduced to 2, 2, 8, and 8, but the rank-4 dot dimension numbers, outlined
// causal where, mixed-precision softmax reduction, and full softmax VJP spelling
// are the same as the GPT2-small B=8, H=12, T=512, D=64 graph.
//
// RAISED-LABEL: func.func public @main(
// RAISED-SAME: %[[Q:[^:]+]]: tensor<2x2x8x8xbf16>,
// RAISED-SAME: %[[K:[^:]+]]: tensor<2x2x8x8xbf16>,
// RAISED-SAME: %[[V:[^:]+]]: tensor<2x2x8x8xbf16>,
// RAISED-SAME: {{%[^:]+}}: tensor<2x2x8x8xbf16>)
// RAISED: %[[SCALE:.+]] = arith.constant {{.*}} : bf16
// RAISED: %[[MASK:.+]] = stablehlo.select
// RAISED-NOT: stablehlo.exponential
// RAISED-NOT: stablehlo.dot_general
// RAISED: %[[ATTN:.+]]:2 = iree_linalg_ext.attention
// RAISED-SAME: decomposition_config = {use_exp2 = false}
// RAISED: ins(%[[Q]], %[[K]], %[[V]], %[[SCALE]], %[[MASK]]
// RAISED: %[[BWD:.+]]:3 = iree_linalg_ext.attention_backward
// RAISED-SAME: decomposition_config = {use_exp2 = false}
// RAISED: ins(%[[Q]], %[[K]], %[[V]], %[[ATTN]]#0, {{%[^,]+}}, %[[ATTN]]#1, %[[SCALE]], %[[MASK]]
// RAISED: return {{.*}}%[[BWD]]#0, %[[BWD]]#1, %[[BWD]]#2

// NEGATIVE-LABEL: func.func public @unsupported_native_shape
// NEGATIVE-NOT: iree_linalg_ext.attention
// NEGATIVE: stablehlo.dot_general
// NEGATIVE-NOT: iree_linalg_ext.attention
// NEGATIVE: return

// OPTION-OFF-LABEL: func.func public @main(
// OPTION-OFF-NOT: iree_linalg_ext.attention
// OPTION-OFF: stablehlo.exponential
// OPTION-OFF-NOT: iree_linalg_ext.attention
// OPTION-OFF: return

// PIPELINE-OFF-LABEL: func.func public @main(
// PIPELINE-OFF-NOT: iree_linalg_ext.attention
// PIPELINE-OFF: linalg.batch_matmul
// PIPELINE-OFF: math.exp
// PIPELINE-OFF-NOT: iree_linalg_ext.attention
// PIPELINE-OFF: return

// PIPELINE-LABEL: func.func public @main(
// PIPELINE-NOT: stablehlo.exponential
// PIPELINE: iree_linalg_ext.attention
// PIPELINE: iree_linalg_ext.attention_backward
// PIPELINE-NOT: stablehlo.exponential
// PIPELINE: return

// PIPELINE-NEGATIVE-LABEL: func.func public @unsupported_native_shape
// PIPELINE-NEGATIVE-NOT: iree_linalg_ext.attention
// PIPELINE-NEGATIVE: linalg.batch_matmul
// PIPELINE-NEGATIVE-NOT: iree_linalg_ext.attention
// PIPELINE-NEGATIVE: return

// CURRENT-LABEL: func.func public @main(
// CURRENT-SAME: %[[Q:[^:]+]]: tensor<2x2x8x8xbf16>,
// CURRENT-SAME: %[[K:[^:]+]]: tensor<2x2x8x8xbf16>,
// CURRENT-SAME: %[[V:[^:]+]]: tensor<2x2x8x8xbf16>,
// CURRENT-SAME: %[[DO:[^:]+]]: tensor<2x2x8x8xbf16>)
// CURRENT: [[KT:%[0-9]+]] = stablehlo.transpose %[[K]], dims = [0, 1, 3, 2]
// CURRENT: [[QK:%[0-9]+]] = stablehlo.dot_general %[[Q]], [[KT]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2]
// CURRENT: [[SCALED:%[0-9]+]] = stablehlo.multiply [[QK]],
// CURRENT: [[MASKED:%[0-9]+]]:2 = call @_where({{%[0-9]+}}, [[SCALED]],
// CURRENT: [[ROW_MAX:%[0-9]+]] = stablehlo.reduce([[MASKED]]#0
// CURRENT-SAME: applies stablehlo.maximum across dimensions = [3]
// CURRENT: [[CENTERED:%[0-9]+]] = stablehlo.subtract [[MASKED]]#0,
// CURRENT: [[EXP:%[0-9]+]] = stablehlo.exponential [[CENTERED]]
// CURRENT: [[EXP_F32:%[0-9]+]] = stablehlo.convert [[EXP]]
// CURRENT: [[SUM:%[0-9]+]] = stablehlo.reduce([[EXP_F32]]
// CURRENT-SAME: applies stablehlo.add across dimensions = [3]
// CURRENT: [[P:%[0-9]+]] = stablehlo.divide [[EXP]],
// CURRENT: [[O:%[0-9]+]] = stablehlo.dot_general [[P]], %[[V]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2]
// CURRENT: [[DO_F32:%[0-9]+]] = stablehlo.convert %[[DO]]
// CURRENT: [[DO_SCALED_F32:%[0-9]+]] = stablehlo.multiply {{%cst[^, ]*}}, [[DO_F32]]
// CURRENT: [[DO_BF16:%[0-9]+]] = stablehlo.convert [[DO_SCALED_F32]]
// CURRENT-SAME: -> tensor<2x2x8x8xbf16>
// CURRENT: [[DVT:%[0-9]+]] = stablehlo.dot_general [[DO_BF16]], [[P]], batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2]
// CURRENT: [[DV:%[0-9]+]] = stablehlo.transpose [[DVT]], dims = [0, 1, 3, 2]
// CURRENT: [[DP:%[0-9]+]] = stablehlo.dot_general [[DO_BF16]], %[[V]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3]
// CURRENT: [[DP_DIV_SUM:%[0-9]+]] = stablehlo.divide [[DP]],
// CURRENT: [[SOFTMAX_VJP:%[0-9]+]] = stablehlo.add [[DP_DIV_SUM]],
// CURRENT: [[DS_UNMASKED:%[0-9]+]] = stablehlo.multiply [[SOFTMAX_VJP]], [[EXP]]
// CURRENT: [[DS_MASKED:%[0-9]+]] = call @_where_0([[MASKED]]#1, [[DS_UNMASKED]])
// CURRENT: [[DS:%[0-9]+]] = stablehlo.multiply [[DS_MASKED]],
// CURRENT: [[DKT:%[0-9]+]] = stablehlo.dot_general [[DS]], %[[Q]], batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2]
// CURRENT: [[DQ:%[0-9]+]] = stablehlo.dot_general [[DS]], [[KT]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3]
// CURRENT: return {{%[0-9]+}}, [[DQ]], {{%[0-9]+}}, [[DV]]

// CURRENT-LABEL: func.func private @tril
// CURRENT: stablehlo.iota dim = 0
// CURRENT: stablehlo.iota dim = 1
// CURRENT: stablehlo.compare GE

// CURRENT-LABEL: func.func private @_where(
// CURRENT: [[PRED:%[0-9]+]] = stablehlo.broadcast_in_dim
// CURRENT: [[SELECTED:%[0-9]+]] = stablehlo.select [[PRED]],
// CURRENT: return [[SELECTED]], [[PRED]]

// CURRENT-LABEL: func.func private @_where_0(
// CURRENT: stablehlo.constant dense<0.000000e+00> : tensor<2x2x8x8xbf16>
// CURRENT: stablehlo.select

// This is the same paired forward/VJP spelling emitted by the encoder models,
// with no causal mask. The colon immediately after the scale in each input
// list proves that the optional mask and its indexing map are both absent.
//
// RAISED-LABEL: func.func public @unmasked_main(
// RAISED-SAME: %[[UQ:[^:]+]]: tensor<2x2x8x8xbf16>,
// RAISED-SAME: %[[UK:[^:]+]]: tensor<2x2x8x8xbf16>,
// RAISED-SAME: %[[UV:[^:]+]]: tensor<2x2x8x8xbf16>)
// RAISED: %[[USCALE:.+]] = arith.constant {{.*}} : bf16
// RAISED-NOT: stablehlo.exponential
// RAISED-NOT: stablehlo.dot_general
// RAISED: %[[UATTN:.+]]:2 = iree_linalg_ext.attention
// RAISED-SAME: decomposition_config = {use_exp2 = false}
// RAISED: ins(%[[UQ]], %[[UK]], %[[UV]], %[[USCALE]] :
// RAISED: %[[UBWD:.+]]:3 = iree_linalg_ext.attention_backward
// RAISED-SAME: decomposition_config = {use_exp2 = false}
// RAISED: ins(%[[UQ]], %[[UK]], %[[UV]], %[[UATTN]]#0, {{%[^,]+}}, %[[UATTN]]#1, %[[USCALE]] :
// RAISED: return {{.*}}%[[UBWD]]#0, %[[UBWD]]#1, %[[UBWD]]#2

// OPTION-OFF-LABEL: func.func public @unmasked_main(
// OPTION-OFF-NOT: iree_linalg_ext.attention
// OPTION-OFF: stablehlo.exponential
// OPTION-OFF-NOT: iree_linalg_ext.attention
// OPTION-OFF: return

// PIPELINE-OFF-LABEL: func.func public @unmasked_main(
// PIPELINE-OFF-NOT: iree_linalg_ext.attention
// PIPELINE-OFF: linalg.batch_matmul
// PIPELINE-OFF: math.exp
// PIPELINE-OFF-NOT: iree_linalg_ext.attention
// PIPELINE-OFF: return

// PIPELINE-LABEL: func.func public @unmasked_main(
// PIPELINE-NOT: stablehlo.exponential
// PIPELINE: iree_linalg_ext.attention
// PIPELINE: iree_linalg_ext.attention_backward
// PIPELINE-NOT: stablehlo.exponential
// PIPELINE: return

// PADDED-DAG: #[[KEY_MASK_MAP:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
// PADDED-LABEL: func.func public @unmasked_main(
// PADDED: %[[KEY_MASK:.+]] = arith.constant dense<
// PADDED-SAME: tensor<16xi1>
// PADDED-DAG: %[[PAD_Q:.+]] = tensor.pad %arg0
// PADDED-SAME: high[0, 0, 8, 0]
// PADDED-DAG: %[[PAD_K:.+]] = tensor.pad %arg1
// PADDED-SAME: high[0, 0, 8, 0]
// PADDED-DAG: %[[PAD_V:.+]] = tensor.pad %arg2
// PADDED-SAME: high[0, 0, 8, 0]
// PADDED: %[[ATTN:.+]]:2 = iree_linalg_ext.attention
// PADDED-SAME: indexing_maps = [{{.*}}#[[KEY_MASK_MAP]], {{.*}}]
// PADDED-SAME: ins(%[[PAD_Q]], %[[PAD_K]], %[[PAD_V]],
// PADDED-SAME: %[[KEY_MASK]]
// PADDED-SAME: tensor<2x2x16x8xbf16>
// PADDED-SAME: tensor<16xi1>
// PADDED: %[[ATTN_BARRIER:.+]]:2 = util.optimization_barrier %[[ATTN]]#0, %[[ATTN]]#1
// PADDED: tensor.extract_slice
// PADDED-SAME: tensor<2x2x16x8xbf16> to tensor<2x2x8x8xbf16>
// PADDED: iree_linalg_ext.attention_backward
// PADDED-SAME: indexing_maps = [{{.*}}#[[KEY_MASK_MAP]], {{.*}}]
// PADDED-SAME: ins({{.*}}%[[KEY_MASK]]
// PADDED-SAME: tensor<2x2x16x8xbf16>
// PADDED: %[[GRAD_BARRIER:.+]]:3 = util.optimization_barrier
// PADDED: tensor.extract_slice %[[GRAD_BARRIER]]#0
// PADDED: tensor.extract_slice %[[GRAD_BARRIER]]#1
// PADDED: tensor.extract_slice %[[GRAD_BARRIER]]#2
// PADDED: return

// An eligible unmasked forward without its shared VJP must remain untouched.
// NEGATIVE-LABEL: func.func public @unmasked_incomplete_backward
// NEGATIVE-NOT: iree_linalg_ext.attention
// NEGATIVE: stablehlo.dot_general
// NEGATIVE: stablehlo.exponential
// NEGATIVE-NOT: iree_linalg_ext.attention
// NEGATIVE: return

// PIPELINE-NEGATIVE-LABEL: func.func public @unmasked_incomplete_backward
// PIPELINE-NEGATIVE-NOT: iree_linalg_ext.attention
// PIPELINE-NEGATIVE: linalg.batch_matmul
// PIPELINE-NEGATIVE-NOT: iree_linalg_ext.attention
// PIPELINE-NEGATIVE: return

module @jit_loss attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x2x8x8xbf16>, %arg1: tensor<2x2x8x8xbf16>, %arg2: tensor<2x2x8x8xbf16>, %arg3: tensor<2x2x8x8xbf16>) -> (tensor<f32> {jax.result_info = "result[0]"}, tensor<2x2x8x8xbf16> {jax.result_info = "result[1][0]"}, tensor<2x2x8x8xbf16> {jax.result_info = "result[1][1]"}, tensor<2x2x8x8xbf16> {jax.result_info = "result[1][2]"}) {
    %0 = stablehlo.transpose %arg1, dims = [0, 1, 3, 2] : (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %1 = stablehlo.dot_general %arg0, %0, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %cst = stablehlo.constant dense<3.535160e-01> : tensor<bf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x2x8x8xbf16>
    %3 = stablehlo.multiply %1, %2 : tensor<2x2x8x8xbf16>
    %c = stablehlo.constant dense<true> : tensor<i1>
    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<8x8xi1>
    %5 = call @tril(%4) : (tensor<8x8xi1>) -> tensor<8x8xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [2, 3] : (tensor<8x8xi1>) -> tensor<1x1x8x8xi1>
    %cst_0 = stablehlo.constant dense<-3.389530e+38> : tensor<bf16>
    %7:2 = call @_where(%6, %3, %cst_0) : (tensor<1x1x8x8xi1>, tensor<2x2x8x8xbf16>, tensor<bf16>) -> (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xi1>)
    %cst_1 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %8 = stablehlo.reduce(%7#0 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<2x2x8x8xbf16>, tensor<bf16>) -> tensor<2x2x8xbf16>
    %cst_2 = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %9 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<bf16>) -> tensor<2x2x8xbf16>
    %10 = stablehlo.maximum %9, %8 : tensor<2x2x8xbf16>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<2x2x8xbf16>) -> tensor<2x2x8x1xbf16>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2, 3] : (tensor<2x2x8x1xbf16>) -> tensor<2x2x8x8xbf16>
    %13 = stablehlo.subtract %7#0, %12 : tensor<2x2x8x8xbf16>
    %14 = stablehlo.exponential %13 : tensor<2x2x8x8xbf16>
    %15 = stablehlo.convert %14 : (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.reduce(%15 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<2x2x8x8xf32>, tensor<f32>) -> tensor<2x2x8xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<2x2x8xf32>) -> tensor<2x2x8x1xf32>
    %18 = stablehlo.convert %17 : (tensor<2x2x8x1xf32>) -> tensor<2x2x8x1xbf16>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2, 3] : (tensor<2x2x8x1xbf16>) -> tensor<2x2x8x8xbf16>
    %20 = stablehlo.divide %14, %19 : tensor<2x2x8x8xbf16>
    %21 = stablehlo.multiply %18, %18 : tensor<2x2x8x1xbf16>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %22 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<bf16>) -> tensor<2x2x8x1xbf16>
    %23 = stablehlo.divide %22, %21 : tensor<2x2x8x1xbf16>
    %24 = stablehlo.dot_general %20, %arg2, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %25 = stablehlo.convert %24 : (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xf32>
    %26 = stablehlo.convert %arg3 : (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<2x2x8x8xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.reduce(%27 init: %cst_5) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<2x2x8x8xf32>, tensor<f32>) -> tensor<f32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<2x2x8x8xf32>
    %30 = stablehlo.multiply %29, %26 : tensor<2x2x8x8xf32>
    %31 = stablehlo.convert %30 : (tensor<2x2x8x8xf32>) -> tensor<2x2x8x8xbf16>
    %32 = stablehlo.dot_general %31, %20, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %33 = stablehlo.transpose %32, dims = [0, 1, 3, 2] : (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %34 = stablehlo.dot_general %31, %arg2, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %35 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2, 3] : (tensor<2x2x8x1xbf16>) -> tensor<2x2x8x8xbf16>
    %36 = stablehlo.multiply %34, %35 : tensor<2x2x8x8xbf16>
    %37 = stablehlo.multiply %36, %14 : tensor<2x2x8x8xbf16>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %38 = stablehlo.reduce(%37 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<2x2x8x8xbf16>, tensor<bf16>) -> tensor<2x2x8xbf16>
    %39 = stablehlo.reshape %38 : (tensor<2x2x8xbf16>) -> tensor<2x2x8x1xbf16>
    %40 = stablehlo.negate %39 : tensor<2x2x8x1xbf16>
    %41 = stablehlo.broadcast_in_dim %18, dims = [0, 1, 2, 3] : (tensor<2x2x8x1xbf16>) -> tensor<2x2x8x8xbf16>
    %42 = stablehlo.divide %34, %41 : tensor<2x2x8x8xbf16>
    %43 = stablehlo.convert %40 : (tensor<2x2x8x1xbf16>) -> tensor<2x2x8x1xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %44 = stablehlo.reduce(%43 init: %cst_8) applies stablehlo.add across dimensions = [3] : (tensor<2x2x8x1xf32>, tensor<f32>) -> tensor<2x2x8xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2] : (tensor<2x2x8xf32>) -> tensor<2x2x8x8xf32>
    %46 = stablehlo.convert %45 : (tensor<2x2x8x8xf32>) -> tensor<2x2x8x8xbf16>
    %47 = stablehlo.add %42, %46 : tensor<2x2x8x8xbf16>
    %48 = stablehlo.multiply %47, %14 : tensor<2x2x8x8xbf16>
    %49 = call @_where_0(%7#1, %48) : (tensor<2x2x8x8xi1>, tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %50 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x2x8x8xbf16>
    %51 = stablehlo.multiply %49, %50 : tensor<2x2x8x8xbf16>
    %52 = stablehlo.dot_general %51, %arg0, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %53 = stablehlo.transpose %52, dims = [0, 1, 3, 2] : (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %54 = stablehlo.dot_general %51, %0, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %55 = stablehlo.transpose %53, dims = [0, 1, 3, 2] : (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    return %28, %54, %55, %33 : tensor<f32>, tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>
  }

  // This is a complete canonical forward/VJP spelling at a sequence length
  // too small for the native Apple schedule. It must remain as StableHLO so
  // the ordinary portable pipeline can compile it.
  func.func public @unsupported_native_shape(
      %q: tensor<2x2x4x8xbf16>, %k: tensor<2x2x4x8xbf16>,
      %v: tensor<2x2x4x8xbf16>, %do: tensor<2x2x4x8xbf16>)
      -> (tensor<2x2x4x8xbf16>, tensor<2x2x4x8xbf16>,
          tensor<2x2x4x8xbf16>, tensor<2x2x4x8xbf16>) {
    %false = stablehlo.constant dense<false> : tensor<4x4xi1>
    %true = stablehlo.constant dense<true> : tensor<4x4xi1>
    %scale = stablehlo.constant dense<3.535160e-01> :
        tensor<2x2x4x4xbf16>
    %backward_scale = stablehlo.constant dense<3.535160e-01> :
        tensor<2x2x4x4xbf16>
    %lowest = stablehlo.constant dense<-3.389530e+38> :
        tensor<2x2x4x4xbf16>
    %neg_inf_rows = stablehlo.constant dense<0xFF80> :
        tensor<2x2x4xbf16>
    %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %zero_bf16 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %zero_scores = stablehlo.constant dense<0.000000e+00> :
        tensor<2x2x4x4xbf16>
    %one = stablehlo.constant dense<1.000000e+00> :
        tensor<2x2x4x1xbf16>
    %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] :
        (tensor<2x2x4x8xbf16>) -> tensor<2x2x8x4xbf16>
    %qk = stablehlo.dot_general %q, %kt,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x4x8xbf16>, tensor<2x2x8x4xbf16>)
        -> tensor<2x2x4x4xbf16>
    %scaled = stablehlo.multiply %qk, %scale : tensor<2x2x4x4xbf16>
    %row = stablehlo.iota dim = 0 : tensor<4x4xi32>
    %column = stablehlo.iota dim = 1 : tensor<4x4xi32>
    %ge = stablehlo.compare GE, %row, %column, SIGNED :
        (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
    %causal = stablehlo.select %ge, %true, %false :
        tensor<4x4xi1>, tensor<4x4xi1>
    %mask = stablehlo.broadcast_in_dim %causal, dims = [2, 3] :
        (tensor<4x4xi1>) -> tensor<2x2x4x4xi1>
    %masked = stablehlo.select %mask, %scaled, %lowest :
        tensor<2x2x4x4xi1>, tensor<2x2x4x4xbf16>
    %row_max = stablehlo.reduce(%masked init: %neg_inf)
        applies stablehlo.maximum across dimensions = [3] :
        (tensor<2x2x4x4xbf16>, tensor<bf16>) -> tensor<2x2x4xbf16>
    %clamped_max = stablehlo.maximum %neg_inf_rows, %row_max :
        tensor<2x2x4xbf16>
    %max_broadcast = stablehlo.broadcast_in_dim %clamped_max,
        dims = [0, 1, 2] :
        (tensor<2x2x4xbf16>) -> tensor<2x2x4x4xbf16>
    %centered = stablehlo.subtract %masked, %max_broadcast :
        tensor<2x2x4x4xbf16>
    %exp = stablehlo.exponential %centered : tensor<2x2x4x4xbf16>
    %exp_f32 = stablehlo.convert %exp :
        (tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xf32>
    %sum = stablehlo.reduce(%exp_f32 init: %zero_f32)
        applies stablehlo.add across dimensions = [3] :
        (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %sum_bf16 = stablehlo.convert %sum :
        (tensor<2x2x4xf32>) -> tensor<2x2x4xbf16>
    %sum_4d = stablehlo.reshape %sum_bf16 :
        (tensor<2x2x4xbf16>) -> tensor<2x2x4x1xbf16>
    %denominator = stablehlo.broadcast_in_dim %sum_4d,
        dims = [0, 1, 2, 3] :
        (tensor<2x2x4x1xbf16>) -> tensor<2x2x4x4xbf16>
    %probability = stablehlo.divide %exp, %denominator :
        tensor<2x2x4x4xbf16>
    %denominator_squared = stablehlo.multiply %sum_4d, %sum_4d :
        tensor<2x2x4x1xbf16>
    %inverse = stablehlo.divide %one, %denominator_squared :
        tensor<2x2x4x1xbf16>
    %output = stablehlo.dot_general %probability, %v,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x4x4xbf16>, tensor<2x2x4x8xbf16>)
        -> tensor<2x2x4x8xbf16>
    %dv_transposed = stablehlo.dot_general %do, %probability,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [2] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x4x8xbf16>, tensor<2x2x4x4xbf16>)
        -> tensor<2x2x8x4xbf16>
    %dv = stablehlo.transpose %dv_transposed, dims = [0, 1, 3, 2] :
        (tensor<2x2x8x4xbf16>) -> tensor<2x2x4x8xbf16>
    %dp = stablehlo.dot_general %do, %v,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x4x8xbf16>, tensor<2x2x4x8xbf16>)
        -> tensor<2x2x4x4xbf16>
    %inverse_broadcast = stablehlo.broadcast_in_dim %inverse,
        dims = [0, 1, 2, 3] :
        (tensor<2x2x4x1xbf16>) -> tensor<2x2x4x4xbf16>
    %dp_times_inverse = stablehlo.multiply %dp, %inverse_broadcast :
        tensor<2x2x4x4xbf16>
    %weighted = stablehlo.multiply %dp_times_inverse, %exp :
        tensor<2x2x4x4xbf16>
    %weighted_reduce = stablehlo.reduce(%weighted init: %zero_bf16)
        applies stablehlo.add across dimensions = [3] :
        (tensor<2x2x4x4xbf16>, tensor<bf16>) -> tensor<2x2x4xbf16>
    %negative = stablehlo.negate %weighted_reduce : tensor<2x2x4xbf16>
    %negative_f32 = stablehlo.convert %negative :
        (tensor<2x2x4xbf16>) -> tensor<2x2x4xf32>
    %negative_4d = stablehlo.reshape %negative_f32 :
        (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %singleton = stablehlo.reduce(%negative_4d init: %zero_f32)
        applies stablehlo.add across dimensions = [3] :
        (tensor<2x2x4x1xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %correction_f32 = stablehlo.broadcast_in_dim %singleton,
        dims = [0, 1, 2] :
        (tensor<2x2x4xf32>) -> tensor<2x2x4x4xf32>
    %correction = stablehlo.convert %correction_f32 :
        (tensor<2x2x4x4xf32>) -> tensor<2x2x4x4xbf16>
    %direct = stablehlo.divide %dp, %denominator :
        tensor<2x2x4x4xbf16>
    %vjp = stablehlo.add %direct, %correction : tensor<2x2x4x4xbf16>
    %unmasked_ds = stablehlo.multiply %vjp, %exp :
        tensor<2x2x4x4xbf16>
    %masked_ds = stablehlo.select %mask, %unmasked_ds, %zero_scores :
        tensor<2x2x4x4xi1>, tensor<2x2x4x4xbf16>
    %ds = stablehlo.multiply %masked_ds, %backward_scale :
        tensor<2x2x4x4xbf16>
    %dk_seed = stablehlo.dot_general %ds, %q,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [2] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x4x4xbf16>, tensor<2x2x4x8xbf16>)
        -> tensor<2x2x4x8xbf16>
    %dk_transposed = stablehlo.transpose %dk_seed, dims = [0, 1, 3, 2] :
        (tensor<2x2x4x8xbf16>) -> tensor<2x2x8x4xbf16>
    %dq = stablehlo.dot_general %ds, %kt,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x4x4xbf16>, tensor<2x2x8x4xbf16>)
        -> tensor<2x2x4x8xbf16>
    %dk = stablehlo.transpose %dk_transposed, dims = [0, 1, 3, 2] :
        (tensor<2x2x8x4xbf16>) -> tensor<2x2x4x8xbf16>
    return %output, %dq, %dk, %dv :
        tensor<2x2x4x8xbf16>, tensor<2x2x4x8xbf16>,
        tensor<2x2x4x8xbf16>, tensor<2x2x4x8xbf16>
  }

  func.func private @tril(%arg0: tensor<8x8xi1>) -> tensor<8x8xi1> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32>
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %c_0 = stablehlo.constant dense<false> : tensor<i1>
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i1>) -> tensor<8x8xi1>
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xi1>
    return %6 : tensor<8x8xi1>
  }
  func.func private @_where(%arg0: tensor<1x1x8x8xi1>, %arg1: tensor<2x2x8x8xbf16>, %arg2: tensor<bf16>) -> (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xi1>) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x1x8x8xi1>) -> tensor<2x2x8x8xi1>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<bf16>) -> tensor<2x2x8x8xbf16>
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x2x8x8xi1>, tensor<2x2x8x8xbf16>
    return %2, %0 : tensor<2x2x8x8xbf16>, tensor<2x2x8x8xi1>
  }
  func.func private @_where_0(%arg0: tensor<2x2x8x8xi1>, %arg1: tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2x2x8x8xbf16>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<2x2x8x8xi1>, tensor<2x2x8x8xbf16>
    return %1 : tensor<2x2x8x8xbf16>
  }
}

// -----

module @jit_unmasked attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @unmasked_main(
      %q: tensor<2x2x8x8xbf16>, %k: tensor<2x2x8x8xbf16>,
      %v: tensor<2x2x8x8xbf16>)
      -> (tensor<f32>, tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>,
          tensor<2x2x8x8xbf16>) {
    %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] :
        (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %qk = stablehlo.dot_general %q, %kt,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>)
        -> tensor<2x2x8x8xbf16>
    %scale_scalar = stablehlo.constant dense<3.535160e-01> : tensor<bf16>
    %scale = stablehlo.broadcast_in_dim %scale_scalar, dims = [] :
        (tensor<bf16>) -> tensor<2x2x8x8xbf16>
    %scaled = stablehlo.multiply %qk, %scale : tensor<2x2x8x8xbf16>
    %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %row_max = stablehlo.reduce(%scaled init: %neg_inf)
        applies stablehlo.maximum across dimensions = [3] :
        (tensor<2x2x8x8xbf16>, tensor<bf16>) -> tensor<2x2x8xbf16>
    %neg_inf_scalar = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %neg_inf_rows = stablehlo.broadcast_in_dim %neg_inf_scalar, dims = [] :
        (tensor<bf16>) -> tensor<2x2x8xbf16>
    %clamped_max = stablehlo.maximum %neg_inf_rows, %row_max :
        tensor<2x2x8xbf16>
    %max_4d = stablehlo.broadcast_in_dim %clamped_max,
        dims = [0, 1, 2] :
        (tensor<2x2x8xbf16>) -> tensor<2x2x8x1xbf16>
    %max_broadcast = stablehlo.broadcast_in_dim %max_4d,
        dims = [0, 1, 2, 3] :
        (tensor<2x2x8x1xbf16>) -> tensor<2x2x8x8xbf16>
    %centered = stablehlo.subtract %scaled, %max_broadcast :
        tensor<2x2x8x8xbf16>
    %exp = stablehlo.exponential %centered : tensor<2x2x8x8xbf16>
    %exp_f32 = stablehlo.convert %exp :
        (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xf32>
    %zero_f32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = stablehlo.reduce(%exp_f32 init: %zero_f32)
        applies stablehlo.add across dimensions = [3] :
        (tensor<2x2x8x8xf32>, tensor<f32>) -> tensor<2x2x8xf32>
    %sum_4d_f32 = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] :
        (tensor<2x2x8xf32>) -> tensor<2x2x8x1xf32>
    %sum_4d = stablehlo.convert %sum_4d_f32 :
        (tensor<2x2x8x1xf32>) -> tensor<2x2x8x1xbf16>
    %denominator = stablehlo.broadcast_in_dim %sum_4d,
        dims = [0, 1, 2, 3] :
        (tensor<2x2x8x1xbf16>) -> tensor<2x2x8x8xbf16>
    %probability = stablehlo.divide %exp, %denominator :
        tensor<2x2x8x8xbf16>
    %denominator_squared = stablehlo.multiply %sum_4d, %sum_4d :
        tensor<2x2x8x1xbf16>
    %one = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %one_4d = stablehlo.broadcast_in_dim %one, dims = [] :
        (tensor<bf16>) -> tensor<2x2x8x1xbf16>
    %inverse = stablehlo.divide %one_4d, %denominator_squared :
        tensor<2x2x8x1xbf16>
    %output = stablehlo.dot_general %probability, %v,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>)
        -> tensor<2x2x8x8xbf16>
    %output_f32 = stablehlo.convert %output :
        (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xf32>
    %squared = stablehlo.multiply %output_f32, %output_f32 :
        tensor<2x2x8x8xf32>
    %two = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %two_broadcast = stablehlo.broadcast_in_dim %two, dims = [] :
        (tensor<f32>) -> tensor<2x2x8x8xf32>
    %twice_output = stablehlo.multiply %two_broadcast, %output_f32 :
        tensor<2x2x8x8xf32>
    %loss_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %loss = stablehlo.reduce(%squared init: %loss_zero)
        applies stablehlo.add across dimensions = [0, 1, 2, 3] :
        (tensor<2x2x8x8xf32>, tensor<f32>) -> tensor<f32>
    %one_f32 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %one_broadcast = stablehlo.broadcast_in_dim %one_f32, dims = [] :
        (tensor<f32>) -> tensor<2x2x8x8xf32>
    %output_grad_f32 = stablehlo.multiply %one_broadcast, %twice_output :
        tensor<2x2x8x8xf32>
    %output_grad = stablehlo.convert %output_grad_f32 :
        (tensor<2x2x8x8xf32>) -> tensor<2x2x8x8xbf16>
    %dv_transposed = stablehlo.dot_general %output_grad, %probability,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [2] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>)
        -> tensor<2x2x8x8xbf16>
    %dv = stablehlo.transpose %dv_transposed, dims = [0, 1, 3, 2] :
        (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %dp = stablehlo.dot_general %output_grad, %v,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>)
        -> tensor<2x2x8x8xbf16>
    %inverse_broadcast = stablehlo.broadcast_in_dim %inverse,
        dims = [0, 1, 2, 3] :
        (tensor<2x2x8x1xbf16>) -> tensor<2x2x8x8xbf16>
    %dp_times_inverse = stablehlo.multiply %dp, %inverse_broadcast :
        tensor<2x2x8x8xbf16>
    %weighted = stablehlo.multiply %dp_times_inverse, %exp :
        tensor<2x2x8x8xbf16>
    %zero_bf16 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %weighted_reduce = stablehlo.reduce(%weighted init: %zero_bf16)
        applies stablehlo.add across dimensions = [3] :
        (tensor<2x2x8x8xbf16>, tensor<bf16>) -> tensor<2x2x8xbf16>
    %negative = stablehlo.negate %weighted_reduce : tensor<2x2x8xbf16>
    %negative_f32 = stablehlo.convert %negative :
        (tensor<2x2x8xbf16>) -> tensor<2x2x8xf32>
    %negative_4d = stablehlo.reshape %negative_f32 :
        (tensor<2x2x8xf32>) -> tensor<2x2x8x1xf32>
    %singleton_zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %singleton = stablehlo.reduce(%negative_4d init: %singleton_zero)
        applies stablehlo.add across dimensions = [3] :
        (tensor<2x2x8x1xf32>, tensor<f32>) -> tensor<2x2x8xf32>
    %correction_f32 = stablehlo.broadcast_in_dim %singleton,
        dims = [0, 1, 2] :
        (tensor<2x2x8xf32>) -> tensor<2x2x8x8xf32>
    %correction = stablehlo.convert %correction_f32 :
        (tensor<2x2x8x8xf32>) -> tensor<2x2x8x8xbf16>
    %direct = stablehlo.divide %dp, %denominator :
        tensor<2x2x8x8xbf16>
    %vjp = stablehlo.add %direct, %correction : tensor<2x2x8x8xbf16>
    %unscaled_ds = stablehlo.multiply %vjp, %exp :
        tensor<2x2x8x8xbf16>
    %backward_scale = stablehlo.broadcast_in_dim %scale_scalar, dims = [] :
        (tensor<bf16>) -> tensor<2x2x8x8xbf16>
    %ds = stablehlo.multiply %unscaled_ds, %backward_scale :
        tensor<2x2x8x8xbf16>
    %dk_seed = stablehlo.dot_general %ds, %q,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [2] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>)
        -> tensor<2x2x8x8xbf16>
    %dk_transposed = stablehlo.transpose %dk_seed, dims = [0, 1, 3, 2] :
        (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %dq = stablehlo.dot_general %ds, %kt,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>)
        -> tensor<2x2x8x8xbf16>
    %dk = stablehlo.transpose %dk_transposed, dims = [0, 1, 3, 2] :
        (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    return %loss, %dq, %dk, %dv :
        tensor<f32>, tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>,
        tensor<2x2x8x8xbf16>
  }
}

// -----

module {
  func.func public @unmasked_incomplete_backward(
      %q: tensor<2x2x8x8xbf16>, %k: tensor<2x2x8x8xbf16>,
      %v: tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16> {
    %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] :
        (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xbf16>
    %qk = stablehlo.dot_general %q, %kt,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>)
        -> tensor<2x2x8x8xbf16>
    %scale = stablehlo.constant dense<3.535160e-01> :
        tensor<2x2x8x8xbf16>
    %scaled = stablehlo.multiply %qk, %scale : tensor<2x2x8x8xbf16>
    %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %row_max = stablehlo.reduce(%scaled init: %neg_inf)
        applies stablehlo.maximum across dimensions = [3] :
        (tensor<2x2x8x8xbf16>, tensor<bf16>) -> tensor<2x2x8xbf16>
    %neg_inf_rows = stablehlo.constant dense<0xFF80> :
        tensor<2x2x8xbf16>
    %clamped_max = stablehlo.maximum %neg_inf_rows, %row_max :
        tensor<2x2x8xbf16>
    %max_broadcast = stablehlo.broadcast_in_dim %clamped_max,
        dims = [0, 1, 2] :
        (tensor<2x2x8xbf16>) -> tensor<2x2x8x8xbf16>
    %centered = stablehlo.subtract %scaled, %max_broadcast :
        tensor<2x2x8x8xbf16>
    %exp = stablehlo.exponential %centered : tensor<2x2x8x8xbf16>
    %exp_f32 = stablehlo.convert %exp :
        (tensor<2x2x8x8xbf16>) -> tensor<2x2x8x8xf32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = stablehlo.reduce(%exp_f32 init: %zero)
        applies stablehlo.add across dimensions = [3] :
        (tensor<2x2x8x8xf32>, tensor<f32>) -> tensor<2x2x8xf32>
    %sum_bf16 = stablehlo.convert %sum :
        (tensor<2x2x8xf32>) -> tensor<2x2x8xbf16>
    %sum_4d = stablehlo.reshape %sum_bf16 :
        (tensor<2x2x8xbf16>) -> tensor<2x2x8x1xbf16>
    %denominator = stablehlo.broadcast_in_dim %sum_4d,
        dims = [0, 1, 2, 3] :
        (tensor<2x2x8x1xbf16>) -> tensor<2x2x8x8xbf16>
    %probability = stablehlo.divide %exp, %denominator :
        tensor<2x2x8x8xbf16>
    %output = stablehlo.dot_general %probability, %v,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x8x8xbf16>, tensor<2x2x8x8xbf16>)
        -> tensor<2x2x8x8xbf16>
    return %output : tensor<2x2x8x8xbf16>
  }
}

// -----

// A complete forward spelling without the complete, shared VJP must remain
// untouched: the paired raise is all-or-nothing.
// NEGATIVE-LABEL: func.func public @incomplete_backward
// NEGATIVE-NOT: iree_linalg_ext.attention
// NEGATIVE: stablehlo.dot_general
// NEGATIVE-NOT: iree_linalg_ext.attention
// NEGATIVE: return
module {
  func.func public @incomplete_backward(
      %q: tensor<2x2x4x8xbf16>, %k: tensor<2x2x4x8xbf16>,
      %v: tensor<2x2x4x8xbf16>) -> tensor<2x2x4x8xbf16> {
    %false = stablehlo.constant dense<false> : tensor<4x4xi1>
    %true = stablehlo.constant dense<true> : tensor<4x4xi1>
    %scale = stablehlo.constant dense<3.535160e-01> : tensor<2x2x4x4xbf16>
    %lowest = stablehlo.constant dense<-3.389530e+38> : tensor<2x2x4x4xbf16>
    %neg_inf_rows = stablehlo.constant dense<0xFF80> : tensor<2x2x4xbf16>
    %neg_inf = stablehlo.constant dense<0xFF80> : tensor<bf16>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %kt = stablehlo.transpose %k, dims = [0, 1, 3, 2] :
        (tensor<2x2x4x8xbf16>) -> tensor<2x2x8x4xbf16>
    %qk = stablehlo.dot_general %q, %kt,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x4x8xbf16>, tensor<2x2x8x4xbf16>)
        -> tensor<2x2x4x4xbf16>
    %scaled = stablehlo.multiply %qk, %scale : tensor<2x2x4x4xbf16>
    %row = stablehlo.iota dim = 0 : tensor<4x4xi32>
    %column = stablehlo.iota dim = 1 : tensor<4x4xi32>
    %ge = stablehlo.compare GE, %row, %column, SIGNED :
        (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
    %causal = stablehlo.select %ge, %true, %false :
        tensor<4x4xi1>, tensor<4x4xi1>
    %mask = stablehlo.broadcast_in_dim %causal, dims = [2, 3] :
        (tensor<4x4xi1>) -> tensor<2x2x4x4xi1>
    %masked = stablehlo.select %mask, %scaled, %lowest :
        tensor<2x2x4x4xi1>, tensor<2x2x4x4xbf16>
    %row_max = stablehlo.reduce(%masked init: %neg_inf)
        applies stablehlo.maximum across dimensions = [3] :
        (tensor<2x2x4x4xbf16>, tensor<bf16>) -> tensor<2x2x4xbf16>
    %clamped_max = stablehlo.maximum %neg_inf_rows, %row_max :
        tensor<2x2x4xbf16>
    %max_broadcast = stablehlo.broadcast_in_dim %clamped_max,
        dims = [0, 1, 2] :
        (tensor<2x2x4xbf16>) -> tensor<2x2x4x4xbf16>
    %centered = stablehlo.subtract %masked, %max_broadcast :
        tensor<2x2x4x4xbf16>
    %exp = stablehlo.exponential %centered : tensor<2x2x4x4xbf16>
    %exp_f32 = stablehlo.convert %exp :
        (tensor<2x2x4x4xbf16>) -> tensor<2x2x4x4xf32>
    %sum = stablehlo.reduce(%exp_f32 init: %zero)
        applies stablehlo.add across dimensions = [3] :
        (tensor<2x2x4x4xf32>, tensor<f32>) -> tensor<2x2x4xf32>
    %sum_bf16 = stablehlo.convert %sum :
        (tensor<2x2x4xf32>) -> tensor<2x2x4xbf16>
    %sum_4d = stablehlo.reshape %sum_bf16 :
        (tensor<2x2x4xbf16>) -> tensor<2x2x4x1xbf16>
    %denominator = stablehlo.broadcast_in_dim %sum_4d,
        dims = [0, 1, 2, 3] :
        (tensor<2x2x4x1xbf16>) -> tensor<2x2x4x4xbf16>
    %probability = stablehlo.divide %exp, %denominator :
        tensor<2x2x4x4xbf16>
    %output = stablehlo.dot_general %probability, %v,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [2],
        precision = [DEFAULT, DEFAULT] :
        (tensor<2x2x4x4xbf16>, tensor<2x2x4x8xbf16>)
        -> tensor<2x2x4x8xbf16>
    return %output : tensor<2x2x4x8xbf16>
  }
}
