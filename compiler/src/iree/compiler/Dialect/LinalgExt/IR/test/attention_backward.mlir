// RUN: iree-opt --split-input-file %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.attention_backward}), canonicalize, cse)" %s | FileCheck %s --check-prefix=DECOMPOSE
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.attention_backward}), canonicalize, cse)" %s | FileCheck %s --check-prefix=CONTRACTIONS

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#lse = affine_map<(b, m, k1, k2, n) -> (b, m)>
#scalar = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (b, m, k2)>

func.func @masked_attention_backward(
    %query: tensor<1x2x3xf32>,
    %key: tensor<1x4x3xf32>,
    %value: tensor<1x4x5xf32>,
    %output: tensor<1x2x5xf32>,
    %output_grad: tensor<1x2x5xf32>,
    %logsumexp: tensor<1x2xf32>,
    %mask_value: tensor<1x2x4xi1>)
    -> (tensor<1x2x3xf32>, tensor<1x4x3xf32>,
        tensor<1x4x5xf32>) {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3xf32>
  %key_grad = tensor.empty() : tensor<1x4x3xf32>
  %value_grad = tensor.empty() : tensor<1x4x5xf32>
  %result:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {use_exp2 = true},
      indexing_maps = [
        #q, #k, #v, #o, #o, #lse, #scalar, #mask, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp,
          %scale, %mask_value :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>,
          tensor<1x2x5xf32>, tensor<1x2x5xf32>, tensor<1x2xf32>,
          f32, tensor<1x2x4xi1>)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>)
      -> tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
}

// ROUNDTRIP-LABEL: func.func @masked_attention_backward
// ROUNDTRIP:       %[[RESULT:.+]]:3 = iree_linalg_ext.attention_backward
// ROUNDTRIP-SAME:    ins(
// ROUNDTRIP-SAME:    tensor<1x2x4xi1>)
// ROUNDTRIP-SAME:    outs(
// ROUNDTRIP-SAME:    -> tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
// ROUNDTRIP:       return %[[RESULT]]#0, %[[RESULT]]#1, %[[RESULT]]#2

// DECOMPOSE-LABEL: func.func @masked_attention_backward
// DECOMPOSE-NOT:   iree_linalg_ext.attention_backward
// DECOMPOSE:       %[[PROB_INIT:[0-9]+]] = tensor.empty() : tensor<1x2x4xf32>
// DECOMPOSE:       %[[QK:[0-9]+]] = linalg.generic{{.*}} ins(%{{[^,]+}}, %{{[^ ]+}} : tensor<1x2x3xf32>, tensor<1x4x3xf32>)
// DECOMPOSE:       %[[QK_BARRIER:[0-9]+]] = util.optimization_barrier %[[QK]]
// DECOMPOSE:       %[[SCORES:[0-9]+]] = linalg.generic{{.*}} ins(%{{[^ ]+}} : tensor<1x2x4xi1>) outs(%[[QK_BARRIER]] : tensor<1x2x4xf32>)
// DECOMPOSE:       %[[SCORE_BARRIER:[0-9]+]] = util.optimization_barrier %[[SCORES]]
// DECOMPOSE:       linalg.generic{{.*}} ins(%{{[^,]+}}, %{{[^,]+}}, %[[SCORE_BARRIER]] : tensor<1x2xf32>, tensor<1x2x4xi1>, tensor<1x2x4xf32>) outs(%[[PROB_INIT]] : tensor<1x2x4xf32>)
// DECOMPOSE:       math.exp2
// DECOMPOSE:       arith.select
// DECOMPOSE:       return %{{.+}}, %{{.+}}, %{{.+}}

// Six contractions: QK, row-dot, dP, dQ, dK, and dV.
// CONTRACTIONS-LABEL: func.func @masked_attention_backward
// CONTRACTIONS-COUNT-6: arith.addf
// CONTRACTIONS:       return

// -----

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#lse = affine_map<(b, m, k1, k2, n) -> (b, m)>
#scalar = affine_map<(b, m, k1, k2, n) -> ()>

func.func @unmasked_attention_backward(
    %query: tensor<1x2x3xf32>,
    %key: tensor<1x4x3xf32>,
    %value: tensor<1x4x5xf32>,
    %output: tensor<1x2x5xf32>,
    %output_grad: tensor<1x2x5xf32>,
    %logsumexp: tensor<1x2xf32>)
    -> (tensor<1x2x3xf32>, tensor<1x4x3xf32>,
        tensor<1x4x5xf32>) {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3xf32>
  %key_grad = tensor.empty() : tensor<1x4x3xf32>
  %value_grad = tensor.empty() : tensor<1x4x5xf32>
  %result:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {use_exp2 = false},
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>,
          tensor<1x2x5xf32>, tensor<1x2x5xf32>, tensor<1x2xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>)
      -> tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
}

// ROUNDTRIP-LABEL: func.func @unmasked_attention_backward
// ROUNDTRIP:       %[[RESULT:.+]]:3 = iree_linalg_ext.attention_backward
// ROUNDTRIP-SAME:    ins(
// ROUNDTRIP-NOT:     tensor<1x2x4xi1>
// ROUNDTRIP-SAME:    outs(
// ROUNDTRIP:       return %[[RESULT]]#0, %[[RESULT]]#1, %[[RESULT]]#2

// DECOMPOSE-LABEL: func.func @unmasked_attention_backward
// DECOMPOSE-NOT:   iree_linalg_ext.attention_backward
// DECOMPOSE-COUNT-1: util.optimization_barrier
// DECOMPOSE:       math.exp
// DECOMPOSE-NOT:   math.exp2
// DECOMPOSE-NOT:   util.optimization_barrier
// DECOMPOSE:       return
