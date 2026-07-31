// RUN: iree-opt --split-input-file --verify-diagnostics %s

#bad_q = affine_map<(b, m, k1, k2, n) -> (b, m, k1 + m)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#lse = affine_map<(b, m, k1, k2, n) -> (b, m)>
#scalar = affine_map<(b, m, k1, k2, n) -> ()>
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>

func.func @non_projected_permutation_map(
    %query: tensor<1x2x3xf32>, %key: tensor<1x4x3xf32>,
    %value: tensor<1x4x5xf32>, %output: tensor<1x2x5xf32>,
    %output_grad: tensor<1x2x5xf32>, %logsumexp: tensor<1x2xf32>) {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3xf32>
  %key_grad = tensor.empty() : tensor<1x4x3xf32>
  %value_grad = tensor.empty() : tensor<1x4x5xf32>
  // expected-error @+1 {{all indexing maps must be projected permutations; map 0 is not}}
  %result:3 = iree_linalg_ext.attention_backward {
      indexing_maps = [
        #bad_q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>,
          tensor<1x2x5xf32>, tensor<1x2x5xf32>, tensor<1x2xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>)
      -> tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
  return
}

// -----

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#lse = affine_map<(b, m, k1, k2, n) -> (b, m)>
#scalar = affine_map<(b, m, k1, k2, n) -> ()>

func.func @non_f32_logsumexp(
    %query: tensor<1x2x3xf32>, %key: tensor<1x4x3xf32>,
    %value: tensor<1x4x5xf32>, %output: tensor<1x2x5xf32>,
    %output_grad: tensor<1x2x5xf32>, %logsumexp: tensor<1x2xf16>) {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3xf32>
  %key_grad = tensor.empty() : tensor<1x4x3xf32>
  %value_grad = tensor.empty() : tensor<1x4x5xf32>
  // expected-error @+1 {{logsumexp must have f32 element type}}
  %result:3 = iree_linalg_ext.attention_backward {
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>,
          tensor<1x2x5xf32>, tensor<1x2x5xf32>, tensor<1x2xf16>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>)
      -> tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
  return
}

// -----

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#lse = affine_map<(b, m, k1, k2, n) -> (b, m)>
#scalar = affine_map<(b, m, k1, k2, n) -> ()>
#bad_mask = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>

func.func @mask_uses_contraction_dimension(
    %query: tensor<1x2x3xf32>, %key: tensor<1x4x3xf32>,
    %value: tensor<1x4x5xf32>, %output: tensor<1x2x5xf32>,
    %output_grad: tensor<1x2x5xf32>, %logsumexp: tensor<1x2xf32>,
    %mask: tensor<1x2x3xi1>) {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3xf32>
  %key_grad = tensor.empty() : tensor<1x4x3xf32>
  %value_grad = tensor.empty() : tensor<1x4x5xf32>
  // expected-error @+1 {{mask map must be a projected permutation of score dimensions}}
  %result:3 = iree_linalg_ext.attention_backward {
      indexing_maps = [
        #q, #k, #v, #o, #o, #lse, #scalar, #bad_mask, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp,
          %scale, %mask :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>,
          tensor<1x2x5xf32>, tensor<1x2x5xf32>, tensor<1x2xf32>,
          f32, tensor<1x2x3xi1>)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>)
      -> tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
  return
}

// -----

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#bad_lse = affine_map<(b, m, k1, k2, n) -> (m, b)>
#scalar = affine_map<(b, m, k1, k2, n) -> ()>

func.func @noncanonical_logsumexp_map(
    %query: tensor<1x2x3xf32>, %key: tensor<1x4x3xf32>,
    %value: tensor<1x4x5xf32>, %output: tensor<1x2x5xf32>,
    %output_grad: tensor<1x2x5xf32>, %logsumexp: tensor<2x1xf32>) {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3xf32>
  %key_grad = tensor.empty() : tensor<1x4x3xf32>
  %value_grad = tensor.empty() : tensor<1x4x5xf32>
  // expected-error @+1 {{logsumexp map must contain the attention batch and query-row dimensions in canonical order}}
  %result:3 = iree_linalg_ext.attention_backward {
      indexing_maps = [
        #q, #k, #v, #o, #o, #bad_lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>,
          tensor<1x2x5xf32>, tensor<1x2x5xf32>, tensor<2x1xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>)
      -> tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
  return
}

// -----

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#lse = affine_map<(b, m, k1, k2, n) -> (b, m)>
#scalar = affine_map<(b, m, k1, k2, n) -> ()>

func.func @f64_query_is_unsupported(
    %query: tensor<1x2x3xf64>, %key: tensor<1x4x3xf32>,
    %value: tensor<1x4x5xf32>, %output: tensor<1x2x5xf32>,
    %output_grad: tensor<1x2x5xf32>, %logsumexp: tensor<1x2xf32>) {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3xf64>
  %key_grad = tensor.empty() : tensor<1x4x3xf32>
  %value_grad = tensor.empty() : tensor<1x4x5xf32>
  // expected-error @+1 {{query must have f16, bf16, or f32 element type}}
  %result:3 = iree_linalg_ext.attention_backward {
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3xf64>, tensor<1x4x3xf32>, tensor<1x4x5xf32>,
          tensor<1x2x5xf32>, tensor<1x2x5xf32>, tensor<1x2xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3xf64>, tensor<1x4x3xf32>, tensor<1x4x5xf32>)
      -> tensor<1x2x3xf64>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
  return
}

// -----

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#lse = affine_map<(b, m, k1, k2, n) -> (b, m)>
#scalar = affine_map<(b, m, k1, k2, n) -> ()>

func.func @f8_query_is_unsupported(
    %query: tensor<1x2x3xf8E4M3FNUZ>, %key: tensor<1x4x3xf32>,
    %value: tensor<1x4x5xf32>, %output: tensor<1x2x5xf32>,
    %output_grad: tensor<1x2x5xf32>, %logsumexp: tensor<1x2xf32>) {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3xf8E4M3FNUZ>
  %key_grad = tensor.empty() : tensor<1x4x3xf32>
  %value_grad = tensor.empty() : tensor<1x4x5xf32>
  // expected-error @+1 {{query must have f16, bf16, or f32 element type}}
  %result:3 = iree_linalg_ext.attention_backward {
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3xf8E4M3FNUZ>, tensor<1x4x3xf32>,
          tensor<1x4x5xf32>, tensor<1x2x5xf32>, tensor<1x2x5xf32>,
          tensor<1x2xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3xf8E4M3FNUZ>, tensor<1x4x3xf32>,
          tensor<1x4x5xf32>)
      -> tensor<1x2x3xf8E4M3FNUZ>, tensor<1x4x3xf32>,
          tensor<1x4x5xf32>
  return
}

// -----

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#lse = affine_map<(b, m, k1, k2, n) -> (b, m)>
#scalar = affine_map<(b, m, k1, k2, n) -> ()>

func.func @non_boolean_use_exp2(
    %query: tensor<1x2x3xf32>, %key: tensor<1x4x3xf32>,
    %value: tensor<1x4x5xf32>, %output: tensor<1x2x5xf32>,
    %output_grad: tensor<1x2x5xf32>, %logsumexp: tensor<1x2xf32>) {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3xf32>
  %key_grad = tensor.empty() : tensor<1x4x3xf32>
  %value_grad = tensor.empty() : tensor<1x4x5xf32>
  // expected-error @+1 {{expected decomposition_config entry 'use_exp2' to be a boolean}}
  %result:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {use_exp2 = 2 : i32},
      indexing_maps = [
        #q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>,
          tensor<1x2x5xf32>, tensor<1x2x5xf32>, tensor<1x2xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>)
      -> tensor<1x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x5xf32>
  return
}
