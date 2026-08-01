// RUN: iree-opt --split-input-file --iree-global-opt-split-attention-backward-for-metal %s | FileCheck %s

#metal_target = #hal.executable.target<"metal-spirv", "metallib">

#q = affine_map<(b, h, m, qk, k2, v) -> (b, h, m, qk)>
#k = affine_map<(b, h, m, qk, k2, v) -> (b, h, k2, qk)>
#v = affine_map<(b, h, m, qk, k2, v) -> (b, h, k2, v)>
#o = affine_map<(b, h, m, qk, k2, v) -> (b, h, m, v)>
#lse = affine_map<(b, h, m, qk, k2, v) -> (b, h, m)>
#scalar = affine_map<(b, h, m, qk, k2, v) -> ()>

util.global private @device = #hal.device.target<"metal", [#metal_target]>
    : !hal.device

// CHECK-LABEL: func.func @metal_only
// CHECK:         iree_linalg_ext.attention_backward {
// CHECK-NOT:     iree_codegen.apple_attention_backward_role
func.func @metal_only(
    %query: tensor<1x2x3x4xbf16>, %key: tensor<1x2x5x4xbf16>,
    %value: tensor<1x2x5x6xbf16>, %output: tensor<1x2x3x6xbf16>,
    %output_grad: tensor<1x2x3x6xbf16>, %logsumexp: tensor<1x2x3xf32>)
    -> (tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
        tensor<1x2x5x6xbf16>)
    attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3x4xbf16>
  %key_grad = tensor.empty() : tensor<1x2x5x4xbf16>
  %value_grad = tensor.empty() : tensor<1x2x5x6xbf16>
  %result:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {qk_attrs = {existing = 7 : i64},
                              use_exp2 = false},
      indexing_maps = [
        #q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>, tensor<1x2x3x6xbf16>,
          tensor<1x2x3x6xbf16>, tensor<1x2x3xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>)
      -> tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
         tensor<1x2x5x6xbf16>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
      tensor<1x2x5x6xbf16>
}

// CHECK-LABEL: func.func @metal_eligible
// CHECK:         iree_linalg_ext.attention_backward {
// CHECK-SAME:      decomposition_config = {
// CHECK-SAME:      dk_attrs = {iree_codegen.apple_attention_backward_causal, iree_codegen.apple_attention_backward_role = "dk_attrs"}
// CHECK-SAME:      dp_attrs = {iree_codegen.apple_attention_backward_role = "dp_attrs"}
// CHECK-SAME:      dq_attrs = {iree_codegen.apple_attention_backward_causal, iree_codegen.apple_attention_backward_role = "dq_attrs"}
// CHECK-SAME:      dv_attrs = {iree_codegen.apple_attention_backward_causal, iree_codegen.apple_attention_backward_role = "dv_attrs"}
// CHECK-SAME:      qk_attrs = {existing = 7 : i64, iree_codegen.apple_attention_backward_role = "qk_attrs"}
// CHECK-SAME:      use_exp2 = false
func.func @metal_eligible(
    %query: tensor<1x2x8x8xbf16>, %key: tensor<1x2x8x8xbf16>,
    %value: tensor<1x2x8x8xbf16>, %output: tensor<1x2x8x8xbf16>,
    %output_grad: tensor<1x2x8x8xbf16>, %logsumexp: tensor<1x2x8xf32>)
    -> (tensor<1x2x8x8xbf16>, tensor<1x2x8x8xbf16>,
        tensor<1x2x8x8xbf16>)
    attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x8x8xbf16>
  %key_grad = tensor.empty() : tensor<1x2x8x8xbf16>
  %value_grad = tensor.empty() : tensor<1x2x8x8xbf16>
  %result:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {
        dk_attrs = {iree_codegen.apple_attention_backward_causal},
        dq_attrs = {iree_codegen.apple_attention_backward_causal},
        dv_attrs = {iree_codegen.apple_attention_backward_causal},
        qk_attrs = {existing = 7 : i64},
        use_exp2 = false},
      indexing_maps = [
        #q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x8x8xbf16>, tensor<1x2x8x8xbf16>,
          tensor<1x2x8x8xbf16>, tensor<1x2x8x8xbf16>,
          tensor<1x2x8x8xbf16>, tensor<1x2x8xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x8x8xbf16>, tensor<1x2x8x8xbf16>,
          tensor<1x2x8x8xbf16>)
      -> tensor<1x2x8x8xbf16>, tensor<1x2x8x8xbf16>,
         tensor<1x2x8x8xbf16>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x8x8xbf16>, tensor<1x2x8x8xbf16>,
      tensor<1x2x8x8xbf16>
}

// CHECK-LABEL: func.func @metal_unsupported_type
// CHECK:         iree_linalg_ext.attention_backward {
// CHECK-NOT:     iree_codegen.apple_attention_backward_role
func.func @metal_unsupported_type(
    %query: tensor<1x2x8x8xf16>, %key: tensor<1x2x8x8xf16>,
    %value: tensor<1x2x8x8xbf16>, %output: tensor<1x2x8x8xbf16>,
    %output_grad: tensor<1x2x8x8xbf16>, %logsumexp: tensor<1x2x8xf32>)
    -> (tensor<1x2x8x8xf16>, tensor<1x2x8x8xf16>,
        tensor<1x2x8x8xbf16>)
    attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x8x8xf16>
  %key_grad = tensor.empty() : tensor<1x2x8x8xf16>
  %value_grad = tensor.empty() : tensor<1x2x8x8xbf16>
  %result:3 = iree_linalg_ext.attention_backward {
      indexing_maps = [
        #q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x8x8xf16>, tensor<1x2x8x8xf16>,
          tensor<1x2x8x8xbf16>, tensor<1x2x8x8xbf16>,
          tensor<1x2x8x8xbf16>, tensor<1x2x8xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x8x8xf16>, tensor<1x2x8x8xf16>,
          tensor<1x2x8x8xbf16>)
      -> tensor<1x2x8x8xf16>, tensor<1x2x8x8xf16>,
         tensor<1x2x8x8xbf16>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x8x8xf16>, tensor<1x2x8x8xf16>,
      tensor<1x2x8x8xbf16>
}

// -----

#cpu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">

#q = affine_map<(b, h, m, qk, k2, v) -> (b, h, m, qk)>
#k = affine_map<(b, h, m, qk, k2, v) -> (b, h, k2, qk)>
#v = affine_map<(b, h, m, qk, k2, v) -> (b, h, k2, v)>
#o = affine_map<(b, h, m, qk, k2, v) -> (b, h, m, v)>
#lse = affine_map<(b, h, m, qk, k2, v) -> (b, h, m)>
#scalar = affine_map<(b, h, m, qk, k2, v) -> ()>

util.global private @device = #hal.device.target<"local", [#cpu_target]>
    : !hal.device

// CHECK-LABEL: func.func @cpu_only
// CHECK:         iree_linalg_ext.attention_backward {
// CHECK-NOT:     iree_codegen.apple_attention_backward_role
func.func @cpu_only(
    %query: tensor<1x2x3x4xbf16>, %key: tensor<1x2x5x4xbf16>,
    %value: tensor<1x2x5x6xbf16>, %output: tensor<1x2x3x6xbf16>,
    %output_grad: tensor<1x2x3x6xbf16>, %logsumexp: tensor<1x2x3xf32>)
    -> (tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
        tensor<1x2x5x6xbf16>)
    attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3x4xbf16>
  %key_grad = tensor.empty() : tensor<1x2x5x4xbf16>
  %value_grad = tensor.empty() : tensor<1x2x5x6xbf16>
  %result:3 = iree_linalg_ext.attention_backward {
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>, tensor<1x2x3x6xbf16>,
          tensor<1x2x3x6xbf16>, tensor<1x2x3xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>)
      -> tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
         tensor<1x2x5x6xbf16>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
      tensor<1x2x5x6xbf16>
}

// -----

#metal_target = #hal.executable.target<"metal-spirv", "metallib">
#cpu_target = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">

#q = affine_map<(b, h, m, qk, k2, v) -> (b, h, m, qk)>
#k = affine_map<(b, h, m, qk, k2, v) -> (b, h, k2, qk)>
#v = affine_map<(b, h, m, qk, k2, v) -> (b, h, k2, v)>
#o = affine_map<(b, h, m, qk, k2, v) -> (b, h, m, v)>
#lse = affine_map<(b, h, m, qk, k2, v) -> (b, h, m)>
#scalar = affine_map<(b, h, m, qk, k2, v) -> ()>

util.global private @device = #hal.device.target<
    "hybrid", [#metal_target, #cpu_target]> : !hal.device

// CHECK-LABEL: func.func @mixed_targets
// CHECK:         iree_linalg_ext.attention_backward {
// CHECK-NOT:     iree_codegen.apple_attention_backward_role
func.func @mixed_targets(
    %query: tensor<1x2x3x4xbf16>, %key: tensor<1x2x5x4xbf16>,
    %value: tensor<1x2x5x6xbf16>, %output: tensor<1x2x3x6xbf16>,
    %output_grad: tensor<1x2x3x6xbf16>, %logsumexp: tensor<1x2x3xf32>)
    -> (tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
        tensor<1x2x5x6xbf16>)
    attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3x4xbf16>
  %key_grad = tensor.empty() : tensor<1x2x5x4xbf16>
  %value_grad = tensor.empty() : tensor<1x2x5x6xbf16>
  %result:3 = iree_linalg_ext.attention_backward {
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>, tensor<1x2x3x6xbf16>,
          tensor<1x2x3x6xbf16>, tensor<1x2x3xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>)
      -> tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
         tensor<1x2x5x6xbf16>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
      tensor<1x2x5x6xbf16>
}

// -----

#q = affine_map<(b, h, m, qk, k2, v) -> (b, h, m, qk)>
#k = affine_map<(b, h, m, qk, k2, v) -> (b, h, k2, qk)>
#v = affine_map<(b, h, m, qk, k2, v) -> (b, h, k2, v)>
#o = affine_map<(b, h, m, qk, k2, v) -> (b, h, m, v)>
#lse = affine_map<(b, h, m, qk, k2, v) -> (b, h, m)>
#scalar = affine_map<(b, h, m, qk, k2, v) -> ()>

// CHECK-LABEL: func.func @unknown_target
// CHECK:         iree_linalg_ext.attention_backward {
// CHECK-NOT:     iree_codegen.apple_attention_backward_role
func.func @unknown_target(
    %query: tensor<1x2x3x4xbf16>, %key: tensor<1x2x5x4xbf16>,
    %value: tensor<1x2x5x6xbf16>, %output: tensor<1x2x3x6xbf16>,
    %output_grad: tensor<1x2x3x6xbf16>, %logsumexp: tensor<1x2x3xf32>)
    -> (tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
        tensor<1x2x5x6xbf16>) {
  %scale = arith.constant 0.5 : f32
  %query_grad = tensor.empty() : tensor<1x2x3x4xbf16>
  %key_grad = tensor.empty() : tensor<1x2x5x4xbf16>
  %value_grad = tensor.empty() : tensor<1x2x5x6xbf16>
  %result:3 = iree_linalg_ext.attention_backward {
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%query, %key, %value, %output, %output_grad, %logsumexp, %scale :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>, tensor<1x2x3x6xbf16>,
          tensor<1x2x3x6xbf16>, tensor<1x2x3xf32>, f32)
      outs(%query_grad, %key_grad, %value_grad :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>)
      -> tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
         tensor<1x2x5x6xbf16>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
      tensor<1x2x5x6xbf16>
}
