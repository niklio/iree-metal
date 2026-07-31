// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.attention_backward}),canonicalize,cse)" %s | FileCheck %s

#q = affine_map<(b, h, m, qk, k2, v) -> (b, h, m, qk)>
#k = affine_map<(b, h, m, qk, k2, v) -> (b, h, k2, qk)>
#v = affine_map<(b, h, m, qk, k2, v) -> (b, h, k2, v)>
#o = affine_map<(b, h, m, qk, k2, v) -> (b, h, m, v)>
#lse = affine_map<(b, h, m, qk, k2, v) -> (b, h, m)>
#scalar = affine_map<(b, h, m, qk, k2, v) -> ()>

func.func @materialized_backward(
    %q: tensor<1x2x3x4xbf16>, %k: tensor<1x2x5x4xbf16>,
    %v: tensor<1x2x5x6xbf16>, %o: tensor<1x2x3x6xbf16>,
    %do: tensor<1x2x3x6xbf16>, %lse: tensor<1x2x3xf32>)
    -> (tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
        tensor<1x2x5x6xbf16>) {
  %scale = arith.constant 0.5 : bf16
  %dq = tensor.empty() : tensor<1x2x3x4xbf16>
  %dk = tensor.empty() : tensor<1x2x5x4xbf16>
  %dv = tensor.empty() : tensor<1x2x5x6xbf16>
  %result:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {
        qk_attrs = {iree_codegen.apple_attention_backward_role = "qk_attrs"},
        dp_attrs = {iree_codegen.apple_attention_backward_role = "dp_attrs"},
        dq_attrs = {iree_codegen.apple_attention_backward_role = "dq_attrs"},
        dk_attrs = {iree_codegen.apple_attention_backward_role = "dk_attrs"},
        dv_attrs = {iree_codegen.apple_attention_backward_role = "dv_attrs"},
        use_exp2 = false},
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%q, %k, %v, %o, %do, %lse, %scale :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>, tensor<1x2x3x6xbf16>,
          tensor<1x2x3x6xbf16>, tensor<1x2x3xf32>, bf16)
      outs(%dq, %dk, %dv :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>)
      -> tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
         tensor<1x2x5x6xbf16>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
      tensor<1x2x5x6xbf16>
}

// CHECK-DAG:   #[[DK_LHS:map[0-9]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>
// CHECK-DAG:   #[[DK_RHS:map[0-9]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
// CHECK-DAG:   #[[DK_OUT:map[0-9]+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @materialized_backward
// CHECK:       iree_codegen.apple_attention_backward_role = "qk_attrs"
// CHECK:       %[[P16:[0-9]+]] = linalg.generic{{.*}} ins({{.*}} : tensor<1x2x3xf32>, tensor<1x2x3x5xf32>) outs({{.*}} : tensor<1x2x3x5xbf16>)
// CHECK:       math.exp
// CHECK:       arith.truncf {{.*}} : f32 to bf16
// CHECK:       %[[P_BARRIER:[0-9]+]] = util.optimization_barrier %[[P16]]
// CHECK:       util.optimization_barrier
// CHECK:       iree_codegen.apple_attention_backward_role = "dp_attrs"
// CHECK:       %[[DS:[0-9]+]] = linalg.generic{{.*}} ins({{.*}} : tensor<1x2x3x5xf32>) outs({{.*}} : tensor<1x2x3x5xbf16>)
// CHECK:       arith.truncf {{.*}} : f32 to bf16
// CHECK:       %[[DQ_BARRIER:[0-9]+]] = util.optimization_barrier %[[DS]]
// CHECK:       iree_codegen.apple_attention_backward_role = "dq_attrs"
// CHECK:       %[[DK_BARRIER:[0-9]+]] = util.optimization_barrier %[[DS]]
// CHECK:       linalg.generic {indexing_maps = [#[[DK_LHS]], #[[DK_RHS]], #[[DK_OUT]]]
// CHECK-SAME:  iree_codegen.apple_attention_backward_role = "dk_attrs"
// CHECK-NOT:   linalg.generic{{.*}} ins({{.*}} : tensor<1x2x3x5xf32>) outs({{.*}} : tensor<1x2x3x5xbf16>)
// CHECK:       util.optimization_barrier %[[P_BARRIER]]
// CHECK:       iree_codegen.apple_attention_backward_role = "dv_attrs"

// CHECK-LABEL: func.func @portable_contraction_attrs
// CHECK:       iree_codegen.test_marker = "preserve"
// CHECK-NOT:   iree_codegen.apple_attention_backward_role
// CHECK:       %[[PORTABLE_P:[0-9]+]] = linalg.generic{{.*}} outs({{.*}} : tensor<1x2x3x5xf32>)
// CHECK:       math.exp
// CHECK-NOT:   arith.truncf
// CHECK:       linalg.yield {{.*}} : f32
func.func @portable_contraction_attrs(
    %q: tensor<1x2x3x4xbf16>, %k: tensor<1x2x5x4xbf16>,
    %v: tensor<1x2x5x6xbf16>, %o: tensor<1x2x3x6xbf16>,
    %do: tensor<1x2x3x6xbf16>, %lse: tensor<1x2x3xf32>)
    -> (tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
        tensor<1x2x5x6xbf16>) {
  %scale = arith.constant 0.5 : bf16
  %dq = tensor.empty() : tensor<1x2x3x4xbf16>
  %dk = tensor.empty() : tensor<1x2x5x4xbf16>
  %dv = tensor.empty() : tensor<1x2x5x6xbf16>
  %result:3 = iree_linalg_ext.attention_backward {
      decomposition_config = {
        qk_attrs = {iree_codegen.test_marker = "preserve"},
        use_exp2 = false},
      indexing_maps = [#q, #k, #v, #o, #o, #lse, #scalar, #q, #k, #v]}
      ins(%q, %k, %v, %o, %do, %lse, %scale :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>, tensor<1x2x3x6xbf16>,
          tensor<1x2x3x6xbf16>, tensor<1x2x3xf32>, bf16)
      outs(%dq, %dk, %dv :
          tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
          tensor<1x2x5x6xbf16>)
      -> tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
         tensor<1x2x5x6xbf16>
  return %result#0, %result#1, %result#2 :
      tensor<1x2x3x4xbf16>, tensor<1x2x5x4xbf16>,
      tensor<1x2x5x6xbf16>
}
