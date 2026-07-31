// RUN: env -u IREE_METAL_COOP_ATTENTION_WIP IREE_METAL_DISABLE_NATIVE_ATTENTION=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION -u IREE_METAL_COOP_ATTENTION_WIP \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>

func.func @matmul_8x8x8(
    %lhs: tensor<8x8xf16>,
    %rhs: tensor<8x8xf16>,
    %acc: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %result = linalg.generic {
      indexing_maps = [#lhs, #rhs, #acc],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<8x8xf16>, tensor<8x8xf16>)
      outs(%acc : tensor<8x8xf32>) {
    ^bb0(%l: f16, %r: f16, %out: f32):
      %lext = arith.extf %l : f16 to f32
      %rext = arith.extf %r : f16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize
// CHECK-LABEL: func.func @matmul_8x8x8(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK-NOT: APPLE_SIMDGROUP

// -----

// A reduction smaller than Apple's 32-lane subgroup must stay off the masked
// subgroup-reduction path. Forcing a reduction tile of 32 for this broadcast
// VJP drops all but one batch contribution on Metal.
#tiny_input = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#tiny_output = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @tiny_broadcast_vjp_reduction(
    %input: tensor<512x768x2xbf16>,
    %init: tensor<512x768xbf16>) -> tensor<512x768xbf16> {
  %result = linalg.generic {
      indexing_maps = [#tiny_input, #tiny_output],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%input : tensor<512x768x2xbf16>)
      outs(%init : tensor<512x768xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %sum = arith.addf %out, %in : bf16
      linalg.yield %sum : bf16
  } -> tensor<512x768xbf16>
  return %result : tensor<512x768xbf16>
}

// CHECK-LABEL: func.func @tiny_broadcast_vjp_reduction(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
