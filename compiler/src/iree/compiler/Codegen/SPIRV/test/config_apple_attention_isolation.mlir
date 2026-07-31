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

// CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce
// CHECK-LABEL: func.func @matmul_8x8x8(
// CHECK-SAME: translation_info = #[[TRANSLATION]]
// CHECK-NOT: APPLE_SIMDGROUP
