// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-llvmgpu-configure-tensor-layouts, canonicalize, cse))' %s | FileCheck %s

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x16_mfma(%lhs: tensor<96x16xf16>,
                           %rhs: tensor<64x16xf16>,
                           %init: tensor<96x64xf32>)
                           -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [3, 2], outer_tile = [1, 1], thread_tile = [32, 2], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [1, 32]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [2, 2], outer_tile = [1, 1], thread_tile = [32, 2], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [1, 32]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [3, 2], outer_tile = [4, 1], thread_tile = [2, 32], element_tile = [4, 1], subgroup_strides = [0, 0], thread_strides = [32, 1]>

// CHECK-LABEL: func.func @matmul_96x64x16_mfma

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

// The physical Apple fragment factors are explicit iteration dimensions.
// Verify the exact native lane mapping for an 8x8 fragment, including a RHS
// whose tensor dimensions are in transposed (N, K) order.

#translation = #iree_codegen.translation_info<
    pipeline = SPIRVAppleVectorDistributeAttention
    workgroup_size = [32, 1, 1]
    subgroup_size = 32>

#apple8_physical_maps = [
  affine_map<(m0, m1, m2, n0, n1, n2, k0, k1, k2) ->
      (m0, m1, m2, k0, k1, k2)>,
  affine_map<(m0, m1, m2, n0, n1, n2, k0, k1, k2) ->
      (n0, n1, n2, k0, k1, k2)>,
  affine_map<(m0, m1, m2, n0, n1, n2, k0, k1, k2) ->
      (m0, m1, m2, n0, n1, n2)>
]

#apple8_physical_traits = {
  indexing_maps = #apple8_physical_maps,
  iterator_types = [
    "parallel", "parallel", "parallel",
    "parallel", "parallel", "parallel",
    "reduction", "reduction", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{
    mma_kind = #iree_gpu.mma_layout<
        APPLE_SIMDGROUP_F32_8x8x8_F16:
        apple_physical_fragment_layout = true>,
    subgroup_basis = [
      [1, 1, 1, 1, 1, 1, 1, 1, 1],
      [0, 3, 4, 1, 5, 6, 2, 7, 8]]
  }>
}

func.func @apple8_physical_transposed_rhs(
    %lhs: tensor<1x2x4x1x2x4xf16>,
    %rhs: tensor<1x2x4x1x2x4xf16>,
    %init: tensor<1x2x4x1x2x4xf32>)
    -> tensor<1x2x4x1x2x4xf32>
    attributes {translation_info = #translation} {
  %out = linalg.generic #apple8_physical_traits
      ins(%lhs, %rhs : tensor<1x2x4x1x2x4xf16>,
                        tensor<1x2x4x1x2x4xf16>)
      outs(%init : tensor<1x2x4x1x2x4xf32>) {
    ^bb0(%l: f16, %r: f16, %acc: f32):
      %lext = arith.extf %l : f16 to f32
      %rext = arith.extf %r : f16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %sum = arith.addf %acc, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<1x2x4x1x2x4xf32>
  return %out : tensor<1x2x4x1x2x4xf32>
}

// CHECK-DAG: #[[$APPLE8_ROW_COL:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1, 1, 1, 1], batch_tile = [1, 1, 1, 1, 1, 1], outer_tile = [1, 1, 1, 1, 1, 1], thread_tile = [1, 2, 4, 1, 2, 2], element_tile = [1, 1, 1, 1, 1, 2], subgroup_strides = [0, 0, 0, 0, 0, 0], thread_strides = [0, 16, 2, 0, 8, 1]>
// CHECK-DAG: #[[$APPLE8_COL_ROW:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1, 1, 1, 1], batch_tile = [1, 1, 1, 1, 1, 1], outer_tile = [1, 1, 1, 1, 1, 1], thread_tile = [1, 2, 2, 1, 2, 4], element_tile = [1, 1, 2, 1, 1, 1], subgroup_strides = [0, 0, 0, 0, 0, 0], thread_strides = [0, 8, 1, 0, 16, 2]>

// CHECK-LABEL: func.func @apple8_physical_transposed_rhs
// CHECK-DAG: %[[APPLE8_LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$APPLE8_ROW_COL]])
// CHECK-DAG: %[[APPLE8_RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$APPLE8_COL_ROW]])
// CHECK-DAG: %[[APPLE8_ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$APPLE8_ROW_COL]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[APPLE8_LHS]], %[[APPLE8_RHS]]
// CHECK-SAME: outs(%[[APPLE8_ACC]]

// -----

// Apple16 adds a native 2x2 outer-tile factor while retaining the same 8x8
// physical lane mapping inside each tile.

#translation = #iree_codegen.translation_info<
    pipeline = SPIRVAppleVectorDistributeAttention
    workgroup_size = [32, 1, 1]
    subgroup_size = 32>

#apple16_physical_maps = [
  affine_map<(m0, m1, m2, m3, n0, n1, n2, n3,
              k0, k1, k2, k3) ->
      (m0, m1, m2, m3, k0, k1, k2, k3)>,
  affine_map<(m0, m1, m2, m3, n0, n1, n2, n3,
              k0, k1, k2, k3) ->
      (n0, n1, n2, n3, k0, k1, k2, k3)>,
  affine_map<(m0, m1, m2, m3, n0, n1, n2, n3,
              k0, k1, k2, k3) ->
      (m0, m1, m2, m3, n0, n1, n2, n3)>
]

#apple16_physical_traits = {
  indexing_maps = #apple16_physical_maps,
  iterator_types = [
    "parallel", "parallel", "parallel", "parallel",
    "parallel", "parallel", "parallel", "parallel",
    "reduction", "reduction", "reduction", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{
    mma_kind = #iree_gpu.mma_layout<
        APPLE_SIMDGROUP_F32_16x16x16_F16:
        apple_physical_fragment_layout = true>,
    subgroup_basis = [
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11]]
  }>
}

func.func @apple16_physical_transposed_rhs(
    %lhs: tensor<1x2x2x4x1x2x2x4xf16>,
    %rhs: tensor<1x2x2x4x1x2x2x4xf16>,
    %init: tensor<1x2x2x4x1x2x2x4xf32>)
    -> tensor<1x2x2x4x1x2x2x4xf32>
    attributes {translation_info = #translation} {
  %out = linalg.generic #apple16_physical_traits
      ins(%lhs, %rhs : tensor<1x2x2x4x1x2x2x4xf16>,
                        tensor<1x2x2x4x1x2x2x4xf16>)
      outs(%init : tensor<1x2x2x4x1x2x2x4xf32>) {
    ^bb0(%l: f16, %r: f16, %acc: f32):
      %lext = arith.extf %l : f16 to f32
      %rext = arith.extf %r : f16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %sum = arith.addf %acc, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<1x2x2x4x1x2x2x4xf32>
  return %out : tensor<1x2x2x4x1x2x2x4xf32>
}

// CHECK-DAG: #[[$APPLE16_ROW_COL:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1, 1, 1, 1, 1, 1], batch_tile = [1, 1, 1, 1, 1, 1, 1, 1], outer_tile = [1, 2, 1, 1, 1, 2, 1, 1], thread_tile = [1, 1, 2, 4, 1, 1, 2, 2], element_tile = [1, 1, 1, 1, 1, 1, 1, 2], subgroup_strides = [0, 0, 0, 0, 0, 0, 0, 0], thread_strides = [0, 0, 16, 2, 0, 0, 8, 1]>
// CHECK-DAG: #[[$APPLE16_COL_ROW:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1, 1, 1, 1, 1, 1], batch_tile = [1, 1, 1, 1, 1, 1, 1, 1], outer_tile = [1, 2, 1, 1, 1, 2, 1, 1], thread_tile = [1, 1, 2, 2, 1, 1, 2, 4], element_tile = [1, 1, 1, 2, 1, 1, 1, 1], subgroup_strides = [0, 0, 0, 0, 0, 0, 0, 0], thread_strides = [0, 0, 8, 1, 0, 0, 16, 2]>

// CHECK-LABEL: func.func @apple16_physical_transposed_rhs
// CHECK-DAG: %[[APPLE16_LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$APPLE16_ROW_COL]])
// CHECK-DAG: %[[APPLE16_RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$APPLE16_COL_ROW]])
// CHECK-DAG: %[[APPLE16_ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$APPLE16_ROW_COL]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[APPLE16_LHS]], %[[APPLE16_RHS]]
// CHECK-SAME: outs(%[[APPLE16_ACC]]

// -----

// Preserve the contraction result layout across a pointwise DPS update. If the
// update folds to a constant (as an all-false attention mask does), the result
// anchor must remain available to distribute the following reduction.

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#contraction_maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#contraction_traits = {
  indexing_maps = #contraction_maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{
    mma_kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
    subgroup_basis = [[1, 1, 1], [0, 1, 2]]
  }>
}

#scalar = affine_map<(m, n) -> ()>
#identity = affine_map<(m, n) -> (m, n)>
#row = affine_map<(m, n) -> (m)>

func.func @pointwise_dps_preserves_result_anchor(
    %lhs: tensor<16x16xf16>,
    %rhs: tensor<16x16xf16>,
    %init: tensor<16x16xf32>,
    %row_init: tensor<16xf32>) -> tensor<16xf32>
    attributes {translation_info = #translation} {
  %scores = linalg.generic #contraction_traits
      ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>)
      outs(%init : tensor<16x16xf32>) {
    ^bb0(%l: f16, %r: f16, %acc: f32):
      %lext = arith.extf %l : f16 to f32
      %rext = arith.extf %r : f16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %acc, %mul : f32
      linalg.yield %add : f32
  } -> tensor<16x16xf32>
  %false = arith.constant false
  %masked = linalg.generic {
      indexing_maps = [#scalar, #identity],
      iterator_types = ["parallel", "parallel"]}
      ins(%false : i1) outs(%scores : tensor<16x16xf32>) {
    ^bb0(%condition: i1, %score: f32):
      %masked_out = arith.constant -3.40282347E+38 : f32
      %selected = arith.select %condition, %score, %masked_out : f32
      linalg.yield %selected : f32
  } -> tensor<16x16xf32>
  %max = linalg.generic {
      indexing_maps = [#identity, #row],
      iterator_types = ["parallel", "reduction"]}
      ins(%masked : tensor<16x16xf32>)
      outs(%row_init : tensor<16xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %next = arith.maximumf %value, %acc : f32
      linalg.yield %next : f32
  } -> tensor<16xf32>
  return %max : tensor<16xf32>
}

// CHECK-LABEL: func.func @pointwise_dps_preserves_result_anchor(
// CHECK: %[[SCORES:.+]] = linalg.generic
// CHECK: %[[SCORES_LAYOUT:.+]] = iree_vector_ext.to_layout %[[SCORES]] to layout(
// CHECK: %[[MASKED:.+]] = linalg.generic
// CHECK-SAME: outs(%[[SCORES_LAYOUT]] : tensor<16x16xf32>)
// CHECK: %[[MASKED_LAYOUT:.+]] = iree_vector_ext.to_layout %[[MASKED]] to layout(
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[MASKED_LAYOUT]] : tensor<16x16xf32>)

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x16_wmmar3(%lhs: tensor<96x16xf16>,
                           %rhs: tensor<64x16xf16>,
                           %init: tensor<96x64xf32>)
                           -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 1], outer_tile = [1, 1], thread_tile = [16, 1], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 0]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [4, 1], outer_tile = [1, 1], thread_tile = [16, 1], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 0]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 4], outer_tile = [8, 1], thread_tile = [2, 16], element_tile = [1, 1], subgroup_strides = [0, 0], thread_strides = [16, 1]>

// CHECK-LABEL: func.func @matmul_96x64x16_wmmar3

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x16_wmmar4(%lhs: tensor<96x16xf16>,
                           %rhs: tensor<64x16xf16>,
                           %init: tensor<96x64xf32>)
                           -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 8], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [4, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 8], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 4], outer_tile = [1, 1], thread_tile = [2, 16], element_tile = [8, 1], subgroup_strides = [0, 0], thread_strides = [16, 1]>

// CHECK-LABEL: func.func @matmul_96x64x16_wmmar4

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [32, 1, 1]
                                              subgroup_size = 32>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x32_wmma_gfx1250(%lhs: tensor<96x32xf16>,
                                        %rhs: tensor<64x32xf16>,
                                        %init: tensor<96x64xf32>) -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x32xf16>, tensor<64x32xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [4, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 4], outer_tile = [1, 1], thread_tile = [2, 16], element_tile = [8, 1], subgroup_strides = [0, 0], thread_strides = [16, 1]>

// CHECK-LABEL: func.func @matmul_96x64x32_wmma_gfx1250

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                              subgroup_basis = [[4, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_128x64x16_multi_subgroup(%lhs: tensor<128x16xf16>,
                                          %rhs: tensor<64x16xf16>,
                                          %init: tensor<128x64xf32>)
                                          -> tensor<128x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<128x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<128x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<128x64xf32>
  return %out : tensor<128x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [4, 1]
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1]
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [4, 1]

// CHECK-LABEL: func.func @matmul_128x64x16_multi_subgroup

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

func.func @linalg_copy(%in : tensor<16x16x16xf16>) -> tensor<16x16x16xf16>
                      attributes { translation_info = #translation } {
  %empty = tensor.empty() : tensor<16x16x16xf16>
  %copied = linalg.copy
            { lowering_config = #iree_gpu.derived_thread_config }
            ins(%in : tensor<16x16x16xf16>)
            outs(%empty : tensor<16x16x16xf16>) -> tensor<16x16x16xf16>
  func.return %copied : tensor<16x16x16xf16>
}

// CHECK-DAG: #[[$LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1], batch_tile = [8, 1, 1], outer_tile = [1, 1, 1], thread_tile = [2, 16, 2], element_tile = [1, 1, 8], subgroup_strides = [0, 0, 0], thread_strides = [32, 2, 1]>

// CHECK-LABEL: func.func @linalg_copy
// CHECK: %[[OUT:.+]] = linalg.copy
// CHECK: to_layout %[[OUT]] to layout(#[[$LAYOUT]])

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>

#gather_trait = {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"],
    lowering_config = #iree_gpu.derived_thread_config
}

func.func @gather_like(%base : tensor<16384x16x32x128xf16>,
                       %indices : tensor<4x64x4xi64>)
                       -> tensor<4x64x4x16x32x128xf16>
                       attributes { translation_info = #translation } {

  %empty = tensor.empty() : tensor<4x64x4x16x32x128xf16>
  %gather = linalg.generic #gather_trait
            ins(%indices : tensor<4x64x4xi64>)
            outs(%empty : tensor<4x64x4x16x32x128xf16>) {
  ^bb0(%in: i64, %out: f16):
    %idx = arith.index_cast %in : i64 to index
    %iv3 = linalg.index 3 : index
    %iv4 = linalg.index 4 : index
    %iv5 = linalg.index 5 : index
    %extracted = tensor.extract %base[%idx, %iv3, %iv4, %iv5] : tensor<16384x16x32x128xf16>
    linalg.yield %extracted : f16
  } -> tensor<4x64x4x16x32x128xf16>

  func.return %gather : tensor<4x64x4x16x32x128xf16>
}

// CHECK-DAG: #[[$LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1, 1, 1, 1], batch_tile = [4, 64, 4, 16, 8, 1], outer_tile = [1, 1, 1, 1, 1, 1], thread_tile = [1, 1, 1, 1, 4, 16], element_tile = [1, 1, 1, 1, 1, 8], subgroup_strides = [0, 0, 0, 0, 0, 0], thread_strides = [0, 0, 0, 0, 16, 1]>

// CHECK-LABEL: func.func @gather_like
// CHECK: %[[OUT:.+]] = linalg.generic
// CHECK: to_layout %[[OUT]] to layout(#[[$LAYOUT]])

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

func.func @dynamic_infer_sizes(%in : tensor<4x32x?x128xf16>) -> tensor<1x1x?x128xf16> attributes { translation_info = #translation } {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %d2 = tensor.dim %in, %c2 : tensor<4x32x?x128xf16>
  %45 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 1024)>(%c0)[%d2]
  %extracted_slice_5 = tensor.extract_slice %in[%c0, %c0, %c0, 0] [1, 1, %45, 128] [1, 1, 1, 1] : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
  %49 = tensor.empty(%45) : tensor<1x1x?x128xf16>
  %50 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_5 : tensor<1x1x?x128xf16>) outs(%49 : tensor<1x1x?x128xf16>) -> tensor<1x1x?x128xf16>
  return %50 : tensor<1x1x?x128xf16>
}

// CHECK-DAG: #[[LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1, 1], batch_tile = [1, 1, 256, 1], outer_tile = [1, 1, 1, 1], thread_tile = [1, 1, 4, 16], element_tile = [1, 1, 1, 8], subgroup_strides = [0, 0, 0, 0], thread_strides = [0, 0, 16, 1]>

// CHECK: %[[EXTRACT:.+]] = tensor.extract_slice %arg0{{.*}} : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty({{.*}}) : tensor<1x1x?x128xf16>
// CHECK: %[[COPY:.+]] = linalg.copy {{.*}} ins(%[[EXTRACT]] : tensor<1x1x?x128xf16>) outs(%[[EMPTY]] : tensor<1x1x?x128xf16>)
// CHECK: iree_vector_ext.to_layout %[[COPY]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#lowering_config = #iree_gpu.lowering_config<{
    subgroup_basis = [[1, 1, 2, 2], [0, 1, 2, 3]],
    lane_basis = [[1, 1, 8, 8], [0, 1, 2, 3]],
    thread = [0, 0, 8, 8]
}>

func.func @dynamic_infer_sizes_lowering_config(%in : tensor<4x32x?x128xf16>) -> tensor<1x1x?x128xf16> attributes { translation_info = #translation } {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %d2 = tensor.dim %in, %c2 : tensor<4x32x?x128xf16>
  %45 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 128)>(%c0)[%d2]
  %extracted_slice_5 = tensor.extract_slice %in[%c0, %c0, %c0, 0] [1, 1, %45, 128] [1, 1, 1, 1] : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
  %49 = tensor.empty(%45) : tensor<1x1x?x128xf16>
  %50 = linalg.copy {lowering_config = #lowering_config} ins(%extracted_slice_5 : tensor<1x1x?x128xf16>) outs(%49 : tensor<1x1x?x128xf16>) -> tensor<1x1x?x128xf16>
  return %50 : tensor<1x1x?x128xf16>
}

// CHECK-DAG: #[[LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 2, 2], batch_tile = [1, 1, 1, 1], outer_tile = [1, 1, 1, 1], thread_tile = [1, 1, 8, 8], element_tile = [1, 1, 8, 8], subgroup_strides = [0, 0, 2, 1], thread_strides = [0, 0, 8, 1]>

// CHECK: %[[EXTRACT:.+]] = tensor.extract_slice %arg0{{.*}} : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty({{.*}}) : tensor<1x1x?x128xf16>
// CHECK: %[[EXTRACTL:.+]] = iree_vector_ext.to_layout %[[EXTRACT]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>
// CHECK: %[[EMPTYL:.+]] = iree_vector_ext.to_layout %[[EMPTY]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>
// CHECK: %[[COPY:.+]] = linalg.copy {{.*}} ins(%[[EXTRACTL]] : tensor<1x1x?x128xf16>) outs(%[[EMPTYL]] : tensor<1x1x?x128xf16>)
// CHECK: iree_vector_ext.to_layout %[[COPY]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>

// -----

// Verify that the batch tile for a dimension that requires ceil division
// (63 / 8 = 8, not 7) is computed correctly.

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [512, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
  affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction", "parallel"],
  lowering_config = #iree_gpu.lowering_config<{
    lane_basis = [[1, 1, 1, 1, 64], [1, 0, 3, 4]],
    subgroup_basis = [[1, 1, 1, 1, 8], [0, 1, 2, 4]],
    thread = [0, 0, 8, 0]
  }>
}

func.func @contraction_ceildiv_batch(%lhs: tensor<1x1x63xf16>,
                                     %rhs: tensor<1x512x63xf16>,
                                     %init: tensor<1x512x1xf32>)
                                     -> tensor<1x512x1xf32>
                                     attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<1x1x63xf16>, tensor<1x512x63xf16>)
                        outs(%init: tensor<1x512x1xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %mul, %out : f32
      linalg.yield %sum : f32
  } -> tensor<1x512x1xf32>
  return %out : tensor<1x512x1xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<{{.*}}batch_tile = [1, 1, 8]{{.*}}element_tile = [1, 1, 8]{{.*}}>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<{{.*}}batch_tile = [1, 1, 8]{{.*}}element_tile = [1, 1, 8]{{.*}}>

// CHECK-LABEL: func.func @contraction_ceildiv_batch

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
