// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_mfma_16x16x16(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<4xf16>, vector<4xf16> into vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_mfma_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4xf32>
//       CHECK:   amdgpu.mfma 16x16x16 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_mfma_32x32x8(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<16xf32>) -> vector<16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<4xf16>, vector<4xf16> into vector<16xf32>
  return %0 : vector<16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_mfma_32x32x8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<16xf32>
//       CHECK:   amdgpu.mfma 32x32x8 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_col_major_multi_mma_mfma_32x32x8(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<16xf32>) -> vector<16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16, col_major = true>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<4xf16>, vector<4xf16> into vector<16xf32>
  return %0 : vector<16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_col_major_multi_mma_mfma_32x32x8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<16xf32>
//       CHECK:   amdgpu.mfma 32x32x8 %[[RHS]] * %[[LHS]] + %[[ACC]]
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>

// -----

#contraction_accesses = [
  affine_map<() -> ()>,
  affine_map<() -> ()>,
  affine_map<() -> ()>
]

func.func @lower_col_major_inner_tiled_virtual_16x16x32(%lhs: vector<8xf16>, %rhs: vector<8xf16>, %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VMFMA_F32_16x16x32_F16, col_major = true>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xf16>, vector<8xf16> into vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_col_major_inner_tiled_virtual_16x16x32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4xf32>
//  CHECK: %[[LHS0:.*]] = vector.extract_strided_slice %[[LHS]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//  CHECK: %[[RHS0:.*]] = vector.extract_strided_slice %[[RHS]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//  CHECK: %[[ACC0:.*]] = amdgpu.mfma 16x16x16 %[[RHS0]] * %[[LHS0]] + %[[ACC]]
//  CHECK: %[[LHS1:.*]] = vector.extract_strided_slice %[[LHS]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//  CHECK: %[[RHS1:.*]] = vector.extract_strided_slice %[[RHS]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//  CHECK: %[[ACC1:.*]] = amdgpu.mfma 16x16x16 %[[RHS1]] * %[[LHS1]] + %[[ACC0]]
//  CHECK: return %[[ACC1]] : vector<4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmmar3_16x16x16(%lhs: vector<16xf16>, %rhs: vector<16xf16>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf16>, vector<16xf16> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmmar3_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x16 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<16xf16>, vector<16xf16>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmmar4_16x16x16(%lhs: vector<8xf16>, %rhs: vector<8xf16>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xf16>, vector<8xf16> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmmar4_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x16 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<8xf16>, vector<8xf16>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_f32_16x16x4_f32(%lhs: vector<2xf32>, %rhs: vector<2xf32>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F32_16x16x4_F32>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<2xf32>, vector<2xf32> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_f32_16x16x4_f32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<2xf32>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<2xf32>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x4 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<2xf32>, vector<2xf32>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_f32_16x16x32_f16(%lhs: vector<16xf16>, %rhs: vector<16xf16>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf16>, vector<16xf16> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_f32_16x16x32_f16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x32 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<16xf16>, vector<16xf16>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_f32_16x16x64_f8E4M3FN(%lhs: vector<32xf8E4M3FN>, %rhs: vector<32xf8E4M3FN>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F32_16x16x64_F8E4M3FN>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<32xf8E4M3FN>, vector<32xf8E4M3FN> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_f32_16x16x64_f8E4M3FN
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<32xf8E4M3FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<32xf8E4M3FN>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x64 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_f32_16x16x128_f8E4M3FN(%lhs: vector<64xf8E4M3FN>, %rhs: vector<64xf8E4M3FN>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F32_16x16x128_F8E4M3FN>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<64xf8E4M3FN>, vector<64xf8E4M3FN> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_f32_16x16x128_f8E4M3FN
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<64xf8E4M3FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<64xf8E4M3FN>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x128 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_f16_16x16x128_f8E4M3FN(%lhs: vector<64xf8E4M3FN>, %rhs: vector<64xf8E4M3FN>, %acc: vector<8xf16>) -> vector<8xf16> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F16_16x16x128_F8E4M3FN>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<64xf8E4M3FN>, vector<64xf8E4M3FN> into vector<8xf16>
  return %0 : vector<8xf16>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_f16_16x16x128_f8E4M3FN
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<64xf8E4M3FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<64xf8E4M3FN>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf16>
//       CHECK:   amdgpu.wmma 16x16x128 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf16>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_mfma_shape_cast_16x16x16(%lhs: vector<1x4xf16>, %rhs: vector<4x1xf16>, %acc: vector<4x1xf32>) -> vector<4x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<1x4xf16>, vector<4x1xf16> into vector<4x1xf32>
  return %0 : vector<4x1xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_mfma_shape_cast_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<1x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4x1xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4x1xf32>
//   CHECK-DAG:   %[[LHSCAST:.+]] = vector.shape_cast %[[LHS]] : vector<1x4xf16> to vector<4xf16>
//   CHECK-DAG:   %[[RHSCAST:.+]] = vector.shape_cast %[[RHS]] : vector<4x1xf16> to vector<4xf16>
//   CHECK-DAG:   %[[ACCCAST:.+]] = vector.shape_cast %[[ACC]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:   %[[MMA:.+]] = amdgpu.mfma 16x16x16 %[[LHSCAST]] * %[[RHSCAST]] + %[[ACCCAST]]
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//       CHECK:   vector.shape_cast %[[MMA]] : vector<4xf32> to vector<4x1xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_inner_tiled_mfma_scale_f32_16x16x128_b32(
      %lhs: vector<32xf4E2M1FN>, %rhs: vector<32xf8E4M3FN>, %lhsScale: vector<1xf8E8M0FNU>, %rhsScale: vector<1xf8E8M0FNU>,
      %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhsScale, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<32xf4E2M1FN>, vector<32xf8E4M3FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_inner_tiled_mfma_scale_f32_16x16x128_b32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<32xf8E4M3FN>
//  CHECK-SAME:   %[[LHS_SCALE:[A-Za-z0-9]+]]: vector<1xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS_SCALE:[A-Za-z0-9]+]]: vector<1xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4xf32>
//  CHECK: %[[CST:.+]] = arith.constant dense<5.877470e-39> : vector<4xf8E8M0FNU>
//  CHECK: %[[LHS_SCALE_SCALAR:.+]] = vector.extract %[[LHS_SCALE]][0]
//  CHECK: %[[LHS_SCALE_LONG:.+]] = vector.insert %[[LHS_SCALE_SCALAR]], %[[CST]] [0]
//  CHECK: %[[RHS_SCALE_SCALAR:.+]] = vector.extract %[[RHS_SCALE]][0]
//  CHECK: %[[RHS_SCALE_LONG:.+]] = vector.insert %[[RHS_SCALE_SCALAR]], %[[CST]] [0]
//  CHECK: amdgpu.scaled_mfma 16x16x128 (%[[LHS_SCALE_LONG]][0] * %[[LHS]]) * (%[[RHS_SCALE_LONG]][0] * %[[RHS]]) + %[[ACC]]
//  CHECK-SAME: vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf8E4M3FN>, vector<4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_inner_tiled_mfma_scale_f32_32x32x64_b32(
      %lhs: vector<32xf4E2M1FN>, %rhs: vector<32xf8E4M3FN>, %lhsScale: vector<1xf8E8M0FNU>, %rhsScale: vector<1xf8E8M0FNU>,
      %acc: vector<16xf32>) -> vector<16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhsScale, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_32x32x64_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<32xf4E2M1FN>, vector<32xf8E4M3FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<16xf32>
  return %0 : vector<16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_inner_tiled_mfma_scale_f32_32x32x64_b32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<32xf8E4M3FN>
//  CHECK-SAME:   %[[LHS_SCALE:[A-Za-z0-9]+]]: vector<1xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS_SCALE:[A-Za-z0-9]+]]: vector<1xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<16xf32>
//  CHECK: %[[CST:.+]] = arith.constant dense<5.877470e-39> : vector<4xf8E8M0FNU>
//  CHECK: %[[LHS_SCALE_SCALAR:.+]] = vector.extract %[[LHS_SCALE]][0]
//  CHECK: %[[LHS_SCALE_LONG:.+]] = vector.insert %[[LHS_SCALE_SCALAR]], %[[CST]] [0]
//  CHECK: %[[RHS_SCALE_SCALAR:.+]] = vector.extract %[[RHS_SCALE]][0]
//  CHECK: %[[RHS_SCALE_LONG:.+]] = vector.insert %[[RHS_SCALE_SCALAR]], %[[CST]] [0]
//  CHECK: amdgpu.scaled_mfma 32x32x64 (%[[LHS_SCALE_LONG]][0] * %[[LHS]]) * (%[[RHS_SCALE_LONG]][0] * %[[RHS]]) + %[[ACC]]
//  CHECK-SAME: vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf8E4M3FN>, vector<16xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_apple_simdgroup_f16_8x8x8_f32(
    %lhs: vector<2xf16>, %rhs: vector<2xf16>,
    %acc: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<APPLE_SIMDGROUP_F32_8x8x8_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<2xf16>, vector<2xf16> into vector<2xf32>
  return %0 : vector<2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_apple_simdgroup_f16_8x8x8_f32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<2xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<2xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<2xf32>
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : i32
//   CHECK-DAG:   %[[C17:.+]] = arith.constant 17 : i32
//   CHECK-DAG:   %[[C12:.+]] = arith.constant 12 : i32
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : i32
//   CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : i32
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : i32
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : i32
//       CHECK:   %[[HARDWARE_LANE:.+]] = gpu.lane_id upper_bound 32
//       CHECK:   %[[HARDWARE_LANE_I32:.+]] = arith.index_cast %[[HARDWARE_LANE]] : index to i32
//       CHECK:   %[[KEEP:.+]] = arith.andi %[[HARDWARE_LANE_I32]], %[[C17]]
//       CHECK:   %[[RIGHT_BIT:.+]] = arith.andi %[[HARDWARE_LANE_I32]], %[[C8]]
//       CHECK:   %[[RIGHT:.+]] = arith.shrui %[[RIGHT_BIT]], %[[C2]]
//       CHECK:   %[[LEFT_BITS:.+]] = arith.andi %[[HARDWARE_LANE_I32]], %[[C6]]
//       CHECK:   %[[LEFT:.+]] = arith.shli %[[LEFT_BITS]], %[[C1]]
//       CHECK:   %[[PARTIAL_SOURCE:.+]] = arith.ori %[[KEEP]], %[[RIGHT]]
//       CHECK:   %[[HARDWARE_SOURCE:.+]] = arith.ori %[[PARTIAL_SOURCE]], %[[LEFT]]
//       CHECK:   %[[PACKED_LHS:.+]] = vector.bitcast %[[LHS]] : vector<2xf16> to vector<1xi32>
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : i32
//       CHECK:   %[[PACKED_RHS:.+]] = vector.bitcast %[[RHS]] : vector<2xf16> to vector<1xi32>
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : i32
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : f32
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : f32
//       CHECK:   gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<8x8xf16, "AOp">
//       CHECK:   gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<8x8xf16, "BOp">
//       CHECK:   gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<8x8xf32, "COp">
//       CHECK:   %[[MMA:.+]] = gpu.subgroup_mma_compute {{.*}} : !gpu.mma_matrix<8x8xf16, "AOp">, !gpu.mma_matrix<8x8xf16, "BOp"> -> !gpu.mma_matrix<8x8xf32, "COp">
//       CHECK:   %[[RESULT0:.+]] = gpu.subgroup_mma_extract_thread_local %[[MMA]]
//       CHECK:   %[[RESULT1:.+]] = gpu.subgroup_mma_extract_thread_local %[[MMA]]
//       CHECK:   %[[CANONICAL_LANE:.+]] = gpu.lane_id upper_bound 32
//       CHECK:   %[[CANONICAL_LANE_I32:.+]] = arith.index_cast %[[CANONICAL_LANE]] : index to i32
//       CHECK:   %[[CANONICAL_KEEP:.+]] = arith.andi %[[CANONICAL_LANE_I32]], %[[C17]]
//       CHECK:   %[[CANONICAL_RIGHT_BITS:.+]] = arith.andi %[[CANONICAL_LANE_I32]], %[[C12]]
//       CHECK:   %[[CANONICAL_RIGHT:.+]] = arith.shrui %[[CANONICAL_RIGHT_BITS]], %[[C1]]
//       CHECK:   %[[CANONICAL_LEFT_BIT:.+]] = arith.andi %[[CANONICAL_LANE_I32]], %[[C2]]
//       CHECK:   %[[CANONICAL_LEFT:.+]] = arith.shli %[[CANONICAL_LEFT_BIT]], %[[C2]]
//       CHECK:   %[[CANONICAL_PARTIAL:.+]] = arith.ori %[[CANONICAL_KEEP]], %[[CANONICAL_RIGHT]]
//       CHECK:   %[[CANONICAL_SOURCE:.+]] = arith.ori %[[CANONICAL_PARTIAL]], %[[CANONICAL_LEFT]]
//       CHECK:   gpu.shuffle idx %[[RESULT0]], %[[CANONICAL_SOURCE]], %[[C32]] : f32
//       CHECK:   gpu.shuffle idx %[[RESULT1]], %[[CANONICAL_SOURCE]], %[[C32]] : f32

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_apple_simdgroup_bf16_8x8x8_f32(
    %lhs: vector<2xbf16>, %rhs: vector<2xbf16>,
    %acc: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<APPLE_SIMDGROUP_F32_8x8x8_BF16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<2xbf16>, vector<2xbf16> into vector<2xf32>
  return %0 : vector<2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_apple_simdgroup_bf16_8x8x8_f32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<2xbf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<2xbf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<2xf32>
//       CHECK:   vector.bitcast %[[LHS]] : vector<2xbf16> to vector<1xi32>
//       CHECK:   gpu.shuffle idx {{.*}} : i32
//       CHECK:   vector.bitcast %[[RHS]] : vector<2xbf16> to vector<1xi32>
//       CHECK:   gpu.shuffle idx {{.*}} : i32
//       CHECK:   gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<8x8xbf16, "AOp">
//       CHECK:   gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<8x8xbf16, "BOp">
//       CHECK:   gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<8x8xf32, "COp">
//       CHECK:   gpu.subgroup_mma_compute {{.*}} : !gpu.mma_matrix<8x8xbf16, "AOp">, !gpu.mma_matrix<8x8xbf16, "BOp"> -> !gpu.mma_matrix<8x8xf32, "COp">

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_apple_simdgroup_f16_16x16x16_f32(
    %lhs: vector<8xf16>, %rhs: vector<8xf16>,
    %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<APPLE_SIMDGROUP_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xf16>, vector<8xf16> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_apple_simdgroup_f16_16x16x16_f32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//   CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//   CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
//   CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : i32
//       CHECK:   %[[HARDWARE_LANE:.+]] = gpu.lane_id upper_bound 32
//       CHECK:   %[[HARDWARE_LANE_I32:.+]] = arith.index_cast %[[HARDWARE_LANE]] : index to i32
//       CHECK:   %[[KEEP:.+]] = arith.andi %[[HARDWARE_LANE_I32]]
//       CHECK:   %[[RIGHT_BIT:.+]] = arith.andi %[[HARDWARE_LANE_I32]]
//       CHECK:   %[[RIGHT:.+]] = arith.shrui %[[RIGHT_BIT]]
//       CHECK:   %[[LEFT_BITS:.+]] = arith.andi %[[HARDWARE_LANE_I32]]
//       CHECK:   %[[LEFT:.+]] = arith.shli %[[LEFT_BITS]]
//       CHECK:   %[[PARTIAL_SOURCE:.+]] = arith.ori %[[KEEP]], %[[RIGHT]]
//       CHECK:   %[[HARDWARE_SOURCE:.+]] = arith.ori %[[PARTIAL_SOURCE]], %[[LEFT]]
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [0], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : i32
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [2], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : i32
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [4], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : i32
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [6], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : i32
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [0], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : i32
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [2], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : i32
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [4], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : i32
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [6], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : i32
// CHECK-COUNT-8:   gpu.shuffle idx {{.*}}, %[[HARDWARE_SOURCE]], %[[C32]] : f32
//       CHECK:   %[[AMAT:.+]] = gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<16x16xf16, "AOp">
//       CHECK:   %[[A0:.+]] = gpu.subgroup_mma_insert_thread_local {{.*}}, %[[AMAT]][%[[C0]]]
//       CHECK:   %[[A1:.+]] = gpu.subgroup_mma_insert_thread_local {{.*}}, %[[A0]][%[[C1]]]
//       CHECK:   %[[A2:.+]] = gpu.subgroup_mma_insert_thread_local {{.*}}, %[[A1]][%[[C2]]]
//       CHECK:   %[[A3:.+]] = gpu.subgroup_mma_insert_thread_local {{.*}}, %[[A2]][%[[C3]]]
//       CHECK:   %[[A4:.+]] = gpu.subgroup_mma_insert_thread_local {{.*}}, %[[A3]][%[[C4]]]
//       CHECK:   %[[A5:.+]] = gpu.subgroup_mma_insert_thread_local {{.*}}, %[[A4]][%[[C5]]]
//       CHECK:   %[[A6:.+]] = gpu.subgroup_mma_insert_thread_local {{.*}}, %[[A5]][%[[C6]]]
//       CHECK:   %[[A7:.+]] = gpu.subgroup_mma_insert_thread_local {{.*}}, %[[A6]][%[[C7]]]
//       CHECK:   %[[BMAT:.+]] = gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-COUNT-8:   gpu.subgroup_mma_insert_thread_local
//       CHECK:   %[[CMAT:.+]] = gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-COUNT-8:   gpu.subgroup_mma_insert_thread_local
//       CHECK:   %[[MMA:.+]] = gpu.subgroup_mma_compute %[[A7]], {{.*}} : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
//       CHECK:   %[[RESULT0:.+]] = gpu.subgroup_mma_extract_thread_local %[[MMA]][%[[C0]]]
//       CHECK:   %[[RESULT1:.+]] = gpu.subgroup_mma_extract_thread_local %[[MMA]][%[[C1]]]
//       CHECK:   %[[RESULT2:.+]] = gpu.subgroup_mma_extract_thread_local %[[MMA]][%[[C2]]]
//       CHECK:   %[[RESULT3:.+]] = gpu.subgroup_mma_extract_thread_local %[[MMA]][%[[C3]]]
//       CHECK:   %[[RESULT4:.+]] = gpu.subgroup_mma_extract_thread_local %[[MMA]][%[[C4]]]
//       CHECK:   %[[RESULT5:.+]] = gpu.subgroup_mma_extract_thread_local %[[MMA]][%[[C5]]]
//       CHECK:   %[[RESULT6:.+]] = gpu.subgroup_mma_extract_thread_local %[[MMA]][%[[C6]]]
//       CHECK:   %[[RESULT7:.+]] = gpu.subgroup_mma_extract_thread_local %[[MMA]][%[[C7]]]
//       CHECK:   %[[CANONICAL_LANE:.+]] = gpu.lane_id upper_bound 32
//       CHECK:   %[[CANONICAL_LANE_I32:.+]] = arith.index_cast %[[CANONICAL_LANE]] : index to i32
//       CHECK:   %[[CANONICAL_KEEP:.+]] = arith.andi %[[CANONICAL_LANE_I32]]
//       CHECK:   %[[CANONICAL_RIGHT_BITS:.+]] = arith.andi %[[CANONICAL_LANE_I32]]
//       CHECK:   %[[CANONICAL_RIGHT:.+]] = arith.shrui %[[CANONICAL_RIGHT_BITS]]
//       CHECK:   %[[CANONICAL_LEFT_BIT:.+]] = arith.andi %[[CANONICAL_LANE_I32]]
//       CHECK:   %[[CANONICAL_LEFT:.+]] = arith.shli %[[CANONICAL_LEFT_BIT]]
//       CHECK:   %[[CANONICAL_PARTIAL:.+]] = arith.ori %[[CANONICAL_KEEP]], %[[CANONICAL_RIGHT]]
//       CHECK:   %[[CANONICAL_SOURCE:.+]] = arith.ori %[[CANONICAL_PARTIAL]], %[[CANONICAL_LEFT]]
// CHECK-COUNT-8:   gpu.shuffle idx {{.*}}, %[[CANONICAL_SOURCE]], %[[C32]] : f32

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_apple_simdgroup_bf16_16x16x16_f32(
    %lhs: vector<8xbf16>, %rhs: vector<8xbf16>,
    %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<APPLE_SIMDGROUP_F32_16x16x16_BF16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xbf16>, vector<8xbf16> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_apple_simdgroup_bf16_16x16x16_f32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<8xbf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<8xbf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [0], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}} : i32
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [2], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}} : i32
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [4], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}} : i32
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [6], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}} : i32
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [0], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}} : i32
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [2], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}} : i32
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [4], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}} : i32
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [6], sizes = [2], strides = [1]}
//       CHECK:   gpu.shuffle idx {{.*}} : i32
// CHECK-COUNT-8:   gpu.shuffle idx {{.*}} : f32
//       CHECK:   gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<16x16xbf16, "AOp">
// CHECK-COUNT-8:   gpu.subgroup_mma_insert_thread_local {{.*}} : bf16, !gpu.mma_matrix<16x16xbf16, "AOp">
//       CHECK:   gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<16x16xbf16, "BOp">
// CHECK-COUNT-8:   gpu.subgroup_mma_insert_thread_local {{.*}} : bf16, !gpu.mma_matrix<16x16xbf16, "BOp">
//       CHECK:   gpu.subgroup_mma_constant_matrix {{.*}} : !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-COUNT-8:   gpu.subgroup_mma_insert_thread_local {{.*}} : f32, !gpu.mma_matrix<16x16xf32, "COp">
//       CHECK:   gpu.subgroup_mma_compute {{.*}} : !gpu.mma_matrix<16x16xbf16, "AOp">, !gpu.mma_matrix<16x16xbf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-COUNT-8:   gpu.subgroup_mma_extract_thread_local
// CHECK-COUNT-8:   gpu.shuffle idx {{.*}} : f32

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_apple16_rhs_4x2_with_transpose_provenance(
    %lhs: vector<2x4xf16>, %acc: vector<2x4xf32>) -> vector<2x4xf32> {
  %rhs_elements = arith.constant dense<[
    [0.000000e+00, 1.000000e+00],
    [2.000000e+00, 3.000000e+00],
    [4.000000e+00, 5.000000e+00],
    [6.000000e+00, 7.000000e+00]
  ]> : vector<4x2xf16>
  %rhs = util.optimization_barrier %rhs_elements : vector<4x2xf16>
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<APPLE_SIMDGROUP_F32_16x16x16_F16>,
    permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>],
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
  } : vector<2x4xf16>, vector<4x2xf16> into vector<2x4xf32>
  return %0 : vector<2x4xf32>
}

func.func @lower_apple16_rhs_4x2_without_provenance(
    %lhs: vector<2x4xf16>, %acc: vector<2x4xf32>) -> vector<2x4xf32> {
  %rhs_elements = arith.constant dense<[
    [8.000000e+00, 9.000000e+00],
    [1.000000e+01, 1.100000e+01],
    [1.200000e+01, 1.300000e+01],
    [1.400000e+01, 1.500000e+01]
  ]> : vector<4x2xf16>
  %rhs = util.optimization_barrier %rhs_elements : vector<4x2xf16>
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<APPLE_SIMDGROUP_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
  } : vector<2x4xf16>, vector<4x2xf16> into vector<2x4xf32>
  return %0 : vector<2x4xf32>
}

func.func @lower_apple16_rhs_2x4(
    %lhs: vector<2x4xf16>, %acc: vector<2x4xf32>) -> vector<2x4xf32> {
  %rhs_elements = arith.constant dense<[
    [0.000000e+00, 2.000000e+00, 4.000000e+00, 6.000000e+00],
    [1.000000e+00, 3.000000e+00, 5.000000e+00, 7.000000e+00]
  ]> : vector<2x4xf16>
  %rhs = util.optimization_barrier %rhs_elements : vector<2x4xf16>
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<APPLE_SIMDGROUP_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
  } : vector<2x4xf16>, vector<2x4xf16> into vector<2x4xf32>
  return %0 : vector<2x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @lower_apple16_rhs_4x2_with_transpose_provenance(
//  CHECK-SAME:   %[[LHS:.+]]: vector<2x4xf16>
//       CHECK:   %[[RHS_ELEMENTS:.+]] = arith.constant dense<{{.*}}> : vector<4x2xf16>
//       CHECK:   %[[RHS:.+]] = util.optimization_barrier %[[RHS_ELEMENTS]] : vector<4x2xf16>
//       CHECK:   %[[LHS8:.+]] = vector.shape_cast %[[LHS]] : vector<2x4xf16> to vector<8xf16>
//  CHECK-NEXT:   %[[RHST:.+]] = vector.transpose %[[RHS]], [1, 0] : vector<4x2xf16> to vector<2x4xf16>
//  CHECK-NEXT:   %[[RHS8:.+]] = vector.shape_cast %[[RHST]] : vector<2x4xf16> to vector<8xf16>
//       CHECK:   vector.extract_strided_slice %[[RHS8]] {offsets = [0], sizes = [2], strides = [1]}

// CHECK-LABEL: func.func @lower_apple16_rhs_4x2_without_provenance(
//  CHECK-SAME:   %[[LHS:.+]]: vector<2x4xf16>
//       CHECK:   %[[RHS_ELEMENTS:.+]] = arith.constant dense<{{.*}}> : vector<4x2xf16>
//       CHECK:   %[[RHS:.+]] = util.optimization_barrier %[[RHS_ELEMENTS]] : vector<4x2xf16>
//       CHECK:   %[[LHS8:.+]] = vector.shape_cast %[[LHS]] : vector<2x4xf16> to vector<8xf16>
//   CHECK-NOT:   vector.transpose %[[RHS]]
//  CHECK-NEXT:   %[[RHS8:.+]] = vector.shape_cast %[[RHS]] : vector<4x2xf16> to vector<8xf16>
//       CHECK:   vector.extract_strided_slice %[[RHS8]] {offsets = [0], sizes = [2], strides = [1]}

// CHECK-LABEL: func.func @lower_apple16_rhs_2x4(
//  CHECK-SAME:   %[[LHS:.+]]: vector<2x4xf16>
//       CHECK:   %[[RHS_ELEMENTS:.+]] = arith.constant dense<{{.*}}> : vector<2x4xf16>
//       CHECK:   %[[RHS:.+]] = util.optimization_barrier %[[RHS_ELEMENTS]] : vector<2x4xf16>
//       CHECK:   %[[LHS8:.+]] = vector.shape_cast %[[LHS]] : vector<2x4xf16> to vector<8xf16>
//   CHECK-NOT:   vector.transpose %[[RHS]]
//  CHECK-NEXT:   %[[RHS8:.+]] = vector.shape_cast %[[RHS]] : vector<2x4xf16> to vector<8xf16>
//       CHECK:   vector.extract_strided_slice %[[RHS8]] {offsets = [0], sizes = [2], strides = [1]}

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_chained_apple16_mmas(
    %lhs0: vector<8xf16>, %rhs0: vector<8xf16>,
    %lhs1: vector<8xf16>, %rhs1: vector<8xf16>,
    %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs0, %rhs0) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<APPLE_SIMDGROUP_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xf16>, vector<8xf16> into vector<8xf32>
  %1 = iree_codegen.inner_tiled ins(%lhs1, %rhs1) outs(%0) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<APPLE_SIMDGROUP_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xf16>, vector<8xf16> into vector<8xf32>
  return %1 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @lower_chained_apple16_mmas(
// CHECK-COUNT-16: gpu.shuffle
//          CHECK: gpu.subgroup_mma_compute
//  CHECK-COUNT-8: gpu.shuffle
//          CHECK: gpu.subgroup_mma_compute
//  CHECK-COUNT-8: gpu.shuffle
//       CHECK: return
