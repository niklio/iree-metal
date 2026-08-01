// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-opt %s --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-expand-dimensions{expand-apple-physical-fragments=true}))"

#physical = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>
}>

func.func @reject_dynamic_physical_fragment(
    %lhs: tensor<?x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<?x48xf32>) -> tensor<?x48xf32> {
  // expected-error @+1 {{Apple physical fragment dimension must be a positive static multiple of the intrinsic size}}
  %result = linalg.matmul {lowering_config = #physical}
      ins(%lhs, %rhs : tensor<?x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<?x48xf32>) -> tensor<?x48xf32>
  return %result : tensor<?x48xf32>
}

// -----

#physical = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>
}>

func.func @reject_nondivisible_physical_fragment(
    %lhs: tensor<30x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<30x48xf32>) -> tensor<30x48xf32> {
  // expected-error @+1 {{Apple physical fragment dimension must be a positive static multiple of the intrinsic size}}
  %result = linalg.matmul {lowering_config = #physical}
      ins(%lhs, %rhs : tensor<30x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<30x48xf32>) -> tensor<30x48xf32>
  return %result : tensor<30x48xf32>
}

// -----

#physical = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>,
  subgroup_basis = [[1, 1, 1]]
}>

func.func @reject_malformed_basis_outer_arity(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<32x48xf32>) -> tensor<32x48xf32> {
  // expected-error @+1 {{malformed subgroup_basis}}
  %result = linalg.matmul {lowering_config = #physical}
      ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}

// -----

#physical = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>,
  lane_basis = [["not-an-integer"], [0, 1, 2]]
}>

func.func @reject_malformed_basis_inner_type(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<32x48xf32>) -> tensor<32x48xf32> {
  // expected-error @+1 {{malformed lane_basis}}
  %result = linalg.matmul {lowering_config = #physical}
      ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}

// -----

#physical = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>,
  subgroup_basis = [[1, 1, 1], [0, 1]]
}>

func.func @reject_basis_mapping_rank(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<32x48xf32>) -> tensor<32x48xf32> {
  // expected-error @+1 {{malformed subgroup_basis}}
  %result = linalg.matmul {lowering_config = #physical}
      ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}

// -----

#physical = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>,
  subgroup_basis = [[1, 1, 1], [0, 1, 3]]
}>

func.func @reject_basis_mapping_out_of_range(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<32x48xf32>) -> tensor<32x48xf32> {
  // expected-error @+1 {{out-of-range mapping in subgroup_basis}}
  %result = linalg.matmul {lowering_config = #physical}
      ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}

// -----

#physical = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>,
  subgroup_basis = [[1, 0, 1], [0, 1, 2]]
}>

func.func @reject_nonpositive_basis_count(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<32x48xf32>) -> tensor<32x48xf32> {
  // expected-error @+1 {{non-positive count in subgroup_basis}}
  %result = linalg.matmul {lowering_config = #physical}
      ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}

// -----

#physical = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>,
  subgroup_basis = [[2, 1, 1], [0, 0, 2]]
}>

func.func @reject_duplicate_basis_mapping(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<32x48xf32>) -> tensor<32x48xf32> {
  // expected-error @+1 {{duplicate mapping in subgroup_basis}}
  %result = linalg.matmul {lowering_config = #physical}
      ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}

// -----

#physical = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>,
  workgroup = [-1, 48, 0]
}>

func.func @reject_negative_tiling_size(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<32x48xf32>) -> tensor<32x48xf32> {
  // expected-error @+1 {{negative size in tiling level 'workgroup'}}
  %result = linalg.matmul {lowering_config = #physical}
      ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}

// -----

#rank_mismatch = #iree_gpu.lowering_config<{
  expand_dims = #iree_gpu.expand_dims<
      [[0], [1, 2]], output_shape = [?, ?, 4]>
}>

func.func @reject_fresh_expand_dims_rank_mismatch(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<32x48xf32>) -> tensor<32x48xf32> {
  // expected-error @+1 {{expand_dims reassociation count must match operation loop rank}}
  %result = linalg.matmul {lowering_config = #rank_mismatch}
      ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}

// -----

#malformed_expand = #iree_gpu.lowering_config<{
  expand_dims
}>

func.func @reject_malformed_expand_dims_type(
    %lhs: tensor<32x64xf16>, %rhs: tensor<64x48xf16>,
    %acc: tensor<32x48xf32>) -> tensor<32x48xf32> {
  // expected-error @+1 {{malformed expand_dims lowering config}}
  %result = linalg.matmul {lowering_config = #malformed_expand}
      ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x48xf16>)
      outs(%acc : tensor<32x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}

// -----

#materialized_maps = [
  affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) ->
      (d0, d1, d2, d3, d8, d9, d10, d11)>,
  affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) ->
      (d4, d5, d6, d7, d8, d9, d10, d11)>,
  affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) ->
      (d4, d5, d6, d7, d0, d1, d2, d3)>
]
#stale_materialized = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>,
  apple_physical_fragment_expansion_materialized,
  subgroup_basis = [[1, 1, 1], [0, 1, 2]]
}>

func.func @reject_stale_materialized_basis(
    %lhs: tensor<1x2x2x4x1x2x2x4xf16>,
    %rhs: tensor<1x2x2x4x1x2x2x4xf16>,
    %acc: tensor<1x2x2x4x1x2x2x4xf32>)
    -> tensor<1x2x2x4x1x2x2x4xf32> {
  // expected-error @+1 {{malformed materialized subgroup_basis}}
  %result = linalg.generic {
      indexing_maps = #materialized_maps,
      iterator_types = [
        "parallel", "parallel", "parallel", "parallel",
        "parallel", "parallel", "parallel", "parallel",
        "reduction", "reduction", "reduction", "reduction"],
      lowering_config = #stale_materialized
    } ins(%lhs, %rhs : tensor<1x2x2x4x1x2x2x4xf16>,
                       tensor<1x2x2x4x1x2x2x4xf16>)
      outs(%acc : tensor<1x2x2x4x1x2x2x4xf32>) {
    ^bb0(%l: f16, %r: f16, %old: f32):
      %lf = arith.extf %l : f16 to f32
      %rf = arith.extf %r : f16 to f32
      %mul = arith.mulf %lf, %rf : f32
      %sum = arith.addf %old, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<1x2x2x4x1x2x2x4xf32>
  return %result : tensor<1x2x2x4x1x2x2x4xf32>
}
