// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: env IREE_METAL_CAUSAL_TRIANGULAR_GRID=1 IREE_METAL_CAUSAL_BWD_BOUNDS=1 iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op,cse))" --mlir-print-local-scope %s | FileCheck %s --check-prefix=TRIANGULAR --implicit-check-not=scf.if
// RUN: env -u IREE_METAL_CAUSAL_TRIANGULAR_GRID -u IREE_METAL_CAUSAL_BWD_BOUNDS iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op,cse))" --mlir-print-local-scope %s | FileCheck %s --check-prefix=BASELINE
// RUN: env IREE_METAL_CAUSAL_TRIANGULAR_GRID=0 IREE_METAL_CAUSAL_BWD_BOUNDS=1 iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op,cse))" --mlir-print-local-scope %s | FileCheck %s --check-prefix=BASELINE
// RUN: env IREE_METAL_CAUSAL_TRIANGULAR_GRID=true IREE_METAL_CAUSAL_BWD_BOUNDS=1 iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op,cse))" --mlir-print-local-scope %s | FileCheck %s --check-prefix=BASELINE
// RUN: not env IREE_METAL_CAUSAL_TRIANGULAR_GRID=1 IREE_METAL_CAUSAL_BWD_BOUNDS=0 iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op,cse))" %s 2>&1 | FileCheck %s --check-prefix=MISSING-BOUNDS

func.func @qk_t512(
    %query: tensor<96x512x64xbf16>,
    %key: tensor<96x512x64xbf16>,
    %init: tensor<96x512x512xf32>) -> tensor<96x512x512xf32>
    attributes {
      translation_info = #iree_codegen.translation_info<
          pipeline = SPIRVAppleVectorDistributeAttention>} {
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(b, m, n, k) -> (b, m, k)>,
        affine_map<(b, m, n, k) -> (b, n, k)>,
        affine_map<(b, m, n, k) -> (b, m, n)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%query, %key : tensor<96x512x64xbf16>, tensor<96x512x64xbf16>)
      outs(%init : tensor<96x512x512xf32>)
      attrs = {
        iree_codegen.apple_attention_backward_causal_score,
        iree_codegen.apple_attention_backward_causal_score_alignment = 128 : i64,
        iree_codegen.apple_attention_backward_role = "qk_attrs",
        lowering_config = #iree_codegen.lowering_config<
            tile_sizes = [[1, 64, 64, 0]]>} {
    ^bb0(%lhs: bf16, %rhs: bf16, %acc: f32):
      %lhs_f32 = arith.extf %lhs : bf16 to f32
      %rhs_f32 = arith.extf %rhs : bf16 to f32
      %product = arith.mulf %lhs_f32, %rhs_f32 : f32
      %sum = arith.addf %product, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<96x512x512xf32>
  return %result : tensor<96x512x512xf32>
}

// TRIANGULAR-LABEL: func.func @qk_t512(
// TRIANGULAR-DAG: %[[QK_C64:.+]] = arith.constant 64 : index
// TRIANGULAR-DAG: %[[QK_C6:.+]] = arith.constant 6 : index
// TRIANGULAR-DAG: %[[QK_C2:.+]] = arith.constant 2 : index
// TRIANGULAR-DAG: %[[QK_C3:.+]] = arith.constant 3 : index
// TRIANGULAR-DAG: %[[QK_C1:.+]] = arith.constant 1 : index
// TRIANGULAR-DAG: %[[QK_C4:.+]] = arith.constant 4 : index
// TRIANGULAR: scf.forall (%[[QK_B:.+]], %[[QK_Y:.+]], %[[QK_ORD:.+]]) in (96, 1, 40)
// TRIANGULAR: %[[QK_COARSE:.+]] = arith.divui %[[QK_ORD]], %[[QK_C4]] : index
// TRIANGULAR: %[[QK_INTRA:.+]] = arith.remui %[[QK_ORD]], %[[QK_C4]] : index
// TRIANGULAR: arith.cmpi uge, %[[QK_COARSE]], %[[QK_C1]] : index
// TRIANGULAR: arith.cmpi uge, %[[QK_COARSE]], %[[QK_C3]] : index
// TRIANGULAR: %[[QK_GE6:.+]] = arith.cmpi uge, %[[QK_COARSE]], %[[QK_C6]] : index
// TRIANGULAR: %[[QK_MACRO_ROW:.+]] = arith.select %[[QK_GE6]], %[[QK_C3]], %{{.+}} : index
// TRIANGULAR: %[[QK_MACRO_BASE:.+]] = arith.select %[[QK_GE6]], %[[QK_C6]], %{{.+}} : index
// TRIANGULAR: %[[QK_MACRO_COL:.+]] = arith.subi %[[QK_COARSE]], %[[QK_MACRO_BASE]] : index
// TRIANGULAR: %[[QK_INTRA_ROW:.+]] = arith.divui %[[QK_INTRA]], %[[QK_C2]] : index
// TRIANGULAR: %[[QK_INTRA_COL:.+]] = arith.remui %[[QK_INTRA]], %[[QK_C2]] : index
// TRIANGULAR: %[[QK_ROW_BASE:.+]] = arith.muli %[[QK_MACRO_ROW]], %[[QK_C2]] : index
// TRIANGULAR: %[[QK_ROW:.+]] = arith.addi %[[QK_ROW_BASE]], %[[QK_INTRA_ROW]] : index
// TRIANGULAR: %[[QK_COL_BASE:.+]] = arith.muli %[[QK_MACRO_COL]], %[[QK_C2]] : index
// TRIANGULAR: %[[QK_COL:.+]] = arith.addi %[[QK_COL_BASE]], %[[QK_INTRA_COL]] : index
// TRIANGULAR: %[[QK_ROW_OFFSET:.+]] = arith.muli %[[QK_ROW]], %[[QK_C64]] : index
// TRIANGULAR: %[[QK_COL_OFFSET:.+]] = arith.muli %[[QK_COL]], %[[QK_C64]] : index
// TRIANGULAR-DAG: tensor.extract_slice {{.*}}[%[[QK_B]], %[[QK_ROW_OFFSET]], 0]
// TRIANGULAR-DAG: tensor.extract_slice {{.*}}[%[[QK_B]], %[[QK_COL_OFFSET]], 0]
// TRIANGULAR-DAG: tensor.extract_slice {{.*}}[%[[QK_B]], %[[QK_ROW_OFFSET]], %[[QK_COL_OFFSET]]]
// TRIANGULAR: linalg.generic
// TRIANGULAR: tensor.parallel_insert_slice {{.*}} into {{.*}}[%[[QK_B]], %[[QK_ROW_OFFSET]], %[[QK_COL_OFFSET]]] [1, 64, 64] [1, 1, 1]
// TRIANGULAR: mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]

// BASELINE-LABEL: func.func @qk_t512(
// BASELINE: scf.forall
// BASELINE-SAME: to (96, 512, 512) step (1, 64, 64)
// BASELINE: mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]

// MISSING-BOUNDS: error: 'linalg.generic' op causal triangular workgroup grid requires IREE_METAL_CAUSAL_BWD_BOUNDS=1

func.func @dp_t1024(
    %output_grad: tensor<8x12x1024x64xbf16>,
    %value: tensor<8x12x1024x64xbf16>,
    %init: tensor<8x12x1024x1024xf32>)
    -> tensor<8x12x1024x1024xf32>
    attributes {
      translation_info = #iree_codegen.translation_info<
          pipeline = SPIRVAppleVectorDistributeAttention>} {
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(b, h, m, n, k) -> (b, h, m, k)>,
        affine_map<(b, h, m, n, k) -> (b, h, n, k)>,
        affine_map<(b, h, m, n, k) -> (b, h, m, n)>],
      iterator_types = [
        "parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%output_grad, %value
          : tensor<8x12x1024x64xbf16>, tensor<8x12x1024x64xbf16>)
      outs(%init : tensor<8x12x1024x1024xf32>)
      attrs = {
        iree_codegen.apple_attention_backward_causal_score,
        iree_codegen.apple_attention_backward_causal_score_alignment = 128 : i64,
        iree_codegen.apple_attention_backward_role = "dp_attrs",
        lowering_config = #iree_codegen.lowering_config<
            tile_sizes = [[1, 1, 64, 128, 0]]>} {
    ^bb0(%lhs: bf16, %rhs: bf16, %acc: f32):
      %lhs_f32 = arith.extf %lhs : bf16 to f32
      %rhs_f32 = arith.extf %rhs : bf16 to f32
      %product = arith.mulf %lhs_f32, %rhs_f32 : f32
      %sum = arith.addf %product, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<8x12x1024x1024xf32>
  return %result : tensor<8x12x1024x1024xf32>
}

// TRIANGULAR-LABEL: func.func @dp_t1024(
// TRIANGULAR-DAG: %[[DP_C128:.+]] = arith.constant 128 : index
// TRIANGULAR-DAG: %[[DP_C64:.+]] = arith.constant 64 : index
// TRIANGULAR-DAG: %[[DP_C7:.+]] = arith.constant 7 : index
// TRIANGULAR-DAG: %[[DP_C28:.+]] = arith.constant 28 : index
// TRIANGULAR-DAG: %[[DP_C2:.+]] = arith.constant 2 : index
// TRIANGULAR: scf.forall (%[[DP_B:.+]], %[[DP_H:.+]], %[[DP_Y:.+]], %[[DP_ORD:.+]]) in (8, 12, 1, 72)
// TRIANGULAR: %[[DP_COARSE:.+]] = arith.divui %[[DP_ORD]], %[[DP_C2]] : index
// TRIANGULAR: %[[DP_INTRA:.+]] = arith.remui %[[DP_ORD]], %[[DP_C2]] : index
// TRIANGULAR-COUNT-6: arith.cmpi uge, %[[DP_COARSE]],
// TRIANGULAR: %[[DP_GE28:.+]] = arith.cmpi uge, %[[DP_COARSE]], %[[DP_C28]] : index
// TRIANGULAR: %[[DP_MACRO_ROW:.+]] = arith.select %[[DP_GE28]], %[[DP_C7]], %{{.+}} : index
// TRIANGULAR: %[[DP_MACRO_BASE:.+]] = arith.select %[[DP_GE28]], %[[DP_C28]], %{{.+}} : index
// TRIANGULAR: %[[DP_MACRO_COL:.+]] = arith.subi %[[DP_COARSE]], %[[DP_MACRO_BASE]] : index
// TRIANGULAR: %[[DP_ROW_BASE:.+]] = arith.muli %[[DP_MACRO_ROW]], %[[DP_C2]] : index
// TRIANGULAR: %[[DP_ROW:.+]] = arith.addi %[[DP_ROW_BASE]], %[[DP_INTRA]] : index
// TRIANGULAR: %[[DP_ROW_OFFSET:.+]] = arith.muli %[[DP_ROW]], %[[DP_C64]] : index
// TRIANGULAR: %[[DP_COL_OFFSET:.+]] = arith.muli %[[DP_MACRO_COL]], %[[DP_C128]] : index
// TRIANGULAR-DAG: tensor.extract_slice {{.*}}[%[[DP_B]], %[[DP_H]], %[[DP_ROW_OFFSET]], 0]
// TRIANGULAR-DAG: tensor.extract_slice {{.*}}[%[[DP_B]], %[[DP_H]], %[[DP_COL_OFFSET]], 0]
// TRIANGULAR-DAG: tensor.extract_slice {{.*}}[%[[DP_B]], %[[DP_H]], %[[DP_ROW_OFFSET]], %[[DP_COL_OFFSET]]]
// TRIANGULAR: linalg.generic
// TRIANGULAR: tensor.parallel_insert_slice {{.*}} into {{.*}}[%[[DP_B]], %[[DP_H]], %[[DP_ROW_OFFSET]], %[[DP_COL_OFFSET]]] [1, 1, 64, 128] [1, 1, 1, 1]
// TRIANGULAR: mapping = [#iree_codegen.workgroup_mapping<z:1>, #iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]

// BASELINE-LABEL: func.func @dp_t1024(
// BASELINE: scf.forall
// BASELINE-SAME: to (8, 12, 1024, 1024) step (1, 1, 64, 128)
// BASELINE: mapping = [#iree_codegen.workgroup_mapping<z:1>, #iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]
