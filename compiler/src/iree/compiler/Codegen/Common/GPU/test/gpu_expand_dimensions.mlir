// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-expand-dimensions))" | FileCheck %s --check-prefixes=CHECK,EARLY
// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-expand-dimensions{expand-apple-physical-fragments=true}))" | FileCheck %s --check-prefixes=CHECK,LATE
// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-expand-dimensions{expand-apple-physical-fragments=true}),func.func(iree-codegen-gpu-expand-dimensions{expand-apple-physical-fragments=true}))" | FileCheck %s --check-prefixes=CHECK,LATE

func.func @expand_matvec(%a: tensor<4x16384xf16>, %b: tensor<1x16384xf16>) -> tensor<4x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x1xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x1xf32>) -> tensor<4x1xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%a, %b : tensor<4x16384xf16>, tensor<1x16384xf16>)
    outs(%fill : tensor<4x1xf32>)
    attrs = {
      lowering_config = #iree_gpu.lowering_config<{
      expand_dims = #iree_gpu.expand_dims<[[0], [1], [2, 3]], output_shape = [?, ?, ?, 8]>,
      lane_basis = [[1, 1, 64, 1], [0, 1, 2, 3]],
      partial_reduction = [0, 0, 64, 0],
      subgroup_basis = [[1, 1, 1, 1], [0, 1, 2, 3]],
      thread = [0, 0, 1, 8],
      workgroup = [4, 1, 0, 0]}>} {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %0 = arith.extf %in : f16 to f32
    %1 = arith.extf %in_0 : f16 to f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// CHECK-LABEL: func.func @expand_matvec
// CHECK: %[[A_EXPAND:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2]] output_shape [4, 2048, 8] : tensor<4x16384xf16> into tensor<4x2048x8xf16>
// CHECK: %[[B_EXPAND:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2]] output_shape [1, 2048, 8] : tensor<1x16384xf16> into tensor<1x2048x8xf16>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]] : tensor<4x2048x8xf16>, tensor<1x2048x8xf16>)

// -----

func.func @expand_multiple_dims(%a: tensor<4x16384xf16>, %b: tensor<4x16384xf16>) -> tensor<4x16384xf16> {
  %empty = tensor.empty() : tensor<4x16384xf16>
  %result = linalg.add {
    lowering_config = #iree_gpu.lowering_config<{
      expand_dims = #iree_gpu.expand_dims<[[0], [1, 2, 3]], output_shape = [?, ?, 2, 4]>
    }>}
    ins(%a, %b : tensor<4x16384xf16>, tensor<4x16384xf16>) outs(%empty : tensor<4x16384xf16>) -> tensor<4x16384xf16>
  return %result : tensor<4x16384xf16>
}

// CHECK-LABEL: func.func @expand_multiple_dims
// CHECK: %[[A_EXPAND:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2, 3]] output_shape [4, 2048, 2, 4] : tensor<4x16384xf16> into tensor<4x2048x2x4xf16>
// CHECK: %[[B_EXPAND:.*]] = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2, 3]] output_shape [4, 2048, 2, 4] : tensor<4x16384xf16> into tensor<4x2048x2x4xf16>
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]] : tensor<4x2048x2x4xf16>, tensor<4x2048x2x4xf16>)

// -----

// Verify that dynamic dimensions are gracefully handled (no expansion occurs).
func.func @no_expand_dynamic_dims(%a: tensor<4x?xf16>, %b: tensor<4x?xf16>) -> tensor<4x128xf16> {
  %empty = tensor.empty() : tensor<4x128xf16>
  %result = linalg.add {
    lowering_config = #iree_gpu.lowering_config<{
      expand_dims = #iree_gpu.expand_dims<[[0], [1, 2]], output_shape = [?, ?, 8]>
    }>}
    ins(%a, %b : tensor<4x?xf16>, tensor<4x?xf16>) outs(%empty : tensor<4x128xf16>) -> tensor<4x128xf16>
  return %result : tensor<4x128xf16>
}

// CHECK-LABEL: func.func @no_expand_dynamic_dim
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.add
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<4x?xf16>, tensor<4x?xf16>)

// -----

// Verify that non-divisible dimensions are gracefully handled (no expansion occurs).
func.func @no_expand_not_divisible(%a: tensor<4x127xf16>, %b: tensor<4x127xf16>) -> tensor<4x127xf16> {
  %empty = tensor.empty() : tensor<4x127xf16>
  %result = linalg.add {
    lowering_config = #iree_gpu.lowering_config<{
      expand_dims = #iree_gpu.expand_dims<[[0], [1, 2]], output_shape = [?, ?, 8]>
    }>}
    ins(%a, %b : tensor<4x127xf16>, tensor<4x127xf16>) outs(%empty : tensor<4x127xf16>) -> tensor<4x127xf16>
  return %result : tensor<4x127xf16>
}

// CHECK-LABEL: func.func @no_expand_not_divisible
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.add
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<4x127xf16>, tensor<4x127xf16>)

// -----

#physical_maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (n, m)>
]
#physical_config = #iree_gpu.lowering_config<{
  workgroup = [32, 48, 0],
  reduction = [0, 0, 64],
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_F16:
      apple_physical_fragment_layout = true>,
  subgroup_basis = [[2, 3, 1], [0, 1, 2]]
}>

// These model the direct contractions seen by causal backward shortening.
// The ordinary early pass must leave their logical rank intact; only the late
// post-tiling invocation is allowed to materialize the physical fragment axes.
func.func @causal_dq_physical_expands_late(
    %lhs: tensor<32x64xf16>, %rhs: tensor<48x64xf16>,
    %acc: tensor<48x32xf32>) -> tensor<48x32xf32> {
  %result = linalg.generic {
      indexing_maps = #physical_maps,
      iterator_types = ["parallel", "parallel", "reduction"],
      lowering_config = #physical_config,
      iree_codegen.apple_attention_backward_causal,
      iree_codegen.apple_attention_backward_role = "dq_attrs"
    } ins(%lhs, %rhs : tensor<32x64xf16>, tensor<48x64xf16>)
      outs(%acc : tensor<48x32xf32>) {
    ^bb0(%l: f16, %r: f16, %old: f32):
      %lf = arith.extf %l : f16 to f32
      %rf = arith.extf %r : f16 to f32
      %mul = arith.mulf %lf, %rf : f32
      %sum = arith.addf %old, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<48x32xf32>
  return %result : tensor<48x32xf32>
}

// EARLY-LABEL: func.func @causal_dq_physical_expands_late
// EARLY-NOT: tensor.expand_shape
// EARLY: linalg.generic
// LATE-LABEL: func.func @causal_dq_physical_expands_late
// LATE-DAG: tensor.expand_shape {{.*}} output_shape [2, 2, 2, 4, 4, 2, 2, 4] : tensor<32x64xf16> into tensor<2x2x2x4x4x2x2x4xf16>
// LATE-DAG: tensor.expand_shape {{.*}} output_shape [3, 2, 2, 4, 4, 2, 2, 4] : tensor<48x64xf16> into tensor<3x2x2x4x4x2x2x4xf16>
// LATE-DAG: tensor.expand_shape {{.*}} output_shape [3, 2, 2, 4, 2, 2, 2, 4] : tensor<48x32xf32> into tensor<3x2x2x4x2x2x2x4xf32>
// LATE: linalg.generic
// LATE-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
// LATE-DAG: apple_physical_fragment_expansion_materialized
// LATE-DAG: reduction = [0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 2, 4]
// LATE-DAG: subgroup_basis = {{\[\[}}2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11{{\]\]}}
// LATE-DAG: workgroup = [2, 2, 2, 4, 3, 2, 2, 4, 0, 0, 0, 0]

func.func @causal_dk_physical_expands_late(
    %lhs: tensor<32x64xf16>, %rhs: tensor<48x64xf16>,
    %acc: tensor<48x32xf32>) -> tensor<48x32xf32> {
  %result = linalg.generic {
      indexing_maps = #physical_maps,
      iterator_types = ["parallel", "parallel", "reduction"],
      lowering_config = #physical_config,
      iree_codegen.apple_attention_backward_causal,
      iree_codegen.apple_attention_backward_role = "dk_attrs"
    } ins(%lhs, %rhs : tensor<32x64xf16>, tensor<48x64xf16>)
      outs(%acc : tensor<48x32xf32>) {
    ^bb0(%l: f16, %r: f16, %old: f32):
      %lf = arith.extf %l : f16 to f32
      %rf = arith.extf %r : f16 to f32
      %mul = arith.mulf %lf, %rf : f32
      %sum = arith.addf %old, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<48x32xf32>
  return %result : tensor<48x32xf32>
}

// EARLY-LABEL: func.func @causal_dk_physical_expands_late
// EARLY-NOT: tensor.expand_shape
// EARLY: linalg.generic
// LATE-LABEL: func.func @causal_dk_physical_expands_late
// LATE: tensor.expand_shape
// LATE: linalg.generic

func.func @causal_dv_physical_expands_late(
    %lhs: tensor<32x64xf16>, %rhs: tensor<48x64xf16>,
    %acc: tensor<48x32xf32>) -> tensor<48x32xf32> {
  %result = linalg.generic {
      indexing_maps = #physical_maps,
      iterator_types = ["parallel", "parallel", "reduction"],
      lowering_config = #physical_config,
      iree_codegen.apple_attention_backward_causal,
      iree_codegen.apple_attention_backward_role = "dv_attrs"
    } ins(%lhs, %rhs : tensor<32x64xf16>, tensor<48x64xf16>)
      outs(%acc : tensor<48x32xf32>) {
    ^bb0(%l: f16, %r: f16, %old: f32):
      %lf = arith.extf %l : f16 to f32
      %rf = arith.extf %r : f16 to f32
      %mul = arith.mulf %lf, %rf : f32
      %sum = arith.addf %old, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<48x32xf32>
  return %result : tensor<48x32xf32>
}

// EARLY-LABEL: func.func @causal_dv_physical_expands_late
// EARLY-NOT: tensor.expand_shape
// EARLY: linalg.generic
// LATE-LABEL: func.func @causal_dv_physical_expands_late
// LATE: tensor.expand_shape
// LATE: linalg.generic

// Model the multi-iteration reduction loop used by production dQ/dK/dV. The
// expanded accumulator must be carried through the loop directly; leaving an
// expand/collapse bridge around the contraction prevents vector iter-arg
// forwarding and materializes an otherwise unnecessary f32 workgroup buffer.
func.func @physical_serial_loop_accumulator(
    %acc: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %lhs = tensor.empty() : tensor<64x128xf16>
  %rhs = tensor.empty() : tensor<64x128xf16>
  %result = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%iter = %acc) -> tensor<64x64xf32> {
    %next = linalg.generic {
        indexing_maps = #physical_maps,
        iterator_types = ["parallel", "parallel", "reduction"],
        lowering_config = #physical_config
      } ins(%lhs, %rhs : tensor<64x128xf16>, tensor<64x128xf16>)
        outs(%iter : tensor<64x64xf32>) {
      ^bb0(%l: f16, %r: f16, %old: f32):
        %lf = arith.extf %l : f16 to f32
        %rf = arith.extf %r : f16 to f32
        %mul = arith.mulf %lf, %rf : f32
        %sum = arith.addf %old, %mul : f32
        linalg.yield %sum : f32
    } -> tensor<64x64xf32>
    scf.yield %next : tensor<64x64xf32>
  }
  return %result : tensor<64x64xf32>
}

// EARLY-LABEL: func.func @physical_serial_loop_accumulator
// EARLY-NOT: tensor.expand_shape
// EARLY: %[[EARLY_LOOP:.+]] = scf.for
// EARLY-SAME: iter_args(%[[EARLY_ITER:.+]] = %arg0) -> (tensor<64x64xf32>)
// EARLY: linalg.generic
// EARLY-SAME: outs(%[[EARLY_ITER]] : tensor<64x64xf32>)
// EARLY: scf.yield
// EARLY-NOT: tensor.collapse_shape
// EARLY: return %[[EARLY_LOOP]] : tensor<64x64xf32>

// LATE-LABEL: func.func @physical_serial_loop_accumulator
// LATE: %[[EXPANDED_ACC:.+]] = tensor.expand_shape %arg0 {{\[}}[0, 1, 2, 3], [4, 5, 6, 7]] output_shape [4, 2, 2, 4, 4, 2, 2, 4] : tensor<64x64xf32> into tensor<4x2x2x4x4x2x2x4xf32>
// LATE: %[[LATE_LOOP:.+]] = scf.for
// LATE-SAME: iter_args(%[[LATE_ITER:.+]] = %[[EXPANDED_ACC]]) -> (tensor<4x2x2x4x4x2x2x4xf32>)
// LATE-NOT: tensor.expand_shape
// LATE-NOT: tensor.collapse_shape
// LATE: linalg.generic
// LATE-SAME: outs(%[[LATE_ITER]] : tensor<4x2x2x4x4x2x2x4xf32>)
// LATE: scf.yield
// LATE-NOT: tensor.expand_shape
// LATE-NOT: tensor.collapse_shape
// LATE: %[[COLLAPSED_RESULT:.+]] = tensor.collapse_shape %[[LATE_LOOP]] {{\[}}[0, 1, 2, 3], [4, 5, 6, 7]] : tensor<4x2x2x4x4x2x2x4xf32> into tensor<64x64xf32>
// LATE: return %[[COLLAPSED_RESULT]] : tensor<64x64xf32>

// -----

// Model the full-store QK dispatch shape that exposed a collapse_shape at the
// interface boundary. The physical expansion must propagate through the BF16
// pointwise epilogue and retarget the full interface store to the expanded
// rank instead of leaving a rank-collapsing bridge behind.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

#qk_physical_maps = [
  affine_map<(b, m, n, k) -> (b, m, k)>,
  affine_map<(b, m, n, k) -> (b, n, k)>,
  affine_map<(b, m, n, k) -> (b, m, n)>
]

#qk_physical_config = #iree_gpu.lowering_config<{
  workgroup = [1, 64, 64, 0],
  reduction = [0, 0, 0, 64],
  mma_kind = #iree_gpu.mma_layout<
      APPLE_SIMDGROUP_F32_16x16x16_BF16:
      apple_physical_fragment_layout = true>,
  subgroup_basis = [[1, 1, 4, 1], [0, 1, 2, 3]]
}>

func.func @physical_qk_t64_full_store(
    %lhs: tensor<1x64x64xbf16>, %rhs: tensor<1x64x64xbf16>,
    %acc: tensor<1x64x64xf32>) {
  %c0 = arith.constant 0 : index
  %target = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
      alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64x64xbf16>>
  %scores = linalg.generic {
      indexing_maps = #qk_physical_maps,
      iterator_types = ["parallel", "parallel", "parallel", "reduction"],
      lowering_config = #qk_physical_config
    } ins(%lhs, %rhs : tensor<1x64x64xbf16>, tensor<1x64x64xbf16>)
      outs(%acc : tensor<1x64x64xf32>) {
    ^bb0(%l: bf16, %r: bf16, %old: f32):
      %lf = arith.extf %l : bf16 to f32
      %rf = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lf, %rf : f32
      %sum = arith.addf %old, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<1x64x64xf32>
  %empty = tensor.empty() : tensor<1x64x64xbf16>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(b, m, n) -> (b, m, n)>,
        affine_map<(b, m, n) -> (b, m, n)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%scores : tensor<1x64x64xf32>)
      outs(%empty : tensor<1x64x64xbf16>) {
    ^bb0(%score: f32, %out: bf16):
      %truncated = arith.truncf %score : f32 to bf16
      linalg.yield %truncated : bf16
  } -> tensor<1x64x64xbf16>
  iree_tensor_ext.dispatch.tensor.store %result, %target,
      offsets = [0, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1]
      : tensor<1x64x64xbf16> ->
        !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64x64xbf16>>
  return
}

// EARLY-LABEL: func.func @physical_qk_t64_full_store
// EARLY-NOT: tensor.expand_shape
// EARLY-NOT: tensor.collapse_shape
// EARLY: iree_tensor_ext.dispatch.tensor.store
// EARLY-SAME: tensor<1x64x64xbf16> ->
// EARLY-SAME: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64x64xbf16>>
// EARLY: return

// LATE-LABEL: func.func @physical_qk_t64_full_store
// LATE-NOT: tensor.collapse_shape
// LATE: %[[EXPANDED_TARGET:.+]] = hal.interface.binding.subspan
// LATE-SAME: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4x2x2x4x4x2x2x4xbf16>>
// LATE: iree_tensor_ext.dispatch.tensor.store %{{.*}}, %[[EXPANDED_TARGET]], offsets = [0, 0, 0, 0, 0, 0, 0, 0, 0], sizes = [1, 4, 2, 2, 4, 4, 2, 2, 4], strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
// LATE-SAME: tensor<1x4x2x2x4x4x2x2x4xbf16> ->
// LATE-SAME: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4x2x2x4x4x2x2x4xbf16>>
// LATE-NOT: tensor.collapse_shape
// LATE: return

// A dynamic unsplit batch dimension remains dynamic while the static M/N/K
// dimensions acquire their physical fragment factors.
func.func @physical_dynamic_batch(
    %lhs: tensor<?x64x64xbf16>, %rhs: tensor<?x64x64xbf16>,
    %acc: tensor<?x64x64xf32>) -> tensor<?x64x64xf32> {
  %result = linalg.generic {
      indexing_maps = #qk_physical_maps,
      iterator_types = ["parallel", "parallel", "parallel", "reduction"],
      lowering_config = #qk_physical_config
    } ins(%lhs, %rhs : tensor<?x64x64xbf16>, tensor<?x64x64xbf16>)
      outs(%acc : tensor<?x64x64xf32>) {
    ^bb0(%l: bf16, %r: bf16, %old: f32):
      %lf = arith.extf %l : bf16 to f32
      %rf = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lf, %rf : f32
      %sum = arith.addf %old, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<?x64x64xf32>
  return %result : tensor<?x64x64xf32>
}

// EARLY-LABEL: func.func @physical_dynamic_batch
// EARLY-NOT: tensor.expand_shape
// EARLY: linalg.generic
// EARLY-SAME: tensor<?x64x64xbf16>

// LATE-LABEL: func.func @physical_dynamic_batch
// LATE-DAG: tensor.expand_shape {{.*}} output_shape [%{{.*}}, 4, 2, 2, 4, 4, 2, 2, 4] : tensor<?x64x64xbf16> into tensor<?x4x2x2x4x4x2x2x4xbf16>
// LATE-DAG: tensor.expand_shape {{.*}} output_shape [%{{.*}}, 4, 2, 2, 4, 4, 2, 2, 4] : tensor<?x64x64xf32> into tensor<?x4x2x2x4x4x2x2x4xf32>
// LATE: linalg.generic
// LATE-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
// LATE-SAME: tensor<?x4x2x2x4x4x2x2x4xbf16>
