// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_FRAGMENTS \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_SCORE_WG64 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=ON
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_FRAGMENTS \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_SCORE_WG64 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=UNSET
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_SCORE_WG64 \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=0 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=ZERO
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_SCORE_WG64 \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=PHYSICAL
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_SCORE_WG64 \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=1 \
// RUN:   IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=COMPACT
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_SCORE_WG64 \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=1 \
// RUN:   IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM=invalid \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=INVALID
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=1 \
// RUN:   IREE_METAL_APPLE_PHYSICAL_SCORE_WG64=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=SCORE-WG64
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=1 \
// RUN:   IREE_METAL_APPLE_PHYSICAL_SCORE_WG64=0 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=PHYSICAL
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=1 \
// RUN:   IREE_METAL_APPLE_PHYSICAL_SCORE_WG64=invalid \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=SCORE-INVALID
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_SCORE_WG64 \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=1 \
// RUN:   iree-opt --iree-gpu-test-target=volta@vulkan \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=NONAPPLE
// RUN: env -u IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_SCORE_WG64 \
// RUN:   IREE_METAL_DISABLE_NATIVE_ATTENTION=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=OFF

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>
#score_lhs = affine_map<(b, m, n, k) -> (b, m, k)>
#score_rhs = affine_map<(b, m, n, k) -> (b, n, k)>
#score_acc = affine_map<(b, m, n, k) -> (b, m, n)>

func.func @untagged(
    %lhs: tensor<16x16xbf16>, %rhs: tensor<16x16xbf16>,
    %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %result = linalg.generic {
      indexing_maps = [#lhs, #rhs, #acc],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<16x16xbf16>, tensor<16x16xbf16>)
      outs(%acc : tensor<16x16xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}

func.func @tagged(
    %lhs: tensor<16x16xbf16>, %rhs: tensor<16x16xbf16>,
    %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %result = linalg.generic {
      iree_codegen.apple_attention_backward_role = "dk_attrs",
      indexing_maps = [#lhs, #rhs, #acc],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<16x16xbf16>, tensor<16x16xbf16>)
      outs(%acc : tensor<16x16xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}

func.func @tagged_qk(
    %lhs: tensor<16x16xbf16>, %rhs: tensor<16x16xbf16>,
    %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %result = linalg.generic {
      iree_codegen.apple_attention_backward_role = "qk_attrs",
      indexing_maps = [#lhs, #rhs, #acc],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<16x16xbf16>, tensor<16x16xbf16>)
      outs(%acc : tensor<16x16xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}

func.func @tagged_dp(
    %lhs: tensor<16x16xbf16>, %rhs: tensor<16x16xbf16>,
    %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %result = linalg.generic {
      iree_codegen.apple_attention_backward_role = "dp_attrs",
      indexing_maps = [#lhs, #rhs, #acc],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<16x16xbf16>, tensor<16x16xbf16>)
      outs(%acc : tensor<16x16xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}

func.func @tagged_qk_score_32x32x64(
    %lhs: tensor<1x32x64xbf16>, %rhs: tensor<1x32x64xbf16>,
    %acc: tensor<1x32x32xf32>) -> tensor<1x32x32xf32> {
  %result = linalg.generic {
      iree_codegen.apple_attention_backward_role = "qk_attrs",
      indexing_maps = [#score_lhs, #score_rhs, #score_acc],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<1x32x64xbf16>, tensor<1x32x64xbf16>)
      outs(%acc : tensor<1x32x32xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<1x32x32xf32>
  return %result : tensor<1x32x32xf32>
}

func.func @tagged_dp_score_32x32x64(
    %lhs: tensor<1x32x64xbf16>, %rhs: tensor<1x32x64xbf16>,
    %acc: tensor<1x32x32xf32>) -> tensor<1x32x32xf32> {
  %result = linalg.generic {
      iree_codegen.apple_attention_backward_role = "dp_attrs",
      indexing_maps = [#score_lhs, #score_rhs, #score_acc],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<1x32x64xbf16>, tensor<1x32x64xbf16>)
      outs(%acc : tensor<1x32x32xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<1x32x32xf32>
  return %result : tensor<1x32x32xf32>
}

func.func @tagged_dq_score_control_32x32x64(
    %lhs: tensor<1x32x64xbf16>, %rhs: tensor<1x32x64xbf16>,
    %acc: tensor<1x32x32xf32>) -> tensor<1x32x32xf32> {
  %result = linalg.generic {
      iree_codegen.apple_attention_backward_role = "dq_attrs",
      indexing_maps = [#score_lhs, #score_rhs, #score_acc],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<1x32x64xbf16>, tensor<1x32x64xbf16>)
      outs(%acc : tensor<1x32x32xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<1x32x32xf32>
  return %result : tensor<1x32x32xf32>
}

func.func @tagged_dq(
    %lhs: tensor<16x16xbf16>, %rhs: tensor<16x16xbf16>,
    %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %result = linalg.generic {
      iree_codegen.apple_attention_backward_role = "dq_attrs",
      indexing_maps = [#lhs, #rhs, #acc],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<16x16xbf16>, tensor<16x16xbf16>)
      outs(%acc : tensor<16x16xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}

func.func @tagged_dv(
    %lhs: tensor<16x16xbf16>, %rhs: tensor<16x16xbf16>,
    %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %result = linalg.generic {
      iree_codegen.apple_attention_backward_role = "dv_attrs",
      indexing_maps = [#lhs, #rhs, #acc],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<16x16xbf16>, tensor<16x16xbf16>)
      outs(%acc : tensor<16x16xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}

func.func @tagged_unknown(
    %lhs: tensor<16x16xbf16>, %rhs: tensor<16x16xbf16>,
    %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %result = linalg.generic {
      iree_codegen.apple_attention_backward_role = "unknown",
      indexing_maps = [#lhs, #rhs, #acc],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<16x16xbf16>, tensor<16x16xbf16>)
      outs(%acc : tensor<16x16xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}

func.func @tagged_unsupported(
    %lhs: tensor<7x7xbf16>, %rhs: tensor<7x7xbf16>,
    %acc: tensor<7x7xf32>) -> tensor<7x7xf32> {
  %result = linalg.generic {
      iree_codegen.apple_attention_backward_role = "dk_attrs",
      indexing_maps = [#lhs, #rhs, #acc],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<7x7xbf16>, tensor<7x7xbf16>)
      outs(%acc : tensor<7x7xf32>) {
    ^bb0(%l: bf16, %r: bf16, %out: f32):
      %lext = arith.extf %l : bf16 to f32
      %rext = arith.extf %r : bf16 to f32
      %mul = arith.mulf %lext, %rext : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<7x7xf32>
  return %result : tensor<7x7xf32>
}

// ON-DAG: #[[APPLE:.+]] = #iree_codegen.translation_info<pipeline = SPIRVAppleVectorDistributeAttention
// ON-LABEL: func.func @untagged(
// ON-NOT: SPIRVAppleVectorDistributeAttention
// ON-NOT: APPLE_SIMDGROUP
// ON-LABEL: func.func @tagged(
// ON-SAME: translation_info = #[[APPLE]]
// ON: mma_kind = #iree_gpu.mma_layout<APPLE_SIMDGROUP_F32_16x16x16_BF16>
// ON-LABEL: func.func @tagged_unsupported(
// ON-NOT: SPIRVAppleVectorDistributeAttention
// ON-NOT: apple_physical_fragment_layout

// UNSET-LABEL: func.func @untagged(
// UNSET-NOT: apple_physical_fragment_layout

// ZERO-NOT: apple_physical_fragment_layout

// PHYSICAL-DAG: #[[SCORE_WG128:.+]] = #iree_codegen.translation_info<pipeline = SPIRVAppleVectorDistributeAttention workgroup_size = [128, 1, 1]
// PHYSICAL-NOT: no_reduce_shared_memory_bank_conflicts = true
// PHYSICAL-LABEL: func.func @untagged(
// PHYSICAL-NOT: apple_physical_fragment_layout
// PHYSICAL-LABEL: func.func @tagged(
// PHYSICAL: apple_physical_fragment_layout = true
// PHYSICAL-LABEL: func.func @tagged_qk(
// PHYSICAL: apple_physical_fragment_layout = true
// PHYSICAL-LABEL: func.func @tagged_dp(
// PHYSICAL: apple_physical_fragment_layout = true
// PHYSICAL-LABEL: func.func @tagged_qk_score_32x32x64(
// PHYSICAL-SAME: translation_info = #[[SCORE_WG128]]
// PHYSICAL: reduction = [0, 0, 0, 64]
// PHYSICAL-SAME: subgroup_basis = {{\[}}[1, 2, 2, 1], [0, 1, 2, 3]{{\]}}
// PHYSICAL-SAME: workgroup = [1, 32, 32, 0]
// PHYSICAL-LABEL: func.func @tagged_dp_score_32x32x64(
// PHYSICAL-SAME: translation_info = #[[SCORE_WG128]]
// PHYSICAL: reduction = [0, 0, 0, 64]
// PHYSICAL-SAME: subgroup_basis = {{\[}}[1, 2, 2, 1], [0, 1, 2, 3]{{\]}}
// PHYSICAL-SAME: workgroup = [1, 32, 32, 0]
// PHYSICAL-LABEL: func.func @tagged_dq_score_control_32x32x64(
// PHYSICAL-SAME: translation_info = #[[SCORE_WG128]]
// PHYSICAL: subgroup_basis = {{\[}}[1, 2, 2, 1], [0, 1, 2, 3]{{\]}}
// PHYSICAL-LABEL: func.func @tagged_dq(
// PHYSICAL: apple_physical_fragment_layout = true
// PHYSICAL-LABEL: func.func @tagged_dv(
// PHYSICAL: apple_physical_fragment_layout = true
// PHYSICAL-LABEL: func.func @tagged_unknown(
// PHYSICAL-NOT: apple_physical_fragment_layout
// PHYSICAL-LABEL: func.func @tagged_unsupported(
// PHYSICAL-NOT: apple_physical_fragment_layout
// PHYSICAL-NOT: no_reduce_shared_memory_bank_conflicts = true

// COMPACT-DAG: #[[COMPACT_TRANSLATION:.+]] = #iree_codegen.translation_info<{{.*}}gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = true
// COMPACT-DAG: #[[PADDED_TRANSLATION:.+]] = #iree_codegen.translation_info<{{.*}}gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false
// COMPACT-LABEL: func.func @untagged(
// COMPACT-NOT: apple_physical_fragment_layout
// COMPACT-LABEL: func.func @tagged(
// COMPACT-SAME: translation_info = #[[COMPACT_TRANSLATION]]
// COMPACT: apple_physical_fragment_layout = true
// COMPACT-LABEL: func.func @tagged_qk(
// COMPACT-SAME: translation_info = #[[COMPACT_TRANSLATION]]
// COMPACT-LABEL: func.func @tagged_dp(
// COMPACT-SAME: translation_info = #[[COMPACT_TRANSLATION]]
// COMPACT-LABEL: func.func @tagged_dq(
// COMPACT-SAME: translation_info = #[[COMPACT_TRANSLATION]]
// COMPACT-LABEL: func.func @tagged_dv(
// COMPACT-SAME: translation_info = #[[COMPACT_TRANSLATION]]
// COMPACT-LABEL: func.func @tagged_unknown(
// COMPACT-SAME: translation_info = #[[PADDED_TRANSLATION]]
// COMPACT-NOT: apple_physical_fragment_layout
// COMPACT-LABEL: func.func @tagged_unsupported(
// COMPACT-NOT: apple_physical_fragment_layout

// INVALID-NOT: no_reduce_shared_memory_bank_conflicts = true

// SCORE-WG64-DAG: #[[SCORE_WG64_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVAppleVectorDistributeAttention workgroup_size = [64, 1, 1]
// SCORE-WG64-DAG: #[[SCORE_CONTROL_WG128_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVAppleVectorDistributeAttention workgroup_size = [128, 1, 1]
// SCORE-WG64-LABEL: func.func @tagged_qk_score_32x32x64(
// SCORE-WG64-SAME: translation_info = #[[SCORE_WG64_TRANSLATION]]
// SCORE-WG64: reduction = [0, 0, 0, 64]
// SCORE-WG64-SAME: subgroup_basis = {{\[}}[1, 1, 2, 1], [0, 1, 2, 3]{{\]}}
// SCORE-WG64-SAME: workgroup = [1, 32, 32, 0]
// SCORE-WG64-LABEL: func.func @tagged_dp_score_32x32x64(
// SCORE-WG64-SAME: translation_info = #[[SCORE_WG64_TRANSLATION]]
// SCORE-WG64: reduction = [0, 0, 0, 64]
// SCORE-WG64-SAME: subgroup_basis = {{\[}}[1, 1, 2, 1], [0, 1, 2, 3]{{\]}}
// SCORE-WG64-SAME: workgroup = [1, 32, 32, 0]
// SCORE-WG64-LABEL: func.func @tagged_dq_score_control_32x32x64(
// SCORE-WG64-SAME: translation_info = #[[SCORE_CONTROL_WG128_TRANSLATION]]
// SCORE-WG64: subgroup_basis = {{\[}}[1, 2, 2, 1], [0, 1, 2, 3]{{\]}}
// SCORE-WG64-SAME: workgroup = [1, 32, 32, 0]

// SCORE-INVALID-DAG: #[[SCORE_INVALID_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVAppleVectorDistributeAttention workgroup_size = [128, 1, 1]
// SCORE-INVALID-LABEL: func.func @tagged_qk_score_32x32x64(
// SCORE-INVALID-SAME: translation_info = #[[SCORE_INVALID_TRANSLATION]]
// SCORE-INVALID: subgroup_basis = {{\[}}[1, 2, 2, 1], [0, 1, 2, 3]{{\]}}
// SCORE-INVALID-LABEL: func.func @tagged_dp_score_32x32x64(
// SCORE-INVALID-SAME: translation_info = #[[SCORE_INVALID_TRANSLATION]]
// SCORE-INVALID: subgroup_basis = {{\[}}[1, 2, 2, 1], [0, 1, 2, 3]{{\]}}

// NONAPPLE-NOT: apple_physical_fragment_layout

// OFF-DAG: #[[LEGACY:.+]] = #iree_codegen.translation_info<pipeline = SPIRVCooperativeMatrixVectorize
// OFF-NOT: APPLE_SIMDGROUP
// OFF-LABEL: func.func @untagged(
// OFF-NOT: SPIRVAppleVectorDistributeAttention
// OFF-NOT: APPLE_SIMDGROUP
// OFF-LABEL: func.func @tagged(
// OFF-SAME: translation_info = #[[LEGACY]]
// OFF-NOT: SPIRVAppleVectorDistributeAttention
// OFF-NOT: APPLE_SIMDGROUP
// OFF-LABEL: func.func @tagged_unsupported(
// OFF-NOT: SPIRVAppleVectorDistributeAttention
// OFF-NOT: APPLE_SIMDGROUP
