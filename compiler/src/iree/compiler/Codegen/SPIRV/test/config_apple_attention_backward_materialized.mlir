// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_FRAGMENTS \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=ON
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   -u IREE_METAL_APPLE_PHYSICAL_FRAGMENTS \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=UNSET
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=0 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=ZERO
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=PHYSICAL
// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   IREE_METAL_APPLE_PHYSICAL_FRAGMENTS=1 \
// RUN:   iree-opt --iree-gpu-test-target=volta@vulkan \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=NONAPPLE
// RUN: env IREE_METAL_DISABLE_NATIVE_ATTENTION=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=OFF

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>

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

// PHYSICAL-LABEL: func.func @untagged(
// PHYSICAL-NOT: apple_physical_fragment_layout
// PHYSICAL-LABEL: func.func @tagged(
// PHYSICAL: apple_physical_fragment_layout = true
// PHYSICAL-LABEL: func.func @tagged_qk(
// PHYSICAL: apple_physical_fragment_layout = true
// PHYSICAL-LABEL: func.func @tagged_dp(
// PHYSICAL: apple_physical_fragment_layout = true
// PHYSICAL-LABEL: func.func @tagged_dq(
// PHYSICAL: apple_physical_fragment_layout = true
// PHYSICAL-LABEL: func.func @tagged_dv(
// PHYSICAL: apple_physical_fragment_layout = true
// PHYSICAL-LABEL: func.func @tagged_unknown(
// PHYSICAL-NOT: apple_physical_fragment_layout
// PHYSICAL-LABEL: func.func @tagged_unsupported(
// PHYSICAL-NOT: apple_physical_fragment_layout

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
