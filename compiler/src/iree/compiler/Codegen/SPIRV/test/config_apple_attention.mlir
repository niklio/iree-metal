// RUN: env IREE_METAL_ATTN_VDIST=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=APPLE
// RUN: env IREE_METAL_ATTN_VDIST=1 \
// RUN:   iree-opt --iree-gpu-test-target=volta@vulkan \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=NONAPPLE
// RUN: env IREE_METAL_ATTN_VDIST=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s -o %t.configured
// RUN: env -u IREE_METAL_ATTN_VDIST -u IREE_METAL_COOP_ATTENTION_WIP \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-spirv-lower-executable-target-pass))' \
// RUN:   %t.configured -o /dev/null

func.func @attention_bf16(
    %query: tensor<96x512x64xbf16>,
    %key: tensor<96x512x64xbf16>,
    %value: tensor<96x512x64xbf16>) -> tensor<96x512x64xbf16> {
  %scale = arith.constant 1.250000e-01 : bf16
  %output = tensor.empty() : tensor<96x512x64xbf16>
  %result = iree_linalg_ext.attention {
      indexing_maps = [
        affine_map<(b, m, k1, k2, n) -> (b, m, k1)>,
        affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>,
        affine_map<(b, m, k1, k2, n) -> (b, k2, n)>,
        affine_map<(b, m, k1, k2, n) -> ()>,
        affine_map<(b, m, k1, k2, n) -> (b, m, n)>
      ]}
      ins(%query, %key, %value, %scale :
          tensor<96x512x64xbf16>, tensor<96x512x64xbf16>,
          tensor<96x512x64xbf16>, bf16)
      outs(%output : tensor<96x512x64xbf16>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<96x512x64xbf16>
  return %result : tensor<96x512x64xbf16>
}

// APPLE-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVAppleVectorDistributeAttention
// APPLE-LABEL: func.func @attention_bf16(
// APPLE-SAME: translation_info = #[[TRANSLATION]]

// NONAPPLE-NOT: SPIRVAppleVectorDistributeAttention
// NONAPPLE-LABEL: func.func @attention_bf16(
// NONAPPLE-NOT: SPIRVAppleVectorDistributeAttention
