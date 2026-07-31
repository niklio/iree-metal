// RUN: env -u IREE_METAL_DISABLE_NATIVE_ATTENTION \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass,func.func(iree-spirv-lower-executable-target-pass))' \
// RUN:   %s -o /dev/null

#query = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#key = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#value = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#scale = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (m, k2)>
#output = affine_map<(b, m, k1, k2, n) -> (b, m, n)>

func.func @attention_bf16_all_false_mask(
    %query: tensor<96x512x64xbf16>,
    %key: tensor<96x512x64xbf16>,
    %value: tensor<96x512x64xbf16>) -> tensor<96x512x64xbf16> {
  %false = arith.constant false
  %mask_empty = tensor.empty() : tensor<512x512xi1>
  %all_false = linalg.fill
      ins(%false : i1)
      outs(%mask_empty : tensor<512x512xi1>) -> tensor<512x512xi1>
  %scale_value = arith.constant 1.250000e-01 : bf16
  %empty = tensor.empty() : tensor<96x512x64xbf16>
  %result = iree_linalg_ext.attention {
      indexing_maps = [#query, #key, #value, #scale, #mask, #output]}
      ins(%query, %key, %value, %scale_value, %all_false :
          tensor<96x512x64xbf16>, tensor<96x512x64xbf16>,
          tensor<96x512x64xbf16>, bf16, tensor<512x512xi1>)
      outs(%empty : tensor<96x512x64xbf16>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<96x512x64xbf16>
  return %result : tensor<96x512x64xbf16>
}
