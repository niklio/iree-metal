// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}))" %s | FileCheck %s

// A non-unique scatter keeps its batch loops serial while distributing the
// update window across a 32-lane Metal workgroup. Fusing this producer would
// materialize 8 * 512 * 32 BF16 values, requiring at least 256 KiB of
// workgroup memory. Keep the update computation in its own dispatch.
util.func public @large_nonunique_scatter_update(
    %updates: tensor<8x512x768xbf16>,
    %indices: tensor<8x512xi32>,
    %original: tensor<50257x768xbf16>) -> tensor<50257x768xbf16> {
  %empty = tensor.empty() : tensor<8x512x768xbf16>
  %scaled = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%updates : tensor<8x512x768xbf16>)
      outs(%empty : tensor<8x512x768xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    linalg.yield %in : bf16
  } -> tensor<8x512x768xbf16>
  %result = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false)
      ins(%scaled, %indices : tensor<8x512x768xbf16>, tensor<8x512xi32>)
      outs(%original : tensor<50257x768xbf16>) {
  ^bb0(%update: bf16, %old: bf16):
    %sum = arith.addf %update, %old : bf16
    iree_linalg_ext.yield %sum : bf16
  } -> tensor<50257x768xbf16>
  util.return %result : tensor<50257x768xbf16>
}

// CHECK-LABEL: util.func public @large_nonunique_scatter_update
//       CHECK: %[[UPDATES:.+]] = flow.dispatch.region
//       CHECK:   %[[SCALED:.+]] = linalg.generic
//       CHECK:   flow.return %[[SCALED]]
//       CHECK: %[[SCATTER:.+]] = flow.dispatch.region
//       CHECK:   iree_linalg_ext.scatter
//  CHECK-SAME:     ins(%[[UPDATES]],
//       CHECK:   flow.return
//       CHECK: util.return %[[SCATTER]]

// -----

// Small non-unique scatter updates stay fused so the resource guard does not
// turn into a blanket performance restriction.
util.func public @small_nonunique_scatter_update(
    %updates: tensor<4x1x128xf32>,
    %indices: tensor<4x1xi32>,
    %original: tensor<8192x128xf32>) -> tensor<8192x128xf32> {
  %empty = tensor.empty() : tensor<4x1x128xf32>
  %scaled = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%updates : tensor<4x1x128xf32>)
      outs(%empty : tensor<4x1x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x1x128xf32>
  %result = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false)
      ins(%scaled, %indices : tensor<4x1x128xf32>, tensor<4x1xi32>)
      outs(%original : tensor<8192x128xf32>) {
  ^bb0(%update: f32, %old: f32):
    %sum = arith.addf %update, %old : f32
    iree_linalg_ext.yield %sum : f32
  } -> tensor<8192x128xf32>
  util.return %result : tensor<8192x128xf32>
}

// CHECK-LABEL: util.func public @small_nonunique_scatter_update
//       CHECK: %[[SCATTER:.+]] = flow.dispatch.region
//       CHECK:   %[[SCALED:.+]] = linalg.generic
//       CHECK:   iree_linalg_ext.scatter
//  CHECK-SAME:     ins(%[[SCALED]],
//       CHECK:   flow.return
//       CHECK: util.return %[[SCATTER]]
