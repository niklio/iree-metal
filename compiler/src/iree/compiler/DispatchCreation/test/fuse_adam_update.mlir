// RUN: env IREE_METAL_FUSE_ADAM_UPDATE=1 iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fuse-multi-use-elementwise-producer{num-iterations=0}))" %s | FileCheck %s
// RUN: sed 's/jit(adamw)/jit(layer_norm)/' %s | env IREE_METAL_FUSE_ADAM_UPDATE=1 iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fuse-multi-use-elementwise-producer{num-iterations=0}))" | FileCheck --check-prefix=NOFUSE %s

#map = affine_map<(d0) -> (d0)>

module @jit_adamw {
util.func public @fuse_adam_update(
    %gradient: tensor<16xf32>, %first: tensor<16xf32>,
    %second: tensor<16xf32>, %parameter: tensor<16xf32>,
    %scale: tensor<f32>)
    -> (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) {
  %first_init = tensor.empty() : tensor<16xf32>
  %second_init = tensor.empty() : tensor<16xf32>
  %parameter_init = tensor.empty() : tensor<16xf32>
  %moments:2 = linalg.generic {
      indexing_maps = [#map, #map, #map, affine_map<(d0) -> ()>, #map, #map],
      iterator_types = ["parallel"]}
      ins(%gradient, %first, %second, %scale
          : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<f32>)
      outs(%first_init, %second_init : tensor<16xf32>, tensor<16xf32>) {
    ^bb0(%g: f32, %m: f32, %v: f32, %s: f32, %m_out: f32, %v_out: f32):
      %scaled_g = arith.mulf %g, %s : f32
      %weighted_g = arith.mulf %scaled_g, %s : f32
      %weighted_m = arith.mulf %m, %s : f32
      %new_m = arith.addf %weighted_m, %weighted_g : f32
      %g_squared = arith.mulf %scaled_g, %scaled_g : f32
      %weighted_g_squared = arith.mulf %g_squared, %s : f32
      %weighted_v = arith.mulf %v, %s : f32
      %new_v = arith.addf %weighted_v, %weighted_g_squared : f32
      linalg.yield %new_m, %new_v : f32, f32
  } -> (tensor<16xf32>, tensor<16xf32>)
  %updated = linalg.generic {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel"]}
      ins(%moments#0, %moments#1, %parameter
          : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>)
      outs(%parameter_init : tensor<16xf32>) {
    ^bb0(%m: f32, %v: f32, %p: f32, %p_out: f32):
      %adjusted_v = arith.addf %v, %m : f32
      %inv_sqrt = math.rsqrt %adjusted_v : f32
      %scaled = arith.mulf %m, %inv_sqrt : f32
      %decay = arith.mulf %p, %m : f32
      %combined = arith.addf %scaled, %decay : f32
      %step = arith.mulf %combined, %inv_sqrt : f32
      %new_p = arith.subf %p, %step : f32
      linalg.yield %new_p : f32
  } -> tensor<16xf32> loc("jit(adamw)/jit(main)/parameter_update")
  %first_result = iree_tensor_ext.compute_barrier.end %moments#0
      : tensor<16xf32> -> tensor<16xf32>
  %second_result = iree_tensor_ext.compute_barrier.end %moments#1
      : tensor<16xf32> -> tensor<16xf32>
  util.return %first_result, %second_result, %updated
      : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
}
}

// CHECK-LABEL: util.func public @fuse_adam_update
// CHECK: %[[FUSED:.+]]:3 = linalg.generic
// CHECK: math.rsqrt
// CHECK: linalg.yield {{.*}}, {{.*}}, {{.*}} : f32, f32, f32
// CHECK: iree_tensor_ext.compute_barrier.end %[[FUSED]]#0
// CHECK: iree_tensor_ext.compute_barrier.end %[[FUSED]]#1
// CHECK: util.return {{.*}}, {{.*}}, %[[FUSED]]#2
// CHECK-NOT: linalg.generic

// NOFUSE-LABEL: util.func public @fuse_adam_update
// NOFUSE: %[[MOMENTS:.+]]:2 = linalg.generic
// NOFUSE: %[[UPDATED:.+]] = linalg.generic
// NOFUSE: util.return {{.*}}, {{.*}}, %[[UPDATED]]
