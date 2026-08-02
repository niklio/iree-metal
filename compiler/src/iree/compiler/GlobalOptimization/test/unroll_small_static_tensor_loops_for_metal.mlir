// RUN: env -u IREE_METAL_NO_STATIC_TENSOR_LOOPS_IN_DISPATCH iree-opt \
// RUN:   --iree-global-opt-unroll-small-static-tensor-loops-for-metal %s \
// RUN:   | FileCheck %s --check-prefix=DISPATCH
// RUN: env IREE_METAL_NO_STATIC_TENSOR_LOOPS_IN_DISPATCH=1 iree-opt \
// RUN:   --iree-global-opt-unroll-small-static-tensor-loops-for-metal %s \
// RUN:   | FileCheck %s --check-prefix=UNROLL

#metal_target = #hal.executable.target<"metal-spirv", "metallib">

util.global private @device = #hal.device.target<"metal", [#metal_target]>
    : !hal.device

// DISPATCH-LABEL: func.func @scan
// DISPATCH:       flow.dispatch.region
// DISPATCH:         scf.for
// DISPATCH:           linalg.generic
// DISPATCH:           tensor.insert_slice
// DISPATCH:           bufferization.materialize_in_destination
// DISPATCH:           bufferization.materialize_in_destination
// DISPATCH:         flow.return
// DISPATCH:       } count() -> (index, index, index) {
// DISPATCH:         %[[ONE:.+]] = arith.constant 1 : index
// DISPATCH:         flow.return %[[ONE]], %[[ONE]], %[[ONE]] : index, index, index
// UNROLL-LABEL: func.func @scan
// UNROLL-NOT:     scf.for
// UNROLL-COUNT-4: linalg.generic
func.func @scan(%input: tensor<4x8xf32>)
    -> (tensor<8xf32>, tensor<4x8xf32>)
    attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %initial_state = arith.constant dense<0.0> : tensor<8xf32>
  %initial_history = tensor.empty() : tensor<4x8xf32>
  %results:2 = scf.for %i = %c0 to %c4 step %c1
      iter_args(%state = %initial_state, %history = %initial_history)
      -> (tensor<8xf32>, tensor<4x8xf32>) {
    %input_slice = tensor.extract_slice %input[%i, 0] [1, 8] [1, 1]
        : tensor<4x8xf32> to tensor<8xf32>
    %empty = tensor.empty() : tensor<8xf32>
    %next_state = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%state, %input_slice : tensor<8xf32>, tensor<8xf32>)
        outs(%empty : tensor<8xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
    } -> tensor<8xf32>
    %next_history = tensor.insert_slice %next_state into %history[%i, 0]
        [1, 8] [1, 1] : tensor<8xf32> into tensor<4x8xf32>
    scf.yield %next_state, %next_history
        : tensor<8xf32>, tensor<4x8xf32>
  }
  return %results#0, %results#1 : tensor<8xf32>, tensor<4x8xf32>
}
