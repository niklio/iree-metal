// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-vector-alloc))" | FileCheck %s

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [0, 0]
>

func.func @test(%vector: vector<16x16xf16>) -> vector<16x16xf16> {
  %out = iree_vector_ext.to_layout %vector to layout(#layout) {shared_memory_conversion} : vector<16x16xf16>
  return %out : vector<16x16xf16>
}


//    CHECK-LABEL: func.func @test
//         CHECK:    gpu.barrier memfence [#gpu.address_space<workgroup>]
//         CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x16xf16, #gpu.address_space<workgroup>>
//         CHECK:    %[[WRITE:.+]] = vector.transfer_write %{{.*}}, %[[ALLOC]]
//         CHECK:    %[[BAR:.+]]   = iree_gpu.value_barrier %[[WRITE]]
//         CHECK:    %[[READ:.+]]  = vector.transfer_read %[[BAR]]
//         CHECK:    %[[OUT:.+]]   = iree_vector_ext.to_layout %[[READ]]

// -----

#coordinate_layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 4],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [4, 1],

  subgroup_strides = [0, 0],
  thread_strides = [0, 0]
>

// A causal-mask coordinate expression is cheap to recompute, but its direct
// and transposed uses require incompatible layouts. Rematerialize the entire
// index DAG for the second use instead of communicating it through workgroup
// memory.
func.func @rematerialize_index_coordinate_dag() -> (vector<4x4xindex>, vector<4x4xindex>) {
  %scale = arith.constant dense<2> : vector<4xindex>
  %offset = arith.constant dense<1> : vector<4xindex>
  %step = vector.step : vector<4xindex>
  %scaled = arith.muli %step, %scale : vector<4xindex>
  %shifted = arith.addi %scaled, %offset : vector<4xindex>
  %coordinates = vector.broadcast %shifted : vector<4xindex> to vector<4x4xindex>
  %direct = iree_vector_ext.to_layout %coordinates to layout(#coordinate_layout) : vector<4x4xindex>
  %transpose = vector.transpose %coordinates, [1, 0] : vector<4x4xindex> to vector<4x4xindex>
  %transposed = iree_vector_ext.to_layout %transpose to layout(#coordinate_layout) : vector<4x4xindex>
  return %direct, %transposed : vector<4x4xindex>, vector<4x4xindex>
}

// CHECK-LABEL: func.func @rematerialize_index_coordinate_dag
// CHECK-COUNT-2: vector.step
// CHECK-COUNT-2: arith.muli
// CHECK-COUNT-2: arith.addi
// CHECK-COUNT-2: vector.broadcast
// CHECK-COUNT-2: iree_vector_ext.to_layout
// CHECK-NOT: bufferization.alloc_tensor{{.*}}index
// CHECK: return

// Do not partially rematerialize the cheap index operations when their shaped
// leaf is an expensive vector read. The layout conflict must stay above the
// whole DAG, leaving each source operation single-copy.
func.func @do_not_rematerialize_index_dag_with_expensive_leaf(%source: memref<4xindex>) -> (vector<4x4xindex>, vector<4x4xindex>) {
  %c0 = arith.constant 0 : index
  %offset = arith.constant dense<1> : vector<4xindex>
  %leaf = vector.transfer_read %source[%c0], %c0 {in_bounds = [true]} : memref<4xindex>, vector<4xindex>
  %shifted = arith.addi %leaf, %offset : vector<4xindex>
  %coordinates = vector.broadcast %shifted : vector<4xindex> to vector<4x4xindex>
  %direct = iree_vector_ext.to_layout %coordinates to layout(#coordinate_layout) : vector<4x4xindex>
  %transpose = vector.transpose %coordinates, [1, 0] : vector<4x4xindex> to vector<4x4xindex>
  %transposed = iree_vector_ext.to_layout %transpose to layout(#coordinate_layout) : vector<4x4xindex>
  return %direct, %transposed : vector<4x4xindex>, vector<4x4xindex>
}

// CHECK-LABEL: func.func @do_not_rematerialize_index_dag_with_expensive_leaf
// CHECK-COUNT-1: vector.transfer_read
// CHECK-COUNT-1: arith.addi
// CHECK-COUNT-1: vector.broadcast
// CHECK: bufferization.alloc_tensor(){{.*}}tensor<4x4xindex
// CHECK: return
