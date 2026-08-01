// RUN: env IREE_METAL_SCATTER_ATOMIC=1 \
// RUN:   IREE_METAL_SCATTER_WINDOW_WORKGROUPS=0 \
// RUN:   iree-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-linalg-ext-to-loops))" \
// RUN:   %s | FileCheck %s --check-prefix=ATOMIC
// RUN: env IREE_METAL_SCATTER_ATOMIC=1 \
// RUN:   IREE_METAL_SCATTER_WINDOW_WORKGROUPS=1 \
// RUN:   iree-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-linalg-ext-to-loops))" \
// RUN:   %s | FileCheck %s --check-prefix=CONFLICT

func.func @scatter_add(
    %original: memref<16x8xf32>, %indices: memref<4x1xi32>,
    %updates: memref<4x8xf32>) {
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(false)
      ins(%updates, %indices : memref<4x8xf32>, memref<4x1xi32>)
      outs(%original : memref<16x8xf32>) {
    ^bb0(%update: f32, %current: f32):
      %sum = arith.addf %current, %update : f32
      iree_linalg_ext.yield %sum : f32
  }
  return
}

// ATOMIC-LABEL: func.func @scatter_add(
// ATOMIC:         memref.atomic_rmw addf

// CONFLICT-LABEL: func.func @scatter_add(
// CONFLICT-NOT:     memref.atomic_rmw
// CONFLICT:         memref.load
// CONFLICT:         arith.addf
// CONFLICT:         memref.store
