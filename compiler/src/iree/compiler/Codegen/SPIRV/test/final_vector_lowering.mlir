// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-spirv-final-vector-lowering))' \
// RUN:   %s | FileCheck %s --check-prefix=DEFAULT
// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-spirv-final-vector-lowering{drop-unit-dims=true}))' \
// RUN:   %s | FileCheck %s --check-prefix=DROP

func.func @drop_unit_dims(
    %lhs: vector<8x1x3xf32>,
    %rhs: vector<8x1x3xf32>) -> vector<8x1x3xf32> {
  %result = arith.addf %lhs, %rhs : vector<8x1x3xf32>
  return %result : vector<8x1x3xf32>
}

// DEFAULT-LABEL: func.func @drop_unit_dims
// DEFAULT:         arith.addf {{.*}} : vector<8x1x3xf32>

// DROP-LABEL: func.func @drop_unit_dims
// DROP:         %[[LHS:.+]] = vector.shape_cast %{{.+}} : vector<8x1x3xf32> to vector<8x3xf32>
// DROP:         %[[RHS:.+]] = vector.shape_cast %{{.+}} : vector<8x1x3xf32> to vector<8x3xf32>
// DROP:         %[[SUM:.+]] = arith.addf %[[LHS]], %[[RHS]] : vector<8x3xf32>
// DROP:         %[[RESULT:.+]] = vector.shape_cast %[[SUM]] : vector<8x3xf32> to vector<8x1x3xf32>
// DROP:         return %[[RESULT]] : vector<8x1x3xf32>
