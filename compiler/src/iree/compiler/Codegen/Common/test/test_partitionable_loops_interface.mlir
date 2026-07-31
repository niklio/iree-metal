// RUN: iree-opt --iree-codegen-test-partitionable-loops-interface --split-input-file %s | FileCheck %s

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
func.func @generic_dynamic(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %init = tensor.empty(%d0, %d2) : tensor<?x?xf32>
  %0 = linalg.generic {
    indexing_maps = [#map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0: tensor<?x?x?xf32>) outs(%init : tensor<?x?xf32>)
    attrs = {__test_interface__ = true} {
      ^bb0(%arg1 : f32, %arg2 : f32):
        linalg.yield %arg1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @generic_dynamic(
//       CHECK:   util.unfoldable_constant dense<[0, 2]> : tensor<2xi32>

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
func.func @generic_unit_dim(%arg0 : tensor<1x?x?xf32>) -> tensor<1x?xf32> {
  %c2 = arith.constant 2 : index
  %d2 = tensor.dim %arg0, %c2 : tensor<1x?x?xf32>
  %init = tensor.empty(%d2) : tensor<1x?xf32>
  %0 = linalg.generic {
    indexing_maps = [#map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0: tensor<1x?x?xf32>) outs(%init : tensor<1x?xf32>)
    attrs = {__test_interface__ = true} {
      ^bb0(%arg1 : f32, %arg2 : f32):
        linalg.yield %arg1 : f32
    } -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}
// CHECK-LABEL: func.func @generic_unit_dim(
//       CHECK:   util.unfoldable_constant dense<2> : tensor<1xi32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @generic_4D(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf32>
  %d3 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf32>
  %init = tensor.empty(%d0, %d1, %d2, %d3) : tensor<?x?x?x?xf32>
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0: tensor<?x?x?x?xf32>) outs(%init : tensor<?x?x?x?xf32>)
    attrs = {__test_interface__ = true} {
      ^bb0(%arg1 : f32, %arg2 : f32):
        linalg.yield %arg1 : f32
    } -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func.func @generic_4D(
//       CHECK:   util.unfoldable_constant dense<[1, 2, 3]> : tensor<3xi32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @generic_4D_unit_dim(%arg0: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x1x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x1x?xf32>
  %d3 = tensor.dim %arg0, %c3 : tensor<?x?x1x?xf32>
  %init = tensor.empty(%d0, %d1, %d3) : tensor<?x?x1x?xf32>
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0: tensor<?x?x1x?xf32>) outs(%init : tensor<?x?x1x?xf32>)
    attrs = {__test_interface__ = true} {
      ^bb0(%arg1 : f32, %arg2 : f32):
        linalg.yield %arg1 : f32
    } -> tensor<?x?x1x?xf32>
  return %0 : tensor<?x?x1x?xf32>
}
// CHECK-LABEL: func.func @generic_4D_unit_dim(
//       CHECK:   util.unfoldable_constant dense<[0, 1, 3]> : tensor<3xi32>

// -----

func.func @named_op(%lhs : tensor<?x?xf32>, %rhs : tensor<?x?xf32>,
    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul {__test_interface__ = true}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @named_op(
//       CHECK:   util.unfoldable_constant dense<[0, 1]> : tensor<2xi32>

// -----

func.func @named_op_unit_dim(%lhs : tensor<1x?xf32>, %rhs : tensor<?x?xf32>,
    %init : tensor<1x?xf32>) -> tensor<1x?xf32> {
  %0 = linalg.matmul {__test_interface__ = true}
      ins(%lhs, %rhs : tensor<1x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<1x?xf32>) -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}
// CHECK-LABEL: func.func @named_op_unit_dim(
//       CHECK:   util.unfoldable_constant dense<1> : tensor<1xi32>

// -----

func.func @mmt4d(%lhs : tensor<?x?x?x?xf32>, %rhs : tensor<?x?x?x?xf32>,
    %init : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.mmt4d {__test_interface__ = true}
      ins(%lhs, %rhs : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%init : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func.func @mmt4d(
//       CHECK:   util.unfoldable_constant dense<[0, 1]> : tensor<2xi32>

// -----

func.func @mmt4d_unit_dim(%lhs : tensor<1x?x?x?xf32>, %rhs : tensor<?x?x?x?xf32>,
    %init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
  %0 = linalg.mmt4d {__test_interface__ = true}
      ins(%lhs, %rhs : tensor<1x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
  return %0 : tensor<1x?x?x?xf32>
}
// CHECK-LABEL: func.func @mmt4d_unit_dim(
//       CHECK:   util.unfoldable_constant dense<[0, 1]> : tensor<2xi32>

// -----

func.func @sort(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.sort
      {__test_interface__ = true}
      dimension(0)
      outs(%arg0 : tensor<?x?xf32>) {
        ^bb0(%arg1 : f32, %arg2 : f32):
          %1  = arith.cmpf ogt, %arg1, %arg2 : f32
          iree_linalg_ext.yield %1 : i1
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @sort(
//       CHECK:   util.unfoldable_constant dense<1> : tensor<1xi32>

// -----

func.func @sort_unit_dim(%arg0 : tensor<?x1xf32>) -> tensor<?x1xf32> {
  %0 = iree_linalg_ext.sort
      {__test_interface__ = true}
      dimension(0)
      outs(%arg0 : tensor<?x1xf32>) {
        ^bb0(%arg1 : f32, %arg2 : f32):
          %1  = arith.cmpf ogt, %arg1, %arg2 : f32
          iree_linalg_ext.yield %1 : i1
      } -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}
// CHECK-LABEL: func.func @sort_unit_dim(
//       CHECK:   util.unfoldable_constant dense<1> : tensor<1xi32>

// -----

func.func @concat_along_last(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.concat dim(1) %arg0, %arg1 {__test_interface__ = true} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @concat_along_last(
//       CHECK:   util.unfoldable_constant dense<0> : tensor<1xi32>

// -----

func.func @concat_along_first(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.concat dim(0) %arg0, %arg1 {__test_interface__ = true} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @concat_along_first(
//       CHECK:   util.unfoldable_constant dense<1> : tensor<1xi32>

// -----

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>

func.func @attention_single_result(
    %query: tensor<2x4x8xf32>, %key: tensor<2x6x8xf32>,
    %value: tensor<2x6x16xf32>) -> tensor<2x4x16xf32> {
  %scale = arith.constant 1.0 : f32
  %output = tensor.empty() : tensor<2x4x16xf32>
  %result = iree_linalg_ext.attention {
      __test_interface__ = true,
      indexing_maps = [#q, #k, #v, #s, #o]}
      ins(%query, %key, %value, %scale :
          tensor<2x4x8xf32>, tensor<2x6x8xf32>,
          tensor<2x6x16xf32>, f32)
      outs(%output : tensor<2x4x16xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<2x4x16xf32>
  return %result : tensor<2x4x16xf32>
}
// CHECK-LABEL: func.func @attention_single_result(
//       CHECK:   util.unfoldable_constant dense<[0, 1, 4]> : tensor<3xi32>

// -----

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#lse = affine_map<(b, m, k1, k2, n) -> (b, m)>

func.func @attention_with_logsumexp(
    %query: tensor<2x4x8xf32>, %key: tensor<2x6x8xf32>,
    %value: tensor<2x6x16xf32>)
    -> (tensor<2x4x16xf32>, tensor<2x4xf32>) {
  %scale = arith.constant 1.0 : f32
  %output = tensor.empty() : tensor<2x4x16xf32>
  %logsumexp = tensor.empty() : tensor<2x4xf32>
  %result:2 = iree_linalg_ext.attention {
      __test_interface__ = true,
      indexing_maps = [#q, #k, #v, #s, #o, #lse]}
      ins(%query, %key, %value, %scale :
          tensor<2x4x8xf32>, tensor<2x6x8xf32>,
          tensor<2x6x16xf32>, f32)
      outs(%output, %logsumexp :
          tensor<2x4x16xf32>, tensor<2x4xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<2x4x16xf32>, tensor<2x4xf32>
  return %result#0, %result#1 :
      tensor<2x4x16xf32>, tensor<2x4xf32>
}
// CHECK-LABEL: func.func @attention_with_logsumexp(
//       CHECK:   util.unfoldable_constant dense<[0, 1]> : tensor<2xi32>

// -----

#q = affine_map<(k1, k2, n) -> (k1)>
#k = affine_map<(k1, k2, n) -> (k2, k1)>
#v = affine_map<(k1, k2, n) -> (k2, n)>
#s = affine_map<(k1, k2, n) -> ()>
#o = affine_map<(k1, k2, n) -> (n)>
#lse = affine_map<(k1, k2, n) -> ()>

// A scalar LSE has no safe partitioning dimension: even partitioning the
// output-only N dimension would make every tile write the same scalar.
func.func @attention_with_scalar_logsumexp(
    %query: tensor<8xf32>, %key: tensor<6x8xf32>,
    %value: tensor<6x16xf32>) -> (tensor<16xf32>, tensor<f32>) {
  %scale = arith.constant 1.0 : f32
  %output = tensor.empty() : tensor<16xf32>
  %logsumexp = tensor.empty() : tensor<f32>
  %result:2 = iree_linalg_ext.attention {
      __test_interface__ = true,
      indexing_maps = [#q, #k, #v, #s, #o, #lse]}
      ins(%query, %key, %value, %scale :
          tensor<8xf32>, tensor<6x8xf32>, tensor<6x16xf32>, f32)
      outs(%output, %logsumexp : tensor<16xf32>, tensor<f32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<16xf32>, tensor<f32>
  return %result#0, %result#1 : tensor<16xf32>, tensor<f32>
}
// CHECK-LABEL: func.func @attention_with_scalar_logsumexp(
//       CHECK:   util.unfoldable_constant dense<> : tensor<0xi32>
