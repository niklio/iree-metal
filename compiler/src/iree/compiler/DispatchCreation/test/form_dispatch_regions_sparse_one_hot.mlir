// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}))" %s | FileCheck %s

util.func public @fuse_sparse_one_hot_with_earliest_consumer(
    %labels: tensor<4xi32>, %logits: tensor<4x17xf32>)
    -> (tensor<4xf32>, tensor<4x17xf32>) {
  %c0 = arith.constant 0.0 : f32
  %one_hot_empty = tensor.empty() : tensor<4x17xf32>
  %one_hot = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%labels : tensor<4xi32>)
      outs(%one_hot_empty : tensor<4x17xf32>) {
    ^bb0(%label: i32, %out: f32):
      %column = linalg.index 1 : index
      %column_i32 = arith.index_cast %column : index to i32
      %matches = arith.cmpi eq, %label, %column_i32 : i32
      %value = arith.uitofp %matches : i1 to f32
      linalg.yield %value : f32
  } -> tensor<4x17xf32>
  %selected_empty = tensor.empty() : tensor<4xf32>
  %selected_init = linalg.fill ins(%c0 : f32)
      outs(%selected_empty : tensor<4xf32>) -> tensor<4xf32>
  %selected = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%logits, %one_hot : tensor<4x17xf32>, tensor<4x17xf32>)
      outs(%selected_init : tensor<4xf32>) {
    ^bb0(%logit: f32, %mask: f32, %sum: f32):
      %masked = arith.mulf %logit, %mask : f32
      %next = arith.addf %sum, %masked : f32
      linalg.yield %next : f32
  } -> tensor<4xf32>
  %gradient_empty = tensor.empty() : tensor<4x17xf32>
  %gradient = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%logits, %one_hot : tensor<4x17xf32>, tensor<4x17xf32>)
      outs(%gradient_empty : tensor<4x17xf32>) {
    ^bb0(%logit: f32, %mask: f32, %out: f32):
      %present = arith.cmpf une, %mask, %c0 : f32
      %value = arith.select %present, %logit, %c0 : f32
      linalg.yield %value : f32
  } -> tensor<4x17xf32>
  util.return %selected, %gradient : tensor<4xf32>, tensor<4x17xf32>
}

// CHECK-LABEL: util.func public @fuse_sparse_one_hot_with_earliest_consumer(
// CHECK-NOT: tensor<4x17xi1>
// CHECK: flow.dispatch.region
// CHECK: linalg.index 0
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK: flow.dispatch.region
// CHECK: "iree-metal.sparse-one-hot-columns"
// CHECK: linalg.index 0
// CHECK: flow.dispatch.region
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME: ins(%arg1, %arg0
// CHECK: arith.cmpi eq
// CHECK-NOT: arith.cmpf

// -----

// Mirrors JAX's canonical full-vocabulary cross-entropy selection, including
// the reversed input order and select-based masking seen after StableHLO
// conversion.
util.func public @rewrite_select_sparse_one_hot_selection(
    %labels: tensor<4xi32>, %logits: tensor<4x17xf32>)
    -> tensor<4xf32> {
  %c0 = arith.constant 0.0 : f32
  %one_hot_empty = tensor.empty() : tensor<4x17xf32>
  %one_hot = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%labels : tensor<4xi32>)
      outs(%one_hot_empty : tensor<4x17xf32>) {
    ^bb0(%label: i32, %out: f32):
      %column = linalg.index 1 : index
      %column_i32 = arith.index_cast %column : index to i32
      %matches = arith.cmpi eq, %label, %column_i32 : i32
      %value = arith.uitofp %matches : i1 to f32
      linalg.yield %value : f32
  } -> tensor<4x17xf32>
  %selected_empty = tensor.empty() : tensor<4xf32>
  %selected_init = linalg.fill ins(%c0 : f32)
      outs(%selected_empty : tensor<4xf32>) -> tensor<4xf32>
  %selected = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%one_hot, %logits : tensor<4x17xf32>, tensor<4x17xf32>)
      outs(%selected_init : tensor<4xf32>) {
    ^bb0(%mask: f32, %logit: f32, %sum: f32):
      %present = arith.cmpf une, %mask, %c0 : f32
      %value = arith.select %present, %logit, %c0 : f32
      %next = arith.addf %sum, %value : f32
      linalg.yield %next : f32
  } -> tensor<4xf32>
  util.return %selected : tensor<4xf32>
}

// CHECK-LABEL: util.func public @rewrite_select_sparse_one_hot_selection(
// CHECK: flow.dispatch.region
// CHECK: linalg.index 0
// CHECK: tensor.extract
// CHECK-NOT: arith.cmpf
// CHECK: flow.return
