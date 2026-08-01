// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: env -u IREE_METAL_GELU_REMAT iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" %s | FileCheck %s --check-prefixes=OFF,NEGATIVE
// RUN: env IREE_METAL_GELU_REMAT=0 iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" %s | FileCheck %s --check-prefixes=OFF,NEGATIVE
// RUN: env IREE_METAL_GELU_REMAT=1 iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" %s | FileCheck %s --check-prefixes=ON,NEGATIVE
// RUN: env IREE_METAL_GELU_REMAT=1 iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse),iree-stablehlo-convert-flash-attention-dispatch,func.func(canonicalize,cse))" %s | FileCheck %s --check-prefixes=ON,NEGATIVE

// ON-LABEL: func.func @paired_tanh_gelu(
// ON-SAME: %[[X:[^:]+]]: tensor<2x3x4xbf16>
// ON-DAG: %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<2x3x4xbf16>
// ON-DAG: %[[SCALE:.+]] = stablehlo.constant dense<7.968750e-01> : tensor<2x3x4xbf16>
// ON-DAG: %[[COEFFICIENT:.+]] = stablehlo.constant dense<4.467770e-02> : tensor<2x3x4xbf16>
// ON-DAG: %[[THREE:.+]] = stablehlo.constant dense<3.000000e+00> : tensor<2x3x4xbf16>
// ON-DAG: %[[HALF:.+]] = stablehlo.constant dense<5.000000e-01> : tensor<2x3x4xbf16>
// ON: %[[OLD_HALF_X:.+]] = stablehlo.multiply %[[HALF]], %[[X]]
// ON: %[[OLD_TANH:.+]] = stablehlo.tanh
// ON: %[[OLD_ONE_PLUS:.+]] = stablehlo.add %[[ONE]], %[[OLD_TANH]]
// ON: %[[OUTPUT:.+]] = stablehlo.multiply %[[OLD_HALF_X]], %[[OLD_ONE_PLUS]]
// ON: %[[DY:.+]] = stablehlo.negate
// ON-NEXT: %[[BARRIER:.+]]:2 = stablehlo.optimization_barrier %[[X]], %[[DY]]
// ON-NEXT: %[[HALF_X:.+]] = stablehlo.multiply %[[HALF]], %[[BARRIER]]#0
// ON-NEXT: %[[X_SQUARED:.+]] = stablehlo.multiply %[[BARRIER]]#0, %[[BARRIER]]#0
// ON-NEXT: %[[X_CUBED:.+]] = stablehlo.multiply %[[X_SQUARED]], %[[BARRIER]]#0
// ON-NEXT: %[[THREE_X_SQUARED:.+]] = stablehlo.multiply %[[THREE]], %[[X_SQUARED]]
// ON-NEXT: %[[CUBIC:.+]] = stablehlo.multiply %[[COEFFICIENT]], %[[X_CUBED]]
// ON-NEXT: %[[INNER:.+]] = stablehlo.add %[[BARRIER]]#0, %[[CUBIC]]
// ON-NEXT: %[[SCALED:.+]] = stablehlo.multiply %[[SCALE]], %[[INNER]]
// ON-NEXT: %[[REMAT_TANH:.+]] = stablehlo.tanh %[[SCALED]]
// ON-NEXT: %[[ONE_MINUS:.+]] = stablehlo.subtract %[[ONE]], %[[REMAT_TANH]]
// ON-NEXT: %[[ONE_PLUS:.+]] = stablehlo.add %[[ONE]], %[[REMAT_TANH]]
// ON-NEXT: %[[HALF_X_DY:.+]] = stablehlo.multiply %[[HALF_X]], %[[BARRIER]]#1
// ON-NEXT: %[[TANH_LEFT:.+]] = stablehlo.multiply %[[HALF_X_DY]], %[[ONE_MINUS]]
// ON-NEXT: %[[TANH_RIGHT:.+]] = stablehlo.multiply %[[TANH_LEFT]], %[[REMAT_TANH]]
// ON-NEXT: %[[TANH_DERIVATIVE:.+]] = stablehlo.add %[[TANH_LEFT]], %[[TANH_RIGHT]]
// ON-NEXT: %[[SCALED_TANH_DERIVATIVE:.+]] = stablehlo.multiply %[[SCALE]], %[[TANH_DERIVATIVE]]
// ON-NEXT: %[[SCALED_CUBIC_DERIVATIVE:.+]] = stablehlo.multiply %[[COEFFICIENT]], %[[SCALED_TANH_DERIVATIVE]]
// ON-NEXT: %[[CUBIC_DERIVATIVE:.+]] = stablehlo.multiply %[[SCALED_CUBIC_DERIVATIVE]], %[[THREE_X_SQUARED]]
// ON-NEXT: %[[INNER_DERIVATIVE:.+]] = stablehlo.add %[[SCALED_TANH_DERIVATIVE]], %[[CUBIC_DERIVATIVE]]
// ON-NEXT: %[[DY_ONE_PLUS:.+]] = stablehlo.multiply %[[BARRIER]]#1, %[[ONE_PLUS]]
// ON-NEXT: %[[LINEAR_DERIVATIVE:.+]] = stablehlo.multiply %[[HALF]], %[[DY_ONE_PLUS]]
// ON-NEXT: %[[DX:.+]] = stablehlo.add %[[INNER_DERIVATIVE]], %[[LINEAR_DERIVATIVE]]
// ON-NOT: stablehlo.tanh
// ON: return %[[OUTPUT]], %[[DX]]

// OFF-LABEL: func.func @paired_tanh_gelu(
// OFF-NOT: stablehlo.optimization_barrier
// OFF: stablehlo.tanh
// OFF-NOT: stablehlo.tanh
// OFF: return

func.func @paired_tanh_gelu(%x: tensor<2x3x4xbf16>, %incoming_grad: tensor<2x3x4xbf16>) -> (tensor<2x3x4xbf16>, tensor<2x3x4xbf16>) {
  %one = stablehlo.constant dense<1.000000e+00> : tensor<2x3x4xbf16>
  %scale = stablehlo.constant dense<7.968750e-01> : tensor<2x3x4xbf16>
  %cubic_coefficient = stablehlo.constant dense<4.467770e-02> : tensor<2x3x4xbf16>
  %three = stablehlo.constant dense<3.000000e+00> : tensor<2x3x4xbf16>
  %half = stablehlo.constant dense<5.000000e-01> : tensor<2x3x4xbf16>
  %half_x = stablehlo.multiply %half, %x : tensor<2x3x4xbf16>
  %x_squared = stablehlo.multiply %x, %x : tensor<2x3x4xbf16>
  %x_cubed = stablehlo.multiply %x_squared, %x : tensor<2x3x4xbf16>
  %three_x_squared = stablehlo.multiply %three, %x_squared : tensor<2x3x4xbf16>
  %cubic = stablehlo.multiply %cubic_coefficient, %x_cubed : tensor<2x3x4xbf16>
  %inner = stablehlo.add %x, %cubic : tensor<2x3x4xbf16>
  %scaled = stablehlo.multiply %scale, %inner : tensor<2x3x4xbf16>
  %tanh = stablehlo.tanh %scaled : tensor<2x3x4xbf16>
  %one_minus_tanh = stablehlo.subtract %one, %tanh : tensor<2x3x4xbf16>
  %one_plus_tanh = stablehlo.add %one, %tanh : tensor<2x3x4xbf16>
  %output = stablehlo.multiply %half_x, %one_plus_tanh : tensor<2x3x4xbf16>
  %dy = stablehlo.negate %incoming_grad : tensor<2x3x4xbf16>
  %half_x_dy = stablehlo.multiply %half_x, %dy : tensor<2x3x4xbf16>
  %dy_one_plus_tanh = stablehlo.multiply %dy, %one_plus_tanh : tensor<2x3x4xbf16>
  %tanh_derivative_left = stablehlo.multiply %half_x_dy, %one_minus_tanh : tensor<2x3x4xbf16>
  %tanh_derivative_right = stablehlo.multiply %tanh_derivative_left, %tanh : tensor<2x3x4xbf16>
  %tanh_derivative = stablehlo.add %tanh_derivative_left, %tanh_derivative_right : tensor<2x3x4xbf16>
  %scaled_tanh_derivative = stablehlo.multiply %scale, %tanh_derivative : tensor<2x3x4xbf16>
  %scaled_cubic_derivative = stablehlo.multiply %cubic_coefficient, %scaled_tanh_derivative : tensor<2x3x4xbf16>
  %cubic_derivative = stablehlo.multiply %scaled_cubic_derivative, %three_x_squared : tensor<2x3x4xbf16>
  %inner_derivative = stablehlo.add %scaled_tanh_derivative, %cubic_derivative : tensor<2x3x4xbf16>
  %linear_derivative = stablehlo.multiply %half, %dy_one_plus_tanh : tensor<2x3x4xbf16>
  %dx = stablehlo.add %inner_derivative, %linear_derivative : tensor<2x3x4xbf16>
  return %output, %dx : tensor<2x3x4xbf16>, tensor<2x3x4xbf16>
}

// -----

// An almost-complete VJP must not be changed. The final linear contribution to
// dX is deliberately absent, so there is no complete paired rematerialization.
//
// NEGATIVE-LABEL: func.func @partial_tanh_gelu_vjp(
// NEGATIVE-NOT: stablehlo.optimization_barrier
// NEGATIVE: stablehlo.tanh
// NEGATIVE-NOT: stablehlo.tanh
// NEGATIVE: return

func.func @partial_tanh_gelu_vjp(%x: tensor<2x3x4xbf16>, %incoming_grad: tensor<2x3x4xbf16>) -> (tensor<2x3x4xbf16>, tensor<2x3x4xbf16>) {
  %one = stablehlo.constant dense<1.000000e+00> : tensor<2x3x4xbf16>
  %scale = stablehlo.constant dense<7.968750e-01> : tensor<2x3x4xbf16>
  %cubic_coefficient = stablehlo.constant dense<4.467770e-02> : tensor<2x3x4xbf16>
  %three = stablehlo.constant dense<3.000000e+00> : tensor<2x3x4xbf16>
  %half = stablehlo.constant dense<5.000000e-01> : tensor<2x3x4xbf16>
  %half_x = stablehlo.multiply %half, %x : tensor<2x3x4xbf16>
  %x_squared = stablehlo.multiply %x, %x : tensor<2x3x4xbf16>
  %x_cubed = stablehlo.multiply %x_squared, %x : tensor<2x3x4xbf16>
  %three_x_squared = stablehlo.multiply %three, %x_squared : tensor<2x3x4xbf16>
  %cubic = stablehlo.multiply %cubic_coefficient, %x_cubed : tensor<2x3x4xbf16>
  %inner = stablehlo.add %x, %cubic : tensor<2x3x4xbf16>
  %scaled = stablehlo.multiply %scale, %inner : tensor<2x3x4xbf16>
  %tanh = stablehlo.tanh %scaled : tensor<2x3x4xbf16>
  %one_minus_tanh = stablehlo.subtract %one, %tanh : tensor<2x3x4xbf16>
  %one_plus_tanh = stablehlo.add %one, %tanh : tensor<2x3x4xbf16>
  %output = stablehlo.multiply %half_x, %one_plus_tanh : tensor<2x3x4xbf16>
  %dy = stablehlo.negate %incoming_grad : tensor<2x3x4xbf16>
  %half_x_dy = stablehlo.multiply %half_x, %dy : tensor<2x3x4xbf16>
  %dy_one_plus_tanh = stablehlo.multiply %dy, %one_plus_tanh : tensor<2x3x4xbf16>
  %tanh_derivative_left = stablehlo.multiply %half_x_dy, %one_minus_tanh : tensor<2x3x4xbf16>
  %tanh_derivative_right = stablehlo.multiply %tanh_derivative_left, %tanh : tensor<2x3x4xbf16>
  %tanh_derivative = stablehlo.add %tanh_derivative_left, %tanh_derivative_right : tensor<2x3x4xbf16>
  %scaled_tanh_derivative = stablehlo.multiply %scale, %tanh_derivative : tensor<2x3x4xbf16>
  %scaled_cubic_derivative = stablehlo.multiply %cubic_coefficient, %scaled_tanh_derivative : tensor<2x3x4xbf16>
  %cubic_derivative = stablehlo.multiply %scaled_cubic_derivative, %three_x_squared : tensor<2x3x4xbf16>
  %partial_dx = stablehlo.add %scaled_tanh_derivative, %cubic_derivative : tensor<2x3x4xbf16>
  return %output, %partial_dx : tensor<2x3x4xbf16>, tensor<2x3x4xbf16>
}
