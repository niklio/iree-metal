// RUN: env -u IREE_METAL_SCATTER_ATOMIC \
// RUN:   IREE_METAL_SCATTER_WINDOW_WORKGROUPS=0 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=OFF \
// RUN:   --implicit-check-not=iree_codegen.apple_scatter_window_workgroups
// RUN: env -u IREE_METAL_SCATTER_ATOMIC \
// RUN:   IREE_METAL_SCATTER_WINDOW_WORKGROUPS=true \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=OFF \
// RUN:   --implicit-check-not=iree_codegen.apple_scatter_window_workgroups
// RUN: env -u IREE_METAL_SCATTER_ATOMIC \
// RUN:   IREE_METAL_SCATTER_WINDOW_WORKGROUPS=1 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=ON
// RUN: env -u IREE_METAL_SCATTER_ATOMIC \
// RUN:   IREE_METAL_SCATTER_WINDOW_WORKGROUPS=1 \
// RUN:   iree-opt --iree-gpu-test-target=volta@vulkan \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=NONAPPLE \
// RUN:   --implicit-check-not=iree_codegen.apple_scatter_window_workgroups
// RUN: env IREE_METAL_SCATTER_WINDOW_WORKGROUPS=1 \
// RUN:   IREE_METAL_SCATTER_ATOMIC=0 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=CONFLICT \
// RUN:   --implicit-check-not=iree_codegen.apple_scatter_window_workgroups
// RUN: env -u IREE_METAL_SCATTER_ATOMIC \
// RUN:   -u IREE_METAL_SCATTER_WINDOW_TILE \
// RUN:   IREE_METAL_SCATTER_WINDOW_WORKGROUPS=1 \
// RUN:   IREE_METAL_SCATTER_SMALL_OUTPUT_WINDOW_TILE=32 \
// RUN:   iree-opt --iree-gpu-test-target=apple@metal \
// RUN:   --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' \
// RUN:   %s | FileCheck %s --check-prefix=BOUNDED

func.func @nonunique_scatter(
    %updates: tensor<8x512x768xf32>,
    %indices: tensor<8x512xi32>,
    %init: tensor<50257x768xf32>) -> tensor<50257x768xf32> {
  %result = iree_linalg_ext.scatter dimension_map = [0]
      unique_indices(false)
      ins(%updates, %indices :
          tensor<8x512x768xf32>, tensor<8x512xi32>)
      outs(%init : tensor<50257x768xf32>) {
    ^bb0(%update: f32, %current: f32):
      %sum = arith.addf %current, %update : f32
      iree_linalg_ext.yield %sum : f32
  } -> tensor<50257x768xf32>
  return %result : tensor<50257x768xf32>
}

// ON-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 1], [0, 0, 1]{{\]}}>
// ON-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute workgroup_size = [32, 1, 1]>
// ON-LABEL: func.func @nonunique_scatter(
// ON-SAME:      translation_info = #[[TRANSLATION]]
// ON:         iree_linalg_ext.scatter {
// ON-SAME:      iree_codegen.apple_scatter_window_workgroups
// ON-SAME:      lowering_config = #[[CONFIG]]

// OFF-LABEL: func.func @nonunique_scatter(
// OFF:         iree_linalg_ext.scatter
// OFF-SAME:      lowering_config = #{{.+}}

// NONAPPLE-LABEL: func.func @nonunique_scatter(
// NONAPPLE:         iree_linalg_ext.scatter
// NONAPPLE-SAME:      lowering_config = #{{.+}}

// CONFLICT: warning: IREE_METAL_SCATTER_WINDOW_WORKGROUPS=1 conflicts with IREE_METAL_SCATTER_ATOMIC; disabling both scatter experiments

func.func @bounded_bf16_scatter(
    %updates: tensor<8x512x768xbf16>,
    %indices: tensor<8x512xi32>,
    %init: tensor<50257x768xbf16>) -> tensor<50257x768xbf16> {
  %result = iree_linalg_ext.scatter dimension_map = [0]
      unique_indices(false)
      ins(%updates, %indices :
          tensor<8x512x768xbf16>, tensor<8x512xi32>)
      outs(%init : tensor<50257x768xbf16>) {
    ^bb0(%update: bf16, %current: bf16):
      %sum = arith.addf %current, %update : bf16
      iree_linalg_ext.yield %sum : bf16
  } -> tensor<50257x768xbf16>
  return %result : tensor<50257x768xbf16>
}

func.func @oversized_bf16_scatter(
    %updates: tensor<8x1024x768xbf16>,
    %indices: tensor<8x1024xi32>,
    %init: tensor<50257x768xbf16>) -> tensor<50257x768xbf16> {
  %result = iree_linalg_ext.scatter dimension_map = [0]
      unique_indices(false)
      ins(%updates, %indices :
          tensor<8x1024x768xbf16>, tensor<8x1024xi32>)
      outs(%init : tensor<50257x768xbf16>) {
    ^bb0(%update: bf16, %current: bf16):
      %sum = arith.addf %current, %update : bf16
      iree_linalg_ext.yield %sum : bf16
  } -> tensor<50257x768xbf16>
  return %result : tensor<50257x768xbf16>
}

// BOUNDED-DAG: #[[DEFAULT_CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 1], [0, 0, 1]{{\]}}>
// BOUNDED-DAG: #[[BF16_CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 32], [0, 0, 1]{{\]}}>
// BOUNDED-LABEL: func.func @nonunique_scatter(
// BOUNDED:         iree_linalg_ext.scatter
// BOUNDED-SAME:      lowering_config = #[[DEFAULT_CONFIG]]
// BOUNDED-LABEL: func.func @bounded_bf16_scatter(
// BOUNDED:         iree_linalg_ext.scatter
// BOUNDED-SAME:      lowering_config = #[[BF16_CONFIG]]
// BOUNDED-LABEL: func.func @oversized_bf16_scatter(
// BOUNDED:         iree_linalg_ext.scatter
// BOUNDED-SAME:      lowering_config = #[[DEFAULT_CONFIG]]
