// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [16], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @subgroup_reduce_f32(%2: tensor<2x512xf32>) -> tensor<2xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %3 = tensor.empty() : tensor<2xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2xf32>) -> tensor<2xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<2x512xf32>) outs(%4 : tensor<2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %out, %in : f32
    linalg.yield %6 : f32
  } -> tensor<2xf32>
  return %5 : tensor<2xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 512]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [128, 1, 1]>
//      CHECK: func.func @subgroup_reduce_f32(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @subgroup_reduce_f16(%2: tensor<16x4096x4096xf16>) -> tensor<16x4096x4096xf16> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f16
  %3 = tensor.empty() : tensor<16x4096x4096xf16>
  %4 = tensor.empty() : tensor<16x4096xf16>
  %5 = linalg.fill ins(%cst : f16) outs(%4 : tensor<16x4096xf16>) -> tensor<16x4096xf16>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<16x4096x4096xf16>) outs(%5 : tensor<16x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %8 = arith.addf %in, %out : f16
    linalg.yield %8 : f16
  } -> tensor<16x4096xf16>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %6 : tensor<16x4096x4096xf16>, tensor<16x4096xf16>) outs(%3 : tensor<16x4096x4096xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.divf %in, %in_0 : f16
    linalg.yield %8 : f16
  } -> tensor<16x4096x4096xf16>
  return %7 : tensor<16x4096x4096xf16>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 512]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//      CHECK: func.func @subgroup_reduce_f16(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 64]]>
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @subgroup_reduce_dynamic(%10: tensor<8x?xf32>) -> tensor<8xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %11 = tensor.empty() : tensor<8xf32>
  %12 = linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%11 : tensor<8xf32>) -> tensor<8xf32>
  %13 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%10 : tensor<8x?xf32>) outs(%12 : tensor<8xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %out: f32):
    %14 = math.powf %in, %cst_0 : f32
    %15 = arith.addf %14, %out : f32
    linalg.yield %15 : f32
  } -> tensor<8xf32>
  return %13 : tensor<8xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 64]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//      CHECK: func.func @subgroup_reduce_dynamic(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// The first two functions verify workgroup_size is limited to subgroup_size
// when the consumer's broadcast dimensions can't be distributed.
// The third function verifies workgroup_size is not limited to subgroup_size
// when the consumer's broadcast dimensions can be distributed.
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [32], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> ()>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map8 = affine_map<(d0, d1) -> ()>

func.func @reduction_with_elementwise_consumer(
    %input: tensor<6144xf32>,
    %other: tensor<64x3x32xf32>,
    %filled: tensor<f32>,
    %empty_out: tensor<64x3x32xf32>
) -> tensor<64x3x32xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %reduction = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["reduction"]
  } ins(%input : tensor<6144xf32>) outs(%filled : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<f32>
  %epilogue = linalg.generic {
    indexing_maps = [#map2, #map3, #map2],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%other, %reduction : tensor<64x3x32xf32>, tensor<f32>)
    outs(%empty_out : tensor<64x3x32xf32>) {
  ^bb0(%in: f32, %in_reduction: f32, %out: f32):
    %0 = arith.addf %in, %in_reduction : f32
    linalg.yield %0 : f32
  } -> tensor<64x3x32xf32>
  return %epilogue : tensor<64x3x32xf32>
}

func.func @batch_reduction_and_elementwise_consumer(
    %input: tensor<128x6144xf32>,
    %other: tensor<128x64x3x32xf32>,
    %filled: tensor<128xf32>,
    %empty_out: tensor<128x64x3x32xf32>
) -> tensor<128x64x3x32xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %reduction = linalg.generic {
    indexing_maps = [#map4, #map5],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<128x6144xf32>) outs(%filled : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<128xf32>
  %epilogue = linalg.generic {
    indexing_maps = [#map6, #map7, #map6],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%other, %reduction : tensor<128x64x3x32xf32>, tensor<128xf32>)
    outs(%empty_out : tensor<128x64x3x32xf32>) {
  ^bb0(%in: f32, %in_reduction: f32, %out: f32):
    %0 = arith.addf %in, %in_reduction : f32
    linalg.yield %0 : f32
  } -> tensor<128x64x3x32xf32>
  return %epilogue : tensor<128x64x3x32xf32>
}

func.func @reduction_with_distributable_elementwise_consumer(
    %input: tensor<6144xf32>,
    %other: tensor<512x12xf32>,
    %filled: tensor<f32>,
    %empty_out: tensor<512x12xf32>
) -> tensor<512x12xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %reduction = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["reduction"]
  } ins(%input : tensor<6144xf32>) outs(%filled : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<f32>
  %epilogue = linalg.generic {
    indexing_maps = [#map4, #map8, #map4],
    iterator_types = ["parallel", "parallel"]
  } ins(%other, %reduction : tensor<512x12xf32>, tensor<f32>)
    outs(%empty_out : tensor<512x12xf32>) {
  ^bb0(%in: f32, %in_reduction: f32, %out: f32):
    %0 = arith.addf %in, %in_reduction : f32
    linalg.yield %0 : f32
  } -> tensor<512x12xf32>
  return %epilogue : tensor<512x12xf32>
}

//  CHECK-DAG: #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [256]{{\]}}>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 256]{{\]}}>
//  CHECK-DAG: #[[CONFIG3:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[], [2048]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION1:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//  CHECK-DAG: #[[TRANSLATION2:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [512, 1, 1]>
//      CHECK: func.func @reduction_with_elementwise_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG1]]
//      CHECK: func.func @batch_reduction_and_elementwise_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION1]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG2]]
//      CHECK: func.func @reduction_with_distributable_elementwise_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION2]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG3]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG3]]

// -----

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [32], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#map9 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map12 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map13 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @fail_reduction_with_nondistributable_consumer(
    %input: tensor<16x64x74xf32>,
    %filled: tensor<16x74xf32>,
    %empty_out: tensor<16x74x64xf32>
) -> tensor<16x74x64xf32> attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %cst_eps = arith.constant 9.99999974E-5 : f32
  %cst_scale = arith.constant 8.000000e+00 : f32

  %reduction = linalg.generic {
    indexing_maps = [#map9, #map13],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%input : tensor<16x64x74xf32>) outs(%filled : tensor<16x74xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<16x74xf32>

  %epilogue = linalg.generic {
    indexing_maps = [#map10, #map11, #map12],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input, %reduction : tensor<16x64x74xf32>, tensor<16x74xf32>)
    outs(%empty_out : tensor<16x74x64xf32>) {
  ^bb0(%in: f32, %in_reduction: f32, %out: f32):
    %0 = arith.maximumf %in, %in_reduction : f32
    %1 = math.sqrt %0 : f32
    %2 = arith.mulf %in, %cst_scale : f32
    %3 = arith.divf %2, %1 : f32
    linalg.yield %3 : f32
  } -> tensor<16x74x64xf32>
  return %epilogue : tensor<16x74x64xf32>
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize {{.*}}>
//      CHECK: func.func @fail_reduction_with_nondistributable_consumer(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-NOT: pipeline = SPIRVSubgroupReduce
//      CHECK: return

// -----

// Static non-multiple reductions smaller than a subgroup must not use the
// subgroup-reduce pipeline. With the transposed input map produced by a
// [T,D] -> [B,T,D] broadcast VJP, that pipeline fans the short reduction out
// from lane 0 through workgroup memory without a barrier. The B=31/32/33 cases
// pin the boundary, while B=577 protects the correct, fast masked-loop path.
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32|b16, subgroup = shuffle,
    subgroup_size_choices = [32], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#broadcast_vjp_input = affine_map<(d0, d1) -> (d1, d0)>
#broadcast_vjp_output = affine_map<(d0, d1) -> (d0)>

func.func @broadcast_vjp_reduce_b2(%input: tensor<2x128xf32>) -> tensor<128xbf16>
    attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %zero = arith.constant 0.000000e+00 : bf16
  %empty = tensor.empty() : tensor<128xbf16>
  %filled = linalg.fill ins(%zero : bf16) outs(%empty : tensor<128xbf16>) -> tensor<128xbf16>
  %result = linalg.generic {
      indexing_maps = [#broadcast_vjp_input, #broadcast_vjp_output],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<2x128xf32>) outs(%filled : tensor<128xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %in_bf16 = arith.truncf %in : f32 to bf16
    %sum = arith.addf %out, %in_bf16 : bf16
    linalg.yield %sum : bf16
  } -> tensor<128xbf16>
  return %result : tensor<128xbf16>
}

func.func @broadcast_vjp_reduce_b31(%input: tensor<31x128xf32>) -> tensor<128xbf16>
    attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %zero = arith.constant 0.000000e+00 : bf16
  %empty = tensor.empty() : tensor<128xbf16>
  %filled = linalg.fill ins(%zero : bf16) outs(%empty : tensor<128xbf16>) -> tensor<128xbf16>
  %result = linalg.generic {
      indexing_maps = [#broadcast_vjp_input, #broadcast_vjp_output],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<31x128xf32>) outs(%filled : tensor<128xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %in_bf16 = arith.truncf %in : f32 to bf16
    %sum = arith.addf %out, %in_bf16 : bf16
    linalg.yield %sum : bf16
  } -> tensor<128xbf16>
  return %result : tensor<128xbf16>
}

func.func @broadcast_vjp_reduce_b32(%input: tensor<32x128xf32>) -> tensor<128xbf16>
    attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %zero = arith.constant 0.000000e+00 : bf16
  %empty = tensor.empty() : tensor<128xbf16>
  %filled = linalg.fill ins(%zero : bf16) outs(%empty : tensor<128xbf16>) -> tensor<128xbf16>
  %result = linalg.generic {
      indexing_maps = [#broadcast_vjp_input, #broadcast_vjp_output],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<32x128xf32>) outs(%filled : tensor<128xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %in_bf16 = arith.truncf %in : f32 to bf16
    %sum = arith.addf %out, %in_bf16 : bf16
    linalg.yield %sum : bf16
  } -> tensor<128xbf16>
  return %result : tensor<128xbf16>
}

func.func @broadcast_vjp_reduce_b33(%input: tensor<33x128xf32>) -> tensor<128xbf16>
    attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %zero = arith.constant 0.000000e+00 : bf16
  %empty = tensor.empty() : tensor<128xbf16>
  %filled = linalg.fill ins(%zero : bf16) outs(%empty : tensor<128xbf16>) -> tensor<128xbf16>
  %result = linalg.generic {
      indexing_maps = [#broadcast_vjp_input, #broadcast_vjp_output],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<33x128xf32>) outs(%filled : tensor<128xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %in_bf16 = arith.truncf %in : f32 to bf16
    %sum = arith.addf %out, %in_bf16 : bf16
    linalg.yield %sum : bf16
  } -> tensor<128xbf16>
  return %result : tensor<128xbf16>
}

func.func @broadcast_vjp_reduce_b577(%input: tensor<577x128xf32>) -> tensor<128xbf16>
    attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %zero = arith.constant 0.000000e+00 : bf16
  %empty = tensor.empty() : tensor<128xbf16>
  %filled = linalg.fill ins(%zero : bf16) outs(%empty : tensor<128xbf16>) -> tensor<128xbf16>
  %result = linalg.generic {
      indexing_maps = [#broadcast_vjp_input, #broadcast_vjp_output],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<577x128xf32>) outs(%filled : tensor<128xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %in_bf16 = arith.truncf %in : f32 to bf16
    %sum = arith.addf %out, %in_bf16 : bf16
    linalg.yield %sum : bf16
  } -> tensor<128xbf16>
  return %result : tensor<128xbf16>
}

//  CHECK-DAG: #[[B2_CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32], [1], [0, 2]{{\]}}>
//  CHECK-DAG: #[[B31_CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32], [1], [0, 31]{{\]}}>
//  CHECK-DAG: #[[SUBGROUP_CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 32]{{\]}}>
//  CHECK-DAG: #[[BASE_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [32, 1, 1]>
//  CHECK-DAG: #[[SUBGROUP_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [32, 1, 1]>
//      CHECK: func.func @broadcast_vjp_reduce_b2(
// CHECK-SAME:     translation_info = #[[BASE_TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[B2_CONFIG]]
//      CHECK: func.func @broadcast_vjp_reduce_b31(
// CHECK-SAME:     translation_info = #[[BASE_TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[B31_CONFIG]]
//      CHECK: func.func @broadcast_vjp_reduce_b32(
// CHECK-SAME:     translation_info = #[[SUBGROUP_TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[SUBGROUP_CONFIG]]
//      CHECK: func.func @broadcast_vjp_reduce_b33(
// CHECK-SAME:     translation_info = #[[SUBGROUP_TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[SUBGROUP_CONFIG]]
//      CHECK: func.func @broadcast_vjp_reduce_b577(
// CHECK-SAME:     translation_info = #[[SUBGROUP_TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[SUBGROUP_CONFIG]]

// -----

// A multi-dimensional root reduction can share its innermost reduction
// dimension with fused row reductions. The subgroup pipeline cannot lower the
// resulting rank-2 vectors, so use the general pipeline.
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  iree_codegen.target_info = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
    compute = fp32|int32, storage = b32, subgroup = shuffle,
    subgroup_size_choices = [32], max_workgroup_sizes = [512, 512, 512],
    max_thread_count_per_workgroup = 512, max_workgroup_memory_bytes = 16384,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>

func.func @multidim_reduction_chain(%input: tensor<3x32xf32>) -> tensor<f32>
    attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
  %zero = arith.constant 0.000000e+00 : f32
  %row_empty = tensor.empty() : tensor<3xf32>
  %row_init = linalg.fill ins(%zero : f32) outs(%row_empty : tensor<3xf32>) -> tensor<3xf32>
  %row = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<3x32xf32>) outs(%row_init : tensor<3xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %sum = arith.addf %acc, %in : f32
    linalg.yield %sum : f32
  } -> tensor<3xf32>
  %scalar_empty = tensor.empty() : tensor<f32>
  %scalar_init = linalg.fill ins(%zero : f32) outs(%scalar_empty : tensor<f32>) -> tensor<f32>
  %total = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> ()>],
      iterator_types = ["reduction", "reduction"]}
      ins(%input, %row : tensor<3x32xf32>, tensor<3xf32>)
      outs(%scalar_init : tensor<f32>) {
  ^bb0(%in: f32, %row_value: f32, %acc: f32):
    %value = arith.addf %in, %row_value : f32
    %sum = arith.addf %acc, %value : f32
    linalg.yield %sum : f32
  } -> tensor<f32>
  return %total : tensor<f32>
}

//  CHECK-DAG: #[[CHAIN_CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = []>
//  CHECK-DAG: #[[CHAIN_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute workgroup_size = [1, 1, 1]>
//      CHECK: func.func @multidim_reduction_chain(
// CHECK-SAME:     translation_info = #[[CHAIN_TRANSLATION]]
//      CHECK:   linalg.generic {{.*}} iterator_types = ["reduction", "reduction"]
// CHECK-SAME:       lowering_config = #[[CHAIN_CONFIG]]
