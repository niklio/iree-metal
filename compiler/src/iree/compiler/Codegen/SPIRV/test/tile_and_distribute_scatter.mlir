// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-distribute)))))' %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 16], [1, 1]]>
#translation = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @static_scatter_update_slice  {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @static_scatter_update_slice layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [16 : index, 1 : index, 1 : index]
    }
    builtin.module {
      func.func @static_scatter_update_slice() {
        %c40 = arith.constant 40 : index
        %c500 = arith.constant 500 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<40x500xi32>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<40x1xi32>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<100x500xi32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        scf.for %arg0 = %workgroup_id_y to %c40 step %workgroup_count_y {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
          scf.for %arg1 = %3 to %c500 step %4 {
            %5 = affine.min affine_map<(d0) -> (16, -d0 + 500)>(%arg1)
            %6 = memref.subview %0[%arg0, %arg1] [1, %5] [1, 1] : memref<40x500xi32> to memref<1x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>>
            %7 = memref.cast %6 : memref<1x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>> to memref<?x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>>
            %8 = memref.subview %1[%arg0, 0] [1, 1] [1, 1] : memref<40x1xi32> to memref<1x1xi32, affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>>
            %9 = memref.cast %8 : memref<1x1xi32, affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>> to memref<?x1xi32, affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>>
            %10 = memref.subview %2[0, %arg1] [100, %5] [1, 1] : memref<100x500xi32> to memref<100x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>>
            iree_linalg_ext.scatter {lowering_config = #config} dimension_map = [0] unique_indices(true) ins(%7, %9 : memref<?x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>>, memref<?x1xi32, affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>>) outs(%10 : memref<100x?xi32, affine_map<(d0, d1)[s0] -> (d0 * 500 + s0 + d1)>>)  {
            ^bb0(%arg2: i32, %arg3: i32):
              iree_linalg_ext.yield %arg2 : i32
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @static_scatter_update_slice()
//       CHECK: %[[ARG0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//       CHECK: %[[ARG1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//       CHECK: %[[ARG2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//       CHECK: scf.for
//       CHECK:   scf.for
//       CHECK:     %[[WG_UPDATE:.+]] = memref.subview %[[ARG0]]
//       CHECK:     %[[WG_INDEX:.+]] = memref.subview %[[ARG1]]
//       CHECK:     %[[WG_TARGET:.+]] = memref.subview %[[ARG2]]
//       CHECK:     %[[TID_X:.+]] = gpu.thread_id x
//       CHECK:     %[[DIM_X:.+]] = gpu.block_dim x
//       CHECK:     scf.for %[[IV_X:.+]] = %[[TID_X]] to %{{.+}} step %[[DIM_X]]
//       CHECK:       %[[T_UPDATE:.+]] = memref.subview %[[WG_UPDATE]][0, %[[IV_X]]] [1, 1] [1, 1]
//       CHECK:       %[[T_INDEX:.+]] = memref.cast %[[WG_INDEX]]
//       CHECK:       %[[T_TARGET:.+]] = memref.subview %[[WG_TARGET]][0, %[[IV_X]]] [100, 1] [1, 1]
//       CHECK:       iree_linalg_ext.scatter
//  CHECK-SAME:         unique_indices(true)
//  CHECK-SAME:         ins(%[[T_UPDATE]], %[[T_INDEX]]
//  CHECK-SAME:         outs(%[[T_TARGET]]

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 32], [0, 0, 1]]>
#translation = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

// This is the buffer-form body of one workgroup after the trailing 32-column
// scatter slice has been formed. The fill and non-unique scatter intentionally
// share the same output memref.
hal.executable private @nonunique_scatter_window_fill {
  hal.executable.variant @metal_spirv_fb target(<"metal-spirv", "metal-msl-fb">) {
    hal.executable.export public @nonunique_scatter_window_fill
        layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    builtin.module {
      func.func @nonunique_scatter_window_fill() {
        %updates = hal.interface.binding.subspan
            layout(#pipeline_layout) binding(0) :
            memref<8x512x32xbf16>
        %indices = hal.interface.binding.subspan
            layout(#pipeline_layout) binding(1) :
            memref<8x512x1xi32>
        %output = hal.interface.binding.subspan
            layout(#pipeline_layout) binding(2) :
            memref<50257x32xbf16>
        %zero = arith.constant 0.000000e+00 : bf16
        linalg.fill ins(%zero : bf16)
            outs(%output : memref<50257x32xbf16>)
        iree_linalg_ext.scatter {
            iree_codegen.apple_scatter_window_workgroups,
            lowering_config = #config}
            dimension_map = [0] unique_indices(false)
            ins(%updates, %indices :
                memref<8x512x32xbf16>, memref<8x512x1xi32>)
            outs(%output : memref<50257x32xbf16>) {
          ^bb0(%update: bf16, %current: bf16):
            %sum = arith.addf %current, %update : bf16
            iree_linalg_ext.yield %sum : bf16
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @nonunique_scatter_window_fill()
//   CHECK-DAG: %[[UPDATES:.+]] = hal.interface.binding.subspan
//   CHECK-DAG: %[[INDICES:.+]] = hal.interface.binding.subspan
//   CHECK-DAG: %[[OUTPUT:.+]] = hal.interface.binding.subspan
//       CHECK: %[[FILL_TID:.+]] = gpu.thread_id x
//       CHECK: %[[FILL_DIM:.+]] = gpu.block_dim x
//       CHECK: scf.for %[[FILL_IV:.+]] = %[[FILL_TID]] to %{{.+}} step %[[FILL_DIM]]
//       CHECK:   %[[FILL_SLICE:.+]] = memref.subview %[[OUTPUT]][0, %[[FILL_IV]]] [50257, 1] [1, 1]
//       CHECK:   linalg.fill
//  CHECK-SAME:     outs(%[[FILL_SLICE]]
//       CHECK: %[[SCATTER_TID:.+]] = gpu.thread_id x
//       CHECK: %[[SCATTER_DIM:.+]] = gpu.block_dim x
//       CHECK: scf.for %[[SCATTER_IV:.+]] = %[[SCATTER_TID]] to %{{.+}} step %[[SCATTER_DIM]]
//       CHECK:   %[[UPDATE_SLICE:.+]] = memref.subview %[[UPDATES]][0, 0, %[[SCATTER_IV]]] [8, 512, 1] [1, 1, 1]
//       CHECK:   %[[INDEX_FULL:.+]] = memref.cast %[[INDICES]]
//       CHECK:   %[[OUTPUT_SLICE:.+]] = memref.subview %[[OUTPUT]][0, %[[SCATTER_IV]]] [50257, 1] [1, 1]
//       CHECK:   iree_linalg_ext.scatter
//  CHECK-SAME:       unique_indices(false)
//  CHECK-SAME:       ins(%[[UPDATE_SLICE]], %[[INDEX_FULL]]
//  CHECK-SAME:       outs(%[[OUTPUT_SLICE]]
//   CHECK-NOT: memref.atomic_rmw
