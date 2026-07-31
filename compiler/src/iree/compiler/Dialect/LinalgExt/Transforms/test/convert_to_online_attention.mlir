// RUN: iree-opt --split-input-file --iree-linalg-ext-convert-attention-to-online-attention %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-linalg-ext-convert-attention-to-online-attention,func.func(iree-linalg-ext-decompose-attention{use-exp2=true}))" %s | FileCheck %s --check-prefix=EXP-MODE

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

func.func @attention(%q: tensor<2x10x4096x128xf16>, %k: tensor<2x10x4096x128xf16>, %v: tensor<2x10x4096x128xf16>)
                     -> tensor<2x10x4096x128xf16> {
  %scale = arith.constant 0.125 : f16
  %acc = tensor.empty() : tensor<2x10x4096x128xf16>
  %out = iree_linalg_ext.attention
         {indexing_maps = [#map, #map1, #map2, #map3, #map4]}
         ins(%q, %k, %v, %scale : tensor<2x10x4096x128xf16>, tensor<2x10x4096x128xf16>, tensor<2x10x4096x128xf16>, f16)
         outs(%acc : tensor<2x10x4096x128xf16>) {
              ^bb0(%score: f32):
                iree_linalg_ext.yield %score : f32
         } -> tensor<2x10x4096x128xf16>
  func.return %out : tensor<2x10x4096x128xf16>
}

// CHECK-LABEL: func.func @attention
// CHECK-SAME: %[[Q:.+]]: tensor<2x10x4096x128xf16>, %[[K:.+]]: tensor<2x10x4096x128xf16>, %[[V:.+]]: tensor<2x10x4096x128xf16>
// CHECK-DAG: %[[ACC_INIT:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[MAX_INIT:.+]] = arith.constant -3.40282347E+38 : f32
// CHECK-DAG: %[[SUM_INIT:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ACC_FILL:.+]] = linalg.fill ins(%[[ACC_INIT]]
// CHECK-DAG: %[[MAX_FILL:.+]] = linalg.fill ins(%[[MAX_INIT]]
// CHECK-DAG: %[[SUM_FILL:.+]] = linalg.fill ins(%[[SUM_INIT]]
// CHECK: %[[OUT:.+]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:         ins(%[[Q]], %[[K]], %[[V]]
// CHECK-SAME:         outs(%[[ACC_FILL]], %[[MAX_FILL]], %[[SUM_FILL]]
// CHECK-NEXT:             ^[[BLOCK:.+]](%[[SCORE:.+]]: f32):
// CHECK-NEXT:               iree_linalg_ext.yield %[[SCORE]] : f32
// CHECK-NEXT:        }
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[OUT]]#2, %[[OUT]]#0
// CHECK: arith.divf
// CHECK: arith.mulf
// CHECK: arith.truncf
// CHECK: linalg.yield

// -----

#mapQ = affine_map<(b, h, m, n, k1, k2) -> (b, h, m, k1)>
#mapK = affine_map<(b, h, m, n, k1, k2) -> (b, h, k2, k1)>
#mapV = affine_map<(b, h, m, n, k1, k2) -> (b, h, k2, n)>
#mapS = affine_map<(b, h, m, n, k1, k2) -> ()>
#mapM = affine_map<(b, h, m, n, k1, k2) -> (b, h, m, k2)>
#mapO = affine_map<(b, h, m, n, k1, k2) -> (b, h, m, n)>

func.func @attention_i1_mask(
    %q: tensor<1x1x2x1xf32>,
    %k: tensor<1x1x2x1xf32>,
    %v: tensor<1x1x2x1xf32>) -> tensor<1x1x2x1xf32> {
  %scale = arith.constant 1.0 : f32
  %mask = arith.constant dense<false> : tensor<1x1x2x2xi1>
  %acc = tensor.empty() : tensor<1x1x2x1xf32>
  %out = iree_linalg_ext.attention
      {indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO]}
      ins(%q, %k, %v, %scale, %mask :
          tensor<1x1x2x1xf32>, tensor<1x1x2x1xf32>,
          tensor<1x1x2x1xf32>, f32, tensor<1x1x2x2xi1>)
      outs(%acc : tensor<1x1x2x1xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x1x2x1xf32>
  return %out : tensor<1x1x2x1xf32>
}

// CHECK-LABEL: func.func @attention_i1_mask
// CHECK: %[[MASK:.+]] = arith.constant dense<false> : tensor<1x1x2x2xi1>
// CHECK: %[[ONLINE:.+]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME: ins(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[MASK]]
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[ONLINE]]#2, %[[ONLINE]]#0
// CHECK: ^bb0(%[[SUM:.+]]: f32, %[[X:.+]]: f32, %{{.+}}: f32):
// CHECK-NEXT: %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[SUM_IS_ZERO:.+]] = arith.cmpf oeq, %[[SUM]], %[[ZERO]] : f32
// CHECK-NEXT: %[[SAFE_SUM:.+]] = arith.select %[[SUM_IS_ZERO]], %[[ONE]], %[[SUM]] : f32
// CHECK-NEXT: %[[RECIPROCAL:.+]] = arith.divf %[[ONE]], %[[SAFE_SUM]] : f32
// CHECK-NEXT: %[[SCALED:.+]] = arith.mulf %[[RECIPROCAL]], %[[X]] : f32
// CHECK-NEXT: %[[ZERO_IF_EMPTY:.+]] = arith.select %[[SUM_IS_ZERO]], %[[ZERO]], %[[SCALED]] : f32
// CHECK-NEXT: linalg.yield %[[ZERO_IF_EMPTY]] : f32

// -----

#mapQ = affine_map<(b, h, m, n, k1, k2) -> (b, h, m, k1)>
#mapK = affine_map<(b, h, m, n, k1, k2) -> (b, h, k2, k1)>
#mapV = affine_map<(b, h, m, n, k1, k2) -> (b, h, k2, n)>
#mapS = affine_map<(b, h, m, n, k1, k2) -> ()>
#mapM = affine_map<(b, h, m, n, k1, k2) -> (b, h, m, k2)>
#mapO = affine_map<(b, h, m, n, k1, k2) -> (b, h, m, n)>
#mapR = affine_map<(b, h, m, n, k1, k2) -> (b, h, m)>

func.func @attention_logsumexp_all_false_mask(
    %q: tensor<1x1x2x1xf32>,
    %k: tensor<1x1x2x1xf32>,
    %v: tensor<1x1x2x1xf32>)
    -> (tensor<1x1x2x1xf32>, tensor<1x1x2xf32>) {
  %scale = arith.constant 1.0 : f32
  %mask = arith.constant dense<false> : tensor<1x1x2x2xi1>
  %acc = tensor.empty() : tensor<1x1x2x1xf32>
  %logsumexp = tensor.empty() : tensor<1x1x2xf32>
  %result:2 = iree_linalg_ext.attention {
      decomposition_config = {use_exp2 = true},
      indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR]}
      ins(%q, %k, %v, %scale, %mask :
          tensor<1x1x2x1xf32>, tensor<1x1x2x1xf32>,
          tensor<1x1x2x1xf32>, f32, tensor<1x1x2x2xi1>)
      outs(%acc, %logsumexp :
          tensor<1x1x2x1xf32>, tensor<1x1x2xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x1x2x1xf32>, tensor<1x1x2xf32>
  return %result#0, %result#1 :
      tensor<1x1x2x1xf32>, tensor<1x1x2xf32>
}

// CHECK-LABEL: func.func @attention_logsumexp_all_false_mask
// CHECK: %[[ONLINE:.+]]:3 = iree_linalg_ext.online_attention
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[ONLINE]]#2, %[[ONLINE]]#0
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[ONLINE]]#1, %[[ONLINE]]#2
// CHECK: ^bb0(%[[MAX:.+]]: f32, %[[SUM:.+]]: f32, %{{.+}}: f32):
// CHECK-NEXT: %[[LOG_SUM:.+]] = math.log2 %[[SUM]] : f32
// CHECK-NEXT: %[[LSE_BASE2:.+]] = arith.addf %[[MAX]], %[[LOG_SUM]] : f32
// CHECK-NEXT: %[[LN2:.+]] = arith.constant 0.693147182 : f32
// CHECK-NEXT: %[[LSE:.+]] = arith.mulf %[[LSE_BASE2]], %[[LN2]] : f32
// CHECK-NEXT: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[SUM_IS_ZERO:.+]] = arith.cmpf oeq, %[[SUM]], %[[ZERO]] : f32
// CHECK-NEXT: %[[NEG_INF:.+]] = arith.constant 0xFF800000 : f32
// CHECK-NEXT: %[[FINAL_LSE:.+]] = arith.select %[[SUM_IS_ZERO]], %[[NEG_INF]], %[[LSE]] : f32
// CHECK-NEXT: linalg.yield %[[FINAL_LSE]] : f32

// -----

#mapQ = affine_map<(b, h, m, n, k1, k2) -> (b, h, m, k1)>
#mapK = affine_map<(b, h, m, n, k1, k2) -> (b, h, k2, k1)>
#mapV = affine_map<(b, h, m, n, k1, k2) -> (b, h, k2, n)>
#mapS = affine_map<(b, h, m, n, k1, k2) -> ()>
#mapO = affine_map<(b, h, m, n, k1, k2) -> (b, h, m, n)>
#mapR = affine_map<(b, h, m, n, k1, k2) -> (b, h, m)>

func.func @attention_natural_exp(
    %q: tensor<1x1x2x1xf32>,
    %k: tensor<1x1x2x1xf32>,
    %v: tensor<1x1x2x1xf32>)
    -> (tensor<1x1x2x1xf32>, tensor<1x1x2xf32>) {
  %scale = arith.constant 1.0 : f32
  %acc = tensor.empty() : tensor<1x1x2x1xf32>
  %logsumexp = tensor.empty() : tensor<1x1x2xf32>
  %result:2 = iree_linalg_ext.attention {
      decomposition_config = {use_exp2 = false},
      indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR]}
      ins(%q, %k, %v, %scale :
          tensor<1x1x2x1xf32>, tensor<1x1x2x1xf32>,
          tensor<1x1x2x1xf32>, f32)
      outs(%acc, %logsumexp :
          tensor<1x1x2x1xf32>, tensor<1x1x2xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x1x2x1xf32>, tensor<1x1x2xf32>
  return %result#0, %result#1 :
      tensor<1x1x2x1xf32>, tensor<1x1x2xf32>
}

// EXP-MODE-LABEL: func.func @attention_natural_exp
// EXP-MODE-NOT: math.exp2
// EXP-MODE: math.exp
// EXP-MODE-NOT: math.log2
// EXP-MODE: math.log
