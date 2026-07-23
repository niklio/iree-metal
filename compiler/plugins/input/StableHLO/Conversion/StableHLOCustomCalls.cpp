// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering CHLO ops to StableHLO and Shape dialect ops,
// taking care of CHLO's broadcasting semantics

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Rewriters.h"
#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"
#include "stablehlo/dialect/BroadcastUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>
#include <cstdlib>

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOCUSTOMCALLS
#define GEN_PASS_DEF_CONVERTFLASHATTENTIONDISPATCH
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

// Computes householder using vector `v` and a `tau` matrice
// with the k-th element. See householder transformation as below:
// https://en.wikipedia.org/wiki/Householder_transformation
static Value computeHouseholder(Value v, Value tau, Value k,
                                ImplicitLocOpBuilder b) {
  auto vTy = cast<ShapedType>(v.getType());

  SmallVector<int64_t> hShape(vTy.getShape());
  hShape.push_back(hShape.back());

  auto hTy = RankedTensorType::get(hShape, vTy.getElementType());
  Value empty = tensor::EmptyOp::create(b, hShape, vTy.getElementType());

  auto outMap = b.getMultiDimIdentityMap(hShape.size());

  SmallVector<AffineExpr> exprs;
  for (int i = 0, s = vTy.getRank(); i < s; ++i) {
    exprs.push_back(b.getAffineDimExpr(i));
  }

  AffineMap vMap = AffineMap::get(hTy.getRank(), 0, exprs, b.getContext());

  exprs.back() = b.getAffineDimExpr(vTy.getRank());
  AffineMap vTMap = AffineMap::get(hTy.getRank(), 0, exprs, b.getContext());

  SmallVector<AffineMap> affineMaps = {vMap, vTMap, outMap};

  SmallVector<utils::IteratorType> iterTypes(hShape.size(),
                                             utils::IteratorType::parallel);

  Value zero =
      arith::ConstantOp::create(b, b.getZeroAttr(vTy.getElementType()));
  Value one =
      arith::ConstantOp::create(b, b.getFloatAttr(vTy.getElementType(), 1.0));

  return linalg::GenericOp::create(
             b, hTy, ValueRange{v, v}, empty, affineMaps, iterTypes,
             [&](OpBuilder &bb, Location loc, ValueRange args) {
               ImplicitLocOpBuilder b(loc, bb);
               SmallVector<Value> indices;
               for (int i = 0, s = hTy.getRank(); i < s; ++i) {
                 indices.push_back(linalg::IndexOp::create(b, loc, i));
               }

               SmallVector<Value> tauIndices(indices.begin(),
                                             indices.end() - 2);
               tauIndices.push_back(k);
               Value t = tensor::ExtractOp::create(b, tau, tauIndices);

               // Generates the lower triangularization of the matrix with
               // one values on the diagonal.
               auto tri = [&](Value v, Value i) {
                 Value eq =
                     arith::CmpIOp::create(b, arith::CmpIPredicate::eq, i, k);
                 Value lt =
                     arith::CmpIOp::create(b, arith::CmpIPredicate::ult, i, k);
                 Value sel = arith::SelectOp::create(b, eq, one, v);
                 return arith::SelectOp::create(b, lt, zero, sel);
               };

               Value v = tri(args[0], indices[indices.size() - 2]);
               Value vT = tri(args[1], indices[indices.size() - 1]);

               Value h = arith::MulFOp::create(b, v, vT);
               h = arith::MulFOp::create(b, h, t);

               Value isDiag = arith::CmpIOp::create(
                   b, arith::CmpIPredicate::eq, indices[indices.size() - 2],
                   indices[indices.size() - 1]);
               Value diag = arith::SelectOp::create(b, isDiag, one, zero);
               Value sub = arith::SubFOp::create(b, diag, h);

               linalg::YieldOp::create(b, sub);
             })
      .getResult(0);
}

// Slices the k-th column of matrix and computes the householder transformation
// for using the `tau` value.
static Value computeHouseholderSlice(Value matrix, Value tau, Value k,
                                     ImplicitLocOpBuilder b) {
  auto matrixTy = cast<ShapedType>(matrix.getType());
  int rank = matrixTy.getRank();

  SmallVector<OpFoldResult> vStrides(rank, b.getIndexAttr(1));
  SmallVector<int64_t> vShape(matrixTy.getShape());
  vShape[vShape.size() - 1] = 1;

  SmallVector<OpFoldResult> vOffsets(rank, b.getIndexAttr(0));
  vOffsets[vOffsets.size() - 1] = k;

  SmallVector<OpFoldResult> vSizes;
  for (auto v : vShape) {
    vSizes.push_back(b.getIndexAttr(v));
  }

  auto sliceTy = RankedTensorType::get(vShape, matrixTy.getElementType());
  Value v = tensor::ExtractSliceOp::create(b, sliceTy, matrix, vOffsets, vSizes,
                                           vStrides);

  SmallVector<ReassociationIndices> reass;
  for (int i = 0; i < rank - 2; ++i) {
    reass.push_back({i});
  }
  reass.push_back({rank - 2, rank - 1});

  ArrayRef<int64_t> collapseVShape(vShape.begin(), vShape.end() - 1);
  auto collapseVTy =
      RankedTensorType::get(collapseVShape, matrixTy.getElementType());
  Value collapseV = tensor::CollapseShapeOp::create(b, collapseVTy, v, reass);

  Value householder = computeHouseholder(collapseV, tau, k, b);
  return householder;
}

struct HouseholderReflectorRewriter final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using Base::Base;
  using OpAdaptor = mlir::stablehlo::CustomCallOp::Adaptor;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getCallTargetName() != "ProductOfElementaryHouseholderReflectors") {
      return rewriter.notifyMatchFailure(
          op, "not ProductOfElementaryHouseholderReflectors");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto matrix = op.getOperand(0);
    auto tau = op.getOperand(1);
    auto matrixTy = cast<ShapedType>(matrix.getType());
    auto tauTy = cast<ShapedType>(tau.getType());
    auto rank = matrixTy.getRank();

    if (isa<ComplexType>(matrixTy.getElementType())) {
      return rewriter.notifyMatchFailure(op, "complex types not supported");
    }

    if (rank < 2) {
      return rewriter.notifyMatchFailure(op, "requires minimum rank 2 matrix");
    }

    // Implementation needs to be checked to work with variable dimension
    // lengths. Should be relatively straightforward.
    if (!matrixTy.hasStaticShape() || !tauTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "not supported for dynamic shapes");
    }

    Value zero = arith::ConstantIndexOp::create(b, 0);
    Value one = arith::ConstantIndexOp::create(b, 1);
    Value k = arith::ConstantIndexOp::create(b, tauTy.getShape().back());
    Value householder0 = computeHouseholderSlice(matrix, tau, zero, b);
    auto scf = scf::ForOp::create(
        b, one, k, one, ValueRange{householder0},
        [&](OpBuilder &bb, Location loc, Value iv, ValueRange args) {
          ImplicitLocOpBuilder b(loc, bb);
          Value householder = computeHouseholderSlice(matrix, tau, iv, b);

          std::vector<int64_t> batch(rank - 2);
          for (int i = 0; i < rank - 2; ++i) {
            batch[i] = i;
          }
          std::vector<int64_t> lhsContract = {rank - 1};
          std::vector<int64_t> rhsContract = {rank - 2};

          auto dotNums = mlir::stablehlo::DotDimensionNumbersAttr::get(
              b.getContext(), batch, batch, lhsContract, rhsContract);
          Value dot = mlir::stablehlo::DotGeneralOp::create(
              b, householder0.getType(), args[0], householder, dotNums, nullptr,
              mlir::stablehlo::DotAlgorithmAttr{});
          scf::YieldOp::create(b, loc, dot);
        });

    SmallVector<OpFoldResult> vOffsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> vStrides(rank, b.getIndexAttr(1));
    SmallVector<int64_t> vShape(matrixTy.getShape());
    SmallVector<OpFoldResult> vSizes;
    for (auto v : vShape) {
      vSizes.push_back(b.getIndexAttr(v));
    }

    auto sliceTy = RankedTensorType::get(vShape, matrixTy.getElementType());
    Value v = tensor::ExtractSliceOp::create(b, sliceTy, scf.getResult(0),
                                             vOffsets, vSizes, vStrides);

    rewriter.replaceOp(op, v);
    return success();
  }
};

struct ShapeAssertionDrop final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using Base::Base;
  using OpAdaptor = mlir::stablehlo::CustomCallOp::Adaptor;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getCallTargetName() != "shape_assertion") {
      return rewriter.notifyMatchFailure(op, "not shape_assertion");
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition.
//===----------------------------------------------------------------------===//

struct LegalizeStableHLOCustomCalls final
    : impl::LegalizeStableHLOCustomCallsBase<LegalizeStableHLOCustomCalls> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    mlir::stablehlo::StablehloDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto f = getOperation();
    MLIRContext *ctx = f.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<HouseholderReflectorRewriter, ShapeAssertionDrop>(ctx);
    if (failed(applyPatternsGreedily(f, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// ---------------------------------------------------------------------------
// ConvertFlashAttentionDispatch: @flash_attention_* custom_call -> wrapper call
// ---------------------------------------------------------------------------

// The shared metal-spirv executable target key for the objects. A config-less
// target is "generic of" any specific metal-msl-fb target
// (ExecutableTargetAttr::isGenericOf short-circuits when either config is
// empty), so the hand-authored object attaches regardless of the concrete
// device target (e.g. the apple-m3 target that --iree-hal-target-device=metal
// produces). Do NOT add a config dict here or it must match the device exactly.
static std::string metalTargetStr() {
  return R"(#hal.executable.target<"metal-spirv", "metal-msl-fb">)";
}

// Builds a module string defining an external-kernel wrapper function `fn`
// (and its hal.executable.source `exe`) for one of the flash kernels,
// specialized to (n, s, d). Mirrors the validated kernels/test/*.mlir.
static std::string buildWrapperModule(StringRef target, StringRef exe,
                                      StringRef fn, int64_t n, int64_t s,
                                      int64_t d, StringRef kernelPath) {
  std::string mt = metalTargetStr();
  std::string t3 = llvm::formatv("tensor<{0}x{1}x{2}xf32>", n, s, d).str();
  std::string t2 = llvm::formatv("tensor<{0}x{1}xf32>", n, s).str();
  int64_t ns = n * s;

  auto exportHeader = [&](StringRef entry, StringRef bindings) {
    return llvm::formatv(
        R"(  hal.executable.source private @{0} attributes {{
    objects = #hal.executable.objects<{{{1} = [#hal.executable.object<{{path = "{4}"}>]}>
  } {{
    hal.executable.export public @{2} ordinal(0)
        layout(#hal.pipeline.layout<constants = 3, bindings = [{3}]>)
        count(%dev: !hal.device, %w: index) -> (index, index, index) {{
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%w]
      %c1 = arith.constant 1 : index
      hal.return %x, %c1, %c1 : index, index, index
    } attributes {{workgroup_size = [64 : index, 1 : index, 1 : index]}
  })",
        exe, mt, entry, bindings, kernelPath)
        .str();
  };
  std::string ro = "#hal.pipeline.binding<storage_buffer, ReadOnly>";
  std::string rw = "#hal.pipeline.binding<storage_buffer>";

  std::string consts = llvm::formatv(
      "    %w = arith.constant {0} : index\n    %n = arith.constant {1} : i32\n"
      "    %s = arith.constant {2} : i32\n    %d = arith.constant {3} : i32\n",
      ns, n, s, d).str();

  if (target == "flash_attention_fwd") {
    std::string bindings = ro + ", " + ro + ", " + ro + ", " + rw + ", " + rw;
    return llvm::formatv(
        "module {{\n{0}\n"
        "  func.func private @{1}(%Q: {2}, %K: {2}, %V: {2}) -> ({2}, {3}) {{\n{4}"
        "    %O, %L = flow.dispatch @{5}::@flash_attention_fwd[%w](%n, %s, %d, %Q, %K, %V)"
        " : (i32, i32, i32, {2}, {2}, {2}) -> ({2}, {3})\n"
        "    return %O, %L : {2}, {3}\n  }\n}\n",
        exportHeader("flash_attention_fwd", bindings), fn, t3, t2, consts, exe)
        .str();
  }
  if (target == "flash_attention_bwd_dq") {
    std::string bindings =
        ro + ", " + ro + ", " + ro + ", " + ro + ", " + ro + ", " + ro + ", " + rw;
    return llvm::formatv(
        "module {{\n{0}\n"
        "  func.func private @{1}(%Q: {2}, %K: {2}, %V: {2}, %dO: {2}, %L: {3}, %D: {3}) -> {2} {{\n{4}"
        "    %dQ = flow.dispatch @{5}::@flash_attention_bwd_dq[%w](%n, %s, %d, %Q, %K, %V, %dO, %L, %D)"
        " : (i32, i32, i32, {2}, {2}, {2}, {2}, {3}, {3}) -> {2}\n"
        "    return %dQ : {2}\n  }\n}\n",
        exportHeader("flash_attention_bwd_dq", bindings), fn, t3, t2, consts, exe)
        .str();
  }
  // flash_attention_bwd_dkdv
  std::string bindings = ro + ", " + ro + ", " + ro + ", " + ro + ", " + ro +
                         ", " + ro + ", " + rw + ", " + rw;
  return llvm::formatv(
      "module {{\n{0}\n"
      "  func.func private @{1}(%Q: {2}, %K: {2}, %V: {2}, %dO: {2}, %L: {3}, %D: {3}) -> ({2}, {2}) {{\n{4}"
      "    %dK, %dV = flow.dispatch @{5}::@flash_attention_bwd_dkdv[%w](%n, %s, %d, %Q, %K, %V, %dO, %L, %D)"
      " : (i32, i32, i32, {2}, {2}, {2}, {2}, {3}, {3}) -> ({2}, {2})\n"
      "    return %dK, %dV : {2}, {2}\n  }\n}\n",
      exportHeader("flash_attention_bwd_dkdv", bindings), fn, t3, t2, consts, exe)
      .str();
}

// Builds a wrapper for the custom simdgroup_matrix GEMM (kernels/gemm.metal):
// C[M,N] f32 = A[M,K] f16 @ B[K,N] f16. Mirrors kernels/test/gemm_test.mlir
// (32x32 tile, 128-thread workgroup, count = ceildiv(N,32) x ceildiv(M,32)).
static std::string buildGemmWrapper(StringRef exe, StringRef fn, int64_t M,
                                    int64_t N, int64_t K, StringRef kernelPath) {
  std::string mt = metalTargetStr();
  // bf16 inputs (was f16): bf16's wide exponent range is what unblocks training loss
  // (fp16 overflows as activations grow). gemm.metal reads bfloat, accumulates f32.
  std::string tA = llvm::formatv("tensor<{0}x{1}xbf16>", M, K).str();
  std::string tB = llvm::formatv("tensor<{0}x{1}xbf16>", K, N).str();
  std::string tC = llvm::formatv("tensor<{0}x{1}xf32>", M, N).str();
  std::string ro = "#hal.pipeline.binding<storage_buffer, ReadOnly>";
  std::string rw = "#hal.pipeline.binding<storage_buffer>";
  return llvm::formatv(
      R"(module {{
  hal.executable.source private @{0} attributes {{
    objects = #hal.executable.objects<{{{1} = [#hal.executable.object<{{path = "{2}"}>]}>
  } {{
    hal.executable.export public @gemm_sg ordinal(0)
        layout(#hal.pipeline.layout<constants = 3, bindings = [{3}, {3}, {4}]>)
        count(%dev: !hal.device, %wn: index, %wm: index) -> (index, index, index) {{
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 32)>()[%wn]
      %y = affine.apply affine_map<()[s0] -> (s0 ceildiv 32)>()[%wm]
      %c1 = arith.constant 1 : index
      hal.return %x, %y, %c1 : index, index, index
    } attributes {{workgroup_size = [128 : index, 1 : index, 1 : index]}
  }
  func.func private @{5}(%A: {6}, %B: {7}) -> {8} {{
    %wn = arith.constant {9} : index
    %wm = arith.constant {10} : index
    %Mi = arith.constant {10} : i32
    %Ni = arith.constant {9} : i32
    %Ki = arith.constant {11} : i32
    %C = flow.dispatch @{0}::@gemm_sg[%wn, %wm](%Mi, %Ni, %Ki, %A, %B) : (i32, i32, i32, {6}, {7}) -> {8}
    return %C : {8}
  }
}
)",
      exe, mt, kernelPath, ro, rw, fn, tA, tB, tC, N, M, K)
      .str();
}

// Fused cross-entropy (kernels/cross_entropy.metal): one threadgroup per row,
// 256 threads, count = M (one workgroup per row). ce_fwd: logits[M,V] f32 +
// targets[M] i32 -> loss[M], lse[M]. ce_bwd: logits + targets + lse -> dlogits[M,V].
static std::string buildCeWrapper(StringRef target, StringRef exe, StringRef fn,
                                  int64_t M, int64_t V, StringRef kernelPath) {
  std::string mt = metalTargetStr();
  std::string tMV = llvm::formatv("tensor<{0}x{1}xf32>", M, V).str();
  std::string tM = llvm::formatv("tensor<{0}xf32>", M).str();
  std::string tMi = llvm::formatv("tensor<{0}xi32>", M).str();
  std::string ro = "#hal.pipeline.binding<storage_buffer, ReadOnly>";
  std::string rw = "#hal.pipeline.binding<storage_buffer>";

  auto exportHeader = [&](StringRef entry, StringRef bindings) {
    return llvm::formatv(
        R"(  hal.executable.source private @{0} attributes {{
    objects = #hal.executable.objects<{{{1} = [#hal.executable.object<{{path = "{4}"}>]}>
  } {{
    hal.executable.export public @{2} ordinal(0)
        layout(#hal.pipeline.layout<constants = 2, bindings = [{3}]>)
        count(%dev: !hal.device, %w: index) -> (index, index, index) {{
      %c1 = arith.constant 1 : index
      hal.return %w, %c1, %c1 : index, index, index
    } attributes {{workgroup_size = [256 : index, 1 : index, 1 : index]}
  })",
        exe, mt, entry, bindings, kernelPath)
        .str();
  };
  // %w (index) = M workgroups; %Mi/%Vi (i32) are the push constants the kernel reads.
  std::string consts = llvm::formatv(
      "    %w = arith.constant {0} : index\n    %Mi = arith.constant {0} : i32\n"
      "    %Vi = arith.constant {1} : i32\n",
      M, V).str();

  if (target == "ce_fwd") {
    std::string bindings = ro + ", " + ro + ", " + rw + ", " + rw;
    return llvm::formatv(
        "module {{\n{0}\n"
        "  func.func private @{1}(%logits: {2}, %targets: {3}) -> ({4}, {4}) {{\n{5}"
        "    %loss, %lse = flow.dispatch @{6}::@ce_fwd[%w](%Mi, %Vi, %logits, %targets)"
        " : (i32, i32, {2}, {3}) -> ({4}, {4})\n"
        "    return %loss, %lse : {4}, {4}\n  }\n}\n",
        exportHeader("ce_fwd", bindings), fn, tMV, tMi, tM, consts, exe)
        .str();
  }
  // ce_bwd
  std::string bindings = ro + ", " + ro + ", " + ro + ", " + rw;
  return llvm::formatv(
      "module {{\n{0}\n"
      "  func.func private @{1}(%logits: {2}, %targets: {3}, %lse: {4}) -> {2} {{\n{5}"
      "    %dlogits = flow.dispatch @{6}::@ce_bwd[%w](%Mi, %Vi, %logits, %targets, %lse)"
      " : (i32, i32, {2}, {3}, {4}) -> {2}\n"
      "    return %dlogits : {2}\n  }\n}\n",
      exportHeader("ce_bwd", bindings), fn, tMV, tMi, tM, consts, exe)
      .str();
}

// Compiler-native flash (Layer 1, FORWARD). Recognize the attention subgraph
//   dot_general(softmax(scale * dot_general(Q, Kᵀ) [+ causal select]), V)
// and rewrite it to a `flash_attention_fwd` custom_call (which the loop below
// lowers to the flash flow.dispatch) — automatic, NO shim. Opt-in via
// NLEARN_COOP_RAISE_FLASH. Layer-1 targets causal + head_dim=64 (kernel scope);
// forward-only, so it's for inference until the Layer-3 backward matcher pairs it.
static void raiseFlashAttentionFwd(ModuleOp module) {
  if (!std::getenv("NLEARN_COOP_RAISE_FLASH"))
    return;
  SmallVector<mlir::stablehlo::DotGeneralOp> cands;
  module.walk([&](mlir::stablehlo::DotGeneralOp dg) { cands.push_back(dg); });
  for (auto av : cands) {
    auto avTy = dyn_cast<RankedTensorType>(av.getType());
    if (!avTy || avTy.getRank() != 4)
      continue;
    auto dn = av.getDotDimensionNumbers();
    if (dn.getLhsBatchingDimensions().size() != 2 ||
        dn.getLhsContractingDimensions().size() != 1 ||
        dn.getLhsContractingDimensions()[0] != 3)
      continue;
    // P (softmax) = divide(exp(subtract(masked_scaled, max)), sum)
    auto div = av.getLhs().getDefiningOp<mlir::stablehlo::DivOp>();
    if (!div)
      continue;
    auto expOp = div.getLhs().getDefiningOp<mlir::stablehlo::ExpOp>();
    if (!expOp)
      continue;
    auto sub = expOp.getOperand().getDefiningOp<mlir::stablehlo::SubtractOp>();
    if (!sub)
      continue;
    Value maskedScaled = sub.getLhs();
    bool causal = false;
    Value scaled = maskedScaled;
    if (auto sel = maskedScaled.getDefiningOp<mlir::stablehlo::SelectOp>()) {
      causal = true;
      scaled = sel.getOperand(1); // inline where: on-true branch = scaled scores
    } else if (auto callOp = maskedScaled.getDefiningOp<func::CallOp>()) {
      // JAX often outlines jnp.where into a func: @_where(mask, scores, negC).
      if (callOp.getNumOperands() == 3) {
        causal = true;
        scaled = callOp.getOperand(1);
      }
    }
    auto mul = scaled.getDefiningOp<mlir::stablehlo::MulOp>();
    if (!mul)
      continue;
    auto qk = mul.getLhs().getDefiningOp<mlir::stablehlo::DotGeneralOp>();
    if (!qk)
      qk = mul.getRhs().getDefiningOp<mlir::stablehlo::DotGeneralOp>();
    if (!qk)
      continue;
    Value Q = qk.getLhs();
    auto kt = qk.getRhs().getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (!kt)
      continue;
    Value K = kt.getOperand();
    Value V = av.getRhs();
    auto qTy = dyn_cast<RankedTensorType>(Q.getType());
    auto kTy = dyn_cast<RankedTensorType>(K.getType());
    auto vTy = dyn_cast<RankedTensorType>(V.getType());
    if (!qTy || !kTy || !vTy || qTy.getRank() != 4)
      continue;
    int64_t B = qTy.getDimSize(0), H = qTy.getDimSize(1), T = qTy.getDimSize(2),
            D = qTy.getDimSize(3);
    if (D != 64 || !causal) // Layer-1: causal + head_dim 64 only
      continue;
    if (kTy.getShape() != qTy.getShape() || vTy.getShape() != qTy.getShape())
      continue;

    // Layer-3: trace the autodiff-decomposed backward from the fwd match, then
    // do a coordinated fwd+bwd raise (must be together: the fwd raise removes the
    // softmax P the bwd references; flash uses the logsumexp L instead).
    //   dK = non-fwd dot_general using Q  -> [B,H,T,D]
    //   dQ = non-fwd dot_general using Kᵀ -> [B,H,T,D]
    //   dV = non-fwd dot_general using P  -> [B,H,D,T] (dVᵀ; needs transpose)
    //   dO = dV's non-P operand
    Value Pv = av.getLhs();
    Value Kt = qk.getRhs();
    auto findOtherDG =
        [&](Value v, Operation *excl) -> mlir::stablehlo::DotGeneralOp {
      for (Operation *u : v.getUsers())
        if (auto dg = dyn_cast<mlir::stablehlo::DotGeneralOp>(u))
          if (dg.getOperation() != excl)
            return dg;
      return nullptr;
    };
    auto dKdg = findOtherDG(Q, qk);
    auto dQdg = findOtherDG(Kt, qk);
    auto dVdg = findOtherDG(Pv, av);
    Value dO;
    if (dVdg)
      dO = (dVdg.getLhs() == Pv) ? dVdg.getRhs() : dVdg.getLhs();
    bool hasBwd = dKdg && dQdg && dVdg && dO;
    if (std::getenv("NLEARN_RAISE_BWD_DEBUG"))
      llvm::errs() << "[bwd-trace] hasBwd=" << hasBwd << "\n";

    // bwd requires dO to have a defining op (to position the bwd ops after it).
    if (hasBwd && !dO.getDefiningOp())
      hasBwd = false;

    MLIRContext *fctx = av.getContext();
    OpBuilder b(av);
    Location loc = av.getLoc();
    Type f32 = b.getF32Type();
    auto n3Ty = RankedTensorType::get({B * H, T, D}, f32);
    auto lTy = RankedTensorType::get({B * H, T}, f32);
    auto to3d = [&](OpBuilder &bld, Value x) -> Value {
      auto xt = cast<RankedTensorType>(x.getType());
      Value f = bld.create<mlir::stablehlo::ConvertOp>(
          loc, RankedTensorType::get(xt.getShape(), f32), x);
      return bld.create<mlir::stablehlo::ReshapeOp>(loc, n3Ty, f);
    };
    auto to4d = [&](OpBuilder &bld, Value x3, Type elemTy) -> Value {
      Value r = bld.create<mlir::stablehlo::ReshapeOp>(
          loc, RankedTensorType::get({B, H, T, D}, f32), x3);
      return bld.create<mlir::stablehlo::ConvertOp>(
          loc, RankedTensorType::get({B, H, T, D}, elemTy), r);
    };
    // Chunk each flash dispatch along the head dim (N=B*H) so a dispatch's grid
    // (n*T/64 threadgroups) stays under the level that triggers a scale-dependent
    // simdgroup_matrix race (validated: n<=48 stable, n=96 races). Target ~256 tg.
    int64_t N = B * H;
    // The intermittent flash NaN is a scale-dependent simdgroup_matrix race that
    // scales with CONCURRENT flash threadgroups (a single n=96 dispatch = 768 tg
    // races; the hero avoids it by dispatching attention sequentially per batch).
    // Fix (validated): chunk along the head dim AND serialize the chunks via an
    // optimization_barrier (below), so at most one flash dispatch runs at a time.
    // Since serialized dispatches run alone, the safe size is the single-dispatch-
    // alone race ceiling — threshold-mapped x8: 192/384/576 tg all bit-identical,
    // 768 tg races (NaN + silent corruption). Budget 384 sits 1.5x under the still-
    // clean 576 and 2x under the racing 768, while halving serialization depth vs
    // 192 (fewer chunks -> recovers the long-seq flash win). Grid per dispatch =
    // chunkHeads * ceil(T/64), so the default is T-adaptive. NLEARN_COOP_FLASH_CHUNK
    // overrides (pass >=N to force a single dispatch for debugging).
    constexpr int64_t kTgBudget = 384;
    int64_t qTiles = std::max<int64_t>(1, (T + 63) / 64);
    int64_t chunkHeads =
        std::min<int64_t>(N, std::max<int64_t>(1, kTgBudget / qTiles));
    if (const char *cs = std::getenv("NLEARN_COOP_FLASH_CHUNK"))
      chunkHeads = std::min<int64_t>(N, std::max<int64_t>(1, atoi(cs)));
    auto chunkedFlash = [&](OpBuilder &bld, StringRef target,
                            ArrayRef<Value> operands,
                            ArrayRef<RankedTensorType> resTys) -> SmallVector<Value> {
      SmallVector<SmallVector<Value>> resChunks(resTys.size());
      Value serialDep; // prev chunk's result[0], threaded to force sequential exec
      for (int64_t c0 = 0; c0 < N; c0 += chunkHeads) {
        int64_t cn = std::min(chunkHeads, N - c0);
        SmallVector<Value> ops;
        for (Value op : operands) {
          auto ot = cast<RankedTensorType>(op.getType());
          SmallVector<int64_t> start(ot.getRank(), 0), stride(ot.getRank(), 1);
          SmallVector<int64_t> limit(ot.getShape().begin(), ot.getShape().end());
          SmallVector<int64_t> sh(ot.getShape().begin(), ot.getShape().end());
          start[0] = c0;
          limit[0] = c0 + cn;
          sh[0] = cn;
          ops.push_back(bld.create<mlir::stablehlo::SliceOp>(
              loc, RankedTensorType::get(sh, ot.getElementType()), op, start,
              limit, stride));
        }
        // Serialize: route the first operand + the prior chunk's output through
        // an optimization_barrier so this dispatch depends on the previous one
        // (avoids concurrent flash dispatches, which trigger the simdgroup_matrix
        // race). Only when chunking (serialDep set).
        if (serialDep) {
          auto bar = bld.create<mlir::stablehlo::OptimizationBarrierOp>(
              loc, ValueRange{ops[0], serialDep});
          ops[0] = bar->getResult(0);
        }
        SmallVector<Type> rt;
        for (auto t : resTys) {
          SmallVector<int64_t> sh(t.getShape().begin(), t.getShape().end());
          sh[0] = cn;
          rt.push_back(RankedTensorType::get(sh, t.getElementType()));
        }
        auto cc = bld.create<mlir::stablehlo::CustomCallOp>(loc, rt, ops);
        cc.setCallTargetName(target);
        for (size_t i = 0; i < resTys.size(); ++i)
          resChunks[i].push_back(cc.getResult(i));
        serialDep = cc.getResult(0);
      }
      SmallVector<Value> out;
      for (size_t i = 0; i < resTys.size(); ++i) {
        if (resChunks[i].size() == 1)
          out.push_back(resChunks[i][0]);
        else
          out.push_back(bld.create<mlir::stablehlo::ConcatenateOp>(
              loc, resTys[i], resChunks[i], 0));
      }
      return out;
    };

    // Forward flash (inserted at `av`; Q/K/V come from early values).
    Value Q3 = to3d(b, Q), K3 = to3d(b, K), V3 = to3d(b, V);
    SmallVector<Value> fwd =
        chunkedFlash(b, "flash_attention_fwd", {Q3, K3, V3}, {n3Ty, lTy});
    Value Ofl = fwd[0], Lfl = fwd[1];
    av.getResult().replaceAllUsesWith(to4d(b, Ofl, avTy.getElementType()));

    if (hasBwd) {
      // Backward flash must be inserted AFTER dO is defined (dominance).
      OpBuilder bB(av);
      bB.setInsertionPointAfter(dO.getDefiningOp());
      Value dO3 = to3d(bB, dO);
      // D[n,s] = sum_d dO*O  as a batched dot (batch [0,1], contract [2]).
      auto ddn = mlir::stablehlo::DotDimensionNumbersAttr::get(
          fctx, {0, 1}, {0, 1}, {2}, {2});
      Value Dr = bB.create<mlir::stablehlo::DotGeneralOp>(
          loc, lTy, dO3, Ofl, ddn, ArrayAttr(),
          mlir::stablehlo::DotAlgorithmAttr());
      SmallVector<Value> dq = chunkedFlash(bB, "flash_attention_bwd_dq",
                                           {Q3, K3, V3, dO3, Lfl, Dr}, {n3Ty});
      SmallVector<Value> dkv =
          chunkedFlash(bB, "flash_attention_bwd_dkdv",
                       {Q3, K3, V3, dO3, Lfl, Dr}, {n3Ty, n3Ty});
      Type dqE = cast<RankedTensorType>(dQdg.getType()).getElementType();
      Type dkE = cast<RankedTensorType>(dKdg.getType()).getElementType();
      Type dvE = cast<RankedTensorType>(dVdg.getType()).getElementType();
      dQdg.getResult().replaceAllUsesWith(to4d(bB, dq[0], dqE));
      dKdg.getResult().replaceAllUsesWith(to4d(bB, dkv[0], dkE));
      // dVdg is [B,H,D,T]; flash dV is [B,H,T,D] -> transpose last two.
      Value dV4 = to4d(bB, dkv[1], dvE);
      Value dVt = bB.create<mlir::stablehlo::TransposeOp>(
          loc, dVdg.getType(), dV4, bB.getDenseI64ArrayAttr({0, 1, 3, 2}));
      dVdg.getResult().replaceAllUsesWith(dVt);
    }
  }
}

struct ConvertFlashAttentionDispatch final
    : impl::ConvertFlashAttentionDispatchBase<ConvertFlashAttentionDispatch> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Layer-1 compiler-native flash: raise attention patterns to flash custom_calls
    // BEFORE collecting/lowering custom_calls below (opt-in NLEARN_COOP_RAISE_FLASH).
    raiseFlashAttentionFwd(module);

    SmallVector<mlir::stablehlo::CustomCallOp> calls;
    module.walk([&](mlir::stablehlo::CustomCallOp op) {
      StringRef t = op.getCallTargetName();
      if (t == "flash_attention_fwd" || t == "flash_attention_bwd_dq" ||
          t == "flash_attention_bwd_dkdv" || t == "gemm" ||
          t == "ce_fwd" || t == "ce_bwd")
        calls.push_back(op);
    });
    if (calls.empty())
      return;

    SymbolTable symbolTable(module);
    for (auto op : calls) {
      StringRef target = op.getCallTargetName();

      // ----- Custom GEMM: A[M,K] @ B[K,N] -> C[M,N] -----
      if (target == "gemm") {
        auto aTy = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
        auto bTy = dyn_cast<RankedTensorType>(op.getOperand(1).getType());
        if (!aTy || !bTy || aTy.getRank() != 2 || bTy.getRank() != 2) {
          op.emitError("gemm custom_call expects 2D A[M,K] and B[K,N]");
          return signalPassFailure();
        }
        int64_t M = aTy.getDimSize(0), K = aTy.getDimSize(1),
                N = bTy.getDimSize(1);
        std::string suffix = llvm::formatv("gemm_{0}x{1}x{2}", M, N, K).str();
        std::string fn = "__" + suffix, exe = "__exe_" + suffix;
        if (!symbolTable.lookup(fn)) {
          const char *envPath = std::getenv("NLEARN_GEMM_KERNEL_PATH");
          StringRef kernelPath =
              envPath ? StringRef(envPath) : StringRef("gemm.metal");
          std::string wrapperStr = buildGemmWrapper(exe, fn, M, N, K, kernelPath);
          auto wrapperMod = parseSourceString<ModuleOp>(wrapperStr, ctx);
          if (!wrapperMod) {
            op.emitError("failed to parse gemm wrapper module");
            return signalPassFailure();
          }
          for (Operation &top : llvm::make_early_inc_range(
                   wrapperMod->getBody()->getOperations())) {
            Operation *cloned = top.clone();
            module.getBody()->push_back(cloned);
            if (auto sym = dyn_cast<SymbolOpInterface>(cloned))
              symbolTable.insert(cloned);
          }
        }
        OpBuilder b(op);
        auto call = b.create<func::CallOp>(op.getLoc(), fn, op.getResultTypes(),
                                           op.getOperands());
        op.replaceAllUsesWith(call.getResults());
        op.erase();
        continue;
      }

      // ----- Fused cross-entropy: ce_fwd / ce_bwd over logits[M,V] -----
      if (target == "ce_fwd" || target == "ce_bwd") {
        auto lTy = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
        if (!lTy || lTy.getRank() != 2) {
          op.emitError("ce custom_call expects 2D logits[M,V]");
          return signalPassFailure();
        }
        int64_t M = lTy.getDimSize(0), V = lTy.getDimSize(1);
        std::string suffix = llvm::formatv("{0}_{1}x{2}", target, M, V).str();
        std::string fn = "__" + suffix, exe = "__exe_" + suffix;
        if (!symbolTable.lookup(fn)) {
          const char *envPath = std::getenv("NLEARN_CE_KERNEL_PATH");
          StringRef kernelPath =
              envPath ? StringRef(envPath) : StringRef("cross_entropy.metal");
          std::string wrapperStr = buildCeWrapper(target, exe, fn, M, V, kernelPath);
          auto wrapperMod = parseSourceString<ModuleOp>(wrapperStr, ctx);
          if (!wrapperMod) {
            op.emitError("failed to parse ce wrapper module");
            return signalPassFailure();
          }
          for (Operation &top : llvm::make_early_inc_range(
                   wrapperMod->getBody()->getOperations())) {
            Operation *cloned = top.clone();
            module.getBody()->push_back(cloned);
            if (auto sym = dyn_cast<SymbolOpInterface>(cloned))
              symbolTable.insert(cloned);
          }
        }
        OpBuilder b(op);
        auto call = b.create<func::CallOp>(op.getLoc(), fn, op.getResultTypes(),
                                           op.getOperands());
        op.replaceAllUsesWith(call.getResults());
        op.erase();
        continue;
      }

      auto qTy = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
      if (!qTy || qTy.getRank() != 3) {
        op.emitError("flash_attention custom_call expects 3D (n,s,d) operand 0");
        return signalPassFailure();
      }
      int64_t n = qTy.getDimSize(0), s = qTy.getDimSize(1), d = qTy.getDimSize(2);
      std::string suffix = llvm::formatv("{0}_{1}x{2}x{3}", target, n, s, d).str();
      std::string fn = "__" + suffix;
      std::string exe = "__exe_" + suffix;

      if (!symbolTable.lookup(fn)) {
        // Absolute path to the hand-authored MSL kernel object. Read from the
        // environment so it isn't baked into the compiler; findFileInPaths
        // resolves absolute paths directly (no --iree-hal-executable-object-
        // search-path needed, which the PJRT plugin's flag parser rejects).
        const char *envPath = std::getenv("NLEARN_FLASH_KERNEL_PATH");
        StringRef kernelPath = envPath ? StringRef(envPath)
                                       : StringRef("flash_attention.metal");
        std::string wrapperStr =
            buildWrapperModule(target, exe, fn, n, s, d, kernelPath);
        auto wrapperMod = parseSourceString<ModuleOp>(wrapperStr, ctx);
        if (!wrapperMod) {
          op.emitError("failed to parse flash wrapper module");
          return signalPassFailure();
        }
        // Clone the wrapper's top-level ops (executable + func) into the module.
        for (Operation &top :
             llvm::make_early_inc_range(wrapperMod->getBody()->getOperations())) {
          Operation *cloned = top.clone();
          module.getBody()->push_back(cloned);
          if (auto sym = dyn_cast<SymbolOpInterface>(cloned))
            symbolTable.insert(cloned);
        }
      }

      OpBuilder b(op);
      auto call = b.create<func::CallOp>(op.getLoc(), fn, op.getResultTypes(),
                                         op.getOperands());
      op.replaceAllUsesWith(call.getResults());
      op.erase();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::stablehlo
