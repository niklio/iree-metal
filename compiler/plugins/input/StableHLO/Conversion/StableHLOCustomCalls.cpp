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
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/BroadcastUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <utility>

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

// Raises the exact canonical StableHLO spellings emitted by JAX for causal or
// unmasked encoder attention and their VJPs as one transaction. This
// deliberately does not share the legacy external-Metal flash gate below:
// these ops stay in IREE and use the normal Attention/AttentionBackward
// lowering paths.
static bool hasI64Values(ArrayRef<int64_t> actual,
                         std::initializer_list<int64_t> expected) {
  return actual.size() == expected.size() &&
         std::equal(actual.begin(), actual.end(), expected.begin());
}

static bool hasStaticTensorType(Value value,
                                std::initializer_list<int64_t> shape,
                                Type elementType = {}) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && type.getRank() == shape.size() &&
         std::equal(type.getShape().begin(), type.getShape().end(),
                    shape.begin()) &&
         (!elementType || type.getElementType() == elementType);
}

static bool hasBroadcastDimensions(mlir::stablehlo::BroadcastInDimOp op,
                                   std::initializer_list<int64_t> expected) {
  return op && hasI64Values(op.getBroadcastDimensions(), expected);
}

static bool hasTransposePermutation(mlir::stablehlo::TransposeOp op,
                                    std::initializer_list<int64_t> expected) {
  return op && hasI64Values(op.getPermutation(), expected);
}

static bool hasDefaultDotSemantics(mlir::stablehlo::DotGeneralOp op) {
  if (op.getAlgorithm()) {
    return false;
  }
  auto precision = op.getPrecisionConfig();
  if (!precision || precision->empty()) {
    return true;
  }
  return llvm::all_of(*precision, [](Attribute attr) {
    auto precisionAttr = dyn_cast<mlir::stablehlo::PrecisionAttr>(attr);
    return precisionAttr &&
           precisionAttr.getValue() == mlir::stablehlo::Precision::DEFAULT;
  });
}

static bool hasDotDimensions(mlir::stablehlo::DotGeneralOp op,
                             std::initializer_list<int64_t> lhsBatch,
                             std::initializer_list<int64_t> rhsBatch,
                             std::initializer_list<int64_t> lhsContract,
                             std::initializer_list<int64_t> rhsContract) {
  if (!op || !hasDefaultDotSemantics(op)) {
    return false;
  }
  auto dims = op.getDotDimensionNumbers();
  return hasI64Values(dims.getLhsBatchingDimensions(), lhsBatch) &&
         hasI64Values(dims.getRhsBatchingDimensions(), rhsBatch) &&
         hasI64Values(dims.getLhsContractingDimensions(), lhsContract) &&
         hasI64Values(dims.getRhsContractingDimensions(), rhsContract);
}

static FloatAttr getSplatFloatAttr(Value value) {
  Attribute constant;
  if (!matchPattern(value, m_Constant(&constant))) {
    return {};
  }
  if (auto scalar = dyn_cast<FloatAttr>(constant)) {
    return scalar;
  }
  auto splat = dyn_cast<SplatElementsAttr>(constant);
  if (!splat) {
    return {};
  }
  return dyn_cast<FloatAttr>(splat.getSplatValue<Attribute>());
}

static BoolAttr getSplatBoolAttr(Value value) {
  Attribute constant;
  if (!matchPattern(value, m_Constant(&constant))) {
    return {};
  }
  if (auto scalar = dyn_cast<BoolAttr>(constant)) {
    return scalar;
  }
  auto splat = dyn_cast<SplatElementsAttr>(constant);
  if (!splat) {
    return {};
  }
  return dyn_cast<BoolAttr>(splat.getSplatValue<Attribute>());
}

static bool isFloatSplat(Value value, double expected) {
  FloatAttr attr = getSplatFloatAttr(value);
  return attr && attr.getValueAsDouble() == expected;
}

static bool isNegativeInfinitySplat(Value value) {
  FloatAttr attr = getSplatFloatAttr(value);
  return attr && attr.getValue().isInfinity() && attr.getValue().isNegative();
}

static bool isLowestFiniteSplat(Value value) {
  FloatAttr attr = getSplatFloatAttr(value);
  if (!attr) {
    return false;
  }
  llvm::APFloat expected = llvm::APFloat::getLargest(
      attr.getValue().getSemantics(), /*Negative=*/true);
  return attr.getValue().bitwiseIsEqual(expected);
}

enum class ReduceCombiner { kAdd, kMaximum };

static bool matchesUnaryReduce(mlir::stablehlo::ReduceOp reduce, Value input,
                               Value init,
                               std::initializer_list<int64_t> dimensions,
                               ReduceCombiner combinerKind) {
  if (!reduce || reduce.getInputs().size() != 1 ||
      reduce.getInitValues().size() != 1 ||
      reduce.getInputs().front() != input ||
      reduce.getInitValues().front() != init ||
      !hasI64Values(reduce.getDimensions(), dimensions) ||
      !llvm::hasSingleElement(reduce.getBody())) {
    return false;
  }
  Block &block = reduce.getBody().front();
  if (block.getNumArguments() != 2 ||
      std::distance(block.begin(), block.end()) != 2) {
    return false;
  }
  Operation &combiner = block.front();
  bool rightCombiner = combinerKind == ReduceCombiner::kAdd
                           ? isa<mlir::stablehlo::AddOp>(combiner)
                           : isa<mlir::stablehlo::MaxOp>(combiner);
  if (!rightCombiner || combiner.getNumOperands() != 2 ||
      combiner.getNumResults() != 1) {
    return false;
  }
  bool consumesArgs = (combiner.getOperand(0) == block.getArgument(0) &&
                       combiner.getOperand(1) == block.getArgument(1)) ||
                      (combiner.getOperand(0) == block.getArgument(1) &&
                       combiner.getOperand(1) == block.getArgument(0));
  auto returnOp = dyn_cast<mlir::stablehlo::ReturnOp>(block.back());
  return consumesArgs && returnOp && returnOp.getNumOperands() == 1 &&
         returnOp.getOperand(0) == combiner.getResult(0);
}

template <typename OpTy>
static Value getOtherBinaryOperand(OpTy op, Value known) {
  if (!op) {
    return {};
  }
  if (op.getLhs() == known) {
    return op.getRhs();
  }
  if (op.getRhs() == known) {
    return op.getLhs();
  }
  return {};
}

struct PairedAttentionMatch {
  mlir::stablehlo::DotGeneralOp outputDot;
  mlir::stablehlo::DotGeneralOp outputGradRoot;
  mlir::stablehlo::TransposeOp valueGradLeaf;
  mlir::stablehlo::DotGeneralOp queryGradLeaf;
  mlir::stablehlo::TransposeOp keyGradLeaf;
  Value query;
  Value key;
  Value value;
  Value outputGrad;
  Value mask;
  FloatAttr scale;
  int64_t paddedSequence;
};

static std::optional<PairedAttentionMatch>
matchPairedAttention(mlir::stablehlo::DotGeneralOp outputDot) {
  auto outputType = dyn_cast<RankedTensorType>(outputDot.getType());
  if (!outputType || outputType.getRank() != 4 ||
      !isa<BFloat16Type>(outputType.getElementType()) ||
      !hasDotDimensions(outputDot, {0, 1}, {0, 1}, {3}, {2})) {
    return std::nullopt;
  }

  auto probabilityDiv =
      outputDot.getLhs().getDefiningOp<mlir::stablehlo::DivOp>();
  Value value = outputDot.getRhs();
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  auto probabilityType =
      probabilityDiv ? dyn_cast<RankedTensorType>(probabilityDiv.getType())
                     : RankedTensorType();
  if (!probabilityDiv || !valueType || !probabilityType ||
      valueType.getRank() != 4 || probabilityType.getRank() != 4) {
    return std::nullopt;
  }

  auto exponential =
      probabilityDiv.getLhs().getDefiningOp<mlir::stablehlo::ExpOp>();
  auto centered = exponential
                      ? exponential.getOperand()
                            .getDefiningOp<mlir::stablehlo::SubtractOp>()
                      : mlir::stablehlo::SubtractOp();
  auto denominatorBroadcast =
      probabilityDiv.getRhs()
          .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  auto denominatorReshape =
      denominatorBroadcast ? denominatorBroadcast.getOperand()
                                 .getDefiningOp<mlir::stablehlo::ReshapeOp>()
                           : mlir::stablehlo::ReshapeOp();
  auto denominatorConvert =
      denominatorReshape ? denominatorReshape.getOperand()
                               .getDefiningOp<mlir::stablehlo::ConvertOp>()
                         : mlir::stablehlo::ConvertOp();
  auto denominatorReduce = denominatorConvert
                               ? denominatorConvert.getOperand()
                                     .getDefiningOp<mlir::stablehlo::ReduceOp>()
                               : mlir::stablehlo::ReduceOp();
  auto exponentialConvert =
      denominatorReduce ? denominatorReduce.getInputs()
                              .front()
                              .getDefiningOp<mlir::stablehlo::ConvertOp>()
                        : mlir::stablehlo::ConvertOp();
  if (!centered || !denominatorReduce || !exponentialConvert ||
      exponentialConvert.getOperand() != exponential.getResult() ||
      !isFloatSplat(denominatorReduce.getInitValues().front(), 0.0) ||
      !matchesUnaryReduce(denominatorReduce, exponentialConvert.getResult(),
                          denominatorReduce.getInitValues().front(), {3},
                          ReduceCombiner::kAdd) ||
      !hasBroadcastDimensions(denominatorBroadcast, {0, 1, 2, 3})) {
    return std::nullopt;
  }

  Value maskedScores = centered.getLhs();
  auto rowMaxBroadcast =
      centered.getRhs().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  auto clampedRowMax =
      rowMaxBroadcast
          ? rowMaxBroadcast.getOperand().getDefiningOp<mlir::stablehlo::MaxOp>()
          : mlir::stablehlo::MaxOp();
  if (!clampedRowMax || !hasBroadcastDimensions(rowMaxBroadcast, {0, 1, 2})) {
    return std::nullopt;
  }
  auto rowMaxReduce =
      clampedRowMax.getLhs().getDefiningOp<mlir::stablehlo::ReduceOp>();
  Value rowMaxClamp = clampedRowMax.getRhs();
  if (!rowMaxReduce) {
    rowMaxReduce =
        clampedRowMax.getRhs().getDefiningOp<mlir::stablehlo::ReduceOp>();
    rowMaxClamp = clampedRowMax.getLhs();
  }
  if (!rowMaxReduce || !isNegativeInfinitySplat(rowMaxClamp) ||
      !isNegativeInfinitySplat(rowMaxReduce.getInitValues().front()) ||
      !matchesUnaryReduce(rowMaxReduce, maskedScores,
                          rowMaxReduce.getInitValues().front(), {3},
                          ReduceCombiner::kMaximum)) {
    return std::nullopt;
  }

  Value mask;
  mlir::stablehlo::BroadcastInDimOp maskBroadcast;
  mlir::stablehlo::MulOp scaledScores;
  if (auto maskedSelect =
          maskedScores.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    maskBroadcast =
        maskedSelect.getPred()
            .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    auto causalMask =
        maskBroadcast
            ? maskBroadcast.getOperand()
                  .getDefiningOp<mlir::stablehlo::SelectOp>()
            : mlir::stablehlo::SelectOp();
    auto causalCompare =
        causalMask
            ? causalMask.getPred().getDefiningOp<mlir::stablehlo::CompareOp>()
            : mlir::stablehlo::CompareOp();
    auto rowIota =
        causalCompare
            ? causalCompare.getLhs().getDefiningOp<mlir::stablehlo::IotaOp>()
            : mlir::stablehlo::IotaOp();
    auto columnIota =
        causalCompare
            ? causalCompare.getRhs().getDefiningOp<mlir::stablehlo::IotaOp>()
            : mlir::stablehlo::IotaOp();
    if (!maskBroadcast || !causalMask || !causalCompare || !rowIota ||
        !columnIota || !hasBroadcastDimensions(maskBroadcast, {2, 3}) ||
        causalCompare.getComparisonDirection() !=
            mlir::stablehlo::ComparisonDirection::GE ||
        causalCompare.getCompareType().value_or(
            mlir::stablehlo::ComparisonType::NOTYPE) !=
            mlir::stablehlo::ComparisonType::SIGNED ||
        rowIota.getIotaDimension() != 0 ||
        columnIota.getIotaDimension() != 1) {
      return std::nullopt;
    }
    BoolAttr causalTrue = getSplatBoolAttr(causalMask.getOnTrue());
    BoolAttr causalFalse = getSplatBoolAttr(causalMask.getOnFalse());
    if (!causalTrue || !causalTrue.getValue() || !causalFalse ||
        causalFalse.getValue() ||
        !isLowestFiniteSplat(maskedSelect.getOnFalse())) {
      return std::nullopt;
    }
    mask = causalMask.getResult();
    scaledScores =
        maskedSelect.getOnTrue().getDefiningOp<mlir::stablehlo::MulOp>();
  } else {
    // The only mask-free form accepted here is the exact encoder spelling:
    // softmax(scale * dot(Q, K^T)). Additive masks and other score transforms
    // deliberately stay on the portable path.
    scaledScores =
        maskedScores.getDefiningOp<mlir::stablehlo::MulOp>();
  }
  auto queryKeyDot =
      scaledScores
          ? scaledScores.getLhs().getDefiningOp<mlir::stablehlo::DotGeneralOp>()
          : mlir::stablehlo::DotGeneralOp();
  Value forwardScale = scaledScores ? scaledScores.getRhs() : Value();
  if (!queryKeyDot && scaledScores) {
    queryKeyDot =
        scaledScores.getRhs().getDefiningOp<mlir::stablehlo::DotGeneralOp>();
    forwardScale = scaledScores.getLhs();
  }
  FloatAttr scale = getSplatFloatAttr(forwardScale);
  auto keyTranspose =
      queryKeyDot
          ? queryKeyDot.getRhs().getDefiningOp<mlir::stablehlo::TransposeOp>()
          : mlir::stablehlo::TransposeOp();
  Value query = queryKeyDot ? queryKeyDot.getLhs() : Value();
  Value key = keyTranspose ? keyTranspose.getOperand() : Value();
  auto queryType =
      query ? dyn_cast<RankedTensorType>(query.getType()) : RankedTensorType();
  auto keyType =
      key ? dyn_cast<RankedTensorType>(key.getType()) : RankedTensorType();
  if (!queryKeyDot || !keyTranspose || !scale ||
      !hasDotDimensions(queryKeyDot, {0, 1}, {0, 1}, {3}, {2}) ||
      !hasTransposePermutation(keyTranspose, {0, 1, 3, 2}) || !queryType ||
      !keyType || queryType != keyType || queryType != valueType ||
      queryType != outputType) {
    return std::nullopt;
  }

  int64_t batch = queryType.getDimSize(0);
  int64_t heads = queryType.getDimSize(1);
  int64_t sequence = queryType.getDimSize(2);
  int64_t headDim = queryType.getDimSize(3);
  Type bf16 = queryType.getElementType();
  Type f32 = Float32Type::get(outputDot.getContext());
  Type i1 = IntegerType::get(outputDot.getContext(), 1);
  // The paired raise is only enabled by the native Metal pipeline, whose
  // forward aggregate has no portable codegen fallback. Preserve the original
  // StableHLO graph unless both contractions fit the default Apple 8x8x8
  // simdgroup inventory. Backward role tagging independently verifies all five
  // of its materialized contractions after target selection.
  int64_t paddedSequence = sequence;
  if (!mask) {
    // The standard ViT-base sequence is one element beyond the aligned native
    // path. Its paired rewrite pads to the best measured physical length and
    // masks the added keys. The environment override is also used by focused
    // tests and schedule sweeps; zero disables the automatic padding of an
    // otherwise unaligned sequence.
    int64_t requestedSequence = sequence == 577 ? 608 : sequence;
    if (const char *value = std::getenv("IREE_METAL_ATTN_PAD_SEQUENCE")) {
      char *end = nullptr;
      long parsed = std::strtol(value, &end, 10);
      if (end != value && *end == '\0') {
        requestedSequence = parsed;
      }
    }
    if (requestedSequence >= sequence && requestedSequence % 8 == 0 &&
        requestedSequence - sequence <= 128) {
      paddedSequence = requestedSequence;
    }
  }
  if (!bf16.isBF16() || sequence < 8 || paddedSequence % 8 != 0 ||
      headDim < 8 || headDim % 8 != 0) {
    return std::nullopt;
  }
  // Broadcast dimensions alone do not prove that the denominator is a row
  // value: e.g. [B,H,1,S] can also broadcast to the score shape. Pin every
  // conversion and reshape so the reduction remains row-wise.
  if (!queryType.hasStaticShape() ||
      !hasStaticTensorType(probabilityDiv.getResult(),
                           {batch, heads, sequence, sequence}, bf16) ||
      !hasStaticTensorType(exponentialConvert.getResult(),
                           {batch, heads, sequence, sequence}, f32) ||
      !hasStaticTensorType(denominatorReduce.getResult(0),
                           {batch, heads, sequence}, f32) ||
      !hasStaticTensorType(denominatorConvert.getResult(),
                           {batch, heads, sequence}, bf16) ||
      !hasStaticTensorType(denominatorReshape.getResult(),
                           {batch, heads, sequence, 1}, bf16) ||
      !hasStaticTensorType(denominatorBroadcast.getResult(),
                           {batch, heads, sequence, sequence}, bf16) ||
      !hasStaticTensorType(value, {batch, heads, sequence, headDim}, bf16) ||
      !hasStaticTensorType(outputDot.getResult(),
                           {batch, heads, sequence, headDim}, bf16)) {
    return std::nullopt;
  }
  if (mask &&
      (!hasStaticTensorType(maskBroadcast.getResult(),
                            {batch, heads, sequence, sequence}, i1) ||
       !hasStaticTensorType(mask, {sequence, sequence}, i1))) {
    return std::nullopt;
  }

  // Find dV = transpose(dot(dO, P)). P may only have this gradient dot in
  // addition to the forward output dot for this exact spelling.
  SmallVector<mlir::stablehlo::DotGeneralOp> valueGradRoots;
  for (Operation *user : probabilityDiv.getResult().getUsers()) {
    auto dot = dyn_cast<mlir::stablehlo::DotGeneralOp>(user);
    if (dot && dot.getRhs() == probabilityDiv.getResult() &&
        hasDotDimensions(dot, {0, 1}, {0, 1}, {2}, {2})) {
      valueGradRoots.push_back(dot);
    }
  }
  if (valueGradRoots.size() != 1) {
    return std::nullopt;
  }
  mlir::stablehlo::DotGeneralOp valueGradRoot = valueGradRoots.front();
  Value outputGrad = valueGradRoot.getLhs();
  if (!hasStaticTensorType(outputGrad, {batch, heads, sequence, headDim},
                           bf16) ||
      !valueGradRoot.getResult().hasOneUse()) {
    return std::nullopt;
  }
  auto valueGradLeaf =
      dyn_cast<mlir::stablehlo::TransposeOp>(*valueGradRoot->user_begin());
  if (!hasTransposePermutation(valueGradLeaf, {0, 1, 3, 2}) ||
      !hasStaticTensorType(valueGradLeaf.getResult(),
                           {batch, heads, sequence, headDim}, bf16)) {
    return std::nullopt;
  }
  if (!probabilityDiv.getResult().hasNUses(2) ||
      !llvm::all_of(probabilityDiv.getResult().getUsers(),
                    [&](Operation *user) {
                      return user == outputDot.getOperation() ||
                             user == valueGradRoot.getOperation();
                    })) {
    return std::nullopt;
  }

  // Find dP = dot(dO, V).
  SmallVector<mlir::stablehlo::DotGeneralOp> probabilityGradDots;
  for (Operation *user : outputGrad.getUsers()) {
    auto dot = dyn_cast<mlir::stablehlo::DotGeneralOp>(user);
    if (dot && dot.getLhs() == outputGrad && dot.getRhs() == value &&
        hasDotDimensions(dot, {0, 1}, {0, 1}, {3}, {3})) {
      probabilityGradDots.push_back(dot);
    }
  }
  if (probabilityGradDots.size() != 1) {
    return std::nullopt;
  }
  mlir::stablehlo::DotGeneralOp probabilityGradDot =
      probabilityGradDots.front();

  // Locate dQ from the shared K transpose, then prove its dS and the complete
  // dK double-transpose leaf.
  SmallVector<mlir::stablehlo::DotGeneralOp> queryGradDots;
  for (Operation *user : keyTranspose.getResult().getUsers()) {
    auto dot = dyn_cast<mlir::stablehlo::DotGeneralOp>(user);
    if (dot && dot != queryKeyDot && dot.getRhs() == keyTranspose.getResult() &&
        hasDotDimensions(dot, {0, 1}, {0, 1}, {3}, {3})) {
      queryGradDots.push_back(dot);
    }
  }
  if (queryGradDots.size() != 1) {
    return std::nullopt;
  }
  mlir::stablehlo::DotGeneralOp queryGradLeaf = queryGradDots.front();
  Value scoreGrad = queryGradLeaf.getLhs();
  if (!hasStaticTensorType(queryGradLeaf.getResult(),
                           {batch, heads, sequence, headDim}, bf16)) {
    return std::nullopt;
  }
  SmallVector<mlir::stablehlo::DotGeneralOp> keyGradDots;
  for (Operation *user : scoreGrad.getUsers()) {
    auto dot = dyn_cast<mlir::stablehlo::DotGeneralOp>(user);
    if (dot && dot.getLhs() == scoreGrad && dot.getRhs() == query &&
        hasDotDimensions(dot, {0, 1}, {0, 1}, {2}, {2})) {
      keyGradDots.push_back(dot);
    }
  }
  if (keyGradDots.size() != 1 || !keyGradDots.front().getResult().hasOneUse()) {
    return std::nullopt;
  }
  auto keyGradInnerTranspose = dyn_cast<mlir::stablehlo::TransposeOp>(
      *keyGradDots.front()->user_begin());
  if (!hasTransposePermutation(keyGradInnerTranspose, {0, 1, 3, 2}) ||
      !keyGradInnerTranspose.getResult().hasOneUse()) {
    return std::nullopt;
  }
  auto keyGradLeaf = dyn_cast<mlir::stablehlo::TransposeOp>(
      *keyGradInnerTranspose->user_begin());
  if (!hasTransposePermutation(keyGradLeaf, {0, 1, 3, 2}) ||
      !hasStaticTensorType(keyGradLeaf.getResult(),
                           {batch, heads, sequence, headDim}, bf16)) {
    return std::nullopt;
  }

  // dS = [where(mask,)] softmax-vjp * the exact same scale splat.
  auto scoreScaleMul = scoreGrad.getDefiningOp<mlir::stablehlo::MulOp>();
  Value maskedScoreGrad = scoreScaleMul ? scoreScaleMul.getLhs() : Value();
  Value backwardScale = scoreScaleMul ? scoreScaleMul.getRhs() : Value();
  if (!getSplatFloatAttr(backwardScale)) {
    std::swap(maskedScoreGrad, backwardScale);
  }
  FloatAttr backwardScaleAttr = getSplatFloatAttr(backwardScale);
  if (!scoreScaleMul || !backwardScaleAttr ||
      !scale.getValue().bitwiseIsEqual(backwardScaleAttr.getValue())) {
    return std::nullopt;
  }
  mlir::stablehlo::MulOp unmaskedScoreGrad;
  if (mask) {
    auto scoreMaskSelect =
        maskedScoreGrad
            ? maskedScoreGrad.getDefiningOp<mlir::stablehlo::SelectOp>()
            : mlir::stablehlo::SelectOp();
    if (!scoreMaskSelect ||
        scoreMaskSelect.getPred() != maskBroadcast.getResult() ||
        !isFloatSplat(scoreMaskSelect.getOnFalse(), 0.0)) {
      return std::nullopt;
    }
    unmaskedScoreGrad =
        scoreMaskSelect.getOnTrue().getDefiningOp<mlir::stablehlo::MulOp>();
  } else {
    unmaskedScoreGrad =
        maskedScoreGrad
            ? maskedScoreGrad.getDefiningOp<mlir::stablehlo::MulOp>()
            : mlir::stablehlo::MulOp();
  }
  Value softmaxVjp =
      getOtherBinaryOperand(unmaskedScoreGrad, exponential.getResult());
  auto softmaxVjpAdd = softmaxVjp
                           ? softmaxVjp.getDefiningOp<mlir::stablehlo::AddOp>()
                           : mlir::stablehlo::AddOp();
  if (!unmaskedScoreGrad || !softmaxVjpAdd) {
    return std::nullopt;
  }

  // One add operand is dP / denominator. The other is the correction term.
  auto directTerm =
      softmaxVjpAdd.getLhs().getDefiningOp<mlir::stablehlo::DivOp>();
  Value correction = softmaxVjpAdd.getRhs();
  if (!directTerm || directTerm.getLhs() != probabilityGradDot.getResult() ||
      directTerm.getRhs() != denominatorBroadcast.getResult()) {
    directTerm = softmaxVjpAdd.getRhs().getDefiningOp<mlir::stablehlo::DivOp>();
    correction = softmaxVjpAdd.getLhs();
  }
  if (!directTerm || directTerm.getLhs() != probabilityGradDot.getResult() ||
      directTerm.getRhs() != denominatorBroadcast.getResult()) {
    return std::nullopt;
  }

  auto correctionConvert =
      correction.getDefiningOp<mlir::stablehlo::ConvertOp>();
  auto correctionBroadcast =
      correctionConvert
          ? correctionConvert.getOperand()
                .getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()
          : mlir::stablehlo::BroadcastInDimOp();
  auto singletonReduce = correctionBroadcast
                             ? correctionBroadcast.getOperand()
                                   .getDefiningOp<mlir::stablehlo::ReduceOp>()
                             : mlir::stablehlo::ReduceOp();
  auto correctionReshape =
      singletonReduce ? singletonReduce.getInputs()
                            .front()
                            .getDefiningOp<mlir::stablehlo::ReshapeOp>()
                      : mlir::stablehlo::ReshapeOp();
  auto correctionToF32 = correctionReshape
                             ? correctionReshape.getOperand()
                                   .getDefiningOp<mlir::stablehlo::ConvertOp>()
                             : mlir::stablehlo::ConvertOp();
  auto correctionNegate =
      correctionToF32
          ? correctionToF32.getOperand().getDefiningOp<mlir::stablehlo::NegOp>()
          : mlir::stablehlo::NegOp();
  auto weightedReduce = correctionNegate
                            ? correctionNegate.getOperand()
                                  .getDefiningOp<mlir::stablehlo::ReduceOp>()
                            : mlir::stablehlo::ReduceOp();
  // As above, prove the singleton is the trailing softmax dimension rather
  // than accepting a shape-compatible reshape that mixes query rows.
  if (!correctionConvert || !correctionBroadcast || !singletonReduce ||
      !correctionReshape || !correctionToF32 || !correctionNegate ||
      !weightedReduce ||
      !hasStaticTensorType(weightedReduce.getInputs().front(),
                           {batch, heads, sequence, sequence}, bf16) ||
      !hasStaticTensorType(weightedReduce.getResult(0),
                           {batch, heads, sequence}, bf16) ||
      !hasStaticTensorType(correctionNegate.getResult(),
                           {batch, heads, sequence}, bf16) ||
      !hasStaticTensorType(correctionToF32.getResult(),
                           {batch, heads, sequence}, f32) ||
      !hasStaticTensorType(correctionReshape.getResult(),
                           {batch, heads, sequence, 1}, f32) ||
      !hasStaticTensorType(singletonReduce.getResult(0),
                           {batch, heads, sequence}, f32) ||
      !hasStaticTensorType(correctionBroadcast.getResult(),
                           {batch, heads, sequence, sequence}, f32) ||
      !hasStaticTensorType(correctionConvert.getResult(),
                           {batch, heads, sequence, sequence}, bf16) ||
      !hasBroadcastDimensions(correctionBroadcast, {0, 1, 2}) ||
      !isFloatSplat(singletonReduce.getInitValues().front(), 0.0) ||
      !matchesUnaryReduce(singletonReduce, correctionReshape.getResult(),
                          singletonReduce.getInitValues().front(), {3},
                          ReduceCombiner::kAdd) ||
      !isFloatSplat(weightedReduce.getInitValues().front(), 0.0) ||
      !matchesUnaryReduce(weightedReduce, weightedReduce.getInputs().front(),
                          weightedReduce.getInitValues().front(), {3},
                          ReduceCombiner::kAdd)) {
    return std::nullopt;
  }
  auto weightedExp = weightedReduce.getInputs()
                         .front()
                         .getDefiningOp<mlir::stablehlo::MulOp>();
  Value weightedProbabilityGrad =
      getOtherBinaryOperand(weightedExp, exponential.getResult());
  auto probabilityGradTimesInverse =
      weightedProbabilityGrad
          ? weightedProbabilityGrad.getDefiningOp<mlir::stablehlo::MulOp>()
          : mlir::stablehlo::MulOp();
  Value inverseBroadcast = getOtherBinaryOperand(
      probabilityGradTimesInverse, probabilityGradDot.getResult());
  auto inverseBroadcastOp =
      inverseBroadcast
          ? inverseBroadcast.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()
          : mlir::stablehlo::BroadcastInDimOp();
  auto inverseDiv = inverseBroadcastOp
                        ? inverseBroadcastOp.getOperand()
                              .getDefiningOp<mlir::stablehlo::DivOp>()
                        : mlir::stablehlo::DivOp();
  auto denominatorSquared =
      inverseDiv ? inverseDiv.getRhs().getDefiningOp<mlir::stablehlo::MulOp>()
                 : mlir::stablehlo::MulOp();
  if (!weightedExp || !probabilityGradTimesInverse || !inverseBroadcastOp ||
      !inverseDiv || !denominatorSquared ||
      !hasBroadcastDimensions(inverseBroadcastOp, {0, 1, 2, 3}) ||
      !isFloatSplat(inverseDiv.getLhs(), 1.0) ||
      denominatorSquared.getLhs() != denominatorReshape.getResult() ||
      denominatorSquared.getRhs() != denominatorReshape.getResult()) {
    return std::nullopt;
  }

  // The backward must be insertable after dO and before the first old gradient
  // root. Keeping all roots in one block rejects accidental cyclic placement.
  Block *block = outputDot->getBlock();
  SmallVector<Operation *> orderedOps = {outputDot,
                                         valueGradRoot,
                                         valueGradLeaf,
                                         probabilityGradDot,
                                         queryGradLeaf,
                                         keyGradDots.front(),
                                         keyGradInnerTranspose,
                                         keyGradLeaf};
  if (llvm::any_of(
          orderedOps,
          [block](Operation *op) { return op->getBlock() != block; }) ||
      !outputDot->isBeforeInBlock(valueGradRoot) ||
      !valueGradRoot->isBeforeInBlock(probabilityGradDot) ||
      !valueGradRoot->isBeforeInBlock(queryGradLeaf) ||
      !valueGradRoot->isBeforeInBlock(keyGradDots.front())) {
    return std::nullopt;
  }
  if (Operation *outputGradDef = outputGrad.getDefiningOp()) {
    if (outputGradDef->getBlock() != block ||
        !outputGradDef->isBeforeInBlock(valueGradRoot)) {
      return std::nullopt;
    }
  }

  return PairedAttentionMatch{outputDot,     valueGradRoot,
                              valueGradLeaf, queryGradLeaf,
                              keyGradLeaf,   query,
                              key,           value,
                              outputGrad,    mask,
                              scale,         paddedSequence};
}

static Value padAttentionSequence(OpBuilder &builder, Location loc, Value value,
                                  int64_t paddedSequence) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> paddedShape(type.getShape());
  paddedShape[2] = paddedSequence;
  auto paddedType =
      RankedTensorType::get(paddedShape, type.getElementType());
  Value zero = arith::ConstantOp::create(
      builder, loc, builder.getZeroAttr(type.getElementType()));
  return tensor::createPadHighOp(paddedType, value, zero, /*nofold=*/false,
                                 loc, builder);
}

static Value sliceAttentionSequence(OpBuilder &builder, Location loc,
                                    Value value, RankedTensorType resultType) {
  int64_t rank = resultType.getRank();
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes =
      llvm::map_to_vector(resultType.getShape(), [&](int64_t size) {
        return OpFoldResult(builder.getIndexAttr(size));
      });
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  return tensor::ExtractSliceOp::create(builder, loc, resultType, value,
                                        offsets, sizes, strides);
}

static Value createPaddedKeyMask(OpBuilder &builder, Location loc,
                                 int64_t sequence, int64_t paddedSequence) {
  auto maskType =
      RankedTensorType::get({paddedSequence}, builder.getI1Type());
  SmallVector<llvm::APInt> maskValues;
  maskValues.reserve(paddedSequence);
  for (int64_t index = 0; index < paddedSequence; ++index) {
    maskValues.emplace_back(/*numBits=*/1, index < sequence);
  }
  auto maskAttr = DenseIntElementsAttr::get(maskType, maskValues);
  return arith::ConstantOp::create(builder, loc, maskType, maskAttr);
}

static void rewritePairedAttention(PairedAttentionMatch match) {
  Location loc = match.outputDot.getLoc();
  MLIRContext *context = match.outputDot.getContext();
  auto queryType = cast<RankedTensorType>(match.query.getType());
  auto keyType = cast<RankedTensorType>(match.key.getType());
  auto valueType = cast<RankedTensorType>(match.value.getType());
  auto outputType = cast<RankedTensorType>(match.outputDot.getType());
  int64_t batch = queryType.getDimSize(0);
  int64_t heads = queryType.getDimSize(1);
  int64_t sequence = queryType.getDimSize(2);
  int64_t originalSequence = sequence;
  bool padSequence = match.paddedSequence != sequence;

  OpBuilder forwardBuilder(match.outputDot);
  Value query = match.query;
  Value key = match.key;
  Value value = match.value;
  Value mask = match.mask;
  if (padSequence) {
    query = padAttentionSequence(forwardBuilder, loc, query,
                                 match.paddedSequence);
    key =
        padAttentionSequence(forwardBuilder, loc, key, match.paddedSequence);
    value =
        padAttentionSequence(forwardBuilder, loc, value, match.paddedSequence);
    mask = createPaddedKeyMask(forwardBuilder, loc, originalSequence,
                               match.paddedSequence);
    queryType = cast<RankedTensorType>(query.getType());
    keyType = cast<RankedTensorType>(key.getType());
    valueType = cast<RankedTensorType>(value.getType());
    outputType = outputType.clone(queryType.getShape());
    sequence = match.paddedSequence;
  }
  Value scale = arith::ConstantOp::create(forwardBuilder, loc, match.scale);
  Value outputInit = tensor::EmptyOp::create(
      forwardBuilder, loc, outputType.getShape(), outputType.getElementType());
  auto logsumexpType = RankedTensorType::get({batch, heads, sequence},
                                             forwardBuilder.getF32Type());
  Value logsumexpInit =
      tensor::EmptyOp::create(forwardBuilder, loc, logsumexpType.getShape(),
                              logsumexpType.getElementType());

  AffineExpr b = forwardBuilder.getAffineDimExpr(0);
  AffineExpr h = forwardBuilder.getAffineDimExpr(1);
  AffineExpr m = forwardBuilder.getAffineDimExpr(2);
  AffineExpr n = forwardBuilder.getAffineDimExpr(3);
  AffineExpr k1 = forwardBuilder.getAffineDimExpr(4);
  AffineExpr k2 = forwardBuilder.getAffineDimExpr(5);
  auto map = [&](std::initializer_list<AffineExpr> results) {
    SmallVector<AffineExpr> resultVector(results);
    return AffineMap::get(/*dimCount=*/6, /*symbolCount=*/0, resultVector,
                          context);
  };
  AffineMap queryMap = map({b, h, m, k1});
  AffineMap keyMap = map({b, h, k2, k1});
  AffineMap valueMap = map({b, h, k2, n});
  AffineMap outputMap = map({b, h, m, n});
  AffineMap logsumexpMap = map({b, h, m});
  AffineMap scaleMap = map({});
  AffineMap maskMap = padSequence ? map({k2}) : map({m, k2});
  SmallVector<AffineMap> forwardMaps = {queryMap, keyMap, valueMap, scaleMap};
  if (mask) {
    forwardMaps.push_back(maskMap);
  }
  forwardMaps.append({outputMap, logsumexpMap});
  DictionaryAttr decompositionConfig =
      forwardBuilder.getDictionaryAttr({forwardBuilder.getNamedAttr(
          IREE::LinalgExt::AttentionOp::getUseExp2AttrStr(),
          forwardBuilder.getBoolAttr(false))});
  DictionaryAttr backwardDecompositionConfig = decompositionConfig;
  if (match.mask) {
    // `match.mask` is present only after the exact lower-triangular StableHLO
    // matcher above has proved row_iota >= column_iota. Do not use the local
    // `mask`: padded unmasked encoder attention creates a key-validity mask
    // later and is not causal.
    NamedAttrList backwardConfig(decompositionConfig);
    for (StringRef role :
         {IREE::LinalgExt::AttentionBackwardOp::getDQAttrStr(),
          IREE::LinalgExt::AttentionBackwardOp::getDKAttrStr(),
          IREE::LinalgExt::AttentionBackwardOp::getDVAttrStr()}) {
      backwardConfig.set(
          role,
          forwardBuilder.getDictionaryAttr({forwardBuilder.getNamedAttr(
              "iree_codegen.apple_attention_backward_causal",
              forwardBuilder.getUnitAttr())}));
    }
    backwardDecompositionConfig =
        backwardConfig.getDictionary(forwardBuilder.getContext());
  }
  SmallVector<Type> forwardResultTypes = {outputType, logsumexpType};
  auto attention = IREE::LinalgExt::AttentionOp::create(
      forwardBuilder, loc, forwardResultTypes, query, key, value, scale, mask,
      outputInit, logsumexpInit,
      forwardBuilder.getAffineMapArrayAttr(forwardMaps), decompositionConfig);
  {
    OpBuilder::InsertionGuard guard(forwardBuilder);
    Block *body = forwardBuilder.createBlock(&attention.getRegion());
    body->addArgument(forwardBuilder.getF32Type(), loc);
    forwardBuilder.setInsertionPointToEnd(body);
    IREE::LinalgExt::YieldOp::create(forwardBuilder, loc, body->getArgument(0));
  }
  Value attentionOutput = attention.getResult(0);
  Value attentionLogsumexp = attention.getResult(1);
  if (padSequence) {
    auto attentionBarrier = IREE::Util::OptimizationBarrierOp::create(
        forwardBuilder, loc,
        ValueRange{attentionOutput, attentionLogsumexp});
    attentionOutput = attentionBarrier.getResult(0);
    attentionLogsumexp = attentionBarrier.getResult(1);
  }

  // Insert after dO is available and before the earliest old gradient root.
  OpBuilder backwardBuilder(match.outputGradRoot);
  Value outputGrad = match.outputGrad;
  if (padSequence) {
    outputGrad = padAttentionSequence(backwardBuilder, loc, outputGrad,
                                      match.paddedSequence);
    outputGrad = IREE::Util::OptimizationBarrierOp::create(
                     backwardBuilder, loc, ValueRange{outputGrad})
                     .getResult(0);
  }
  Value queryGradInit = tensor::EmptyOp::create(
      backwardBuilder, loc, queryType.getShape(), queryType.getElementType());
  Value keyGradInit = tensor::EmptyOp::create(
      backwardBuilder, loc, keyType.getShape(), keyType.getElementType());
  Value valueGradInit = tensor::EmptyOp::create(
      backwardBuilder, loc, valueType.getShape(), valueType.getElementType());
  SmallVector<AffineMap> backwardMaps = {
      queryMap, keyMap,  valueMap, outputMap,
      outputMap, logsumexpMap, scaleMap};
  if (mask) {
    backwardMaps.push_back(maskMap);
  }
  backwardMaps.append({queryMap, keyMap, valueMap});
  SmallVector<Type> backwardResultTypes = {queryType, keyType, valueType};
  auto attentionBackward = IREE::LinalgExt::AttentionBackwardOp::create(
      backwardBuilder, loc, backwardResultTypes, query, key, value,
      attentionOutput, outputGrad, attentionLogsumexp, scale, mask,
      queryGradInit,
      keyGradInit, valueGradInit,
      backwardBuilder.getAffineMapArrayAttr(backwardMaps),
      backwardDecompositionConfig);

  // Mutate only the externally visible leaves, and only after both operations
  // have been constructed. Replace the forward result last.
  Value queryGrad = attentionBackward.getResult(0);
  Value keyGrad = attentionBackward.getResult(1);
  Value valueGrad = attentionBackward.getResult(2);
  Value output = attentionOutput;
  if (padSequence) {
    auto gradientBarrier = IREE::Util::OptimizationBarrierOp::create(
        backwardBuilder, loc, ValueRange{queryGrad, keyGrad, valueGrad});
    queryGrad = gradientBarrier.getResult(0);
    keyGrad = gradientBarrier.getResult(1);
    valueGrad = gradientBarrier.getResult(2);
    queryGrad = sliceAttentionSequence(
        backwardBuilder, loc, queryGrad,
        cast<RankedTensorType>(match.queryGradLeaf.getType()));
    keyGrad = sliceAttentionSequence(
        backwardBuilder, loc, keyGrad,
        cast<RankedTensorType>(match.keyGradLeaf.getType()));
    valueGrad = sliceAttentionSequence(
        backwardBuilder, loc, valueGrad,
        cast<RankedTensorType>(match.valueGradLeaf.getType()));
    output = sliceAttentionSequence(
        forwardBuilder, loc, output,
        cast<RankedTensorType>(match.outputDot.getType()));
  }
  match.queryGradLeaf.getResult().replaceAllUsesWith(queryGrad);
  match.keyGradLeaf.getResult().replaceAllUsesWith(keyGrad);
  match.valueGradLeaf.getResult().replaceAllUsesWith(valueGrad);
  match.outputDot.getResult().replaceAllUsesWith(output);
}

// The canonical JAX tanh-GELU VJP retains the full forward elementwise chain
// until the backward is available. On Metal that forces large rank-3 BF16
// intermediates to survive across the intervening FFN contraction. Recognize
// the complete canonical forward/VJP pair and rebuild the derivative behind an
// optimization barrier. This deliberately trades a second small elementwise
// chain (including tanh) for avoiding the saved forward intermediates.
//
// Keep this matcher intentionally exact. Apart from guarding correctness, the
// strict spelling makes the environment variable a safe experiment: a JAX or
// StableHLO canonicalization change simply leaves the original graph intact.
struct PairedTanhGeluMatch {
  Value input;
  Value outputGrad;
  Value inputGrad;
  Value one;
  Value half;
  Value three;
  Value cubicCoefficient;
  Value tanhScale;
  Operation *outputGradDef;
};

static bool isTypedFloatSplat(Value value, Type type, double expected) {
  return value.getType() == type && isFloatSplat(value, expected);
}

template <typename OpTy, typename Predicate>
static OpTy findUniqueBinaryUser(Value known, Predicate predicate) {
  OpTy found;
  for (Operation *user : known.getUsers()) {
    auto op = dyn_cast<OpTy>(user);
    Value other = getOtherBinaryOperand(op, known);
    if (!other || !predicate(other)) {
      continue;
    }
    if (found) {
      return OpTy();
    }
    found = op;
  }
  return found;
}

static bool hasExactlyUsers(Value value,
                            std::initializer_list<Operation *> expected) {
  SmallVector<Operation *> remaining(expected);
  for (OpOperand &use : value.getUses()) {
    auto it = llvm::find(remaining, use.getOwner());
    if (it == remaining.end()) {
      return false;
    }
    remaining.erase(it);
  }
  return remaining.empty();
}

static std::optional<PairedTanhGeluMatch>
matchPairedTanhGelu(mlir::stablehlo::TanhOp tanh) {
  auto type = dyn_cast<RankedTensorType>(tanh.getType());
  if (!type || type.getRank() != 3 ||
      !isa<BFloat16Type>(type.getElementType())) {
    return std::nullopt;
  }

  // Forward: 0.5*x*(1+tanh(0.796875*(x+0.044677734375*x^3))).
  auto tanhArgument = tanh.getOperand().getDefiningOp<mlir::stablehlo::MulOp>();
  if (!tanhArgument) {
    return std::nullopt;
  }
  Value tanhScale;
  Value innerValue;
  if (isTypedFloatSplat(tanhArgument.getLhs(), type, 0.796875)) {
    tanhScale = tanhArgument.getLhs();
    innerValue = tanhArgument.getRhs();
  } else if (isTypedFloatSplat(tanhArgument.getRhs(), type, 0.796875)) {
    tanhScale = tanhArgument.getRhs();
    innerValue = tanhArgument.getLhs();
  } else {
    return std::nullopt;
  }

  auto inner = innerValue.getDefiningOp<mlir::stablehlo::AddOp>();
  if (!inner) {
    return std::nullopt;
  }
  mlir::stablehlo::MulOp cubicTerm;
  Value input;
  Value cubicCoefficient;
  Value inputCubed;
  for (auto [possibleInput, possibleCubic] :
       {std::pair<Value, Value>{inner.getLhs(), inner.getRhs()},
        std::pair<Value, Value>{inner.getRhs(), inner.getLhs()}}) {
    auto multiply = possibleCubic.getDefiningOp<mlir::stablehlo::MulOp>();
    if (!multiply) {
      continue;
    }
    Value coefficient;
    Value cubed;
    if (isTypedFloatSplat(multiply.getLhs(), type, 0.044677734375)) {
      coefficient = multiply.getLhs();
      cubed = multiply.getRhs();
    } else if (isTypedFloatSplat(multiply.getRhs(), type, 0.044677734375)) {
      coefficient = multiply.getRhs();
      cubed = multiply.getLhs();
    } else {
      continue;
    }
    if (cubicTerm) {
      return std::nullopt;
    }
    cubicTerm = multiply;
    input = possibleInput;
    cubicCoefficient = coefficient;
    inputCubed = cubed;
  }
  if (!cubicTerm || input.getType() != type) {
    return std::nullopt;
  }
  auto inputCubedOp = inputCubed.getDefiningOp<mlir::stablehlo::MulOp>();
  Value inputSquared = getOtherBinaryOperand(inputCubedOp, input);
  auto inputSquaredOp =
      inputSquared ? inputSquared.getDefiningOp<mlir::stablehlo::MulOp>()
                   : mlir::stablehlo::MulOp();
  if (!inputCubedOp || !inputSquaredOp || inputSquaredOp.getLhs() != input ||
      inputSquaredOp.getRhs() != input) {
    return std::nullopt;
  }

  auto threeInputSquared = findUniqueBinaryUser<mlir::stablehlo::MulOp>(
      inputSquared,
      [&](Value other) { return isTypedFloatSplat(other, type, 3.0); });
  Value three = getOtherBinaryOperand(threeInputSquared, inputSquared);
  auto halfInput = findUniqueBinaryUser<mlir::stablehlo::MulOp>(
      input, [&](Value other) { return isTypedFloatSplat(other, type, 0.5); });
  Value half = getOtherBinaryOperand(halfInput, input);
  if (!threeInputSquared || !three || !halfInput || !half) {
    return std::nullopt;
  }

  mlir::stablehlo::SubtractOp oneMinusTanh;
  for (Operation *user : tanh.getResult().getUsers()) {
    auto subtract = dyn_cast<mlir::stablehlo::SubtractOp>(user);
    if (!subtract || subtract.getRhs() != tanh.getResult() ||
        !isTypedFloatSplat(subtract.getLhs(), type, 1.0)) {
      continue;
    }
    if (oneMinusTanh) {
      return std::nullopt;
    }
    oneMinusTanh = subtract;
  }
  auto onePlusTanh = findUniqueBinaryUser<mlir::stablehlo::AddOp>(
      tanh.getResult(),
      [&](Value other) { return isTypedFloatSplat(other, type, 1.0); });
  Value one = getOtherBinaryOperand(onePlusTanh, tanh.getResult());
  if (!oneMinusTanh || !onePlusTanh || !one || oneMinusTanh.getLhs() != one) {
    return std::nullopt;
  }
  auto output = findUniqueBinaryUser<mlir::stablehlo::MulOp>(
      onePlusTanh.getResult(),
      [&](Value other) { return other == halfInput.getResult(); });
  if (!output) {
    return std::nullopt;
  }

  // Reverse: match the exact canonical VJP, including its deliberately
  // factored (1-tanh)*(1+tanh) spelling and reuse of 3*x^2.
  mlir::stablehlo::MulOp halfInputTimesGrad;
  Value outputGrad;
  auto tanhDerivativeLeft = findUniqueBinaryUser<mlir::stablehlo::MulOp>(
      oneMinusTanh.getResult(), [&](Value other) {
        auto multiply = other.getDefiningOp<mlir::stablehlo::MulOp>();
        Value possibleGrad =
            getOtherBinaryOperand(multiply, halfInput.getResult());
        if (!multiply || !possibleGrad || possibleGrad.getType() != type) {
          return false;
        }
        halfInputTimesGrad = multiply;
        outputGrad = possibleGrad;
        return true;
      });
  if (!tanhDerivativeLeft || !halfInputTimesGrad || !outputGrad) {
    return std::nullopt;
  }
  auto gradTimesOnePlus = findUniqueBinaryUser<mlir::stablehlo::MulOp>(
      onePlusTanh.getResult(),
      [&](Value other) { return other == outputGrad; });
  auto tanhDerivativeRight = findUniqueBinaryUser<mlir::stablehlo::MulOp>(
      tanhDerivativeLeft.getResult(),
      [&](Value other) { return other == tanh.getResult(); });
  if (!gradTimesOnePlus || !tanhDerivativeRight) {
    return std::nullopt;
  }
  auto tanhDerivative = findUniqueBinaryUser<mlir::stablehlo::AddOp>(
      tanhDerivativeLeft.getResult(),
      [&](Value other) { return other == tanhDerivativeRight.getResult(); });
  if (!tanhDerivative) {
    return std::nullopt;
  }
  auto scaledTanhDerivative = findUniqueBinaryUser<mlir::stablehlo::MulOp>(
      tanhDerivative.getResult(),
      [&](Value other) { return other == tanhScale; });
  if (!scaledTanhDerivative) {
    return std::nullopt;
  }
  auto scaledCubicDerivative = findUniqueBinaryUser<mlir::stablehlo::MulOp>(
      scaledTanhDerivative.getResult(),
      [&](Value other) { return other == cubicCoefficient; });
  if (!scaledCubicDerivative) {
    return std::nullopt;
  }
  auto cubicDerivative = findUniqueBinaryUser<mlir::stablehlo::MulOp>(
      scaledCubicDerivative.getResult(),
      [&](Value other) { return other == threeInputSquared.getResult(); });
  if (!cubicDerivative) {
    return std::nullopt;
  }
  auto innerDerivative = findUniqueBinaryUser<mlir::stablehlo::AddOp>(
      scaledTanhDerivative.getResult(),
      [&](Value other) { return other == cubicDerivative.getResult(); });
  auto linearDerivative = findUniqueBinaryUser<mlir::stablehlo::MulOp>(
      gradTimesOnePlus.getResult(), [&](Value other) { return other == half; });
  if (!innerDerivative || !linearDerivative) {
    return std::nullopt;
  }
  auto inputGrad = findUniqueBinaryUser<mlir::stablehlo::AddOp>(
      innerDerivative.getResult(),
      [&](Value other) { return other == linearDerivative.getResult(); });
  if (!inputGrad || inputGrad.getResult().use_empty()) {
    return std::nullopt;
  }

  // Prove that every retained forward intermediate is used only by this pair.
  // Besides making false positives harder, this guarantees that canonical DCE
  // can remove the old VJP uses so the forward values stop crossing the FFN
  // contraction boundary.
  if (!hasExactlyUsers(inputSquared, {inputCubedOp, threeInputSquared}) ||
      !hasExactlyUsers(inputCubed, {cubicTerm}) ||
      !hasExactlyUsers(cubicTerm.getResult(), {inner}) ||
      !hasExactlyUsers(inner.getResult(), {tanhArgument}) ||
      !hasExactlyUsers(tanhArgument.getResult(), {tanh}) ||
      !hasExactlyUsers(halfInput.getResult(), {output, halfInputTimesGrad}) ||
      !hasExactlyUsers(tanh.getResult(),
                       {oneMinusTanh, onePlusTanh, tanhDerivativeRight}) ||
      !hasExactlyUsers(oneMinusTanh.getResult(), {tanhDerivativeLeft}) ||
      !hasExactlyUsers(onePlusTanh.getResult(), {output, gradTimesOnePlus}) ||
      !hasExactlyUsers(threeInputSquared.getResult(), {cubicDerivative}) ||
      !hasExactlyUsers(halfInputTimesGrad.getResult(), {tanhDerivativeLeft}) ||
      !hasExactlyUsers(tanhDerivativeLeft.getResult(),
                       {tanhDerivativeRight, tanhDerivative}) ||
      !hasExactlyUsers(tanhDerivativeRight.getResult(), {tanhDerivative}) ||
      !hasExactlyUsers(tanhDerivative.getResult(), {scaledTanhDerivative}) ||
      !hasExactlyUsers(scaledTanhDerivative.getResult(),
                       {scaledCubicDerivative, innerDerivative}) ||
      !hasExactlyUsers(scaledCubicDerivative.getResult(), {cubicDerivative}) ||
      !hasExactlyUsers(cubicDerivative.getResult(), {innerDerivative}) ||
      !hasExactlyUsers(gradTimesOnePlus.getResult(), {linearDerivative}) ||
      !hasExactlyUsers(innerDerivative.getResult(), {inputGrad}) ||
      !hasExactlyUsers(linearDerivative.getResult(), {inputGrad}) ||
      output.getResult().use_empty()) {
    return std::nullopt;
  }

  Operation *outputGradDef = outputGrad.getDefiningOp();
  Block *block = tanh->getBlock();
  SmallVector<Operation *> matchedOps = {tanhArgument,
                                         inner,
                                         cubicTerm,
                                         inputCubedOp,
                                         inputSquaredOp,
                                         threeInputSquared,
                                         halfInput,
                                         oneMinusTanh,
                                         onePlusTanh,
                                         output,
                                         halfInputTimesGrad,
                                         tanhDerivativeLeft,
                                         gradTimesOnePlus,
                                         tanhDerivativeRight,
                                         tanhDerivative,
                                         scaledTanhDerivative,
                                         scaledCubicDerivative,
                                         cubicDerivative,
                                         innerDerivative,
                                         linearDerivative,
                                         inputGrad};
  if (!outputGradDef || outputGradDef->getBlock() != block ||
      llvm::any_of(
          matchedOps,
          [block](Operation *op) { return op->getBlock() != block; }) ||
      !output->isBeforeInBlock(outputGradDef) ||
      !outputGradDef->isBeforeInBlock(inputGrad)) {
    return std::nullopt;
  }

  return PairedTanhGeluMatch{
      input,        outputGrad, inputGrad.getResult(), one,
      half,         three,      cubicCoefficient,      tanhScale,
      outputGradDef};
}

static void rewritePairedTanhGelu(PairedTanhGeluMatch match) {
  OpBuilder builder(match.outputGradDef);
  builder.setInsertionPointAfter(match.outputGradDef);
  Location loc = match.inputGrad.getLoc();
  auto barrier = mlir::stablehlo::OptimizationBarrierOp::create(
      builder, loc, ValueRange{match.input, match.outputGrad});
  Value input = barrier->getResult(0);
  Value outputGrad = barrier->getResult(1);

  // Reproduce JAX's rematerialized computation and VJP operation order. Keep
  // the two x*x operations distinct here; the canonicalizer may CSE them after
  // both have been placed on the backward side of the barrier.
  Value halfInput =
      mlir::stablehlo::MulOp::create(builder, loc, match.half, input);
  Value inputSquaredForCube =
      mlir::stablehlo::MulOp::create(builder, loc, input, input);
  Value inputCubed =
      mlir::stablehlo::MulOp::create(builder, loc, inputSquaredForCube, input);
  Value inputSquaredForDerivative =
      mlir::stablehlo::MulOp::create(builder, loc, input, input);
  Value threeInputSquared = mlir::stablehlo::MulOp::create(
      builder, loc, match.three, inputSquaredForDerivative);
  Value cubicTerm = mlir::stablehlo::MulOp::create(
      builder, loc, match.cubicCoefficient, inputCubed);
  Value inner = mlir::stablehlo::AddOp::create(builder, loc, input, cubicTerm);
  Value tanhArgument =
      mlir::stablehlo::MulOp::create(builder, loc, match.tanhScale, inner);
  Value tanh = mlir::stablehlo::TanhOp::create(builder, loc, tanhArgument);
  Value oneMinusTanh =
      mlir::stablehlo::SubtractOp::create(builder, loc, match.one, tanh);
  Value onePlusTanh =
      mlir::stablehlo::AddOp::create(builder, loc, match.one, tanh);
  Value halfInputTimesGrad =
      mlir::stablehlo::MulOp::create(builder, loc, halfInput, outputGrad);
  Value tanhDerivativeLeft = mlir::stablehlo::MulOp::create(
      builder, loc, halfInputTimesGrad, oneMinusTanh);
  Value tanhDerivativeRight =
      mlir::stablehlo::MulOp::create(builder, loc, tanhDerivativeLeft, tanh);
  Value tanhDerivative = mlir::stablehlo::AddOp::create(
      builder, loc, tanhDerivativeLeft, tanhDerivativeRight);
  Value scaledTanhDerivative = mlir::stablehlo::MulOp::create(
      builder, loc, match.tanhScale, tanhDerivative);
  Value scaledCubicDerivative = mlir::stablehlo::MulOp::create(
      builder, loc, match.cubicCoefficient, scaledTanhDerivative);
  Value cubicDerivative = mlir::stablehlo::MulOp::create(
      builder, loc, scaledCubicDerivative, threeInputSquared);
  Value innerDerivative = mlir::stablehlo::AddOp::create(
      builder, loc, scaledTanhDerivative, cubicDerivative);
  Value gradTimesOnePlus =
      mlir::stablehlo::MulOp::create(builder, loc, outputGrad, onePlusTanh);
  Value linearDerivative = mlir::stablehlo::MulOp::create(
      builder, loc, match.half, gradTimesOnePlus);
  Value inputGrad = mlir::stablehlo::AddOp::create(
      builder, loc, innerDerivative, linearDerivative);
  match.inputGrad.replaceAllUsesWith(inputGrad);
}

static void rematerializePairedTanhGelu(ModuleOp module) {
  const char *value = std::getenv("IREE_METAL_GELU_REMAT");
  if (!value || StringRef(value) != "1") {
    return;
  }
  SmallVector<mlir::stablehlo::TanhOp> candidates;
  module.walk(
      [&](mlir::stablehlo::TanhOp tanh) { candidates.push_back(tanh); });
  for (mlir::stablehlo::TanhOp candidate : candidates) {
    std::optional<PairedTanhGeluMatch> match = matchPairedTanhGelu(candidate);
    if (match) {
      rewritePairedTanhGelu(*match);
    }
  }
}

static void raisePairedAttention(ModuleOp module) {
  SmallVector<mlir::stablehlo::DotGeneralOp> candidates;
  module.walk(
      [&](mlir::stablehlo::DotGeneralOp dot) { candidates.push_back(dot); });
  for (mlir::stablehlo::DotGeneralOp candidate : candidates) {
    if (candidate->use_empty()) {
      continue;
    }
    std::optional<PairedAttentionMatch> match = matchPairedAttention(candidate);
    if (match) {
      rewritePairedAttention(*match);
    }
  }
}

// Compiler-native flash (Layer 1, FORWARD). Recognize the attention subgraph
//   dot_general(softmax(scale * dot_general(Q, Kᵀ) [+ causal select]), V)
// and rewrite it to a `flash_attention_fwd` custom_call (which the loop below
// lowers to the flash flow.dispatch) — automatic, NO shim. Opt-in via
// IREE_METAL_COOP_RAISE_FLASH. Layer-1 targets causal + head_dim=64 (kernel scope);
// forward-only, so it's for inference until the Layer-3 backward matcher pairs it.
static void raiseFlashAttentionFwd(ModuleOp module) {
  if (!std::getenv("IREE_METAL_COOP_RAISE_FLASH"))
    return;
  SmallVector<mlir::stablehlo::DotGeneralOp> cands;
  module.walk([&](mlir::stablehlo::DotGeneralOp dg) { cands.push_back(dg); });
  for (auto av : cands) {
    if (av->use_empty()) {
      continue;
    }
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
    if (std::getenv("IREE_METAL_RAISE_BWD_DEBUG"))
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
    // chunkHeads * ceil(T/64), so the default is T-adaptive. IREE_METAL_COOP_FLASH_CHUNK
    // overrides (pass >=N to force a single dispatch for debugging).
    constexpr int64_t kTgBudget = 384;
    int64_t qTiles = std::max<int64_t>(1, (T + 63) / 64);
    int64_t chunkHeads =
        std::min<int64_t>(N, std::max<int64_t>(1, kTgBudget / qTiles));
    if (const char *cs = std::getenv("IREE_METAL_COOP_FLASH_CHUNK"))
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
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, IREE::LinalgExt::IREELinalgExtDialect,
                    IREE::Util::UtilDialect,
                    mlir::stablehlo::StablehloDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    rematerializePairedTanhGelu(module);

    // The native paired raise and the legacy external-Metal forward raise are
    // intentionally mutually exclusive. An explicitly requested native pass
    // remains authoritative; the rollback controls pipeline construction.
    // Both explicit flash custom calls and any custom calls created by the
    // legacy path are still lowered below.
    if (raiseNativeAttention) {
      raisePairedAttention(module);
    }
    if (!raiseNativeAttention && !suppressLegacyAttentionRaise) {
      raiseFlashAttentionFwd(module);
    }

    SmallVector<mlir::stablehlo::CustomCallOp> calls;
    module.walk([&](mlir::stablehlo::CustomCallOp op) {
      StringRef t = op.getCallTargetName();
      if (t == "flash_attention_fwd" || t == "flash_attention_bwd_dq" ||
          t == "flash_attention_bwd_dkdv" || t == "gemm" || t == "ce_fwd" ||
          t == "ce_bwd") {
        calls.push_back(op);
      }
    });
    if (calls.empty()) {
      return;
    }

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
          const char *envPath = std::getenv("IREE_METAL_GEMM_KERNEL_PATH");
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
          const char *envPath = std::getenv("IREE_METAL_CE_KERNEL_PATH");
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
        const char *envPath = std::getenv("IREE_METAL_FLASH_KERNEL_PATH");
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
