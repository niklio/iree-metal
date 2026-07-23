// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Passes.cpp - Pipelines from Linalg ops to SPIR-V -------------------===//
//
// This file contains various pipelines to lower IREE HAL executables containing
// Linalg ops to SPIR-V.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-spirv-lowering-pass-pipeline"

namespace mlir::iree_compiler {

static llvm::cl::opt<int> clSPIRVIndexingBits(
    "iree-spirv-index-bits",
    llvm::cl::desc("Set the bit width of indices in SPIR-V."),
    // nlearn: 64-bit indices needed for deep UNROLLED models (buffer offsets exceed 2^32, e.g. 4.5B at
    // 12L unrolled). Combined with --iree-dispatch-creation-fuse-multi-use=false (which fixes the grad-graph
    // miscompile), this unlocks full unrolling -> ~65% MFU (vs scan ~45%). 64-bit alone NaN'd only because
    // the fuse-multi-use miscompile was still present.
    llvm::cl::init(64));

//===----------------------------------------------------------------------===//
// Bufferization Configuration
//===----------------------------------------------------------------------===//

static FailureOr<Value> gpuAllocateWorkgroupMemoryFn(OpBuilder &builder,
                                                     Location loc,
                                                     MemRefType memRefType,
                                                     ValueRange dynamicSizes,
                                                     unsigned alignment) {
  auto workgroupSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  MemRefType allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), workgroupSpace);
  auto allocOp = memref::AllocOp::create(builder, loc, allocType, dynamicSizes,
                                         builder.getI64IntegerAttr(alignment));
  return allocOp.getResult();
}

static FailureOr<Value> gpuAllocateFunctionMemoryFn(OpBuilder &builder,
                                                    Location loc,
                                                    MemRefType memRefType,
                                                    ValueRange dynamicSizes,
                                                    unsigned alignment) {
  std::optional<unsigned> space =
      spirv::mapVulkanStorageClassToMemorySpace(spirv::StorageClass::Function);
  MemRefType allocType = MemRefType::get(
      memRefType.getShape(), memRefType.getElementType(), {}, *space);
  auto allocaOp =
      memref::AllocaOp::create(builder, loc, allocType, dynamicSizes,
                               builder.getI64IntegerAttr(alignment));
  return allocaOp.getResult();
}

static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  auto fromType = cast<MemRefType>(from.getType());
  auto toType = cast<MemRefType>(to.getType());

  bool needsBarrier = hasSharedMemoryAddressSpace(fromType) ||
                      hasSharedMemoryAddressSpace(toType);
  if (needsBarrier) {
    // We aren't using global memory for communication or destructively
    // overwriting it in a way that may be visible to other workitems (and,
    // commonly, we're copyng to or from workgroup memory and thas's what we
    // must synchronize on), so we use a local-memory-only barrier here.
    gpu::BarrierOp::create(builder, loc, gpu::AddressSpace::Workgroup);
  }
  Operation *copy = memref::CopyOp::create(builder, loc, from, to);
  if (needsBarrier) {
    setMarker(copy, getCopyToWorkgroupMemoryMarker());
    gpu::BarrierOp::create(builder, loc, gpu::AddressSpace::Workgroup);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

static void addTileAndDistributeToWorkgroupsPasses(
    OpPassManager &funcPassManager,
    bool useFuseTensorPadWithConsumerPass = false,
    bool useWARForCooperativeMatrixCodegen = false) {
  funcPassManager.addPass(createConvertAccGEMMToGEMMPass());
  funcPassManager.addPass(
      createTileAndDistributeToWorkgroupsUsingForallOpPass());
  funcPassManager.addPass(createFoldReshapeIntoInterfaceTensorPass());
  funcPassManager.addPass(createBufferizeDispatchTensorLoadStorePass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

/// Adds passes to lower vector ops to meet SPIR-V requirements.
void addSPIRVVectorLoweringPasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createSPIRVInitialVectorLoweringPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createSPIRVFinalVectorLoweringPass());
}

static void addBufferizePasses(OpPassManager &funcPassManager,
                               BufferizationOptions::AllocationFn fn) {
  BufferizationOptions::AllocationFn allocationFn = fn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn);
}

static void
addSPIRVBufferizePasses(OpPassManager &funcPassManager,
                        BufferizationOptions::AllocationFn allocationFn) {
  // Resolve dim ops first so that we don't have compute Linalg ops lingering on
  // because of dim op usage. This avoids bufferizing those compute ops just for
  // their shape dimensions.
  funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  addBufferizePasses(funcPassManager, allocationFn);
  // Distribute immediately after bufferization to avoid losing attribute
  // annotations in subsequent transformations. This is a bit fragile right now
  // but we expect upstream for loops to eventually recognize distribution as a
  // first-class attribute then we don't need this.
  funcPassManager.addPass(createGPUDistributeScfForPass());
  funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createCleanupBufferAllocViewPass());
}

/// Adds passes to materialize structured ops as loops. This replaces structured
/// ops with loop nests containing payloads, so it should be invoked after
/// tiling and vectorization and before buffer transformations.
static void addLoopMaterializationPasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(IREE::LinalgExt::createLinalgExtToLoopsPass());
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createConvertLinalgToLoopsPass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

/// Adds passes to lowering MemRefs. This folds MemRef subviews, flattens n-D
/// MemRef into 1-D ones, vectorizes load/store when possible, and performs
/// cross loop nest optimizations. This should be invoked after structured op
/// lowering and before final SPIR-V conversion.
static void addMemRefLoweringPasses(OpPassManager &modulePassManager) {
  // TODO: query this from the target.
  auto getIndexBitwidth = [](mlir::FunctionOpInterface) { return 32; };

  {
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass(createConvertComplexToStandardPass)
        // Math dialect ops rewrites, approximations, casts.
        .addPass(createMathTransformPass)
        .addPass(createPadDynamicAllocPass)
        .addPass(
            [&]() { return createGPUCheckResourceUsagePass(getIndexBitwidth); })

        // Fold load/store from/to subview ops into the original memref when
        // possible. In SPIR-V we don't use memref descriptor so it's not
        // possible to handle subview ops.
        .addPass(memref::createFoldMemRefAliasOpsPass)
        .addPass(createConvertUnsupportedFloatArithPass)
        .addPass(createEmulateNarrowTypePass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)

        // Turn scalar load/store from memrefs into vectorized ones if possible.
        // This gives better memory access patterns, which is very important for
        // perf.
        .addPass(createSPIRVVectorizeLoadStorePass)
        // Perform optimizations that need to across the scf.for region
        // boundary.
        .addPass(createForOpCanonicalizationPass)
        // Perform various vector-level cross-op optimizations like load-store
        // forwarding, shape casting and casting op cancelling.
        .addPass([&]() { return createOptimizeVectorTransferPass(); })
        .addPass(createSPIRVBreakDownLargeVectorPass)

        // Perform optimizations that need to across the scf.for region
        // boundary.
        .addPass(createForOpCanonicalizationPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass([&]() { return createOptimizeVectorTransferPass(); });
  }

  // Turn multi-dimension memref into one-dimension. This is needed for
  // SPIR-V because we don't use upstream memref descriptors.
  modulePassManager.addPass(createFlattenMemRefSubspanPass());

  FunctionLikeNest(modulePassManager)
      .addPass(createSPIRVEraseStorageBufferStaticShapePass)
      .addPass(createCSEPass);
}

/// Adds passes to perform the final SPIR-V conversion.
static void addSPIRVLoweringPasses(OpPassManager &modulePassManager) {
  FunctionLikeNest(modulePassManager)
      .addPass(createPropagateDispatchSizeBoundsPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPass(createIREECodegenLowerAffinePass)
      .addPass([]() {
        return IREE::Util::createOptimizeIntArithmeticPass(
            IREE::Util::OptimizeIntArithmeticPassOptions{/*narrowToI32=*/true});
      })

      // Lower ApplyScale before the i64 Emulation Pass so that new 64-bit ops
      // are also emulated if not supported by the target.
      .addPass([&]() {
        return createTosaToArithPass({/*includeApplyRescale=*/true,
                                      /*use32BitApplyRescale=*/true});
      })
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPass(createSPIRVMapMemRefStorageClassPass)
      .addPass(createSPIRVEmulateI64Pass)
      .addPass(createConvertBf16ArithToF32Pass)
      .addPass([]() {
        // Convert unsupported float buffer types to integer types.
        // SPIR-V doesn't natively support bf16 or fp8 types, so we convert
        // them to integer types of the same bit width for storage.
        return createConvertUnsupportedFloatToIntBuffersPass(
            ConvertUnsupportedFloatToIntBuffersPassOptions{
                /*includeBf16=*/true,
                /*includeF8E5M2=*/true,
                /*includeF8E4M3FN=*/true,
                /*includeF8E5M2FNUZ=*/true,
                /*includeF8E4M3FNUZ=*/true,
                /*includeF8E8M0FNU=*/true,
            });
      })
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  modulePassManager.addPass(createSPIRVConvertGPUTargetPass());
  modulePassManager.addPass(createConvertToSPIRVPass(clSPIRVIndexingBits));

  auto getTargetEnv = [](spirv::ModuleOp moduleOp) {
    return moduleOp->getParentOfType<mlir::ModuleOp>()
        ->getAttrOfType<spirv::TargetEnvAttr>(spirv::getTargetEnvAttrName());
  };

  OpPassManager &spirvModulePassManager =
      modulePassManager.nest<spirv::ModuleOp>();
  spirvModulePassManager.addPass(
      spirv::createUnifyAliasedResourcePass(getTargetEnv));
  spirvModulePassManager.addPass(spirv::createSPIRVLowerABIAttributesPass());
  spirvModulePassManager.addPass(createCanonicalizerPass());
  spirvModulePassManager.addPass(createCSEPass());
  spirvModulePassManager.addPass(spirv::createSPIRVRewriteInsertsPass());
  spirvModulePassManager.addPass(spirv::createSPIRVCanonicalizeGLPass());
  spirvModulePassManager.addPass(spirv::createSPIRVUpdateVCEPass());
}

//===----------------------------------------------------------------------===//
// Pass Pipelines
//===----------------------------------------------------------------------===//

void addSPIRVBaseLoweringPassPipeline(OpPassManager &funcPassManager) {
  addBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addLoopMaterializationPasses(funcPassManager);
}

void addSPIRVBaseDistributePassPipeline(OpPassManager &funcPassManager) {
  addTileAndDistributeToWorkgroupsPasses(funcPassManager);

  addBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);

  // Tile and distribute to GPU invocations.
  funcPassManager.addPass(createSPIRVTileAndDistributePass());
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addLoopMaterializationPasses(funcPassManager);
}

void addSPIRVBaseVectorizePassPipeline(OpPassManager &funcPassManager) {
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/true);

  funcPassManager.addPass(createFoldAffineMinInDistributedLoopsPass());
  funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());

  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Tile to GPU invocations and vectorize.
  funcPassManager.addPass(createGPUCreateFastSlowPathPass());
  funcPassManager.addPass(createGPUTilePass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  {
    GenericVectorizationPassOptions options;
    funcPassManager.addPass(createGenericVectorizationPass(options));
  }
  addSPIRVVectorLoweringPasses(funcPassManager);
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Bufferize and distribute.
  addSPIRVBufferizePasses(funcPassManager, gpuAllocateFunctionMemoryFn);

  // Generate loop nests for all remaining ops and remove trivial loops.
  addLoopMaterializationPasses(funcPassManager);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  funcPassManager.addPass(createOptimizeVectorTransferPass());
}

// nlearn: flash-attention. Mirrors the Winograd pipeline (decompose a linalg_ext
// aggregate op mid-pipeline, then generically vectorize) but for
// iree_linalg_ext.attention: convert to the online form, tile the reduction
// (K/scores) dim to serial loops so the softmax runs per-tile (scores stay
// un-materialized), decompose to matmul+softmax, then vectorize + SPIR-V lower.
// First increment: attention COMPILES on metal-spirv via this path (matmuls
// generic-vectorized); routing the per-tile matmuls onto coop is a follow-up.
namespace {
// nlearn: de-alias online-softmax SCRATCH writes to flash-loop iter_args so the
// tensor-level flash loop can one-shot-bufferize. Upstream one-shot-bufferize
// FAILS ("yield not equivalent to iter bbArg") when a loop's yielded value isn't
// buffer-equivalent to its iter_arg — the decomposed online-softmax reuses the
// max iter_arg for BOTH the yielded new-max (a reduction) AND the exp2-correction
// (an elementwise scratch write). For each flash-loop iter_arg, we give a fresh
// tensor.empty to every ELEMENTWISE (all-parallel) linalg op that writes that
// iter_arg as a DPS init but whose result does NOT feed that iter_arg's yielded
// value (i.e. scratch, not the accumulator chain). Full-overwrite elementwise
// ops ignore their init, so this is semantics-preserving and only removes the
// false buffer aliasing. Reduction accumulators (kept) still alias their iter_arg.
struct DeAliasFlashScratchPass
    : PassWrapper<DeAliasFlashScratchPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeAliasFlashScratchPass)
  StringRef getArgument() const final { return "nlearn-dealias-flash-scratch"; }
  void runOnOperation() override {
    getOperation()->walk([](scf::ForOp forOp) {
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (auto [idx, bbArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        // Backward-reachable set from this iter_arg's yielded value.
        llvm::SmallDenseSet<Value> reach;
        SmallVector<Value> wl{yield.getOperand(idx)};
        while (!wl.empty()) {
          Value v = wl.pop_back_val();
          if (!reach.insert(v).second)
            continue;
          if (Operation *def = v.getDefiningOp())
            for (Value o : def->getOperands())
              wl.push_back(o);
        }
        for (OpOperand &use : llvm::make_early_inc_range(bbArg.getUses())) {
          auto dps = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
          if (!dps || !dps.isDpsInit(&use))
            continue;
          auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
          if (!linalgOp || linalgOp.getNumReductionLoops() != 0)
            continue; // keep reduction accumulators aliased to the iter_arg
          if (reach.contains(use.getOwner()->getResult(0)))
            continue; // on the accumulator chain -> keep
          auto t = dyn_cast<RankedTensorType>(use.get().getType());
          if (!t || !t.hasStaticShape())
            continue;
          // Does the op READ its output init (read-modify, e.g. the online-
          // softmax correction exp2(m_old - m_new) reads m_old via the max
          // iter_arg)? If so, a fresh empty would feed garbage — instead give it
          // a COPY of the iter_arg captured at the LOOP-BODY START (before the
          // rowmax reduction overwrites the iter_arg's buffer), preserving m_old.
          bool readsInit = false;
          if (linalgOp->getNumRegions() == 1 &&
              !linalgOp->getRegion(0).empty()) {
            Block &body = linalgOp->getRegion(0).front();
            if (use.getOperandNumber() < body.getNumArguments())
              readsInit =
                  !body.getArgument(use.getOperandNumber()).use_empty();
          }
          Location loc = use.getOwner()->getLoc();
          Value newOut;
          if (readsInit) {
            OpBuilder cb(forOp.getBody(), forOp.getBody()->begin());
            Value dst = cb.create<tensor::EmptyOp>(loc, t.getShape(),
                                                   t.getElementType());
            newOut = cb.create<linalg::CopyOp>(loc, bbArg, dst).getResult(0);
          } else {
            OpBuilder b(use.getOwner());
            newOut =
                b.create<tensor::EmptyOp>(loc, t.getShape(), t.getElementType());
          }
          use.set(newOut);
        }
      }
    });
  }
};
} // namespace

namespace {
// nlearn: attach a THREAD lowering_config to the decomposed online-softmax
// generics (the non-matmul linalg ops — the qk/pv matmuls carry a coop
// decomposition_config already) so GPUApplyTilingLevel(Thread) distributes the
// M query-rows across the subgroup's threads. Viable in the shared-memory-staged
// path because the scores live in regular row-major smem (no coop layout needed
// for the softmax). Tiles the parallel loop of size == mBlock (the workgroup M
// tile) to 1 (one query-row per thread).
struct ConfigSoftmaxThreadsPass
    : PassWrapper<ConfigSoftmaxThreadsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConfigSoftmaxThreadsPass)
  int64_t mBlock = 1;
  StringRef getArgument() const final { return "nlearn-config-softmax-threads"; }
  void runOnOperation() override {
    int64_t m = getenv("NLEARN_COOP_ATTN_M") ? atoi(getenv("NLEARN_COOP_ATTN_M"))
                                             : mBlock;
    getOperation()->walk([&](linalg::LinalgOp op) {
      SmallVector<int64_t> ranges = op.getStaticLoopRanges();
      SmallVector<utils::IteratorType> iters = op.getIteratorTypesArray();
      SmallVector<int64_t> threadTile(ranges.size(), 0);
      bool found = false;
      for (auto [i, r] : llvm::enumerate(ranges)) {
        if (iters[i] == utils::IteratorType::parallel && r == m) {
          threadTile[i] = 1;
          found = true;
          break;
        }
      }
      if (!found)
        return;
      Builder b(op.getContext());
      auto dict = DictionaryAttr::get(
          op.getContext(),
          {b.getNamedAttr("thread", b.getI64ArrayAttr(threadTile))});
      setLoweringConfig(op,
                        IREE::GPU::LoweringConfigAttr::get(op.getContext(), dict));
    });
  }
};

// nlearn (NLEARN_COOP_ATTN_BARRIER): isolate the qk (scores) contraction from its
// softmax REDUCTION epilogue by inserting a util.optimization_barrier on its result.
// In decomposed attention the qk matmul feeds (through the parallel scale generic) a
// rowmax REDUCTION; the coop passes (SPIRVTileToCooperativeOps) INFINITE-LOOP on this
// matmul->reduction fusion (verified: hangs even on a trivial [64,64] single-tile).
// Standalone matmuls coop cleanly, so breaking the qk->reduction edge should let qk
// become a clean coop matmul (materialized scores) — mirrors the GlobalOpt
// SplitTransposeEpilogue/IsolateBatchMatmul barrier trick, but post-decompose in
// codegen. pv (clean parallel divide epilogue) needs no barrier.
static bool feedsReductionForward(Value v, int depth) {
  if (depth < 0)
    return false;
  for (Operation *user : v.getUsers()) {
    auto lin = dyn_cast<linalg::LinalgOp>(user);
    if (!lin)
      continue;
    // A non-matmul reduction consumer = the rowmax/rowsum softmax reduction.
    if (lin.getNumReductionLoops() != 0 &&
        !isa<linalg::ContractionOpInterface>(user))
      return true;
    // A parallel elementwise generic (e.g. the scale) — recurse through it.
    if (lin.getNumReductionLoops() == 0)
      for (Value r : user->getResults())
        if (feedsReductionForward(r, depth - 1))
          return true;
  }
  return false;
}

// nlearn (cont80d): fold arith.extf(f16->f32) on a vector.contract's A/B operands
// INTO the contract, so the decomposed attention qk/pv matmuls become f16xf16->f32
// (the Apple coop-native mixed-precision form) instead of f32xf32->f32 (which the
// coop intrinsic doesn't match -> falls to scalar). Numerically identical (coop
// accumulates in f32). This is the last blocker to coop firing in COOPSMEM.
// nlearn (cont83): true if `v` reaches a vector.multi_reduction/reduction forward
// (through elementwise ops) — i.e. it's the qk scores feeding the softmax.
static bool vectorFeedsReduction(Value v, int depth) {
  if (depth < 0)
    return false;
  for (Operation *user : v.getUsers()) {
    if (isa<vector::MultiDimReductionOp, vector::ReductionOp>(user))
      return true;
    if (user->getNumResults() == 1 &&
        !isa<vector::ContractionOp>(user) &&
        isa<VectorType>(user->getResult(0).getType()))
      if (vectorFeedsReduction(user->getResult(0), depth - 1))
        return true;
  }
  return false;
}

// nlearn (cont83, NLEARN_COOP_ATTN_VBARRIER): insert a VECTOR-level
// util.optimization_barrier on the qk vector.contract result (the one feeding the
// softmax reduction), right before SPIRVVectorToGPUSubgroupMMA. convertVectorToMMAOps
// won't propagate the COp mma type across the barrier, so the softmax stays as
// regular vector ops (off the matrix units) while qk/pv convert to coop separately —
// avoiding the invalid f16-accumulator coop op (cont81e/82b) without C-promotion.
struct VectorContractBarrierPass
    : PassWrapper<VectorContractBarrierPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorContractBarrierPass)
  StringRef getArgument() const final { return "nlearn-vector-contract-barrier"; }
  void runOnOperation() override {
    SmallVector<Operation *> qks;
    getOperation()->walk([&](vector::ContractionOp c) {
      if (vectorFeedsReduction(c.getResult(), /*depth=*/6))
        qks.push_back(c.getOperation());
    });
    for (Operation *op : qks) {
      Value res = op->getResult(0);
      if (!res.use_empty() &&
          isa<IREE::Util::OptimizationBarrierOp>(*res.getUsers().begin()))
        continue;
      OpBuilder b(op->getContext());
      b.setInsertionPointAfter(op);
      auto bar = IREE::Util::OptimizationBarrierOp::create(b, op->getLoc(), res);
      res.replaceAllUsesExcept(bar->getResult(0), bar);
    }
  }
};

struct FoldContractExtPass
    : PassWrapper<FoldContractExtPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldContractExtPass)
  StringRef getArgument() const final { return "nlearn-fold-contract-ext"; }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    vector::populateFoldArithExtensionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct QkScoreBarrierPass
    : PassWrapper<QkScoreBarrierPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QkScoreBarrierPass)
  StringRef getArgument() const final { return "nlearn-qk-score-barrier"; }
  void runOnOperation() override {
    SmallVector<Operation *> qks;
    getOperation()->walk([&](linalg::LinalgOp op) {
      if (!isa<linalg::ContractionOpInterface>(op.getOperation()))
        return;
      if (op->getNumResults() != 1)
        return;
      Value res = op->getResult(0);
      // Already isolated behind a barrier?
      if (!res.use_empty() &&
          isa<IREE::Util::OptimizationBarrierOp>(*res.getUsers().begin()))
        return;
      if (feedsReductionForward(res, /*depth=*/3))
        qks.push_back(op.getOperation());
    });
    for (Operation *op : qks) {
      Value res = op->getResult(0);
      OpBuilder b(op->getContext());
      b.setInsertionPointAfter(op);
      auto bar = IREE::Util::OptimizationBarrierOp::create(b, op->getLoc(), res);
      res.replaceAllUsesExcept(bar->getResult(0), bar);
    }
  }
};
} // namespace

void addSPIRVVectorDistributeAttentionPassPipeline(
    OpPassManager &funcPassManager) {
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/true);
  funcPassManager.addPass(createFoldAffineMinInDistributedLoopsPass());
  funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Decompose the attention op within codegen (post dispatch/config), so its
  // per-tile matmuls become plain linalg the rest of the pipeline lowers. Doing
  // this here (not as preprocessing) avoids dispatch-region-formation re-fusing
  // the pieces back into an unconfigured attention-shaped dispatch.
  funcPassManager.addPass(
      IREE::LinalgExt::createConvertAttentionToOnlineAttentionPass());
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Reduction;
    options.allowZeroSlices = true;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
  }

  // nlearn (NLEARN_COOP_ATTN_THREAD): distribute the query-M block across the
  // subgroup's threads by tiling the ATTENTION/online_attention op (which
  // carries the "thread" tiling level in its config and implements
  // TilingInterface) to M->1 BEFORE decompose. Each thread then decomposes to a
  // scalar M=1 per-query attention — no vector<M> to legalize, no coop-layout
  // interface, full occupancy. Doing this pre-decompose is the fix for cont57:
  // post-decompose the softmax generics carry NO config so the thread pass
  // couldn't reach them.
  if (getenv("NLEARN_COOP_ATTN_THREAD")) {
    GPUApplyTilingLevelPassOptions topts;
    topts.tilingLevel = IREE::GPU::TilingLevel::Thread;
    topts.allowZeroSlices = true;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(topts));
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  funcPassManager.addPass(IREE::LinalgExt::createDecomposeAttentionPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // nlearn (NLEARN_COOP_ATTN_SMEM): shared-memory-staged flash. Mirror the coop
  // MATMUL pipeline's order (bufferize-to-workgroup-mem -> promote C -> coop) for
  // the decomposed attention so the qk scores land in a REGULAR row-major SHARED-
  // MEMORY buffer (via the opaque coop store) and the inter-matmul softmax runs
  // as a normal smem reduction — NO coop layout, sidestepping the research-grade
  // Apple simdgroup-layout blocker that walls vector-distribute. Returns early to
  // skip the tensor-level vectorization + function-memory end-bufferize below.
  // nlearn (NLEARN_COOP_ATTN_COOPSMEM): the CORRECT-ORDER coop flash (cont78d root
  // cause: the DECOMP branch called the coop passes PRE-bufferize so they never
  // fired -> huge vector<64x64> contracts hung SPIRVInitialVectorLowering). Mirror
  // addSPIRVCooperativeMatrixVectorizePassPipeline's order — bufferize(smem) FIRST,
  // then TileAndPromote(promoteC) -> TileToCoop -> GenericVectorization ->
  // VectorizeToCoop — so the decomposed qk/pv matmuls become real simdgroup_matrix
  // coop ops (scores staged in smem via the workgroup allocator). The softmax
  // reductions between them are left to the normal vector lowering (they break down
  // fine; only the CONTRACTs hung, and those are now coop). GPUDistribute resolves
  // any thread foralls post-bufferize.
  if (getenv("NLEARN_COOP_ATTN_COOPSMEM")) {
    funcPassManager.addPass(std::make_unique<DeAliasFlashScratchPass>());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    // nlearn (cont82, NLEARN_COOP_ATTN_BARRIER): isolate the qk coop result from the
    // softmax so convertVectorToMMAOps doesn't propagate the COp mma type THROUGH the
    // softmax to the pv-A truncf (which makes an invalid f16-accumulator coop op,
    // cont81e). The barrier forces the softmax to read a regular vector, not a coop
    // mma_load, keeping the softmax off the matrix units while qk/pv stay coop.
    if (getenv("NLEARN_COOP_ATTN_BARRIER")) {
      funcPassManager.addPass(std::make_unique<QkScoreBarrierPass>());
      funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    }
    addBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);
    // C-promotion (doPromoteCMatrix) SEGFAULTS on the attention (the qk matmul's C
    // feeds the softmax reduction, not a plain store) — cont78g. Skip it by default
    // (coop can store to smem directly); opt back in with NLEARN_COOP_ATTN_PROMOTEC.
    funcPassManager.addPass(
        createSPIRVTileAndPromotePass(SPIRVTileAndPromotePassOptions{
            /*promoteCMatrix=*/getenv("NLEARN_COOP_ATTN_PROMOTEC") != nullptr,
            /*skipThreadLevel=*/true,
            /*skipOperandPromotion=*/true}));
    funcPassManager.addPass(createRemoveSingleIterationLoopPass());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    funcPassManager.addPass(createSPIRVTileToCooperativeOpsPass());
    funcPassManager.addPass(createRemoveSingleIterationLoopPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    {
      GenericVectorizationPassOptions options;
      funcPassManager.addPass(createGenericVectorizationPass(options));
    }
    funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
    // Fold extf(f16) on the contract operands so qk/pv are f16xf16->f32 = coop-native.
    funcPassManager.addPass(std::make_unique<FoldContractExtPass>());
    funcPassManager.addPass(createSPIRVVectorizeToCooperativeOpsPass());
    funcPassManager.addPass(createCSEPass());
    // Vector-level barrier on the qk scores so the softmax doesn't fuse into coop.
    if (getenv("NLEARN_COOP_ATTN_VBARRIER"))
      funcPassManager.addPass(std::make_unique<VectorContractBarrierPass>());
    // nlearn (cont81): convert the (f16-native) qk/pv vector.contract + its operand
    // loads/store to gpu.subgroup_mma ops NOW, before the vector lowering below —
    // otherwise addSPIRVVectorLoweringPasses scalarizes the contract AND its operand
    // transfer_reads (cont80f/81), destroying the coop matmul. Once they're mma ops
    // the vector lowering leaves them alone and only lowers the softmax.
    funcPassManager.addPass(createSPIRVVectorToGPUSubgroupMMAPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    funcPassManager.addPass(createSPIRVBreakDownLargeVectorPass());
    addSPIRVVectorLoweringPasses(funcPassManager);
    // nlearn (cont80): addSPIRVVectorLoweringPasses lowers the softmax
    // multi_reduction to a vector<M>-wide from_elements/addf (e.g. vector<16> for an
    // M-tile=16), which ConvertToSPIRV can't legalize (Apple has no Vector16 cap).
    // The FIRST BreakDownLargeVector above runs BEFORE this lowering so it misses
    // them — run it again AFTER to split the reduction vectors to SPIR-V-legal <=4.
    funcPassManager.addPass(createSPIRVBreakDownLargeVectorPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    funcPassManager.addPass(createGPUDistributePass());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    addLoopMaterializationPasses(funcPassManager);
    funcPassManager.addPass(createOptimizeVectorTransferPass());
    return;
  }

  if (getenv("NLEARN_COOP_ATTN_SMEM")) {
    // De-alias the online-softmax scratch writes to the flash-loop iter_args so
    // the flash loop one-shot-bufferizes (see the pass comment).
    funcPassManager.addPass(std::make_unique<DeAliasFlashScratchPass>());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    // Distribute the WHOLE per-query flash (qk + softmax + pv) across the
    // subgroup threads: tile the query-M dim to 1 so each thread computes one
    // query's full attention. Tile-and-fuse here is DESIRED (fuses qk/pv into the
    // per-query loop). No coop for qk/pv — those are small; the flash-structure
    // win is avoiding the global T×T materialization, not matrix-unit qk/pv.
    // First target: a correct, compiling thread-distributed flash to benchmark.
    funcPassManager.addPass(std::make_unique<ConfigSoftmaxThreadsPass>());
    {
      GPUApplyTilingLevelPassOptions topts;
      topts.tilingLevel = IREE::GPU::TilingLevel::Thread;
      topts.allowZeroSlices = true;
      funcPassManager.addPass(createGPUApplyTilingLevelPass(topts));
    }
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    // Vectorize the now-scalar (M=1) per-thread ops, break down any leftover
    // wide vectors, lower, then bufferize (scores/accumulators end up per-thread
    // or in shared memory via the workgroup allocator).
    {
      GenericVectorizationPassOptions options;
      funcPassManager.addPass(createGenericVectorizationPass(options));
    }
    funcPassManager.addPass(createSPIRVBreakDownLargeVectorPass());
    addSPIRVVectorLoweringPasses(funcPassManager);
    funcPassManager.addPass(createForOpCanonicalizationPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    addBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);
    // Resolve the (now bufferized) scf.forall(thread) into gpu.thread_id-indexed
    // code — GPUDistribute requires the forall to be bufferized first.
    funcPassManager.addPass(createGPUDistributePass());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    addLoopMaterializationPasses(funcPassManager);
    funcPassManager.addPass(createOptimizeVectorTransferPass());
    return;
  }

  // nlearn (NLEARN_COOP_ATTN_DECOMP): route the decomposed qk/pv matmuls (which
  // carry SPIR-V coop lowering_configs from decomposition_config) onto the
  // matrix units via the existing coop passes, instead of generic vectorization
  // (which can't legalize M>=16 and is scalar-slow at M=1). The coop passes only
  // touch matmuls that HAVE a lowering_config; the softmax generics fall through
  // to GenericVectorization below.
  if (getenv("NLEARN_COOP_ATTN_DECOMP")) {
    // nlearn: shared-memory-staged flash (the tractable path — softmax on a
    // regular row-major smem buffer, no coop layout) needs a proper pipeline
    // RESTRUCTURE to bufferize the scores then SPIRVTileAndPromote(promoteC) —
    // grafting the promote pass into this pre-bufferize tensor-level path
    // SEGFAULTS (verified cont64). Left as the documented next build step.
    // nlearn (NLEARN_COOP_ATTN_BARRIER): isolate the qk contraction from its
    // reduction epilogue so the coop passes don't infinite-loop on the
    // matmul->reduction fusion (cont78b) — try to make qk a clean coop matmul.
    if (getenv("NLEARN_COOP_ATTN_BARRIER")) {
      funcPassManager.addPass(std::make_unique<QkScoreBarrierPass>());
      funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    }
    funcPassManager.addPass(createSPIRVTileToCooperativeOpsPass());
    funcPassManager.addPass(createSPIRVVectorizeToCooperativeOpsPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  // Vectorize the decomposed matmul+softmax generics. The decomposed ops have no
  // per-op lowering_config (GPUTile would error "missing lowering configuration"),
  // so vectorize generically without the GPU tile step. (First increment: get
  // attention to COMPILE; per-tile coop routing + proper decompositionConfig are
  // the follow-up perf steps.)
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  {
    GenericVectorizationPassOptions options;
    funcPassManager.addPass(createGenericVectorizationPass(options));
  }
  // nlearn: the online-softmax reductions between the (now coop) qk/pv matmuls
  // get generic-vectorized to coop-tile-wide vectors (e.g. vector<16xf32> from
  // an M-tile=16), which ConvertToSPIRV can't legalize ("failed to legalize
  // arith.constant vector<16xf32>"). Break them down to SPIR-V-legal widths
  // (<=4) here, exactly as the coop matmul pipeline does for its scalar-fallback
  // leftovers. No-op for the M=1 path (already <=4-wide).
  funcPassManager.addPass(createSPIRVBreakDownLargeVectorPass());
  addSPIRVVectorLoweringPasses(funcPassManager);
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addSPIRVBufferizePasses(funcPassManager, gpuAllocateFunctionMemoryFn);
  addLoopMaterializationPasses(funcPassManager);
  funcPassManager.addPass(createOptimizeVectorTransferPass());
}

void addSPIRVWinogradVectorizePassPipeline(OpPassManager &funcPassManager) {
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/true);

  funcPassManager.addPass(createFoldAffineMinInDistributedLoopsPass());
  funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());

  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createGPUTilePass());
  funcPassManager.addPass(
      IREE::LinalgExt::createDecomposeWinogradTransformPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Tile to GPU invocations and vectorize.
  funcPassManager.addPass(createSPIRVAnnotateWinogradLoopsPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  {
    GenericVectorizationPassOptions options;
    options.enableCleanup = true;
    funcPassManager.addPass(createGenericVectorizationPass(options));
  }
  addSPIRVVectorLoweringPasses(funcPassManager);
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Bufferize and distribute.
  addSPIRVBufferizePasses(funcPassManager, gpuAllocateFunctionMemoryFn);

  // Generate loop nests for all remaining ops and remove trivial loops.
  addLoopMaterializationPasses(funcPassManager);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  funcPassManager.addPass(createOptimizeVectorTransferPass());
}

void addSPIRVCooperativeMatrixVectorizePassPipeline(
    OpPassManager &funcPassManager, unsigned pipelineDepth,
    unsigned storeStage) {
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/false,
      // nlearn: the WAR padding (64->68) creates a strided C-staging subview that
      // FoldMemRefAliasOps can't fold to a static offset on the bf16-store path.
      // Env-toggle to test disabling it.
      /*useWARForCooperativeMatrixCodegen=*/getenv("NLEARN_COOP_NO_WAR") == nullptr);

  addBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);

  // Tile to GPU workgroups and promote.
  funcPassManager.addPass(
      createSPIRVTileAndPromotePass(SPIRVTileAndPromotePassOptions{
          /*promoteCMatrix=*/getenv("NLEARN_COOP_NO_CPROMOTE") == nullptr,
          /*skipThreadLevel=*/true,
          /*skipOperandPromotion=*/true}));
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  // Run canonicalization patterns to propagate constant shape sizes after
  // removing trip-one loops.
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Tile and distribute to GPU subgroups.
  funcPassManager.addPass(createSPIRVTileToCooperativeOpsPass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Multi-buffer depending on pipeline depth and distribute to shared memory.
  if (pipelineDepth > 0) {
    funcPassManager.addPass(createGPUMultiBufferingPass(
        GPUMultiBufferingPassOptions{pipelineDepth + 1}));
  }
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());

  // Reduce bank conflicts by padding.
  {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = detail::bankConflictReductionPaddingBits;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }

  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  // Performs high-level n-D mechanical vectorization. This does not perform
  // unrolling or lowering, which is done later.
  {
    GenericVectorizationPassOptions options;
    funcPassManager.addPass(createGenericVectorizationPass(options));
  }

  // With subview ops, vector hoisting won't kick in. So fold memref subview ops
  // before performing vector unrolling and hoisting.
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());

  // Vectorize to cooperative ops.
  funcPassManager.addPass(createSPIRVVectorizeToCooperativeOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  // Run canonicalization patterns to propagate constant shape sizes after
  // removing trip-one loops.
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  funcPassManager.addPass(createOptimizeVectorTransferPass());

  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createSPIRVVectorToGPUSubgroupMMAPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  // nlearn: matmuls that DON'T convert to subgroup_mma (e.g. some training
  // backward contractions) fall back to scalar inside the coop pipeline, leaving
  // coop-native-shaped vectors (vector<16x16xf32> accumulators, vector<16xf32>
  // temporaries). Unlike the scalar tile-and-vectorize pipeline, the coop pipeline
  // did NOT run SPIRVBreakDownLargeVector, so those large vectors reached
  // ConvertToSPIRV and failed to legalize ("failed to legalize arith.constant
  // vector<16xf32>"), killing the whole compile. Break them down here so the
  // scalar-fallback path legalizes. For genuinely-converted coop dispatches this
  // is a no-op (their data lives in !gpu.mma_matrix, not large vectors).
  funcPassManager.addPass(createSPIRVBreakDownLargeVectorPass());
  addSPIRVVectorLoweringPasses(funcPassManager);

  if (pipelineDepth > 0) {
    PipeliningSchedulingStrategy schedule =
        storeStage == 0 ? PipeliningSchedulingStrategy::loadStoreStage0
                        : PipeliningSchedulingStrategy::loadGlobalStage0;
    GPUPipeliningPassOptions pipelieningOptions = {};
    pipelieningOptions.epiloguePeeling = true;
    pipelieningOptions.depth = pipelineDepth;
    pipelieningOptions.scheduleIndex = llvm::to_underlying(schedule);
    funcPassManager.addPass(createGPUPipeliningPass(pipelieningOptions));
  }
}

void addSPIRVMatmulPromoteVectorizePassPipeline(OpPassManager &funcPassManager,
                                                unsigned pipelineDepth,
                                                unsigned storeStage) {
  // Guards against 0 for consistency with older user provided tuning configs.
  pipelineDepth = pipelineDepth ? pipelineDepth : 1;
  LLVM_DEBUG(llvm::dbgs() << "Non-zero Pipeline Depth: " << pipelineDepth
                          << "\n";);
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/false,
      // nlearn: the WAR padding (64->68) creates a strided C-staging subview that
      // FoldMemRefAliasOps can't fold to a static offset on the bf16-store path.
      // Env-toggle to test disabling it.
      /*useWARForCooperativeMatrixCodegen=*/getenv("NLEARN_COOP_NO_WAR") == nullptr);

  // Promote to workgroups and tile to threads.
  funcPassManager.addPass(createGPUTensorTileToSerialLoopsPass());
  funcPassManager.addPass(createGPUTensorAlloc());
  funcPassManager.addPass(createGPUTensorTilePass());

  // Performs high-level n-D mechanical vectorization. This does not perform
  // unrolling or lowering, which is done later.
  {
    GenericVectorizationPassOptions options;
    options.enableCleanup = false;
    options.maxVectorSize = 4096;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  // Bufferize.
  addBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);

  // Distribute scf.forall to GPU threads.
  funcPassManager.addPass(createGPUDistributePass());

  if (pipelineDepth > 1 || storeStage == 0) {
    GPUMultiBufferingPassOptions multibufferingOptions = {
        storeStage == 0 ? pipelineDepth + 1 : pipelineDepth};
    funcPassManager.addPass(createGPUMultiBufferingPass(multibufferingOptions));
  }

  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());

  {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = detail::bankConflictReductionPaddingBits;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }

  // With subview ops, vector hoisting won't kick in. So fold memref subview ops
  // before performing vector unrolling and hoisting.
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());

  funcPassManager.addPass(createSPIRVInitialVectorLoweringPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createSPIRVFinalVectorLoweringPass());

  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());

  // Hoist loop invariant code to avoid pipelining it.
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  PipeliningSchedulingStrategy schedule =
      storeStage == 0 ? PipeliningSchedulingStrategy::loadStoreStage0
                      : PipeliningSchedulingStrategy::loadGlobalStage0;
  GPUPipeliningPassOptions pipelieningOptions = {};
  pipelieningOptions.epiloguePeeling = true;
  pipelieningOptions.depth = pipelineDepth;
  pipelieningOptions.scheduleIndex = llvm::to_underlying(schedule);
  funcPassManager.addPass(createGPUPipeliningPass(pipelieningOptions));

  addLoopMaterializationPasses(funcPassManager);
}

void addSPIRVSubgroupReducePassPipeline(OpPassManager &funcPassManager) {
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/true);

  // Fuse input parallel ops into the reduction op so that we don't need to
  // create temporary allocations during bufferization.
  funcPassManager.addPass(createRematerializeParallelOpsPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());

  funcPassManager.addPass(createGPUTileReductionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());

  // Performs high-level n-D mechanical vectorization. This does not perform
  // unrolling or lowering, which is done later.
  {
    GenericVectorizationPassOptions options;
    options.enableVectorMasking = true;
    options.useConfiguredVectorSizes = false;
    options.enableCleanup = false;
    options.generateContract = false;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Bufferize and distribute.
  // We bufferize before distributing to threads there; so we are still at the
  // block level. Therefore, need to allocate workgroup memory.
  addSPIRVBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());

  // Simplify the IR for vector distribution.
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());

  // Handle vector reduction operations specifically.
  VectorReductionToGPUPassOptions options;
  options.expandSubgroupReduction = false;
  funcPassManager.addPass(createVectorReductionToGPUPass(options));
  // Perform normal vector unrolling and lowering transformations. This breaks
  // vectors down to native machine size.
  addSPIRVVectorLoweringPasses(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

static void buildSPIRVCodegenConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
    addEncodingToNopPasses(funcPassManager);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());
  modulePassManager.addPass(createSPIRVSelectLoweringStrategyPass());
}

void buildSPIRVCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  variantPassManager.addPass(createSpecializeExportsPass());
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildSPIRVCodegenConfigurationPassPipelineImpl(modulePassManager);
}

void buildSPIRVCodegenPassPipeline(OpPassManager &variantPassManager) {
  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(
        createSPIRVLowerExecutableUsingTransformDialectPass());
    FunctionLikeNest(modulePassManager)
        .addPass(createSPIRVLowerExecutableTargetPass)
        .addPass(createVerifyWorkgroupDistributionPass);
    addMemRefLoweringPasses(modulePassManager);
    FunctionLikeNest(modulePassManager).addPass(createGpuEliminateBarriers);
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(createResolveWorkgroupCountHintsPass());
  variantPassManager.addPass(IREE::Util::createDropCompilerHintsPass(
      IREE::Util::DropCompilerHintsPassOptions{/*keepAssumeInt=*/true}));

  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    addSPIRVLoweringPasses(modulePassManager);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Using SPIR-V pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

// NOTE: this runs on the top-level program module containing all hal.executable
// ops.
void buildSPIRVLinkingPassPipeline(OpPassManager &modulePassManager) {
  auto &executablePassManager =
      modulePassManager.nest<IREE::HAL::ExecutableOp>();
  // Trim the allowed target environment (version/capability/extension/etc.) to
  // the minimal requirement needed by compiled spirv.module ops. This helps to
  // increase the chance of linking different variant ops together.
  executablePassManager.addNestedPass<IREE::HAL::ExecutableVariantOp>(
      createSPIRVTrimExecutableTargetEnvPass());
  // Materialize the minimal required target environment into proper device
  // queries to execute in the runtime.
  executablePassManager.addNestedPass<IREE::HAL::ExecutableVariantOp>(
      createSPIRVMaterializeExecutableConditionsPass());
  // Link together executables. This may produce some IR duplication.
  modulePassManager.addPass(createSPIRVLinkExecutablesPass());

  // Cleanup IR duplication.
  modulePassManager.addNestedPass<IREE::HAL::ExecutableOp>(
      mlir::createCanonicalizerPass());
}

//===---------------------------------------------------------------------===//
// Register SPIR-V Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"
} // namespace

void registerCodegenSPIRVPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> SPIRVConfigPipeline(
      "iree-codegen-spirv-configuration-pipeline",
      "Runs the pipeline for configuring the lowering from linalg to SPIR-V on "
      "all functions in a module",
      [](OpPassManager &modulePassManager) {
        buildSPIRVCodegenConfigurationPassPipelineImpl(modulePassManager);
      });

  static PassPipelineRegistration<> LinalgSPIRVPipeline(
      "iree-codegen-linalg-to-spirv-pipeline",
      "Runs the progressive lowering pipeline from linalg to SPIR-V",
      [](OpPassManager &variantPassManager) {
        buildSPIRVCodegenPassPipeline(variantPassManager);
      });
}

} // namespace mlir::iree_compiler
