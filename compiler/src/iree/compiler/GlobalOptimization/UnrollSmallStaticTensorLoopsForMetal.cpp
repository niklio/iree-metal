// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_UNROLLSMALLSTATICTENSORLOOPSFORMETALPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

// These limits cover short scans while keeping expansion bounded. Count all
// nested operations in the body so regions inside linalg ops contribute to the
// code-size estimate as well.
constexpr uint64_t kMaxTripCount = 32;
constexpr uint64_t kMaxExpandedOperationCount = 1024;

static bool hasOnlyMetalExecutableTargets(ModuleOp moduleOp) {
  IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
  if (failed(deviceAnalysis.run())) {
    return false;
  }
  SetVector<IREE::HAL::ExecutableTargetAttr> targets;
  deviceAnalysis.gatherAllExecutableTargets(targets);
  return !targets.empty() && llvm::all_of(targets, [](auto target) {
    return target.getBackend().getValue() == "metal-spirv";
  });
}

static bool isEligibleLoop(scf::ForOp forOp) {
  // Limit this to tensor programs. Scalar loops should remain available for
  // later lowering, where they may execute efficiently on the host or device.
  if (!llvm::any_of(forOp.getResultTypes(),
                    [](Type type) { return isa<RankedTensorType>(type); })) {
    return false;
  }

  std::optional<APInt> staticTripCount = forOp.getStaticTripCount();
  if (!staticTripCount || staticTripCount->isZero() ||
      staticTripCount->getActiveBits() > 64) {
    return false;
  }
  uint64_t tripCount = staticTripCount->getZExtValue();
  if (tripCount > kMaxTripCount) {
    return false;
  }

  uint64_t operationCount = 0;
  forOp.getRegion().walk([&](Operation *) { ++operationCount; });
  return operationCount <= kMaxExpandedOperationCount / tripCount;
}

// Make loop-carried destination-style results alias their corresponding
// iteration arguments. Device codegen requires this equivalence when
// bufferizing scf.for inside a dispatch. The update is safe for parallel
// elementwise destination-style ops: each output element overwrites only the
// input element consumed by the same invocation.
static bool makeYieldsDestinationPassing(scf::ForOp forOp,
                                         IRRewriter &rewriter) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> replacements(yieldOp.getOperands());
  SmallVector<std::pair<OpOperand *, Value>> initReplacements;

  // Preflight all destination retargeting before changing the IR. A yielded
  // tensor may already be destination-passing (for example, insert_slice).
  // Otherwise only retarget elementwise linalg results whose old output is not
  // read and whose loop-carried input uses the same indexing map. This is the
  // in-place update pattern emitted for scan state and avoids introducing
  // cross-element read-after-write hazards.
  for (auto [index, iterArg, yielded] :
       llvm::enumerate(forOp.getRegionIterArgs(), yieldOp.getOperands())) {
    if (!isa<RankedTensorType>(iterArg.getType())) {
      continue;
    }
    auto opResult = dyn_cast<OpResult>(yielded);
    if (opResult) {
      auto dpsOp = dyn_cast<DestinationStyleOpInterface>(opResult.getOwner());
      if (dpsOp) {
        OpOperand *initOperand =
            dpsOp.getDpsInitOperand(opResult.getResultNumber());
        if (!initOperand || initOperand->get().getType() != iterArg.getType()) {
          return false;
        }
        if (initOperand->get() != iterArg) {
          auto linalgOp = dyn_cast<linalg::LinalgOp>(opResult.getOwner());
          if (!linalgOp || !linalg::isElementwise(linalgOp) ||
              !linalgOp.getMatchingBlockArgument(initOperand).use_empty()) {
            return false;
          }
          AffineMap outputMap = linalgOp.getMatchingIndexingMap(initOperand);
          for (OpOperand &use : iterArg.getUses()) {
            Operation *user = use.getOwner();
            if (user == linalgOp) {
              if (!linalgOp.isDpsInput(&use) ||
                  linalgOp.getMatchingIndexingMap(&use) != outputMap) {
                return false;
              }
              continue;
            }
            // Earlier users finish before the destination is overwritten. In
            // scan bodies these are the elementwise producers feeding the
            // yielded op. Reject later or nested users, which could observe the
            // updated storage through an old tensor value.
            if (user->getBlock() != linalgOp->getBlock() ||
                !user->isBeforeInBlock(linalgOp)) {
              return false;
            }
          }
          initReplacements.emplace_back(initOperand, iterArg);
        }
      }
    }

    if (yielded == iterArg) {
      continue;
    }

    rewriter.setInsertionPoint(yieldOp);
    replacements[index] = bufferization::MaterializeInDestinationOp::create(
                              rewriter, yieldOp.getLoc(), yielded, iterArg)
                              ->getResult(0);
  }
  for (auto &initReplacement : initReplacements) {
    OpOperand *initOperand = initReplacement.first;
    Value replacement = initReplacement.second;
    rewriter.modifyOpInPlace(initOperand->getOwner(),
                             [&]() { initOperand->set(replacement); });
  }
  rewriter.modifyOpInPlace(yieldOp,
                           [&]() { yieldOp->setOperands(replacements); });
  return true;
}

class UnrollSmallStaticTensorLoopsForMetalPass final
    : public impl::UnrollSmallStaticTensorLoopsForMetalPassBase<
          UnrollSmallStaticTensorLoopsForMetalPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (!hasOnlyMetalExecutableTargets(moduleOp)) {
      return;
    }

    // Collect outermost loops only. Expanding both a nested loop and its parent
    // can multiply code size beyond the per-loop budget.
    SmallVector<scf::ForOp> loops;
    moduleOp.walk([&](scf::ForOp forOp) {
      if (!forOp->getParentOfType<scf::ForOp>() && isEligibleLoop(forOp)) {
        loops.push_back(forOp);
      }
    });
    bool formDeviceDispatch =
        ::getenv("IREE_METAL_NO_STATIC_TENSOR_LOOPS_IN_DISPATCH") == nullptr;
    IRRewriter rewriter(&getContext());
    for (scf::ForOp forOp : loops) {
      if (formDeviceDispatch) {
        Location loc = forOp.getLoc();
        if (!makeYieldsDestinationPassing(forOp, rewriter)) {
          forOp.emitRemark(
              "failed to form dispatch for eligible static tensor loop");
          continue;
        }
        FailureOr<IREE::Flow::DispatchRegionOp> dispatchOp =
            IREE::Flow::wrapOpInDispatchRegion(rewriter, forOp);
        if (failed(dispatchOp)) {
          forOp.emitRemark(
              "failed to form dispatch for eligible static tensor loop");
          continue;
        }
        // A single thread is sufficient for these bounded tensor loops and
        // keeps their loop-carried writes inside one ordered GPU invocation.
        // An explicit 1x1x1 grid also selects the scalar SPIR-V lowering path,
        // avoiding invalid partially distributed writes from nested loops.
        Region &countRegion = dispatchOp->getWorkgroupCount();
        Block *countBlock = rewriter.createBlock(&countRegion);
        rewriter.setInsertionPointToStart(countBlock);
        Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
        IREE::Flow::ReturnOp::create(rewriter, loc, ValueRange{one, one, one});
        continue;
      }
      if (failed(loopUnrollFull(forOp))) {
        forOp.emitRemark("failed to unroll eligible static tensor loop");
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
