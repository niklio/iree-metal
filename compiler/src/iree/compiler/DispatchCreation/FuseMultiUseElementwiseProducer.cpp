// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- FusionOfTensorsOps.cpp - Pass to fuse operations on tensors-------===//
//
// Pass to fuse operations on tensors after conversion to Linalg. Uses the
// patterns from MLIR for fusion linalg operations on tensors, and a few
// patterns to fuse these with IREE specific operations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <limits>

#define DEBUG_TYPE "iree-dispatch-creation-fusion-of-tensor-ops"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FUSEMULTIUSEELEMENTWISEPRODUCERPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

// TODO: Remove this and the backing code once consteval is beyond being
// rolled back.
static llvm::cl::opt<int64_t> clLinalgMaxConstantFoldElements(
    "iree-codegen-linalg-max-constant-fold-elements",
    llvm::cl::desc("Maximum number of elements to try to constant fold."),
    llvm::cl::init(0));

/// Check if any of the use dominates all other uses of the operation.
static std::optional<OpOperand *> getFusableUse(Operation *op,
                                                DominanceInfo &dominanceInfo) {
  auto uses = op->getUses();
  for (OpOperand &source : uses) {
    Operation *sourceOp = source.getOwner();
    bool dominatesAllUsers = true;
    for (OpOperand &target : uses) {
      Operation *targetOp = target.getOwner();
      if (sourceOp != targetOp &&
          !dominanceInfo.properlyDominates(sourceOp, targetOp,
                                           /*enclosingOpOk=*/false)) {
        dominatesAllUsers = false;
        break;
      }
    }
    if (dominatesAllUsers) {
      return &source;
    }
  }
  return std::nullopt;
}

static OpOperand *getFirstUseInConsumer(Operation *producer,
                                        Operation *consumer) {
  for (OpOperand &opOperand : consumer->getOpOperands()) {
    if (opOperand.get().getDefiningOp() == producer) {
      return &opOperand;
    }
  }
  return nullptr;
}

static SmallVector<OpOperand *> getAllUsesInConsumer(Operation *producer,
                                                     Operation *consumer) {
  SmallVector<OpOperand *> allUses;
  for (OpOperand &opOperand : consumer->getOpOperands()) {
    if (opOperand.get().getDefiningOp() == producer) {
      allUses.push_back(&opOperand);
    }
  }
  return allUses;
}

/// Perform the fusion of `rootOp` with all the operations in `fusableOps`
/// using elementwise fusion.
static LogicalResult doMultiUseFusion(Operation *rootOp,
                                      llvm::SetVector<Operation *> &fusableOps,
                                      RewriterBase &rewriter) {
  assert(rootOp && "root op cant be null");

  LLVM_DEBUG({
    llvm::dbgs() << "Fusion root : \n";
    rootOp->print(llvm::dbgs());
    llvm::dbgs() << "\nFused with :";

    for (auto producer : fusableOps) {
      llvm::dbgs() << "\t";
      producer->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  SmallVector<Operation *> fusedOpsVec = llvm::to_vector(fusableOps);
  mlir::computeTopologicalSorting(fusedOpsVec);

  Operation *consumerOp = rootOp;
  OpBuilder::InsertionGuard g(rewriter);
  for (Operation *producerOp : llvm::reverse(fusedOpsVec)) {
    // Fuse all uses from producer -> consumer. It has been checked
    // before that all uses are fusable.
    while (OpOperand *fusedOperand =
               getFirstUseInConsumer(producerOp, consumerOp)) {
      rewriter.setInsertionPoint(consumerOp);
      FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
          linalg::fuseElementwiseOps(rewriter, fusedOperand);
      if (failed(fusionResult)) {
        return rewriter.notifyMatchFailure(consumerOp,
                                           "failed to fuse with producer");
      }
      for (auto replacement : fusionResult->replacements) {
        rewriter.replaceUsesWithIf(
            replacement.first, replacement.second, [&](OpOperand &use) {
              return use.getOwner() != fusionResult->fusedOp &&
                     fusableOps.count(use.getOwner()) == 0;
            });
      }
      consumerOp = fusionResult->fusedOp;
      if (failed(cast<linalg::GenericOp>(consumerOp).verify())) {
        return consumerOp->emitOpError("failed to verify op");
      }
    }
  }
  return success();
}

static FailureOr<unsigned> fuseMultiUseProducers(
    Operation *funcOp, MLIRContext *context, DominanceInfo &dominanceInfo,
    const FuseMultiUseElementwiseProducerPassOptions &options) {
  OpBuilder builder(context);
  llvm::MapVector<Operation *, llvm::SetVector<Operation *>> fusedOps;
  DenseMap<Operation *, Operation *> opToRootMap;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](linalg::GenericOp genericOp) {
        if (options.intraDispatch ==
            IREE::Flow::isNonNullAndOutsideDispatch(genericOp)) {
          return;
        }

        // 1. Only look at all parallel consumers.
        if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
          return;
        }

        // Dequantization-like operations should be fused with consumers to keep
        // the smaller bit width on the dispatch boundary.
        if (IREE::LinalgExt::isBitExtendOp(genericOp)) {
          return;
        }

        Operation *fusableProducer = nullptr;
        for (OpOperand &operand : genericOp->getOpOperands()) {
          // 2. Only fuse with `linalg.generic` producers that arent
          //    already part of another fusion group.
          auto producer = dyn_cast_if_present<linalg::GenericOp>(
              operand.get().getDefiningOp());
          if (!producer || opToRootMap.count(producer)) {
            continue;
          }

          // 3. For now do not fuse with ops in another block.
          if (producer->getBlock() != genericOp->getBlock()) {
            continue;
          }

          // 4. Basic fusability checks.
          if (!linalg::areElementwiseOpsFusable(&operand)) {
            continue;
          }

          // 5. Only consider all parallel `producer` with same iteration space
          //    as the consumer.
          if (producer.getNumLoops() != producer.getNumParallelLoops() ||
              genericOp.getNumLoops() != producer.getNumLoops()) {
            continue;
          }

          // 6. Check that the `genericOp` dominates all uses of `producer`.
          std::optional<OpOperand *> fusableUse =
              getFusableUse(producer, dominanceInfo);
          if (!fusableUse || fusableUse.value()->getOwner() != genericOp) {
            continue;
          }

          // 7. Skip bit-extend-like `producer` ops as we would rather fuse
          //    by cloning the producer instead of multi-use fusion.
          if (!options.intraDispatch &&
              IREE::LinalgExt::isBitExtendOp(producer)) {
            continue;
          }

          // 8. Skip bit-truncate-like `producer` ops as we would rather fuse
          //    these operations with their producers.
          if (!options.intraDispatch &&
              IREE::LinalgExt::isBitTruncateOp(producer)) {
            continue;
          }

          // 9. All uses from `producer` -> `consumer` need to be fusable.
          //    Without this the `producer` is still live, and there is no
          //    advantage to do the fusion.
          if (llvm::any_of(getAllUsesInConsumer(producer, genericOp),
                           [&](OpOperand *use) {
                             return !linalg::areElementwiseOpsFusable(use);
                           })) {
            continue;
          }

          fusableProducer = producer;
          break;
        }
        if (!fusableProducer) {
          return;
        }

        // If the `genericOp` is already part of a fusion group, just add the
        // the `fusableProducer` to the same group.
        llvm::SetVector<Operation *> &fusedOpSet = fusedOps[genericOp];
        fusedOpSet.insert(fusableProducer);
        opToRootMap[fusableProducer] = genericOp;
        return;
      });

  if (fusedOps.empty()) {
    return 0;
  }

  IRRewriter rewriter(context);
  for (auto it = fusedOps.rbegin(), ie = fusedOps.rend(); it != ie; ++it) {
    if (failed(doMultiUseFusion(it->first, it->second, rewriter))) {
      return funcOp->emitOpError("failed multi use fusion");
    }
  }

  RewritePatternSet fusionPatterns(context);
  linalg::populateEraseUnusedOperandsAndResultsPatterns(fusionPatterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(fusionPatterns)))) {
    return funcOp->emitOpError("multi use producer -> consumer fusion failed");
  }
  return fusedOps.size();
}

// Fuses the two-result Adam moment update produced by the ordinary multi-use
// pass into its single-result parameter update. The generic elementwise fusion
// utility stops at a multi-result producer, leaving the largest parameters to
// make two complete memory passes. Keep this deliberately narrow: FP32,
// identity-mapped, all-parallel moment/parameter updates with rsqrt in the
// consumer body.
struct FuseAdamMomentAndParameterUpdate final
    : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    std::string location;
    llvm::raw_string_ostream locationStream(location);
    consumer.getLoc().print(locationStream);
    if (!StringRef(location).contains_insensitive("adamw")) {
      return failure();
    }
    if (consumer.getNumDpsInputs() != 3 || consumer.getNumDpsInits() != 1 ||
        consumer.getNumResults() != 1 ||
        consumer.getNumLoops() != consumer.getNumParallelLoops() ||
        !consumer.getMatchingIndexingMap(consumer.getDpsInitOperand(0))
             .isIdentity() ||
        !llvm::hasSingleElement(consumer.getBody()->getOps<math::RsqrtOp>()) ||
        llvm::range_size(consumer.getBody()->getOps<arith::MulFOp>()) != 3 ||
        llvm::range_size(consumer.getBody()->getOps<arith::AddFOp>()) != 2 ||
        !llvm::hasSingleElement(consumer.getBody()->getOps<arith::SubFOp>())) {
      return failure();
    }

    linalg::GenericOp producer;
    SmallVector<unsigned> producerInputNumbers;
    for (auto [inputNumber, operand] :
         llvm::enumerate(consumer.getDpsInputOperands())) {
      auto candidate =
          operand->get().getDefiningOp<linalg::GenericOp>();
      if (!candidate || candidate.getNumResults() != 2) {
        continue;
      }
      if (producer && producer != candidate) {
        return failure();
      }
      producer = candidate;
      producerInputNumbers.push_back(inputNumber);
    }
    if (!producer || producerInputNumbers.size() != 2 ||
        producer->getBlock() != consumer->getBlock() ||
        producer.getNumDpsInputs() != 4 || producer.getNumDpsInits() != 2 ||
        producer.getNumLoops() != producer.getNumParallelLoops() ||
        producer.getNumLoops() != consumer.getNumLoops() ||
        producer.getIteratorTypesArray() !=
            consumer.getIteratorTypesArray()) {
      return failure();
    }
    unsigned identityProducerInputs = 0;
    unsigned scalarProducerInputs = 0;
    for (OpOperand *operand : producer.getDpsInputOperands()) {
      AffineMap map = producer.getMatchingIndexingMap(operand);
      identityProducerInputs += map.isIdentity();
      scalarProducerInputs += map.getNumResults() == 0;
    }
    if (identityProducerInputs != 3 || scalarProducerInputs != 1 ||
        llvm::range_size(producer.getBody()->getOps<arith::MulFOp>()) != 6 ||
        llvm::range_size(producer.getBody()->getOps<arith::AddFOp>()) != 2) {
      return failure();
    }

    SmallVector<bool> consumedResults(2, false);
    for (unsigned inputNumber : producerInputNumbers) {
      auto result = dyn_cast<OpResult>(
          consumer.getDpsInputOperand(inputNumber)->get());
      if (!result || result.getOwner() != producer ||
          result.getResultNumber() >= consumedResults.size() ||
          consumedResults[result.getResultNumber()]) {
        return failure();
      }
      consumedResults[result.getResultNumber()] = true;
      if (!consumer.getMatchingIndexingMap(
                       consumer.getDpsInputOperand(inputNumber))
               .isIdentity()) {
        return failure();
      }
    }
    if (!llvm::all_of(consumedResults, [](bool value) { return value; })) {
      return failure();
    }

    auto isIdentityF32Init = [](linalg::GenericOp op, OpOperand *operand) {
      auto type = dyn_cast<ShapedType>(operand->get().getType());
      return type && type.getElementType().isF32() &&
             op.getMatchingIndexingMap(operand).isIdentity();
    };
    if (!isIdentityF32Init(producer, producer.getDpsInitOperand(0)) ||
        !isIdentityF32Init(producer, producer.getDpsInitOperand(1)) ||
        !isIdentityF32Init(consumer, consumer.getDpsInitOperand(0))) {
      return failure();
    }

    SmallVector<Value> inputs(producer.getDpsInputs());
    SmallVector<AffineMap> maps;
    for (OpOperand *operand : producer.getDpsInputOperands()) {
      maps.push_back(producer.getMatchingIndexingMap(operand));
    }
    SmallVector<unsigned> consumerArgumentToNewInput(
        consumer.getNumDpsInputs(), std::numeric_limits<unsigned>::max());
    for (auto [inputNumber, operand] :
         llvm::enumerate(consumer.getDpsInputOperands())) {
      if (operand->get().getDefiningOp() == producer) {
        continue;
      }
      consumerArgumentToNewInput[inputNumber] = inputs.size();
      inputs.push_back(operand->get());
      maps.push_back(consumer.getMatchingIndexingMap(operand));
    }

    SmallVector<Value> inits(producer.getDpsInits());
    llvm::append_range(inits, consumer.getDpsInits());
    for (unsigned index = 0; index < producer.getNumDpsInits(); ++index) {
      maps.push_back(producer.getMatchingIndexingMap(
          producer.getDpsInitOperand(index)));
    }
    maps.push_back(
        consumer.getMatchingIndexingMap(consumer.getDpsInitOperand(0)));

    SmallVector<Type> resultTypes(producer->getResultTypes());
    llvm::append_range(resultTypes, consumer->getResultTypes());
    Block *producerBody = producer.getBlock();
    Block *consumerBody = consumer.getBlock();
    auto replacement = linalg::GenericOp::create(
        rewriter, consumer.getLoc(), resultTypes, inputs, inits, maps,
        consumer.getIteratorTypesArray(),
        [&](OpBuilder &builder, Location loc, ValueRange arguments) {
          IRMapping producerMapping;
          for (auto [index, argument] :
               llvm::enumerate(producerBody->getArguments())) {
            unsigned newIndex = index < producer.getNumDpsInputs()
                                    ? index
                                    : inputs.size() +
                                          index - producer.getNumDpsInputs();
            producerMapping.map(argument, arguments[newIndex]);
          }
          for (Operation &operation : producerBody->without_terminator()) {
            builder.clone(operation, producerMapping);
          }
          auto producerYield =
              cast<linalg::YieldOp>(producerBody->getTerminator());
          SmallVector<Value> producerValues;
          for (Value value : producerYield.getValues()) {
            producerValues.push_back(producerMapping.lookup(value));
          }

          IRMapping consumerMapping;
          for (unsigned inputNumber = 0;
               inputNumber < consumer.getNumDpsInputs(); ++inputNumber) {
            Value value = consumer.getDpsInputOperand(inputNumber)->get();
            if (auto result = dyn_cast<OpResult>(value);
                result && result.getOwner() == producer) {
              consumerMapping.map(consumerBody->getArgument(inputNumber),
                                  producerValues[result.getResultNumber()]);
            } else {
              consumerMapping.map(
                  consumerBody->getArgument(inputNumber),
                  arguments[consumerArgumentToNewInput[inputNumber]]);
            }
          }
          consumerMapping.map(
              consumerBody->getArgument(consumer.getNumDpsInputs()),
              arguments[inputs.size() + producer.getNumDpsInits()]);
          for (Operation &operation : consumerBody->without_terminator()) {
            builder.clone(operation, consumerMapping);
          }
          auto consumerYield =
              cast<linalg::YieldOp>(consumerBody->getTerminator());
          SmallVector<Value> yieldedValues(producerValues);
          for (Value value : consumerYield.getValues()) {
            yieldedValues.push_back(consumerMapping.lookup(value));
          }
          linalg::YieldOp::create(builder, loc, yieldedValues);
        });

    for (auto [index, result] : llvm::enumerate(producer->getResults())) {
      rewriter.replaceUsesWithIf(result, replacement.getResult(index),
                                 [&](OpOperand &use) {
                                   return use.getOwner() != consumer;
                                 });
    }
    rewriter.replaceOp(consumer, replacement.getResults().drop_front(2));
    rewriter.eraseOp(producer);
    return success();
  }
};

namespace {

/// Pass to fuse linalg on tensor operations as well as fusion of hal.interface*
/// operations with linalg.tensor_reshape operation.
struct FuseMultiUseElementwiseProducerPass final
    : impl::FuseMultiUseElementwiseProducerPassBase<
          FuseMultiUseElementwiseProducerPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void FuseMultiUseElementwiseProducerPass::runOnOperation() {
  Operation *funcOp = getOperation();
  MLIRContext *context = funcOp->getContext();
  FuseMultiUseElementwiseProducerPassOptions options{intraDispatch,
                                                     numIterations};

  // Run fusion of producer with consumer when producer has multiple uses.
  // For now run this sequence a fixed times (2 by default). Ideally we
  // would run it till no candidates exist.
  for (auto i : llvm::seq<unsigned>(0, options.numIterations)) {
    (void)i;
    auto &dominanceInfo = getAnalysis<DominanceInfo>();
    FailureOr<unsigned> numOfFusableCandidates =
        fuseMultiUseProducers(funcOp, context, dominanceInfo, options);
    if (failed(numOfFusableCandidates)) {
      funcOp->emitError("failed to fuse multi-use producers");
      return signalPassFailure();
    }
    if (numOfFusableCandidates.value() == 0) {
      break;
    }
  }

  const char *fuseAdamUpdate = getenv("IREE_METAL_FUSE_ADAM_UPDATE");
  if (fuseAdamUpdate && StringRef(fuseAdamUpdate) != "0") {
    RewritePatternSet patterns(context);
    patterns.add<FuseAdamMomentAndParameterUpdate>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp->emitError("failed to fuse Adam moment and parameter updates");
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
