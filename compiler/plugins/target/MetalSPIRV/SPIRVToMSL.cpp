// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/MetalSPIRV/SPIRVToMSL.h"

#include <cstdlib>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

// Disable exception handling in favor of assertions.
#define SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
#include "third_party/spirv_cross/spirv_msl.hpp"

#define DEBUG_TYPE "spirv-to-msl"

/// The [[buffer(N)]] index for push constants.
/// Note that this MUST be kept consistent with the Metal HAL driver.
#define IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX 3

namespace mlir::iree_compiler {

namespace {
class SPIRVToMSLCompiler : public SPIRV_CROSS_NAMESPACE::CompilerMSL {
public:
  using CompilerMSL::CompilerMSL;

  MetalShader::ThreadGroupSize
  getWorkgroupSizeForEntryPoint(StringRef entryName) {
    const auto &entryPoint = get_entry_point(
        entryName.str(), spv::ExecutionModel::ExecutionModelGLCompute);
    const auto &workgroupSize = entryPoint.workgroup_size;
    // TODO(antiagainst): support specialization constant.
    if (workgroupSize.constant != 0) {
      return {0, 0, 0};
    }
    return {workgroupSize.x, workgroupSize.y, workgroupSize.z};
  }

  // A struct containing a resource descriptor's information.
  struct Descriptor {
    uint32_t set;
    uint32_t binding;

    Descriptor(uint32_t s, uint32_t b) : set(s), binding(b) {}

    friend bool operator<(const Descriptor &l, const Descriptor &r) {
      return std::tie(l.set, l.binding) < std::tie(r.set, r.binding);
    }
  };

  // Updates `descriptors` with resource set and binding number pairs in
  // increasing order, and `hasPushConstant` if with push constants.
  // Returns true if no unsupported cases are encountered.
  bool getResources(SmallVectorImpl<Descriptor> *descriptors,
                    bool *hasPushConstant) {
    descriptors->clear();
    *hasPushConstant = false;

    // Iterate over all variables in the SPIR-V blob.
    bool hasUnknownCase = false;
    ir.for_each_typed_id<SPIRV_CROSS_NAMESPACE::SPIRVariable>(
        [&](uint32_t id, SPIRV_CROSS_NAMESPACE::SPIRVariable &var) {
          auto storage = var.storage;
          switch (storage) {
            // Non-interface variables. We don't care.
          case spv::StorageClassFunction:
          case spv::StorageClassPrivate:
          case spv::StorageClassWorkgroup:
            // Builtin variables. We don't care either.
          case spv::StorageClassInput:
            return;
          case spv::StorageClassPushConstant:
            *hasPushConstant = true;
            return;
          case spv::StorageClassUniform:
          case spv::StorageClassStorageBuffer: {
            uint32_t setNo = get_decoration(id, spv::DecorationDescriptorSet);
            uint32_t bindingNo = get_decoration(id, spv::DecorationBinding);
            descriptors->emplace_back(setNo, bindingNo);
            return;
          }
          default:
            break;
          }
          hasUnknownCase = true;
        });

    llvm::sort(*descriptors);
    return !hasUnknownCase;
  }

  bool requiresMSL31() {
    bool required = false;
    ir.for_each_typed_id<SPIRV_CROSS_NAMESPACE::SPIRType>(
        [&](uint32_t, SPIRV_CROSS_NAMESPACE::SPIRType &type) {
          required |=
              type.op == spv::OpTypeCooperativeMatrixKHR ||
              type.basetype == SPIRV_CROSS_NAMESPACE::SPIRType::BFloat16;
        });
    return required;
  }

  Options getCompilationOptions(IREE::HAL::MetalTargetPlatform platform) {
    // TODO(antiagainst): fill out the following according to the Metal GPU
    // family.
    SPIRVToMSLCompiler::Options spvCrossOptions;
    switch (platform) {
    case IREE::HAL::MetalTargetPlatform::macOS:
      spvCrossOptions.platform = SPIRVToMSLCompiler::Options::Platform::macOS;
      break;
    case IREE::HAL::MetalTargetPlatform::iOS:
    case IREE::HAL::MetalTargetPlatform::iOSSimulator:
      spvCrossOptions.platform = SPIRVToMSLCompiler::Options::Platform::iOS;
      break;
    }
    // Apple cooperative matrices and native bfloat types require MSL 3.1.
    // Derive that floor from the serialized SPIR-V rather than an ambient
    // feature flag so cross-compilation is reproducible in a fresh process.
    // Metal 4 tensor emission remains an independent development opt-in.
    if (std::getenv("IREE_METAL_MSL4") ||
        std::getenv("IREE_METAL_MSL4_MATMUL2D")) {
      spvCrossOptions.msl_version =
          SPIRVToMSLCompiler::Options::make_msl_version(4, 0);
    } else if (requiresMSL31()) {
      spvCrossOptions.msl_version =
          SPIRVToMSLCompiler::Options::make_msl_version(3, 1);
    } else {
      spvCrossOptions.msl_version =
          SPIRVToMSLCompiler::Options::make_msl_version(3, 0);
    }
    // Enable using Metal argument buffers. It is more akin to Vulkan descriptor
    // sets, which is how IREE HAL models resource bindings and mappings.
    spvCrossOptions.argument_buffers = true;
    return spvCrossOptions;
  }
};

// iree-metal / Metal 4 cooperative-tensor port (task#28).
// For a plain bf16xbf16->bf16 (f32-accumulate) matmul dispatch, spirv-cross emits
// the coop-matrix simdgroup_multiply_accumulate path (validated ~3.26 TFLOP/s on the
// FFN GEMM). Measured in isolation, a DEVICE-DIRECT Metal 4 matmul2d over the same
// 64x64 workgroup tile reaches ~3.58 (+10%, 95% of jax-metal) and the batched attention
// bmm reaches +50% — the structure-preserving swap loses (per-subgroup matmul2d on 16x16
// = 1.35-2.33 < 3.26), so we substitute the WHOLE kernel body with a device-direct
// matmul2d microkernel. IREE's dispatch already uses a [64,64] workgroup tile with a
// 128-thread (4-simdgroup) workgroup and grid=(ceil(N/64),ceil(M/64)) — which matches
// matmul2d<execution_simdgroups<4>> over a 64x64 tile exactly, so we keep IREE's dispatch
// and only replace the emitted MSL. Gated on IREE_METAL_MSL4_MATMUL2D (inert by default).
//
// Returns the substitute MSL, or std::nullopt to fall back to the spirv-cross output.
static std::optional<std::string>
tryEmitMatmul2dMSL(StringRef reviseName, StringRef origName, size_t numBuffers,
                   bool hasPushConstant) {
  if (!std::getenv("IREE_METAL_MSL4_MATMUL2D"))
    return std::nullopt;
  // Conservative first target: a pure static-shaped matmul dispatch with exactly
  // A,B,C buffers and no push constants (no dynamic dims / fused operands yet).
  if (numBuffers != 3 || hasPushConstant)
    return std::nullopt;

  // batch_matmul: "..._batch_matmul_<B>x<M>x<N>x<K>_bf16xbf16xf32" (f32 output;
  // attention score/value matmuls keep the accumulator in f32 for softmax). Each
  // workgroup computes one 64x64 tile of one batch (IREE tiles [1,64,64], grid
  // (N/64, M/64, B), wg.z = batch), mapping 1:1 onto matmul2d<simdgroups<4>>.
  // NOTE: batched attention matmuls in the backward often carry a FUSED elementwise
  // epilogue (scale / mask) in the same dispatch. Substituting the whole kernel drops
  // that epilogue -> nan on real models (bert GNORM=nan), even though the bare matmul
  // is numerically exact in isolation. So the batched path is behind its OWN opt-in
  // flag (separate from the validated pure-2D IREE_METAL_MSL4_MATMUL2D) until an
  // epilogue-safety guard (detect non-coop-matmul ops in the module -> fall back) lands.
  if (size_t bmk = origName.find("_batch_matmul_");
      bmk != StringRef::npos && std::getenv("IREE_METAL_MSL4_BMM")) {
    StringRef brest = origName.substr(bmk + 14);
    if (!brest.contains("bf16xbf16xf32"))
      return std::nullopt;
    StringRef bdimsStr = brest.split('_').first;
    SmallVector<StringRef, 4> bd;
    bdimsStr.split(bd, 'x');
    if (bd.size() != 4)
      return std::nullopt;
    long long B = 0, M = 0, N = 0, K = 0;
    if (bd[0].getAsInteger(10, B) || bd[1].getAsInteger(10, M) ||
        bd[2].getAsInteger(10, N) || bd[3].getAsInteger(10, K))
      return std::nullopt;
    if (M % 64 || N % 64 || K % 8)
      return std::nullopt;
    if (std::getenv("IREE_METAL_MSL4_MATMUL2D_LOG"))
      llvm::errs() << "[matmul2d] SUBSTITUTED " << origName << " (batch B=" << B
                   << " M=" << M << " N=" << N << " K=" << K << ")\n";
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "#include <metal_stdlib>\n"
          "#include <metal_tensor>\n"
          "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
          "using namespace metal;\n"
          "using namespace mpp::tensor_ops;\n"
          "struct spvDescriptorSetBuffer0 {\n"
          "  device bfloat* A [[id(0)]];\n"
          "  device bfloat* B [[id(1)]];\n"
          "  device float* C [[id(2)]];\n"  // f32 accumulator output
          "};\n"
          "kernel void "
       << reviseName
       << "(constant spvDescriptorSetBuffer0& s [[buffer(0)]], "
          "uint3 wg [[threadgroup_position_in_grid]], "
          "uint3 lid [[thread_position_in_threadgroup]]) {\n"
          "  uint b = wg.z;\n"
          "  device bfloat* Ab = s.A + b*"
       << (M * K)
       << ";\n"
          "  device bfloat* Bb = s.B + b*"
       << (K * N)
       << ";\n"
          "  device float* Cb = s.C + b*"
       << (M * N)
       << ";\n"
          "  tensor<device bfloat, dextents<int32_t,2>, tensor_inline> A(Ab, "
          "dextents<int32_t,2>("
       << K << ", " << M
       << "));\n"
          "  tensor<device bfloat, dextents<int32_t,2>, tensor_inline> B(Bb, "
          "dextents<int32_t,2>("
       << N << ", " << K
       << "));\n"
          "  tensor<device float, dextents<int32_t,2>, tensor_inline> C(Cb, "
          "dextents<int32_t,2>("
       << N << ", " << M
       << "));\n"
          "  constexpr auto d = matmul2d_descriptor(64, 64, "
          "static_cast<int>(dynamic_extent), false, false, false);\n"
          "  matmul2d<d, execution_simdgroups<4>> op;\n"
          "  auto mA = A.slice(0, wg.y*64);\n"
          "  auto mB = B.slice(wg.x*64, 0);\n"
          "  auto mC = C.slice(wg.x*64, wg.y*64);\n"
          "  op.run(mA, mB, mC);\n"
          "}\n";
    return os.str();
  }

  // Parse "..._matmul_<M>x<N>x<K>_bf16xbf16xf32" out of the original entry name.
  size_t mk = origName.find("_matmul_");
  if (mk == StringRef::npos)
    return std::nullopt;
  StringRef rest = origName.substr(mk + 8);
  if (!rest.contains("bf16xbf16xf32"))
    return std::nullopt;
  auto [dimsStr, tail] = rest.split('_');
  SmallVector<StringRef, 3> dims;
  dimsStr.split(dims, 'x');
  if (dims.size() != 3)
    return std::nullopt;
  long long M = 0, N = 0, K = 0;
  if (dims[0].getAsInteger(10, M) || dims[1].getAsInteger(10, N) ||
      dims[2].getAsInteger(10, K))
    return std::nullopt;
  // matmul2d works on a 64x64 workgroup tile; require the problem to be tiled by 64
  // so the grid IREE dispatches matches (no ragged edge handling in this first brick).
  if (M % 64 || N % 64 || K % 8)
    return std::nullopt;

  if (std::getenv("IREE_METAL_MSL4_MATMUL2D_LOG"))
    llvm::errs() << "[matmul2d] SUBSTITUTED " << origName << " (M=" << M
                 << " N=" << N << " K=" << K << ")\n";

  std::string s;
  llvm::raw_string_ostream os(s);
  os << "#include <metal_stdlib>\n"
        "#include <metal_tensor>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "using namespace metal;\n"
        "using namespace mpp::tensor_ops;\n"
        "struct spvDescriptorSetBuffer0 {\n"
        "  device bfloat* A [[id(0)]];\n"
        "  device bfloat* B [[id(1)]];\n"
        "  device bfloat* C [[id(2)]];\n"
        "};\n"
        "kernel void "
     << reviseName
     << "(constant spvDescriptorSetBuffer0& s [[buffer(0)]], "
        "uint3 wg [[threadgroup_position_in_grid]], "
        "uint3 lid [[thread_position_in_threadgroup]]) {\n"
        "  tensor<device bfloat, dextents<int32_t,2>, tensor_inline> A(s.A, "
        "dextents<int32_t,2>("
     << K << ", " << M
     << "));\n"
        "  tensor<device bfloat, dextents<int32_t,2>, tensor_inline> B(s.B, "
        "dextents<int32_t,2>("
     << N << ", " << K
     << "));\n"
        "  tensor<device bfloat, dextents<int32_t,2>, tensor_inline> C(s.C, "
        "dextents<int32_t,2>("
     << N << ", " << M
     << "));\n"
        "  constexpr auto d = matmul2d_descriptor(64, 64, "
        "static_cast<int>(dynamic_extent), false, false, false);\n"
        "  matmul2d<d, execution_simdgroups<4>> op;\n"
        "  auto mA = A.slice(0, wg.y*64);\n"
        "  auto mB = B.slice(wg.x*64, 0);\n"
        "  auto mC = C.slice(wg.x*64, wg.y*64);\n"
        // The HW intrinsic (WMMAR4_F32_16x16x16_BF16) accumulates in f32 and
        // converts to bf16 on store, matching IREE's f32-accumulate semantics.
        "  op.run(mA, mB, mC);\n"
        "}\n";
  return os.str();
}
} // namespace

std::optional<std::pair<MetalShader, std::string>>
crossCompileSPIRVToMSL(IREE::HAL::MetalTargetPlatform targetPlatform,
                       llvm::ArrayRef<uint32_t> spvBinary,
                       StringRef entryPoint) {
  SPIRVToMSLCompiler spvCrossCompiler(spvBinary.data(), spvBinary.size());

  // All spirv-cross operations work on the current entry point. It should be
  // set right after the cross compiler construction.
  spvCrossCompiler.set_entry_point(
      entryPoint.str(), spv::ExecutionModel::ExecutionModelGLCompute);

  SmallVector<SPIRVToMSLCompiler::Descriptor> descriptors;
  bool hasPushConstant = false;
  if (!spvCrossCompiler.getResources(&descriptors, &hasPushConstant)) {
    return std::nullopt;
  }

  // Explicitly set the argument buffer [[id(N)]] location for each SPIR-V
  // resource variable.
  for (const auto &descriptor : descriptors) {
    SPIRV_CROSS_NAMESPACE::MSLResourceBinding binding = {};
    binding.stage = spv::ExecutionModelGLCompute;
    binding.desc_set = descriptor.set;
    binding.binding = descriptor.binding;
    // We only interact with buffers in IREE.
    binding.msl_buffer = descriptor.binding;

    spvCrossCompiler.add_msl_resource_binding(binding);
  }
  // If push constants are used, explicitly set its [[buffer(N)]] location too.
  if (hasPushConstant) {
    SPIRV_CROSS_NAMESPACE::MSLResourceBinding binding = {};
    binding.stage = spv::ExecutionModelGLCompute;
    binding.desc_set =
        SPIRV_CROSS_NAMESPACE::ResourceBindingPushConstantDescriptorSet;
    binding.binding = SPIRV_CROSS_NAMESPACE::ResourceBindingPushConstantBinding;
    binding.msl_buffer = IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX;

    spvCrossCompiler.add_msl_resource_binding(binding);
  }

  auto spvCrossOptions = spvCrossCompiler.getCompilationOptions(targetPlatform);
  spvCrossCompiler.set_msl_options(spvCrossOptions);
  uint32_t languageVersion = 196608u; // MTLLanguageVersion3_0
  if (spvCrossOptions.supports_msl_version(4, 0)) {
    languageVersion = 262144u; // MTLLanguageVersion4_0
  } else if (spvCrossOptions.supports_msl_version(3, 1)) {
    languageVersion = 196609u; // MTLLanguageVersion3_1
  }

  std::string mslSource = spvCrossCompiler.compile();
  // Get the revised entry point name. Cross compiling to MSL generates source
  // code, where we may run into the case that we are using reserved keyword for
  // the entry point name, e.g., `abs`. Under such circumstances, it will be
  // revised to avoid collision.
  const auto &spirvEntryPoint = spvCrossCompiler.get_entry_point(
      entryPoint.str(), spv::ExecutionModel::ExecutionModelGLCompute);

  // iree-metal (task#28): optionally substitute a device-direct Metal 4 matmul2d
  // microkernel for a plain matmul dispatch. Falls back to the spirv-cross MSL
  // above when the gate is off or the dispatch is not a supported matmul shape.
  if (auto m2d = tryEmitMatmul2dMSL(spirvEntryPoint.name, spirvEntryPoint.orig_name,
                                    descriptors.size(), hasPushConstant)) {
    mslSource = std::move(*m2d);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Original entry point name: '" << spirvEntryPoint.orig_name
                 << "'\n";
    llvm::dbgs() << "Revised entry point name: '" << spirvEntryPoint.name
                 << "'\n";
    llvm::dbgs() << "Generated MSL:\n-----\n" << mslSource << "\n-----\n";
  });

  auto workgroupSize =
      spvCrossCompiler.getWorkgroupSizeForEntryPoint(entryPoint);
  if (!workgroupSize.x || !workgroupSize.y || !workgroupSize.z) {
    return std::nullopt;
  }
  return std::make_pair(
      MetalShader{std::move(mslSource), workgroupSize, languageVersion},
      spirvEntryPoint.name);
}

} // namespace mlir::iree_compiler
