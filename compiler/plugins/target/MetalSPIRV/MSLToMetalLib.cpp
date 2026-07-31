// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/MetalSPIRV/MSLToMetalLib.h"

#include <stdlib.h>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-msl-to-metal-lib"

namespace mlir::iree_compiler::IREE::HAL {

static std::optional<llvm::StringRef>
getMetalLanguageStandard(uint32_t languageVersion) {
  switch (languageVersion) {
  case 196608u: // MTLLanguageVersion3_0
    return "metal3.0";
  case 196609u: // MTLLanguageVersion3_1
    return "metal3.1";
  case 262144u: // MTLLanguageVersion4_0
    return "metal4.0";
  default:
    return std::nullopt;
  }
}

/// Returns the command to compile the given MSL source file into Metal library.
std::optional<std::string> detail::buildMetalCompileCommand(
    MetalTargetPlatform platform, uint32_t languageVersion,
    llvm::StringRef mslFile, llvm::StringRef libFile) {
  const char *sdk = "";
  switch (platform) {
  case MetalTargetPlatform::macOS:
    sdk = "macosx";
    break;
  case MetalTargetPlatform::iOS:
    sdk = "iphoneos";
    break;
  case MetalTargetPlatform::iOSSimulator:
    sdk = "iphonesimulator";
    break;
  }

  std::optional<llvm::StringRef> languageStandard =
      getMetalLanguageStandard(languageVersion);
  if (!languageStandard) {
    return std::nullopt;
  }

  // Metal shader offline compilation involves two steps:
  // 1. Compile the MSL source code into an Apple IR (AIR) file,
  // 2. Compile the AIR file into a Metal library.
  return llvm::Twine("xcrun -sdk ")
      .concat(sdk)
      .concat(" metal -std=")
      .concat(*languageStandard)
      .concat(" -c ")
      .concat(mslFile)
      .concat(" -o - | xcrun -sdk ")
      .concat(sdk)
      .concat(" metallib - -o ")
      .concat(libFile)
      .str();
}

/// Returns the given command via system shell.
static LogicalResult runSystemCommand(llvm::StringRef command) {
  LLVM_DEBUG(llvm::dbgs() << "Running system command: '" << command << "'\n");
  int exitCode = system(command.data());
  if (exitCode == 0) {
    return success();
  }
  llvm::errs() << "Failed to run system command '" << command
               << "' with error code: " << exitCode << "\n";
  return failure();
}

std::unique_ptr<llvm::MemoryBuffer>
compileMSLToMetalLib(MetalTargetPlatform targetPlatform,
                     uint32_t languageVersion, llvm::StringRef mslCode,
                     llvm::StringRef entryPoint) {
  llvm::SmallString<32> mslFile, airFile, libFile;
  int mslFd = 0;
  llvm::sys::fs::createTemporaryFile(entryPoint, "metal", mslFd, mslFile);
  llvm::sys::fs::createTemporaryFile(entryPoint, "metallib", libFile);
  llvm::FileRemover mslRemover(mslFile.c_str());
  llvm::FileRemover libRemover(libFile.c_str());

  { // Write input MSL code to the temporary file.
    llvm::raw_fd_ostream inputStream(mslFd, /*shouldClose=*/true);
    inputStream << mslCode << "\n";
  }

  std::optional<std::string> command = detail::buildMetalCompileCommand(
      targetPlatform, languageVersion, mslFile, libFile);
  if (!command) {
    llvm::errs() << "Unsupported MSL language version " << languageVersion
                 << " for offline Metal compilation\n";
    return nullptr;
  }
  if (failed(runSystemCommand(*command))) {
    return nullptr;
  }

  auto fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(libFile, /*isText=*/false);
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << "Failed to open generated metallib file '" << libFile
                 << "' with error: " << error.message();
    return nullptr;
  }

  return std::move(*fileOrErr);
}

} // namespace mlir::iree_compiler::IREE::HAL
