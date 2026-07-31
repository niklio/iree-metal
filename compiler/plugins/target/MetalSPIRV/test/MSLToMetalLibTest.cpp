// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/MetalSPIRV/MSLToMetalLib.h"

#include <optional>

#include "gtest/gtest.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

TEST(MSLToMetalLibTest, LanguageStandards) {
  EXPECT_EQ(detail::buildMetalCompileCommand(MetalTargetPlatform::macOS,
                                             196608u, "input.metal",
                                             "output.metallib"),
            "xcrun -sdk macosx metal -std=metal3.0 -c input.metal -o - | "
            "xcrun -sdk macosx metallib - -o output.metallib");
  EXPECT_EQ(detail::buildMetalCompileCommand(MetalTargetPlatform::iOS, 196609u,
                                             "input.metal", "output.metallib"),
            "xcrun -sdk iphoneos metal -std=metal3.1 -c input.metal -o - | "
            "xcrun -sdk iphoneos metallib - -o output.metallib");
  EXPECT_EQ(detail::buildMetalCompileCommand(MetalTargetPlatform::iOSSimulator,
                                             262144u, "input.metal",
                                             "output.metallib"),
            "xcrun -sdk iphonesimulator metal -std=metal4.0 -c input.metal "
            "-o - | xcrun -sdk iphonesimulator metallib - -o "
            "output.metallib");
  EXPECT_EQ(detail::buildMetalCompileCommand(MetalTargetPlatform::macOS, 0u,
                                             "input.metal", "output.metallib"),
            std::nullopt);
}

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
