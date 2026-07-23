# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_cmake_installdir "${CMAKE_INSTALL_LIBDIR}/cmake/IREE")

# Generate export sets.
# See runtime/src/CMakeLists.txt
iree_generate_export_targets(
  EXPORT_SET Runtime
  INSTALL_DESTINATION ${_cmake_installdir}
)

# See compiler/bindings/c/CMakeLists.txt
# nlearn §13: only generate the Compiler export set when the compiler is built. With
# IREE_BUILD_COMPILER=OFF the c_loader links the (stubbed) IREELLVMIncludeSetup which isn't
# in any export set, which fails generate. We consume the build-tree dylib, not installs.
if(IREE_BUILD_COMPILER)
  iree_generate_export_targets(
    EXPORT_SET Compiler
    INSTALL_DESTINATION ${_cmake_installdir}
  )
endif()

if(IREE_BUILD_COMPILER)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/IREECompilerConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/IREECompilerConfig.cmake
    @ONLY
  )
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/IREECompilerConfig.cmake
    DESTINATION ${_cmake_installdir}
    COMPONENT IREECMakeExports
  )
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/IREERuntimeConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/IREERuntimeConfig.cmake
  @ONLY
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/IREERuntimeConfig.cmake
  DESTINATION ${_cmake_installdir}
  COMPONENT IREECMakeExports
)
