#!/usr/bin/python3

# Copyright 2024 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Early splice the _setup_support directory onto the python path.
import os
from pathlib import Path
import sys

THIS_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "_setup_support"))

import iree_pjrt_setup
from setuptools import setup, find_namespace_packages

README = r"""
iree-metal is an experimental JAX PJRT backend for Apple GPUs. Preview wheels
are tested only with the exact compiler, JAX, Python, hardware, and macOS matrix
published at https://github.com/niklio/iree-metal/releases.
"""

# Setup and get version information.
CMAKE_BUILD_DIR_ABS = os.environ.get(
    "IREE_PJRT_CMAKE_BUILD_DIR", os.path.join(THIS_DIR, "build", "cmake")
)


class CMakeBuildPy(iree_pjrt_setup.BaseCMakeBuildPy):
    def build_default_configuration(self):
        print("*****************************", file=sys.stderr)
        print("* Building base runtime     *", file=sys.stderr)
        print("*****************************", file=sys.stderr)
        self.build_configuration(
            CMAKE_BUILD_DIR_ABS,
            extra_cmake_args=(
                "-DIREE_HAL_DRIVER_METAL=ON",
                # The plugin loads libIREECompiler dynamically from the compiler
                # wheel. The lightweight interface stub in integrations/pjrt
                # lets this wheel build without compiling LLVM/MLIR a second time.
                "-DIREE_BUILD_COMPILER=OFF",
                "-DLLVM_PARALLEL_LINK_JOBS=1",
            ),
        )
        print("Target populated.", file=sys.stderr)


iree_pjrt_setup.populate_built_package(
    os.path.join(
        CMAKE_BUILD_DIR_ABS,
        "python",
        "iree",
        "_pjrt_libs",
        "metal",
    )
)


setup(
    name=f"iree-pjrt-plugin-metal{iree_pjrt_setup.PACKAGE_SUFFIX}",
    version=f"{iree_pjrt_setup.PACKAGE_VERSION}",
    author="The IREE Authors and iree-metal contributors",
    license="Apache-2.0 WITH LLVM-exception",
    description="Experimental iree-metal PJRT plugin for Apple GPUs",
    long_description=README,
    long_description_content_type="text/markdown",
    url=iree_pjrt_setup.PROJECT_URL,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=iree_pjrt_setup.PYTHON_REQUIRES,
    packages=[
        "jax_plugins.iree_metal",
        "iree._pjrt_libs.metal",
    ],
    package_dir={
        "jax_plugins.iree_metal": "jax_plugins/iree_metal",
        "iree._pjrt_libs.metal": "build/cmake/python/iree/_pjrt_libs/metal",
    },
    package_data={
        "iree._pjrt_libs.metal": ["pjrt_plugin_iree_metal.*"],
    },
    cmdclass={
        "build": iree_pjrt_setup.PjrtPluginBuild,
        "build_py": CMakeBuildPy,
        "bdist_wheel": iree_pjrt_setup.bdist_wheel,
        "install": iree_pjrt_setup.platlib_install,
    },
    zip_safe=False,  # Needs to reference embedded shared libraries.
    project_urls={
        "Documentation": "https://github.com/niklio/iree-metal/blob/iree-metal/docs/metal/developer-preview.md",
        "Issues": "https://github.com/niklio/iree-metal/issues",
        "Source": "https://github.com/niklio/iree-metal",
    },
    entry_points={
        # We must advertise which Python modules should be treated as loadable
        # plugins. This augments the path based scanning that Jax does, which
        # is not always robust to all packaging circumstances.
        "jax_plugins": [
            "iree-metal = jax_plugins.iree_metal",
        ],
    },
    install_requires=iree_pjrt_setup.install_requires,
)
