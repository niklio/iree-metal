"""Platform-tagged metapackage for the iree-metal developer preview."""

from __future__ import annotations

import os
import re
from pathlib import Path

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


HERE = Path(__file__).resolve().parent
VERSION = os.environ.get("IREE_METAL_VERSION", "")
if not re.fullmatch(r"3\.11\.0\.dev[0-9]{8}(?:[0-9]{2})?", VERSION):
    raise RuntimeError(
        "IREE_METAL_VERSION must be a release preview version such as "
        "3.11.0.dev2026080401"
    )


class MetalPreviewWheel(bdist_wheel):
    """Prevent this metadata-only wheel from installing on unsupported hosts."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", "macosx_13_0_arm64"


setup(
    name="iree-metal",
    version=VERSION,
    description="Independent experimental IREE JAX backend for Apple GPUs",
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="iree-metal contributors",
    url="https://github.com/niklio/iree-metal",
    project_urls={
        "Documentation": "https://github.com/niklio/iree-metal/blob/release/developer-preview/docs/metal/developer-preview.md",
        "Issues": "https://github.com/niklio/iree-metal/issues",
        "Source": "https://github.com/niklio/iree-metal",
    },
    license="Apache-2.0 WITH LLVM-exception",
    python_requires=">=3.12,<3.13",
    install_requires=[
        f"iree-base-compiler-iree-metal=={VERSION}",
        f"iree-pjrt-plugin-metal-iree-metal=={VERSION}",
    ],
    packages=[],
    cmdclass={"bdist_wheel": MetalPreviewWheel},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: MacOS X",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
)
