# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from importlib import metadata
import os
from pathlib import Path
import platform

import jax._src.xla_bridge as xb

logger = logging.getLogger(__name__)

PREVIEW_PROFILE = "preview-20260802"
_PREVIEW_PROFILE_ALIASES = {"preview", PREVIEW_PROFILE}
_PREVIEW_FEATURE_GATES = (
    "IREE_METAL_CAUSAL_BWD_BOUNDS",
    "IREE_METAL_CAUSAL_FWD_BOUNDS",
    "IREE_METAL_CAUSAL_TRIANGULAR_GRID",
    "IREE_METAL_FFN_PAD_M64",
    "IREE_METAL_GELU_REMAT",
    "IREE_METAL_LN_PAIRED",
    "IREE_METAL_SCATTER_WINDOW_WORKGROUPS",
)
_PREVIEW_COMPILER_OPTIONS = (
    # Embed MSL and compile it through the runtime Metal API. This keeps the
    # preview usable without installing Xcode's optional Metal Toolchain.
    "--iree-metal-compile-to-metallib=false",
    # The current multi-use path can miscompile a transformer gradient graph.
    "--iree-dispatch-creation-fuse-multi-use=false",
    "--iree-dispatch-creation-enable-aggressive-fusion=true",
)


def _verify_compiler_distribution() -> bool:
    """Rejects mixed stock/preview compiler installations.

    The preview compiler has a distinct distribution name but necessarily
    provides the same ``iree.compiler`` import namespace as stock IREE. Check
    the plugin's installed requirements so upstream builds retain their normal
    behavior while preview builds fail clearly if pip produced a hybrid env.
    """
    try:
        requirements = metadata.requires("iree-pjrt-plugin-metal-iree-metal") or []
    except metadata.PackageNotFoundError:
        # Source/editable development may not have distribution metadata yet.
        return False

    is_preview = any(
        requirement.lower().replace("_", "-").startswith(
            "iree-base-compiler-iree-metal"
        )
        for requirement in requirements
    )
    if not is_preview:
        return False

    try:
        metadata.distribution("iree-base-compiler-iree-metal")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "The iree-metal preview requires its matching "
            "iree-base-compiler-iree-metal wheel."
        ) from exc

    try:
        metadata.distribution("iree-base-compiler")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "Stock iree-base-compiler and the iree-metal preview compiler are "
            "installed together and share files. Create a clean virtual environment."
        )

    from iree.compiler import version as compiler_version

    if compiler_version.PACKAGE_SUFFIX != "-iree-metal":
        raise RuntimeError(
            "The installed iree.compiler files do not carry the iree-metal fork "
            "marker. Create a clean virtual environment and reinstall both preview wheels."
        )
    return True


def _configure_preview_profile() -> tuple[str, str]:
    """Applies the versioned preview profile and returns compiler options.

    The old campaign interface required users to coordinate several process
    environment variables. Keep those internal rollout gates for now, but set
    them from one versioned profile so the installed backend works by default.
    Every gate still accepts an explicit value for compiler debugging.
    """
    requested = os.environ.get("IREE_METAL_PROFILE", PREVIEW_PROFILE).strip().lower()
    if requested in _PREVIEW_PROFILE_ALIASES:
        active_profile = PREVIEW_PROFILE
        for name in _PREVIEW_FEATURE_GATES:
            os.environ.setdefault(name, "1")
    elif requested == "baseline":
        active_profile = "baseline"
    else:
        supported = f"{PREVIEW_PROFILE}, preview, baseline"
        raise RuntimeError(
            f"Unsupported IREE_METAL_PROFILE={requested!r}; choose one of: {supported}"
        )

    compiler_options = list(_PREVIEW_COMPILER_OPTIONS)
    extra_options = os.environ.get("IREE_PJRT_IREE_COMPILER_OPTIONS", "").strip()
    if extra_options:
        # Preserve the existing advanced escape hatch. Appending lets an
        # explicit user option take precedence where IREE accepts repeats.
        compiler_options.append(extra_options)
    return active_profile, " ".join(compiler_options)


def probe_iree_compiler_dylib() -> str:
    """Locates the IREE compiler dylib.

    Honors the IREE_PJRT_COMPILER_LIB_PATH env var when set so a locally-built
    (e.g. patched) libIREECompiler can be used instead of the one bundled with
    the installed iree.compiler wheel. Falls back to probing the installed
    package otherwise.
    """
    override = os.environ.get("IREE_PJRT_COMPILER_LIB_PATH")
    if override:
        return override

    # TODO: Move this out of the ctypes API initialization.
    from iree.compiler.api import ctypes_dl

    return ctypes_dl._probe_iree_compiler_dylib()


def _find_native_library() -> Path:
    """Locates the PJRT plugin shared library.

    CMake emits the shared library with a platform-dependent suffix: ".so" on
    Linux and (depending on the target type) ".dylib" or ".so" on macOS. Probe
    the candidates rather than hard-coding a single suffix so the plugin loads
    regardless of how the dylib was named at build time.
    """
    import iree._pjrt_libs.metal as lib_package

    base = Path(lib_package.__file__).resolve().parent
    stem = "pjrt_plugin_iree_metal"
    # On Darwin prefer .dylib but fall back to .so (and vice-versa elsewhere).
    suffixes = (".dylib", ".so") if platform.system() == "Darwin" else (".so", ".dylib")
    for suffix in suffixes:
        candidate = base / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    # Return the conventional path for the platform so the warning below points
    # at the expected location.
    return base / f"{stem}{suffixes[0]}"


def initialize():
    is_preview = _verify_compiler_distribution()
    path = _find_native_library()
    if not path.exists():
        logger.warning(
            f"WARNING: Native library {path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )
    options = {
        "COMPILER_LIB_PATH": str(probe_iree_compiler_dylib()),
    }
    if is_preview:
        active_profile, compiler_options = _configure_preview_profile()
        options["IREE_COMPILER_OPTIONS"] = compiler_options
        # The upstream plugin defaults to debug logging. A packaged preview
        # should be quiet unless the user asks for diagnostics.
        options["LOG_LEVEL"] = os.environ.get("IREE_PJRT_LOG_LEVEL", "error")
        logger.info("iree-metal profile: %s", active_profile)

    xb.register_plugin(
        "iree_metal",
        priority=500,
        library_path=str(path),
        options=options,
    )
