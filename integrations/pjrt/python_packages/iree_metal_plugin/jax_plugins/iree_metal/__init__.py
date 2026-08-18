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

PREVIEW_PROFILE = "preview-20260818"
_PREVIEW_PROFILE_ALIASES = {"preview", PREVIEW_PROFILE}
_PREVIEW_FEATURE_GATES = (
    "IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM",
    "IREE_METAL_APPLE_PHYSICAL_FRAGMENTS",
    "IREE_METAL_APPLE_PHYSICAL_SCORE_WG64",
    "IREE_METAL_CAUSAL_BWD_BOUNDS",
    "IREE_METAL_CAUSAL_FWD_BOUNDS",
    "IREE_METAL_CAUSAL_TRIANGULAR_GRID",
    "IREE_METAL_FFN_PAD_M64",
    "IREE_METAL_FUSE_ADAM_UPDATE",
    "IREE_METAL_GELU_REMAT",
    "IREE_METAL_LN_PAIRED",
    "IREE_METAL_MSL4_VIT_ATTN_VALUE",
    "IREE_METAL_MSL4_VIT_FFN_COMPACT_EPILOGUE",
    "IREE_METAL_MSL4_VIT_FFN_DW",
    "IREE_METAL_MSL4_VIT_FFN_REDUCTION",
    "IREE_METAL_MSL4_VIT_FFN_RECONSTRUCT_EPILOGUE",
    "IREE_METAL_MSL4_VIT_GELU_SAVED_COMPACT",
    "IREE_METAL_MSL4_VIT_STATIC_SLICES",
    "IREE_METAL_NATIVE_EXP",
    "IREE_METAL_SCATTER_WINDOW_WORKGROUPS",
    "IREE_METAL_SPLIT_RESOURCE_ONLY",
    "IREE_METAL_VIT_DEAD_ATTN_PAD_FILL",
    "IREE_METAL_VIT_DEAD_ATTN_SCRATCH_PAD_FILL",
    "IREE_METAL_VIT_DEAD_QKV_PAD_FILL",
    "IREE_METAL_VIT_FFN18_DIRECT_COOP",
    "IREE_METAL_VIT_FORWARD_LARGE_FFN_COPY",
    "IREE_METAL_VIT_FORWARD_LARGE_FFN_TRANSPOSE",
    "IREE_METAL_VIT_FORWARD_SMALL_MATMUL_COPY",
    "IREE_METAL_VIT_FUSED_GELU_OUTPUT",
    "IREE_METAL_VIT_POSITIONAL_SCATTER",
    "IREE_METAL_VIT_SIMDGROUP_TRANSPOSE",
)
# These exact-graph padding rewrites were validated against the JAX 0.6.1 ViT
# lowering used by Preview 4. JAX 0.11.1 produces a different backward graph;
# the raw-pad matmul and dead-row-fill rewrites can silently corrupt unsampled
# gradients there. Keep those two experiments available for explicit developer
# opt-in, but never turn them on from the current-JAX release profile. The
# attention-border, attention-scratch, and QKV fill eliminations above have
# separately passed the current-JAX full-gradient and sustained replay checks.
_LEGACY_VIT_PADDING_GATES = frozenset(
    {
        "IREE_METAL_MSL4_VIT_RAW_PAD_MATMUL",
        "IREE_METAL_VIT_DEAD_ROW_PAD_FILL",
    }
)
_PREVIEW_PARAMETER_DEFAULTS = {
    # Two stages improved every-model HF10 geometric mean while three stages
    # exceeded the useful shared-memory depth and regressed sustained latency.
    "IREE_METAL_ATTN_PREFETCH_STAGES": "2",
    # Exact-shape Metal 4 kernels for the ViT-base FFN and projection paths.
    "IREE_METAL_MSL4_VIT_FFN_F32": "dynamic",
    "IREE_METAL_MSL4_VIT_FFN_TRANSPOSED": "all",
    "IREE_METAL_MSL4_VIT_PROJECTION": "298",
    "IREE_METAL_MSL4_VIT_PROJECTION_TILE": "128x32",
    "IREE_METAL_MSL4_VIT_WALK_BLOCK_M": "4",
    "IREE_METAL_ATTN_PV_SUBGROUP_SEED": "2",
    "IREE_METAL_ATTN_PV_MN_TILE_SEED": "4",
    "IREE_METAL_ATTN_PV_K_TILE_SEED": "1",
    "IREE_METAL_VIT_POSITIONAL_SCATTER_WIDTH": "8",
    # Use a full subgroup for small-output scatters and bounded static BF16
    # embedding gradients. Large/dynamic BF16 batches and large-output F32
    # scatters remain resource-derived so the 32 KiB Metal invariant holds.
    "IREE_METAL_SCATTER_SMALL_OUTPUT_WINDOW_TILE": "32",
    # Amortize the remaining ViT gradient-layout kernel.
    "IREE_METAL_VIT_TRANSPOSE_HEAD_TILE": "2",
}
_PREVIEW_COMPILER_OPTIONS = (
    # Embed MSL and compile it through the runtime Metal API. This keeps the
    # preview usable without installing Xcode's optional Metal Toolchain.
    "--iree-metal-compile-to-metallib=false",
    # Multi-use fusion removes redundant optimizer and integrated-step
    # materializations. Reduction producers retain the fork's stricter
    # correctness boundary in FormDispatchRegions.
    "--iree-dispatch-creation-fuse-multi-use=true",
    "--iree-dispatch-creation-enable-aggressive-fusion=true",
    # Bound partial-reduction tiles for global training losses while retaining
    # aggressive fusion for the transformer body.
    "--iree-dispatch-creation-enable-split-reduction=true",
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
        for name, value in _PREVIEW_PARAMETER_DEFAULTS.items():
            os.environ.setdefault(name, value)
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


def _enable_jax_buffer_donation() -> bool:
    """Enables JAX's existing StableHLO donation lowering for iree_metal.

    JAX currently guards input/output alias generation with a private list of
    platform names instead of querying the PJRT plugin. IREE already consumes
    the resulting ``tf.aliasing_output`` annotation, so register iree_metal
    with that lowering until JAX exposes a public plugin capability API.

    Returns whether the active JAX version exposed the compatibility hook.
    """
    try:
        from jax._src.interpreters import mlir as jax_mlir
    except ImportError:
        return False

    platforms = getattr(jax_mlir, "_platforms_with_donation", None)
    if not isinstance(platforms, list):
        return False
    if "iree_metal" not in platforms:
        platforms.append("iree_metal")
    return True


def _enable_jax_persistent_cache() -> bool:
    """Allows JAX's cache to use this backend's PJRT serialization support.

    Current JAX releases still gate the persistent cache on a hard-coded list
    of built-in platform names. The backend now implements executable
    serialization and deserialization, so preserve JAX's normal checks while
    admitting iree_metal after that legacy platform-name gate rejects it.
    """
    try:
        from jax._src import compilation_cache
    except ImportError:
        return False

    original = compilation_cache.is_cache_used
    if getattr(original, "_iree_metal_enabled", False):
        return True

    def is_cache_used(backend):
        used = original(backend)
        if used or getattr(backend, "platform", None) != "iree_metal":
            return used
        if not compilation_cache._is_cache_enabled():
            return False
        if not getattr(backend, "supports_executable_serialization", True):
            return False
        # The original check has already serialized access through JAX's cache
        # mutex and marked this task checked. Record the corrected result for
        # subsequent calls.
        compilation_cache._cache_used = True
        return True

    is_cache_used._iree_metal_enabled = True
    compilation_cache.is_cache_used = is_cache_used
    return True


def _register_jax_capabilities() -> bool:
    """Uses JAX's public plugin capability API when it is available."""
    try:
        from jax.extend import backend as jax_backend
    except ImportError:
        return False
    register = getattr(jax_backend, "register_backend_capabilities", None)
    if register is None:
        return False

    from . import _dlpack

    register(
        "iree_metal",
        supports_buffer_donation=True,
        supports_persistent_cache=True,
        dlpack_device_type=_dlpack._K_DL_METAL,
        dlpack_exporter=_dlpack.export,
    )
    return True


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
        if not _register_jax_capabilities():
            logger.info(
                "JAX does not yet expose backend capability registration; "
                "using the versioned released-JAX compatibility bridge."
            )
            if not _enable_jax_buffer_donation():
                logger.warning(
                    "This JAX version does not expose its platform donation "
                    "registry; donate_argnums will remain unavailable."
                )
            if not _enable_jax_persistent_cache():
                logger.warning(
                    "This JAX version does not expose its persistent cache hook."
                )
            from . import _dlpack

            if not _dlpack.enable_legacy():
                logger.warning("This JAX version does not expose its DLPack hook.")
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
