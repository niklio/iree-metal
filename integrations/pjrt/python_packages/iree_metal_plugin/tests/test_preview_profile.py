# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
import types
import unittest
from unittest import mock

# Profile resolution is pure Python and does not need a JAX installation. Stub
# only the import surface used by the plugin module so packaging CI can run this
# test before dependency wheels are installed.
jax_module = types.ModuleType("jax")
jax_module.__path__ = []
jax_src_module = types.ModuleType("jax._src")
jax_src_module.__path__ = []
xla_bridge_module = types.ModuleType("jax._src.xla_bridge")
xla_bridge_module.register_plugin = mock.Mock()
sys.modules.setdefault("jax", jax_module)
sys.modules.setdefault("jax._src", jax_src_module)
sys.modules.setdefault("jax._src.xla_bridge", xla_bridge_module)

from jax_plugins import iree_metal


class PreviewProfileTest(unittest.TestCase):
    def test_preview_includes_shipped_physical_apple_attention_gates(self):
        self.assertTrue(
            {
                "IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM",
                "IREE_METAL_APPLE_PHYSICAL_FRAGMENTS",
                "IREE_METAL_APPLE_PHYSICAL_SCORE_WG64",
            }.issubset(iree_metal._PREVIEW_FEATURE_GATES)
        )

    def test_preview_includes_fused_adam_update(self):
        self.assertIn(
            "IREE_METAL_FUSE_ADAM_UPDATE", iree_metal._PREVIEW_FEATURE_GATES
        )

    def test_preview_limits_split_reduction_to_resource_cases(self):
        self.assertIn(
            "IREE_METAL_SPLIT_RESOURCE_ONLY", iree_metal._PREVIEW_FEATURE_GATES
        )

    def test_preview_uses_two_attention_prefetch_stages(self):
        self.assertEqual(
            iree_metal._PREVIEW_PARAMETER_DEFAULTS[
                "IREE_METAL_ATTN_PREFETCH_STAGES"
            ],
            "2",
        )

    def test_preview_includes_validated_vit_gates(self):
        self.assertTrue(
            {
                "IREE_METAL_MSL4_VIT_FFN_COMPACT_EPILOGUE",
                "IREE_METAL_MSL4_VIT_FFN_DW",
                "IREE_METAL_MSL4_VIT_FFN_REDUCTION",
                "IREE_METAL_MSL4_VIT_FFN_RECONSTRUCT_EPILOGUE",
                "IREE_METAL_MSL4_VIT_GELU_SAVED_COMPACT",
                "IREE_METAL_MSL4_VIT_RAW_PAD_MATMUL",
                "IREE_METAL_VIT_FFN18_DIRECT_COOP",
                "IREE_METAL_VIT_POSITIONAL_SCATTER",
                "IREE_METAL_VIT_SIMDGROUP_TRANSPOSE",
            }.issubset(iree_metal._PREVIEW_FEATURE_GATES)
        )
        self.assertEqual(
            iree_metal._PREVIEW_PARAMETER_DEFAULTS[
                "IREE_METAL_MSL4_VIT_FFN_F32"
            ],
            "dynamic",
        )
        self.assertEqual(
            iree_metal._PREVIEW_PARAMETER_DEFAULTS[
                "IREE_METAL_MSL4_VIT_PROJECTION_TILE"
            ],
            "128x32",
        )
        self.assertEqual(
            iree_metal._PREVIEW_PARAMETER_DEFAULTS[
                "IREE_METAL_VIT_TRANSPOSE_HEAD_TILE"
            ],
            "2",
        )
        self.assertEqual(
            iree_metal._PREVIEW_PARAMETER_DEFAULTS[
                "IREE_METAL_VIT_POSITIONAL_SCATTER_WIDTH"
            ],
            "8",
        )

    def test_preview_is_the_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            profile, options = iree_metal._configure_preview_profile()
            self.assertEqual(profile, iree_metal.PREVIEW_PROFILE)
            for name in iree_metal._PREVIEW_FEATURE_GATES:
                self.assertEqual(os.environ[name], "1")
            self.assertEqual(os.environ["IREE_METAL_ATTN_PREFETCH_STAGES"], "2")
            self.assertIn("--iree-metal-compile-to-metallib=false", options)
            self.assertIn("--iree-dispatch-creation-fuse-multi-use=true", options)
            self.assertIn(
                "--iree-dispatch-creation-enable-split-reduction=true", options
            )

    def test_baseline_does_not_enable_feature_gates(self):
        with mock.patch.dict(
            os.environ, {"IREE_METAL_PROFILE": "baseline"}, clear=True
        ):
            profile, _ = iree_metal._configure_preview_profile()
            self.assertEqual(profile, "baseline")
            for name in iree_metal._PREVIEW_FEATURE_GATES:
                self.assertNotIn(name, os.environ)
            self.assertNotIn("IREE_METAL_ATTN_PREFETCH_STAGES", os.environ)

    def test_explicit_gate_and_compiler_options_are_preserved(self):
        with mock.patch.dict(
            os.environ,
            {
                "IREE_METAL_FFN_PAD_M64": "0",
                "IREE_METAL_FUSE_ADAM_UPDATE": "0",
                "IREE_METAL_ATTN_PREFETCH_STAGES": "1",
                "IREE_PJRT_IREE_COMPILER_OPTIONS": "--iree-opt-level=O1",
            },
            clear=True,
        ):
            _, options = iree_metal._configure_preview_profile()
            self.assertEqual(os.environ["IREE_METAL_FFN_PAD_M64"], "0")
            self.assertEqual(os.environ["IREE_METAL_FUSE_ADAM_UPDATE"], "0")
            self.assertEqual(os.environ["IREE_METAL_ATTN_PREFETCH_STAGES"], "1")
            self.assertTrue(options.endswith("--iree-opt-level=O1"))

    def test_unknown_profile_fails_closed(self):
        with mock.patch.dict(
            os.environ, {"IREE_METAL_PROFILE": "mystery"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                iree_metal._configure_preview_profile()


if __name__ == "__main__":
    unittest.main()
