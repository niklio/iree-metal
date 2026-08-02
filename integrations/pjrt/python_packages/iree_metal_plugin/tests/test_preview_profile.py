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
    def test_preview_is_the_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            profile, options = iree_metal._configure_preview_profile()
            self.assertEqual(profile, iree_metal.PREVIEW_PROFILE)
            for name in iree_metal._PREVIEW_FEATURE_GATES:
                self.assertEqual(os.environ[name], "1")
            self.assertIn("--iree-metal-compile-to-metallib=false", options)
            self.assertIn("--iree-dispatch-creation-fuse-multi-use=false", options)

    def test_baseline_does_not_enable_feature_gates(self):
        with mock.patch.dict(
            os.environ, {"IREE_METAL_PROFILE": "baseline"}, clear=True
        ):
            profile, _ = iree_metal._configure_preview_profile()
            self.assertEqual(profile, "baseline")
            for name in iree_metal._PREVIEW_FEATURE_GATES:
                self.assertNotIn(name, os.environ)

    def test_explicit_gate_and_compiler_options_are_preserved(self):
        with mock.patch.dict(
            os.environ,
            {
                "IREE_METAL_FFN_PAD_M64": "0",
                "IREE_PJRT_IREE_COMPILER_OPTIONS": "--iree-opt-level=O1",
            },
            clear=True,
        ):
            _, options = iree_metal._configure_preview_profile()
            self.assertEqual(os.environ["IREE_METAL_FFN_PAD_M64"], "0")
            self.assertTrue(options.endswith("--iree-opt-level=O1"))

    def test_unknown_profile_fails_closed(self):
        with mock.patch.dict(
            os.environ, {"IREE_METAL_PROFILE": "mystery"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                iree_metal._configure_preview_profile()


if __name__ == "__main__":
    unittest.main()
