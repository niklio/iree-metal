#!/bin/bash
# Installs the preview in a disposable venv and exercises JAX compilation,
# execution, and reverse-mode autodiff through the Metal PJRT plugin.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 WHEELHOUSE" >&2
  exit 2
fi

wheelhouse="$(cd "$1" && pwd)"
python_bin="${IREE_METAL_PYTHON:-python3}"
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/iree-metal-smoke.XXXXXX")"

cleanup() {
  rm -rf "${smoke_root}"
}
trap cleanup EXIT

shopt -s nullglob
compiler_wheels=("${wheelhouse}"/iree_base_compiler_iree_metal-*.whl)
plugin_wheels=("${wheelhouse}"/iree_pjrt_plugin_metal_iree_metal-*.whl)
if [[ ${#compiler_wheels[@]} -ne 1 || ${#plugin_wheels[@]} -ne 1 ]]; then
  echo "error: expected one compiler wheel and one plugin wheel in ${wheelhouse}" >&2
  exit 2
fi

"${python_bin}" -m venv "${smoke_root}/venv"
venv_python="${smoke_root}/venv/bin/python"
"${venv_python}" -m pip install \
  --no-index \
  --find-links "${wheelhouse}" \
  --find-links "${wheelhouse}/dependencies" \
  "${compiler_wheels[0]}" "${plugin_wheels[0]}"
"${venv_python}" -m pip check

env \
  -u IREE_PJRT_IREE_COMPILER_OPTIONS \
  -u IREE_PJRT_LOG_LEVEL \
  -u IREE_METAL_PROFILE \
  -u IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM \
  -u IREE_METAL_APPLE_PHYSICAL_FRAGMENTS \
  -u IREE_METAL_APPLE_PHYSICAL_SCORE_WG64 \
  -u IREE_METAL_CAUSAL_BWD_BOUNDS \
  -u IREE_METAL_CAUSAL_FWD_BOUNDS \
  -u IREE_METAL_CAUSAL_TRIANGULAR_GRID \
  -u IREE_METAL_FFN_PAD_M64 \
  -u IREE_METAL_GELU_REMAT \
  -u IREE_METAL_LN_PAIRED \
  -u IREE_METAL_SCATTER_WINDOW_WORKGROUPS \
  JAX_PLATFORMS=iree_metal \
  DEVELOPER_DIR=/iree-metal-preview-does-not-use-xcode-tools \
  "${venv_python}" - <<'PY'
import jax
import jax.numpy as jnp
import os

assert jax.default_backend() == "iree_metal", jax.devices()
expected_gates = (
    "IREE_METAL_APPLE_PHYSICAL_BACKWARD_COMPACT_SMEM",
    "IREE_METAL_APPLE_PHYSICAL_FRAGMENTS",
    "IREE_METAL_APPLE_PHYSICAL_SCORE_WG64",
    "IREE_METAL_CAUSAL_BWD_BOUNDS",
    "IREE_METAL_CAUSAL_FWD_BOUNDS",
    "IREE_METAL_CAUSAL_TRIANGULAR_GRID",
    "IREE_METAL_FFN_PAD_M64",
    "IREE_METAL_GELU_REMAT",
    "IREE_METAL_LN_PAIRED",
    "IREE_METAL_SCATTER_WINDOW_WORKGROUPS",
)
assert all(os.environ.get(name) == "1" for name in expected_gates), os.environ

@jax.jit
def loss(a, b):
    y = a @ b
    return jnp.mean(jnp.tanh(y).astype(jnp.float32))

a = jnp.arange(64 * 96, dtype=jnp.float32).reshape(64, 96).astype(jnp.bfloat16) / 100
b = jnp.arange(96 * 128, dtype=jnp.float32).reshape(96, 128).astype(jnp.bfloat16) / 100
value = loss(a, b)
grad = jax.grad(loss)(a, b)
value.block_until_ready()
grad.block_until_ready()
assert bool(jnp.isfinite(value))
assert bool(jnp.all(jnp.isfinite(grad)))
print("iree-metal smoke test passed", jax.__version__, jax.devices())
PY
