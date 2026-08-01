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
plugin_wheels=("${wheelhouse}"/iree_pjrt_plugin_metal-*.whl)
if [[ ${#compiler_wheels[@]} -ne 1 || ${#plugin_wheels[@]} -ne 1 ]]; then
  echo "error: expected one compiler wheel and one plugin wheel in ${wheelhouse}" >&2
  exit 2
fi

"${python_bin}" -m venv "${smoke_root}/venv"
venv_python="${smoke_root}/venv/bin/python"
"${venv_python}" -m pip install --upgrade pip
"${venv_python}" -m pip install "${compiler_wheels[0]}" "${plugin_wheels[0]}"
"${venv_python}" -m pip check

JAX_PLATFORMS=iree_metal "${venv_python}" - <<'PY'
import jax
import jax.numpy as jnp

assert jax.default_backend() == "iree_metal", jax.devices()

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
