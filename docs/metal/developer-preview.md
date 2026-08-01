# iree-metal developer preview

iree-metal is an experimental JAX backend for Apple GPUs. The developer preview
is intended for people who can tolerate sharp edges, report reduced test cases,
and keep a CPU fallback environment available. It is not a drop-in production
replacement for Apple's supported ML frameworks.

## Preview support contract

The first preview deliberately has a narrow support envelope:

- Apple silicon only; the initial validation target is an M4 Mac.
- The wheel is compiled with a macOS 13 deployment target. That is not yet a
  claim that every macOS 13 configuration works; the actual minimum runtime
  version and older Apple GPU generations remain release validation items.
- CPython 3.12. The compiler uses the stable ABI, but later Python releases are
  not claimed until the complete bundle is tested on them.
- JAX and JAXLIB 0.6.1 exactly. The plugin wheel pins this pair so an unrelated
  package installation cannot silently upgrade it to an incompatible version.
- BF16 transformer and vision-model training shapes are the primary optimized
  workload. General JAX coverage is incomplete.

Each release's manifest is authoritative for the source revision, submodule
revisions, build host, deployment target, Python, Xcode, and JAX versions.

## Install from a GitHub preview release

Create a clean environment and download both wheels from the same release. Do
not mix a plugin wheel and compiler wheel from different releases.
The plugin wheel contains the IREE runtime it needs; a separate runtime wheel is
not required for JAX use.

Do not install the preview compiler and stock `iree-base-compiler` in the same
environment. They expose the same `iree.compiler` import namespace even though
their distribution names differ. Use the dedicated virtual environment below.

```bash
python3.12 -m venv .venv-iree-metal
source .venv-iree-metal/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  ./iree_base_compiler_iree_metal-*.whl \
  ./iree_pjrt_plugin_metal-*.whl
```

Verify the downloaded files before installing:

```bash
shasum -a 256 -c SHA256SUMS
```

## First run

The plugin registers as `iree_metal`. Select it explicitly while the backend is
experimental:

```bash
JAX_PLATFORMS=iree_metal python - <<'PY'
import jax
import jax.numpy as jnp

print(jax.__version__, jax.devices())
x = jnp.ones((128, 128), dtype=jnp.bfloat16)
y = jax.jit(lambda a: a @ a)(x)
print(y.block_until_ready()[0, 0])
PY
```

The first execution includes compilation and is not representative of steady
state performance.

## Known limitations

- Only the exact JAX version named above is supported in the first preview.
- Dynamic shapes, multi-device execution, distributed execution, and broad JAX
  test-suite compatibility are not release claims yet.
- Performance is shape-sensitive. Published benchmark claims apply only to the
  disclosed model, batch, sequence, dtype, direction, and measurement method.
- A successful compile does not establish numerical correctness. Preview release
  candidates are gated on both output and gradient checks for the published
  model board.
- The forked distribution name makes its origin visible to package tooling, but
  it still shares an import namespace with stock IREE and requires a separate
  virtual environment.

## Reporting a problem

Use the iree-metal preview issue template. Include the release manifest, the
output of the following command, and the smallest reproducer that still fails:

```bash
JAX_PLATFORMS=iree_metal python -c \
  'import jax, platform; print(platform.platform()); print(jax.__version__); print(jax.devices())'
```

Do not include model weights, access tokens, private paths, or other secrets in
an issue.

## Building the wheels

On an Apple-silicon Mac with initialized submodules, Python 3.12, CMake, Ninja,
and Xcode installed:

```bash
export IREE_METAL_VERSION=3.11.0.devYYYYMMDD
export IREE_METAL_PYTHON=python3.12
./build_tools/iree_metal/build_preview_wheels.sh
./build_tools/iree_metal/smoke_test_wheels.sh ./wheelhouse
```

The build produces two wheels, `SHA256SUMS`, a license inventory, and a
provenance manifest. Hosted CI repeats the build and structural validation in an
isolated macOS arm64 runner. The execution smoke test remains a separate gate on
a physical M4 because standard hosted runners do not promise Metal GPU access.
