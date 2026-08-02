# iree-metal developer preview

iree-metal is an experimental JAX backend for Apple GPUs. The preview is for
developers who can tolerate sharp edges, provide reduced bug reports, and keep
a CPU fallback environment. It is not a drop-in production replacement for an
Apple-supported framework.

## Support boundary

The first preview deliberately has a narrow validation envelope:

- Apple M4 hardware. Other Apple GPU generations are not yet claimed.
- The exact macOS and Xcode build versions in the attached manifest. Wheels use
  a macOS 13 deployment target, but that tag alone is not evidence that older
  macOS releases were tested.
- CPython 3.12.
- JAX and JAXLIB 0.6.1 exactly.
- BF16 transformer and vision-model training shapes as the primary optimized
  workload. General JAX coverage is incomplete.

Each release manifest is authoritative for source and submodule revisions,
build host, deployment target, Python, toolchain, JAX, and the active compiler
profile.

## Install the offline bundle

Download `iree-metal-preview-3.11.0.dev20260802-macos-arm64.tar.gz` from the
GitHub prerelease. The bundle contains the two project wheels and all locked
runtime dependencies needed for an offline installation. Do not mix wheels
from different releases and do not install stock `iree-base-compiler` in the
same environment; the two compiler distributions share the `iree.compiler`
import namespace.

```bash
shasum -a 256 -c \
  iree-metal-preview-3.11.0.dev20260802-macos-arm64.tar.gz.sha256
tar -xzf iree-metal-preview-3.11.0.dev20260802-macos-arm64.tar.gz
cd iree-metal-preview-3.11.0.dev20260802
shasum -a 256 -c SHA256SUMS

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index \
  --find-links . \
  --find-links dependencies \
  ./iree_base_compiler_iree_metal-*.whl \
  ./iree_pjrt_plugin_metal_iree_metal-*.whl
python -m pip check
```

For an online install from individually downloaded assets, pass
`-r requirements-macos-arm64-py312.txt --require-hashes` along with the two
local project wheels.

GitHub-built assets can also be checked against their build provenance:

```bash
gh attestation verify \
  iree_base_compiler_iree_metal-*.whl \
  --repo niklio/iree-metal
```

## Run JAX

The plugin registers as `iree_metal`. The only required runtime selection is:

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

No campaign environment flags are required. The preview wheel selects the
versioned `preview-20260802` profile, uses runtime MSL compilation so Xcode's
optional Metal Toolchain is not needed, and applies the tested fusion and
optimization settings internally. The first execution includes compilation
and is not representative of steady-state performance.

The single supported diagnostic rollback is:

```bash
IREE_METAL_PROFILE=baseline JAX_PLATFORMS=iree_metal python app.py
```

The baseline profile keeps the safe Metal compilation/fusion settings but
turns off the fork's seven preview optimization gates. Individual legacy gates
remain implementation details for compiler developers and are not part of the
public compatibility interface.

## Numeric contract

The optimized profile's causal-attention bounds transformations assume finite
model inputs. They skip mathematically masked tiles, which is equivalent for
finite values but can change NaN/Inf propagation through expressions such as
`0 * NaN`. Use `IREE_METAL_PROFILE=baseline` when diagnosing non-finite values
or when strict propagation of arbitrary non-finite inputs is required.

Output and gradient tolerances are recorded in the attached verifier evidence.
A successful compile or a fast timing does not by itself establish numerical
correctness.

## What the verifier measures

The independent release verifier runs 166 operation/shape/dtype cases through
the exact installed wheels. For each case it records compilation, execution,
timeouts, output comparison, and synchronized execution timing against the
disclosed `jax-metal` baseline. Its geometric execution-time ratio is a
diagnostic comparison across that case board, not an end-to-end model, API
coverage, or universal performance-parity claim.

Release evidence is sanitized to remove credentials and private filesystem
paths. Benchmark claims are valid only for the disclosed hardware, software,
inputs, tolerances, synchronization, and aggregation method.

Maintainers package a passing run with
`build_tools/iree_metal/package_verifier_evidence.py`. The command rejects a
candidate manifest with runtime overrides, wheel hashes that differ from the
release wheelhouse, an incomplete or failing 166-case board, a corrupt evidence
database, private filesystem paths, or common credential patterns.

## Known limitations

- Dynamic shapes, multiple devices, distributed execution, and broad JAX test
  compatibility are not release claims.
- Performance is shape-sensitive and compilation can be substantial.
- Only the exact JAX version and wheel pair in the release are supported.
- Older macOS releases and non-M4 Apple GPUs are not validated by this preview.
- The forked compiler distribution still shares an import namespace with stock
  IREE and therefore requires a dedicated virtual environment.
- The optimized profile has the finite-input causal-attention contract above.

## Report a problem

Use the iree-metal preview issue template. Attach the release manifest and the
smallest sanitized reproducer. Include:

```bash
JAX_PLATFORMS=iree_metal python -c \
  'import jax, platform; print(platform.platform()); print(jax.__version__); print(jax.devices())'
```

Do not attach model weights, credentials, compiler dumps containing private
source, or other sensitive data.

## Build from source

Use a clean recursive clone on Apple silicon with Python 3.12, CMake, Ninja,
and Xcode:

```bash
git clone --recursive https://github.com/niklio/iree-metal.git
cd iree-metal
git checkout iree-metal-v3.11.0.dev20260802
export IREE_METAL_VERSION=3.11.0.dev20260802
export IREE_METAL_PYTHON=python3.12
./build_tools/iree_metal/build_preview_wheels.sh
./build_tools/iree_metal/smoke_test_wheels.sh ./wheelhouse
```

The build rejects dirty or mismatched submodules, verifies and applies the
repository's locked third-party source overlays, embeds no private checkout
paths, pins and hashes build/runtime dependencies, normalizes wheel archives,
records provenance, and produces an SPDX SBOM. It reverses the overlays only if
their trees remain unchanged. Release tags are built again in GitHub Actions;
execution and autodiff are gated separately on a physical M4.
