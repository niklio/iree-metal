# iree-metal developer preview

iree-metal is an experimental JAX backend for Apple GPUs. The preview is for
developers who can tolerate sharp edges, provide reduced bug reports, and keep
a CPU fallback environment. It is not a drop-in production replacement for an
Apple-supported framework.

## Support boundary

The preview deliberately has a narrow validation envelope:

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

Download `iree-metal-preview-3.11.0.dev2026080801-macos-arm64.tar.gz` from the
GitHub prerelease. The bundle contains the two project wheels and all locked
runtime dependencies needed for an offline installation. Do not mix wheels
from different releases and do not install stock `iree-base-compiler` in the
same environment; the two compiler distributions share the `iree.compiler`
import namespace.

```bash
shasum -a 256 -c \
  iree-metal-preview-3.11.0.dev2026080801-macos-arm64.tar.gz.sha256
tar -xzf iree-metal-preview-3.11.0.dev2026080801-macos-arm64.tar.gz
cd iree-metal-preview-3.11.0.dev2026080801
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
versioned `preview-20260808` profile, uses runtime MSL compilation so Xcode's
optional Metal Toolchain is not needed, and applies the tested fusion and
optimization settings internally. The first execution includes compilation
and is not representative of steady-state performance.

The single supported diagnostic rollback is:

```bash
IREE_METAL_PROFILE=baseline JAX_PLATFORMS=iree_metal python app.py
```

The baseline profile keeps the safe Metal compilation/fusion settings but
turns off the fork's preview optimization gates. Individual legacy gates
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

The independent `iree-metal-verifier` runs two gates through disposable
environments installed from the exact release wheelhouse:

- 233 BF16/F32 semantic cases spanning JIT, reverse-mode autodiff, vectorization,
  reductions, gathers, scatters, transformer primitives, shapes, and index
  patterns. Each records compilation, execution, determinism, timeout, and
  output comparison against the JAX 0.6.1 CPU oracle.
- 10 deterministic BF16 forward-and-backward transformer and vision workloads.
  Each validates loss, gradient norm, persistent per-leaf gradient signatures,
  replay determinism, and synchronized throughput.

The model performance gate compares the release wheels with a live, artifact-keyed
`jax-metal` 0.4.34 reference on the same disclosed Apple M4. For each model, one
GPU lock covers a balanced candidate/reference/reference/candidate crossover.
Every fresh worker runs 16 synchronized steps, discards three warmups and the
slowest remaining sample, then averages the rest. The per-model ratio divides the
geometric mean of the two candidate throughputs by the geometric mean of the two
reference throughputs. Reversing the order controls for monotonic thermal and
GPU-frequency drift without assuming a fixed idle state. The exact reference
binaries and package metadata are hashed into the evidence. The release requires
all 243 checks to pass and the geometric mean of the 10 per-model ratios to be
strictly greater than 1.00x. This is a bounded board comparison, not a claim of
universal API or hardware parity.

Because JAX Metal is the harness reference, a process that fails during backend
startup may be retried once; the failed attempt remains in the evidence. Candidate
failures, numerical failures, and post-startup reference failures are not retried.

Release evidence is sanitized to remove credentials and private filesystem
paths. Benchmark claims are valid only for the disclosed hardware, software,
inputs, tolerances, synchronization, and aggregation method.

Maintainers package a passing run with
`build_tools/iree_metal/package_verifier_evidence.py`. The command rejects a
candidate manifest with runtime overrides, wheel hashes that differ from the
release wheelhouse, any failure among the 233 semantic or 10 model checks,
model geometric-mean parity at or below 1.00x, a corrupt evidence database,
private filesystem paths, or common credential patterns.

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
git checkout iree-metal-v3.11.0.dev2026080801
export IREE_METAL_VERSION=3.11.0.dev2026080801
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
