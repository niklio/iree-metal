# iree-metal

iree-metal is an experimental, open-source JAX backend for Apple GPUs. It is
an independently maintained fork of [IREE](https://github.com/iree-org/iree),
focused on BF16 transformer and vision-model workloads through IREE's Metal
runtime and Apple `simdgroup_matrix` instructions.

This project is not an Apple product and is not an official IREE distribution.
The first developer preview is intentionally narrow: Apple M4, CPython 3.12,
and JAX/JAXLIB 0.6.1. Its wheel tag has a macOS 13 deployment target, but the
release manifest names the newer macOS version actually validated.

## Install the developer preview

Download the preview bundle from
[GitHub Releases](https://github.com/niklio/iree-metal/releases). It includes
the two project wheels, a complete hashed dependency wheelhouse, checksums,
license inventory, provenance manifest, and SBOM. Exact-wheel verifier evidence
is attached alongside the bundle.

```bash
tar -xzf iree-metal-preview-3.11.0.dev2026080201-macos-arm64.tar.gz
cd iree-metal-preview-3.11.0.dev2026080201
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index \
  --find-links . --find-links dependencies \
  ./iree_base_compiler_iree_metal-*.whl \
  ./iree_pjrt_plugin_metal_iree_metal-*.whl
JAX_PLATFORMS=iree_metal python -c \
  'import jax; print(jax.devices())'
```

No performance campaign flags are required. The wheels select the tested
`preview-20260802` profile internally. For diagnosis, the one supported
rollback is `IREE_METAL_PROFILE=baseline`.

Read the [developer preview guide](docs/metal/developer-preview.md) before using
the backend. It documents the numeric contract, support boundary, verification
method, and known limitations.

## Project status

The release verifier exercises 166 operation/shape/dtype cases against the
exact wheels, checking outputs and execution. This is a backend diagnostic
suite, not a claim of universal JAX or end-to-end application parity. Exact
results and machine-readable evidence are attached to each release.

The fork carries independently reviewable changes in the compiler, runtime,
PJRT plugin, LLVM/MLIR, SPIRV-Cross, and StableHLO. Third-party modifications
live as checksummed source overlays in `submodule-patches/`; the release build
verifies their exact resulting Git trees. A recursive clone of a release tag
therefore contains every modification without auxiliary fork repositories.

## Contributing and support

- [Contributing](CONTRIBUTING.md)
- [Support and compatibility policy](SUPPORT.md)
- [Security policy](SECURITY.md)
- [Governance](GOVERNANCE.md)
- [Code of conduct](CODE_OF_CONDUCT.md)

Bug reports should use the iree-metal developer-preview issue template and
include a sanitized minimal reproducer plus the attached release manifest.

## License and upstream relationship

iree-metal retains IREE's Apache License 2.0 with LLVM Exceptions and all
upstream notices. See [LICENSE](LICENSE). Changes that are generally useful and
sufficiently isolated are intended to be proposed upstream; preview-specific
packaging and experimental optimization profiles may remain fork-only.

<details>
<summary>About upstream IREE</summary>

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment,
pronounced "eerie") is an MLIR-based compiler and runtime. Visit
[iree.dev](https://iree.dev/) and the
[upstream repository](https://github.com/iree-org/iree) for the official
project.

</details>
