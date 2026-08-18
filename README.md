# iree-metal

iree-metal is an experimental, open-source JAX backend for Apple GPUs. It is
an independently maintained fork of [IREE](https://github.com/iree-org/iree),
focused on BF16 transformer and vision-model workloads through IREE's Metal
runtime and Apple `simdgroup_matrix` instructions.

This project is not an Apple product and is not an official IREE distribution.
The developer preview is intentionally narrow: Apple M4, CPython 3.12, and
JAX/JAXLIB 0.10.2 through the latest verified release below 0.12. Its wheel tag
has a macOS 13 deployment target, but the release manifest names the newer
macOS version actually validated.

## Install the developer preview

Use a fresh Python 3.12 virtual environment. Do not install the stock
`iree-base-compiler` package in the same environment because it shares the
`iree.compiler` import namespace with this preview.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install iree-metal==3.11.0.dev2026081801
```

The metapackage installs the matching compiler and plugin wheels. A fresh
environment resolves the release's locked JAX/JAXLIB 0.11.1 pair; the plugin
metadata admits the separately verified `>=0.10.2,<0.12` compatibility range.

Then select the backend when running your program:

```bash
JAX_PLATFORMS=iree_metal python -c \
  'import jax; print(jax.devices())'
```

That is the only required runtime setting. The wheels select the tested
`preview-20260818` profile automatically; no tuning flags are needed. For
diagnosis, the one supported rollback is `IREE_METAL_PROFILE=baseline`.

Advanced users who need an offline installation, exact dependency locking,
checksums, or GitHub provenance verification can download the self-contained
bundle from the
[Developer Preview 5 release](https://github.com/niklio/iree-metal/releases/tag/iree-metal-v3.11.0.dev2026081801)
and follow the [developer preview guide](docs/metal/developer-preview.md). The
guide also documents the numeric contract, support boundary, and known
limitations.

## Project status

The release verifier first runs a clean-wheel JAX/PJRT compatibility matrix,
then keeps four independent artifact gates against the exact wheels: 233
semantic operation/shape/dtype cases, nine decoder-training operator and
resource cases, 10 deterministic BF16 forward-and-backward model workloads,
and 12 paired decoder-training performance workloads. Correctness and resource
checks must all pass. The model-board geometric mean must exceed 1.00x its live
artifact-keyed `jax-metal` reference; decoder-training performance must meet
the aggregate, per-phase, per-workload, and stability thresholds recorded in
the release evidence on the disclosed Apple M4 system.
This is a bounded release claim, not universal JAX compatibility. Exact results
and machine-readable evidence are attached to each release.

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
