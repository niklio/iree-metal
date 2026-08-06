# iree-metal

iree-metal is an experimental JAX backend for Apple GPUs, built on an
independently maintained fork of [IREE](https://github.com/iree-org/iree).

> **Developer preview:** This project is not an Apple product or an official
> IREE distribution. Compatibility is intentionally limited while the backend
> is under active development.

## System requirements

- An Apple Silicon Mac (`arm64`)
- macOS 13 or newer
- Python 3.12

The current release is tested on Apple M4. Other Apple Silicon Macs meet the
wheel's installation requirements but are not yet part of the performance
claim.

## Install

Start with a fresh virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install iree-metal
```

The [`iree-metal`](https://pypi.org/project/iree-metal/) package installs the
matching compiler, Metal PJRT plugin, and tested JAX dependencies. Do not add
the stock `iree-base-compiler` package to the same environment because both
packages provide the `iree.compiler` namespace.

## Verify the installation

```bash
JAX_PLATFORMS=iree_metal python -c \
  'import jax; print(jax.devices())'
```

You should see one Apple GPU device. Run an existing JAX program with the same
environment variable:

```bash
JAX_PLATFORMS=iree_metal python your_program.py
```

No tuning flags are required. To diagnose a possible optimization issue, use
the baseline profile:

```bash
IREE_METAL_PROFILE=baseline JAX_PLATFORMS=iree_metal python your_program.py
```

## Project status

The preview is verified against a bounded correctness and BF16 model suite on
Apple M4; it is not a claim of universal JAX compatibility. See the
[developer preview guide](docs/metal/developer-preview.md) for supported
workloads, known limitations, offline installation, checksums, and reproducible
benchmark evidence.

The exact wheels and release evidence are available from
[Developer Preview 3](https://github.com/niklio/iree-metal/releases/tag/iree-metal-v3.11.0.dev2026080401).

## Contributing and support

- [Contributing](CONTRIBUTING.md)
- [Support and compatibility policy](SUPPORT.md)
- [Security policy](SECURITY.md)
- [Governance](GOVERNANCE.md)
- [Code of conduct](CODE_OF_CONDUCT.md)

Bug reports should use the developer-preview issue template and include a
small, sanitized reproducer.

## License and upstream relationship

iree-metal retains IREE's Apache License 2.0 with LLVM Exceptions and all
upstream notices. See [LICENSE](LICENSE).

<details>
<summary>About upstream IREE</summary>

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment,
pronounced "eerie") is an MLIR-based compiler and runtime. Visit
[iree.dev](https://iree.dev/) and the
[upstream repository](https://github.com/iree-org/iree) for the official
project.

</details>
