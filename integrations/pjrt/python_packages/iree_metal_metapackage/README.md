# iree-metal

`iree-metal` is an independent, experimental developer preview of a JAX backend
for Apple GPUs. It is maintained at
[niklio/iree-metal](https://github.com/niklio/iree-metal). It is not an official release of iree-org, the IREE project, or the Linux Foundation.

This distribution is a platform-specific metapackage. It installs the exact,
version-matched compiler and PJRT plugin wheels that passed the developer
preview release gates; it contains no executable code itself.

## Supported preview configuration

- Apple M4
- macOS arm64
- CPython 3.12
- JAX and JAXLIB 0.6.1

Other Apple GPUs, Python/JAX versions, dynamic shapes, multiple devices, and
broad JAX compatibility are not release claims.

## Install

Developer preview versions are prereleases, so request them explicitly:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --pre iree-metal
JAX_PLATFORMS=iree_metal python -c \
  'import jax; print(jax.__version__, jax.devices())'
```

The compiler distribution shares the `iree.compiler` import namespace with
stock IREE. Use a dedicated virtual environment and do not install
`iree-base-compiler` alongside this preview.

See the [complete support and numeric
contract](https://github.com/niklio/iree-metal/blob/release/developer-preview/docs/metal/developer-preview.md)
and the release notes before use.
