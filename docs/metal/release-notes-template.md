# iree-metal developer preview VERSION

This is an experimental prerelease of the iree-metal JAX backend for Apple
GPUs. Use it in a dedicated virtual environment and retain a CPU fallback.

## What is in this release

- `iree_base_compiler_iree_metal-VERSION-cp312-abi3-macosx_*_arm64.whl`
- `iree_pjrt_plugin_metal-VERSION-py3-none-macosx_*_arm64.whl`
- `SHA256SUMS`
- `iree-metal-preview-VERSION.manifest.txt`
- `THIRD_PARTY_LICENSES.txt`

The two wheels are one tested unit. Mixing versions is unsupported.

## Tested configuration

- Source revision: `IREE_REVISION`
- Hardware: `HARDWARE`
- macOS and Xcode: `TOOLCHAIN`
- Python: 3.12
- JAX/JAXLIB: 0.6.1
- Deployment target: `DEPLOYMENT_TARGET`

Copy the exact values from the attached manifest. Do not infer broader hardware
or OS support from the wheel's compatibility tag.

## Correctness and performance

Insert the generated release-candidate model table here. It must identify:

- model/revision and whether weights are synthetic or published;
- batch, sequence, hidden dimensions, dtype, and forward versus backward;
- warmup count, timed iterations, statistic, synchronization, and power mode;
- baseline software versions and the physical machine used for both sides;
- output and gradient tolerances; and
- all failures or excluded shapes, not just the aggregate.

Link the immutable JSON/CSV inputs used to render the table. Performance numbers
from the active optimization campaign are not release evidence until rerun from
these exact wheels.

## Install and verify

Follow [developer-preview.md](developer-preview.md). Verify `SHA256SUMS`, install
both wheels together in a new environment, and explicitly select
`JAX_PLATFORMS=iree_metal`.

## Known limitations

Copy the limitations from the preview guide and add any release-specific
correctness, crash, compilation, or performance issues. Every known workaround
should name the flag and its default value.

## Feedback

Use the iree-metal developer preview issue template and attach the manifest plus
a sanitized minimal reproducer. This preview has no production support or API
stability commitment.
