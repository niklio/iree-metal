# Developer preview release checklist

This checklist separates stable release engineering from results that must be
regenerated after the performance campaign lands.

## Can be completed before parity

- [x] Isolate release work from the optimization checkout.
- [x] Give the forked compiler a non-conflicting distribution name.
- [x] Pin the plugin to the matching compiler wheel and tested JAX/JAXLIB pair.
- [x] Script macOS arm64 compiler and PJRT plugin wheels from one source revision.
- [x] Record source/submodule/toolchain provenance and SHA-256 checksums.
- [x] Automate installation, compilation, execution, and autodiff in a fresh venv.
- [x] Add a manually triggered, non-publishing CI workflow.
- [x] Draft installation, limitations, and problem-reporting documentation.
- [x] Add a preview-specific issue template.
- [x] Reject strong credential patterns, unexpected top-level wheel contents,
      and non-system dylib dependencies.
- [ ] Check native debug/string tables for embedded build paths and make the
      wheel build reproducible across two clean workspaces.
- [x] Generate a source license inventory beside release assets.
- [x] Verify the fork marker and reject an environment containing stock IREE.
- [ ] Test an offline install from only the documented release bundle.
- [ ] Decide whether the first distribution is GitHub Releases only or also PyPI;
      reserve package names before announcing them.
- [ ] Draft release notes with an explicit feedback channel and deprecation policy.
- [ ] Configure a sufficiently resourced macOS builder if the standard hosted
      runner cannot complete the compiler wheel within its RAM/disk limits.

## Must wait for the parity campaign's release candidate

- [ ] Rebase this branch onto the selected, clean parity commit.
- [ ] Ensure every modified submodule revision is committed and reachable from a
      public remote; a superproject tag must never depend on a dirty submodule.
- [ ] Build from a fresh recursive clone, not the campaign checkout.
- [ ] Run the full correctness board against the exact wheels being released.
- [ ] Rerun baselines on the disclosed hardware and store hardware metadata.
- [ ] Generate final performance tables and release claims from those results.
- [ ] Test at least two fresh M4 machines or runner environments.
- [ ] Determine the actual minimum supported macOS and Apple GPU generation, or
      retain the narrow M4-only preview claim.
- [ ] Tag the immutable source revision and attach wheels, checksums, manifest,
      licenses, benchmark inputs, and release notes to one prerelease.
- [ ] Install the public assets once more and rerun the smoke/correctness gates.

## Do not do before the release candidate

- Do not bake current throughput numbers into package metadata or evergreen docs.
- Do not promise universal JAX compatibility or support for untested Apple GPUs.
- Do not publish wheels built from the live checkout or dirty submodules.
- Do not let the plugin accept arbitrary newer JAX versions until compatibility
  CI exists for them.
- Do not submit upstream-facing patches as one large performance fork; preserve
  independently reviewable compiler, runtime, PJRT, and third-party changes.
