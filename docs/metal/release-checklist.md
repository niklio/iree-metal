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
- [x] Reject private build paths, normalize wheel archives, and suppress
      nondeterministic Mach-O UUIDs.
- [x] Generate a source license inventory beside release assets.
- [x] Verify the fork marker and reject an environment containing stock IREE.
- [ ] Test an offline install from only the final GitHub release bundle.
- [x] Select GitHub Releases only for the first preview and give both project
      distributions fork-specific names.
- [x] Draft release notes with an explicit feedback channel and deprecation policy.
- [ ] Configure a sufficiently resourced macOS builder if the standard hosted
      runner cannot complete the compiler wheel within its RAM/disk limits.

## Must wait for the parity campaign's release candidate

- [ ] Rebase this branch onto the selected, clean parity commit.
- [x] Carry every third-party change as a checksummed source overlay and verify
      its exact resulting Git tree from public upstream submodule bases.
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
