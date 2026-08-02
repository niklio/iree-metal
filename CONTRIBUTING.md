# Contributing to iree-metal

Thanks for helping improve the open Apple-GPU path for JAX. By participating,
you agree to follow the [code of conduct](CODE_OF_CONDUCT.md) and to certify
your commits under the Developer Certificate of Origin using `git commit -s`.

## Before opening a change

Use a GitHub issue for substantial compiler behavior, public API, packaging, or
support-envelope changes. Small fixes can go directly to a pull request. For a
bug, start from the iree-metal preview issue template and reduce it to the
smallest JAX program that still reproduces the problem.

Never include private model weights, credentials, personal paths, or customer
data in issues, traces, compiler dumps, or verifier evidence.

## Development workflow

1. Recursively clone your fork and create a topic branch.
2. Keep changes scoped to one compiler/runtime concern when possible.
3. Add focused IR tests for compiler transformations and a JAX reproducer for
   user-visible behavior.
4. Run the affected IREE test targets plus the wheel structural validator.
5. For Metal runtime or performance work, test on physical Apple silicon and
   report correctness separately from timing.

Release wheels are never built from unrecorded source. Changes to LLVM,
SPIRV-Cross, or StableHLO must regenerate the corresponding patch and locked
tree/checksum in `submodule-patches/`; both are reviewed in the same pull
request.

## Benchmark reports

Disclose hardware, macOS/Xcode versions, power mode, model and revision,
shapes, dtype, forward/backward direction, warmups, timed iterations,
synchronization, statistic, baseline versions, tolerances, failures, and all
compiler profile changes. A compile or timing result without an output check is
not accepted as correctness evidence.

## Upstream-first changes

This repository is based on IREE. If a fix is backend-independent or useful to
the wider IREE community, maintainers may ask that it be proposed to upstream
IREE first or in parallel. Do not represent a fork pull request as an upstream
IREE review.
