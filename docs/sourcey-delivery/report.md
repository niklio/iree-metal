# iree-metal documentation report

## Overview

This delivery publishes navigable, source-linked documentation for
[`niklio/iree-metal`](https://github.com/niklio/iree-metal), covering the Metal
compiler target, Metal HAL driver, and PJRT integration. Sourcey 3.6.5 renders
Doxygen XML plus three maintainer guides into a static, searchable site.

The source is pinned to commit
[`35a8adbfbafd0958c14a1797d710a5cb9559fd9a`](https://github.com/niklio/iree-metal/commit/35a8adbfbafd0958c14a1797d710a5cb9559fd9a).
The maintained IREE fork is Apache-2.0 licensed and was active in August 2026.

## Coverage and verification

- 63 documented pages, represented by 64 HTML entry files including the root entry point.
- 60 API-reference entry files and three focused maintainer guides.
- 830 searchable records: 63 pages, 439 sections, 222 functions, 98 variables,
  two enums, five enum values, and one additional type record.
- 64 C, C++, and Objective-C source files across
  `compiler/plugins/target/MetalSPIRV`,
  `runtime/src/iree/hal/drivers/metal`, and
  `integrations/pjrt/src/iree_pjrt`.
- Source links resolve to repository-relative paths and line numbers at the pinned commit.
- A governed Runx validation rebuilt the site, checked 8,626 local references,
  found zero missing targets, found zero stale API landing links, and sealed the
  result under receipt
  `runx:receipt:sha256:92aac9e6694802bf4ef1f368c8fab6a673897e9978a3128b30e24c81d556ba6f`.
- The largest HTML file is 132,860 bytes, comfortably below a 1 MB response limit.

## Maintainer-facing gaps

- **Public boundary:** `EXTRACT_ALL=YES` intentionally exposes useful
  implementation seams, but it also includes internal declarations and anonymous
  numeric pages. Define Doxygen groups or explicit inclusion rules so supported
  interfaces are distinct from implementation details.
- **Missing contracts:** Many declarations have signatures but no explanation of
  parameters, ownership, error behavior, or lifecycle guarantees. The most
  consequential gaps are around buffers, synchronization, compiler jobs, and PJRT
  object conversion.
- **Task-flow navigation:** Navigation is organized by namespace and type.
  Contributor paths such as SPIR-V to MSL, executable loading, device creation,
  and buffer transfer would make the cross-subsystem flow easier to follow.
- **Runnable examples:** Add a minimal end-to-end JAX/PJRT example and focused
  compiler, HAL, and buffer examples. Tested snippets would clarify setup and
  object lifetimes.
- **Objective-C and macros:** Conditional Objective-C declarations and
  macro-expanded IREE APIs may not be represented perfectly by Doxygen. Compare
  those pages with their headers and add parser definitions or hand-written notes
  where extraction loses semantics.

## Reproduction

From the documentation project directory:

```bash
npm install
npm run build
```

`npm run build` regenerates Doxygen XML from the pinned source checkout, builds
the static Sourcey site, and normalizes the API landing-page links.
