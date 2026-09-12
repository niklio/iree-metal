---
title: iree-metal reference
description: Source-linked documentation for the Metal compiler target, runtime HAL driver, and PJRT integration.
---

# iree-metal reference

This site documents the source surfaces that connect JAX programs to Apple's Metal GPU stack in [iree-metal](https://github.com/niklio/iree-metal). Every available definition links to the exact source revision.

The reference is pinned to commit [`35a8adb`](https://github.com/niklio/iree-metal/commit/35a8adbfbafd0958c14a1797d710a5cb9559fd9a). That makes signatures and source locations stable even as the development branch moves.

<CardGroup cols={3}>
  <Card title="Metal compiler" icon="code-bracket" href="/iree-metal/sourcey-docs/api.html">
    SPIR-V-to-MSL conversion, Metal library compilation, and target registration.
  </Card>
  <Card title="Metal HAL" icon="cpu-chip" href="/iree-metal/sourcey-docs/api.html">
    Device, allocator, buffer, command-buffer, executable, and synchronization surfaces.
  </Card>
  <Card title="PJRT bridge" icon="command-line" href="/iree-metal/sourcey-docs/api.html">
    Compiler, platform, tensor, layout, and backend client integration used by JAX.
  </Card>
</CardGroup>

## What is covered

- `compiler/plugins/target/MetalSPIRV`: the compiler path that converts SPIR-V to Metal Shading Language and packages Metal libraries.
- `runtime/src/iree/hal/drivers/metal`: the Metal HAL driver's public and internal C/Objective-C interfaces.
- `integrations/pjrt/src/iree_pjrt`: the PJRT bridge used to expose IREE-backed devices and compilation to JAX.

The API pages are extracted from the repository rather than copied by hand. Use the source links beside declarations when you need implementation context.

## Scope

This is reference material for the experimental fork, not the canonical IREE documentation. For the project overview, supported workflows, and installation notes, start with the [iree-metal README](https://github.com/niklio/iree-metal/blob/35a8adbfbafd0958c14a1797d710a5cb9559fd9a/README.md).
