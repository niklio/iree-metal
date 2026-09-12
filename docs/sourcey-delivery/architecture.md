---
title: Architecture map
description: How the documented Metal compiler, HAL runtime, and PJRT layers fit together.
---

# Architecture map

The documented code spans three boundaries. Reading them in execution order makes the reference easier to navigate.

## 1. PJRT integration

JAX discovers a PJRT plugin and calls the standard PJRT C API. The `iree_pjrt` layer translates those calls into IREE compiler and runtime operations.

- `Platform` and backend client classes describe the available device family.
- Compiler helpers turn StableHLO programs and compile options into IREE artifacts.
- Tensor and layout utilities translate host-side shapes, layouts, and buffers.
- Backend-specific clients select CPU, CUDA, ROCm, Vulkan, or the fork's Metal path.

## 2. Compiler target

The `MetalSPIRV` plugin owns the final Metal-specific compilation stages.

- SPIR-V modules are converted to Metal Shading Language.
- MSL is compiled into a Metal library representation.
- Target options and serialization hooks register the backend with IREE's compiler.

## 3. Metal HAL runtime

The HAL driver takes compiled executables and schedules them on a Metal device.

- The driver and device surfaces establish the runtime object graph.
- Allocators and buffers own device-visible storage.
- Command buffers encode transfers and dispatches.
- Executable caches and executable objects load compiled kernels.
- Shared events provide synchronization between submitted work.

## Source boundaries

The API reference intentionally includes internal interfaces because this fork is experimental and its integration points are the useful debugging surface. Definitions remain tied to the pinned commit, so readers can distinguish the documented snapshot from later upstream changes.

