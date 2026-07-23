# iree-metal

A fork of [IREE](https://github.com/iree-org/iree) that makes ordinary JAX models run on
Apple GPUs by targeting the GPU's matrix units for `bfloat16` matmuls. It exists because
Apple's `jax-metal` — the tuned way to run JAX on Apple Silicon — is closed and frozen at
JAX 0.4.34, so there was no open, current-JAX path that used the hardware properly.

## What the fork actually changes

Stock IREE compiles a `bf16 @ bf16 → bf16` matmul (what real Hugging Face models emit) to
scalar Metal code that ignores the GPU's matrix units and runs ~2× too slow. The fork's
core change is a compiler pass that promotes the matmul accumulator to f32
(`RaiseContractionAccumulatorToF32`), which is enough to make IREE's cooperative-matrix
codegen emit `simdgroup_bfloat8x8` — Apple's actual matrix-unit instructions — instead of
falling back to scalar loops. That one pass is the bulk of the speedup and is default-on.

Around it, a handful of supporting changes make it hold up on real models:

- **Cooperative-matrix codegen fixes** so the coop pipeline emits matrix-unit code instead
  of silently dropping to scalar, plus a graceful scalar fallback when a graph can't be
  legalized.
- **Operand padding** that zero-pads non-multiple-of-16 matmul dimensions, so odd-shaped
  models (e.g. ViT at sequence length 577) still reach the matrix units — exact, since
  padding a contraction dim with zeros changes nothing.
- **Coverage fixes** for modern architectures: a llama-family path (RoPE + RMSNorm +
  SwiGLU) and a fix for a `bf16`-accumulator store that produced NaNs on grouped-query
  attention.
- **A PJRT Metal plugin** so JAX drives the patched runtime directly.

## Where it stands

Measured on a small suite of standard transformer and vision models (`bf16`, forward and
backward, on an Apple M3 / 16 GB): correct on every model, at roughly **two-thirds of
`jax-metal`'s throughput** — up from about a quarter at the fork point.

The remaining third is whole-graph fusion — keeping intermediates on-chip through a
matmul's K-loop, the way Apple's MPSGraph and XLA do. Isolated matmuls are already at
`jax-metal` parity, so the gap is scheduling, not raw matrix-unit throughput; closing it
is a much deeper codegen effort. Benchmark harness and per-model numbers:
[`~/iree-fork-bench`](https://github.com/niklio/nlearn).

Compiler diffs and submodule patches (LLVM `bf16` MMA, SPIRV-Cross MSL emission) are under
`nlearn-submodule-patches/`.

---

<details>
<summary>Upstream IREE README</summary>

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment, pronounced
"eerie") is an [MLIR](https://mlir.llvm.org/)-based end-to-end compiler and runtime that
lowers ML models to a unified IR scaling from datacenter to mobile and edge. See
[the IREE website](https://iree.dev/) for project details, guides, and build instructions.
Licensed under Apache 2.0 with LLVM Exceptions; see [LICENSE](LICENSE).

</details>
