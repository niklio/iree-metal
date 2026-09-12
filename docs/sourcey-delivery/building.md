---
title: Reproduce these docs
description: Rebuild the static reference from the pinned source revision.
---

# Reproduce these docs

The published reference is built from `niklio/iree-metal` at commit `35a8adbfbafd0958c14a1797d710a5cb9559fd9a`.

## Inputs

- Source repository: [`niklio/iree-metal`](https://github.com/niklio/iree-metal)
- License: [Apache-2.0](https://github.com/niklio/iree-metal/blob/35a8adbfbafd0958c14a1797d710a5cb9559fd9a/LICENSE)
- Site toolchain: [Sourcey 3.6.5](https://github.com/sourcey/sourcey/releases/tag/v3.6.5)
- Extraction format: Doxygen XML

## Build outline

```bash
git clone --branch iree-metal --single-branch https://github.com/niklio/iree-metal.git
cd iree-metal/docs/sourcey-delivery
npm install
npm run build
```

## Coverage notes

The build favors interfaces that explain the fork's end-to-end Metal path. Tests, build scripts, generated shader bodies, and unrelated IREE subsystems are excluded. Undocumented declarations remain visible because they are often the exact seams a contributor needs while tracing compilation or runtime behavior.

The result is static HTML with search metadata, `llms.txt`, and `llms-full.txt`. No documentation runtime or external API is needed after publication.
