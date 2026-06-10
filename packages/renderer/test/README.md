# Tests

Test suite for `@headless-three/renderer`, using Node's built-in test runner (`node --test`).

## Running

```bash
npm run build        # or: npm run build:debug  (faster, dev)
npm test
```

The tests require the native binary (`headless_three_renderer.*.node`) to be built and the TypeScript `dist/` to be generated. `npm run build` produces both.

## What is covered

- **`smoke.test.mjs`** — module loads, `Renderer` can be constructed, basic scenes render to correctly-sized PNG buffers, renderer instances are reusable.
- **`corpus.test.mjs`** — generated representative scenes render without crashes and produce visible non-background pixels for transparent layers, skinned/morphed geometry, physical IBL + shadows, instanced points/lines, LOD/groups, and pathological geometry.
- **`scenes.test.mjs`** — scene-level invariants: `rgba` format produces `width * height * 4` bytes, meshes cover reasonable portions of the frame, different materials produce different pixel statistics, PBR scenes show lighting gradients, line/point topologies render without error, empty scene renders background color.

The harness intentionally tests **invariants** (dimensions, color statistics, non-emptiness) rather than exact pixel matches, so the same suite passes across Metal / Vulkan / DX12 / llvmpipe without per-platform snapshot drift.
The generated corpus is a broader no-crash/visibility sweep; committed browser-generated golden images and external glTF/VRM assets remain future work.

## CI software rendering

On Linux CI runners without a GPU, `mesa-vulkan-drivers` provides `lavapipe` (software Vulkan) which wgpu auto-selects. The CI workflow installs it in the test job.

## Adding visual-regression snapshots (optional)

Pixel-accurate snapshots are not committed by default because they differ across GPU backends. If you want to add them for a single canonical platform (e.g. macOS), store reference PNGs under `test/__snapshots__/` and gate the comparison behind `process.env.UPDATE_SNAPSHOTS === '1'` to regenerate.
