# Tests

Test suite for `@headless-three/renderer`, using Node's built-in test runner (`node --test`).

## Running

```bash
npm run build        # or: npm run build:debug  (faster, dev)
npm test
```

The tests require the native binary (`headless_three_renderer.*.node`) to be built and the TypeScript `dist/` to be generated. `npm run build` produces both.

## What is covered

- **`smoke.test.mjs`** — module loads, `Renderer` can be constructed, loader/avatar helper exports are present, basic scenes render to correctly-sized PNG buffers, renderer instances are reusable.
- **`gltf.test.mjs`** — local glTF fixtures load geometry/materials, vertex colors, morph targets, skinning attributes, encoded data-URI textures, GLB bufferView images, synthetic and real external VRM/VRMA extension fixtures, external image files, external buffers, texture color-role metadata, and committed Khronos glTF Sample Assets through the exported local glTF helper, including external and embedded compressed-image failure guidance, the loader plugin/configuration hook, and render through the documented Node loader setup.
- **`corpus.test.mjs`** — generated representative scenes render without crashes and produce visible non-background pixels for transparent layers, custom blend equations/factors, 4x-MSAA alpha-to-coverage, stencil-masked render state, ArrayCamera/CubeCamera outputs, focused depth/distance material output, 2D/equirectangular/cube background textures, signed raw DataTextures, toon alpha-map cutouts, LightProbe lit-material models, mask/object-id/normal render modes, ShadowMaterial fog opt-out, skinned/morphed geometry, avatar-style toon/skinned combinations, physical IBL + shadows, instanced points/lines, LOD/groups, and pathological geometry.
- **`golden.test.mjs`** — optional browser-reference comparison for the generated corpus. It is skipped unless `HEADLESS_THREE_BROWSER_REFERENCE_DIR` points at generated PNGs and `manifest.json`, or the platform default `browser-reference/references/<platform>-<arch>/` directory exists; CI runs the skipped harness so the entry point stays covered until references are committed.
- **`scale.test.mjs`** — many-mesh, raw texture-heavy, encoded texture-heavy, and supported 64-light budget scenes render, and larger visible non-ambient light sets fail clearly.
- **`scenes.test.mjs`** — scene-level invariants: `rgba` format produces `width * height * 4` bytes, meshes cover reasonable portions of the frame, different materials produce different pixel statistics, PBR scenes show lighting gradients, line/point topologies render without error, empty scene renders background color.

The default harness intentionally tests **invariants** (dimensions, color statistics, non-emptiness) rather than exact pixel matches, so the same suite passes across Metal / Vulkan / DX12 / llvmpipe without per-platform snapshot drift.
The generated corpus, minimal local glTF fixtures, and committed Khronos glTF Sample Assets are broader no-crash/visibility sweeps and run in CI; committed browser-generated golden images and broader avatar-scale VRM/VRMA assets remain future work.

## CI software rendering

On Linux CI runners without a GPU, `mesa-vulkan-drivers` provides `lavapipe` (software Vulkan) which wgpu auto-selects. The CI workflow installs it in the test job.

## Browser Reference Corpus

`test/browser-reference/index.html` renders the WebGL-compatible subset of the generated corpus with `THREE.WebGLRenderer` in a real browser and offers one PNG download per fixture plus `manifest.json`. Each fixture pre-clears the full canvas before applying viewport/scissor settings so saved PNGs do not retain pixels from earlier fixtures, fixture-level background/environment controls are applied to the browser scene before rendering, and supported linear `outputColorSpace` aliases are normalized in the generated manifest while unsupported values fail during generation. Renderer-only corpus fixtures, such as Three.js features that WebGLRenderer cannot reference, still run through `test:corpus`. Serve the repository root with any static server, open the page, and save the downloaded files into a local reference directory:

```bash
# from the repository root
python3 -m http.server 4173
# open http://localhost:4173/packages/renderer/test/browser-reference/
```

Compare those browser PNGs against the headless renderer with:

```bash
HEADLESS_THREE_BROWSER_REFERENCE_DIR=/path/to/browser-pngs pnpm run test:golden
```

If references are committed under `packages/renderer/test/browser-reference/references/<platform>-<arch>/`,
`pnpm run test:golden` detects that platform directory without the environment
variable.

Use `HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF` to tune the mean RGBA tolerance for the chosen browser/GPU reference platform. The default is `18`.

## Adding visual-regression snapshots (optional)

Pixel-accurate snapshots are not committed by default because they differ across GPU backends. If you want to add them for a single canonical platform, prefer browser-generated references from `test/browser-reference/`, save them under the platform reference directory, and keep `HEADLESS_THREE_BROWSER_REFERENCE_DIR` for ad hoc local comparisons.

Scale-test counts and platform notes are documented in [`docs/scale-budgets.md`](../../../docs/scale-budgets.md).
