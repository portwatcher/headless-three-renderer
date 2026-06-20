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
- **`docs.test.mjs`** — public Markdown links resolve, the release checklist keeps compatibility/golden-reference gates documented, and the Khronos glTF Sample Asset Coverage doc stays synchronized with the committed `gltf.test.mjs` fixture constants.
- **`gltf.test.mjs`** — local glTF fixtures load geometry/materials, vertex colors, morph targets, skinning attributes, encoded data-URI textures, GLB bufferView images, minimal synthetic, generated humanoid, practical full-avatar, and real external VRM/VRMA extension fixtures, external image files, external buffers, URI-encoded local filenames with spaces, texture color-role metadata, and committed Khronos glTF Sample Assets through the exported local glTF helper, including large textured architectural scenes, WebP image payloads, diffuse-transmission metadata, volume-scatter metadata, legacy spec-gloss metadata, animation-pointer metadata, clearcoat/transmission variant metadata, material variants, animation clips, external and embedded compressed-image failure guidance, the loader plugin/configuration hook, and render through the documented Node loader setup.
- **`corpus.test.mjs`** — generated representative scenes render without crashes and produce visible non-background pixels for transparent layers, custom blend equations/factors, 4x-MSAA alpha-to-coverage, stencil-masked render state, ArrayCamera/CubeCamera outputs including update capture, focused depth/distance material output, 2D/equirectangular plus renderer-only raw, CubeUV cube, and packed CubeUV background textures, signed raw DataTextures, toon alpha-map cutouts, LightProbe lit-material models, material env maps including renderer-only CubeUV cube and packed CubeUV material env maps, mask/object-id/normal/depth render modes, ShadowMaterial fog opt-out, skinned/morphed geometry, avatar-style toon/skinned combinations, physical IBL + shadows, instanced points/lines, LOD/groups, and pathological geometry.
- **`golden.test.mjs`** — browser-reference comparison for the generated corpus. It uses committed platform references when `browser-reference/references/<platform>-<arch>/` exists, accepts explicit `HEADLESS_THREE_BROWSER_REFERENCE_DIR` inputs, requires the committed Linux x64 references in CI, and otherwise skips platforms without committed references.
- **`scale.test.mjs`** — 1,936-mesh, mixed 144-mesh raw-texture/light, 7,056-instance, 225-map raw texture-heavy, 169-map encoded texture-heavy, and supported 64-light budget scenes render; generated NodePerformanceTest-shaped glTF graphs with 10,000 nodes/meshes/materials/texture definitions load; and larger visible non-ambient light sets fail clearly.
- **`scenes.test.mjs`** — scene-level invariants: `rgba` format produces `width * height * 4` bytes, meshes cover reasonable portions of the frame, different materials produce different pixel statistics, LOD distance/zoom/hysteresis selection behaves consistently, EffectComposer helper flows including CopyShader passes and OutputPass render through target readback, PBR scenes show lighting gradients, line/point topologies render without error, empty scene renders background color.

The default harness intentionally tests **invariants** (dimensions, color statistics, non-emptiness) rather than exact pixel matches, so the same suite passes across Metal / Vulkan / DX12 / llvmpipe without per-platform snapshot drift.
The generated corpus, committed Linux x64 browser references, minimal local glTF fixtures, committed humanoid/full-avatar VRM/VRMA fixtures, and committed Khronos glTF Sample Assets are broader no-crash/visibility sweeps and run in CI; additional platform browser references and tighter visual tolerances for the remaining Linux x64 exceptions remain future work.

## CI software rendering

On Linux CI runners without a GPU, `mesa-vulkan-drivers` provides `lavapipe` (software Vulkan) which wgpu auto-selects. The CI workflow installs it in the test job.

## Browser Reference Corpus

`test/browser-reference/index.html` renders the WebGL-compatible subset of the generated corpus with `THREE.WebGLRenderer` in a real browser and offers one PNG download per fixture plus `manifest.json`. Each fixture pre-clears the full canvas before applying viewport/scissor settings so saved PNGs do not retain pixels from earlier fixtures, fixture-level background/environment controls are applied to the browser scene before rendering, and supported linear `outputColorSpace` aliases are normalized in the generated manifest while unsupported values fail during generation. Renderer-only corpus fixtures, such as Three.js features that WebGLRenderer cannot reference, still run through `test:corpus`. Serve the repository root with any static server, open the page, and save the downloaded files into a local reference directory:

```bash
# from the repository root
python3 -m http.server 4173
# open http://localhost:4173/packages/renderer/test/browser-reference/
```

For reproducible local generation, install Playwright only in the workspace
where you are regenerating references, then run the optional headless wrapper:

```bash
pnpm add -D playwright
pnpm exec playwright install chromium
pnpm -C packages/renderer run generate:browser-reference -- --output test/browser-reference/references/<platform>-<arch>
```

Compare those browser PNGs against the headless renderer with:

```bash
HEADLESS_THREE_BROWSER_REFERENCE_DIR=/path/to/browser-pngs pnpm run test:golden
```

If references are committed under `packages/renderer/test/browser-reference/references/<platform>-<arch>/`,
`pnpm run test:golden` detects that platform directory without the environment
variable.

The harness defaults to a mean RGBA tolerance of `18` and carries two fixture-scoped
tolerances for known browser parity gaps in the Linux x64 corpus. Use
`HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF` to override that policy for ad hoc
local comparisons against a chosen browser/GPU reference platform.
Set `HEADLESS_THREE_REQUIRE_BROWSER_REFERENCES=1` in CI when golden references are expected to be present; without it, `test:golden` keeps the no-reference skip mode for platforms where references have not been committed yet.

## Adding visual-regression snapshots (optional)

Pixel-accurate snapshots are not committed by default because they differ across GPU backends. If you want to add them for a single canonical platform, prefer browser-generated references from `test/browser-reference/`, save them under the platform reference directory, and keep `HEADLESS_THREE_BROWSER_REFERENCE_DIR` for ad hoc local comparisons.

Scale-test counts and platform notes are documented in [`docs/scale-budgets.md`](../../../docs/scale-budgets.md).
