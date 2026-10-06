# Release Notes

## 0.4.4

- Lit materials now use Three.js r155+ physical light units for indirect
  diffuse light. AmbientLight, HemisphereLight, LightProbe, and light-map
  irradiance reach Standard, Physical, Lambert, Phong, and Toon materials as
  `irradiance * diffuse / PI`; Standard/Physical diffuse excludes metalness,
  and `aoMap` applies to all of these terms. 0.4.3 used the full ambient color
  without the `1 / PI` and also lit metals with it: a white
  `MeshStandardMaterial` plane under AmbientLight 0.7 and DirectionalLight 1.2
  rendered 241 instead of the 172 of Three.js r180 WebGL, and creator VRM
  faces clipped to white under brighter lights. They now match the browser
  within 2 levels.
- Ambient lights are summed as color × intensity (0.4.3 clamped the summed
  colors) and also apply with an environment map (0.4.3 dropped them). A
  visible AmbientLight, also with zero intensity, turns off the renderer's
  no-light preview fallback. The fallback stays for lit materials in scenes
  without any light source; a light map now counts as a light source.
- HemisphereLight takes its sky direction from the normalized light world
  position, as Three.js does, instead of from its rotation. This also applies
  to Pixiv MToon.
- Standard/Physical direct diffuse no longer has a Fresnel weight, as in
  Three.js `RE_Direct_Physical` (about 4 % brighter for dielectrics).
- Added `test:lighting`, run on every CI platform, with expected values
  checked against Three.js r180 and r183 WebGL output. `test:mtoon` adds
  three-vrm 3.4.4 regressions for shading shift textures (sRGB decode of its
  color-texture assignment, generated-face edge/inside values) and for lit and
  shade factors changed after a `VRMC_materials_mtoon` load and before the
  first frame; these MToon paths already matched in 0.4.3.
- Browser references are unchanged because they are WebGL output. All 109
  golden assertions pass against both committed reference sets; for example,
  `skinned-morphed-plane` drops from a mean diff of 5.90 to 0.28 and
  `mesh-standard-displacement-map` from 7.25 to 2.16. Tests that assumed the
  old ambient scale now multiply ambient intensities by PI, light metallic
  glTF samples with a neutral environment map, or render default-material
  geometry checks as dielectrics.
- `material.dithering`, `material.precision`, `wireframeLinewidth`,
  `wireframeLinecap`, and `wireframeLinejoin` are no longer validated because
  they have no native effect; invalid values are ignored instead of failing.
- Known remaining differences: sRGB output uses a 2.2 gamma curve, so very
  dark tones are up to about 8 levels lighter than WebGL; direct specular
  keeps a Schlick-GGX geometry term, so off-normal metal highlights differ by
  a few levels; and `scene.environment` image-based lighting renders brighter
  than Three.js and also lights Lambert, Phong, and Toon materials, which
  Three.js does not.

## 0.4.3

- Replaced native per-frame byte fingerprints with XXH3 for texture, mesh,
  and uniform caches. Full payload validation is retained, including in-place
  pixel edits with Three.js `needsUpdate`; this is a local cache key, not an asset
  integrity or security hash.
- Cache CopyShader/OutputShader classification by exact GLSL source in a bounded
  map, avoiding repeated whitespace scans of unchanged MToon shaders. Replacing
  the source still changes material support checks immediately.
- Added a reused-renderer regression for live texture and shader replacement.
  Output semantics and golden tolerances are unchanged; existing browser
  references remain authoritative for this performance-only release.

## 0.4.2

- Replaced the generic toon approximation for Pixiv MToon with its own native
  lighting path: authored shade color/texture, shading shift and toony factors,
  rim/matcap textures and expression factors, and VRM0 shade clamping. Ambient
  irradiance now uses the same Lambert normalization as Pixiv WebGL, avoiding
  washed-out skin and clothing. World- and screen-coordinate outlines extrude
  back faces using the authored width texture and outline color/lighting mix.
- Shade/rim maps use direct GPU texture uploads and color decoding instead
  of repeated CPU PBR texture packing. Constant-color outlines skip lighting
  textures. Independent texture transforms and UV channels have regressions.
- Fixed transformed SkinnedMesh geometry and normals: CPU skinning returns
  mesh-local vertices, so the object world transform must still be applied.
- Added targeted regressions for lighting, dynamic factors/textures, outlines,
  and transformed skinning. UV animation and debug modes remain unsupported.
- A same-model, same-pose WebGL comparison at 600×720 reduced mean RGB error
  from 44.04 to 1.65 out of 255 in the consuming NPCify application. Its private
  model is not committed. Regenerated all 104 WebGL corpus images with Three.js
  r183 and passed all 109 golden assertions without changing tolerances. The
  generator now accepts r183's unpacked distance shader as well as the older
  packed output. Existing committed Linux x64/macOS arm64 references also pass
  and remain unchanged; other platforms retain the documented no-reference
  skip path. Targeted MToon tests now run on every CI platform.

## 0.4.1

- Added a native toon surface adapter for Pixiv `MToonMaterial`, fixing real
  VRM loads failing at the `ShaderMaterial` boundary. Base-color textures,
  color/opacity expression bindings, alpha testing, normal/emissive maps,
  sidedness, and vertex-color opt-out use the existing native material path.
  The original material and its uniforms are retained, including live updates.
- MToon parity remains partial: shading uses native Three.js toon lighting;
  authored shade ramps, rim lighting, matcaps, UV animation, and extruded
  outlines are not translated. Outline draw groups are omitted instead of
  incorrectly painting their unextruded surfaces over the avatar.
- Added tests using the real optional Pixiv loader and committed Seed-san
  avatar, plus texture-alpha, live color/opacity, vertex-color, outline-group,
  and unsupported generic shader regression coverage.

## 0.4.0

- Moved pooled render, conversion completion, and diagnostic readback waits off
  N-API/libuv async work onto one renderer-owned media thread. Pooled render and
  conversion now wait once for the exact final submission instead of creating
  two global completion bubbles. Synchronous reservation also occurs before
  JS scene extraction, preserving fixed-capacity `error`/`drop-newest`
  scheduling under overload.
- Added capability-gated `i420-planes` and `GpuFramePool.renderI420()`: GPU
  BT.601 limited-range conversion, tight Y/U/V packing, only 1.5 B/pixel CPU
  readback, preallocated per-slot GPU/readback resources, and reusable exact
  caller buffers. Added a real optional `@roamhq/wrtc` source test across every
  packaged platform, running on the production Node 20 runtime. Linux and
  macOS additionally verify the sink dimensions and bytes. Windows cannot make
  that second assertion because `@roamhq/wrtc` 0.10.0 corrupts RTCVideoSink
  dimensions on Node 20 and 24, while accepting the same source frame. The
  consumer receives a documented plain `{ width, height, data }` frame rather
  than renderer-specific native metadata; the rest of renderer CI remains on
  Node 24.
- On Apple M4/Metal/Node 24 with `UV_THREADPOOL_SIZE=1`, the released 0.3.0
  1080p NV12 pool averaged 3.169 ms; 0.4.0 averaged 1.598 ms after removing the
  second completion bubble. An unrelated PBKDF2 probe changed from 0.535 ms
  idle / 5.212 ms during a 4K pooled frame to 0.409 ms / 0.038 ms. The new
  1080p packed-I420 benchmark read 3,110,400 bytes instead of 8,294,400 bytes.
- In the same environment, a 100-frame `@roamhq/wrtc` run averaged 2.986 ms for
  legacy RGBA readback + libyuv conversion + `onFrame`, versus 1.772 ms for GPU
  packed I420 + `onFrame` (1.69x throughput, with renderer caller-buffer reuse).
- Added a Linux Vulkan single-device prerequisite bootstrap that safely enables
  the external-memory, DMA-BUF, DRM-modifier, foreign-queue, and semaphore-fd
  extensions when available, and reports that state separately from support.
  DMA-BUF/encoder surfaces remain unavailable until a VA-created surface can be
  imported, synchronized, encoded, lifetime-tested, and fd-leak-tested on
  matched AMD amdgpu/VCN hardware; no unusable VkImage or fd is exposed.
- In a production-equivalent container on the matched AMD amdgpu/VCN host,
  Vulkan reported `encoderSurface.prerequisitesReady=true` while correctly
  keeping encoder-surface/DMA-BUF support false. All 10 GPU media tests passed.
  A 720x720, 120-frame run (after 20 warm-up frames) reduced mean/p95 output
  time from 1.702/2.302 ms for legacy RGBA to 0.996/1.034 ms for packed I420,
  and reduced readback from 2,073,600 to 777,600 bytes per frame with one fixed
  allocation reused for the remaining 139 submissions.

## 0.3.0

- Added a genuinely asynchronous, fixed-capacity `GpuFramePool` with default
  triple buffering, synchronous pre-libuv reservation, `error`/`drop-newest`
  overflow behavior, reuse statistics, deterministic close, and no per-frame
  output surface allocation after warm-up.
- Added real GPU compute conversion to truthful `nv12-planes` and optional
  `p010-planes` outputs. Both expose separate Y and interleaved UV textures,
  BT.709 limited-range/centered-chroma metadata, and validated plane content;
  P010 uses upper-10-bit word placement.
- Added explicit per-plane native handle, dimensions, logical row semantics,
  backend state restoration, external-use acknowledgement, and unsafe-slot
  retirement contracts.
- Kept DMA-BUF, IOSurface/CVPixelBuffer, shared D3D12, and encoder-native
  multi-planar surface capabilities false with wgpu 29-specific blockers rather
  than presenting separate textures as portable encoder surfaces.

## 0.2.0

- Added capability-gated `Renderer.renderGpuFrame()` output for a leased,
  submission-complete native GPU texture without CPU RGBA readback. Borrowed
  Metal, Vulkan, and D3D12 handles have explicit same-device lifetime rules.
- Added the future DMA-BUF lease surface with precise unsupported capability
  reporting. Current wgpu-managed Vulkan textures are not falsely advertised as
  exportable because their allocations lack external-memory flags.
- Split large Rust, TypeScript, JavaScript test, and WGSL sources into real
  modules and added a repository-wide 800-line source guard.

## 0.1.11

- Package metadata now targets `0.1.11` for a compatibility release that restores conformance coverage against Three.js `0.183.x`.
- The renderer backend and conformance suite now cover the CommonRenderer timestamp, bind group, DOM element, XR, and example module surface changes introduced by the Three.js `0.183.x` upgrade.

## 0.1.10

- Package metadata now targets `0.1.10` for a metadata-only npm publish that refreshes the npm README and keyword list from the GitHub package metadata.
- The renderer package README is now kept below npm registry README metadata limits and links to the full GitHub compatibility and loader documentation.

## 0.1.9

- Reusable `Renderer` instances now retain native mesh buffers after a seed render and send compact native mesh references for unchanged, cacheable geometry on later frames. This reduces repeated JS-to-native geometry payloads for transform-heavy animation with mostly static mesh attributes.
- The cache is conservative: meshes that need native vertex re-preparation for displacement, normal maps, bump maps, clearcoat normal maps, or anisotropy continue sending full geometry payloads.
- Package metadata now targets `0.1.9` for the root package and optional native binary packages.
