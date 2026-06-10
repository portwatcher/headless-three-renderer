# TODO: Three.js Scene Support Roadmap

## Goal

Render normal Three.js scene graphs in Node.js through the TypeScript adapter,
N-API, and the native `wgpu` renderer.

Non-goal: provide a WebGL context in Node.js or emulate the full
`THREE.WebGLRenderer` API/state machine. The public contract is scene input and
image/target output, not browser WebGL compatibility.

## Current Status

Current state: strong alpha for mesh-centric still rendering.

Already supported in the current implementation:

- `THREE.Scene` or `THREE.Object3D` root + `THREE.Camera` input, PNG/RGBA output, reusable renderer, and target-like raw RGBA output.
- Meshes, skinned meshes, sprites, geometry groups, vertex colors, indices, normals, UVs, and object world transforms.
- `THREE.InstancedMesh` CPU expansion with `instanceMatrix` and `instanceColor`.
- `THREE.InstancedBufferGeometry` CPU expansion for mesh, point, and line geometry with common per-instance offset and color attributes, including dashed line expansion.
- CPU skinning and morph target baking before native rendering.
- `MeshBasicMaterial`, `MeshDepthMaterial`, `MeshDistanceMaterial`, `MeshLambertMaterial`, `MeshMatcapMaterial`, `MeshNormalMaterial`, `MeshPhongMaterial`, `MeshToonMaterial`, `ShadowMaterial`, `MeshStandardMaterial`, and a substantial `MeshPhysicalMaterial` subset.
- Base color, opacity, premultiplied alpha, base/sprite/point/line/matcap/emissive/light/toon gradient/physical color map, supported background texture, and equirectangular/cube scene-environment and reflection-probe LDR IBL sRGB/linear color-space handling, output color-space controls, base color `flipY`, normal/bump maps, metallic/roughness/specular maps, AO/light maps with selected UV channels, Phong specular maps with selected UV channels, toon gradient maps, alpha/displacement maps, alpha test, alpha hash, transparency sorting, side/culling, and texture wrap modes.
- Base color, supported background, sprite/point color and alpha, line, normal, bump, displacement, metallic/roughness, emissive, AO, light, specular, alpha, and physical-extension map texture UV transforms (`offset`, `repeat`, `rotation`, `center`, `flipY`, and explicit matrix), with direct explicit-matrix conformance for base/background, sprite/point color and alpha, line, common mesh material maps, and current physical-extension maps.
- Base/background, normal/bump, metallic/roughness, emissive, AO/light, alpha, Phong specular, toon gradient, matcap color-map, and packed physical-extension texture-group sampler wrap modes, `magFilter`/`minFilter` (`NearestFilter`/`LinearFilter` family), and compatible native texture anisotropy.
- Material `depthTest`, `depthWrite`, `colorWrite`, `polygonOffset`, `alphaHash`, `premultipliedAlpha`, stencil state, built-in blending modes (`NoBlending`, `NormalBlending`, `AdditiveBlending`, `SubtractiveBlending`, and `MultiplyBlending`), and custom blend equations/factors in the main pass.
- Material-local `clippingPlanes`, `THREE.ClippingGroup`, plus render-option global `clippingPlanes`, up to eight world-space planes, including local/group `clipIntersection`, group `clipShadows`, and `options.localClippingEnabled` opt-out.
- Clearcoat, sheen, anisotropy, specular intensity/color, roughness-aware scene-color transmission/thickness, IOR, attenuation, and related physical material maps.
- Ambient, directional, point, spot, hemisphere, RectAreaLight approximation, and LightProbe diffuse lighting, up to 16 direct lights.
- Environment IBL from equirectangular or cube `scene.environment`/reflection probes, plus single shared material-level env maps for supported IBL material paths, including `MeshBasicMaterial` legacy combine/reflectivity/refraction handling.
- Scene background colors plus 2D, equirectangular, and raw or encoded six-face cube texture backgrounds, including `backgroundIntensity`, approximate texture `backgroundBlurriness`, equirectangular/cube `backgroundRotation`, and equirectangular/cube `environmentRotation`; unsupported background rotations fail clearly.
- Render-option viewport and scissor rectangles in output pixel coordinates.
- Flat auxiliary `options.renderMode` passes for white masks and RGB object IDs.
- Directional, spot, point, and directional cascaded shadow maps for a single shadow-casting light; common PCF shadow options including `radius` and PCF-irrelevant `blurSamples` inputs are accepted; custom depth/distance shadow material alpha cutouts and displacement are honored; additional shadow-casting lights fail clearly.
- ACES Filmic tone mapping, output color-space controls, simple post-processing, custom WGSL fragment bodies, lines, and point billboards.
- `THREE.Sprite`/`SpriteMaterial` CPU billboard expansion with color, opacity, texture maps, scene fog, center, scale, rotation, size attenuation, layers, and render ordering; explicit sprite shadow flags fail clearly.
- `LineDashedMaterial` dash/gap/scale patterns and custom `lineDistance` attributes for common line, line segment, line loop, and instanced line cases.
- Object/camera layer filtering for renderable objects and lights, plus native draw-order keys for group order, `renderOrder`, material id, projected geometry bounding-sphere z, and object/insertion ties.
- `THREE.LOD` auto-update selection for the active camera.
- `THREE.Fog` and `THREE.FogExp2` in mesh, shadow, sprite, point, and line material paths, using camera-distance fog with `material.fog` opt-out.

Distance to the project goal:

- Ready for many static or pre-animated glTF/VRM-style render jobs.
- Not yet full Three.js scene compatibility.
- The remaining work is mostly breadth, edge-case parity, conformance testing, packaging hygiene, and performance, not one single missing renderer feature.

Treat the goal as achieved only when a published compatibility matrix and golden-image regression corpus pass across the supported platforms.

## P0 - Contract, Tests, And Packaging

| Task | Impact | Notes |
|---|---:|---|
| Add golden-image parity tests | High | Use browser Three.js/WebGLRenderer only as an offline/reference test generator. Production runtime remains Node + `wgpu`. |
| Build a curated scene corpus | High | Initial generated invariant corpus covers transparent materials, skinning, morphs, avatar-style toon/skinned combinations, IBL, shadows, points/lines, LOD/groups, and pathological geometry in CI; committed glTF fixtures cover `GLTFLoader` geometry/material loading plus encoded data-URI texture loading in CI. Remaining work is committed glTF Sample Assets, external VRM/VRMA fixtures, and browser-generated references. |

## P1 - Scene Graph And Object Coverage

| Task | Impact | Notes |
|---|---:|---|
| `InstancedBufferGeometry` edge cases | High | Basic mesh, point, line, and dashed-line CPU expansion supports common per-instance offset/color attributes; remaining work is arbitrary custom vertex attributes and shader integration. |
| `renderOrder` and sorting edge cases | High | Group order, `renderOrder`, material id, projected geometry bounding-sphere z, and object/insertion ties are honored; remaining work is `materialVariant`, transmissive buckets, and custom sort callback parity. |
| Sprites and billboards | High | `THREE.Sprite`/`SpriteMaterial` CPU billboards support center, scale, rotation, perspective size attenuation controls, opacity, scene fog with `material.fog` opt-out, texture and alpha maps with UV transforms, texture-map sRGB color-space decode, layers, render ordering, and main-pass clipping; explicit shadow flags fail clearly. Remaining work is sprite shadow rendering and exact WebGL edge-case parity. |
| Points material parity | Medium | `PointsMaterial` CPU billboards support size, opacity, perspective size attenuation, depth-independent orthographic sizing, point-sprite map/alpha-map UVs, map/alpha-map UV transforms, sRGB color-space decode for maps, vertex colors, scene fog with `material.fog` opt-out, main-pass clipping, common instanced geometry offset/color attributes, and directional/spot shadow casting from expanded billboards; `receiveShadow` and point-light cube shadows fail clearly. Remaining work is hardware cap behavior, circular point-sprite edge cases, point-light shadow casting, and broader receive-shadow parity. Direct conformance covers both perspective attenuation and orthographic size stability. |
| Line material parity edge cases | Medium | Current lines support unlit color, opacity, scene fog with `material.fog` opt-out, vertex colors, common instanced geometry offset/color attributes, alpha maps with UV channel selection, and `LineBasicMaterial.map` texture RGB/alpha with sRGB color-space decode, texture UV transforms, and `texture.channel` UV selection; `LineDashedMaterial` dash/gap/scale patterns render for common non-instanced and instanced cases with custom `lineDistance` attributes, reconstructed map/alpha-map UVs, texture UV transforms including explicit matrices, `texture.channel` UV selection for maps, and interpolated vertex colors. Non-default `linewidth` fails clearly. Remaining work is thick-line support and broader line material option parity. |
| Scene background textures | Medium | 2D, equirectangular, and raw or encoded six-face cube texture background mappings are supported with output color-space conversion and approximate texture blurriness; equirectangular/cube mappings sample from camera direction. PMREM/CubeUV background mappings fail clearly. Remaining work is exact blur parity, newer mapping controls, and exact color-space parity. |
| Clipping planes | Medium | Render-option global planes, scene-level `THREE.ClippingGroup`, and material-local `clippingPlanes`/`clipIntersection` affect the main pass for meshes, sprites, points, and lines, up to eight planes; nested parent/child `ClippingGroup` union/intersection composition has direct conformance coverage; `material.clipShadows` and group `clipShadows` clip shadow-caster fragments in the shadow pass; and `options.localClippingEnabled: false` disables material-local planes while preserving global/group planes. Remaining work is browser-generated nested `ClippingGroup` golden parity and alpha-to-coverage smoothing. |

## P1 - Materials And Textures

| Task | Impact | Notes |
|---|---:|---|
| Texture transform parity edge cases | Very high | Base color, supported background, sprite/point color and alpha, line, normal, bump, displacement, metallic/roughness, emissive, AO, light, specular, alpha, and physical-extension map transforms are supported, including direct conformance for explicit base/background, sprite/point color and alpha, line, matcap, normal/bump, displacement, emissive, metallic/roughness, AO/light, Phong specular, alpha, and current physical-extension texture matrices. Direct conformance also covers explicit-matrix plus color-space interactions for current color-producing transform slots. Remaining work is exact browser golden-tested parity. |
| Texture color-space parity | Very high | Base color, sprite, point, line, matcap, emissive, light, toon gradient, sheen color, physical specular color, supported background texture maps, and equirectangular/cube scene-environment and reflection-probe LDR IBL inputs honor explicit `THREE.SRGBColorSpace`/`THREE.LinearSRGBColorSpace`; material and texture background output conversion supports `THREE.SRGBColorSpace`/`THREE.LinearSRGBColorSpace`; explicit texture matrices compose with color-space decode for current color-producing transform slots; base color maps honor `flipY`; and standard material shaders honor `premultipliedAlpha`. Remaining work is exact color-space handling for remaining texture slots and WebGLRenderer state parity. |
| Additional UV channel coverage | High | Base color maps, normal/bump maps, matcap color maps, displacement maps, metallic/roughness maps, `emissiveMap`, `aoMap`, `lightMap`, `alphaMap`, Phong `specularMap`, line and dashed-line color maps, and current physical-extension maps honor texture UV channels; future slots still need matching channel support. |
| More texture slots | High | Unsupported iridescence-related physical maps fail clearly. Remaining work is adding those maps and deepening exact displacement edge-case parity. |
| More material classes | High | Scene-level environment/reflection probes are the main supported IBL path, including direct conformance for `MeshPhongMaterial` specular environment reflection and `specularMap` masking. A single shared material-level reflection `envMap` is supported for `MeshBasicMaterial`, `MeshStandardMaterial`, `MeshPhysicalMaterial`, `MeshPhongMaterial`, and `MeshLambertMaterial` through the native IBL path with per-material intensity and shared rotation; Basic legacy multiply/mix/add combine modes, `reflectivity`, and refraction mappings with `refractionRatio` are honored, while unsupported material classes, non-default Phong/Lambert legacy combine/reflectivity, non-Basic refraction mappings, multiple distinct material env maps, and multiple distinct material env-map rotations fail clearly. `Object3D.customDepthMaterial` and `customDistanceMaterial` are honored for shadow caster alpha-tested and displacement material inputs. `MeshToonMaterial` has direct conformance for fallback bands, gradient maps, base-map UV channels, emissive-map UV channels, light-map secondary UVs, and alpha-map cutouts. `ShadowMaterial` covers the current transparent receiver path with color, opacity, scene fog, fog opt-out, and output conversion. Remaining work is deepening `ShadowMaterial` exact parity, `PointsMaterial` cap/receive/point-light shadow parity, `SpriteMaterial` shadow rendering parity, `MeshDistanceMaterial` point-shadow parity, broader custom shadow material translation, `MeshToonMaterial` browser-golden edge-case parity, exact non-Basic material-level envMap/refraction parity, plus `LineDashedMaterial` edge cases. |
| Physical material parity gaps | High | Iridescence and dispersion inputs fail clearly, scene-color transmission uses a roughness/IOR-aware filter, and direct conformance covers transmission volume attenuation color/distance behavior. Remaining work is adding/spec-checking unsupported features, exact attenuation parity against browser references, exact transmission mip/bicubic parity, and clearcoat/normal/specular BRDF edge cases against Three.js. |
| Blending modes and render state | High | `NoBlending`, `NormalBlending`, `AdditiveBlending`, `SubtractiveBlending`, `MultiplyBlending`, `CustomBlending` equations/factors including constant color/alpha factors, `depthTest`, `depthWrite`, `colorWrite`, `polygonOffset`, `alphaHash`, `premultipliedAlpha`, transparent default depth writes, and stencil state are honored in the main pass; `alphaToCoverage` fails clearly. Remaining work is alpha-to-coverage support. |
| Shader material strategy | High | `ShaderMaterial`, `RawShaderMaterial`, NodeMaterial, and built-in material `onBeforeCompile` customizations fail clearly unless a headless WGSL fragment override is provided. Current custom WGSL fragments are not Three.js shader parity. Remaining work is a fuller strategy for Three.js shader graph/material translation and `onBeforeCompile` parity. |
| Compressed texture path | Medium | KTX2/Basis/`THREE.CompressedTexture` inputs fail clearly with pre-decode guidance across material, background, and environment texture slots. Remaining work is native compressed texture decode support. |
| Mip/filter/aniso settings | Medium | Base/background, normal/bump, metallic/roughness, emissive, AO/light, alpha, Phong specular, toon gradient, matcap color-map, and packed physical-extension texture groups respect wrap modes plus `NearestFilter`/`LinearFilter`-family `magFilter`/`minFilter`; material/background sampler slots accept non-default texture anisotropy and use native sampler anisotropy when the effective sampler is linear-filtered, while explicit mipmap arrays still fail clearly across material/background/environment texture slots. Conflicting sampler settings, including anisotropy, between maps packed into one native texture fail clearly. Remaining work is generated mip chains and deeper mipmap parity. |

## P1 - Lighting, Shadows, And IBL

| Task | Impact | Notes |
|---|---:|---|
| Multiple shadow-casting lights | Very high | A single shadow-casting directional/spot/point light is supported; additional visible shadow-casting lights fail clearly. Remaining work is native multi-shadow-map rendering. |
| Shadow behavior parity | High | Common `shadow.camera`, rectangular directional/spot/cascaded map sizes, square point-light cube-face maps, bias, normal bias, `shadow.radius` PCF scaling, PCF-irrelevant `shadow.blurSamples` inputs, cast/receive rules, texture alpha-tested casters, directional/spot point-billboard shadow casters, and custom depth/distance shadow material alpha cutouts and displacement are supported; non-square point-light shadow map sizes and point-light point-billboard shadow casters fail clearly. Remaining work is VSM-specific blur behavior if a VSM shadow mode is added, broader custom shadow material translation, transparency parity, and deeper golden-tested behavior. |
| `RectAreaLight` | Medium | One-sided finite-area approximation is supported for direct lighting. Remaining work is exact Three.js LTC/LUT parity and golden tests. |
| `LightProbe` support | Medium | Three.js `LightProbe` spherical harmonics contribute diffuse indirect lighting and honor visibility plus camera layer filtering; remaining work is exact golden-tested parity with all material/environment combinations. |
| Environment/background rotation controls | Medium | `scene.environmentIntensity`, `scene.backgroundIntensity`, `options.backgroundIntensity`, approximate texture `backgroundBlurriness`, equirectangular/cube `scene.backgroundRotation`, and equirectangular/cube `scene.environmentRotation` are supported for current background/environment texture mappings; unsupported background rotations fail clearly. Remaining work is exact blur parity and exact background/environment mapping controls. |
| PMREM compatibility | Medium | Cube reflection environment inputs are converted through the renderer's CPU IBL precompute path; refraction and PMREM/CubeUV environment inputs fail clearly. Remaining work is deciding whether to accept Three.js PMREM outputs, prefiltered cubemaps, or keep CPU precompute from source environment maps. |

## P2 - Cameras, Targets, And Render Pipeline Features

| Task | Impact | Notes |
|---|---:|---|
| Array/cube camera support | Medium | `THREE.ArrayCamera` and `THREE.CubeCamera` fail clearly. Remaining work is native support for environment capture and stereo-ish workflows. |
| Render target depth output | Medium | Single-output targets with `depthTexture` populate RGBA8 normalized depth readback using depth-tested visible geometry; direct conformance covers near/far ordering, texture writeback, scissor clipping, and alpha-map/alpha-test cutouts. Remaining work is native float depth texture readback and exact WebGL render-target depth semantics. |
| Multiple render targets | Medium | Single-output target texture objects, one-element `target.texture` arrays, and `target.textures[0]` write back RGBA8 data; multiple color attachment targets fail clearly. Remaining work is native MRT support for masks, normals, IDs, and deferred-style workflows. |
| MSAA controls | Medium | Render-option and target sample counts greater than 1 fail clearly. Remaining work is configurable sample count and resolve behavior. |
| Viewport/scissor support | Medium | Render-option viewport/scissor rectangles are supported in output pixel coordinates; remaining work is exact WebGLRenderer state-machine parity and broader tiled-render ergonomics. |
| Object ID/mask render modes | Low | Initial `options.renderMode: "mask"` and `"object-id"` passes are supported for flat segmentation output, including base texture and alpha-map cutouts plus target reverse lookup metadata. Remaining work is MRT integration. |

## P2 - Performance And Scale

| Task | Impact | Notes |
|---|---:|---|
| Persistent resource cache | High | Reuse GPU buffers, textures, pipelines, IBL maps, and bind groups across renders. |
| Incremental scene updates | High | Avoid rebuilding every mesh/material/texture for animation frames. |
| GPU skinning/morphing path | Medium | CPU baking is simple and correct enough for stills, but expensive for dense animated characters. |
| Large scene memory budget tests | Medium | Scale regression tests render many meshes, many unique textures, and the supported 16-light budget in CI, scenes with more than 16 visible non-ambient lights fail clearly, and platform-specific scale budget notes are documented. Remaining work is larger memory ceilings. |
| Parallel texture decode/precompute | Medium | IBL and image decode can dominate render latency. |

## P2 - Node Loader Ergonomics

| Task | Impact | Notes |
|---|---:|---|
| Document Node loader setup | High | Node `GLTFLoader` setup is documented for local files, `FileLoader`/`fetch`, encoded image buffers, PNG/JPEG/WebP data URI and Blob URL images, GLB bufferView image handling, path resolution, and optional image polyfills. Remaining work is keeping examples synced with future helper APIs. |
| Provide asset loading helpers | Medium | `createEncodedImageTextureLoader`, `EncodedImageTextureLoader`, `installLocalFileFetch`, and `resolveLocalAssetPath` help Node loaders expose renderer-supported encoded texture buffers from local files, PNG/JPEG/WebP data URIs, and PNG/JPEG/WebP Blob URLs used by GLB bufferView images. Remaining work is broader helper coverage for KTX2/Basis pre-decode and higher-level glTF/VRM loading. |
| Add end-to-end glTF/VRM examples | High | `examples/render-gltf.mjs` renders a local glTF/GLB asset fully in Node using the exported loader helpers, tests render committed geometry/material and encoded-texture glTF fixtures through `GLTFLoader`, and the generated corpus includes a synthetic avatar-style toon/skinned scene. Remaining work is committed glTF Sample Assets plus VRM/VRMA-specific examples and external fixtures. |

## Acceptance Criteria For "Fully Supports Three.js Scenes"

- Compatibility matrix exists and is kept versioned with every release.
- A representative scene corpus renders without crashes and with acceptable visual diffs against Three.js reference output.
- Unsupported features fail clearly or are explicitly marked unsupported in docs.
- `InstancedMesh`, core material classes, texture transforms/color spaces, layers, render order, multiple shadow lights, sprites, and common glTF/VRM scenes are covered by tests.
- Packaged npm artifacts install and render on Linux x64, Linux arm64, macOS x64, macOS arm64, and Windows x64.
- The public API remains scene-oriented: `render(scene, camera, options)` and target/readback output.
