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
- Base color, opacity, premultiplied alpha, base/matcap/emissive/light/physical color map sRGB color-space handling, output color-space controls, base color `flipY`, normal/bump maps, metallic/roughness/specular maps, AO/light maps with selected UV channels, Phong specular maps with selected UV channels, toon gradient maps, alpha/displacement maps, alpha test, alpha hash, transparency sorting, side/culling, and texture wrap modes.
- Base color, normal, bump, displacement, metallic/roughness, emissive, AO, light, specular, alpha, and physical-extension map texture UV transforms (`offset`, `repeat`, `rotation`, `center`, `flipY`, and explicit matrix).
- Base/background, normal/bump, metallic/roughness, emissive, AO/light, alpha, Phong specular, and packed physical-extension texture-group sampler wrap modes and `magFilter`/`minFilter` (`NearestFilter`/`LinearFilter` family).
- Material `depthTest`, `depthWrite`, `colorWrite`, `polygonOffset`, `alphaHash`, `premultipliedAlpha`, stencil state, built-in blending modes (`NoBlending`, `NormalBlending`, `AdditiveBlending`, `SubtractiveBlending`, and `MultiplyBlending`), and custom blend equations/factors in the main pass.
- Material-local `clippingPlanes` plus render-option global `clippingPlanes` in the main pass, up to eight world-space planes, including local `clipIntersection`.
- Clearcoat, sheen, anisotropy, specular intensity/color, transmission/thickness, IOR, attenuation, and related physical material maps.
- Ambient, directional, point, spot, hemisphere, RectAreaLight approximation, and LightProbe diffuse lighting, up to 16 direct lights.
- Environment IBL from equirectangular `scene.environment`/reflection probes.
- Scene background colors and 2D texture backgrounds, including `backgroundIntensity` and approximate 2D texture `backgroundBlurriness`; non-default background/environment rotations fail clearly.
- Render-option viewport and scissor rectangles in output pixel coordinates.
- Directional, spot, point, and directional cascaded shadow maps for a single shadow-casting light; additional shadow-casting lights fail clearly.
- ACES Filmic tone mapping, output color-space controls, simple post-processing, custom WGSL fragment bodies, lines, and point billboards.
- `THREE.Sprite`/`SpriteMaterial` CPU billboard expansion with color, opacity, texture maps, center, scale, rotation, size attenuation, layers, and render ordering.
- `LineDashedMaterial` dash/gap patterns for common line, line segment, line loop, and instanced line cases.
- Object/camera layer filtering for renderable objects and lights, plus native draw-order keys for group order, `renderOrder`, material id, projected z, and object/insertion ties.
- `THREE.LOD` auto-update selection for the active camera.
- `THREE.Fog` and `THREE.FogExp2` in standard material shaders, using camera-distance fog.

Distance to the project goal:

- Ready for many static or pre-animated glTF/VRM-style render jobs.
- Not yet full Three.js scene compatibility.
- The remaining work is mostly breadth, edge-case parity, conformance testing, packaging hygiene, and performance, not one single missing renderer feature.

Treat the goal as achieved only when a published compatibility matrix and golden-image regression corpus pass across the supported platforms.

## P0 - Contract, Tests, And Packaging

| Task | Impact | Notes |
|---|---:|---|
| Add golden-image parity tests | High | Use browser Three.js/WebGLRenderer only as an offline/reference test generator. Production runtime remains Node + `wgpu`. |
| Build a curated scene corpus | High | Initial generated invariant corpus covers transparent materials, skinning, morphs, IBL, shadows, points/lines, LOD/groups, and pathological geometry in CI; a minimal committed glTF fixture covers `GLTFLoader` integration in CI. Remaining work is committed glTF Sample Assets, VRM/VRMA fixtures, and browser-generated references. |

## P1 - Scene Graph And Object Coverage

| Task | Impact | Notes |
|---|---:|---|
| `InstancedBufferGeometry` edge cases | High | Basic mesh, point, line, and dashed-line CPU expansion supports common per-instance offset/color attributes; remaining work is arbitrary custom vertex attributes and shader integration. |
| `renderOrder` and sorting edge cases | High | Group order, `renderOrder`, material id, projected z, and object/insertion ties are honored; remaining work is exact Three.js bounding-sphere z, materialVariant, transmissive buckets, and custom sort callback parity. |
| Sprites and billboards | High | `THREE.Sprite`/`SpriteMaterial` CPU billboards support scale, rotation, opacity, texture maps, layers, render ordering, and main-pass clipping; remaining work is exact shadow/WebGL edge-case parity. |
| Points material parity | Medium | `PointsMaterial` CPU billboards support size, perspective size attenuation, maps, alpha maps, vertex colors, main-pass clipping, and common instanced geometry offset/color attributes; remaining work is hardware cap behavior, circular point-sprite edge cases, and shadows. |
| Line material parity edge cases | Medium | Current lines support unlit color, vertex colors, common instanced geometry offset/color attributes, and `LineBasicMaterial.map` texture alpha/UVs; `LineDashedMaterial` dash/gap patterns render for common non-instanced and instanced cases with reconstructed map UVs and interpolated vertex colors. Non-default `linewidth` fails clearly. Remaining work is thick-line support and broader line material option parity. |
| Scene background textures | Medium | 2D texture backgrounds are supported with output color-space conversion and approximate 2D texture blurriness; cube/equirect background mappings fail clearly. Remaining work is rendering cube/equirect backgrounds, cube/equirect blur, newer mapping controls, and exact color-space parity. |
| Clipping planes | Medium | Render-option global planes and material-local `clippingPlanes`/`clipIntersection` affect the main pass for meshes, sprites, points, and lines, up to eight planes; `clipShadows` fails clearly. Remaining work is exact `WebGLRenderer.localClippingEnabled` state, shadow-pass clipping support, alpha-to-coverage smoothing, and golden-tested parity. |

## P1 - Materials And Textures

| Task | Impact | Notes |
|---|---:|---|
| Texture transform parity edge cases | Very high | Base color, normal, bump, displacement, metallic/roughness, emissive, AO, light, specular, alpha, and physical-extension map transforms are supported; remaining work is exact matrix/color-space interactions and golden-tested parity. |
| Texture color-space parity | Very high | Base color, matcap, emissive, light, sheen color, and physical specular color maps honor `THREE.SRGBColorSpace`; material and 2D texture background output conversion supports `THREE.SRGBColorSpace`/`THREE.LinearSRGBColorSpace`; base color maps honor `flipY`; and standard material shaders honor `premultipliedAlpha`. Remaining work is exact color-space handling for remaining texture slots and WebGLRenderer state parity. |
| Additional UV channel coverage | High | Base color maps, normal/bump maps, matcap color maps, displacement maps, metallic/roughness maps, `emissiveMap`, `aoMap`, `lightMap`, `alphaMap`, Phong `specularMap`, and current physical-extension maps honor texture UV channels; future slots still need matching channel support. |
| More texture slots | High | Unsupported iridescence-related physical maps fail clearly. Remaining work is adding those maps and deepening exact displacement edge-case parity. |
| More material classes | High | Material-level `envMap` reflection inputs fail clearly; scene-level environment/reflection probes are the supported IBL path. Remaining work is deepening `ShadowMaterial` color-space/clipShadows parity, `PointsMaterial` cap/shadow parity, `SpriteMaterial` shadow parity, `MeshDistanceMaterial` point-shadow parity, `MeshToonMaterial` exact edge-case parity, `MeshPhongMaterial` environment/reflection parity, plus `LineDashedMaterial` edge cases. |
| Physical material parity gaps | High | Iridescence and dispersion inputs fail clearly. Remaining work is adding/spec-checking those features, deeper attenuation parity, transmission roughness behavior, and clearcoat/normal/specular BRDF edge cases against Three.js. |
| Blending modes and render state | High | `NoBlending`, `NormalBlending`, `AdditiveBlending`, `SubtractiveBlending`, `MultiplyBlending`, `CustomBlending` equations/factors, `depthTest`, `depthWrite`, `colorWrite`, `polygonOffset`, `alphaHash`, `premultipliedAlpha`, and stencil state are honored in the main pass; `alphaToCoverage` fails clearly. Remaining work is alpha-to-coverage support, constant blend edge-case parity, and exact transparent depth-write parity. |
| Shader material strategy | High | `ShaderMaterial`, `RawShaderMaterial`, NodeMaterial, and built-in material `onBeforeCompile` customizations fail clearly unless a headless WGSL fragment override is provided. Current custom WGSL fragments are not Three.js shader parity. Remaining work is a fuller strategy for Three.js shader graph/material translation and `onBeforeCompile` parity. |
| Compressed texture path | Medium | KTX2/Basis/`THREE.CompressedTexture` inputs fail clearly with pre-decode guidance. Remaining work is native compressed texture decode support. |
| Mip/filter/aniso settings | Medium | Base/background, normal/bump, metallic/roughness, emissive, AO/light, alpha, Phong specular, and packed physical-extension texture groups respect wrap modes plus `NearestFilter`/`LinearFilter`-family `magFilter`/`minFilter`; non-default texture anisotropy fails clearly. Remaining work is generated mip chains, native anisotropic filtering, and conflicting per-channel sampler settings inside the same packed physical texture. |

## P1 - Lighting, Shadows, And IBL

| Task | Impact | Notes |
|---|---:|---|
| Multiple shadow-casting lights | Very high | A single shadow-casting directional/spot/point light is supported; additional visible shadow-casting lights fail clearly. Remaining work is native multi-shadow-map rendering. |
| Shadow behavior parity | High | Common `shadow.camera`, square map size, bias, normal bias, cast/receive rules, and alpha-tested casters are supported; non-square shadow map sizes fail clearly. Remaining work is rectangular map sizes, radius/blur behavior, transparency parity, and deeper golden-tested behavior. |
| `RectAreaLight` | Medium | One-sided finite-area approximation is supported for direct lighting. Remaining work is exact Three.js LTC/LUT parity and golden tests. |
| `LightProbe` support | Medium | Three.js `LightProbe` spherical harmonics contribute diffuse indirect lighting; remaining work is exact golden-tested parity with all material/environment combinations. |
| Environment/background rotation controls | Medium | `scene.environmentIntensity`, `scene.backgroundIntensity`, `options.backgroundIntensity`, and approximate 2D texture `backgroundBlurriness` are supported; non-default background/environment rotations fail clearly. Remaining work is native rotation, cube/equirect blur, and exact background/environment mapping controls. |
| PMREM compatibility | Medium | Cube, refraction, and PMREM/CubeUV environment inputs fail clearly. Remaining work is deciding whether to accept Three.js PMREM outputs, prefiltered cubemaps, or keep CPU precompute from source environment maps. |

## P2 - Cameras, Targets, And Render Pipeline Features

| Task | Impact | Notes |
|---|---:|---|
| Array/cube camera support | Medium | `THREE.ArrayCamera` and `THREE.CubeCamera` fail clearly. Remaining work is native support for environment capture and stereo-ish workflows. |
| Render target depth output | Medium | Target `depthTexture` requests fail clearly. Remaining work is exposing depth texture/readback for downstream tools. |
| Multiple render targets | Medium | Multiple color attachment targets fail clearly. Remaining work is native MRT support for masks, normals, IDs, and deferred-style workflows. |
| MSAA controls | Medium | Render-option and target sample counts greater than 1 fail clearly. Remaining work is configurable sample count and resolve behavior. |
| Viewport/scissor support | Medium | Render-option viewport/scissor rectangles are supported in output pixel coordinates; remaining work is exact WebGLRenderer state-machine parity and broader tiled-render ergonomics. |
| Object ID/mask render modes | Low | Useful for server-side pipelines, segmentation, and picking. |

## P2 - Performance And Scale

| Task | Impact | Notes |
|---|---:|---|
| Persistent resource cache | High | Reuse GPU buffers, textures, pipelines, IBL maps, and bind groups across renders. |
| Incremental scene updates | High | Avoid rebuilding every mesh/material/texture for animation frames. |
| GPU skinning/morphing path | Medium | CPU baking is simple and correct enough for stills, but expensive for dense animated characters. |
| Large scene memory budget tests | Medium | Stress many meshes/textures/lights and confirm predictable failure modes. |
| Parallel texture decode/precompute | Medium | IBL and image decode can dominate render latency. |

## P2 - Node Loader Ergonomics

| Task | Impact | Notes |
|---|---:|---|
| Document Node loader setup | High | Node `GLTFLoader` setup is documented for local files, `FileLoader`/`fetch`, encoded image buffers, path resolution, and optional image polyfills. Remaining work is keeping examples synced with future helper APIs. |
| Provide asset loading helpers | Medium | `createEncodedImageTextureLoader`, `EncodedImageTextureLoader`, `installLocalFileFetch`, and `resolveLocalAssetPath` help Node loaders expose renderer-supported encoded texture buffers. Remaining work is broader helper coverage for embedded images, KTX2/Basis pre-decode, and higher-level glTF/VRM loading. |
| Add end-to-end glTF/VRM examples | High | `examples/render-gltf.mjs` renders a local glTF/GLB asset fully in Node using the exported loader helpers, and tests now render a committed minimal glTF fixture through `GLTFLoader`. Remaining work is committed glTF Sample Assets plus VRM/VRMA-specific examples and fixtures. |

## Acceptance Criteria For "Fully Supports Three.js Scenes"

- Compatibility matrix exists and is kept versioned with every release.
- A representative scene corpus renders without crashes and with acceptable visual diffs against Three.js reference output.
- Unsupported features fail clearly or are explicitly marked unsupported in docs.
- `InstancedMesh`, core material classes, texture transforms/color spaces, layers, render order, multiple shadow lights, sprites, and common glTF/VRM scenes are covered by tests.
- Packaged npm artifacts install and render on Linux x64, Linux arm64, macOS x64, macOS arm64, and Windows x64.
- The public API remains scene-oriented: `render(scene, camera, options)` and target/readback output.
