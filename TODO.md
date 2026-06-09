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

- `THREE.Scene` + `THREE.Camera` input, PNG/RGBA output, reusable renderer, and target-like raw RGBA output.
- Meshes, skinned meshes, geometry groups, vertex colors, indices, normals, UVs, and object world transforms.
- CPU skinning and morph target baking before native rendering.
- `MeshBasicMaterial`, `MeshLambertMaterial`, `MeshStandardMaterial`, and a substantial `MeshPhysicalMaterial` subset.
- Base color, opacity, base color maps, normal maps, emissive maps, metallic/roughness maps, AO maps, alpha test, transparency sorting, side/culling, and texture wrap modes.
- Clearcoat, sheen, anisotropy, transmission/thickness, IOR, attenuation, and related physical material maps.
- Ambient, directional, point, spot, and hemisphere lights, up to 16 lights.
- Environment IBL from equirectangular `scene.environment`/reflection probes.
- Directional, spot, point, and directional cascaded shadow maps, limited to the first shadow-casting light.
- ACES Filmic tone mapping, simple post-processing, custom WGSL fragment bodies, lines, and points.

Distance to the project goal:

- Ready for many static or pre-animated glTF/VRM-style render jobs.
- Not yet full Three.js scene compatibility.
- The remaining work is mostly breadth, edge-case parity, conformance testing, packaging hygiene, and performance, not one single missing renderer feature.

Treat the goal as achieved only when a published compatibility matrix and golden-image regression corpus pass across the supported platforms.

## P0 - Contract, Tests, And Packaging

| Task | Impact | Notes |
|---|---:|---|
| Publish a compatibility matrix | High | Document supported object types, materials, texture features, lights, shadows, render targets, and known gaps. |
| Add golden-image parity tests | High | Use browser Three.js/WebGLRenderer only as an offline/reference test generator. Production runtime remains Node + `wgpu`. |
| Build a curated scene corpus | High | Include glTF Sample Assets, VRM/VRMA fixtures, transparent materials, skinning, morphs, IBL, shadows, points/lines, and pathological geometry. |
| Split smoke tests from conformance tests | Medium | Current tests prove core paths render, but not visual parity against Three.js. |
| Remove or sync duplicate root `api/` sources | High | `api/` and `packages/renderer/api/` have drifted. Keep one source of truth to avoid publishing stale behavior. |
| Fix package naming/docs drift | Medium | Root README still references `headless-three-renderer`; package name is `@headless-three/renderer`. |
| Add release artifact verification | High | Install the packed package in a clean temp project and render a scene on every published target. |

## P1 - Scene Graph And Object Coverage

| Task | Impact | Notes |
|---|---:|---|
| `THREE.InstancedMesh` | Very high | Required for many real Three.js/glTF scenes. Support `instanceMatrix`, `instanceColor`, bounds, and shadows. |
| `InstancedBufferGeometry` attributes | High | Needed for custom instanced content beyond `InstancedMesh`. |
| Object/camera layers | High | Honor `object.layers` and `camera.layers` during scene flattening. |
| `renderOrder` and group sorting semantics | High | Needed for deterministic transparent/overlay scenes. |
| Sprites and billboards | High | Support `THREE.Sprite`/`SpriteMaterial`, including scale, rotation, opacity, and texture maps. |
| Points material parity | Medium | Current points render, but `PointsMaterial.size`, size attenuation, maps, alpha maps, and per-point behavior need parity. |
| Line material parity | Medium | Current lines render, but dashed lines, linewidth limits, alpha maps, and line material options need a contract. |
| LOD handling | Medium | Choose the same object level as Three.js for the active camera. |
| Scene background textures | Medium | Support texture/cube/equirect backgrounds, not only color backgrounds. |
| Fog | Medium | Support scene fog/fogExp2 in material shaders. |
| Clipping planes | High | Global and local clipping planes affect many production scenes. |

## P1 - Materials And Textures

| Task | Impact | Notes |
|---|---:|---|
| Texture transform parity | Very high | Support `offset`, `repeat`, `rotation`, `center`, `matrix`, and `matrixAutoUpdate`. |
| Texture color-space parity | Very high | Honor `texture.colorSpace`, `renderer.outputColorSpace` equivalent settings, `flipY`, premultiply alpha, and sRGB/linear rules. |
| UV channel support | High | `aoMap` and `lightMap` commonly use `uv2`; normal/base maps use `uv`. |
| More texture slots | High | Add `alphaMap`, `bumpMap`, `displacementMap`, `lightMap`, `specularMap`, `matcap`, and relevant physical material extension maps. |
| More material classes | High | Add explicit support for `MeshPhongMaterial`, `MeshToonMaterial`, `MeshNormalMaterial`, `MeshMatcapMaterial`, `MeshDepthMaterial`, `MeshDistanceMaterial`, `ShadowMaterial`, `SpriteMaterial`, `LineDashedMaterial`, and richer `PointsMaterial`. |
| Physical material parity gaps | High | Add/spec-check iridescence, specular intensity/color, dispersion, attenuation, transmission roughness behavior, and clearcoat/normal edge cases against Three.js. |
| Blending modes | High | Support Three.js blending modes, blend equations/factors, premultiplied alpha, depthTest, depthWrite, colorWrite, polygonOffset, and alpha hash/coverage where feasible. |
| Shader material strategy | High | Decide and document scope for `ShaderMaterial`, `RawShaderMaterial`, `onBeforeCompile`, and NodeMaterial. Current custom WGSL fragments are not Three.js shader parity. |
| Compressed texture path | Medium | Support KTX2/Basis or document the required pre-decode pipeline. |
| Mip/filter/aniso settings | Medium | Respect min/mag filters, mipmaps, anisotropy, and sampler settings where `wgpu` supports them. |

## P1 - Lighting, Shadows, And IBL

| Task | Impact | Notes |
|---|---:|---|
| Multiple shadow-casting lights | Very high | Current renderer uses the first shadow-casting directional/spot/point light only. |
| Shadow behavior parity | High | Match Three.js `shadow.camera`, map size, bias, normal bias, radius/blur behavior, cast/receive rules, transparency and alpha-tested casters. |
| `RectAreaLight` | Medium | Common in product/studio scenes. Requires area light approximation or LUT path. |
| `LightProbe` support | Medium | Map Three.js probes into ambient/IBL terms. |
| Environment rotation/intensity/background intensity | Medium | Track newer Three.js environment/background controls. |
| PMREM compatibility | Medium | Decide whether to accept Three.js PMREM outputs, prefiltered cubemaps, or keep CPU precompute from source environment maps. |

## P2 - Cameras, Targets, And Render Pipeline Features

| Task | Impact | Notes |
|---|---:|---|
| Array/cube camera support | Medium | Needed for some environment capture and stereo-ish workflows. |
| Render target depth output | Medium | Expose depth texture/readback for downstream tools. |
| Multiple render targets | Medium | Useful for masks, normals, IDs, and deferred-style workflows. |
| MSAA controls | Medium | Add configurable sample count and resolve behavior. |
| Viewport/scissor support | Medium | Important for tiled renders, atlases, and subviews. |
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
| Document Node loader setup | High | GLTFLoader examples need clear recipes for `FileLoader`, image buffers, path resolution, and optional fetch/polyfills. |
| Provide asset loading helpers | Medium | Optional helpers can normalize texture image data into the renderer-supported form. |
| Add end-to-end glTF/VRM examples | High | Keep simple examples in the repo that run fully in Node. |

## Acceptance Criteria For "Fully Supports Three.js Scenes"

- Compatibility matrix exists and is kept versioned with every release.
- A representative scene corpus renders without crashes and with acceptable visual diffs against Three.js reference output.
- Unsupported features fail clearly or are explicitly marked unsupported in docs.
- `InstancedMesh`, core material classes, texture transforms/color spaces, layers, render order, multiple shadow lights, sprites, and common glTF/VRM scenes are covered by tests.
- Packaged npm artifacts install and render on Linux x64, Linux arm64, macOS x64, macOS arm64, and Windows x64.
- The public API remains scene-oriented: `render(scene, camera, options)` and target/readback output.
