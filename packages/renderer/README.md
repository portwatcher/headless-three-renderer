# @headless-three/renderer

Headless `wgpu` renderer for Three.js scenes in Node.js.

This package exists for Node.js environments where WebGL is not available. You build or load a normal Three.js scene graph, pass the `THREE.Scene` or `THREE.Object3D` root and `THREE.Camera` to this package, and the native addon renders it with `wgpu`.

```bash
npm install @headless-three/renderer three
```

```js
import fs from 'node:fs'
import * as THREE from 'three'
import { render } from '@headless-three/renderer'

const scene = new THREE.Scene()
scene.background = new THREE.Color(0.04, 0.045, 0.05)

const geometry = new THREE.BoxGeometry(1, 1, 1)
const material = new THREE.MeshBasicMaterial({ color: 0xe84d3d })
scene.add(new THREE.Mesh(geometry, material))

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
camera.position.set(2.5, 1.8, 3.2)
camera.lookAt(0, 0, 0)

const imageBuffer = render(scene, camera, {
  width: 512,
  height: 512,
})

fs.writeFileSync('render.png', imageBuffer)
```

With local glTF/GLB assets, render the loaded root directly:

```js
import fs from 'node:fs'
import * as THREE from 'three'
import { loadGltfFromFile, render } from '@headless-three/renderer'

const gltf = await loadGltfFromFile('./model.glb')

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
camera.position.set(2, 1.5, 4)
camera.lookAt(0, 0, 0)

const imageBuffer = render(gltf.scene, camera, {
  width: 1024,
  height: 1024,
})

fs.writeFileSync('render.png', imageBuffer)
```

For local Node.js glTF/GLB loading with external buffers or texture files, see
the [Node loader setup guide](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/node-loader-setup.md).
The repository also includes a runnable
[local glTF example](https://github.com/portwatcher/headless-three-renderer/blob/main/examples/render-gltf.mjs).

The module exports a convenience `render(scene, camera, options)` function and a reusable `Renderer` class:

```js
import { Renderer } from '@headless-three/renderer'
const renderer = new Renderer()
const imageBuffer = renderer.render(scene, camera, { width: 512, height: 512 })
```

`Renderer.sortObjects`, `Renderer.setOpaqueSort(fn)`, `Renderer.setTransparentSort(fn)`, and the matching `render()` options (`sortObjects`, `opaqueSort`, `transparentSort`) control native draw-list sorting; invalid option or setter values fail clearly.

It also exports Node loader helpers:

- `applyVrmAnimation(vrm, vrmAnimation, options)`: creates a VRMA animation clip with `createVRMAnimationClip`, seeks a `THREE.AnimationMixer` to `options.time`, and updates the VRM scene for still-frame rendering.
- `loadGltfFromFile(filePath, options)`: loads local `.gltf` or `.glb` files with encoded texture handlers and local `file://` buffer support already installed; malformed helper paths and option containers fail clearly.
- `loadVrmFromFile(filePath, options)`: loads local VRM files with `@pixiv/three-vrm`'s `VRMLoaderPlugin` registered. The Pixiv package remains an optional dependency in your project.
- `loadVrmAnimationFromFile(filePath, options)`: loads local VRMA files with `@pixiv/three-vrm-animation`'s `VRMAnimationLoaderPlugin` registered. The animation package remains optional.
- `createNodeGltfLoader(rootDir, options)`: creates a configured `GLTFLoader` bundle for advanced flows, including plugin registration through `options.configureLoader`; malformed helper boolean options, callback hooks, and custom managers fail clearly.
- `createEncodedImageTextureLoader(rootDir)` / `EncodedImageTextureLoader`: a `LoadingManager` image handler for local PNG/JPEG/WebP files, PNG/JPEG/WebP data URIs, and PNG/JPEG/WebP Blob URLs that exposes encoded buffers directly to renderer-supported texture slots; malformed helper paths fail clearly.
- `installLocalFileFetch()`: a small `file://` fetch bridge for Three.js `FileLoader` when loading local external glTF buffers.
- `resolveLocalAssetPath(url, rootDir)`: shared path resolution for local loader helpers.

## Supported Three.js Surface

See the versioned [compatibility matrix](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/compatibility.md) for the public support contract, known gaps, and platform package status. Scale-test budgets and platform notes are documented in [docs/scale-budgets.md](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/scale-budgets.md).

The public API accepts only Three.js-like objects:

- `scene`: a `THREE.Scene` or `THREE.Object3D` root; malformed scene containers and visibility flags fail clearly.
- `camera`: a `THREE.Camera`, including perspective and orthographic cameras. Malformed camera containers, invalid aspect-derived dimensions, clipping distances, and matrix values fail clearly. `THREE.ArrayCamera` composes sub-camera viewports for PNG, raw RGBA, and target output. `THREE.CubeCamera` renders six RGBA faces plus optional depth faces into `WebGLCubeRenderTarget.texture.image`/`source.data`, nonzero `activeMipmapLevel` writes the active mip entry, and captured color textures can be reused as cube background/environment inputs; exact WebGL face semantics remain limited.
- `options`: an options object; malformed option containers fail clearly.
- `options.width` and `options.height`: output pixel size. Defaults to `512 x 512`; invalid explicit dimensions fail clearly.
- `options.background`: `[r, g, b]`, `[r, g, b, a]`, a `THREE.Color`, a supported 2D/equirectangular/cube texture, or `null` to clear `scene.background` for one render. Defaults to `scene.background`; option-supplied backgrounds use option-supplied background controls rather than scene background controls, and malformed scene or option background values fail clearly.
- `options.backgroundIntensity`: overrides `scene.backgroundIntensity` for supported color and texture backgrounds; invalid values fail clearly.
- `options.backgroundBlurriness`: overrides `scene.backgroundBlurriness` for supported texture backgrounds; invalid values fail clearly.
- `options.backgroundRotation`: overrides `scene.backgroundRotation` for supported equirectangular and cube texture backgrounds; explicit option rotation values are always validated, and invalid or unsupported rotations fail clearly.
- `options.environmentIntensity`: overrides `scene.environmentIntensity` or reflection-probe intensity for supported scene environments; invalid values fail clearly.
- `options.environmentRotation`: overrides `scene.environmentRotation` for supported scene environments; explicit option rotation values are always validated, and invalid values fail clearly.
- `options.viewport`: `[x, y, width, height]` or `{ x, y, width, height }` output pixel rectangle, using a top-left origin, for viewport-limited draws; invalid rectangles fail clearly.
- `options.scissor`: `[x, y, width, height]` or `{ x, y, width, height }` output pixel rectangle, using a top-left origin, for scissor-clipped draws; invalid rectangles fail clearly.
- `options.clippingPlanes`: global world-space clipping planes for the render.
- `options.localClippingEnabled`: `false` disables material-local clipping planes while preserving `options.clippingPlanes`; defaults to `true`, and invalid values fail clearly.
- `options.format`: `'png'` by default, or `'rgba'` for raw RGBA8 bytes; unsupported values fail clearly.
- `options.outputColorSpace`: `THREE.SRGBColorSpace` (`'srgb'`, default) or `THREE.LinearSRGBColorSpace` (`'srgb-linear'`) for material and texture background output conversion; unsupported values fail clearly.
- `options.renderMode`: `'color'` by default, `'mask'` for white visible geometry on black, `'object-id'` for flat RGB object IDs, or `'normal'` for view-space normal colors; invalid values fail clearly.
- `options.target`: a non-array target-like object populated for a single color output, including `target.texture`, `target.textures[0]`, or one-element `target.texture` arrays. Top-level `target.data` remains raw RGBA8; color textures can also request Alpha/Red/RG/RGB/RGBA formats plus normalized `FloatType`, signed/unsigned integer, packed color, or `HalfFloatType` readback arrays.
- `options.postProcessing`: built-in post effects (`exposure`, `contrast`, `saturation`, `vignette`, `grayscale`, `invert`); malformed containers and invalid effect values fail clearly.

### Geometry & Scene

- `THREE.Mesh` and `THREE.SkinnedMesh`
- `THREE.InstancedMesh` with `instanceMatrix` and `instanceColor`; invalid explicit instance counts fail clearly
- `THREE.InstancedBufferGeometry` for mesh, point, line, and dashed-line geometry with common offset/color attributes and `meshPerAttribute` repeat values; invalid explicit instance counts or per-attribute repeat values fail clearly
- `THREE.BatchedMesh` common packed-geometry batches are CPU-expanded with per-instance matrices, colors, visibility flags, packed geometry groups/material arrays when present, common per-object sphere frustum culling, range-local internal sorting, `sortObjects=false`, and `customSort`; malformed batch internals/culling/sort controls fail clearly, while exact culling/source-group preservation edge cases and native batched drawing remain planned
- `THREE.BufferGeometry` positions, indices, normals, and UV coordinates, with invalid attribute values failing clearly
- `THREE.Sprite`/`SpriteMaterial` CPU billboards with center, scale, rotation, perspective size attenuation controls, opacity, texture maps, scene fog, layers, render ordering, main-pass clipping, directional/spot/point shadow casting, and alpha-tested custom depth/distance shadow material cutouts; invalid billboard scalar and size-attenuation values fail clearly
- geometry groups with material arrays
- mesh world transforms, object visibility flags, and object/camera layer masks, with invalid transform matrix, visibility, or layer values failing clearly
- `THREE.LOD` camera-distance/zoom level selection, with invalid auto-update flags, camera zoom, or level distance/hysteresis values failing clearly
- vertex colors, with invalid `material.vertexColors` values failing clearly
- scene background color plus 2D, equirectangular, and raw or encoded six-face cube texture backgrounds with `backgroundIntensity`, approximate texture blur, equirectangular/cube `scene.backgroundRotation`/`options.backgroundRotation`, `options.environmentIntensity`, and equirectangular/cube `scene.environmentRotation`/`options.environmentRotation`; invalid background color/control/rotation values, PMREM/CubeUV backgrounds, and unsupported background rotations fail clearly
- render-option viewport/scissor rectangles and render-target viewport/scissor fields in output pixel coordinates, with malformed target scissor flags failing clearly
- perspective, orthographic, and custom projection matrices

### Materials & Textures

- material base color, opacity, and visibility, with invalid color/opacity/visible values failing clearly
- `material.map` (base color texture) — PNG, JPEG, WebP, and raw one-channel, two-channel, RGB, or RGBA numeric DataTexture inputs, including byte, signed/unsigned normalized integer, packed 16-bit color, float, and half-float typed data, with `texture.channel` UV selection and sRGB color-space decode
- base, sprite, point, line, matcap, emissive, light, sheen color, and physical specular color maps decode `THREE.SRGBColorSpace`; unsupported texture color-space/encoding values fail clearly
- base, 2D background, sprite/point color and alpha, line, matcap, normal/bump, displacement, emissive, metallic/roughness, AO/light, Phong specular, alpha, and current physical-extension maps honor texture UV transforms, including explicit texture matrices for those covered slots and color-space decode after explicit matrices for current color-producing transform slots; invalid transform and transform-boolean values fail clearly
- `texture.channel` supports channels 0-3 on supported map slots; channels 1-3 route selected non-primary UV attributes through the available native UV streams, simple mesh draws with only `material.map` and `material.alphaMap` can use distinct selected non-primary channels for those two maps, and other mixed non-primary channels in one material draw fail clearly
- material and texture background output conversion supports `THREE.SRGBColorSpace` and `THREE.LinearSRGBColorSpace`; texture backgrounds decode `THREE.SRGBColorSpace`
- base/background, normal/bump, displacement, metallic/roughness, emissive, AO/light, alpha, Phong specular, toon gradient, matcap color-map, and packed physical-extension texture-group wrap modes plus `NearestFilter`/`LinearFilter`-family `magFilter` and `minFilter`, including generated mip chains for mipmap min filters, raw explicit mip chains for unpacked 2D material/background texture uploads, half-float raw mip level decoding, and clear failures for unsupported sampler constants, invalid mipmap controls, or invalid anisotropy values
- PBR metallic/roughness via `MeshStandardMaterial` and `MeshPhysicalMaterial`
- `MeshPhysicalMaterial` clearcoat, sheen, anisotropy, scalar iridescence, specular intensity/color, IOR, attenuation, approximate dispersion, and roughness-aware environment-backed or scene-color transmission / refraction; invalid physical color/scalar values fail clearly
- physical material extension maps for clearcoat, clearcoat roughness, clearcoat normals, sheen color/roughness, anisotropy, iridescence factor/thickness, specular color/intensity, transmission, and thickness; all current physical-extension maps include primary/secondary `texture.channel` UV selection, texture transforms including explicit matrices, packed texture-group sampler settings, clear failures for incompatible packed samplers, and sheen/specular color maps include sRGB color-space decode
- custom WGSL fragment bodies via `material.userData.headlessThreeRenderer.fragmentWgsl`; `ShaderMaterial`, `RawShaderMaterial`, NodeMaterial, and `onBeforeCompile` customizations require this explicit override path
- metallic/roughness map (`material.metalnessMap` / `material.roughnessMap`) with primary/secondary `texture.channel` UV selection and wrap/filter sampler settings
- normal map with configurable `normalScale`, plus bump map with `bumpScale`, both with primary/secondary `texture.channel` UV selection and wrap/filter sampler settings; invalid scalar values fail clearly
- `MeshNormalMaterial` and `MeshMatcapMaterial` normal-map output
- `material.flatShading` per-face normals for triangle meshes without normal maps
- `MeshMatcapMaterial.map` color maps with primary/secondary `texture.channel` UV selection and transforms
- displacement map CPU-baked into triangle vertices with `displacementScale`, `displacementBias`, primary/secondary `texture.channel` UV selection, and texture transforms; invalid scale/bias values fail clearly
- `MeshToonMaterial.gradientMap` red-channel diffuse ramps with sRGB color-space decode and wrap/filter sampler settings; direct conformance also covers toon base-map UV channels, emissive-map UV channels, light-map secondary UVs, and alpha-map cutouts
- `MeshDepthMaterial.depthPacking`: basic, RGBA, RGB, and RG packing, with clear failures for unsupported depth-packing constants
- `MeshDistanceMaterial` `referencePosition`, `nearDistance`, and `farDistance` overrides, with invalid range values failing clearly, plus alpha-map cutouts and CPU-baked displacement
- main-pass `material.wireframe` output for supported mesh materials, including direct coverage for `MeshBasicMaterial`, `MeshDepthMaterial`, and `MeshDistanceMaterial`
- `Object3D.customDepthMaterial` and `customDistanceMaterial` for mesh shadow caster alpha-tested and displacement material inputs, plus alpha-tested sprite/point billboard shadow cutouts
- emissive color, intensity, and emissive map, with primary/secondary `texture.channel` UV selection, sRGB color-space decode, and wrap/filter sampler settings; invalid color/intensity values fail clearly
- light maps with `lightMapIntensity`, primary/secondary `texture.channel` UV selection, texture transforms, sRGB color-space decode, and wrap/filter sampler settings; invalid intensity values fail clearly
- occlusion map (`material.aoMap`) applied to indirect lighting, with primary/secondary `texture.channel` UV selection and wrap/filter sampler settings; invalid intensity values fail clearly
- alpha map (`material.alphaMap`) using Three.js' green-channel opacity convention, with primary/secondary `texture.channel` UV selection and wrap/filter sampler settings
- `MeshPhongMaterial.specularMap` red-channel specular strength, with finite `shininess`, primary/secondary `texture.channel` UV selection, texture transforms, wrap/filter sampler settings, and masking for scene-level, reflection-probe, and supported material-level environment specular reflections
- `MeshBasicMaterial.envMap` for one shared material-level reflection or refraction map, including legacy multiply/mix/add combine modes, `reflectivity`, and `refractionRatio`; invalid env-map scalar values fail clearly
- `MeshStandardMaterial`, `MeshPhysicalMaterial` (PBR), `MeshLambertMaterial` (diffuse-only), and `MeshBasicMaterial` (unlit)
- `ShadowMaterial` transparent receiver output with color, opacity, scene fog, fog opt-out, and output color-space conversion
- `material.side`: `FrontSide`, `BackSide`, `DoubleSide`, with clear failures for unsupported side constants
- `material.fog = false` opt-out for scene fog on mesh, shadow, sprite, point, and line material paths; invalid fog color/parameter values fail clearly
- alpha test (`material.alphaTest`) with fragment discard; invalid values fail clearly
- native draw ordering honors group order, `renderOrder`, material id, WebGL material variant, transmissive/transparent buckets, projected geometry bounding-sphere z, object/insertion ties, `sortObjects`, and custom opaque/transparent sort callbacks with object/material/geometry/group render-item metadata, including source-object metadata for BatchedMesh-expanded draws; transparency sorting is back-to-front with `material.depthWrite` overrides, including Three.js' default transparent depth writes; invalid `renderOrder` and sort-control values fail clearly
- material render state: `depthTest`, `depthFunc`, `depthWrite`, `colorWrite`, `polygonOffset`, `alphaHash`, `alphaToCoverage` on 4x MSAA renders, `premultipliedAlpha`, stencil state, built-in blending modes, `CustomBlending` equations/factors, and clear failures for unsupported render-state constants or invalid boolean/numeric values
- render-option global clipping planes and material-local clipping planes, with `options.localClippingEnabled: false` available to ignore material-local planes; invalid plane/control values, invalid clipping control booleans, and over-budget global/group/material combinations beyond eight active planes fail clearly
- single shared material-level reflection/refraction `envMap` inputs are supported for `MeshBasicMaterial`, and shared reflection `envMap` inputs are supported for `MeshStandardMaterial`, `MeshPhysicalMaterial`, `MeshPhongMaterial`, and `MeshLambertMaterial` through the native IBL path; `envMap` properties on material classes that do not consume material environment maps are ignored, while unsupported material classes, unsupported material env-map options, non-Basic refraction mappings, PMREM/CubeUV mappings, multiple distinct material env maps, and multiple distinct material env-map rotations fail clearly
- texture wrap modes: repeat, mirror, clamp-to-edge
- mipmap min filters generate native mip chains from supported material/background texture source levels; raw `DataTexture`-style explicit mipmap arrays upload for unpacked 2D material/background texture slots, while packed physical-extension maps and environment/reflection-probe explicit mip arrays still fail clearly
- texture anisotropy values greater than 1 use native anisotropic samplers for supported material/background texture slots when the effective sampler is linear-filtered; invalid anisotropy values fail clearly
- line material arrays honor geometry groups; `LineBasicMaterial.linewidth` and `LineDashedMaterial.linewidth` values greater than 1 expand to camera-facing quads; dashed line material segments honor dash/gap/scale settings and custom `lineDistance` attributes, and preserve map UV transforms including explicit matrices, selected `texture.channel` UVs, and interpolated vertex colors for common `LineDashedMaterial` cases, including instanced line geometry; invalid line scalar values fail clearly

Texture image data can be:

- Raw one-channel, two-channel, RGB, or RGBA numeric pixels via `THREE.DataTexture` (or any image with `.data`, `.width`, `.height`), including `UnsignedByteType`, normalized `ByteType`/`ShortType`/`UnsignedShortType`/`IntType`/`UnsignedIntType`, packed `UnsignedShort4444Type`/`UnsignedShort5551Type`, normalized float arrays, and `HalfFloatType` `Uint16Array` binary16 data
- Encoded PNG, JPEG, or WebP image buffers (auto-decoded on the native side)

Compressed KTX2/Basis/`THREE.CompressedTexture` inputs are not decoded in-process; pre-decode them to RGB/RGBA data or an encoded PNG/JPEG/WebP image before rendering. Browser `Image`/`ImageBitmap`/canvas-like texture objects are not readable in Node and fail clearly until normalized to encoded bytes or raw pixel data. Mismatched-length raw texture payloads fail clearly.

### Lights

- `THREE.AmbientLight` — uniform ambient illumination
- `THREE.DirectionalLight` — sun-like parallel light with position/target
- `THREE.PointLight` — omnidirectional light with distance/decay attenuation
- `THREE.SpotLight` — cone light with angle, penumbra, distance, and decay
- `THREE.HemisphereLight` — sky/ground gradient ambient light
- `THREE.RectAreaLight` — one-sided finite-area direct-light approximation
- `THREE.LightProbe` — diffuse spherical-harmonics indirect lighting, with invalid coefficient values failing clearly

Lights are automatically extracted from the scene, with invalid light color, numeric controls, transform matrix values, shadow flags, and shadow option containers failing clearly. The shader uses a Cook-Torrance PBR BRDF (GGX/Trowbridge-Reitz distribution, Schlick-GGX geometry, Schlick Fresnel) with Three.js-compatible physically-based attenuation for punctual lights. Up to 64 direct lights per scene are supported. One visible directional, spot, or point light may cast shadows; additional shadow-casting lights fail clearly until native multi-shadow-map rendering lands. When no lights are present, meshes render with a hemispherical ambient fallback.

### Image-Based Lighting (IBL)

Environment maps set on `scene.environment` are supported for image-based lighting. A single shared material-level reflection `envMap` can also feed the same native IBL path for `MeshBasicMaterial`, `MeshStandardMaterial`, `MeshPhysicalMaterial`, `MeshPhongMaterial`, and `MeshLambertMaterial`, with per-material intensity and one shared material env-map rotation. Material `envMap` properties on classes that Three.js does not shade with material environment maps are accepted as no-ops. The renderer CPU-precomputes:

- **Diffuse irradiance cubemap** — cosine-weighted hemisphere convolution
- **Prefiltered specular cubemap** — GGX importance-sampled at multiple roughness mip levels
- **BRDF integration LUT** — split-sum approximation lookup table

Supported input formats: equirectangular images in RGB/RGBA byte data, Float16 (`HalfFloatType`), or Float32 (`FloatType`), plus raw or encoded six-face cube reflection textures. Scene-environment, reflection-probe, and supported material-level LDR inputs honor explicit `THREE.SRGBColorSpace` and `THREE.LinearSRGBColorSpace`; omitted color space defaults to sRGB for compatibility. `MeshBasicMaterial.envMap` supports legacy multiply/mix/add combine modes with `reflectivity` plus refraction mappings with `refractionRatio`; Phong/Lambert material env maps honor legacy multiply/mix/add combine modes plus `reflectivity`; malformed scene-environment/reflection-probe values, non-Basic refraction mappings, PMREM/CubeUV environment inputs, multiple distinct material env maps, multiple distinct material env-map rotations, invalid environment intensity values, invalid material env-map scalar values, and unsupported raw environment texture channel layouts fail clearly until those paths land. `scene.environmentIntensity` is respected for scene environments, reflection-probe intensity applies only when `scene.environment` is absent, and `options.environmentIntensity` can override scene or reflection-probe intensity for one render.

Scene-level reflection probes are supported through `scene.userData.headlessThreeRenderer.reflectionProbe` or the first entry in `reflectionProbes`. Probe textures use the same equirectangular and cube texture formats as `scene.environment` and feed the same diffuse/specular IBL path.

### Skinning / Skeletal Animation

`THREE.SkinnedMesh` objects are automatically detected and skinned on the CPU. The renderer reads `skinIndex` and `skinWeight` attributes, computes bone matrices from `skeleton.bones` and `skeleton.boneInverses`, and transforms vertex positions and normals before sending them to the GPU. Invalid bone, inverse bind, and mesh bind matrix values fail clearly.

Compatible with:

- **Three.js** `SkinnedMesh` + `Skeleton` + `AnimationMixer`
- **@pixiv/three-vrm** — VRM humanoid avatars
- **VRMA** — VRM Animation files via `VRMAnimationLoaderPlugin` + `createVRMAnimationClip`

The repository includes runnable local examples for [glTF/GLB](https://github.com/portwatcher/headless-three-renderer/blob/main/examples/render-gltf.mjs) and [VRM/VRMA](https://github.com/portwatcher/headless-three-renderer/blob/main/examples/render-vrm.mjs) assets.

Use `applyVrmAnimation()` or your own `AnimationMixer`, then call `scene.updateMatrixWorld(true)` before `render()` to bake the current pose:

```js
import * as THREE from 'three'
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm'
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from '@pixiv/three-vrm-animation'
import { applyVrmAnimation, loadVrmAnimationFromFile, loadVrmFromFile, render } from '@headless-three/renderer'

// Load VRM model
const modelGltf = await loadVrmFromFile('./avatar.vrm', { VRMLoaderPlugin })
const vrm = modelGltf.userData.vrm
VRMUtils.removeUnnecessaryVertices(vrm.scene)
VRMUtils.removeUnnecessaryJoints(vrm.scene)
vrm.scene.rotation.y = Math.PI

// Load VRMA animation
const animGltf = await loadVrmAnimationFromFile('./dance.vrma', {
  VRMLoaderPlugin,
  VRMAnimationLoaderPlugin,
})
const vrmAnimation = animGltf.userData.vrmAnimations[0]

// Animate to a specific time
await applyVrmAnimation(vrm, vrmAnimation, {
  createVRMAnimationClip,
  time: 1.5,
})

// Update world matrices then render
vrm.scene.updateMatrixWorld(true)

const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20)
camera.position.set(0, 1.2, 3)
camera.lookAt(0, 1, 0)

const imageBuffer = render(vrm.scene, camera, {
  width: 1024,
  height: 1024,
})
```

### Morph Targets / Blend Shapes

Morph targets are applied on the CPU before rendering. Both **relative** (glTF default) and **absolute** (legacy Three.js) modes are supported. Position and normal morphs are applied based on `mesh.morphTargetInfluences`, with invalid influence values and malformed `geometry.morphTargetsRelative` values failing clearly. This is compatible with:

- glTF morph targets via `GLTFLoader`
- VRM blend shapes / expressions from `@pixiv/three-vrm`
- Blender shape keys exported to glTF

### Shadows

Directional, spot, point, and up to four directional cascaded shadow maps are supported for one visible shadow-casting light. Set `light.castShadow = true`, configure `light.shadow.camera`, and mark meshes with `mesh.castShadow = true` / `mesh.receiveShadow = true`. Common shadow options including `light.shadow.bias`, `light.shadow.normalBias`, `light.shadow.radius`, and PCF-irrelevant `light.shadow.blurSamples` inputs are accepted, explicit mesh `material.shadowSide` values filter shadow-caster faces, and `material.alphaToCoverage` approximates shadow-caster alpha cutouts with a 0.5 cutoff. `Object3D.customDepthMaterial` is honored for directional/spot mesh shadow caster alpha-tested and displacement inputs, and `customDistanceMaterial` is honored for point-light mesh shadow caster alpha-tested and displacement inputs; sprite and point billboard shadow casters also honor alpha-tested custom depth/distance material cutouts. `THREE.Sprite` and `THREE.Points` can cast directional/spot/point shadows from expanded billboard quads, and `receiveShadow` is accepted as a WebGL-compatible no-op on their unlit material paths. Malformed shadow flags/containers, invalid shadow numeric values, additional shadow-casting lights, malformed or over-budget cascade hints, and non-square point-light `light.shadow.mapSize` values fail clearly until native multi-shadow, deeper cascade support, and rectangular cube-face support land. The renderer renders a depth-only pass and samples it with 3×3 PCF and a normal-offset bias.

Directional cascades can be provided with `light.userData.headlessThreeRenderer.shadowCascades`, where each cascade has finite `{ left, right, top, bottom, near, far, split }` bounds.

### Tone Mapping

Output uses the Narkowicz ACES Filmic tone mapping fit with a three.js-compatible `1/0.6` exposure pre-scale, matching `THREE.ACESFilmicToneMapping`.

### Render Targets & Post-Processing

`renderToTarget(scene, camera, target, options)` and `options.target` populate a target-like object with `{ width, height, data }` plus `target.texture.image.data` when a texture object is present. Top-level target rendering defaults to raw RGBA8 and `target.data` remains RGBA8 for compatibility; color textures receive Alpha/Red/RG/RGB/RGBA byte data by requested format, normalized `Float32Array` data for `THREE.FloatType`, signed or unsigned normalized integer arrays for `ByteType`/`ShortType`/`IntType` and `UnsignedShortType`/`UnsignedIntType`, packed `Uint16Array` data for `UnsignedShort4444Type`/`UnsignedShort5551Type`, packed `Uint32Array` RGB9_E5 data for `UnsignedInt5999Type`, or `Uint16Array` half-float data for `HalfFloatType`. A target `depthTexture` object receives normalized depth readback for the same viewport/scissor and visible depth-tested geometry, including base-texture and alpha-map alpha-tested cutouts plus transparent material default/explicit `depthWrite` behavior; `THREE.FloatType` depth textures receive scalar `Float32Array` data, `HalfFloatType` depth textures receive `Uint16Array` half-float data, `UnsignedByteType`/`UnsignedShortType`/`UnsignedIntType` depth textures receive scalar unsigned typed arrays, `UnsignedInt248Type` receives `Uint32Array` data with normalized depth24 in the high bits and zero stencil bytes, and plain depth target objects receive RGBA8 bytes. Single-attachment target texture arrays and MRT-shaped targets write their first texture. Multiple color attachments and 4x MSAA sample counts resolve into target readback buffers; malformed targets, target image containers, nested target texture/mipmap/source containers, malformed target scissor flags, unsupported sample counts, unsupported color target texture formats/types, explicit depth texture types, and depth texture format/type pairings fail clearly.

Built-in post-processing can be enabled with `options.postProcessing`. Supported effects are exposure, contrast, saturation, vignette, grayscale, and invert; malformed containers and invalid effect values fail clearly.

`options.renderMode` can request flat auxiliary passes. `'mask'` clears to black and writes white for visible geometry. `'object-id'` clears to RGB zero and encodes each object's adapter sort ID plus one into RGB bytes, making `format: 'rgba'` the preferred inspection path. `'normal'` clears to black and writes view-space normal colors matching `MeshNormalMaterial` for visible geometry. Target-based object-id renders populate `target.objectIdEntries` and `target.objectIdMap` for reverse lookup from encoded RGB IDs. These modes bypass scene backgrounds, lighting, environment, fog, and post-processing while preserving depth testing, culling, clipping planes, base texture alpha, `material.alphaMap`, `alphaTest`, and `alphaHash`; invalid render modes fail clearly.

### Custom WGSL Fragment Materials

Materials can provide a WGSL fragment body with `material.userData.headlessThreeRenderer.fragmentWgsl`. The body runs inside the renderer's standard vertex, uniform, color, UV, and base-texture setup and returns a `vec4<f32>`.

Three.js `ShaderMaterial`, `RawShaderMaterial`, and NodeMaterial are not translated directly; provide the headless WGSL fragment override above or use a built-in material.

### Lines and Points

`THREE.Line`, `LineSegments`, `LineLoop`, and `THREE.Points` are supported. Lines and points render as unlit (basic) primitives and ignore lighting / normals. Opacity, line material arrays with geometry groups, scene fog, and `material.fog = false` are honored. `LineBasicMaterial.map` samples line UVs, including texture UV transforms, channel 0-3 `texture.channel` UV selection through one non-primary UV stream, texture RGB with sRGB color-space decode, and alpha-tested texture alpha; `LineBasicMaterial.linewidth` and `LineDashedMaterial.linewidth` values greater than 1 expand to camera-facing quads. Dashed lines honor custom `lineDistance` attributes, and dashed line maps preserve texture UV transforms, selected `texture.channel` UVs, and sRGB color-space decode while reconstructing dash segments. `PointsMaterial` maps and alpha maps use per-point geometry UVs when present and point-sprite UVs otherwise, honor texture UV transforms, decode sRGB color maps, shrink with perspective distance by default, keep orthographic point size independent of camera depth, cast directional/spot/point shadows from the expanded billboard quads, accept `receiveShadow` as a WebGL-compatible no-op, and honor alpha-tested custom depth/distance material cutouts in shadow passes. Invalid point size, point size attenuation, line width, and dashed-line scalar values fail clearly.
