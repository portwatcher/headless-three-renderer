# @headless-three/renderer

Headless `wgpu` renderer for Three.js scenes in Node.js.

This package exists for Node.js environments where WebGL is not available. You build or load a normal Three.js scene, pass the `THREE.Scene` and `THREE.Camera` to this package, and the native addon renders it with `wgpu`.

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

With `GLTFLoader`, render the loaded Three.js scene directly:

```js
import fs from 'node:fs'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { render } from '@headless-three/renderer'

const gltf = await new GLTFLoader().loadAsync('./model.glb')

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

It also exports Node loader helpers:

- `createEncodedImageTextureLoader(rootDir)` / `EncodedImageTextureLoader`: a `LoadingManager` image handler for local PNG/JPEG/WebP files that exposes encoded buffers directly to renderer-supported texture slots.
- `installLocalFileFetch()`: a small `file://` fetch bridge for Three.js `FileLoader` when loading local external glTF buffers.
- `resolveLocalAssetPath(url, rootDir)`: shared path resolution for local loader helpers.

## Supported Three.js Surface

See the versioned [compatibility matrix](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/compatibility.md) for the public support contract, known gaps, and platform package status.

The public API accepts only Three.js-like objects:

- `scene`: a `THREE.Scene`.
- `camera`: a `THREE.Camera`, including perspective and orthographic cameras. `THREE.ArrayCamera` and `THREE.CubeCamera` fail clearly until native support lands.
- `options.width` and `options.height`: output pixel size. Defaults to `512 x 512`.
- `options.background`: `[r, g, b]`, `[r, g, b, a]`, a `THREE.Color`, or a supported 2D texture. Defaults to `scene.background`.
- `options.backgroundIntensity`: overrides `scene.backgroundIntensity` for supported color and 2D texture backgrounds.
- `options.backgroundBlurriness`: overrides `scene.backgroundBlurriness` for supported 2D texture backgrounds.
- `options.viewport`: `[x, y, width, height]` or `{ x, y, width, height }` output pixel rectangle, using a top-left origin, for viewport-limited draws.
- `options.scissor`: `[x, y, width, height]` or `{ x, y, width, height }` output pixel rectangle, using a top-left origin, for scissor-clipped draws.
- `options.format`: `'png'` by default, or `'rgba'` for raw RGBA8 bytes.
- `options.outputColorSpace`: `THREE.SRGBColorSpace` (`'srgb'`, default) or `THREE.LinearSRGBColorSpace` (`'srgb-linear'`) for material and 2D texture background output conversion.
- `options.target`: a target-like object populated with raw RGBA8 readback data for a single color output.
- `options.postProcessing`: built-in post effects (`exposure`, `contrast`, `saturation`, `vignette`, `grayscale`, `invert`).

### Geometry & Scene

- `THREE.Mesh` and `THREE.SkinnedMesh`
- `THREE.InstancedMesh` with `instanceMatrix` and `instanceColor`
- `THREE.BufferGeometry` positions, indices, normals, and UV coordinates
- geometry groups with material arrays
- mesh world transforms
- `THREE.LOD` camera-distance level selection
- vertex colors
- scene background color and 2D texture backgrounds with `backgroundIntensity` and 2D texture blur; cube/equirect backgrounds fail clearly until native support lands
- render-option viewport and scissor rectangles in output pixel coordinates
- perspective, orthographic, and custom projection matrices

### Materials & Textures

- material base color and opacity
- `material.map` (base color texture) — PNG, JPEG, WebP, and raw RGBA8 DataTexture, with `texture.channel` UV selection and sRGB color-space decode
- base, matcap, emissive, light, sheen color, and physical specular color maps decode `THREE.SRGBColorSpace`
- material and 2D texture background output conversion supports `THREE.SRGBColorSpace` and `THREE.LinearSRGBColorSpace`
- base/background, normal/bump, metallic/roughness, emissive, AO/light, alpha, Phong specular, and packed physical-extension texture-group wrap modes plus `NearestFilter`/`LinearFilter`-family `magFilter` and `minFilter`
- PBR metallic/roughness via `MeshStandardMaterial` and `MeshPhysicalMaterial`
- `MeshPhysicalMaterial` clearcoat, sheen, anisotropy, specular intensity/color, and environment-backed or scene-color transmission / refraction
- physical material extension maps for clearcoat, clearcoat roughness, clearcoat normals, sheen color/roughness, anisotropy, specular color/intensity, transmission, and thickness; all current physical-extension maps include `texture.channel` UV selection, packed texture-group sampler settings, and sheen/specular color maps include sRGB color-space decode
- custom WGSL fragment bodies via `material.userData.headlessThreeRenderer.fragmentWgsl`; `ShaderMaterial`, `RawShaderMaterial`, NodeMaterial, and `onBeforeCompile` customizations require this explicit override path
- metallic/roughness map (`material.metalnessMap` / `material.roughnessMap`) with `texture.channel` UV selection and wrap/filter sampler settings
- normal map with configurable `normalScale`, plus bump map with `bumpScale`, both with `texture.channel` UV selection and wrap/filter sampler settings
- `MeshNormalMaterial` and `MeshMatcapMaterial` normal-map output
- `material.flatShading` per-face normals for triangle meshes without normal maps
- `MeshMatcapMaterial.map` color maps with `texture.channel` UV selection and transforms
- displacement map CPU-baked into triangle vertices with `displacementScale`, `displacementBias`, `texture.channel` UV selection, and texture transforms
- `MeshToonMaterial.gradientMap` red-channel diffuse ramps
- `MeshDepthMaterial.depthPacking`: basic, RGBA, RGB, and RG packing
- `MeshDistanceMaterial` `referencePosition`, `nearDistance`, and `farDistance` overrides
- `MeshDepthMaterial` and `MeshDistanceMaterial` wireframe output
- emissive color, intensity, and emissive map, with `texture.channel` UV selection, sRGB color-space decode, and wrap/filter sampler settings
- light maps with `lightMapIntensity`, `texture.channel` UV selection, texture transforms, sRGB color-space decode, and wrap/filter sampler settings
- occlusion map (`material.aoMap`) applied to indirect lighting, with `texture.channel` UV selection and wrap/filter sampler settings
- alpha map (`material.alphaMap`) using Three.js' green-channel opacity convention, with `texture.channel` UV selection and wrap/filter sampler settings
- `MeshPhongMaterial.specularMap` red-channel specular strength, with `texture.channel` UV selection, texture transforms, and wrap/filter sampler settings
- `MeshStandardMaterial`, `MeshPhysicalMaterial` (PBR), `MeshLambertMaterial` (diffuse-only), and `MeshBasicMaterial` (unlit)
- `material.side`: `FrontSide`, `BackSide`, `DoubleSide`
- `material.fog = false` opt-out for scene fog
- alpha test (`material.alphaTest`) with fragment discard
- transparency sorting (back-to-front) with separate no-depth-write pipeline
- material render state: `depthTest`, `depthWrite`, `colorWrite`, `polygonOffset`, `alphaHash`, `premultipliedAlpha`, stencil state, built-in blending modes, and `CustomBlending` equations/factors
- material-level `envMap` reflection/refraction inputs fail clearly; use `scene.environment` or reflection probes for supported IBL
- unsupported `alphaToCoverage` and `clipShadows` material states fail clearly
- unsupported `MeshPhysicalMaterial` iridescence and dispersion inputs fail clearly
- texture wrap modes: repeat, mirror, clamp-to-edge
- dashed line material segments preserve map UVs and interpolated vertex colors for common `LineDashedMaterial` cases
- line material `linewidth` values other than 1 fail clearly until thick-line support lands

Texture image data can be:

- Raw RGBA8 pixels via `THREE.DataTexture` (or any image with `.data`, `.width`, `.height`)
- Encoded PNG, JPEG, or WebP image buffers (auto-decoded on the native side)

Compressed KTX2/Basis/`THREE.CompressedTexture` inputs are not decoded in-process; pre-decode them to RGBA data or an encoded PNG/JPEG/WebP image before rendering.

### Lights

- `THREE.AmbientLight` — uniform ambient illumination
- `THREE.DirectionalLight` — sun-like parallel light with position/target
- `THREE.PointLight` — omnidirectional light with distance/decay attenuation
- `THREE.SpotLight` — cone light with angle, penumbra, distance, and decay
- `THREE.HemisphereLight` — sky/ground gradient ambient light
- `THREE.RectAreaLight` — one-sided finite-area direct-light approximation

Lights are automatically extracted from the scene. The shader uses a Cook-Torrance PBR BRDF (GGX/Trowbridge-Reitz distribution, Schlick-GGX geometry, Schlick Fresnel) with Three.js-compatible physically-based attenuation for punctual lights. Up to 16 lights per scene. When no lights are present, meshes render with a hemispherical ambient fallback.

### Image-Based Lighting (IBL)

Environment maps set on `scene.environment` are supported for image-based lighting. The renderer CPU-precomputes:

- **Diffuse irradiance cubemap** — cosine-weighted hemisphere convolution
- **Prefiltered specular cubemap** — GGX importance-sampled at multiple roughness mip levels
- **BRDF integration LUT** — split-sum approximation lookup table

Supported input formats: equirectangular images in RGBA8, Float16 (`HalfFloatType`), or Float32 (`FloatType`). `scene.environmentIntensity` is respected.

Scene-level reflection probes are supported through `scene.userData.headlessThreeRenderer.reflectionProbe` or the first entry in `reflectionProbes`. Probe textures use the same equirectangular texture formats as `scene.environment` and feed the same diffuse/specular IBL path.

### Skinning / Skeletal Animation

`THREE.SkinnedMesh` objects are automatically detected and skinned on the CPU. The renderer reads `skinIndex` and `skinWeight` attributes, computes bone matrices from `skeleton.bones` and `skeleton.boneInverses`, and transforms vertex positions and normals before sending them to the GPU.

Compatible with:

- **Three.js** `SkinnedMesh` + `Skeleton` + `AnimationMixer`
- **@pixiv/three-vrm** — VRM humanoid avatars
- **VRMA** — VRM Animation files via `VRMAnimationLoaderPlugin` + `createVRMAnimationClip`

Call `mixer.update(dt)` and `scene.updateMatrixWorld(true)` before `render()` to bake the current pose:

```js
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm'
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from '@pixiv/three-vrm-animation'
import { render } from '@headless-three/renderer'

const gltfLoader = new GLTFLoader()
gltfLoader.register((parser) => new VRMLoaderPlugin(parser))
gltfLoader.register((parser) => new VRMAnimationLoaderPlugin(parser))

// Load VRM model
const modelGltf = await gltfLoader.loadAsync('./avatar.vrm')
const vrm = modelGltf.userData.vrm
VRMUtils.removeUnnecessaryVertices(vrm.scene)
VRMUtils.removeUnnecessaryJoints(vrm.scene)
vrm.scene.rotation.y = Math.PI

// Load VRMA animation
const animGltf = await gltfLoader.loadAsync('./dance.vrma')
const vrmAnimation = animGltf.userData.vrmAnimations[0]
const clip = createVRMAnimationClip(vrmAnimation, vrm)

// Animate to a specific time
const mixer = new THREE.AnimationMixer(vrm.scene)
mixer.clipAction(clip).play()
mixer.update(1.5) // seek to 1.5 seconds

// Update world matrices then render
vrm.update(0)
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

Morph targets are applied on the CPU before rendering. Both **relative** (glTF default) and **absolute** (legacy Three.js) modes are supported. Position and normal morphs are applied based on `mesh.morphTargetInfluences`. This is compatible with:

- glTF morph targets via `GLTFLoader`
- VRM blend shapes / expressions from `@pixiv/three-vrm`
- Blender shape keys exported to glTF

### Shadows

Directional, spot, point, and directional cascaded shadow maps are supported. Set `light.castShadow = true`, configure `light.shadow.camera`, and mark meshes with `mesh.castShadow = true` / `mesh.receiveShadow = true`. The renderer picks the first shadow-casting directional, spot, or point light, renders a depth-only pass, and samples it with 3×3 PCF and a normal-offset bias.

Directional cascades can be provided with `light.userData.headlessThreeRenderer.shadowCascades`, where each cascade has `{ left, right, top, bottom, near, far, split }` bounds.

### Tone Mapping

Output uses the Narkowicz ACES Filmic tone mapping fit with a three.js-compatible `1/0.6` exposure pre-scale, matching `THREE.ACESFilmicToneMapping`.

### Render Targets & Post-Processing

`renderToTarget(scene, camera, target, options)` and `options.target` populate a target-like object with `{ width, height, data }` plus `target.texture.image.data` when a texture object is present. Target rendering defaults to raw RGBA8. Target `depthTexture`, multiple color attachments, and sample counts greater than 1 fail clearly until depth readback, MRT, and MSAA support land.

Built-in post-processing can be enabled with `options.postProcessing`. Supported effects are exposure, contrast, saturation, vignette, grayscale, and invert.

### Custom WGSL Fragment Materials

Materials can provide a WGSL fragment body with `material.userData.headlessThreeRenderer.fragmentWgsl`. The body runs inside the renderer's standard vertex, uniform, color, UV, and base-texture setup and returns a `vec4<f32>`.

Three.js `ShaderMaterial`, `RawShaderMaterial`, and NodeMaterial are not translated directly; provide the headless WGSL fragment override above or use a built-in material.

### Lines and Points

`THREE.Line`, `THREE.LineSegments`, `THREE.LineLoop`, and `THREE.Points` are supported. Lines and points render as unlit (basic) primitives and ignore lighting / normals. Non-dashed `LineBasicMaterial.map` samples line UVs, including alpha-tested texture alpha.
