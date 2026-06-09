# headless-three

Headless Three.js ecosystem — pnpm monorepo.

## Packages

| Package | Description |
|---|---|
| [`@headless-three/renderer`](./packages/renderer) | Headless wgpu renderer for Three.js scenes in Node.js |

## Development

```bash
pnpm install
pnpm -r build
pnpm -r test
```

Releases are tag-driven: push `v<semver>` to trigger the publish workflow.
# headless-three-renderer

Headless `wgpu` renderer for Three.js scenes in Node.js.

This package exists for Node.js environments where WebGL is not available. You build or load a normal Three.js scene, pass the `THREE.Scene` and `THREE.Camera` to this package, and the native addon renders it with `wgpu`.

```bash
npm install headless-three-renderer three
```

```js
import fs from 'node:fs'
import * as THREE from 'three'
import { render } from 'headless-three-renderer'

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
import { render } from 'headless-three-renderer'

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

The module exports a convenience `render(scene, camera, options)` function and a reusable `Renderer` class:

```js
import { Renderer } from 'headless-three-renderer'
const renderer = new Renderer()
const imageBuffer = renderer.render(scene, camera, { width: 512, height: 512 })
```

## Supported Three.js Surface

The public API accepts only Three.js-like objects:

- `scene`: a `THREE.Scene`.
- `camera`: a `THREE.Camera`, including perspective and orthographic cameras.
- `options.width` and `options.height`: output pixel size. Defaults to `512 x 512`.
- `options.background`: `[r, g, b]`, `[r, g, b, a]`, or a `THREE.Color`. Defaults to `scene.background` when it is a color.
- `options.format`: `'png'` by default, or `'rgba'` for raw RGBA8 bytes.
- `options.target`: a target-like object populated with raw RGBA8 readback data.
- `options.postProcessing`: built-in post effects (`exposure`, `contrast`, `saturation`, `vignette`, `grayscale`, `invert`).

### Geometry & Scene

- `THREE.Mesh` and `THREE.SkinnedMesh`
- `THREE.BufferGeometry` positions, indices, normals, and UV coordinates
- geometry groups with material arrays
- mesh world transforms
- vertex colors
- scene background color
- perspective, orthographic, and custom projection matrices

### Materials & Textures

- material base color and opacity
- `material.map` (base color texture) — PNG, JPEG, WebP, and raw RGBA8 DataTexture
- PBR metallic/roughness via `MeshStandardMaterial` and `MeshPhysicalMaterial`
- `MeshPhysicalMaterial` clearcoat, sheen, anisotropy, and environment-backed or scene-color transmission / refraction
- physical material extension maps for clearcoat, clearcoat roughness, clearcoat normals, sheen color/roughness, anisotropy, transmission, and thickness
- custom WGSL fragment bodies via `material.userData.headlessThreeRenderer.fragmentWgsl`
- metallic/roughness map (`material.metalnessMap` / `material.roughnessMap`)
- normal map with configurable `normalScale`
- emissive color, intensity, and emissive map
- occlusion map (`material.aoMap`) applied to indirect lighting
- `MeshStandardMaterial`, `MeshPhysicalMaterial` (PBR), `MeshLambertMaterial` (diffuse-only), and `MeshBasicMaterial` (unlit)
- `material.side`: `FrontSide`, `BackSide`, `DoubleSide`
- alpha test (`material.alphaTest`) with fragment discard
- transparency sorting (back-to-front) with separate no-depth-write pipeline
- texture wrap modes: repeat, mirror, clamp-to-edge

Texture image data can be:

- Raw RGBA8 pixels via `THREE.DataTexture` (or any image with `.data`, `.width`, `.height`)
- Encoded PNG, JPEG, or WebP image buffers (auto-decoded on the native side)

### Lights

- `THREE.AmbientLight` — uniform ambient illumination
- `THREE.DirectionalLight` — sun-like parallel light with position/target
- `THREE.PointLight` — omnidirectional light with distance/decay attenuation
- `THREE.SpotLight` — cone light with angle, penumbra, distance, and decay
- `THREE.HemisphereLight` — sky/ground gradient ambient light

Lights are automatically extracted from the scene. The shader uses a Cook-Torrance PBR BRDF (GGX/Trowbridge-Reitz distribution, Schlick-GGX geometry, Schlick Fresnel) with Three.js-compatible physically-based attenuation. Up to 16 lights per scene. When no lights are present, meshes render with a hemispherical ambient fallback.

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
import { render } from 'headless-three-renderer'

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

`renderToTarget(scene, camera, target, options)` and `options.target` populate a target-like object with `{ width, height, data }` plus `target.texture.image.data` when a texture object is present. Target rendering defaults to raw RGBA8.

Built-in post-processing can be enabled with `options.postProcessing`. Supported effects are exposure, contrast, saturation, vignette, grayscale, and invert.

### Custom WGSL Fragment Materials

Materials can provide a WGSL fragment body with `material.userData.headlessThreeRenderer.fragmentWgsl`. The body runs inside the renderer's standard vertex, uniform, color, UV, and base-texture setup and returns a `vec4<f32>`.

### Lines and Points

`THREE.Line`, `THREE.LineSegments`, `THREE.LineLoop`, and `THREE.Points` are supported. Lines and points render as unlit (basic) primitives and ignore lighting / normals.
