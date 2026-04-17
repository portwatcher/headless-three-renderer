# headless-three-renderer

Headless `wgpu` renderer for Three.js scenes in Node.js.

This package exists for Node.js environments where WebGL is not available. You build or load a normal Three.js scene, pass the `THREE.Scene` and `THREE.Camera` to this package, and the native addon renders it with `wgpu`.

```bash
npm install
npm run build
```

```js
const fs = require('node:fs')
const THREE = require('three')
const renderer = require('./')

const scene = new THREE.Scene()
scene.background = new THREE.Color(0.04, 0.045, 0.05)

const geometry = new THREE.BoxGeometry(1, 1, 1)
const material = new THREE.MeshBasicMaterial({ color: 0xe84d3d })
scene.add(new THREE.Mesh(geometry, material))

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
camera.position.set(2.5, 1.8, 3.2)
camera.lookAt(0, 0, 0)

const imageBuffer = renderer.render(scene, camera, {
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
import renderer from './index.js'

const gltf = await new GLTFLoader().loadAsync('./model.glb')

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
camera.position.set(2, 1.5, 4)
camera.lookAt(0, 0, 0)

const imageBuffer = renderer.render(gltf.scene, camera, {
  width: 1024,
  height: 1024,
})

fs.writeFileSync('render.png', imageBuffer)
```

The module exports a convenience `render(scene, camera, options)` function and a reusable `Renderer` class:

```js
const { Renderer } = require('./')
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
- metallic/roughness map (`material.metalnessMap` / `material.roughnessMap`)
- normal map with configurable `normalScale`
- emissive color, intensity, and emissive map
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
import renderer from './index.js'

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

const imageBuffer = renderer.render(vrm.scene, camera, {
  width: 1024,
  height: 1024,
})
```

### Morph Targets / Blend Shapes

Morph targets are applied on the CPU before rendering. Both **relative** (glTF default) and **absolute** (legacy Three.js) modes are supported. Position and normal morphs are applied based on `mesh.morphTargetInfluences`. This is compatible with:

- glTF morph targets via `GLTFLoader`
- VRM blend shapes / expressions from `@pixiv/three-vrm`
- Blender shape keys exported to glTF

### Not Yet Implemented

Shadows, custom shaders, render targets, occlusion maps, lines, points, `MeshBasicMaterial`/`MeshLambertMaterial` unlit paths, back-face culling per `material.side`, and ACES Filmic tone mapping (currently uses Reinhard).
