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

For local Node.js glTF/GLB loading with external buffers or texture files, see the [Node loader setup guide](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/node-loader-setup.md). The repository also includes runnable [glTF](https://github.com/portwatcher/headless-three-renderer/blob/main/examples/render-gltf.mjs) and [VRM/VRMA](https://github.com/portwatcher/headless-three-renderer/blob/main/examples/render-vrm.mjs) examples.

## API

The module exports a convenience `render(scene, camera, options)` function and a reusable `Renderer` class:

```js
import { Renderer } from '@headless-three/renderer'

const renderer = new Renderer()
const imageBuffer = renderer.render(scene, camera, { width: 512, height: 512 })
```

`Renderer.renderAsync(scene, camera, options)` is a Promise-returning compatibility wrapper around the same scene-output contract.

`Renderer.renderGpuFrame(scene, camera, options)` returns a same-device GPU texture lease without RGBA readback when `Renderer.getGpuOutputCapabilities().texture.supported` is true. Native handles are borrowed `bigint` values and must be released deterministically. See the [GPU-native output contract](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/gpu-native-output.md).

`Renderer.sortObjects`, `Renderer.opaque`, `Renderer.transparent`, `Renderer.setOpaqueSort(fn)`, `Renderer.setTransparentSort(fn)`, and the matching `render()` options (`sortObjects`, `opaque`, `transparent`, `opaqueSort`, `transparentSort`) control native draw-list sorting and bucket inclusion.

It also exports Node loader helpers:

- `loadGltfFromFile(filePath, options)`: loads local `.gltf` or `.glb` files from relative paths, absolute paths, or `file://` URLs with encoded texture handlers and local `file://` buffer support already installed.
- `loadVrmFromFile(filePath, options)`: loads local VRM files with `@pixiv/three-vrm`'s `VRMLoaderPlugin` registered. The Pixiv package remains an optional dependency in your project.
- `loadVrmAnimationFromFile(filePath, options)`: loads local VRMA files with `@pixiv/three-vrm-animation`'s `VRMAnimationLoaderPlugin` registered. The animation package remains optional.
- `applyVrmAnimation(vrm, vrmAnimation, options)`: creates and applies a VRMA animation clip for still-frame rendering.
- `createNodeGltfLoader(rootDir, options)`: creates a configured `GLTFLoader` bundle for advanced flows, including plugin registration through `options.configureLoader`.
- `createEncodedImageTextureLoader(rootDir, manager)` / `EncodedImageTextureLoader`: a `LoadingManager` image handler for local PNG/JPEG/WebP files, PNG/JPEG/WebP data URIs, and PNG/JPEG/WebP Blob URLs.
- `installLocalFileFetch()`: a small `file://` fetch bridge for Three.js `FileLoader` when loading local external glTF buffers.
- `resolveLocalAssetPath(url, rootDir)`: shared path resolution for local loader helpers.

## Supported Three.js Surface

See the versioned [compatibility matrix](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/compatibility.md) for the public support contract, known gaps, and platform package status. Scale-test budgets and platform notes are documented in [docs/scale-budgets.md](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/scale-budgets.md).

The current release supports:

- Scene roots as `THREE.Scene` or `THREE.Object3D`, plus perspective, orthographic, array, and cube cameras.
- PNG output by default, raw RGBA output with `format: 'rgba'`, viewport and scissor rectangles, render targets, cube targets, readback helpers, selected color attachments, and depth texture readback.
- Scene backgrounds from colors, CSS color strings, 2D textures, equirectangular textures, cube textures, and supported PMREM/CubeUV inputs.
- Output color-space controls, tone mapping controls, clear-color state, clipping planes, post-processing effects, and auxiliary render modes for mask, object-id, normal, and depth output.
- `THREE.Mesh`, `SkinnedMesh`, `InstancedMesh`, `InstancedBufferGeometry`, `BatchedMesh`, sprites, points, lines, dashed lines, LOD, morph targets, geometry groups, vertex colors, frustum culling, layers, render ordering, and custom sort callbacks.
- Common built-in materials including `MeshBasicMaterial`, `MeshLambertMaterial`, `MeshPhongMaterial`, `MeshStandardMaterial`, `MeshPhysicalMaterial`, `MeshNormalMaterial`, `MeshMatcapMaterial`, `MeshToonMaterial`, `MeshDepthMaterial`, `MeshDistanceMaterial`, `ShadowMaterial`, sprite, point, line, and dashed-line materials.
- PBR metallic/roughness lighting, physical material extensions, alpha testing, alpha hash, alpha-to-coverage, wireframe output, fog, blending, stencil state, local and global clipping, and material render-state validation.
- Raw texture data and encoded PNG/JPEG/WebP texture buffers, including common channel formats, color-space handling, UV transforms, selected UV channels, wrap modes, filters, generated mipmaps, explicit raw mipmaps on supported slots, anisotropy, canvas-like texture readback, and WebP helper probing.
- Ambient, directional, point, spot, hemisphere, rect-area, and light-probe lighting, plus directional, spot, point, and cascaded directional shadows within the documented native shadow-layer budget.
- Image-based lighting from scene environments, reflection probes, and supported material `envMap` inputs, including equirectangular, six-face cube, CubeUV, and packed PMREM/CubeUV sources.
- VRM and VRMA still-frame flows through the optional Pixiv loaders, CPU skinning, skeletal animation, and morph targets.
- WebGLRenderer/CommonRenderer compatibility state for many high-level renderer properties and no-op lifecycle methods that existing Three.js integrations probe.
- Custom WGSL fragment bodies through `material.userData.headlessThreeRenderer.fragmentWgsl`.

ShaderMaterial, RawShaderMaterial, NodeMaterial, browser WebGL context access, real WebXR session binding, compressed texture decoding, live video texture readback, WebGPU compute, and direct low-level WebGL buffer or texture binding are outside the scene-oriented API unless the compatibility matrix documents a covered adapter path.

Invalid option containers, malformed Three.js-like objects, unsupported constants, incompatible texture formats, and unsupported renderer escape hatches fail with explicit errors instead of falling through to a browser-only path.

## Native Packages

The main package depends on optional native packages for the current platform:

- `@headless-three/renderer-darwin-arm64`
- `@headless-three/renderer-darwin-x64`
- `@headless-three/renderer-linux-arm64-gnu`
- `@headless-three/renderer-linux-x64-gnu`
- `@headless-three/renderer-win32-x64-msvc`

Install the main package, not the platform package directly.

## Documentation

- [Compatibility matrix](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/compatibility.md)
- [Node loader setup](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/node-loader-setup.md)
- [Scale budgets](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/scale-budgets.md)
- [GPU-native output](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/gpu-native-output.md)
- [Release notes](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/release-notes.md)
- [glTF example](https://github.com/portwatcher/headless-three-renderer/blob/main/examples/render-gltf.mjs)
- [VRM/VRMA example](https://github.com/portwatcher/headless-three-renderer/blob/main/examples/render-vrm.mjs)
