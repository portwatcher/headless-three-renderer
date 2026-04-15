# gltf-renderer-rs

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

Current rendering support is intentionally focused on static triangle meshes:

- `THREE.Mesh`
- `THREE.BufferGeometry` positions and indices
- geometry groups with material arrays
- mesh world transforms
- material base color and opacity
- vertex colors
- scene background color
- perspective, orthographic, and custom projection matrices

Not yet implemented: textures, lights, shadows, skinning, morph targets, custom shaders, render targets, lines, points, and Three.js material models beyond flat color/vertex color. Those require additional `wgpu` pipeline and shader work.
