# Node Loader Setup

This renderer accepts normal Three.js scene objects, but Three.js loaders still
need Node-aware file and image loading. Browser-oriented examples often rely on
DOM image elements, object URLs, or WebGL upload behavior that is not available
in a plain Node process.

## Load A Local glTF Or GLB

Prefer `GLTFLoader.parse()` with bytes read from disk. Pass a base URL ending in
`/` so relative `.bin`, `.png`, `.jpg`, and `.webp` references resolve against
the model directory.

```js
import fs from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import {
  createEncodedImageTextureLoader,
  installLocalFileFetch,
  render,
} from '@headless-three/renderer'

async function loadGltfFromFile(filePath) {
  installLocalFileFetch()

  const absolute = path.resolve(filePath)
  const root = path.dirname(absolute)
  const bytes = await fs.readFile(absolute)

  const manager = new THREE.LoadingManager()
  const encodedImages = createEncodedImageTextureLoader(root)
  manager.addHandler(/^blob:/i, encodedImages)
  manager.addHandler(/^data:image\/(?:png|jpe?g|webp)/i, encodedImages)
  manager.addHandler(/\.(png|jpe?g|webp)$/i, encodedImages)

  const loader = new GLTFLoader(manager)
  const baseUrl = pathToFileURL(`${root}${path.sep}`).href

  return await new Promise((resolve, reject) => {
    loader.parse(arrayBufferView(bytes), baseUrl, resolve, reject)
  })
}

function arrayBufferView(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
}

const gltf = await loadGltfFromFile('./model.gltf')

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
camera.position.set(2, 1.5, 4)
camera.lookAt(0, 0, 0)

const imageBuffer = render(gltf.scene, camera, { width: 1024, height: 1024 })
await fs.writeFile('render.png', imageBuffer)
```

The repository includes the same pattern as a runnable script:

```bash
pnpm --filter @headless-three/renderer build:ts
node examples/render-gltf.mjs ./model.gltf render.png
```

`createEncodedImageTextureLoader()` returns a Three.js loader handler whose
textures expose encoded PNG/JPEG/WebP bytes through `texture.image` and
`texture.source.data`. The renderer decodes those bytes natively, so no DOM
`Image`, canvas, or WebGL context is needed for external image files or
PNG/JPEG/WebP data URI or Blob URL image references.

## FileLoader And Fetch

Three.js `FileLoader` uses `fetch()`. Modern Node includes `fetch`, but its
built-in implementation does not consistently read `file://` URLs. The
`installLocalFileFetch()` adds the `ProgressEvent` global expected by Three.js,
handles local external `.bin` files, and delegates HTTP(S) URLs to the native
fetch implementation.

Use a `file://` base URL from `pathToFileURL()` when the glTF references local
external `.bin` buffers. `GLTFLoader` resolves those URLs before calling
`FileLoader`.

## Embedded Images

For images embedded directly as PNG/JPEG/WebP data URIs in `.gltf` JSON, register
the encoded-buffer loader for data URI image schemes:

```js
const encodedImages = createEncodedImageTextureLoader(root)
manager.addHandler(/^data:image\/(?:png|jpe?g|webp)/i, encodedImages)
manager.addHandler(/\.(png|jpe?g|webp)$/i, encodedImages)
```

For images embedded in GLB files or glTF bufferViews, `GLTFLoader` converts the
bufferView into a `Blob` URL and then uses its internal image loader. On modern
Node versions with `Blob`, `URL.createObjectURL`, and `fetch(blobUrl)` support,
register the same encoded-buffer loader for Blob URLs:

```js
manager.addHandler(/^blob:/i, encodedImages)
```

If your Node version or loader stack does not provide Blob URL fetch support,
install the needed polyfills or preprocess embedded images as external
PNG/JPEG/WebP files or data URIs so the encoded-buffer loader can handle them.

After loading, texture slots should expose one of the renderer-supported image
forms:

- `texture.image = Buffer | Uint8Array` for encoded PNG/JPEG/WebP bytes.
- `texture.source.data = Buffer | Uint8Array` for encoded PNG/JPEG/WebP bytes.
- `texture.image = { data, width, height }` for raw RGB/RGBA data.

Compressed KTX2/Basis textures are not decoded by the renderer yet. Decode them
to RGB/RGBA data or convert them to PNG/JPEG/WebP before rendering.

## Optional Polyfills

Only install polyfills that your loader path actually uses:

- `fetch`: needed by Three.js `FileLoader` on older Node versions.
- `Blob` and object URLs: needed for embedded glTF images.
- `createImageBitmap`: needed if you want `GLTFLoader` to use
  `ImageBitmapLoader` for embedded images.
- DOM `Image`/`document.createElementNS`: needed only when falling back to
  Three.js `TextureLoader`/`ImageLoader`.

If a polyfilled image loader returns browser image objects instead of encoded
bytes or raw RGB/RGBA data, normalize those textures before calling `render()`.
The renderer fails clearly for browser `Image`/`ImageBitmap`/canvas-like
texture objects because their pixels are not readable in headless Node.
