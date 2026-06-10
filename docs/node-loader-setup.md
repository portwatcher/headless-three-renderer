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
import { pathToFileURL, fileURLToPath } from 'node:url'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { render } from '@headless-three/renderer'

async function loadGltfFromFile(filePath) {
  installLocalFileFetch()

  const absolute = path.resolve(filePath)
  const root = path.dirname(absolute)
  const bytes = await fs.readFile(absolute)

  const manager = new THREE.LoadingManager()
  manager.addHandler(/\.(png|jpe?g|webp)$/i, new EncodedImageTextureLoader(root))

  const loader = new GLTFLoader(manager)
  const baseUrl = pathToFileURL(`${root}${path.sep}`).href

  return await new Promise((resolve, reject) => {
    loader.parse(arrayBufferView(bytes), baseUrl, resolve, reject)
  })
}

class EncodedImageTextureLoader {
  constructor(rootDir) {
    this.rootDir = rootDir
  }

  setCrossOrigin() {
    return this
  }

  setRequestHeader() {
    return this
  }

  load(url, onLoad, _onProgress, onError) {
    const texture = new THREE.Texture()
    fs.readFile(localAssetPath(url, this.rootDir)).then((buffer) => {
      texture.image = buffer
      texture.source.data = buffer
      texture.needsUpdate = true
      onLoad?.(texture)
    }, onError)
    return texture
  }
}

function localAssetPath(url, rootDir) {
  if (/^data:/i.test(url)) {
    throw new Error('Data URI textures should be decoded or written to files before loading in Node.')
  }
  if (/^file:/i.test(url)) return fileURLToPath(url)
  if (/^[a-z][a-z0-9+.-]*:/i.test(url)) {
    throw new Error(`Remote texture URL is not a local file: ${url}`)
  }
  return path.resolve(rootDir, url)
}

function arrayBufferView(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
}

function installLocalFileFetch() {
  const marker = Symbol.for('headless-three-renderer.local-file-fetch')
  if (globalThis[marker]) return

  if (typeof globalThis.ProgressEvent === 'undefined') {
    globalThis.ProgressEvent = class ProgressEvent extends Event {
      constructor(type, init = {}) {
        super(type)
        this.lengthComputable = init.lengthComputable ?? false
        this.loaded = init.loaded ?? 0
        this.total = init.total ?? 0
      }
    }
  }

  const nativeFetch = globalThis.fetch
  globalThis.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input?.url
    if (typeof url === 'string' && url.startsWith('file:')) {
      const buffer = await fs.readFile(fileURLToPath(url))
      return new Response(buffer)
    }
    if (!nativeFetch) throw new Error('global fetch is not available')
    return nativeFetch(input, init)
  }

  globalThis[marker] = true
}

const gltf = await loadGltfFromFile('./model.gltf')

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
camera.position.set(2, 1.5, 4)
camera.lookAt(0, 0, 0)

const imageBuffer = render(gltf.scene, camera, { width: 1024, height: 1024 })
await fs.writeFile('render.png', imageBuffer)
```

The custom image handler returns a `THREE.Texture` whose `image` and
`source.data` are encoded PNG/JPEG/WebP bytes. The renderer decodes those bytes
natively, so no DOM `Image`, canvas, or WebGL context is needed for external
image files.

## FileLoader And Fetch

Three.js `FileLoader` uses `fetch()`. Modern Node includes `fetch`, but its
built-in implementation does not consistently read `file://` URLs. The
`installLocalFileFetch()` bridge above adds the `ProgressEvent` global expected
by Three.js, handles local external `.bin` files, and delegates HTTP(S) URLs to
the native fetch implementation.

Use a `file://` base URL from `pathToFileURL()` when the glTF references local
external `.bin` buffers. `GLTFLoader` resolves those URLs before calling
`FileLoader`.

## Embedded Images

For images embedded in GLB files or glTF bufferViews, `GLTFLoader` converts the
bufferView into a `Blob` URL and then uses its internal image loader. In plain
Node, that path needs an image implementation such as a `createImageBitmap`
polyfill, or a preprocessing step that rewrites embedded images as external
PNG/JPEG/WebP files so the encoded-buffer loader above can handle them.

After loading, texture slots should expose one of the renderer-supported image
forms:

- `texture.image = Buffer | Uint8Array` for encoded PNG/JPEG/WebP bytes.
- `texture.source.data = Buffer | Uint8Array` for encoded PNG/JPEG/WebP bytes.
- `texture.image = { data, width, height }` for raw RGBA8 data.

Compressed KTX2/Basis textures are not decoded by the renderer yet. Decode them
to RGBA data or convert them to PNG/JPEG/WebP before rendering.

## Optional Polyfills

Only install polyfills that your loader path actually uses:

- `fetch`: needed by Three.js `FileLoader` on older Node versions.
- `Blob` and object URLs: needed for embedded glTF images.
- `createImageBitmap`: needed if you want `GLTFLoader` to use
  `ImageBitmapLoader` for embedded images.
- DOM `Image`/`document.createElementNS`: needed only when falling back to
  Three.js `TextureLoader`/`ImageLoader`.

If a polyfilled image loader returns browser image objects instead of encoded
bytes or raw RGBA data, normalize those textures before calling `render()`.
