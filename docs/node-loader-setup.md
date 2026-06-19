# Node Loader Setup

This renderer accepts normal Three.js scene objects, but Three.js loaders still
need Node-aware file and image loading. Browser-oriented examples often rely on
DOM image elements, object URLs, or WebGL upload behavior that is not available
in a plain Node process.

## Load A Local glTF Or GLB

Use `loadGltfFromFile()` for local `.gltf` or `.glb` files. Its `filePath`
argument accepts relative paths, absolute paths, and `file://` URLs. It reads
bytes from disk, installs the local `file://` fetch bridge used by Three.js
`FileLoader`, and registers encoded PNG/JPEG/WebP image handlers so external
files, data URI images, and embedded GLB bufferView images expose
renderer-supported buffers.

```js
import fs from 'node:fs/promises'
import * as THREE from 'three'
import {
  loadGltfFromFile,
  render,
} from '@headless-three/renderer'

const gltf = await loadGltfFromFile('./model.gltf')

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
camera.position.set(2, 1.5, 4)
camera.lookAt(0, 0, 0)

const imageBuffer = render(gltf.scene, camera, { width: 1024, height: 1024 })
await fs.writeFile('render.png', imageBuffer)
```

If you need custom loader plugins, register them through the configuration hook
before the file is parsed:

```js
import { loadGltfFromFile } from '@headless-three/renderer'
import { VRMLoaderPlugin } from '@pixiv/three-vrm'

const gltf = await loadGltfFromFile('./avatar.vrm', {
  configureLoader(loader) {
    loader.register((parser) => new VRMLoaderPlugin(parser))
  },
})
```

For VRM avatars, `loadVrmFromFile()` installs `@pixiv/three-vrm`'s
`VRMLoaderPlugin` for you. Pass the plugin constructor explicitly when you
already imported the Pixiv package and want to avoid an extra dynamic import:

```js
import { VRMLoaderPlugin } from '@pixiv/three-vrm'
import { loadVrmFromFile } from '@headless-three/renderer'

const gltf = await loadVrmFromFile('./avatar.vrm', { VRMLoaderPlugin })
const vrm = gltf.userData.vrm
```

For VRMA clips, use `loadVrmAnimationFromFile()` with
`@pixiv/three-vrm-animation` installed. Use `applyVrmAnimation()` to create a
VRMA clip, seek an animation mixer to a still-frame time with `setTime()` or the
`update()` fallback, and update the avatar before rendering. Pass
`updateDelta` to control the VRM update step, or `updateVrm: false` when the
caller will update the avatar manually.

For fully custom loading flows, `createNodeGltfLoader(rootDir)` returns the
configured `{ loader, manager, encodedImages }` bundle so callers can add more
handlers or reuse the loader directly.

Local helper paths are normalized through `resolveLocalAssetPath(url, rootDir)`.
Relative paths resolve under `rootDir`, POSIX absolute paths and Windows
drive-letter paths are preserved as local files, and `file://` URLs are decoded
with Node's `fileURLToPath()`. URI-encoded local filenames such as
`Texture%20BaseColor.png` and external buffers with literal spaces resolve
through the same local file paths. Data URI images are handled by the encoded
image loader, Blob URLs are handled by the Blob URL image path, and remote
HTTP(S) asset URLs are rejected by the local helpers.

The repository includes the same pattern as a runnable script:

```bash
pnpm --filter @headless-three/renderer build:ts
node examples/render-gltf.mjs ./model.gltf render.png
```

For VRM avatars and optional VRMA animation clips, install the Pixiv loader
packages in your project and use the VRM example:

```bash
pnpm add @pixiv/three-vrm @pixiv/three-vrm-animation
pnpm --filter @headless-three/renderer build:ts
node examples/render-vrm.mjs ./avatar.vrm ./dance.vrma render.png
```

`createEncodedImageTextureLoader()` returns the lower-level
`EncodedImageTextureLoader` Three.js loader handler used by `loadGltfFromFile()`.
It supports both callback-style `load()` and promise-style `loadAsync()`. Its
textures expose encoded PNG/JPEG/WebP bytes through `texture.image` and
`texture.source.data`, and it reports optional `LoadingManager`
`itemStart`/`itemEnd`/`itemError` hooks when a manager is supplied. The renderer
decodes those bytes natively, so no DOM `Image`, canvas, or WebGL context is
needed for external image files or PNG/JPEG/WebP data URI or Blob URL image
references.

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

For images embedded in GLB files as PNG/JPEG/WebP bufferViews, the exported
helpers normalize those embedded image references to data URIs before
`GLTFLoader` parses the asset. This avoids the browser-only default image loader
path that Three.js otherwise uses for bufferView images in Node.

For custom flows that still produce Blob URLs, register the same encoded-buffer
loader for Blob URLs:

```js
manager.addHandler(/^blob:/i, encodedImages)
```

If your loader stack bypasses the exported helpers, preprocess embedded images
as external PNG/JPEG/WebP files or data URIs so the encoded-buffer loader can
handle them.

After loading, texture slots should expose one of the renderer-supported image
forms:

- `texture.image = Buffer | Uint8Array` for encoded PNG/JPEG/WebP bytes.
- `texture.source.data = Buffer | Uint8Array` for encoded PNG/JPEG/WebP bytes.
- `texture.image = { data, width, height }` for raw one-channel, two-channel, RGB, or RGBA data.

Compressed KTX2/Basis textures are not decoded by the renderer yet. External or
embedded compressed image payloads fail during helper loading with pre-decode
guidance; decode them to RGB/RGBA data or convert them to PNG/JPEG/WebP before
rendering.

## Optional Polyfills

Only install polyfills that your loader path actually uses:

- `fetch`: needed by Three.js `FileLoader` on older Node versions.
- `Blob` and object URLs: needed for embedded glTF images.
- `createImageBitmap`: needed if you want `GLTFLoader` to use
  `ImageBitmapLoader` for embedded images.
- DOM `Image`/`document.createElementNS`: needed only when falling back to
  Three.js `TextureLoader`/`ImageLoader`.

If a polyfilled image loader returns browser image objects instead of encoded
bytes or raw one-channel, two-channel, RGB, or RGBA data, normalize those
textures before calling `render()`.
The renderer fails clearly for browser `Image`/`ImageBitmap`/canvas-like
texture objects because their pixels are not readable in headless Node.
