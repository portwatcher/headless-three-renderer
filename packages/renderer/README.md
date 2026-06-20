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

`Renderer.renderAsync(scene, camera, options)` is a Promise-returning compatibility wrapper around the same scene-output contract.

`Renderer.sortObjects`, `Renderer.setOpaqueSort(fn)`, `Renderer.setTransparentSort(fn)`, and the matching `render()` options (`sortObjects`, `opaqueSort`, `transparentSort`) control native draw-list sorting; invalid option or setter values fail clearly.
`Renderer.opaque` and `Renderer.transparent` are validated CommonRenderer compatibility flags that gate opaque and transmissive/transparent bucket rendering.

It also exports Node loader helpers:

- `applyVrmAnimation(vrm, vrmAnimation, options)`: creates a VRMA animation clip with `createVRMAnimationClip`, accepting either direct VRM/VRMA objects or the glTF wrappers returned by `loadVrmFromFile()`/`loadVrmAnimationFromFile()`, seeks a `THREE.AnimationMixer` to `options.time` through `setTime()` or `update()` fallback, and updates the VRM scene for still-frame rendering unless `updateVrm: false` is passed.
- `loadGltfFromFile(filePath, options)`: loads local `.gltf` or `.glb` files from relative paths, absolute paths, or `file://` URLs with encoded texture handlers and local `file://` buffer support already installed; malformed helper paths, option containers, and glTF image metadata fail clearly.
- `loadVrmFromFile(filePath, options)`: loads local VRM files with `@pixiv/three-vrm`'s `VRMLoaderPlugin` registered. The Pixiv package remains an optional dependency in your project.
- `loadVrmAnimationFromFile(filePath, options)`: loads local VRMA files with `@pixiv/three-vrm-animation`'s `VRMAnimationLoaderPlugin` registered. The animation package remains optional.
- `createNodeGltfLoader(rootDir, options)`: creates a configured `GLTFLoader` bundle for advanced flows, including plugin registration through `options.configureLoader`; `rootDir` accepts relative paths, absolute paths, and `file://` URLs, and malformed helper boolean options, callback hooks, and custom managers fail clearly.
- `createEncodedImageTextureLoader(rootDir, manager)` / `EncodedImageTextureLoader`: a `LoadingManager` image handler with `load()` and `loadAsync()` for local PNG/JPEG/WebP files, PNG/JPEG/WebP data URIs, and PNG/JPEG/WebP Blob URLs that exposes encoded buffers directly to renderer-supported texture slots, reports optional manager item start/end/error hooks, and resolves `setPath()` directory prefixes for relative, absolute, and `file://` paths without rewriting data, Blob, absolute, or fully-qualified asset URLs; `rootDir` accepts relative paths, absolute paths, and `file://` URLs, and malformed helper paths, callbacks, and manager objects fail clearly.
- `installLocalFileFetch()`: a small `file://` fetch bridge for Three.js `FileLoader` when loading local external glTF buffers.
- `resolveLocalAssetPath(url, rootDir)`: shared path resolution for local loader helpers, covering relative paths under relative, absolute, or `file://` roots plus POSIX/Windows absolute paths and `file://` asset URLs while rejecting remote asset/root URLs.

## Supported Three.js Surface

See the versioned [compatibility matrix](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/compatibility.md) for the public support contract, known gaps, and platform package status. Scale-test budgets and platform notes are documented in [docs/scale-budgets.md](https://github.com/portwatcher/headless-three-renderer/blob/main/docs/scale-budgets.md).

The public API accepts only Three.js-like objects:

- `scene`: a `THREE.Scene` or `THREE.Object3D` root; malformed scene/children containers and visibility flags fail clearly.
- `camera`: a `THREE.Camera`, including perspective and orthographic cameras. Malformed camera/userData containers, invalid aspect-derived dimensions, clipping distances, and matrix containers or values fail clearly. `THREE.ArrayCamera` composes sub-camera viewports for PNG, raw RGBA, and target output, with malformed sub-camera containers failing clearly. `THREE.CubeCamera` renders six RGBA faces plus optional depth faces into `WebGLCubeRenderTarget.texture.image`/`source.data`, nonzero `activeMipmapLevel` writes the active mip entry, `CubeCamera.update(renderer, scene)` works with the reusable renderer's minimal target state while preserving inert `Renderer.xr` enabled/cameraAutoUpdate/framebuffer-scale/controller/reference-space/session/base-layer/binding/frame/environment/depth-texture/depth/foveation/camera-texture/event-target dispatch probes, and captured color textures can be reused as cube background/environment inputs; real XR session binding and malformed child-camera containers fail clearly, while exact WebGL face semantics remain limited.
- `options`: an options object; malformed option containers fail clearly.
- `options.width` and `options.height`: output pixel size. Defaults to `512 x 512`; invalid explicit dimensions fail clearly.
- `options.background`: `[r, g, b]`, `[r, g, b, a]`, a CSS color string, a `THREE.Color`, a supported 2D/equirectangular/cube texture, or `null` to clear `scene.background` for one render. Defaults to `scene.background`; option-supplied backgrounds use option-supplied background controls rather than scene background controls, and malformed scene or option background values fail clearly.
- `options.backgroundIntensity`: overrides `scene.backgroundIntensity` for supported color and texture backgrounds; invalid values fail clearly.
- `options.backgroundBlurriness`: overrides `scene.backgroundBlurriness` for supported texture backgrounds; invalid values fail clearly.
- `options.backgroundRotation`: overrides `scene.backgroundRotation` for supported equirectangular and cube texture backgrounds; explicit option rotation values are always validated, and invalid or unsupported rotations fail clearly.
- `options.environmentIntensity`: overrides `scene.environmentIntensity` or reflection-probe intensity for supported scene environments; invalid values fail clearly.
- `options.environmentRotation`: overrides `scene.environmentRotation` for supported scene environments; explicit option rotation values are always validated, and invalid values fail clearly.
- `options.viewport`: `[x, y, width, height]` or `{ x, y, width, height }` output pixel rectangle, using a top-left origin, for viewport-limited draws; invalid rectangles fail clearly.
- `options.scissor`: `[x, y, width, height]` or `{ x, y, width, height }` output pixel rectangle, using a top-left origin, for scissor-clipped draws; invalid rectangles fail clearly.
- `options.clippingPlanes`: global world-space clipping planes for the render; reusable `Renderer.clippingPlanes` provides the same value as a renderer-state fallback.
- `options.localClippingEnabled`: `false` disables material-local clipping planes while preserving global clipping planes; reusable `Renderer.localClippingEnabled` provides the same value as a renderer-state fallback, defaults to `true`, and invalid values fail clearly.
- `options.format`: `'png'` by default, or `'rgba'` for raw RGBA8 bytes; unsupported values fail clearly.
- `options.outputColorSpace`: `THREE.SRGBColorSpace` (`'srgb'`, default) or `THREE.LinearSRGBColorSpace` (`'srgb-linear'`, `'linear-srgb'`, `'linearsrgb'`, or `'linear'`) for material and texture background output conversion; reusable `Renderer.outputColorSpace`, `currentColorSpace`, and `_outputColorSpace` provide the same value as a renderer-state fallback, and unsupported values fail clearly.
- `options.renderMode`: `'color'` by default, `'mask'` for white visible geometry on black, `'object-id'` for flat RGB object IDs, or `'normal'` for view-space normal colors; invalid values fail clearly.
- `options.target`: a non-array target-like object populated for a color output, including `target.texture`, `target.textures[0]`, one-element `target.texture` arrays, or MRT-shaped targets. `Renderer.setRenderTarget(target); renderer.render(...)` supports the same regular target writeback through minimal reusable-renderer target state, and `Renderer.readRenderTargetPixels()`/`readRenderTargetPixelsAsync()` can copy stored target color data, explicit cube target faces, or selected color attachment indices into caller-provided buffers. Async readback can allocate a matching output buffer and accepts the common-renderer `(target, x, y, width, height, textureIndex, faceIndex)` argument shape. Top-level `target.data` remains raw RGBA8; color textures can also request Alpha/Red/RG/RGB/RGBA and RedIntegerFormat/RGIntegerFormat/RGBIntegerFormat/RGBAIntegerFormat plus normalized `FloatType`, signed/unsigned integer, packed color, or `HalfFloatType` readback arrays. Regular-camera, ArrayCamera, and CubeCamera MRT-shaped targets can populate secondary texture attachments when each secondary texture declares `texture.userData.headlessThreeRenderer.renderMode` as `'color'`, `'mask'`, `'object-id'`, or `'normal'`; `Renderer.getMRT()` returns `null`, `Renderer.setMRT(null)` is accepted as a clear operation, and non-null `Renderer.setMRT()` fails clearly because native MRT shader outputs are not supported.
- `options.postProcessing`: built-in post effects (`exposure`, `contrast`, `saturation`, `vignette`, `grayscale`, `invert`); malformed containers and invalid effect values fail clearly.

### Geometry & Scene

- `THREE.Mesh` and `THREE.SkinnedMesh`, including WebGL-style bounding-sphere frustum culling with `frustumCulled=false` opt-out
- `THREE.InstancedMesh` with `instanceMatrix` and `instanceColor`; invalid explicit instance counts fail clearly
- `THREE.InstancedBufferGeometry` for mesh, point, line, and dashed-line geometry with common offset/color attributes, selected instanced UV attributes, Three.js' default `instanceCount = Infinity`, and `meshPerAttribute` repeat values; invalid explicit instance counts or per-attribute repeat values fail clearly
- `THREE.BatchedMesh` common packed-geometry batches are CPU-expanded with per-instance matrices, colors, visibility flags, deleted/inactive instance entries, deleted/inactive packed geometry ranges, packed geometry groups/material arrays including partial range/group intersections, common per-object sphere frustum culling, range-local internal sorting including transparent material-array groups, `sortObjects=false`, and `customSort` callback context/camera/list handling; malformed batch internals including packed matrix/color texture containers, instance table entries, cached culling bounds, culling controls, and sort controls fail clearly, while exact culling/source-group translation edge cases and native batched drawing remain planned
- `THREE.BufferGeometry` positions, indices, normals, and UV coordinates, with malformed attribute/data/bounding-sphere containers and invalid attribute values failing clearly
- `THREE.Sprite`/`SpriteMaterial` CPU billboards with center, scale, rotation, perspective size attenuation controls, opacity, texture maps, `alphaHash` and `alphaToCoverage` opacity cutouts, scene fog, layers, render ordering, frustum culling, main-pass clipping, directional/spot/point shadow casting, directional/spot custom-depth cutouts, point custom-distance shadow cutouts, and custom/source base-map and alpha-map texture transforms on directional custom-depth and point custom-distance shadow paths; invalid billboard scalar and size-attenuation values fail clearly
- geometry groups with material arrays, with malformed material containers failing clearly
- mesh world transforms, object visibility and `frustumCulled` flags, and object/camera layer containers/masks, with invalid transform matrix, visibility, culling, or layer values failing clearly
- `THREE.LOD` camera-distance/zoom level selection, with invalid auto-update flags, camera zoom, or level distance/hysteresis values failing clearly
- vertex colors, with invalid `material.vertexColors` values failing clearly
- scene background color plus reusable `Renderer.setClearColor()`/`setClearAlpha()` fallback state with hex, CSS string, and color-like clear-color inputs, CSS string scene/option background colors, and 2D, equirectangular, raw, encoded, CubeUV-mapped readable six-face cube, and packed 2D PMREM/CubeUV sharp-atlas texture backgrounds with `backgroundIntensity`, approximate texture blur, equirectangular/cube `scene.backgroundRotation`/`options.backgroundRotation`, `options.environmentIntensity`, and equirectangular/cube `scene.environmentRotation`/`options.environmentRotation`; invalid background color/control/rotation values, invalid renderer clear-color values, malformed packed PMREM/CubeUV background layouts, and unsupported background rotations fail clearly
- render-option viewport/scissor rectangles, render-target dimensions and viewport/scissor fields, reusable `Renderer.setSize()`/`getSize()` plus `setDrawingBufferSize()`/`getDrawingBufferSize()` and `setPixelRatio()`/`getPixelRatio()` compatibility state, inert `Renderer.domElement` output/client-size/style mirror plus style-property, attribute, event-target, and canvas-export probes, `Renderer.outputColorSpace`/`currentColorSpace`/`_outputColorSpace` output-conversion state, read-only `Renderer.coordinateSystem` WebGL coordinate signal, `Renderer.toneMapping`/`currentToneMapping`/`toneMappingExposure` material tone-mapping state plus `Renderer.needsFrameBufferTarget=false` inline-output probe, `Renderer.clippingPlanes` global clipping fallback state, `Renderer.localClippingEnabled` material-local clipping state, `Renderer.info` inert compatibility counters with WebGLInfo- and CommonRenderer-style `update()`/`reset()`/`dispose()`, inert `Renderer.debug` shader-diagnostic state, inert `Renderer.inspector` lifecycle and copy-hook state, conservative `Renderer.capabilities` WebGL limit, draw-buffer, texture probes, and renderer-level `getMaxAnisotropy()`, and conservative `hasFeature()`/`hasFeatureAsync()`/`hasCompatibility()` plus `isOccluded()` probes, inert `Renderer.extensions`, scratch `Renderer.properties`/`renderLists` helpers, inert `Renderer.state` buffer and state-level setter probes with raw WebGL binding/upload failures, inert `Renderer.xr` enabled/cameraAutoUpdate/framebuffer-scale/controller/reference-space/session/base-layer/binding/frame/environment/depth-texture/depth/foveation/camera-texture/event-target dispatch probes, validated inert WebGLRenderer-style constructor parameters for common context attributes plus read-only `alpha`/`depth`/`stencil`/`logarithmicDepthBuffer` metadata, CommonRenderer-style `isRenderer`/`initialized` probes, `highPrecision=false` metadata with `true` failing clearly, `samples`/`currentSamples`/`isOutputTarget` sample/output-target probes, default `outputBufferType` with `getOutputBufferType()`/`getColorBufferType()`, cloned `getContextAttributes()` readback, and clear `getContext()` unsupported failures, `setViewport()`/`getViewport()`/`getCurrentViewport()` and `setScissor()`/`setScissorTest()` state in output pixel coordinates with default-only common-renderer viewport depth-range validation, `Renderer.shadowMap.enabled` shadow gating plus `shadowMap.autoUpdate`/`needsUpdate`/`type` compatibility state, clear color/depth/stencil value state, material-set `compile()`/`compileAsync()` compatibility hooks, no-op `init()`/`initRenderTarget()`/`initTexture()`/`initTextureAsync()`/`hasInitialized()`/`clear()`/`clearAsync()`/`clearTarget()`/`clearColor()`/`clearColorAsync()`/`clearDepth()`/`clearDepthAsync()`/`clearStencil()`/`clearStencilAsync()`/`resetState()`/`resetGLState()`/`dispose()`/`forceContextLoss()`/`forceContextRestore()`/`setAnimationLoop()`/`getAnimationLoop()` hooks, and no-op `autoClear`/`autoClearColor`/`autoClearDepth`/`autoClearStencil` flags for pass-owned buffers and object-lifetime native cleanup; `getContext()`, `domElement.getContext()`, `domElement` canvas export/capture APIs, `setRenderTargetTextures()`, `setRenderTargetFramebuffer()`, and real `Renderer.xr.setSession()` binding fail clearly because this package has no browser WebGL/WebXR context or external WebGL texture/framebuffer binding; `copyFramebufferToTexture()` supports level-0 CPU copies from the active render target's readable color data into readable raw textures and fails clearly without an active readable target, with out-of-bounds destinations, or with channel-mismatched/non-zero-level inputs; `copyTextureToTexture()` supports level-0 CPU copies from readable raw, canvas-like, or OffscreenCanvas-backed source textures into readable raw destination textures and fails clearly for unreadable, out-of-bounds, channel-mismatched, non-raw destination, or non-zero-level inputs. Invalid sizes, size updateStyle flags, pixel ratios, non-default viewport depth ranges, constructor parameters including constructor-level `samples > 1` or non-default `outputBufferType`, output color spaces, tone-mapping constants/exposure values, clipping plane values/booleans, compile material values, info booleans/update values, debug values, inspector values, capability/extension/property/render-list/state probes, DOM style/attribute/event/canvas probes, XR booleans/values/events/controller/reference-space binding, shadow-map booleans/type constants, clear values/booleans, resource-init targets/textures, external WebGL texture/framebuffer handles, copy targets, unsupported texture-copy inputs, animation-loop callbacks, auto-clear flags, and malformed target scissor flags fail clearly
- `Renderer.shadowMap.transmitted` is accepted as validated boolean compatibility state alongside `shadowMap.autoUpdate`, `shadowMap.needsUpdate`, and `shadowMap.type`
- `Renderer.debug.getShaderAsync()` validates scene, camera, and object inputs but fails clearly because generated backend shader source is not exposed
- `Renderer.onDeviceLost` is assignable callback state for CommonRenderer compatibility; native device-loss handling is not exposed
- `Renderer.setEffects()` is accepted as a validated no-op WebGLRenderer compatibility hook; use `options.postProcessing` for actual effects
- `Renderer.transmissionResolutionScale` is stored as positive finite WebGLRenderer compatibility state and scales the scene-color texture sampled by physical transmission
- `Renderer.isWebGLRenderer` is exposed for Three.js helper branches that use WebGLRenderer-compatible readback signatures, including `LightProbeGenerator.fromCubeRenderTarget()` cube-target readback, standard CopyShader `ShaderPass`/`TexturePass`/`SavePass` fullscreen flows, and `EXRExporter`/`KTX2Exporter` render-target export paths, while WebGL context access still fails clearly
- `Renderer.state` exposes inert `buffers.color`, `buffers.depth`, and `buffers.stencil` no-op setter probes, state-level blending/material/flip-sided/cull-face/line-width/polygon-offset/scissor-test/scissor/viewport probes, `buffers.depth.getReversed()`, no-op `reset()`/`unbindTexture()` hooks, and clear failures for raw WebGL binding/upload state methods
- `Renderer.renderBufferDirect()`, `Renderer.renderObject()`, non-null `Renderer.setRenderObjectFunction()`, non-null `Renderer.setOutputRenderTarget()`/`setCanvasTarget()`, direct `Renderer.setTexture2D()`/`setTextureCube()`/`setTextureCubeDynamic()`/`setTexture3D()`/`setTexture2DArray()`, `Renderer.compute()`/`computeAsync()`, `Renderer.getArrayBufferAsync()`, `Renderer.resolveTimestampsAsync()`, and `Renderer.waitForGPU()` calls fail clearly because direct WebGL buffer binding, renderer-internal render-object dispatch, material program dispatch, backend-owned common-renderer output/canvas targets, browser WebGL texture units, direct texture binding, WebGPU compute pipelines, storage-buffer readback, timestamp query pools, and direct GPU synchronization are outside the scene-oriented API; `Renderer.getRenderObjectFunction()` returns `null`, `Renderer.setRenderObjectFunction(null)` is accepted as a clear operation, `Renderer.getOutputRenderTarget()`/`getCanvasTarget()` return `null`, and `Renderer.setOutputRenderTarget(null)`/`setCanvasTarget(null)` are accepted as clear operations
- perspective, orthographic, and custom projection matrices

### Materials & Textures

- material base color, opacity, and visibility, including CSS string material color inputs, with malformed color containers and invalid color/opacity/visible values failing clearly
- `material.map` (base color texture) — PNG, JPEG, WebP, and raw one-channel, two-channel, RGB, or RGBA numeric DataTexture inputs, including byte, signed/unsigned normalized integer, packed 16-bit color, float, and half-float typed data, with `texture.channel` UV selection plus raw `texture.premultiplyAlpha` and sRGB color-space decode
- base, sprite, point, line/dashed-line, matcap, metallic/roughness, emissive, AO, light, Phong specular, alpha, sheen color, and physical specular color maps honor `THREE.SRGBColorSpace`/`THREE.LinearSRGBColorSpace`, including the documented linear string aliases; unsupported texture color-space/encoding values fail clearly
- base, 2D background, sprite/point color and alpha, line/dashed-line, matcap, normal/bump, displacement, emissive, metallic/roughness, AO/light, Phong specular, alpha, and current physical-extension maps honor texture UV transforms, including explicit texture matrices for those covered slots and color-space decode after explicit matrices for current color-producing transform slots; malformed transform vector containers and invalid transform or transform-boolean values fail clearly
- `texture.channel` supports channels 0-3 on supported map slots, with malformed channel values failing clearly; channels 1-3 route selected non-primary UV attributes through the available native UV streams, mesh material draws can share up to two distinct selected texture channels across current supported slots, and draws requiring three or more distinct texture channels fail clearly
- material and texture background output conversion supports `THREE.SRGBColorSpace` and `THREE.LinearSRGBColorSpace`; texture backgrounds honor `THREE.SRGBColorSpace`/`THREE.LinearSRGBColorSpace`, including the documented linear string aliases, and raw 2D texture backgrounds plus current raw IBL inputs honor `texture.premultiplyAlpha`
- base/background, normal/bump, displacement, metallic/roughness, emissive, AO/light, alpha, Phong specular, toon gradient, matcap color-map, and packed physical-extension texture-group wrap modes plus `NearestFilter`/`LinearFilter`-family `magFilter` and `minFilter`, including direct coverage for background texture repeat/mirrored wrapping, base color-map repeat/mirrored wrapping, normal-map repeat/mirrored wrapping, bump-map repeat/mirrored wrapping, displacement-map repeat/mirrored wrapping, metalness-map repeat/mirrored wrapping, roughness-map repeat/mirrored wrapping, matcap color-map repeat/mirrored wrapping, Phong specular-map repeat/mirrored wrapping, AO-map repeat/mirrored wrapping, emissive-map repeat/mirrored wrapping, light-map repeat/mirrored wrapping, alpha-map repeat/mirrored wrapping, sprite color-map repeat/mirrored wrapping, sprite alpha-map repeat/mirrored wrapping, point color-map repeat/mirrored wrapping, point alpha-map repeat/mirrored wrapping, line color-map repeat/mirrored wrapping, line alpha-map repeat/mirrored wrapping, dashed-line color-map repeat/mirrored wrapping, dashed-line alpha-map repeat/mirrored wrapping, toon gradient-map horizontal repeat/mirrored wrapping, iridescence factor/thickness repeat/mirrored wrapping and filter behavior, generated mip chains for mipmap min filters, raw explicit mip chains for unpacked 2D material/background texture uploads, half-float raw mip level decoding, WebGL-compatible `unpackAlignment` values for tightly packed readable uploads, and clear failures for unsupported sampler constants, invalid mipmap controls, invalid `unpackAlignment`, or invalid anisotropy values
- PBR metallic/roughness via `MeshStandardMaterial` and `MeshPhysicalMaterial`
- `MeshPhysicalMaterial` clearcoat, sheen, anisotropy, scalar iridescence, specular intensity/color, IOR, attenuation, approximate dispersion, and roughness-aware environment-backed or scene-color transmission / refraction; malformed physical color containers and invalid physical color/scalar values fail clearly
- physical material extension maps for clearcoat, clearcoat roughness, clearcoat normals, sheen color/roughness, anisotropy, iridescence factor/thickness, specular color/intensity, transmission, and thickness; all current physical-extension maps include primary/secondary `texture.channel` UV selection, texture transforms including explicit matrices, packed texture-group sampler settings with direct iridescence factor/thickness repeat/mirrored wrap/filter coverage, clear failures for incompatible packed samplers, and sRGB color-space decode for clearcoat, clearcoat roughness, anisotropy, iridescence factor/thickness, transmission, thickness, and sheen/specular color map RGB channels
- custom WGSL fragment bodies via `material.userData.headlessThreeRenderer.fragmentWgsl`; `ShaderMaterial`, `RawShaderMaterial`, NodeMaterial, and `onBeforeCompile` customizations require this explicit override path, and malformed material userData/renderer hint containers fail clearly
- metallic/roughness map (`material.metalnessMap` / `material.roughnessMap`) with color-space decode, primary/secondary `texture.channel` UV selection, texture transforms, metalness-map and roughness-map horizontal/vertical repeat/mirrored wrapping, and nearest/linear filtering
- normal map with configurable `normalScale`, primary/secondary `texture.channel` UV selection, color-space decode, horizontal/vertical repeat/mirrored wrapping, and nearest/linear filtering, plus bump map with `bumpScale`, color-space decode, and horizontal/vertical repeat/mirrored wrapping; invalid scalar values fail clearly
- `MeshNormalMaterial` and `MeshMatcapMaterial` normal-map output
- `material.flatShading` per-face normals for triangle meshes without normal maps
- `MeshMatcapMaterial.map` color maps with primary/secondary `texture.channel` UV selection, transforms, horizontal/vertical repeat/mirrored wrapping, and nearest/linear filtering
- displacement map CPU-baked into triangle vertices with `displacementScale`, `displacementBias`, primary/secondary `texture.channel` UV selection, texture color-space decode, texture transforms, horizontal/vertical repeat/mirrored wrapping, and nearest/linear sampler filtering; invalid scale/bias values fail clearly
- `MeshToonMaterial.gradientMap` red-channel diffuse ramps with sRGB color-space decode, horizontal repeat/mirrored wrapping, and nearest/linear filtering; direct conformance also covers toon normal/bump-map lighting perturbation, base-map UV channels, emissive-map UV channels, light-map secondary UVs, and alpha-map cutouts
- `MeshDepthMaterial.depthPacking`: basic, RGBA, RGB, and RG packing, with clear failures for unsupported depth-packing constants
- `MeshDistanceMaterial` `referencePosition`, `nearDistance`, and `farDistance` overrides, with invalid range/reference values and malformed material userData/renderer hint containers failing clearly, plus alpha-map cutouts and CPU-baked displacement
- main-pass `material.wireframe` output for supported mesh materials, including direct coverage for `MeshBasicMaterial`, `MeshDepthMaterial`, and `MeshDistanceMaterial`; legacy mesh wireframe line hints (`wireframeLinewidth`, `wireframeLinecap`, `wireframeLinejoin`) are validated and accepted as native no-ops
- `Object3D.customDepthMaterial` and `customDistanceMaterial` for mesh shadow caster alpha-tested inputs, including selected base-map and alpha-map UV channels, base-map and alpha-map texture transforms, displacement, visibility flags, and wireframe material inputs, plus source-material base/alpha maps with selected base-map and alpha-map UV channels, base-map and alpha-map texture transforms, opacity, clipping, displacement, `shadowSide`, and wireframe state on custom shadow casters, alpha-tested sprite/point billboard shadow cutouts, sprite custom-depth/custom-distance source/custom base-map and alpha-map texture transforms, and point-billboard custom-distance base-map/alpha-map cutouts with selected geometry UV channels and texture transforms; malformed custom shadow material containers and visibility values fail clearly
- emissive color, intensity, and emissive map, with primary/secondary `texture.channel` UV selection, texture transforms, sRGB color-space decode, horizontal/vertical repeat/mirrored wrapping, and nearest/linear filtering; malformed color containers and invalid color/intensity values fail clearly
- light maps with `lightMapIntensity`, primary/secondary `texture.channel` UV selection, texture transforms, sRGB color-space decode, horizontal/vertical repeat/mirrored wrapping, and nearest/linear filtering; invalid intensity values fail clearly
- occlusion map (`material.aoMap`) applied to indirect lighting, with color-space decode, primary/secondary `texture.channel` UV selection, texture transforms, horizontal/vertical repeat/mirrored wrapping, and nearest/linear filtering; invalid intensity values fail clearly
- alpha map (`material.alphaMap`) using Three.js' green-channel opacity convention, with color-space decode, primary/secondary `texture.channel` UV selection, texture transforms, horizontal/vertical repeat/mirrored wrapping, and nearest/linear filtering
- `MeshPhongMaterial.specularMap` red-channel specular strength, with finite `shininess`, color-space decode, primary/secondary `texture.channel` UV selection, texture transforms, horizontal/vertical repeat/mirrored wrapping, nearest/linear filtering, and masking for scene-level, reflection-probe, and supported material-level environment specular reflections
- `MeshBasicMaterial`, `MeshLambertMaterial`, and `MeshPhongMaterial` material env maps for one shared material-level reflection or refraction map, including legacy multiply/mix/add combine modes, `reflectivity`, and `refractionRatio`; invalid env-map scalar values fail clearly
- `MeshStandardMaterial`, `MeshPhysicalMaterial` (PBR), `MeshLambertMaterial` (diffuse-only), and `MeshBasicMaterial` (unlit)
- `ShadowMaterial` transparent receiver output with color, opacity, scene fog, Fog/FogExp2 fog opt-out, and output color-space conversion
- `material.side`: `FrontSide`, `BackSide`, `DoubleSide`, with clear failures for unsupported side constants
- `material.fog = false` opt-out for scene fog on mesh, shadow, sprite, point, and line material paths; CSS string fog colors are accepted, while malformed fog color containers and invalid fog color/parameter values fail clearly
- alpha test (`material.alphaTest`) with fragment discard and alpha-to-coverage threshold smoothing on multisampled main-pass renders; invalid values fail clearly
- native draw ordering honors group order, `renderOrder`, material id, WebGL material variant, transmissive/transparent buckets, projected geometry bounding-sphere z, object/insertion ties, `sortObjects`, and custom opaque/transparent sort callbacks with object/material/geometry/group render-item metadata, including source-object metadata for BatchedMesh-expanded draws; transparency sorting is back-to-front with `material.depthWrite` overrides, including Three.js' default transparent depth writes; invalid `renderOrder` and sort-control values fail clearly
- renderer-level `opaque` and `transparent` flags can skip opaque or transmissive/transparent render buckets, with invalid flag values failing clearly
- material render state: `depthTest`, `depthFunc`, `depthWrite`, `colorWrite`, `polygonOffset`, `alphaHash`, `alphaToCoverage` on 4x MSAA renders including output-alpha and alpha-test threshold coverage, `premultipliedAlpha`, `toneMapped=false` output opt-out, boolean-validated `dithering`, supported `precision` strings, and mesh wireframe line hints as native no-ops, stencil state, built-in blending modes, `CustomBlending` equations/factors, and clear failures for unsupported render-state constants or invalid boolean/numeric values
- render-option global clipping planes, reusable `Renderer.clippingPlanes` global fallback planes, and material-local clipping planes, with `options.localClippingEnabled: false` and reusable `Renderer.localClippingEnabled = false` available to ignore material-local planes, alpha-to-coverage smoothing for MSAA main-pass clipping edges, and `material.clipShadows`/group `clipShadows` clipping mesh and line shadow casters; malformed plane containers, invalid plane/control values, invalid clipping control booleans, and over-budget global/group/material combinations beyond eight active planes fail clearly
- single shared material-level reflection/refraction `envMap` inputs are supported for `MeshBasicMaterial`, `MeshLambertMaterial`, and `MeshPhongMaterial`, and shared reflection `envMap` inputs are supported for `MeshStandardMaterial` and `MeshPhysicalMaterial` through the native IBL path, including CubeUV-mapped six-face cube and packed 2D PMREM/CubeUV sharp/base-atlas env maps; `envMap` properties on material classes that do not consume material environment maps are ignored, while unsupported material classes, unsupported material env-map options, PBR material refraction mappings, malformed packed 2D PMREM/CubeUV mappings, multiple distinct material env maps, and multiple distinct material env-map rotations fail clearly
- texture wrap modes: repeat, mirror, clamp-to-edge
- mipmap min filters generate native mip chains from supported material/background texture source levels; raw `DataTexture`-style explicit mipmap arrays upload for unpacked 2D material/background texture slots, while malformed texture source containers plus packed physical-extension maps and environment/reflection-probe/material-envMap explicit mip arrays fail clearly
- texture anisotropy values greater than 1 use native anisotropic samplers for supported material/background texture slots when the effective sampler is linear-filtered; invalid anisotropy values fail clearly
- line material arrays honor geometry groups; `LineBasicMaterial.linewidth` and `LineDashedMaterial.linewidth` values greater than 1 expand to camera-facing quads; `linecap`/`linejoin` and unlit `receiveShadow` are accepted as WebGL-compatible no-ops, with invalid cap/join values failing clearly; dashed line material segments honor dash/gap/scale settings, treat `scale=0` as solid like WebGL, and support custom `lineDistance` attributes including descending spans, keep missing `lineDistance` attributes solid like WebGL, and preserve map UV transforms including explicit matrices, selected `texture.channel` UVs, selected instanced map UV attributes, and interpolated vertex colors for common `LineDashedMaterial` cases, including instanced line geometry; invalid negative or non-finite line scalar values fail clearly

Texture image data can be:

- Raw one-channel, two-channel, RGB, or RGBA numeric pixels via `THREE.DataTexture` (or any image with `.data`, `.width`, `.height`), including `UnsignedByteType`, normalized `ByteType`/`ShortType`/`UnsignedShortType`/`IntType`/`UnsignedIntType`, packed `UnsignedShort4444Type`/`UnsignedShort5551Type`, normalized float arrays, and `HalfFloatType` `Uint16Array` binary16 data
- Encoded PNG, JPEG, or WebP image buffers (auto-decoded on the native side)

Compressed KTX2/Basis/`THREE.CompressedTexture` inputs and compressed texture format constants are not decoded in-process; pre-decode them to RGB/RGBA data or an encoded PNG/JPEG/WebP image before rendering. Canvas-like texture images that expose `getContext("2d").getImageData()` are read directly in Node, and image-like objects can be read through an available `OffscreenCanvas`/2D canvas polyfill that supports `drawImage()` plus `getImageData()`; opaque browser `Image`/`ImageBitmap` objects still fail clearly when no readable or drawable pixel path is available. Mismatched-length raw texture payloads fail clearly.

### Lights

- `THREE.AmbientLight` — uniform ambient illumination
- `THREE.DirectionalLight` — sun-like parallel light with position/target
- `THREE.PointLight` — omnidirectional light with distance/decay attenuation
- `THREE.SpotLight` — cone light with angle, penumbra, distance, and decay
- `THREE.HemisphereLight` — sky/ground gradient ambient light
- `THREE.RectAreaLight` — one-sided finite-area direct-light approximation
- `THREE.LightProbe` — diffuse spherical-harmonics indirect lighting plus `LightProbeGenerator.fromCubeRenderTarget()` cube-target readback interop, with invalid coefficient values failing clearly

Lights are automatically extracted from the scene, with CSS string light colors accepted and malformed light color/target containers, invalid light color, numeric controls, transform matrix values, shadow flags, and shadow option containers failing clearly. The shader uses a Cook-Torrance PBR BRDF (GGX/Trowbridge-Reitz distribution, Schlick-GGX geometry, Schlick Fresnel) with Three.js-compatible physically-based attenuation for punctual lights. Up to 64 direct lights per scene are supported. Visible directional, spot, and point lights may cast shadows while their packed native shadow-map usage stays within twelve array layers. When no lights are present, meshes render with a hemispherical ambient fallback.

### Image-Based Lighting (IBL)

Environment maps set on `scene.environment` are supported for image-based lighting. A single shared material-level reflection `envMap` can also feed the same native IBL path for `MeshBasicMaterial`, `MeshStandardMaterial`, `MeshPhysicalMaterial`, `MeshPhongMaterial`, and `MeshLambertMaterial`, with per-material intensity and one shared material env-map rotation. Material `envMap` properties on classes that Three.js does not shade with material environment maps are accepted as no-ops. The renderer CPU-precomputes:

- **Diffuse irradiance cubemap** — cosine-weighted hemisphere convolution
- **Prefiltered specular cubemap** — GGX importance-sampled at multiple roughness mip levels
- **BRDF integration LUT** — split-sum approximation lookup table

Supported input formats: equirectangular images in RGB/RGBA byte data, Float16 (`HalfFloatType`), or Float32 (`FloatType`), plus raw or encoded six-face cube reflection textures, CubeUV-mapped six-face cube inputs, and packed 2D PMREM/CubeUV sharp/base-atlas inputs routed through the same CPU IBL precompute. Raw scene-environment, reflection-probe, and supported material-level IBL inputs honor `texture.premultiplyAlpha`. Scene-environment, reflection-probe, and supported material-level LDR inputs honor explicit `THREE.SRGBColorSpace` and `THREE.LinearSRGBColorSpace`, including the documented linear string aliases; omitted color space defaults to sRGB for compatibility. `MeshBasicMaterial`, `MeshLambertMaterial`, and `MeshPhongMaterial` env maps support legacy multiply/mix/add combine modes with `reflectivity` plus refraction mappings with `refractionRatio`; malformed scene-environment/reflection-probe values, malformed reflection-probe hint containers, unsupported IBL texture input classes, raw data layouts/types, explicit mipmaps, color-space/legacy-encoding values, PBR material refraction mappings, malformed packed 2D PMREM/CubeUV environment inputs, multiple distinct material env maps, multiple distinct material env-map rotations, invalid environment intensity values, and invalid material env-map scalar values fail clearly. Exact PMREM prefiltered LOD semantics remain planned beyond decoded sharp/base atlas faces. `scene.environmentIntensity` is respected for scene environments, reflection-probe intensity applies only when `scene.environment` is absent, and `options.environmentIntensity` can override scene or reflection-probe intensity for one render.

Scene-level reflection probes are supported through `scene.userData.headlessThreeRenderer.reflectionProbe` or the first entry in `reflectionProbes`. Probe textures use the same equirectangular and cube texture formats as `scene.environment` and feed the same diffuse/specular IBL path.

`scene.userData` must be an object when present; `scene.userData.headlessThreeRenderer` and the legacy `scene.userData.headlessRenderer` key must also be objects when present.

### Skinning / Skeletal Animation

`THREE.SkinnedMesh` objects are automatically detected and skinned on the CPU. The renderer reads `skinIndex` and `skinWeight` attributes, computes bone matrices from `skeleton.bones` and `skeleton.boneInverses`, and transforms vertex positions and normals before sending them to the GPU. Malformed mesh/skeleton containers and invalid bone, inverse bind, and mesh bind matrix values fail clearly.

Compatible with:

- **Three.js** `SkinnedMesh` + `Skeleton` + `AnimationMixer`
- **@pixiv/three-vrm** — VRM humanoid avatars
- **VRMA** — VRM Animation files via `VRMAnimationLoaderPlugin` + `createVRMAnimationClip`

The repository includes runnable local examples for [glTF/GLB](https://github.com/portwatcher/headless-three-renderer/blob/main/examples/render-gltf.mjs) and [VRM/VRMA](https://github.com/portwatcher/headless-three-renderer/blob/main/examples/render-vrm.mjs) assets.

Use `applyVrmAnimation()` or your own `AnimationMixer`, then call `scene.updateMatrixWorld(true)` before `render()` to bake the current pose. `applyVrmAnimation()` accepts the glTF objects returned by the local VRM/VRMA helpers or direct `vrm`/`vrmAnimation` objects; it updates the avatar by default, and accepts `updateVrm: false` if your render pipeline performs that update separately.

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

// Animate to a specific time
await applyVrmAnimation(modelGltf, animGltf, {
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

Morph targets are applied on the CPU before rendering. Both **relative** (glTF default) and **absolute** (legacy Three.js) modes are supported. Position and normal morphs are applied based on `mesh.morphTargetInfluences`, with malformed morph attribute containers, invalid influence values, and malformed `geometry.morphTargetsRelative` values failing clearly. This is compatible with:

- glTF morph targets via `GLTFLoader`
- VRM blend shapes / expressions from `@pixiv/three-vrm`
- Blender shape keys exported to glTF

### Shadows

Directional, spot, point, and up to four directional cascaded shadow maps are supported across a packed twelve-layer native depth texture array. Directional and spot shadows use one layer, point shadows use six cube-face layers, and directional cascades use one layer per cascade; shadow-light sets that exceed that budget fail clearly. Reusable `Renderer.shadowMap.enabled` defaults to true for the current scene-oriented behavior; set it to false to suppress renderer-owned shadow maps. `shadowMap.autoUpdate` and `shadowMap.needsUpdate` are accepted as compatibility state; `shadowMap.type` selects a single-compare `THREE.BasicShadowMap` path or the current 3×3 PCF path for `THREE.PCFShadowMap`, `THREE.PCFSoftShadowMap`, and `THREE.VSMShadowMap`. Set `light.castShadow = true`, configure `light.shadow.camera`, and mark meshes with `mesh.castShadow = true` / `mesh.receiveShadow = true`. Common shadow options including `light.shadow.bias`, `light.shadow.normalBias`, `light.shadow.radius`, and PCF-irrelevant `light.shadow.blurSamples` inputs are accepted, explicit mesh `material.shadowSide` values filter shadow-caster faces, and `material.alphaToCoverage` approximates shadow-caster alpha cutouts with a 0.5 cutoff. `Object3D.customDepthMaterial` is honored for directional/spot mesh shadow caster alpha-tested inputs, including selected base-map and alpha-map UV channels, base-map and alpha-map texture transforms, displacement inputs, and visibility flags, and for sprite/point billboard shadow caster alpha-tested inputs; `customDistanceMaterial` is honored for point-light mesh shadow caster alpha-tested inputs, including selected base-map and alpha-map UV channels, base-map and alpha-map texture transforms, displacement inputs, and visibility flags, and for sprite/point billboard shadow caster alpha-tested inputs, with sprite custom-depth/custom-distance base/alpha maps honoring texture transforms and point-billboard custom-distance base/alpha maps honoring selected geometry UV channels plus texture transforms. Source-material base/alpha maps including selected base-map and alpha-map UV channels, base-map and alpha-map texture transforms, `alphaHash` and `alphaToCoverage` opacity cutouts, and `shadowSide` are carried onto custom mesh shadow casters, and source-material base/alpha texture transforms are also carried onto the covered sprite custom-depth/custom-distance and point-light point-billboard custom-distance paths. `THREE.Sprite` and `THREE.Points` can cast directional/spot/point shadows from expanded billboard quads, and `receiveShadow` is accepted as a WebGL-compatible no-op on their unlit material paths. Malformed shadow flags/containers, invalid shadow numeric values, invalid renderer shadow-map booleans/type constants, invalid custom shadow material visibility values, malformed light userData/cascade hint containers/values, over-budget cascade/layer sets, and non-square point-light `light.shadow.mapSize` values fail clearly until true shadow atlas allocation, deeper cascade support, and rectangular cube-face support land. The renderer renders depth-only passes and samples them with a normal-offset bias.

`shadowMap.transmitted` is stored as compatibility state; it currently does not change scene output because transmitted shadow-map passes are not modeled separately.

Directional cascades can be provided with `light.userData.headlessThreeRenderer.shadowCascades`, where each cascade has finite `{ left, right, top, bottom, near, far, split }` bounds.

`light.userData` must be an object when present before cascade hints are read.

### Tone Mapping

Output uses the Narkowicz ACES Filmic tone mapping fit by default with a three.js-compatible `toneMappingExposure / 0.6` exposure pre-scale, matching `THREE.ACESFilmicToneMapping`. Reusable `Renderer.toneMapping` and `currentToneMapping` support `THREE.NoToneMapping`, `THREE.LinearToneMapping`, `THREE.ReinhardToneMapping`, `THREE.CineonToneMapping`, `THREE.ACESFilmicToneMapping`, `THREE.CustomToneMapping`, `THREE.AgXToneMapping`, and `THREE.NeutralToneMapping`; `THREE.CustomToneMapping` uses Three.js' default identity custom function because GLSL shader-chunk customizations are not translated; `Renderer.toneMappingExposure` defaults to `1`; `Renderer.needsFrameBufferTarget` reports `false` because output conversion is inline in the native render pass; unsupported tone-mapping constants and invalid exposure values fail clearly.

### Render Targets & Post-Processing

`renderToTarget(scene, camera, target, options)`, `options.target`, and `Renderer.setRenderTarget(target); renderer.render(scene, camera, options)` populate a target-like object with `{ width, height, data }` plus `target.texture.image.data` when a texture object is present. `Renderer.readRenderTargetPixels()` and `readRenderTargetPixelsAsync()` copy stored CPU color data for regular targets, explicit cube target faces, and selected color attachment indices into caller-provided buffers; async readback can allocate a matching output buffer and accepts the common-renderer `(target, x, y, width, height, textureIndex, faceIndex)` argument shape. Top-level target rendering defaults to raw RGBA8 and `target.data` remains RGBA8 for compatibility; color textures receive Alpha/Red/RG/RGB/RGBA and RedIntegerFormat/RGIntegerFormat/RGBIntegerFormat/RGBAIntegerFormat channel data by requested format, normalized `Float32Array` data for `THREE.FloatType`, signed or unsigned normalized integer arrays for `ByteType`/`ShortType`/`IntType` and `UnsignedShortType`/`UnsignedIntType`, packed `Uint16Array` data for `UnsignedShort4444Type`/`UnsignedShort5551Type`, packed `Uint32Array` RGB9_E5 data for `UnsignedInt5999Type`, packed `Uint32Array` R11F_G11F_B10F data for `UnsignedInt101111Type`, or `Uint16Array` half-float data for `HalfFloatType`, including through target texture arrays, `target.textures`, MRT-shaped targets, and `options.target`. Regular-camera, ArrayCamera, and CubeCamera MRT-shaped targets can populate secondary color textures with explicit multi-pass auxiliary outputs when each secondary texture declares `texture.userData.headlessThreeRenderer.renderMode` as `'color'`, `'mask'`, `'object-id'`, or `'normal'`; `Renderer.getMRT()` returns `null`, `Renderer.setMRT(null)` is accepted as a clear operation, and arbitrary native MRT shader outputs including non-null `Renderer.setMRT()` remain unsupported. A target `depthTexture` object receives normalized depth readback for the same viewport/scissor and visible depth-tested geometry, including base-texture and alpha-map alpha-tested cutouts, `alphaHash` cutouts, and transparent material default/explicit `depthWrite` behavior; `THREE.FloatType` depth textures receive scalar `Float32Array` data, `HalfFloatType` depth textures receive `Uint16Array` half-float data, `UnsignedByteType`/`UnsignedShortType`/`UnsignedIntType` depth textures receive scalar unsigned typed arrays, `UnsignedInt248Type` receives `Uint32Array` data with normalized depth24 in the high bits and zero stencil bytes, and plain depth target objects receive RGBA8 bytes. 4x MSAA sample counts resolve into target readback buffers; unhinted secondary color attachments, malformed targets, target image containers, nested target texture/mipmap/source containers, malformed target scissor flags, array/3D target texture objects, regular-camera cube target texture objects, unsupported sample counts, unsupported color target texture formats/types, compressed target texture objects or compressed target format constants, explicit depth texture types, and depth texture format/type pairings fail clearly.

Built-in post-processing can be enabled with `options.postProcessing`. Supported effects are exposure, contrast, saturation, vignette, grayscale, and invert; malformed containers and invalid effect values fail clearly.

`options.renderMode` can request flat auxiliary passes. `'mask'` clears to black and writes white for visible geometry. `'object-id'` clears to RGB zero and encodes each object's adapter sort ID plus one into RGB bytes, making `format: 'rgba'` the preferred inspection path. `'normal'` clears to black and writes view-space normal colors matching `MeshNormalMaterial` for visible geometry. Target-based object-id renders populate `target.objectIdEntries` and `target.objectIdMap` for reverse lookup from encoded RGB IDs. These modes bypass scene backgrounds, lighting, environment, fog, and post-processing while preserving depth testing, culling, clipping planes, base texture alpha, `material.alphaMap`, `alphaTest`, and `alphaHash`; invalid render modes fail clearly.

### Custom WGSL Fragment Materials

Materials can provide a WGSL fragment body with `material.userData.headlessThreeRenderer.fragmentWgsl`. The body runs inside the renderer's standard vertex, uniform, color, UV, and base-texture setup and returns a `vec4<f32>`.

Three.js `ShaderMaterial`, `RawShaderMaterial`, and NodeMaterial are not translated directly; provide the headless WGSL fragment override above or use a built-in material.

`material.userData` must be an object when present; `material.userData.headlessThreeRenderer` and the legacy `material.userData.headlessRenderer` key must also be objects when present.

### Lines and Points

`THREE.Line`, `LineSegments`, `LineLoop`, and `THREE.Points` are supported. Lines and points render as unlit (basic) primitives, ignore lighting / normals, and honor bounding-sphere frustum culling with `frustumCulled=false` opt-out; point culling also accounts for rendered billboard size. Opacity, line material arrays with geometry groups, scene fog, `material.fog = false`, `alphaHash` opacity cutouts, and 4x-MSAA `alphaToCoverage` opacity cutouts are honored. `LineBasicMaterial.map` samples line UVs, including texture UV transforms, horizontal/vertical repeat/mirrored wrapping, channel 0-3 `texture.channel` UV selection through one non-primary UV stream, texture RGB with sRGB color-space decode, and alpha-tested texture alpha; line alpha maps honor selected `texture.channel` UVs and horizontal/vertical repeat/mirrored wrapping. Line objects cast directional/spot/point shadows with base-map, alpha-map, `alphaHash`, and `alphaToCoverage` cutouts. `LineBasicMaterial.linewidth` and `LineDashedMaterial.linewidth` values greater than 1 expand to camera-facing quads, while `linecap`, `linejoin`, and unlit `receiveShadow` are accepted as WebGL-compatible no-ops. Dashed lines honor custom `lineDistance` attributes including descending spans, keep `scale=0` or missing `lineDistance` attributes solid like WebGL, and dashed line maps/alpha maps preserve texture UV transforms, horizontal/vertical repeat/mirrored wrapping, selected `texture.channel` UVs, selected instanced map UV attributes, and sRGB color-space decode while reconstructing dash segments. `PointsMaterial` maps and alpha maps use per-point geometry UVs when present and point-sprite UVs otherwise, keep untextured point-sprite corners square/visible, honor texture UV transforms and horizontal/vertical repeat/mirrored wrapping, decode sRGB color maps, apply `alphaHash` and 4x-MSAA `alphaToCoverage` opacity cutouts, shrink with perspective distance by default, keep orthographic point size independent of camera depth, cast directional/spot/point shadows from the expanded billboard quads, accept `receiveShadow` as a WebGL-compatible no-op, and honor alpha-tested custom depth/distance material cutouts in shadow passes, including selected geometry UV channels for point-light custom-distance base/alpha maps. Invalid point size, point size attenuation, line width, line cap/join strings, and negative or non-finite dashed-line scalar values fail clearly.
