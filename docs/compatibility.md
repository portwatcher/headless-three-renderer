# @headless-three/renderer Compatibility Matrix

This matrix is the public support contract for the current renderer. The package accepts normal Three.js scene objects and renders through the TypeScript adapter, N-API, and native `wgpu`; it does not provide a WebGL context or emulate `THREE.WebGLRenderer`.

Status keys:

- **Supported**: implemented and covered by smoke or conformance tests.
- **Partial**: implemented for common cases, with known gaps listed here.
- **Unsupported**: not implemented; scenes should be preprocessed or should expect clear documented gaps.

## Public API

| Feature | Status | Notes |
|---|---|---|
| `render(scene, camera, options)` | Supported | Returns PNG by default or raw RGBA when `options.format` is `'rgba'`. |
| `Renderer` reusable instance | Supported | Reuses the native renderer object across calls. |
| `renderToTarget()` / `options.target` | Supported | Populates target-like `{ width, height, data, texture.image.data }` fields with RGBA8 for a single color output. Unsupported target depth, MRT, and MSAA fields fail clearly. |
| Post-processing options | Supported | Exposure, contrast, saturation, vignette, grayscale, and invert. |
| WebGLRenderer-compatible state machine | Unsupported | Non-goal. The contract is scene input and image/target output. |

## Scene Graph And Geometry

| Feature | Status | Notes |
|---|---|---|
| `THREE.Scene` traversal | Supported | Honors `visible === false` on ancestors. |
| `THREE.Mesh` | Supported | Indexed and non-indexed `BufferGeometry`, groups, material arrays, transforms, normals, UVs, and vertex colors. |
| `THREE.SkinnedMesh` | Supported | CPU skinning from skeleton bones and inverse bind matrices. |
| `THREE.InstancedMesh` | Supported | CPU-expanded per instance with `instanceMatrix` and `instanceColor`; this favors compatibility over native GPU instancing performance. |
| Morph targets | Supported | CPU-baked relative and absolute position/normal morph targets. |
| `THREE.Line`, `LineSegments`, `LineLoop` | Partial | Rendered as unlit line lists. `LineBasicMaterial.map` samples line UVs, including alpha-tested texture alpha. `LineDashedMaterial` dash/gap patterns are CPU-expanded for common cases with reconstructed map UVs and interpolated vertex colors. Non-default `linewidth` fails clearly; thick-line support and broader line material options remain limited. |
| `THREE.Points` | Partial | `PointsMaterial` is CPU-expanded into camera-facing quads with size, perspective size attenuation, maps, alpha maps, vertex colors, layers, render ordering, main-pass clipping, and common instanced-geometry offset/color attributes. Hardware point-size caps, exact circular point-sprite edge cases, line instancing parity, and shadows remain planned. |
| Object/camera layers | Supported | Renderable objects and lights are filtered against the active camera layers. |
| `renderOrder` | Partial | Native sorting includes inherited `THREE.Group.renderOrder`/`groupOrder`, object `renderOrder`, material id, projected z, and stable object/insertion ties. Exact bounding-sphere z, materialVariant, and specialized bucket parity remain limited. |
| `InstancedBufferGeometry` | Partial | CPU-expanded for mesh and point geometry with instanced vertex colors and common instanced offset attributes (`instanceOffset`, `instancePosition`, `offset`, `translate`, `translation`). Arbitrary custom vertex attributes and line instancing still require additional strategy. |
| `THREE.Sprite` / billboards | Partial | CPU-expanded into camera-facing quads with `SpriteMaterial` color, opacity, map, alphaMap, center, scale, rotation, size attenuation, layers, render ordering, and main-pass clipping. Sprite shadows and exact WebGL edge cases remain planned. |
| LOD | Supported | `THREE.LOD` auto-update selects the active child from camera distance/zoom before traversal. |
| Clipping planes | Partial | Render-option global `clippingPlanes` and material-local `clippingPlanes`/`clipIntersection` are evaluated in the main pass for meshes, sprites, points, and lines, up to eight world-space planes. `material.clipShadows` fails clearly; `WebGLRenderer.localClippingEnabled`, shadow-pass clipping support, alpha-to-coverage smoothing, and exact golden parity remain planned. |
| Fog | Partial | `THREE.Fog` and `THREE.FogExp2` are applied in material shaders using camera-distance fog. Exact Three.js view-depth parity is not yet golden-tested. |

## Cameras And Output

| Feature | Status | Notes |
|---|---|---|
| Perspective cameras | Supported | Uses the camera projection and inverse world matrices. |
| Orthographic/custom projection cameras | Supported | Any camera with `projectionMatrix` and `matrixWorldInverse`. |
| Array/cube cameras | Unsupported | `THREE.ArrayCamera` and `THREE.CubeCamera` fail clearly. Planned. |
| Scene background color | Supported | `THREE.Color` scene background or `options.background`, including `scene.backgroundIntensity` and `options.backgroundIntensity`. |
| Background textures/cubemaps | Partial | 2D texture backgrounds from `scene.background` or `options.background` are rendered with wrap modes, UV transforms, background intensity, approximate `backgroundBlurriness`, and output color-space conversion. Cube/equirect background mappings fail clearly; rendering cube/equirect backgrounds and exact mapping/color-space/blur parity remain planned. |
| Output color space | Partial | `options.outputColorSpace` supports `THREE.SRGBColorSpace` (`"srgb"`, default) and `THREE.LinearSRGBColorSpace` (`"srgb-linear"`) for material and 2D texture background output conversion. Exact WebGLRenderer color-management state parity remains planned. |
| Viewport/scissor | Partial | `options.viewport` and `options.scissor` accept `[x, y, width, height]` or `{ x, y, width, height }` pixel rectangles in top-left output coordinates. Exact WebGLRenderer state-machine parity remains out of scope for the scene-oriented API. |
| Depth output / MRT | Unsupported | Target `depthTexture` and multiple color attachment requests fail clearly. Planned. |
| MSAA controls | Unsupported | Render-option and target sample counts greater than 1 fail clearly. Planned. |

## Materials And Textures

| Feature | Status | Notes |
|---|---|---|
| `MeshBasicMaterial` | Supported | Unlit path. |
| `MeshLambertMaterial` | Supported | Diffuse-only lighting path. |
| `MeshPhongMaterial` | Partial | Blinn-Phong direct lighting with `specular`, `shininess`, and `specularMap` strength is supported. Material-level `envMap` fails clearly; use scene-level environment/reflection probes for supported IBL. Combine modes and exact transform/color-space edge cases remain planned. |
| `MeshStandardMaterial` | Supported | PBR metallic/roughness path. |
| `MeshPhysicalMaterial` | Partial | Clearcoat, sheen, anisotropy, specular intensity/color, transmission, IOR, thickness, attenuation, and related maps are supported. Iridescence and dispersion fail clearly; transmission roughness behavior and several edge cases remain planned. |
| `MeshNormalMaterial` | Supported | View-space normal color output is supported, including `flatShading`, tangent-space normal maps, and bump maps. CPU-baked displacement affects triangle geometry. |
| `MeshMatcapMaterial` | Supported | Matcap texture sampling from view-space normals is supported, including `flatShading`, separate color maps with `texture.channel` UV selection and transforms, tangent-space normal maps, and bump maps. CPU-baked displacement affects triangle geometry. |
| `MeshDepthMaterial` | Partial | Basic, RGBA, RGB, and RG depth packing are supported, including wireframe output, base/alpha map alpha testing, and CPU-baked displacement through the common material path. |
| `MeshDistanceMaterial` | Partial | Red-channel distance output is supported, including wireframe output, optional `referencePosition`, `nearDistance`, and `farDistance` overrides plus base/alpha map alpha testing and CPU-baked displacement through the common material path. Point-shadow automation and packed distance variants remain planned. |
| `MeshToonMaterial` | Partial | Lit toon shading supports Three.js' no-gradient-map fallback bands, red-channel `gradientMap` ramps, and the common map, normal, bump, emissive, AO, light-map, alpha, displacement, side, fog, and shadow paths. Exact edge-case parity remains planned. |
| `ShadowMaterial` | Partial | Transparent shadow receiver output is supported for the renderer's current shadow map path, including scene fog and `material.fog` opt-out. Exact color-space behavior, `clipShadows`, and multiple shadow-casting lights remain planned. |
| Base color and opacity | Supported | Includes transparent sorting path. |
| Vertex colors | Supported | Multiplied with material color. |
| `material.side` | Supported | Front, back, and double-sided culling. |
| Alpha test | Supported | Fragment discard cutoff. |
| Alpha hash | Supported | `material.alphaHash` applies deterministic stochastic discard before depth/color writes. `material.alphaToCoverage` fails clearly; alpha-to-coverage smoothing remains planned. |
| Blending and depth/color state | Partial | Built-in `NoBlending`, `NormalBlending`, `AdditiveBlending`, `SubtractiveBlending`, and `MultiplyBlending` modes, `CustomBlending` equations/factors, plus `material.depthTest`, `material.depthWrite`, `material.colorWrite`, `material.polygonOffset`, standard material `premultipliedAlpha`, and stencil state are honored in the main pass. `material.alphaToCoverage` fails clearly; alpha-to-coverage support, constant blend edge-case parity, custom WGSL premultiplied-output wrapping, and exact transparent depth-write parity remain planned. |
| Base color map | Supported | Raw RGBA8 `DataTexture` and encoded PNG/JPEG/WebP buffers. `texture.channel` selects `uv`, `uv1`, `uv2`, or `uv3` when present; base color maps honor `flipY === false` and decode `texture.colorSpace === THREE.SRGBColorSpace` before shading. |
| Texture wrap modes | Supported | Repeat, mirrored repeat, and clamp-to-edge for base color maps. |
| Normal and bump maps | Supported | Tangent-space normal maps include `normalScale`; bump maps include `bumpScale`. `texture.channel` UV selection, texture transforms, wrap modes, and `NearestFilter`/`LinearFilter` sampler settings are honored. |
| Displacement map | Partial | Triangle meshes with UVs and normals are CPU-displaced along object-space normals using the red channel, `displacementScale`, `displacementBias`, `texture.channel` UV selection, and texture transforms. Exact WebGL vertex-shader edge cases, line/point displacement, and shadow-reference nuances remain planned. |
| Metallic/roughness maps | Supported | glTF convention: roughness in G, metalness in B. `texture.channel` selects `uv`, `uv1`, `uv2`, or `uv3` when present, and texture transforms plus wrap/filter sampler settings are honored. |
| Emissive map | Supported | Combined with emissive color/intensity. `texture.channel` selects `uv`, `uv1`, `uv2`, or `uv3` when present, and texture transforms, sRGB color-space decode, plus wrap/filter sampler settings are honored. |
| AO map | Supported | Red channel is applied to indirect lighting. `texture.channel` selects `uv`, `uv1`, `uv2`, or `uv3` when present, and texture transforms plus wrap/filter sampler settings are honored. |
| Light map | Partial | RGB channels add baked indirect diffuse lighting with `lightMapIntensity`; `texture.channel` selects `uv`, `uv1`, `uv2`, or `uv3` when present, and texture transforms, sRGB color-space decode, plus wrap/filter sampler settings are honored. Broader light-map parity remains planned. |
| Specular map | Partial | Red channel modulates `MeshPhongMaterial` specular strength; `texture.channel` selects `uv`, `uv1`, `uv2`, or `uv3` when present, and texture transforms plus wrap/filter sampler settings are honored. Non-Phong uses remain planned. |
| Alpha map | Supported | Green channel multiplies material opacity and participates in alpha testing/blending. `texture.channel` selects `uv`, `uv1`, `uv2`, or `uv3` when present, and `offset`, `repeat`, `rotation`, `center`, explicit `matrix`, `flipY`, plus wrap/filter sampler settings are honored. |
| Toon gradient map | Supported | `MeshToonMaterial.gradientMap` samples the red channel as the direct diffuse ramp at `dot(N, L) * 0.5 + 0.5`. |
| Physical extension maps | Partial | Clearcoat, sheen, anisotropy, specular color/intensity, transmission, and thickness map subsets are supported; all current physical-extension maps honor `texture.channel` UV selection, packed texture-group sampler settings, and sheen/specular color maps decode `THREE.SRGBColorSpace`. Conflicting sampler settings between channels packed into the same physical texture remain planned. |
| Custom WGSL fragment body | Partial | Supported through `material.userData.headlessThreeRenderer.fragmentWgsl`, including as an explicit opt-in path for `ShaderMaterial`/`RawShaderMaterial` inputs and materials with `onBeforeCompile` customizations. This is not Three.js shader parity. |
| Texture transforms | Partial | Base color, matcap color, normal, bump, displacement, metallic/roughness, emissive, AO, light, specular, alpha, clearcoat, clearcoat roughness, clearcoat normal, sheen color/roughness, anisotropy, specular color/intensity, transmission, and thickness maps honor `offset`, `repeat`, `rotation`, `center`, `flipY`, and explicit `matrix` when `matrixAutoUpdate === false`. Exact matrix/color-space interactions still need golden parity. |
| Texture color-space parity | Partial | Base color, matcap, emissive, light, sheen color, and physical specular color maps decode `THREE.SRGBColorSpace`; base color maps honor `flipY`; material and 2D texture background output conversion supports `THREE.SRGBColorSpace`/`THREE.LinearSRGBColorSpace`; and standard material shaders honor `premultipliedAlpha`. Remaining slot parity and exact WebGLRenderer color-management state parity remain planned. |
| Texture sampler settings | Partial | Base color/matcap, background, normal/bump, metallic/roughness, emissive, AO/light, alpha, Phong specular, and packed physical-extension texture groups honor wrap modes plus `NearestFilter`/`LinearFilter`-family `magFilter` and `minFilter`. Non-default texture anisotropy fails clearly; generated mip chains, native anisotropic filtering, and conflicting per-channel sampler settings inside the same packed physical texture remain planned. |
| Additional texture slots | Partial | Unsupported iridescence-related physical map slots fail clearly. Adding those slots remains planned. |
| Additional material classes | Partial | Remaining class work is mostly deeper parity for supported classes plus unsupported/custom material strategies documented above. Material-level `envMap`, `ShaderMaterial`, `RawShaderMaterial`, NodeMaterial, and built-in material `onBeforeCompile` customizations fail clearly unless a documented alternative path is used. |
| Compressed textures | Unsupported | KTX2/Basis/`THREE.CompressedTexture` inputs fail with a clear pre-decode error. Decode to RGBA `DataTexture` data or an encoded PNG/JPEG/WebP image before rendering. |

## Lighting, Shadows, And IBL

| Feature | Status | Notes |
|---|---|---|
| Ambient light | Supported | Summed scene ambient lights, camera-layer filtered. |
| Directional light | Supported | Position/target direction and intensity. |
| Point light | Supported | Distance and decay attenuation. |
| Spot light | Supported | Angle, penumbra, distance, and decay. |
| Hemisphere light | Supported | Sky/ground gradient ambient term. |
| `RectAreaLight` | Partial | One-sided finite-area direct-light approximation using the light center, local `-Z` direction, width, and height. Exact Three.js LTC/LUT behavior remains planned. |
| Light layers | Supported | Lights are filtered against the active camera layers. |
| Light limit | Partial | Up to 16 lights. |
| Shadows | Partial | Directional, spot, point, and directional cascaded shadow maps are supported for one shadow-casting light; additional visible shadow-casting lights fail clearly. |
| Shadow behavior parity | Partial | Supports common camera bounds, square map size, bias, normal bias, cast/receive flags, and alpha-tested casters. Non-square shadow map sizes fail clearly; native multiple-shadow-map rendering, rectangular maps, and blur/radius parity remain planned. |
| Equirectangular environment IBL | Supported | CPU precomputes diffuse irradiance, prefiltered specular, and BRDF LUT from equirectangular inputs. Cube, refraction, and PMREM/CubeUV environment mappings fail clearly. |
| Reflection probe hint | Supported | Uses `scene.userData.headlessThreeRenderer.reflectionProbe` or first probe in `reflectionProbes`. |
| Environment intensity | Supported | `scene.environmentIntensity`. |
| `LightProbe` | Partial | Spherical-harmonics coefficients from visible, camera-layer-matching `THREE.LightProbe` objects contribute diffuse indirect lighting. Exact golden-tested parity across every material/environment combination remains planned. |
| Environment/background rotation and intensity controls | Partial | `scene.environmentIntensity`, `scene.backgroundIntensity`, `options.backgroundIntensity`, and approximate 2D texture `backgroundBlurriness` are supported. Non-default `scene.backgroundRotation` and `scene.environmentRotation` fail clearly; native rotation, cube/equirect background blur, and newer mapping controls remain planned. |

## Test Coverage

| Area | Status | Notes |
|---|---|---|
| Smoke tests | Supported | Module load, simple renders, output dimensions, reusable renderer. |
| Conformance invariants | Supported | Scene-level render invariants for materials, PBR, IBL, shadows, post-processing, lines, points, layers, and render order. |
| Packed artifact verification | Supported | CI packs the package and current-platform native binary, installs them in a clean temp project, and renders a scene. |
| Node loader setup docs/helpers | Supported | Local `GLTFLoader` setup is documented for file paths, external buffers, encoded image files, and optional image polyfills. The package exports Node helpers for encoded image texture loading and local `file://` fetch bridging. |
| Golden-image parity corpus | Partial | The repo has scene invariants but not a committed browser-generated golden corpus yet. |
| Curated external asset corpus | Unsupported | Planned for glTF Sample Assets, VRM/VRMA, and pathological geometry fixtures. |

## Packaging

| Platform package | Status |
|---|---|
| `@headless-three/renderer-linux-x64-gnu` | Supported |
| `@headless-three/renderer-linux-arm64-gnu` | Supported |
| `@headless-three/renderer-darwin-x64` | Supported |
| `@headless-three/renderer-darwin-arm64` | Supported |
| `@headless-three/renderer-win32-x64-msvc` | Supported |

Published packages are verified by installing the packed root package and matching native package in a clean project, then rendering a real Three.js scene.
