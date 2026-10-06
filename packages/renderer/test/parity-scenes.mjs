// Scene helpers for the WebGL parity cases. They only use the injected THREE namespace, so the
// same code runs in Node (test/parity.test.mjs) and in Chrome (browser-reference/parity.html).

export const createParityHelpers = function (THREE) {
  const AVATAR_LIGHT_SCALE = Math.PI / 1.9
  const linearColor = (r, g = r, b = r) => new THREE.Color().setRGB(r, g, b, THREE.LinearSRGBColorSpace)

  const ortho = () => {
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
    camera.position.z = 2
    camera.updateMatrixWorld(true)
    return camera
  }
  const persp = (position = [0, 0, 3.2], target = [0, 0, 0], fov = 40, aspect = 1) => {
    const camera = new THREE.PerspectiveCamera(fov, aspect, 0.1, 50)
    camera.position.set(...position)
    camera.lookAt(...target)
    camera.updateMatrixWorld(true)
    return camera
  }
  const sceneWith = (...objects) => {
    const scene = new THREE.Scene()
    scene.background = linearColor(0.02, 0.025, 0.03)
    scene.add(...objects)
    return scene
  }
  // npcify AVATAR_LIGHTS: AmbientLight 0.7 and DirectionalLight 1.2 at (1, 1.5, 1), times PI / 1.9.
  const avatarLights = () => {
    const key = new THREE.DirectionalLight(0xffffff, 1.2 * AVATAR_LIGHT_SCALE)
    key.position.set(1, 1.5, 1)
    return [new THREE.AmbientLight(0xffffff, 0.7 * AVATAR_LIGHT_SCALE), key]
  }
  const sphere = (material, radius = 1, segments = 96) =>
    new THREE.Mesh(new THREE.SphereGeometry(radius, segments, segments / 2), material)
  const stripes = (materials) => {
    const group = new THREE.Group()
    const width = 2 / materials.length
    materials.forEach((material, index) => {
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, 2), material)
      mesh.position.x = -1 + width * (index + 0.5)
      group.add(mesh)
    })
    return group
  }
  const dataTexture = (width, height, texel, options = {}) => {
    const data = new Uint8Array(width * height * 4)
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) data.set(texel(x, y), (y * width + x) * 4)
    }
    const texture = new THREE.DataTexture(data, width, height)
    texture.colorSpace = options.colorSpace ?? THREE.NoColorSpace
    texture.magFilter = options.magFilter ?? THREE.LinearFilter
    texture.minFilter = options.minFilter ?? THREE.LinearFilter
    texture.wrapS = texture.wrapT = options.wrap ?? THREE.ClampToEdgeWrapping
    texture.generateMipmaps = options.generateMipmaps ?? false
    texture.flipY = options.flipY ?? false
    texture.needsUpdate = true
    return texture
  }
  // Red follows u, green follows v: shows texture orientation and transforms.
  const gradientTexture = (options) => dataTexture(16, 16, (x, y) => [Math.round(x * 17), Math.round(y * 17), 96, 255], {
    colorSpace: THREE.SRGBColorSpace,
    ...options,
  })
  const bumpyNormalMap = () => dataTexture(32, 32, (x, y) => {
    const nx = Math.sin((x / 32) * Math.PI * 4) * 0.5
    const ny = Math.cos((y / 32) * Math.PI * 4) * 0.5
    const nz = Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny))
    return [Math.round((nx * 0.5 + 0.5) * 255), Math.round((ny * 0.5 + 0.5) * 255), Math.round((nz * 0.5 + 0.5) * 255), 255]
  }, { wrap: THREE.RepeatWrapping })

  // Equirectangular environment: warm sky, cool ground, a bright sun. Rows start at the -Y pole
  // (flipY = false), so a renderer that ignores flipY shows the sky below the horizon.
  const environmentRadiance = (u, v, hdr) => {
    const y = Math.sin((v - 0.5) * Math.PI)
    const phi = (u - 0.5) * 2 * Math.PI
    let rgb
    if (y >= 0) {
      rgb = [0.25 + 0.55 * y + 0.1 * Math.cos(phi), 0.22 + 0.35 * y + 0.05 * Math.sin(phi), 0.18 + 0.65 * y - 0.05 * Math.cos(phi)]
    } else {
      rgb = [0.12 + 0.06 * Math.cos(phi), 0.1, 0.08 + 0.05 * Math.sin(phi * 2)]
    }
    const du = Math.min(Math.abs(u - 0.62), 1 - Math.abs(u - 0.62)) * 2
    const dv = v - 0.78
    const sun = Math.exp(-(du * du + dv * dv) / 0.002) * (hdr ? 12 : 0.6)
    return [rgb[0] + sun, rgb[1] + sun * 0.9, rgb[2] + sun * 0.75]
  }
  const environment = (kind = 'hdr', width = 256) => {
    const height = width / 2
    if (kind === 'hdr') {
      const data = new Uint16Array(width * height * 4)
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const rgb = environmentRadiance((x + 0.5) / width, (y + 0.5) / height, true)
          const i = (y * width + x) * 4
          data.set([...rgb, 1].map((value) => THREE.DataUtils.toHalfFloat(value)), i)
        }
      }
      const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.HalfFloatType)
      texture.mapping = THREE.EquirectangularReflectionMapping
      texture.colorSpace = THREE.LinearSRGBColorSpace
      texture.magFilter = THREE.LinearFilter
      texture.minFilter = THREE.LinearFilter
      texture.needsUpdate = true
      return texture
    }
    const encode = (value) => {
      const v = Math.min(Math.max(value, 0), 1)
      return Math.round(255 * (v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055))
    }
    const texture = dataTexture(width, height, (x, y) => [
      ...environmentRadiance((x + 0.5) / width, (y + 0.5) / height, false).map(kind === 'srgb' ? encode : (v) => Math.round(Math.min(v, 1) * 255)),
      255,
    ], { colorSpace: kind === 'srgb' ? THREE.SRGBColorSpace : THREE.NoColorSpace })
    texture.mapping = THREE.EquirectangularReflectionMapping
    return texture
  }

  // Six-face cube environment; each face has a horizontal gradient that shows the cube mirroring.
  const cubeEnvironment = (size) => {
    const faceColors = [[230, 70, 60], [60, 210, 80], [70, 90, 230], [230, 220, 70], [70, 220, 220], [220, 70, 220]]
    const faces = faceColors.map((color) => dataTexture(size, size, (x) => [
      ...color.map((channel) => Math.round(channel * (0.4 + (0.6 * x) / Math.max(size - 1, 1)))),
      255,
    ]))
    const texture = new THREE.CubeTexture(faces)
    texture.colorSpace = THREE.SRGBColorSpace
    texture.magFilter = THREE.LinearFilter
    texture.minFilter = THREE.LinearFilter
    texture.generateMipmaps = false
    texture.needsUpdate = true
    return texture
  }

  // Interior sample points of a centered sphere (radius about 28 px in a 64 x 64 frame).
  const spherePoints = [[32, 32], [22, 32], [42, 32], [32, 22], [32, 42], [24, 24], [40, 24], [24, 40], [40, 40], [15, 32], [49, 32], [32, 15], [32, 49]]
  const gridPoints = (size, cells) => {
    const points = []
    for (let y = 0; y < cells; y += 1) {
      for (let x = 0; x < cells; x += 1) points.push([Math.floor(((x + 0.5) * size) / cells), Math.floor(((y + 0.5) * size) / cells)])
    }
    return points
  }
  const rowPoints = (width, count, y) => Array.from({ length: count }, (_, i) => [Math.floor(((i + 0.5) * width) / count), y])

  return {
    linearColor, ortho, persp, sceneWith, avatarLights, sphere, stripes, dataTexture, gradientTexture,
    bumpyNormalMap, environment, cubeEnvironment, spherePoints, gridPoints, rowPoints,
  }
}
