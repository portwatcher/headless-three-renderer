import * as THREE from 'three'
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js'
import { createSceneCorpus } from '../corpus.mjs'
import {
  BROWSER_REFERENCE_MANIFEST_FILE,
  createBrowserReferenceFixtures,
  createBrowserReferenceManifest,
  normalizeBrowserReferenceOutputColorSpace,
} from './manifest.mjs'

const fixturesEl = document.getElementById('fixtures')
const statusEl = document.getElementById('status')
const downloadAllButton = document.getElementById('download-all')
const downloadManifestLink = document.getElementById('download-manifest')

const renderer = new THREE.WebGLRenderer({
  alpha: true,
  antialias: true,
  preserveDrawingBuffer: true,
  stencil: true,
})
renderer.setPixelRatio(1)
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.PCFShadowMap
RectAreaLightUniformsLib.init()

const downloadLinks = []

try {
  const fixtures = createBrowserReferenceFixtures(createSceneCorpus())
  const manifest = createBrowserReferenceManifest(fixtures)
  setupManifestDownload(manifest)

  for (const fixture of fixtures) {
    const reference = renderFixture(fixture)
    fixturesEl.appendChild(reference.element)
    downloadLinks.push(reference.link)
    await nextFrame()
  }

  statusEl.textContent = `Rendered ${fixtures.length} browser reference PNGs. Save them and ${BROWSER_REFERENCE_MANIFEST_FILE} in one reference directory.`
  downloadAllButton.disabled = false
} catch (error) {
  statusEl.textContent = error instanceof Error ? error.message : String(error)
  console.error(error)
}

downloadAllButton.addEventListener('click', () => {
  downloadLinks.forEach((link, index) => {
    setTimeout(() => link.click(), index * 150)
  })
})

function setupManifestDownload(manifest) {
  const json = `${JSON.stringify(manifest, null, 2)}\n`
  const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }))
  downloadManifestLink.href = url
  downloadManifestLink.download = BROWSER_REFERENCE_MANIFEST_FILE
  downloadManifestLink.removeAttribute('aria-disabled')
  downloadLinks.push(downloadManifestLink)
}

function renderFixture(fixture) {
  const width = fixture.options.width
  const height = fixture.options.height

  if (fixture.camera?.isPerspectiveCamera === true && fixture.camera.aspect !== width / height) {
    fixture.camera.aspect = width / height
    fixture.camera.updateProjectionMatrix()
  }

  renderer.setSize(width, height, false)
  renderer.outputColorSpace = outputColorSpace(fixture.options.outputColorSpace)

  let restoreRendererOptions = () => {}
  let restoreSceneOptions = () => {}
  let restoreRenderMode = () => {}
  let dataUrl
  try {
    restoreRendererOptions = applyFixtureRendererOptions(fixture)
    restoreSceneOptions = applyFixtureSceneOptions(fixture)
    restoreRenderMode = applyFixtureRenderMode(fixture)
    clearFullCanvas(fixture.scene, width, height)
    applyFixtureRenderRectangles(fixture, width, height)
    fixture.scene.updateMatrixWorld(true)
    fixture.camera.updateMatrixWorld(true)
    renderer.render(fixture.scene, fixture.camera)
    dataUrl = renderer.domElement.toDataURL('image/png')
  } finally {
    restoreRenderMode()
    restoreSceneOptions()
    restoreRendererOptions()
  }

  const image = new Image(width, height)
  image.alt = fixture.name
  image.src = dataUrl

  const link = document.createElement('a')
  link.href = dataUrl
  link.download = `${fixture.name}.png`
  link.textContent = 'Download PNG'

  const element = document.createElement('article')
  element.className = 'fixture'
  const title = document.createElement('strong')
  title.textContent = fixture.name
  element.append(image, title, link)

  return { element, link }
}

function clearFullCanvas(scene, width, height) {
  const previousClearColor = renderer.getClearColor(new THREE.Color())
  const previousClearAlpha = renderer.getClearAlpha()
  const background = scene?.background
  const colorBackground = background?.isColor === true ? background : null
  const backgroundIntensity = typeof scene?.backgroundIntensity === 'number' && Number.isFinite(scene.backgroundIntensity)
    ? scene.backgroundIntensity
    : 1

  renderer.setScissorTest(false)
  renderer.setViewport(0, 0, width, height)
  renderer.setClearColor(
    colorBackground ? background.clone().multiplyScalar(backgroundIntensity) : new THREE.Color(0, 0, 0),
    colorBackground ? 1 : 0,
  )
  renderer.clear(true, true, true)
  renderer.setClearColor(previousClearColor, previousClearAlpha)
}

function applyFixtureRenderRectangles(fixture, width, height) {
  const viewport = rectangleFromOption(fixture.options.viewport) ?? { x: 0, y: 0, width, height }
  renderer.setViewport(
    viewport.x,
    height - viewport.y - viewport.height,
    viewport.width,
    viewport.height,
  )

  const scissor = rectangleFromOption(fixture.options.scissor)
  if (scissor) {
    renderer.setScissor(
      scissor.x,
      height - scissor.y - scissor.height,
      scissor.width,
      scissor.height,
    )
    renderer.setScissorTest(true)
  } else {
    renderer.setScissorTest(false)
  }
}

function rectangleFromOption(rectangle) {
  if (!rectangle) return null
  if (Array.isArray(rectangle)) {
    return {
      x: rectangle[0],
      y: rectangle[1],
      width: rectangle[2],
      height: rectangle[3],
    }
  }
  return {
    x: rectangle.x,
    y: rectangle.y,
    width: rectangle.width ?? rectangle.z,
    height: rectangle.height ?? rectangle.w,
  }
}

function applyFixtureRendererOptions(fixture) {
  const previousSortObjects = renderer.sortObjects
  const previousClippingPlanes = renderer.clippingPlanes
  const previousLocalClippingEnabled = renderer.localClippingEnabled

  renderer.sortObjects = fixture.options.sortObjects ?? true
  renderer.setOpaqueSort(fixture.options.opaqueSort ?? null)
  renderer.setTransparentSort(fixture.options.transparentSort ?? null)
  renderer.clippingPlanes = fixture.options.clippingPlanes ?? []
  renderer.localClippingEnabled = fixture.options.localClippingEnabled ?? false

  return () => {
    renderer.sortObjects = previousSortObjects
    renderer.setOpaqueSort(null)
    renderer.setTransparentSort(null)
    renderer.clippingPlanes = previousClippingPlanes
    renderer.localClippingEnabled = previousLocalClippingEnabled
  }
}

function applyFixtureSceneOptions(fixture) {
  const optionKeys = [
    'background',
    'backgroundIntensity',
    'backgroundBlurriness',
    'backgroundRotation',
    'environmentIntensity',
    'environmentRotation',
  ]
  if (!optionKeys.some((key) => Object.prototype.hasOwnProperty.call(fixture.options, key))) {
    return () => {}
  }
  if (fixture.scene?.isScene !== true) {
    throw new Error('Browser reference scene-level options require a THREE.Scene fixture.')
  }

  const previous = {
    background: fixture.scene.background,
    backgroundIntensity: fixture.scene.backgroundIntensity,
    backgroundBlurriness: fixture.scene.backgroundBlurriness,
    backgroundRotation: fixture.scene.backgroundRotation,
    environmentIntensity: fixture.scene.environmentIntensity,
    environmentRotation: fixture.scene.environmentRotation,
  }
  if (Object.prototype.hasOwnProperty.call(fixture.options, 'background')) {
    fixture.scene.background = fixture.options.background
  }
  if (Object.prototype.hasOwnProperty.call(fixture.options, 'backgroundIntensity')) {
    fixture.scene.backgroundIntensity = fixture.options.backgroundIntensity
  }
  if (Object.prototype.hasOwnProperty.call(fixture.options, 'backgroundBlurriness')) {
    fixture.scene.backgroundBlurriness = fixture.options.backgroundBlurriness
  }
  if (Object.prototype.hasOwnProperty.call(fixture.options, 'backgroundRotation')) {
    fixture.scene.backgroundRotation = fixture.options.backgroundRotation
  }
  if (Object.prototype.hasOwnProperty.call(fixture.options, 'environmentIntensity')) {
    fixture.scene.environmentIntensity = fixture.options.environmentIntensity
  }
  if (Object.prototype.hasOwnProperty.call(fixture.options, 'environmentRotation')) {
    fixture.scene.environmentRotation = fixture.options.environmentRotation
  }

  return () => {
    fixture.scene.background = previous.background
    fixture.scene.backgroundIntensity = previous.backgroundIntensity
    fixture.scene.backgroundBlurriness = previous.backgroundBlurriness
    fixture.scene.backgroundRotation = previous.backgroundRotation
    fixture.scene.environmentIntensity = previous.environmentIntensity
    fixture.scene.environmentRotation = previous.environmentRotation
  }
}

function applyFixtureRenderMode(fixture) {
  const mode = fixture.options.renderMode ?? 'color'
  if (mode === 'color') {
    return () => {}
  }
  if (mode !== 'normal' && mode !== 'mask') {
    throw new Error(`Browser reference generation only supports color, mask, and normal render modes; received ${mode}.`)
  }
  if (fixture.scene?.isScene !== true) {
    throw new Error('Browser reference render modes require a THREE.Scene fixture.')
  }

  const previousOverrideMaterial = fixture.scene.overrideMaterial
  const previousBackground = fixture.scene.background
  const overrideMaterial = mode === 'normal'
    ? new THREE.MeshNormalMaterial()
    : new THREE.MeshBasicMaterial({ color: 0xffffff })

  fixture.scene.overrideMaterial = overrideMaterial
  fixture.scene.background = new THREE.Color(0, 0, 0)

  return () => {
    fixture.scene.overrideMaterial = previousOverrideMaterial
    fixture.scene.background = previousBackground
    overrideMaterial.dispose()
  }
}

function outputColorSpace(value) {
  return normalizeBrowserReferenceOutputColorSpace(value)
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve))
}
