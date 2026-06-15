import * as THREE from 'three'
import { createSceneCorpus } from '../corpus.mjs'
import { BROWSER_REFERENCE_MANIFEST_FILE, createBrowserReferenceManifest } from './manifest.mjs'

const fixturesEl = document.getElementById('fixtures')
const statusEl = document.getElementById('status')
const downloadAllButton = document.getElementById('download-all')
const downloadManifestLink = document.getElementById('download-manifest')

const renderer = new THREE.WebGLRenderer({
  alpha: true,
  antialias: true,
  preserveDrawingBuffer: true,
})
renderer.setPixelRatio(1)
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.PCFShadowMap

const downloadLinks = []

try {
  const fixtures = createSceneCorpus()
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
  applyFixtureRenderRectangles(fixture, width, height)
  renderer.outputColorSpace = outputColorSpace(fixture.options.outputColorSpace)

  const restoreRendererOptions = applyFixtureRendererOptions(fixture)
  let dataUrl
  try {
    fixture.scene.updateMatrixWorld(true)
    fixture.camera.updateMatrixWorld(true)
    renderer.render(fixture.scene, fixture.camera)
    dataUrl = renderer.domElement.toDataURL('image/png')
  } finally {
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

function outputColorSpace(value) {
  if (value === THREE.LinearSRGBColorSpace || value === 'srgb-linear') {
    return THREE.LinearSRGBColorSpace
  }
  return THREE.SRGBColorSpace
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve))
}
