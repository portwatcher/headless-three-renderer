import * as THREE from 'three'
import { createSceneCorpus } from '../corpus.mjs'

const fixturesEl = document.getElementById('fixtures')
const statusEl = document.getElementById('status')
const downloadAllButton = document.getElementById('download-all')

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
  for (const fixture of createSceneCorpus()) {
    const reference = renderFixture(fixture)
    fixturesEl.appendChild(reference.element)
    downloadLinks.push(reference.link)
    await nextFrame()
  }

  statusEl.textContent = `Rendered ${downloadLinks.length} browser reference PNGs. Save them as <fixture-name>.png in a reference directory.`
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

function renderFixture(fixture) {
  const width = fixture.options.width
  const height = fixture.options.height

  if (fixture.camera?.isPerspectiveCamera === true && fixture.camera.aspect !== width / height) {
    fixture.camera.aspect = width / height
    fixture.camera.updateProjectionMatrix()
  }

  renderer.setSize(width, height, false)
  renderer.setViewport(0, 0, width, height)
  renderer.setScissorTest(false)
  renderer.outputColorSpace = outputColorSpace(fixture.options.outputColorSpace)

  fixture.scene.updateMatrixWorld(true)
  fixture.camera.updateMatrixWorld(true)
  renderer.render(fixture.scene, fixture.camera)

  const dataUrl = renderer.domElement.toDataURL('image/png')
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

function outputColorSpace(value) {
  if (value === THREE.LinearSRGBColorSpace || value === 'srgb-linear') {
    return THREE.LinearSRGBColorSpace
  }
  return THREE.SRGBColorSpace
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve))
}
