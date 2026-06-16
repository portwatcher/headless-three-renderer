import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import * as THREE from 'three'
import native from '../native.js'
import pkg from '../dist/index.js'
import { createSceneCorpus } from './corpus.mjs'
import {
  BROWSER_REFERENCE_MANIFEST_FILE,
  createBrowserReferenceFixtures,
  createBrowserReferenceManifest,
  normalizeBrowserReferenceOutputColorSpace,
} from './browser-reference/manifest.mjs'

const { Renderer } = pkg

const referenceDir = process.env.HEADLESS_THREE_BROWSER_REFERENCE_DIR
const maxMeanDiff = Number(process.env.HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF ?? 18)

test('browser reference manifest normalizes outputColorSpace aliases', () => {
  const fixtures = [
    { name: 'constant', options: { width: 1, height: 1, outputColorSpace: THREE.LinearSRGBColorSpace } },
    { name: 'hyphen-alias', options: { width: 1, height: 1, outputColorSpace: 'linear-srgb' } },
    { name: 'compact-alias', options: { width: 1, height: 1, outputColorSpace: 'linearsrgb' } },
    { name: 'short-alias', options: { width: 1, height: 1, outputColorSpace: 'linear' } },
    { name: 'default-srgb', options: { width: 1, height: 1 } },
    { name: 'renderer-only', browserReference: false, options: { width: 1, height: 1, outputColorSpace: 'linear' } },
  ]
  const manifest = createBrowserReferenceManifest(fixtures)

  assert.deepEqual(
    manifest.fixtures.map((fixture) => [fixture.name, fixture.outputColorSpace]),
    [
      ['constant', THREE.LinearSRGBColorSpace],
      ['hyphen-alias', THREE.LinearSRGBColorSpace],
      ['compact-alias', THREE.LinearSRGBColorSpace],
      ['short-alias', THREE.LinearSRGBColorSpace],
      ['default-srgb', THREE.SRGBColorSpace],
    ],
  )
  assert.equal(normalizeBrowserReferenceOutputColorSpace('srgb'), THREE.SRGBColorSpace)
})

test('generated corpus matches browser WebGLRenderer golden references', {
  skip: referenceDir
    ? false
    : 'set HEADLESS_THREE_BROWSER_REFERENCE_DIR to a directory of browser-generated corpus PNGs',
}, async (t) => {
  assert.ok(Number.isFinite(maxMeanDiff) && maxMeanDiff >= 0, 'HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF must be a non-negative number')

  const fixtures = createBrowserReferenceFixtures(createSceneCorpus())
  const manifest = await readReferenceManifest(referenceDir)
  validateReferenceManifest(manifest, createBrowserReferenceManifest(fixtures))

  const renderer = new Renderer()
  for (const fixture of fixtures) {
    await t.test(fixture.name, async () => {
      const referencePath = path.join(referenceDir, `${fixture.name}.png`)
      const referencePng = await readFile(referencePath)
      const reference = native.decodeImage(referencePng)
      const width = fixture.options.width
      const height = fixture.options.height

      assert.equal(reference.width, width, `${fixture.name} reference width mismatch`)
      assert.equal(reference.height, height, `${fixture.name} reference height mismatch`)

      const actual = renderer.render(fixture.scene, fixture.camera, {
        ...fixture.options,
        format: 'rgba',
      })
      const metrics = diffRgba(actual, reference.data)

      assert.ok(
        metrics.mean <= maxMeanDiff,
        `${fixture.name} browser-reference mean RGBA diff ${metrics.mean.toFixed(3)} exceeded ${maxMeanDiff}; max channel diff ${metrics.max}`,
      )
    })
  }
})

async function readReferenceManifest(referenceDir) {
  const manifestPath = path.join(referenceDir, BROWSER_REFERENCE_MANIFEST_FILE)
  let raw
  try {
    raw = await readFile(manifestPath, 'utf8')
  } catch (error) {
    throw new Error(
      `Browser reference manifest is required at ${manifestPath}. Regenerate references with test/browser-reference/index.html and save ${BROWSER_REFERENCE_MANIFEST_FILE} with the PNG files.`,
      { cause: error },
    )
  }

  try {
    return JSON.parse(raw)
  } catch (error) {
    throw new Error(`Browser reference manifest at ${manifestPath} is not valid JSON.`, { cause: error })
  }
}

function validateReferenceManifest(actual, expected) {
  assert.equal(actual.schemaVersion, expected.schemaVersion, 'browser reference manifest schemaVersion mismatch')
  assert.equal(actual.generator, expected.generator, 'browser reference manifest generator mismatch')
  assert.equal(actual.renderer, expected.renderer, 'browser reference manifest renderer mismatch')
  assert.equal(
    actual.threeRevision,
    expected.threeRevision,
    'browser reference manifest Three.js revision mismatch; regenerate references with the current dependency version',
  )
  assert.deepEqual(
    actual.fixtures,
    expected.fixtures,
    'browser reference manifest fixture list mismatch; regenerate references from the current corpus',
  )
}

function diffRgba(actual, expected) {
  assert.equal(actual.length, expected.length, 'RGBA buffers must have matching lengths')

  let total = 0
  let max = 0
  for (let i = 0; i < actual.length; i += 1) {
    const diff = Math.abs(actual[i] - expected[i])
    total += diff
    max = Math.max(max, diff)
  }

  return {
    max,
    mean: total / actual.length,
  }
}
