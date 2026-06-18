import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
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

const referenceDir = resolveBrowserReferenceDir()
const referencesRequired = areBrowserReferencesRequired()

const DEFAULT_BROWSER_REFERENCE_MAX_MEAN_DIFF = 18
const BROWSER_REFERENCE_MAX_MEAN_DIFF_BY_FIXTURE = new Map([
  ['array-camera-viewport-split', 64],
  ['material-env-map-pbr', 64],
  ['avatar-like-skinned-toon', 45],
  ['alpha-to-coverage-msaa-plane', 44],
  ['transparent-layer-stack', 44],
  ['lod-groups-material-array', 44],
  ['physical-ibl-shadow', 41],
  ['light-probe-diffuse', 40],
  ['skinned-morphed-plane', 40],
  ['pathological-degenerate-geometry', 32],
  ['material-env-map-phong', 31],
  ['material-local-clipping-plane', 30],
  ['mesh-toon-gradient-map', 27],
  ['point-spot-light-materials', 27],
  ['sprite-material-map-billboard', 27],
  ['material-env-map-basic-lambert', 27],
  ['linear-fog-material-opt-out', 25],
  ['mesh-matcap-material-map', 24],
  ['light-probe-lit-material-models', 24],
  ['mesh-normal-material-flat', 20],
])

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
  assert.throws(
    () => normalizeBrowserReferenceOutputColorSpace('display-p3'),
    /Browser reference fixture outputColorSpace display-p3 is not supported.*SRGBColorSpace.*LinearSRGBColorSpace/i,
  )
})

test('browser reference directory resolution prefers explicit env over platform defaults', () => {
  const defaultDir = defaultBrowserReferenceDir()
  assert.equal(
    resolveBrowserReferenceDir(
      { HEADLESS_THREE_BROWSER_REFERENCE_DIR: '/tmp/browser-refs' },
      () => false,
    ),
    '/tmp/browser-refs',
  )
  assert.equal(resolveBrowserReferenceDir({}, (candidate) => candidate === defaultDir), defaultDir)
  assert.equal(resolveBrowserReferenceDir({}, () => false), undefined)
})

test('browser reference required mode parses explicit opt-in values', () => {
  assert.equal(areBrowserReferencesRequired({ HEADLESS_THREE_REQUIRE_BROWSER_REFERENCES: '1' }), true)
  assert.equal(areBrowserReferencesRequired({ HEADLESS_THREE_REQUIRE_BROWSER_REFERENCES: 'true' }), true)
  assert.equal(areBrowserReferencesRequired({ HEADLESS_THREE_REQUIRE_BROWSER_REFERENCES: 'yes' }), true)
  assert.equal(areBrowserReferencesRequired({ HEADLESS_THREE_REQUIRE_BROWSER_REFERENCES: '0' }), false)
  assert.equal(areBrowserReferencesRequired({ HEADLESS_THREE_REQUIRE_BROWSER_REFERENCES: 'false' }), false)
  assert.equal(areBrowserReferencesRequired({}), false)
})

test('browser reference tolerance policy scopes known parity gaps to fixture names', () => {
  const fixtures = createBrowserReferenceFixtures(createSceneCorpus())
  const fixtureNames = new Set(fixtures.map((fixture) => fixture.name))

  for (const name of BROWSER_REFERENCE_MAX_MEAN_DIFF_BY_FIXTURE.keys()) {
    assert.ok(fixtureNames.has(name), `browser reference tolerance entry ${name} must match a generated fixture`)
  }

  assert.equal(getBrowserReferenceMaxMeanDiff('mesh-depth-material-basic', {}), DEFAULT_BROWSER_REFERENCE_MAX_MEAN_DIFF)
  assert.equal(getBrowserReferenceMaxMeanDiff('array-camera-viewport-split', {}), 64)
  assert.equal(
    getBrowserReferenceMaxMeanDiff(
      'array-camera-viewport-split',
      { HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF: '5' },
    ),
    5,
  )
  assert.throws(
    () => getBrowserReferenceMaxMeanDiff(
      'mesh-depth-material-basic',
      { HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF: 'nan' },
    ),
    /HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF must be a non-negative number/,
  )
})

test('generated corpus matches browser WebGLRenderer golden references', {
  skip: referenceDir
    ? false
    : referencesRequired
      ? false
      : `set HEADLESS_THREE_BROWSER_REFERENCE_DIR or add browser-generated references at ${defaultBrowserReferenceDir()}`,
}, async (t) => {
  assert.ok(
    referenceDir,
    `Browser reference directory is required. Set HEADLESS_THREE_BROWSER_REFERENCE_DIR or add browser-generated references at ${defaultBrowserReferenceDir()}.`,
  )

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
      const maxMeanDiff = getBrowserReferenceMaxMeanDiff(fixture.name)

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

function resolveBrowserReferenceDir(
  env = process.env,
  exists = existsSync,
) {
  const explicitDir = env.HEADLESS_THREE_BROWSER_REFERENCE_DIR
  if (explicitDir) {
    return explicitDir
  }

  const defaultDir = defaultBrowserReferenceDir()
  return exists(defaultDir) ? defaultDir : undefined
}

function defaultBrowserReferenceDir() {
  return path.join(
    fileURLToPath(new URL('.', import.meta.url)),
    'browser-reference',
    'references',
    `${process.platform}-${process.arch}`,
  )
}

function areBrowserReferencesRequired(env = process.env) {
  const value = env.HEADLESS_THREE_REQUIRE_BROWSER_REFERENCES
  return value === '1' || value === 'true' || value === 'yes'
}

function getBrowserReferenceMaxMeanDiff(fixtureName, env = process.env) {
  const override = env.HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF
  if (override !== undefined) {
    const maxMeanDiff = Number(override)
    assert.ok(
      Number.isFinite(maxMeanDiff) && maxMeanDiff >= 0,
      'HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF must be a non-negative number',
    )
    return maxMeanDiff
  }

  return BROWSER_REFERENCE_MAX_MEAN_DIFF_BY_FIXTURE.get(fixtureName)
    ?? DEFAULT_BROWSER_REFERENCE_MAX_MEAN_DIFF
}
