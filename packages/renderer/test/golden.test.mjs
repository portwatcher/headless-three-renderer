import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import native from '../native.js'
import pkg from '../dist/index.js'
import { createSceneCorpus } from './corpus.mjs'

const { Renderer } = pkg

const referenceDir = process.env.HEADLESS_THREE_BROWSER_REFERENCE_DIR
const maxMeanDiff = Number(process.env.HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF ?? 18)

test('generated corpus matches browser WebGLRenderer golden references', {
  skip: referenceDir
    ? false
    : 'set HEADLESS_THREE_BROWSER_REFERENCE_DIR to a directory of browser-generated corpus PNGs',
}, async (t) => {
  assert.ok(Number.isFinite(maxMeanDiff) && maxMeanDiff >= 0, 'HEADLESS_THREE_REFERENCE_MAX_MEAN_DIFF must be a non-negative number')

  const renderer = new Renderer()
  for (const fixture of createSceneCorpus()) {
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
