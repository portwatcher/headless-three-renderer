import test from 'node:test'
import assert from 'node:assert/strict'
import { access, readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO_ROOT = fileURLToPath(new URL('../../../', import.meta.url))
const COMPATIBILITY_DOC = path.join(REPO_ROOT, 'docs', 'compatibility.md')
const GLTF_SAMPLE_ASSETS_DOC = path.join(REPO_ROOT, 'docs', 'gltf-sample-assets.md')
const GLTF_TEST = path.join(REPO_ROOT, 'packages', 'renderer', 'test', 'gltf.test.mjs')

test('public documentation links point at committed files', async () => {
  const markdownFiles = [
    path.join(REPO_ROOT, 'README.md'),
    path.join(REPO_ROOT, 'docs', 'compatibility.md'),
    path.join(REPO_ROOT, 'docs', 'gltf-sample-assets.md'),
    path.join(REPO_ROOT, 'packages', 'renderer', 'test', 'README.md'),
  ]

  for (const markdownFile of markdownFiles) {
    const text = await readFile(markdownFile, 'utf8')
    const links = [...text.matchAll(/\[[^\]]+\]\(([^)]+)\)/g)]
      .map((match) => match[1].split('#')[0])
      .filter((href) => href && !/^[a-z]+:/i.test(href) && !href.startsWith('/'))

    for (const href of links) {
      const target = path.resolve(path.dirname(markdownFile), href)
      await assert.doesNotReject(
        () => access(target),
        `${path.relative(REPO_ROOT, markdownFile)} links to missing file ${href}`,
      )
    }
  }
})

test('compatibility matrix links to synchronized Khronos glTF Sample Assets coverage', async () => {
  const [compatibility, sampleAssetsDoc, gltfTest] = await Promise.all([
    readFile(COMPATIBILITY_DOC, 'utf8'),
    readFile(GLTF_SAMPLE_ASSETS_DOC, 'utf8'),
    readFile(GLTF_TEST, 'utf8'),
  ])

  assert.match(
    compatibility,
    /\[Khronos glTF Sample Asset Coverage\]\(\.\/gltf-sample-assets\.md\)/,
    'compatibility matrix should link to the committed Khronos sample-asset list',
  )

  const fixtureNames = [...gltfTest.matchAll(/gltf-sample-assets', '([^']+)'/g)]
    .map((match) => match[1])
    .sort((a, b) => a.localeCompare(b))
  const uniqueFixtureNames = [...new Set(fixtureNames)]
  assert.ok(uniqueFixtureNames.length > 100, 'docs test should cover the committed Khronos fixture corpus')

  const documentedNames = [...sampleAssetsDoc.matchAll(/^- `([^`]+)`$/gm)]
    .map((match) => match[1])
  assert.deepEqual(
    documentedNames,
    uniqueFixtureNames,
    'docs/gltf-sample-assets.md must match the committed Khronos glTF Sample Assets fixture list',
  )
})
