import test from 'node:test'
import assert from 'node:assert/strict'
import { access, readFile, readdir } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO_ROOT = fileURLToPath(new URL('../../../', import.meta.url))
const COMPATIBILITY_DOC = path.join(REPO_ROOT, 'docs', 'compatibility.md')
const GLTF_SAMPLE_ASSETS_DOC = path.join(REPO_ROOT, 'docs', 'gltf-sample-assets.md')
const NODE_LOADER_SETUP_DOC = path.join(REPO_ROOT, 'docs', 'node-loader-setup.md')
const RELEASE_CHECKLIST_DOC = path.join(REPO_ROOT, 'docs', 'release-checklist.md')
const CI_WORKFLOW = path.join(REPO_ROOT, '.github', 'workflows', 'CI.yml')
const PACKAGE_JSON = path.join(REPO_ROOT, 'packages', 'renderer', 'package.json')
const ANIMATED_PROFILE_SCRIPT = path.join(REPO_ROOT, 'packages', 'renderer', 'scripts', 'profile-animated-scene.mjs')
const API_INDEX = path.join(REPO_ROOT, 'packages', 'renderer', 'api', 'index.ts')
const GLTF_TEST = path.join(REPO_ROOT, 'packages', 'renderer', 'test', 'gltf.test.mjs')

test('public documentation links point at committed files', async () => {
  const markdownFiles = [
    path.join(REPO_ROOT, 'README.md'),
    path.join(REPO_ROOT, 'docs', 'compatibility.md'),
    path.join(REPO_ROOT, 'docs', 'gltf-sample-assets.md'),
    path.join(REPO_ROOT, 'docs', 'node-loader-setup.md'),
    path.join(REPO_ROOT, 'docs', 'release-checklist.md'),
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
  const gltfTestParts = (await readdir(path.dirname(GLTF_TEST)))
    .filter((name) => /^gltf\.test\.part-\d+\.mjs$/.test(name))
    .sort()
  const [compatibility, sampleAssetsDoc, gltfTest] = await Promise.all([
    readFile(COMPATIBILITY_DOC, 'utf8'),
    readFile(GLTF_SAMPLE_ASSETS_DOC, 'utf8'),
    Promise.all(gltfTestParts.map((name) => readFile(path.join(path.dirname(GLTF_TEST), name), 'utf8')))
      .then((parts) => parts.join('\n')),
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

test('release checklist gates compatibility and golden-reference updates', async () => {
  const checklist = await readFile(RELEASE_CHECKLIST_DOC, 'utf8')

  assert.match(
    checklist,
    /\[Three\.js compatibility matrix\]\(\.\/compatibility\.md\)/,
    'release checklist should require compatibility matrix updates',
  )
  assert.match(
    checklist,
    /release notes/i,
    'release checklist should tie compatibility changes to release notes',
  )
  assert.match(
    checklist,
    /packages\/renderer\/test\/browser-reference\/references\/<platform>-<arch>\//,
    'release checklist should name the committed platform reference directory',
  )
  assert.match(
    checklist,
    /HEADLESS_THREE_REQUIRE_BROWSER_REFERENCES=1/,
    'release checklist should document required golden-reference mode',
  )
  assert.match(
    checklist,
    /HEADLESS_THREE_BROWSER_REFERENCE_DIR=/,
    'release checklist should document explicit browser-reference inputs',
  )
  assert.match(
    checklist,
    /fixture-scoped golden tolerance/i,
    'release checklist should require tolerance changes to be reviewed',
  )
  assert.match(
    checklist,
    /generate:browser-reference/,
    'release checklist should document the headless browser-reference generator',
  )
  assert.match(
    checklist,
    /pnpm -C packages\/renderer run test:golden/,
    'release checklist should require the golden-reference harness',
  )
})

test('node loader setup docs name every exported loader helper', async () => {
  const [doc, apiIndex] = await Promise.all([
    readFile(NODE_LOADER_SETUP_DOC, 'utf8'),
    readFile(API_INDEX, 'utf8'),
  ])
  const loaderExportBlock = apiIndex.match(/export \{\r?\n([\s\S]*?)\r?\n\} from '\.\/loaders'/)
  assert.ok(loaderExportBlock, 'api/index.ts should re-export public loader helpers from ./loaders')
  const exportedLoaderNames = loaderExportBlock[1]
    .split('\n')
    .map((line) => line.replace(/[,\s]/g, ''))
    .filter(Boolean)

  for (const name of exportedLoaderNames) {
    const documentedNamePattern = new RegExp(`\`${escapeRegExp(name)}(?:\\([^\`]*\\))?\``)
    assert.match(
      doc,
      documentedNamePattern,
      `docs/node-loader-setup.md should name exported loader helper ${name}`,
    )
  }
})

test('compatibility matrix and CI stay synchronized with packaged platform targets', async () => {
  const [compatibility, ciWorkflow, packageJson] = await Promise.all([
    readFile(COMPATIBILITY_DOC, 'utf8'),
    readFile(CI_WORKFLOW, 'utf8'),
    readFile(PACKAGE_JSON, 'utf8'),
  ])
  const { napi } = JSON.parse(packageJson)
  const packageNamesByTarget = new Map([
    ['x86_64-apple-darwin', '@headless-three/renderer-darwin-x64'],
    ['aarch64-apple-darwin', '@headless-three/renderer-darwin-arm64'],
    ['x86_64-pc-windows-msvc', '@headless-three/renderer-win32-x64-msvc'],
    ['x86_64-unknown-linux-gnu', '@headless-three/renderer-linux-x64-gnu'],
    ['aarch64-unknown-linux-gnu', '@headless-three/renderer-linux-arm64-gnu'],
  ])

  assert.deepEqual(
    [...napi.targets].sort(),
    [...packageNamesByTarget.keys()].sort(),
    'docs test package target map must match packages/renderer/package.json napi.targets',
  )

  for (const target of napi.targets) {
    const packageName = packageNamesByTarget.get(target)
    assert.match(
      compatibility,
      new RegExp(escapeRegExp(packageName)),
      `compatibility matrix should list ${packageName}`,
    )
    assert.match(
      ciWorkflow,
      new RegExp(escapeRegExp(target)),
      `CI matrix should include ${target}`,
    )
  }
})

test('animated-scene profiler stays exposed as a package script', async () => {
  const packageJson = JSON.parse(await readFile(PACKAGE_JSON, 'utf8'))
  const profileScript = await readFile(ANIMATED_PROFILE_SCRIPT, 'utf8')
  assert.equal(
    packageJson.scripts['profile:animated'],
    'node scripts/profile-animated-scene.mjs',
    'package.json should expose the animated-scene profiler',
  )
  await assert.doesNotReject(
    () => access(ANIMATED_PROFILE_SCRIPT),
    'profile:animated should point at a committed script',
  )
  assert.match(
    profileScript,
    /--mode=NAME/,
    'profile:animated should document workload modes',
  )
  assert.match(
    profileScript,
    /--all-modes/,
    'profile:animated should document same-settings all-mode profile runs',
  )
  assert.match(
    profileScript,
    /mixed.*transform.*material.*static.*instanced/s,
    'profile:animated should keep mixed, transform, material, static, and instanced workload modes available',
  )
})

function escapeRegExp(value) {
  return value.replace(/[\\^$.*+?()[\]{}|]/g, '\\$&')
}
