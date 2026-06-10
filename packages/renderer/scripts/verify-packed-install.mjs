#!/usr/bin/env node

import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { cp, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const packageRoot = path.resolve(here, '..')
const repoRoot = path.resolve(packageRoot, '..', '..')
const tmp = mkdtempSync(path.join(tmpdir(), 'headless-three-pack-'))

try {
  const rootPackage = JSON.parse(await readFile(path.join(packageRoot, 'package.json'), 'utf8'))
  const platform = await currentPlatformPackage()
  const nativePackageRoot = path.join(packageRoot, 'npm', platform.packageDir)
  const nativeBinary = path.join(packageRoot, platform.binaryName)

  assert.ok(existsSync(path.join(packageRoot, 'dist', 'index.js')), 'dist/index.js is missing; run build:ts first')
  assert.ok(existsSync(nativeBinary), `${platform.binaryName} is missing; build or download the native artifact first`)
  assert.ok(existsSync(nativePackageRoot), `${nativePackageRoot} is missing`)

  const nativeStage = path.join(tmp, platform.packageDir)
  await cp(nativePackageRoot, nativeStage, { recursive: true })
  await cp(nativeBinary, path.join(nativeStage, platform.binaryName))
  await rewritePackageJson(path.join(nativeStage, 'package.json'), (pkg) => {
    pkg.version = rootPackage.version
    return pkg
  })
  const nativeTarball = npmPack(nativeStage, tmp)

  const rootStage = path.join(tmp, 'renderer')
  mkdirSync(rootStage)
  await cp(path.join(packageRoot, 'dist'), path.join(rootStage, 'dist'), { recursive: true })
  for (const file of ['native.js', 'native.d.ts', 'README.md', 'LICENSE']) {
    await cp(path.join(packageRoot, file), path.join(rootStage, file))
  }
  await cp(path.join(packageRoot, 'package.json'), path.join(rootStage, 'package.json'))
  await rewritePackageJson(path.join(rootStage, 'package.json'), (pkg) => {
    pkg.optionalDependencies = {
      [platform.packageName]: pathToFileURL(nativeTarball).href,
    }
    return pkg
  })
  const rootTarball = npmPack(rootStage, tmp)

  const project = path.join(tmp, 'project')
  mkdirSync(project)
  const threeSpec = existsSync(path.join(repoRoot, 'node_modules', 'three', 'package.json'))
    ? pathToFileURL(path.join(repoRoot, 'node_modules', 'three')).href
    : rootPackage.devDependencies?.three ?? rootPackage.peerDependencies?.three ?? 'latest'
  writeFileSync(path.join(project, 'package.json'), JSON.stringify({
    name: 'verify-packed-renderer',
    private: true,
    type: 'module',
    dependencies: {
      '@headless-three/renderer': pathToFileURL(rootTarball).href,
      three: threeSpec,
    },
  }, null, 2))

  run('npm', ['install', '--no-audit', '--fund=false'], project)

  const verifyScript = path.join(project, 'verify.mjs')
  writeFileSync(verifyScript, `
import assert from 'node:assert/strict'
import * as THREE from 'three'
import { render } from '@headless-three/renderer'

const scene = new THREE.Scene()
scene.background = new THREE.Color(0, 0, 0)
scene.add(new THREE.Mesh(
  new THREE.BoxGeometry(1, 1, 1),
  new THREE.MeshBasicMaterial({ color: 0x22ccff }),
))
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
camera.position.set(0, 0, 3)
camera.lookAt(0, 0, 0)
const rgba = render(scene, camera, { width: 32, height: 32, format: 'rgba' })
assert.equal(rgba.length, 32 * 32 * 4)
assert.ok(rgba.some((value) => value !== 0), 'render output should be non-empty')
`)
  run('node', [verifyScript], project)

  console.log(`Verified packed install for ${platform.packageName}`)
} finally {
  if (process.env.KEEP_PACK_VERIFY_TMP === '1') {
    console.log(`Keeping ${tmp}`)
  } else {
    rmSync(tmp, { recursive: true, force: true })
  }
}

async function currentPlatformPackage() {
  const platform = process.platform
  const arch = process.arch

  if (platform === 'darwin' && arch === 'x64') return platformPackage('darwin-x64')
  if (platform === 'darwin' && arch === 'arm64') return platformPackage('darwin-arm64')
  if (platform === 'win32' && arch === 'x64') return platformPackage('win32-x64-msvc')
  if (platform === 'linux' && arch === 'x64') return platformPackage(`linux-x64-${await linuxLibc()}`)
  if (platform === 'linux' && arch === 'arm64') return platformPackage(`linux-arm64-${await linuxLibc()}`)

  throw new Error(`No release artifact package is configured for ${platform}/${arch}`)
}

function platformPackage(packageDir) {
  return {
    packageDir,
    packageName: `@headless-three/renderer-${packageDir}`,
    binaryName: `headless_three_renderer.${packageDir}.node`,
  }
}

async function linuxLibc() {
  try {
    const ldd = await readFile('/usr/bin/ldd', 'utf8')
    if (ldd.includes('musl')) return 'musl'
  } catch {}

  return 'gnu'
}

async function rewritePackageJson(file, rewrite) {
  const pkg = rewrite(JSON.parse(await readFile(file, 'utf8')))
  await writeFile(file, `${JSON.stringify(pkg, null, 2)}\n`)
}

function npmPack(cwd, destination) {
  const before = new Set(safeReaddir(destination))
  run('npm', ['pack', '--pack-destination', destination], cwd)
  const after = safeReaddir(destination)
    .filter((file) => file.endsWith('.tgz') && !before.has(file))
    .map((file) => path.join(destination, file))
  assert.equal(after.length, 1, `expected npm pack in ${cwd} to create one tarball`)
  return after[0]
}

function safeReaddir(dir) {
  try {
    return readdirSync(dir)
  } catch {
    return []
  }
}

function run(command, args, cwd) {
  execFileSync(command, args, {
    cwd,
    stdio: 'inherit',
    env: {
      ...process.env,
      npm_config_ignore_scripts: 'false',
    },
  })
}
