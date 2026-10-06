// Regenerates test/parity-references.json: renders test/parity-cases.mjs with Three.js
// WebGLRenderer in Chrome and records the sample pixels. The native lighting model follows
// Three.js r180, so the references need a three@0.180 package (r181+ changed PBR lighting):
//   npm install --prefix <dir> three@0.180.0
//   node test/browser-reference/generate-parity.mjs --browser-executable <chrome> --three <dir>/node_modules/three
// Options: --three-vrm <@pixiv/three-vrm package dir>, --swiftshader, --output <json>,
// --check <json> (compare with a reference file instead of writing), --any-revision.
import { createReadStream, existsSync, realpathSync } from 'node:fs'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { connectChromeDevTools, findFreePort, waitForChromeDevToolsUrl, waitForProcessExit } from './chrome-devtools.mjs'

const REFERENCE_REVISION = '180'
const repoRoot = fileURLToPath(new URL('../../../../', import.meta.url))
const packageRoot = fileURLToPath(new URL('../../', import.meta.url))
const options = parseOptions(process.argv.slice(2))
const roots = {
  '/__three__/': realpathSync(options.three ?? path.join(packageRoot, 'node_modules', 'three')),
  '/__three_vrm__/': realpathSync(options.threeVrm ?? path.join(packageRoot, 'node_modules', '@pixiv', 'three-vrm')),
  '/': repoRoot,
}

const server = createServer(async (request, response) => {
  const pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://127.0.0.1').pathname)
  const prefix = Object.keys(roots).find((candidate) => pathname.startsWith(candidate))
  const root = roots[prefix]
  const filePath = path.resolve(root, pathname.slice(prefix.length))
  const relative = path.relative(root, filePath)
  if (relative.startsWith('..') || path.isAbsolute(relative) || !existsSync(filePath) || !(await stat(filePath)).isFile()) {
    response.writeHead(404).end('Not found')
    return
  }
  const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json' }
  response.writeHead(200, { 'content-type': types[path.extname(filePath)] ?? 'application/octet-stream' })
  createReadStream(filePath).pipe(response)
})
await new Promise((resolve) => server.listen({ host: '127.0.0.1', port: 0 }, resolve))
const { port } = server.address()

try {
  const result = await renderInChrome(`http://127.0.0.1:${port}/packages/renderer/test/browser-reference/parity.html`)
  if (result.three !== REFERENCE_REVISION && !options.anyRevision) {
    throw new Error(`parity references need Three.js r${REFERENCE_REVISION}, got r${result.three}; pass --three <three@0.${REFERENCE_REVISION} package dir>`)
  }
  const references = {
    generator: {
      three: result.three,
      renderer: result.renderer,
      userAgent: result.userAgent,
      date: new Date().toISOString().slice(0, 10),
    },
    cases: result.cases,
  }
  if (options.check) {
    const expected = JSON.parse(await readFile(options.check, 'utf8'))
    let worst = 0
    for (const [name, samples] of Object.entries(expected.cases)) {
      samples.forEach((sample, index) => sample.forEach((value, channel) => {
        worst = Math.max(worst, Math.abs(value - (references.cases[name]?.[index]?.[channel] ?? Infinity)))
      }))
    }
    console.log(`${result.renderer}, Three.js r${result.three}: largest channel difference ${worst}`)
  } else {
    const output = options.output ?? path.join(packageRoot, 'test', 'parity-references.json')
    await writeFile(output, `${JSON.stringify(references, null, 1).replace(/\[\n\s+(\d+),\n\s+(\d+),\n\s+(\d+)\n\s+\]/g, '[$1, $2, $3]')}\n`)
    console.log(`Wrote ${Object.keys(references.cases).length} parity cases from ${result.renderer}, Three.js r${result.three}, to ${output}`)
  }
} finally {
  server.close()
}

async function renderInChrome(url) {
  const debugPort = await findFreePort()
  const userDataDir = await mkdtemp(path.join(tmpdir(), 'headless-three-parity-'))
  const angle = options.swiftshader
    ? ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
    : ['--ignore-gpu-blocklist', '--enable-gpu']
  const browser = spawn(options.browserExecutable, [
    '--headless=new',
    '--no-first-run',
    '--no-default-browser-check',
    '--force-color-profile=srgb',
    ...angle,
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${userDataDir}`,
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] })
  let stderrTail = ''
  let spawnError
  browser.stderr.on('data', (chunk) => {
    stderrTail = `${stderrTail}${String(chunk)}`.slice(-4096)
  })
  browser.once('error', (error) => {
    spawnError = error
  })
  let cdp
  try {
    const wsUrl = await waitForChromeDevToolsUrl(debugPort, browser, () => stderrTail, () => spawnError)
    cdp = await connectChromeDevTools(wsUrl)
    await cdp.send('Page.enable')
    await cdp.send('Runtime.enable')
    await cdp.send('Page.navigate', { url })
    await cdp.waitFor('Page.loadEventFired', 30000)
    const evaluation = await cdp.send('Runtime.evaluate', {
      expression: 'globalThis.__HEADLESS_THREE_PARITY_READY__',
      awaitPromise: true,
      returnByValue: true,
    }, 180000)
    if (evaluation.exceptionDetails) {
      throw new Error(evaluation.exceptionDetails.exception?.description ?? evaluation.exceptionDetails.text)
    }
    return evaluation.result.value
  } finally {
    cdp?.close()
    browser.kill('SIGTERM')
    await waitForProcessExit(browser)
    await rm(userDataDir, { recursive: true, force: true })
  }
}

function parseOptions(args) {
  const parsed = { swiftshader: false }
  for (let i = 0; i < args.length; i += 1) {
    const value = () => {
      if (i + 1 >= args.length) throw new Error(`${args[i]} requires a value`)
      return args[++i]
    }
    if (args[i] === '--') continue
    else if (args[i] === '--browser-executable') parsed.browserExecutable = value()
    else if (args[i] === '--three') parsed.three = value()
    else if (args[i] === '--three-vrm') parsed.threeVrm = value()
    else if (args[i] === '--output') parsed.output = value()
    else if (args[i] === '--check') parsed.check = value()
    else if (args[i] === '--swiftshader') parsed.swiftshader = true
    else if (args[i] === '--any-revision') parsed.anyRevision = true
    else throw new Error(`Unknown option ${args[i]}`)
  }
  if (!parsed.browserExecutable) throw new Error('--browser-executable <chrome> is required')
  return parsed
}
