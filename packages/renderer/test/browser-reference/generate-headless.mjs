import { createReadStream } from 'node:fs'
import { mkdir, readdir, stat, unlink, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { BROWSER_REFERENCE_MANIFEST_FILE } from './manifest.mjs'

const repoRoot = fileURLToPath(new URL('../../../../', import.meta.url))
const browserReferencePage = '/packages/renderer/test/browser-reference/'
const readyGlobal = '__HEADLESS_THREE_BROWSER_REFERENCE_READY__'

const contentTypes = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.png', 'image/png'],
])

let options
try {
  options = parseArgs(process.argv.slice(2))
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error))
  printHelp()
  process.exit(1)
}

if (options.help) {
  printHelp()
  process.exit(0)
}

if (!options.outputDir) {
  console.error('Missing required --output <directory>.')
  printHelp()
  process.exit(1)
}

const server = await startStaticServer(options.port)
const { port } = server.address()
const baseUrl = `http://127.0.0.1:${port}`

let browser
try {
  const playwright = await loadPlaywright()
  const browserType = playwright[options.browser]
  if (!browserType) {
    throw new Error(`Unsupported Playwright browser "${options.browser}". Use chromium, firefox, or webkit.`)
  }

  browser = await browserType.launch({ headless: !options.headed })
  const page = await browser.newPage()
  page.on('console', (message) => {
    if (message.type() === 'error') {
      console.error(`[browser] ${message.text()}`)
    }
  })

  await page.goto(`${baseUrl}${browserReferencePage}`, { waitUntil: 'load' })
  const result = await page.evaluate(
    async ({ readyGlobal }) => {
      const ready = globalThis[readyGlobal]
      if (!ready || typeof ready.then !== 'function') {
        throw new Error(`Browser reference page did not expose ${readyGlobal}.`)
      }
      return ready
    },
    { readyGlobal },
  )

  await writeBrowserReferences(result, options.outputDir)
  console.log(`Wrote ${result.fixtures.length} browser reference PNGs and ${BROWSER_REFERENCE_MANIFEST_FILE} to ${options.outputDir}`)
} finally {
  await browser?.close()
  await new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) reject(error)
      else resolve()
    })
  })
}

async function loadPlaywright() {
  try {
    return await import('playwright')
  } catch (error) {
    throw new Error(
      [
        'The headless browser reference generator requires Playwright installed in this workspace.',
        'Install it only when regenerating references:',
        '  pnpm add -D playwright',
        '  pnpm exec playwright install chromium',
      ].join('\n'),
      { cause: error },
    )
  }
}

async function writeBrowserReferences(result, outputDir) {
  assertBrowserReferenceResult(result)
  await mkdir(outputDir, { recursive: true })

  const fixtureDataUrls = new Map(result.fixtures.map((fixture) => [fixture.name, fixture.dataUrl]))
  const expectedPngFiles = new Set(result.manifest.fixtures.map((fixture) => fixture.file))
  await removeStaleReferencePngs(outputDir, expectedPngFiles)

  for (const fixture of result.manifest.fixtures) {
    const dataUrl = fixtureDataUrls.get(fixture.name)
    if (!dataUrl) {
      throw new Error(`Generated browser reference result is missing fixture ${fixture.name}.`)
    }
    await writeFile(path.join(outputDir, fixture.file), pngBufferFromDataUrl(dataUrl))
  }

  await writeFile(
    path.join(outputDir, BROWSER_REFERENCE_MANIFEST_FILE),
    `${JSON.stringify(result.manifest, null, 2)}\n`,
  )
}

async function removeStaleReferencePngs(outputDir, expectedPngFiles) {
  const entries = await readdir(outputDir)
  await Promise.all(entries.map(async (entry) => {
    if (!entry.endsWith('.png') || expectedPngFiles.has(entry)) return
    await unlink(path.join(outputDir, entry))
  }))
}

function assertBrowserReferenceResult(result) {
  if (!result || typeof result !== 'object') {
    throw new Error('Browser reference page returned an invalid result.')
  }
  if (!result.manifest || typeof result.manifest !== 'object' || !Array.isArray(result.manifest.fixtures)) {
    throw new Error('Browser reference page returned an invalid manifest.')
  }
  if (!Array.isArray(result.fixtures)) {
    throw new Error('Browser reference page returned an invalid fixture list.')
  }
}

function pngBufferFromDataUrl(dataUrl) {
  const match = /^data:image\/png;base64,([A-Za-z0-9+/=]+)$/.exec(dataUrl)
  if (!match) {
    throw new Error('Browser reference fixture did not return a PNG data URL.')
  }
  return Buffer.from(match[1], 'base64')
}

function startStaticServer(port) {
  const server = createServer(async (request, response) => {
    try {
      if (request.method !== 'GET' && request.method !== 'HEAD') {
        response.writeHead(405)
        response.end()
        return
      }

      const url = new URL(request.url ?? '/', 'http://127.0.0.1')
      const pathname = decodeURIComponent(url.pathname)
      const relativePath = pathname.endsWith('/') ? `${pathname.slice(1)}index.html` : pathname.slice(1)
      const filePath = path.resolve(repoRoot, relativePath)
      if (!isInside(repoRoot, filePath)) {
        response.writeHead(403)
        response.end('Forbidden')
        return
      }

      const fileStat = await stat(filePath)
      if (!fileStat.isFile()) {
        response.writeHead(404)
        response.end('Not found')
        return
      }

      response.writeHead(200, {
        'content-length': fileStat.size,
        'content-type': contentTypes.get(path.extname(filePath)) ?? 'application/octet-stream',
      })
      if (request.method === 'HEAD') {
        response.end()
      } else {
        createReadStream(filePath).pipe(response)
      }
    } catch (error) {
      if (error?.code === 'ENOENT') {
        response.writeHead(404)
        response.end('Not found')
        return
      }
      response.writeHead(500)
      response.end(error instanceof Error ? error.message : String(error))
    }
  })

  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port }, () => {
      server.off('error', reject)
      resolve(server)
    })
  })
}

function isInside(root, candidate) {
  const relative = path.relative(root, candidate)
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative))
}

function parseArgs(args) {
  const parsed = {
    browser: 'chromium',
    headed: false,
    help: false,
    outputDir: undefined,
    port: 0,
  }

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]
    if (arg === '--') {
      continue
    } else if (arg === '--help' || arg === '-h') {
      parsed.help = true
    } else if (arg === '--headed') {
      parsed.headed = true
    } else if (arg === '--browser') {
      parsed.browser = requireValue(args, index, arg)
      index += 1
    } else if (arg === '--output') {
      parsed.outputDir = path.resolve(requireValue(args, index, arg))
      index += 1
    } else if (arg === '--port') {
      parsed.port = parsePort(requireValue(args, index, arg))
      index += 1
    } else {
      throw new Error(`Unknown browser reference generator option: ${arg}`)
    }
  }

  return parsed
}

function requireValue(args, index, option) {
  const value = args[index + 1]
  if (!value || value.startsWith('--')) {
    throw new Error(`Missing value for ${option}.`)
  }
  return value
}

function parsePort(value) {
  const port = Number(value)
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    throw new Error(`Invalid --port value ${value}.`)
  }
  return port
}

function printHelp() {
  console.log(`
Usage:
  node test/browser-reference/generate-headless.mjs --output <directory> [options]

Options:
  --output <directory>   Directory for PNG files and ${BROWSER_REFERENCE_MANIFEST_FILE}.
  --browser <name>       Playwright browser type: chromium, firefox, or webkit. Default: chromium.
  --headed               Show the browser while rendering.
  --port <number>        Static server port. Default: 0, which selects a free port.
  -h, --help             Show this help.
`.trim())
}
