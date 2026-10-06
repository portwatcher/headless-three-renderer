// Minimal Chrome DevTools Protocol client for the browser reference generators.
import { createServer as createNetServer } from 'node:net'

export function findFreePort() {
  const server = createNetServer()
  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port: 0 }, () => {
      const { port } = server.address()
      server.close((error) => {
        if (error) reject(error)
        else resolve(port)
      })
    })
  })
}

export async function waitForChromeDevToolsUrl(port, browser, readStderrTail, readSpawnError) {
  const deadline = Date.now() + 30000
  let lastError
  while (Date.now() < deadline) {
    const spawnError = readSpawnError()
    if (spawnError) {
      throw new Error(`Browser executable failed to start: ${spawnError.message}`)
    }
    if (browser.exitCode !== null) {
      const stderrTail = readStderrTail()
      throw new Error(`Browser executable exited before DevTools became available.${stderrTail ? `\n${stderrTail}` : ''}`)
    }
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/list`)
      if (response.ok) {
        const targets = await response.json()
        const page = targets.find((target) => target.type === 'page' && target.webSocketDebuggerUrl)
        if (page) return page.webSocketDebuggerUrl
      }
    } catch (error) {
      lastError = error
    }
    await delay(150)
  }
  throw new Error(`Timed out waiting for Chrome DevTools on port ${port}.`, { cause: lastError })
}

export function connectChromeDevTools(wsUrl) {
  const socket = new WebSocket(wsUrl)
  const pending = new Map()
  const queuedEvents = []
  const waiters = []
  let nextId = 1

  return new Promise((resolve, reject) => {
    socket.addEventListener('open', () => {
      socket.addEventListener('message', (event) => {
        const message = JSON.parse(event.data)
        if (message.id && pending.has(message.id)) {
          const request = pending.get(message.id)
          pending.delete(message.id)
          if (message.error) {
            request.reject(new Error(`${message.error.message}: ${message.error.data ?? ''}`))
          } else {
            request.resolve(message.result)
          }
          return
        }
        if (message.method) {
          const waiterIndex = waiters.findIndex((waiter) => waiter.method === message.method)
          if (waiterIndex >= 0) {
            const [waiter] = waiters.splice(waiterIndex, 1)
            clearTimeout(waiter.timer)
            waiter.resolve(message.params)
          } else {
            queuedEvents.push(message)
          }
        }
      })

      resolve({
        send(method, params = {}, timeoutMs = 30000) {
          const id = nextId++
          return new Promise((resolveRequest, rejectRequest) => {
            const timer = setTimeout(() => {
              pending.delete(id)
              rejectRequest(new Error(`Timed out waiting for ${method}.`))
            }, timeoutMs)
            pending.set(id, {
              resolve(value) {
                clearTimeout(timer)
                resolveRequest(value)
              },
              reject(error) {
                clearTimeout(timer)
                rejectRequest(error)
              },
            })
            socket.send(JSON.stringify({ id, method, params }))
          })
        },
        waitFor(method, timeoutMs = 30000) {
          const queuedIndex = queuedEvents.findIndex((message) => message.method === method)
          if (queuedIndex >= 0) {
            const [message] = queuedEvents.splice(queuedIndex, 1)
            return Promise.resolve(message.params)
          }
          return new Promise((resolveEvent, rejectEvent) => {
            const timer = setTimeout(() => {
              const waiterIndex = waiters.findIndex((waiter) => waiter.resolve === resolveEvent)
              if (waiterIndex >= 0) waiters.splice(waiterIndex, 1)
              rejectEvent(new Error(`Timed out waiting for ${method}.`))
            }, timeoutMs)
            waiters.push({ method, resolve: resolveEvent, timer })
          })
        },
        close() {
          socket.close()
        },
      })
    }, { once: true })
    socket.addEventListener('error', reject, { once: true })
  })
}

export function waitForProcessExit(child) {
  if (child.exitCode !== null) return Promise.resolve()
  return new Promise((resolve) => {
    const forceKillTimer = setTimeout(() => {
      child.kill('SIGKILL')
    }, 5000)
    const giveUpTimer = setTimeout(resolve, 10000)
    child.once('exit', () => {
      clearTimeout(forceKillTimer)
      clearTimeout(giveUpTimer)
      resolve()
    })
  })
}

export function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
