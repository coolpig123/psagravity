/// <reference lib="webworker" />
// Web Worker of the static site: runs the repo's Python spread model in the browser with Pyodide
// (https://pyodide.org, loaded from the jsDelivr CDN: Python + numpy / pandas / xarray / scipy / pyproj).
// The Python code is py-bundle.json (made by vite.config.ts from spread_model, gee_layers and
// israel_static/browser); the data files are fetched from public/<area>/ only when a request needs them.
//
// Messages in:  { init: { dataUrl, bundleUrl } } once, then { id, path, body } per request
// Messages out: { progress: text } while working, { id, status, body } per request (body = JSON text)

const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v314.0.8/full/'
const PACKAGES = ['numpy', 'pandas', 'xarray', 'scipy', 'pyproj']
const PY_DIR = '/repo/static/py' // firespread.survival looks two folders up for the repo: keep it this deep
const DATA_DIR = '/data'

// The parts of Pyodide used here.
type Pyodide = {
  loadPackage: (names: string[]) => Promise<void>
  runPython: (code: string) => unknown
  pyimport: (name: string) => { files_for: (body: string) => string; handle: (path: string, body: string) => string }
  FS: {
    mkdirTree: (path: string) => void
    writeFile: (path: string, data: string | Uint8Array) => void
  }
}

const scope = self as unknown as DedicatedWorkerGlobalScope
const progress = (text: string) => scope.postMessage({ progress: text })

let dataUrl = ''
let ready: Promise<Pyodide> | null = null
const haveFiles = new Set<string>()

function mkdirFor(py: Pyodide, path: string) {
  py.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')))
}

async function fetchInto(py: Pyodide, name: string): Promise<void> {
  if (haveFiles.has(name)) return
  const res = await fetch(dataUrl + name)
  if (!res.ok) throw new Error(`data file ${name}: ${res.status}`)
  const path = `${DATA_DIR}/${name}`
  mkdirFor(py, path)
  py.FS.writeFile(path, new Uint8Array(await res.arrayBuffer()))
  haveFiles.add(name)
}

async function start(bundleUrl: string): Promise<Pyodide> {
  progress('Loading Python (first visit: about half a minute, then cached by the browser)…')
  const { loadPyodide } = (await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs`)) as {
    loadPyodide: (options: { indexURL: string }) => Promise<Pyodide>
  }
  const py = await loadPyodide({ indexURL: PYODIDE_URL })
  const [bundle] = await Promise.all([
    fetch(bundleUrl).then((r) => {
      if (!r.ok) throw new Error(`py-bundle.json: ${r.status}`)
      return r.json() as Promise<Record<string, string>>
    }),
    py.loadPackage(PACKAGES),
  ])
  for (const [name, source] of Object.entries(bundle)) {
    const path = `${PY_DIR}/${name}`
    mkdirFor(py, path)
    py.FS.writeFile(path, source)
  }
  py.FS.mkdirTree(DATA_DIR)
  await Promise.all([fetchInto(py, 'index.json'), fetchInto(py, 'static.npz')])
  py.runPython(`import sys\nsys.path.insert(0, ${JSON.stringify(PY_DIR)})\nimport staticfire, firespread.web`)
  progress('')
  return py
}

scope.onmessage = async (e: MessageEvent) => {
  const m = e.data as { init?: { dataUrl: string; bundleUrl: string }; id?: number; path?: string; body?: string }
  if (m.init) {
    dataUrl = m.init.dataUrl
    ready = start(m.init.bundleUrl)
    ready.catch((err) => progress(`Python could not start: ${err instanceof Error ? err.message : err}`))
    return
  }
  if (m.id === undefined) return
  try {
    if (!ready) throw new Error('the worker was not initialised')
    const py = await ready
    const staticfire = py.pyimport('staticfire')
    progress('Reading the downloaded data…')
    const files = JSON.parse(staticfire.files_for(m.body!)) as string[]
    await Promise.all(files.map((f) => fetchInto(py, f)))
    progress('Running the model in your browser…')
    const answer = staticfire.handle(m.path!, m.body!) // "status\n{json}"
    const cut = answer.indexOf('\n')
    scope.postMessage({ id: m.id, status: Number(answer.slice(0, cut)), body: answer.slice(cut + 1) })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    scope.postMessage({
      id: m.id,
      status: 500,
      body: JSON.stringify({ error: `the Python model failed in the browser: ${message}` }),
    })
  } finally {
    progress('')
  }
}
