import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { readdirSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

// The Python the static site runs in the browser (src/staticWorker.ts), as one JSON file {path: source}:
// the spread model and the three geelayers modules it needs straight from the repo (no copies to keep in step),
// plus israel_static/browser (the inputs from downloaded files, and a numpy stand-in for xgboost).
// Served at /py-bundle.json by the dev server and written into the build.
function pythonBundle(): Plugin {
  const repo = resolve(import.meta.dirname, '..')
  const collect = (): string => {
    const files: Record<string, string> = {}
    const add = (dir: string, prefix: string, keep: (name: string) => boolean) => {
      for (const name of readdirSync(join(repo, dir)))
        if (keep(name)) files[prefix + name] = readFileSync(join(repo, dir, name), 'utf8')
    }
    add('spread_model/src/firespread', 'firespread/', (n) => n.endsWith('.py') && n !== '__main__.py')
    add('spread_model/src/firespread/models', 'firespread/models/', (n) => n.endsWith('.json'))
    // not geelayers/__init__.py: it imports Earth Engine
    add('gee_layers/src/geelayers', 'geelayers/', (n) => ['grid.py', 'derive.py', 'perimeter.py'].includes(n))
    files['geelayers/__init__.py'] = ''
    add('israel_static/browser', '', (n) => n.endsWith('.py'))
    return JSON.stringify(files)
  }
  return {
    name: 'python-bundle',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url?.split('?')[0].endsWith('/py-bundle.json')) {
          res.setHeader('Content-Type', 'application/json')
          res.end(collect())
        } else next()
      })
    },
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'py-bundle.json', source: collect() })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  // GitHub Pages serves the site under /<repo>/: the workflow sets BASE_PATH. Locally it stays at /.
  base: process.env.BASE_PATH ?? '/',
  // Don't run Babel on pre-bundled dependencies in the cache folder.
  plugins: [react({ exclude: [/\/node_modules\//, /psagravity-vite/] }), pythonBundle()],
  // Keep Vite's cache out of OneDrive, which locks files while syncing.
  cacheDir: join(tmpdir(), 'psagravity-vite'),
  // MapLibre loads its worker file next to itself; pre-bundling breaks that path.
  optimizeDeps: { exclude: ['maplibre-gl'] },
  worker: { format: 'es' },
  // Forward API calls to the Python server (backend/server.py), so the browser sees one origin.
  // 127.0.0.1, not localhost: the Python server listens on IPv4 only.
  // If it is not running, answer 503 with a clear message instead of a bare 500.
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        configure: (proxy) => {
          proxy.on('error', (_err, _req, res) => {
            if (!('writeHead' in res) || res.headersSent) return
            res.writeHead(503, { 'Content-Type': 'application/json' })
            res.end(JSON.stringify({ error: 'backend not running' }))
          })
        },
      },
    },
  },
})
