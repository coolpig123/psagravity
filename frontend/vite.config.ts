import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

// https://vite.dev/config/
export default defineConfig({
  // Don't run Babel on pre-bundled dependencies in the cache folder.
  plugins: [react({ exclude: [/\/node_modules\//, /psagravity-vite/] })],
  // Keep Vite's cache out of OneDrive, which locks files while syncing.
  cacheDir: join(tmpdir(), 'psagravity-vite'),
  // MapLibre loads its worker file next to itself; pre-bundling breaks that path.
  optimizeDeps: { exclude: ['maplibre-gl'] },
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
