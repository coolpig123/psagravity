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
})
