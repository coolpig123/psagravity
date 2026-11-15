// Animated wind on the map, in the style of earth.nullschool.net (Cameron Beccario, MIT licence):
// a colour overlay of wind speed under the map's own layers, and thousands of particles drifting with the
// wind on a canvas above it, leaving short fading trails, brighter where the wind is stronger.
// The wind is a u/v grid (MERRA-2, ~55 km cells) from POST /api/wind covering the whole visible map (the server
// widens the view to 5 deg tiles, so small pans and zooms need no new data); it is bilinearly interpolated.
import * as maplibregl from 'maplibre-gl'

// Answer of POST /api/wind (firedata.merra2.wind_grid).
export type WindGrid = {
  time_utc: string
  hour_start_utc: string
  lat: number[] // south -> north, cell centres
  lon: number[] // west -> east
  u: number[][] // [lat][lon], m/s towards the east
  v: number[][] // [lat][lon], m/s towards the north
  speed_max: number
  mean_speed_ms: number
  mean_dir_deg: number
  source: string
  resolution: string
  bbox?: [number, number, number, number] // [west, south, east, north] where the wind is complete (no faded edge)
}

// Zoomed far out, only part of the view gets wind (at most this many degrees), to keep downloads small.
export const MAX_VIEW_LON_DEG = 30
export const MAX_VIEW_LAT_DEG = 20

// The visible map as [west, south, east, north], clamped to the world. If it is wider than MAX_VIEW_* degrees,
// it is cut to that size around `focus` (the polygon, kept inside the view) or else the middle of the view.
export function viewBox(map: maplibregl.Map, focus?: [number, number]): [number, number, number, number] {
  const b = map.getBounds()
  let [w, e] = [Math.max(-180, b.getWest()), Math.min(180, b.getEast())]
  let [s, n] = [Math.max(-85, b.getSouth()), Math.min(85, b.getNorth())]
  // centre c, clamped so the cut stays inside [lo, hi]
  const cut = (lo: number, hi: number, c: number, span: number): [number, number] => {
    const mid = Math.min(Math.max(c, lo + span / 2), hi - span / 2)
    return [mid - span / 2, mid + span / 2]
  }
  if (e - w > MAX_VIEW_LON_DEG) [w, e] = cut(w, e, focus ? focus[0] : (w + e) / 2, MAX_VIEW_LON_DEG)
  if (n - s > MAX_VIEW_LAT_DEG) [s, n] = cut(s, n, focus ? focus[1] : (s + n) / 2, MAX_VIEW_LAT_DEG)
  return [w, s, e, n]
}

// True if the grid already has complete wind for the whole box (then no new request is needed).
export function covers(g: WindGrid, box: [number, number, number, number]): boolean {
  if (!g.bbox) return false
  return box[0] >= g.bbox[0] && box[1] >= g.bbox[1] && box[2] <= g.bbox[2] && box[3] <= g.bbox[3]
}

// earth.nullschool.net "wind" overlay: segmented colour scale by speed (m/s).
const SPEED_SCALE: [number, [number, number, number]][] = [
  [0, [37, 74, 255]],
  [1, [0, 100, 254]],
  [3, [0, 200, 254]],
  [5, [37, 193, 146]],
  [7, [0, 230, 0]],
  [9, [0, 250, 0]],
  [11, [254, 225, 0]],
  [13, [254, 174, 0]],
  [15, [220, 74, 29]],
  [17, [180, 0, 50]],
  [19, [254, 0, 150]],
  [21, [151, 50, 222]],
  [24, [86, 54, 222]],
  [27, [42, 132, 222]],
  [29, [64, 199, 222]],
  [100, [255, 255, 255]],
]
export const LEGEND_MAX_MS = 20

export function speedColor(s: number): [number, number, number] {
  for (let i = 1; i < SPEED_SCALE.length; i++) {
    const [s1, c1] = SPEED_SCALE[i]
    if (s <= s1) {
      const [s0, c0] = SPEED_SCALE[i - 1]
      const f = Math.max(0, (s - s0) / (s1 - s0))
      return [0, 1, 2].map((k) => Math.round(c0[k] + f * (c1[k] - c0[k]))) as [number, number, number]
    }
  }
  return SPEED_SCALE[SPEED_SCALE.length - 1][1]
}

// CSS gradient of the overlay colours from 0 to LEGEND_MAX_MS, for the legend.
export function legendGradient(): string {
  const stops = SPEED_SCALE.filter(([s]) => s <= LEGEND_MAX_MS).map(
    ([s, c]) => `rgb(${c.join(',')}) ${((100 * s) / LEGEND_MAX_MS).toFixed(0)}%`,
  )
  return `linear-gradient(to right, ${stops.join(', ')})`
}

// As earth.nullschool.net: particles brighter where the wind is stronger (0 .. 17 m/s in steps). Here white with
// rising opacity, because the map under the overlay is lighter than nullschool's black globe.
const INTENSITY: string[] = []
for (let a = 0.6; a <= 1.0001; a += 0.025) INTENSITY.push(`rgba(255, 255, 255, ${a.toFixed(3)})`)
const INTENSITY_MAX_MS = 17

const MAX_AGE = 100 // frames a particle lives before it is reborn somewhere else
const FRAME_MS = 40 // ~25 frames per second, as nullschool
const FADE = 0.96 // trail alpha kept per frame
const PARTICLES_PER_PX = 1 / 250 // particles per screen pixel of the wind area
const MAX_PARTICLES = 5000
const LINE_WIDTH = 1.25
// The overlay colours are darkened, so the light particles stand out as on nullschool's dark globe;
// the map stays readable underneath. The last cells fade out, so the wind area has no hard edge.
const OVERLAY_OPACITY = 0.75
const OVERLAY_DARKEN = 0.45
const EDGE_FADE_CELLS = 1
const OVERLAY_ID = 'wind-overlay'

type Particle = { x: number; y: number; age: number }

// Web Mercator y (0 at the north edge of the world, 1 at the south), as maplibregl.MercatorCoordinate,
// written out because it runs for every particle in every frame.
const mercY = (lat: number) => 0.5 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / (2 * Math.PI)
const latOfMercY = (y: number) => (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI

// Bilinear u, v at (lng, lat), or null outside the grid / where it has no data.
export function sampleWind(g: WindGrid, lng: number, lat: number): [number, number] | null {
  const nLat = g.lat.length
  const nLon = g.lon.length
  if (nLat < 2 || nLon < 2) return null
  const fi = ((lat - g.lat[0]) / (g.lat[nLat - 1] - g.lat[0])) * (nLat - 1)
  const fj = ((lng - g.lon[0]) / (g.lon[nLon - 1] - g.lon[0])) * (nLon - 1)
  if (!(fi >= 0 && fi <= nLat - 1 && fj >= 0 && fj <= nLon - 1)) return null
  const i = Math.min(Math.floor(fi), nLat - 2)
  const j = Math.min(Math.floor(fj), nLon - 2)
  const di = fi - i
  const dj = fj - j
  const lerp = (a: number[][]) =>
    (a[i][j] * (1 - dj) + a[i][j + 1] * dj) * (1 - di) + (a[i + 1][j] * (1 - dj) + a[i + 1][j + 1] * dj) * di
  return [lerp(g.u), lerp(g.v)]
}

// Distance (in grid cells) from (lng, lat) to the nearest edge of the grid.
function edgeCells(g: WindGrid, lng: number, lat: number): number {
  const fi = ((lat - g.lat[0]) / (g.lat[g.lat.length - 1] - g.lat[0])) * (g.lat.length - 1)
  const fj = ((lng - g.lon[0]) / (g.lon[g.lon.length - 1] - g.lon[0])) * (g.lon.length - 1)
  return Math.max(0, Math.min(fi, g.lat.length - 1 - fi, fj, g.lon.length - 1 - fj))
}

export class WindLayer {
  private map: maplibregl.Map
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private grid: WindGrid | null = null
  private visible = false
  private moving = false
  private particles: Particle[] = []
  private frame = 0
  private lastStep = 0
  private beforeLayer: string | undefined
  // screen <-> map (the map is flat: no pitch, no rotation): lng linear in x, Mercator y linear in y
  private view = { w: 0, h: 0, lng0: 0, dLng: 0, my0: 0, dMy: 0 }
  private area = { x0: 0, y0: 0, x1: 0, y1: 0 } // wind grid on screen, clipped to the canvas

  constructor(map: maplibregl.Map, parent: HTMLElement, beforeLayer?: string) {
    this.map = map
    this.beforeLayer = beforeLayer
    this.canvas = document.createElement('canvas')
    this.canvas.className = 'wind-canvas'
    parent.appendChild(this.canvas)
    this.ctx = this.canvas.getContext('2d')!
    map.on('movestart', this.onMoveStart)
    map.on('moveend', this.onMoveEnd)
    map.on('resize', this.onMoveEnd)
    this.resize()
  }

  destroy() {
    cancelAnimationFrame(this.frame)
    this.map.off('movestart', this.onMoveStart)
    this.map.off('moveend', this.onMoveEnd)
    this.map.off('resize', this.onMoveEnd)
    this.removeOverlay()
    this.canvas.remove()
  }

  setData(grid: WindGrid | null) {
    this.grid = grid
    this.removeOverlay()
    if (grid) this.addOverlay(grid)
    this.restart()
  }

  setVisible(on: boolean) {
    this.visible = on
    if (this.map.getLayer(OVERLAY_ID)) this.map.setLayoutProperty(OVERLAY_ID, 'visibility', on ? 'visible' : 'none')
    this.canvas.style.display = on ? '' : 'none'
    this.restart()
  }

  // ---- speed overlay: an image on the map, drawn under the prediction and the polygon ----

  private addOverlay(g: WindGrid) {
    const [w, e] = [g.lon[0], g.lon[g.lon.length - 1]]
    const [s, n] = [g.lat[0], g.lat[g.lat.length - 1]]
    const W = 512
    const H = 512
    const c = document.createElement('canvas')
    c.width = W
    c.height = H
    const cx = c.getContext('2d')!
    const img = cx.createImageData(W, H)
    const top = mercY(n)
    const bottom = mercY(s)
    for (let r = 0; r < H; r++) {
      const lat = latOfMercY(top + ((r + 0.5) / H) * (bottom - top)) // rows evenly spaced on the Mercator map
      for (let q = 0; q < W; q++) {
        const lng = w + ((q + 0.5) / W) * (e - w)
        const uv = sampleWind(g, lng, lat)
        const k = 4 * (r * W + q)
        if (!uv) continue
        const [R, G, B] = speedColor(Math.hypot(uv[0], uv[1]))
        const edge = Math.min(edgeCells(g, lng, lat) / EDGE_FADE_CELLS, 1)
        img.data[k] = R * OVERLAY_DARKEN
        img.data[k + 1] = G * OVERLAY_DARKEN
        img.data[k + 2] = B * OVERLAY_DARKEN
        img.data[k + 3] = 255 * edge
      }
    }
    cx.putImageData(img, 0, 0)
    this.map.addSource(OVERLAY_ID, {
      type: 'image',
      url: c.toDataURL('image/png'),
      coordinates: [[w, n], [e, n], [e, s], [w, s]],
    })
    const before = this.beforeLayer && this.map.getLayer(this.beforeLayer) ? this.beforeLayer : undefined
    this.map.addLayer(
      {
        id: OVERLAY_ID,
        type: 'raster',
        source: OVERLAY_ID,
        layout: { visibility: this.visible ? 'visible' : 'none' },
        paint: { 'raster-opacity': OVERLAY_OPACITY, 'raster-resampling': 'linear', 'raster-fade-duration': 0 },
      },
      before,
    )
  }

  private removeOverlay() {
    if (this.map.getLayer(OVERLAY_ID)) this.map.removeLayer(OVERLAY_ID)
    if (this.map.getSource(OVERLAY_ID)) this.map.removeSource(OVERLAY_ID)
  }

  // ---- particles ----

  // Like nullschool: no animation while the map moves, fresh particles once it stops.
  private onMoveStart = () => {
    this.moving = true
    cancelAnimationFrame(this.frame)
    this.clear()
  }

  private onMoveEnd = () => {
    this.moving = false
    this.resize()
    this.restart()
  }

  private resize() {
    const box = this.map.getContainer()
    const dpr = window.devicePixelRatio || 1
    const w = box.clientWidth
    const h = box.clientHeight
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr)
      this.canvas.height = Math.round(h * dpr)
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const nw = this.map.unproject([0, 0])
    const se = this.map.unproject([w, h])
    const my0 = mercY(nw.lat)
    this.view = { w, h, lng0: nw.lng, dLng: (se.lng - nw.lng) / w, my0, dMy: (mercY(se.lat) - my0) / h }
  }

  private toLngLat(x: number, y: number): [number, number] {
    const v = this.view
    return [v.lng0 + x * v.dLng, latOfMercY(v.my0 + y * v.dMy)]
  }

  private clear() {
    this.ctx.clearRect(0, 0, this.view.w, this.view.h)
  }

  private restart() {
    cancelAnimationFrame(this.frame)
    this.clear()
    this.particles = []
    const g = this.grid
    if (!g || !this.visible || this.moving) return
    // Where the wind grid is on screen.
    const a = this.map.project([g.lon[0], g.lat[g.lat.length - 1]])
    const b = this.map.project([g.lon[g.lon.length - 1], g.lat[0]])
    this.area = {
      x0: Math.max(0, a.x),
      y0: Math.max(0, a.y),
      x1: Math.min(this.view.w, b.x),
      y1: Math.min(this.view.h, b.y),
    }
    const px = Math.max(0, this.area.x1 - this.area.x0) * Math.max(0, this.area.y1 - this.area.y0)
    if (px < 1) return
    const n = Math.min(MAX_PARTICLES, Math.round(px * PARTICLES_PER_PX))
    for (let k = 0; k < n; k++) this.particles.push(this.spawn(Math.floor(Math.random() * MAX_AGE)))
    this.lastStep = 0
    this.frame = requestAnimationFrame(this.tick)
  }

  private spawn(age = 0): Particle {
    const { x0, y0, x1, y1 } = this.area
    return { x: x0 + Math.random() * (x1 - x0), y: y0 + Math.random() * (y1 - y0), age }
  }

  private tick = (now: number) => {
    this.frame = requestAnimationFrame(this.tick)
    if (now - this.lastStep < FRAME_MS) return
    this.lastStep = now
    const g = this.grid!
    const ctx = this.ctx
    const { w, h } = this.view
    // Pixels per frame for 1 m/s: scales with the map height, so it looks the same on any screen.
    const k = 0.25 * (h / 600)

    // Fade the old trails.
    ctx.globalCompositeOperation = 'destination-in'
    ctx.fillStyle = `rgba(0, 0, 0, ${FADE})`
    ctx.fillRect(0, 0, w, h)
    ctx.globalCompositeOperation = 'source-over'

    // One path per brightness level (fast: a few strokes per frame instead of one per particle).
    const buckets: number[][] = INTENSITY.map(() => [])
    for (let p = 0; p < this.particles.length; p++) {
      const pt = this.particles[p]
      if (pt.age++ > MAX_AGE) {
        this.particles[p] = this.spawn()
        continue
      }
      const [lng, lat] = this.toLngLat(pt.x, pt.y)
      const uv = sampleWind(g, lng, lat)
      if (!uv) {
        pt.age = MAX_AGE + 1 // left the grid: reborn next frame
        continue
      }
      const nx = pt.x + uv[0] * k
      const ny = pt.y - uv[1] * k // v is northward, screen y is downward
      const speed = Math.hypot(uv[0], uv[1])
      const b = Math.floor((Math.min(speed, INTENSITY_MAX_MS) / INTENSITY_MAX_MS) * (INTENSITY.length - 1))
      buckets[b].push(pt.x, pt.y, nx, ny)
      pt.x = nx
      pt.y = ny
    }
    ctx.lineWidth = LINE_WIDTH
    buckets.forEach((seg, b) => {
      if (!seg.length) return
      ctx.beginPath()
      ctx.strokeStyle = INTENSITY[b]
      for (let s = 0; s < seg.length; s += 4) {
        ctx.moveTo(seg[s], seg[s + 1])
        ctx.lineTo(seg[s + 2], seg[s + 3])
      }
      ctx.stroke()
    })
  }
}
