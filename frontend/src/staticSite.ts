// The website without a Python server (the GitHub Pages build, VITE_STATIC_DATA=haifa): the data of one area was
// downloaded in advance into public/<area>/ (israel_static/scripts/build_haifa_data.py), so nothing needs an
// account or an API key. staticFetch() answers the three server calls from those files:
//   /api/wind     read here, from the monthly MERRA-2 files
//   /api/predict  the same Python model, run in the browser by staticWorker.ts (Pyodide)
//   /api/polygon  the model inputs around the polygon, as a table (also Python in the worker)
import type { WindGrid } from './wind'

export const STATIC_AREA: string = import.meta.env.VITE_STATIC_DATA ?? ''
export const STATIC = STATIC_AREA !== ''
export const DATA_URL = `${import.meta.env.BASE_URL}${STATIC_AREA}/`

type Period = { first_utc: string; last_utc: string; dataset: string }
type WindMonth = { month: string; first_hour_utc: string; hours: number }

// public/<area>/index.json, written by build_haifa_data.py (only the parts the page itself reads).
export type StaticIndex = {
  name: string
  center: [number, number]
  bbox: [number, number, number, number] // [west, south, east, north] where polygons can be drawn
  default_start_utc?: string
  weather: Record<string, Period>
  wind: {
    source: string
    resolution: string
    lat: number[]
    lon: number[]
    first_hour_utc: string
    last_hour_utc: string
    months: WindMonth[]
    scale: number
  } | null
  firms: unknown | null
  nisar: { snapshots: { time_utc: string }[] } | null
  s2: { snapshots: { time_utc: string }[] } | null
}

let indexPromise: Promise<StaticIndex> | null = null

export function staticIndex(): Promise<StaticIndex> {
  indexPromise ??= fetch(`${DATA_URL}index.json`).then((r) => {
    if (!r.ok) throw new Error(`the downloaded data is missing (${DATA_URL}index.json: ${r.status})`)
    return r.json() as Promise<StaticIndex>
  })
  return indexPromise
}

// "2026-08-09T21:00Z", "2026-08-09 21:00" (taken as UTC, as the Python server does) -> ms, or NaN.
export function parseUtc(text: string): number {
  let s = text.trim().replace(' ', 'T')
  if (!/(z|[+-]\d\d:?\d\d)$/i.test(s)) s += 'Z'
  return Date.parse(s)
}

const day = (iso: string) => iso.slice(0, 10)

// The times with downloaded weather, for the hint under the toolbar.
export function periodText(index: StaticIndex): string {
  const all = Object.values(index.weather)
  if (!all.length) return ''
  const first = all.map((p) => p.first_utc).sort()[0]
  const last = all.map((p) => p.last_utc).sort().at(-1)!
  return `${day(first)} to ${day(last)}`
}

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

// ---- wind ----------------------------------------------------------------------------------

const windFiles = new Map<string, Promise<Int16Array>>()

function windMonth(month: string): Promise<Int16Array> {
  let file = windFiles.get(month)
  if (!file) {
    file = fetch(`${DATA_URL}wind/${month}.bin`).then(async (r) => {
      if (!r.ok) throw new Error(`wind file ${month}.bin: ${r.status}`)
      return new Int16Array(await r.arrayBuffer())
    })
    file.catch(() => windFiles.delete(month)) // a failed download is tried again next time
    windFiles.set(month, file)
  }
  return file
}

// POST /api/wind: the MERRA-2 hour that contains `start`, over the whole downloaded box (whatever the view).
async function staticWind(body: { start?: string }): Promise<Response> {
  const index = await staticIndex()
  const w = index.wind
  if (!w) return json(422, { error: 'no wind was downloaded for this site' })
  const t = parseUtc(String(body.start ?? ''))
  if (Number.isNaN(t)) return json(400, { error: 'bad request: start: an ISO time in UTC is required' })
  const hour = Math.floor(t / 3_600_000) * 3_600_000
  const month = new Date(hour).toISOString().slice(0, 7)
  const entry = w.months.find((m) => m.month === month)
  const k = entry ? (hour - Date.parse(entry.first_hour_utc)) / 3_600_000 : -1
  if (!entry || k < 0 || k >= entry.hours)
    return json(422, {
      error:
        `the downloaded MERRA-2 wind covers ${day(w.first_hour_utc)} to ${day(w.last_hour_utc)} only ` +
        '(MERRA-2 is published 2-6 weeks after each month): choose a Start inside it',
    })
  const data = await windMonth(month)
  const nLat = w.lat.length
  const nLon = w.lon.length
  const cells = nLat * nLon
  const grid = (offset: number) =>
    Array.from({ length: nLat }, (_, i) =>
      Array.from({ length: nLon }, (_, j) => data[offset + i * nLon + j] * w.scale),
    )
  const u = grid(k * cells) // the file holds all u (hour, lat, lon), then all v
  const v = grid((entry.hours + k) * cells)
  let [sumU, sumV, speedMax] = [0, 0, 0]
  for (let i = 0; i < nLat; i++)
    for (let j = 0; j < nLon; j++) {
      sumU += u[i][j]
      sumV += v[i][j]
      speedMax = Math.max(speedMax, Math.hypot(u[i][j], v[i][j]))
    }
  const [meanU, meanV] = [sumU / cells, sumV / cells]
  const answer: WindGrid = {
    time_utc: new Date(hour + 1_800_000).toISOString(), // MERRA-2 times are the middle of the averaging hour
    hour_start_utc: new Date(hour).toISOString(),
    lat: w.lat,
    lon: w.lon,
    u,
    v,
    speed_max: speedMax,
    mean_speed_ms: Math.hypot(meanU, meanV),
    mean_dir_deg: ((Math.atan2(-meanU, -meanV) * 180) / Math.PI + 360) % 360,
    source: w.source,
    resolution: w.resolution,
    bbox: [w.lon[1], w.lat[1], w.lon[nLon - 2], w.lat[nLat - 2]],
    stride: 1,
    global_lon: false,
  }
  return json(200, answer)
}

// ---- the Python model, in a worker -----------------------------------------------------------

type Waiting = { resolve: (answer: { status: number; body: string }) => void }
let worker: Worker | null = null
let nextId = 0
const waiting = new Map<number, Waiting>()
const statusListeners = new Set<(text: string) => void>()

// What the worker is doing ("Loading Python…", "Running the model…"), for the page.
export function onStaticStatus(listener: (text: string) => void): () => void {
  statusListeners.add(listener)
  return () => statusListeners.delete(listener)
}

function startWorker(): Worker {
  if (worker) return worker
  worker = new Worker(new URL('./staticWorker.ts', import.meta.url), { type: 'module' })
  worker.onmessage = (e: MessageEvent) => {
    const m = e.data as { id?: number; status?: number; body?: string; progress?: string }
    if (m.progress !== undefined) statusListeners.forEach((f) => f(m.progress!))
    if (m.id !== undefined) {
      waiting.get(m.id)?.resolve({ status: m.status ?? 500, body: m.body ?? '{"error":"no answer"}' })
      waiting.delete(m.id)
    }
  }
  worker.onerror = (e) => {
    // the worker itself failed to start (blocked CDN, old browser): answer everything that waits
    const body = JSON.stringify({ error: `the Python model could not start in this browser: ${e.message || 'worker error'}` })
    waiting.forEach((w) => w.resolve({ status: 500, body }))
    waiting.clear()
    worker = null
  }
  worker.postMessage({
    init: {
      dataUrl: new URL(DATA_URL, location.href).href,
      bundleUrl: new URL(`${import.meta.env.BASE_URL}py-bundle.json`, location.href).href,
    },
  })
  return worker
}

// Start loading Python as soon as the page opens, so the first Predict does not wait for it.
export function warmUp(): void {
  if (STATIC) startWorker()
}

function callPython(path: string, body: string): Promise<{ status: number; body: string }> {
  const w = startWorker()
  const id = ++nextId
  return new Promise((resolve) => {
    waiting.set(id, { resolve })
    w.postMessage({ id, path, body })
  })
}

// Stands in for fetch(API_BASE + path, init) on the static site.
export async function staticFetch(path: string, init: RequestInit): Promise<Response> {
  const body = typeof init.body === 'string' ? init.body : '{}'
  if (path === '/api/wind') return staticWind(JSON.parse(body))
  const answer = await callPython(path, body)
  return new Response(answer.body, { status: answer.status, headers: { 'Content-Type': 'application/json' } })
}
