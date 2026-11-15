import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import { Check, Download, Flame, Pencil, Ruler, Trash2, Undo2 } from 'lucide-react'
import 'maplibre-gl/dist/maplibre-gl.css'
import './App.css'
import LocationSearch from './LocationSearch'
import StartTimePicker from './StartTimePicker'
import { boxCovers, legendGradient, LEGEND_MAX_MS, sampleWind, strideFor, tileBox, viewBox, WindLayer, type WindGrid } from './wind'

type Point = [number, number] // [lng, lat]
type Mode = 'idle' | 'drawing' | 'locked'

const COLOR = '#d62828' // brand red, same as the title and main buttons

// OpenStreetMap standard tiles. Usage policy (https://operations.osmfoundation.org/policies/tiles/):
// light use only and a visible "© OpenStreetMap contributors" credit (see the credit link below the map).
const OSM_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      maxzoom: 19,
    },
  },
  layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
}
const MAX_AREA_KM2 = 400 // above this the area is shown in red
const DEFAULT_START = '2026-08-09T21:00Z' // Bug Fire, first afternoon: easy to test with
const EMPTY: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] }

// Where the Python server is: same origin by default (Vite proxy in npm run dev). The GitHub Pages build has no
// server (VITE_NO_SERVER=1), unless VITE_API_BASE points to one that is reachable from the browser.
const API_BASE: string = import.meta.env.VITE_API_BASE ?? ''
const NO_SERVER = import.meta.env.VITE_NO_SERVER === '1' && !API_BASE
const NO_SERVER_MESSAGE = 'This online version has no Python server: run the app locally to use it (see frontend/README.md)'
const api = (path: string, init: RequestInit) => {
  if (NO_SERVER) throw new Error(NO_SERVER_MESSAGE)
  return fetch(API_BASE + path, init)
}

// Answer of POST /api/predict (spread_model/src/firespread/web.py).
type WeatherTimelineStep = {
  time_utc: string
  wind_speed_ms?: number | null
  wind_dir_deg?: number | null
  temp_c?: number | null
  rh_pct?: number | null
}

type Prediction = {
  cells: GeoJSON.FeatureCollection // one square per cell, properties.p = ignition probability
  bounds: [number, number, number, number]
  summary: {
    main_direction?: string
    main_direction_deg?: number
    direction_strength?: number
    expected_new_area_km2?: number
    weather_time_utc?: string
    weather_source?: string
    weather_dataset?: string
    weather_steps?: number
    weather_note?: string
    nisar_passes_used?: string
    nisar_coverage_pct?: number
    s2_clear_pct?: string
    s2_burned_pct?: string
    sim_steps?: number
    sim_expected_area_km2?: number
    sim_reached_grid_edge?: number
    sim_time_developing?: number
    sim_weather_steps?: number
    sim_forecast_hours?: number
    weather_timeline?: WeatherTimelineStep[]
    n_cells: number // cells at risk within the CA steps
    n_map_cells?: number // squares on the map (also the arrival-time cells)
    max_prob: number | null
    notes: string[]
    // arrival times (survival model, spread_model/src/firespread/survival.py); false if it was skipped
    arrival?: boolean
    arrival_horizons_h?: number[]
    arrival_model?: string
    arrival_area_6h_km2?: number
    arrival_area_12h_km2?: number
    arrival_area_24h_km2?: number
    arrival_area_48h_km2?: number
    // VIIRS detections (NASA FIRMS) in the 12 h before the start that mark the active front; 0 / absent = whole polygon
    active_detections?: number
    active_front_km2?: number
  }
}

// What the prediction cells are coloured by: CA chance (p), median arrival hours (t50), or chance within N h.
type View = 'p' | 'pe' | 't50' | 'p6' | 'p12' | 'p24' | 'p48'
const VIEW_LABELS: Record<View, string> = {
  p: 'Chance of burning (CA steps)',
  pe: 'Chance the fire ever gets there',
  t50: 'Arrival time if it gets there (median hours)',
  p6: 'Chance within 6 h',
  p12: 'Chance within 12 h',
  p24: 'Chance within 24 h',
  p48: 'Chance within 48 h',
}
const CHANCE_COLORS: (number | string)[] = [0, '#ffd23f', 0.25, '#ff9f1c', 0.5, '#f3722c', 0.75, '#d62828', 1, '#6a040f']
// soon = dark red, later = yellow
const HOURS_COLORS: (number | string)[] = [0, '#6a040f', 6, '#d62828', 12, '#f3722c', 24, '#ff9f1c', 48, '#ffd23f', 96, '#fff3b0']

function viewPaint(view: View) {
  const value: maplibregl.ExpressionSpecification = ['coalesce', ['get', view], view === 't50' ? 1000 : 0]
  const ramp = ['interpolate', ['linear'], value, ...(view === 't50' ? HOURS_COLORS : CHANCE_COLORS)] as unknown as
    maplibregl.ExpressionSpecification
  const opacity = (view === 't50'
    ? ['interpolate', ['linear'], value, 0, 0.9, 48, 0.7, 96, 0.45]
    : ['interpolate', ['linear'], value, 0, 0.45, 0.3, 0.75, 1, 0.9]) as unknown as maplibregl.ExpressionSpecification
  // arrival view: the time if the fire gets there, so only cells it has a real chance (>= 10%) to reach;
  // chance views: hide ~0 chances
  const filter = (view === 't50'
    ? ['>=', ['coalesce', ['get', 'pe'], 0], 0.1]
    : ['>', ['coalesce', ['get', view], 0], 0.01]) as unknown as maplibregl.FilterSpecification
  return { ramp, opacity, filter }
}

const fmtHours = (h: number | null | undefined) =>
  h === null || h === undefined ? '–' : h < 1 ? `${Math.round(h * 60)} min` : h < 72 ? `${h.toFixed(1)} h` : `${(h / 24).toFixed(1)} days`
const fmtPct = (p: number | null | undefined) => (p === null || p === undefined ? '–' : `${Math.round(p * 100)}%`)

type WindStatus = { state: 'idle' | 'loading' | 'ok' | 'error'; message?: string }

// 16-point compass name of a direction in degrees (0 = north, clockwise).
function compass(deg: number): string {
  const names = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
  return names[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16]
}

// "2026-08-09T21:00:00+00:00" -> "21:00-22:00": the hour a MERRA-2 hourly mean covers.
function hourRange(hourStartIso: string): string {
  const h = new Date(hourStartIso).getUTCHours()
  const two = (n: number) => String(n).padStart(2, '0')
  return `${two(h)}:00–${two((h + 1) % 24)}:00`
}

// Area of the closed polygon on the Earth's surface, in km² (spherical formula, as in Turf.js).
function areaKm2(points: Point[]): number {
  if (points.length < 3) return 0
  const R = 6371.0088 // mean Earth radius, km
  const rad = (d: number) => (d * Math.PI) / 180
  let sum = 0
  for (let i = 0; i < points.length; i++) {
    const prev = points[(i - 1 + points.length) % points.length]
    const curr = points[i]
    const next = points[(i + 1) % points.length]
    sum += (rad(next[0]) - rad(prev[0])) * Math.sin(rad(curr[1]))
  }
  return Math.abs((sum * R * R) / 2)
}

// Distance in Web Mercator units, so it is not distorted by latitude like raw degrees.
function distance(a: Point, b: Point): number {
  const ma = maplibregl.MercatorCoordinate.fromLngLat(a)
  const mb = maplibregl.MercatorCoordinate.fromLngLat(b)
  return Math.hypot(ma.x - mb.x, ma.y - mb.y)
}

// Insert p into the closed ring between the two neighbours where it adds the least
// outline length. The closing edge (last -> first) is checked first, so a point placed
// "in order" is simply appended.
function insertNearest(ring: Point[], p: Point): Point[] {
  if (ring.length < 3) return [...ring, p]
  let best = ring.length
  let bestCost = Infinity
  for (let k = 0; k < ring.length; k++) {
    const i = (ring.length - 1 + k) % ring.length // start with the closing edge
    const a = ring[i]
    const b = ring[(i + 1) % ring.length]
    const cost = distance(a, p) + distance(p, b) - distance(a, b)
    if (cost < bestCost) {
      bestCost = cost
      best = i + 1
    }
  }
  return [...ring.slice(0, best), p, ...ring.slice(best)]
}

// Points are stored as lng/lat, so the shape stays in place when the map moves or resizes.
function toGeoJSON(points: Point[], mode: Mode): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = []
  if (mode === 'locked') {
    features.push({
      type: 'Feature',
      properties: {},
      geometry: { type: 'Polygon', coordinates: [[...points, points[0]]] },
    })
  } else {
    // Outline closed back to the first point, so the shape always looks like a polygon.
    // One line per edge: the dash pattern restarts at each corner, so moving or adding a point
    // only shifts the dashes of the edges touching it (one long line would shift them all).
    const edges = points.length === 2 ? 1 : points.length >= 3 ? points.length : 0
    for (let i = 0; i < edges; i++) {
      features.push({
        type: 'Feature',
        properties: {},
        geometry: { type: 'LineString', coordinates: [points[i], points[(i + 1) % points.length]] },
      })
    }
    // Each point carries its outline index, so it can be picked up and dragged.
    points.forEach((p, i) => {
      features.push({ type: 'Feature', properties: { i }, geometry: { type: 'Point', coordinates: p } })
    })
  }
  return { type: 'FeatureCollection', features }
}

function App() {
  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [points, setPoints] = useState<Point[]>([]) // outline order
  const [history, setHistory] = useState<Point[][]>([]) // outline before each add / move, for Undo
  const [mode, setMode] = useState<Mode>('idle')
  const [showRegions, setShowRegions] = useState(false)
  const modeRef = useRef(mode)
  modeRef.current = mode
  const pointsRef = useRef(points)
  pointsRef.current = points
  const [start, setStart] = useState(DEFAULT_START) // UTC time the prediction starts
  const [weatherSource, setWeatherSource] = useState<'auto' | 'meteomatics' | 'rtma' | 'gfs' | 'era5_land'>('auto')
  const [forecastHours, setForecastHours] = useState(6)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState('')
  const [view, setView] = useState<View>('p')
  const requestId = useRef(0) // answers of older requests (polygon cleared or redrawn meanwhile) are dropped
  const windLayer = useRef<WindLayer | null>(null)
  const [showWind, setShowWind] = useState(true)
  const [wind, setWind] = useState<WindGrid | null>(null)
  const [windStatus, setWindStatus] = useState<WindStatus>({ state: 'idle' })
  const windRequest = useRef(0) // same as requestId, for the wind
  // The last wind request (on its way, answered or failed): its start time and the area it covers.
  const windAsked = useRef<{ start: string; box: [number, number, number, number]; stride: number } | null>(null)
  const windTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const [viewTick, setViewTick] = useState(0) // +1 after each pan / zoom, so the wind can follow the view

  useEffect(() => {
    const map = new maplibregl.Map({
      container: container.current!,
      style: OSM_STYLE,
      center: [0, 0],
      zoom: 1,
      maplibreLogo: false,
      attributionControl: false,
      // Flat 2D map only: no tilting (pitch) and no rotating.
      maxPitch: 0,
      dragRotate: false,
      pitchWithRotate: false,
      touchPitch: false,
    })
    map.touchZoomRotate.disableRotation()
    map.keyboard.disableRotation()
    mapRef.current = map
    // Dev server only: lets browser tests move the map (window.__map.jumpTo(...)).
    if (import.meta.env.DEV) (window as unknown as { __map: maplibregl.Map }).__map = map

    map.on('load', () => {
      // Where get_fire_inputs has full data (green) or not (red); built by backend/make_regions.py.
      map.addSource('regions', { type: 'geojson', data: `${import.meta.env.BASE_URL}regions.geojson` })
      map.addLayer({
        id: 'regions-fill',
        type: 'fill',
        source: 'regions',
        layout: { visibility: 'none' },
        paint: {
          'fill-color': ['case', ['get', 'available'], '#00ff00', '#ff0000'],
          'fill-opacity': 0.3,
        },
      })
      map.addSource('polygon', { type: 'geojson', data: toGeoJSON([], 'idle') })
      map.addLayer({
        id: 'polygon-fill',
        type: 'fill',
        source: 'polygon',
        filter: ['==', ['geometry-type'], 'Polygon'],
        paint: { 'fill-color': COLOR, 'fill-opacity': 0.25 },
      })
      // Solid outline once locked (Polygon), dashed while still drawing (LineString).
      map.addLayer({
        id: 'polygon-line',
        type: 'line',
        source: 'polygon',
        filter: ['==', ['geometry-type'], 'Polygon'],
        paint: { 'line-color': COLOR, 'line-width': 2 },
      })
      map.addLayer({
        id: 'polygon-draft',
        type: 'line',
        source: 'polygon',
        filter: ['==', ['geometry-type'], 'LineString'],
        paint: { 'line-color': COLOR, 'line-width': 2, 'line-dasharray': [3, 2] },
      })
      map.addLayer({
        id: 'polygon-points',
        type: 'circle',
        source: 'polygon',
        filter: ['==', ['geometry-type'], 'Point'],
        paint: {
          'circle-color': COLOR,
          'circle-radius': 5,
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 2,
        },
      })
      // Chance each cell burns within the simulated steps: yellow -> orange -> red, under the polygon outline.
      // Low chances are drawn lighter, so the likely path stands out.
      map.addSource('prediction', { type: 'geojson', data: EMPTY })
      const ramp: maplibregl.ExpressionSpecification = [
        'interpolate', ['linear'], ['get', 'p'], 0, '#ffd23f', 0.25, '#ff9f1c', 0.5, '#f3722c', 0.75, '#d62828', 1, '#6a040f',
      ]
      map.addLayer(
        {
          id: 'prediction-fill',
          type: 'fill',
          source: 'prediction',
          paint: {
            'fill-color': ramp,
            'fill-outline-color': ramp, // no light seams between neighbouring cells
            'fill-opacity': ['interpolate', ['linear'], ['get', 'p'], 0, 0.45, 0.3, 0.75, 1, 0.9],
          },
        },
        'polygon-line',
      )
      // Wind animation: speed colours under the prediction, particles on a canvas above the map.
      windLayer.current = new WindLayer(map, container.current!.parentElement!, 'prediction-fill')
      map.on('moveend', () => setViewTick((t) => t + 1))
      setLoaded(true)
    })

    // Click a prediction cell (when not drawing): its chance and arrival time.
    map.on('click', 'prediction-fill', (e) => {
      if (modeRef.current === 'drawing') return
      const q = e.features?.[0]?.properties
      if (!q) return
      const arrival = q.pe !== undefined
      const html =
        `<b>Chance of burning (CA steps):</b> ${fmtPct(q.p)}` +
        (arrival
          ? `<br><b>Chance the fire ever gets here:</b> ${fmtPct(q.pe)}` +
            (q.t50 === null || q.t50 === undefined
              ? '<br><b>Arrival, if it does:</b> no estimate (the chance hardly grows over 6–48 h)'
              : `<br><b>Arrival, if it does:</b> median ${fmtHours(q.t50)} (half of the cases ${fmtHours(q.t25)} – ${fmtHours(q.t75)})`) +
            `<br><b>Chance within</b> 6 h ${fmtPct(q.p6)} · 12 h ${fmtPct(q.p12)} · 24 h ${fmtPct(q.p24)} · 48 h ${fmtPct(q.p48)}`
          : '')
      new maplibregl.Popup({ closeButton: true, maxWidth: '320px' }).setLngLat(e.lngLat).setHTML(html).addTo(map)
    })
    map.on('mouseenter', 'prediction-fill', () => {
      if (modeRef.current !== 'drawing') map.getCanvas().style.cursor = 'pointer'
    })
    map.on('mouseleave', 'prediction-fill', () => {
      if (modeRef.current !== 'drawing') map.getCanvas().style.cursor = ''
    })

    // A 'click' only fires if the mouse barely moved; a drag pans the map instead.
    map.on('click', (e) => {
      if (modeRef.current !== 'drawing') return
      // Clicking on an existing point does not add a new one.
      if (map.queryRenderedFeatures(e.point, { layers: ['polygon-points'] }).length) return
      const p: Point = [e.lngLat.lng, e.lngLat.lat]
      setHistory((h) => [...h, pointsRef.current])
      setPoints((ring) => insertNearest(ring, p))
    })

    // Drag an existing point: hover shows a move cursor, press and drag moves the point.
    let dragged: Point | null = null
    let beforeDrag: Point[] = [] // outline when the drag started, saved for Undo
    let movedDuringDrag = false
    map.on('mouseenter', 'polygon-points', () => {
      if (modeRef.current === 'drawing' && !dragged) map.getCanvas().style.cursor = 'move'
    })
    map.on('mouseleave', 'polygon-points', () => {
      if (modeRef.current === 'drawing' && !dragged) map.getCanvas().style.cursor = 'crosshair'
    })
    map.on('mousedown', 'polygon-points', (e) => {
      if (modeRef.current !== 'drawing' || e.originalEvent.button !== 0) return
      const i = e.features?.[0]?.properties?.i
      if (typeof i !== 'number') return
      e.preventDefault() // stop the map from panning during this drag
      dragged = pointsRef.current[i]
      beforeDrag = pointsRef.current
      movedDuringDrag = false
      map.getCanvas().style.cursor = 'grabbing'
    })
    map.on('mousemove', (e) => {
      if (!dragged) return
      const old = dragged
      const moved: Point = [e.lngLat.lng, e.lngLat.lat]
      dragged = moved
      movedDuringDrag = true
      setPoints((ring) => ring.map((q) => (q === old ? moved : q)))
    })
    // Listen on the whole window, so releasing the button outside the map also ends the drag.
    const endDrag = () => {
      if (!dragged) return
      dragged = null
      map.getCanvas().style.cursor = 'move'
      // The whole drag is one Undo step (a press without moving is not saved).
      if (movedDuringDrag) setHistory((h) => [...h, beforeDrag])
    }
    window.addEventListener('mouseup', endDrag)

    return () => {
      window.removeEventListener('mouseup', endDrag)
      windLayer.current?.destroy()
      windLayer.current = null
      map.remove()
      mapRef.current = null
      setLoaded(false)
    }
  }, [])

  // Redraw the shape whenever the points or the mode change.
  useEffect(() => {
    const source = mapRef.current?.getSource<maplibregl.GeoJSONSource>('polygon')
    source?.setData(toGeoJSON(points, mode))
  }, [points, mode, loaded])

  // Show or hide the available / unavailable regions.
  useEffect(() => {
    if (!loaded) return
    mapRef.current?.setLayoutProperty('regions-fill', 'visibility', showRegions ? 'visible' : 'none')
  }, [showRegions, loaded])

  // Crosshair cursor while drawing.
  useEffect(() => {
    if (mapRef.current) mapRef.current.getCanvas().style.cursor = mode === 'drawing' ? 'crosshair' : ''
  }, [mode, loaded])

  // Show the prediction cells (or nothing) on the map.
  useEffect(() => {
    const source = mapRef.current?.getSource<maplibregl.GeoJSONSource>('prediction')
    source?.setData(prediction ? prediction.cells : EMPTY)
    // Zoom to the predicted area so it is easy to see.
    if (prediction && prediction.cells.features.length) {
      const [w, s, e, n] = prediction.bounds
      mapRef.current?.fitBounds([[w, s], [e, n]], { padding: 30, duration: 800 })
    }
  }, [prediction, loaded])

  // Colour the cells by the chosen view.
  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map?.getLayer('prediction-fill')) return
    const { ramp, opacity, filter } = viewPaint(view)
    map.setPaintProperty('prediction-fill', 'fill-color', ramp)
    map.setPaintProperty('prediction-fill', 'fill-outline-color', ramp)
    map.setPaintProperty('prediction-fill', 'fill-opacity', opacity)
    map.setFilter('prediction-fill', filter)
  }, [view, loaded])

  // A prediction without arrival times can only be shown as the CA chance.
  useEffect(() => {
    if (prediction && !prediction.summary.arrival) setView('p')
  }, [prediction])

  // The wind is always available (polygon or not): fetch the MERRA-2 wind over the whole visible map for the
  // start time, at any zoom (coarser grid when zoomed far out). After a pan or zoom, fetch again only if the view
  // leaves the area of the last request (answered, on its way or failed), or if zooming in deserves a finer grid,
  // so map resizes and small moves never start a new request; the old wind stays meanwhile. A new start time
  // always fetches again. Nothing is fetched while the wind is switched off.
  useEffect(() => {
    const map = mapRef.current
    const t = start.trim()
    if (!showWind || !t || !map || !loaded) {
      windRequest.current++
      clearTimeout(windTimer.current)
      windAsked.current = null
      setWind(null)
      setWindStatus({ state: 'idle' })
      return
    }
    const box = viewBox(map)
    const want = tileBox(box)
    const wantStride = strideFor(want)
    const asked = windAsked.current
    if (
      asked &&
      asked.start === t &&
      boxCovers(asked.box, box) &&
      (asked.stride <= wantStride || asked.box.every((x, k) => x === want[k]))
    )
      return
    const id = ++windRequest.current
    windAsked.current = { start: t, box: want, stride: wantStride }
    if (!asked || asked.start !== t) setWindStatus({ state: 'loading' })
    clearTimeout(windTimer.current) // a newer request replaces one not sent yet (typing a start time)
    windTimer.current = setTimeout(async () => {
      try {
        const res = await api('/api/wind', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bbox: box, start: t }),
        })
        if (res.status === 503) throw new Error('the Python server is not running (npm run dev)')
        const data = await res.json().catch(() => null)
        if (res.status === 404 && data?.no_data)
          throw new Error('no MERRA-2 data for this start time yet (published 2-6 weeks after each month)')
        // A server started before the website was updated: no /api/wind (404) or the older version (polygon).
        if (res.status === 404 || (res.status === 400 && /polygon/.test(data?.error ?? '')))
          throw new Error('the Python server is older than the website: stop npm run dev (Ctrl+C) and start it again')
        if (!res.ok) throw new Error(data?.error ?? `the server answered ${res.status}`)
        if (id !== windRequest.current) return
        const grid = data as WindGrid
        if (windAsked.current && grid.stride) windAsked.current.stride = grid.stride
        setWind(grid)
        setWindStatus({ state: 'ok' })
      } catch (err) {
        if (id !== windRequest.current) return
        setWind(null)
        setWindStatus({ state: 'error', message: err instanceof Error ? err.message : String(err) })
      }
    }, 400)
  }, [showWind, start, viewTick, loaded])

  useEffect(() => {
    windLayer.current?.setData(wind)
  }, [wind, loaded])

  useEffect(() => {
    windLayer.current?.setVisible(showWind)
  }, [showWind, loaded])

  // Clearing or redrawing the polygon removes the prediction and ignores one still on its way.
  const dropPrediction = () => {
    requestId.current++
    setPrediction(null)
    setPredicting(false)
    setError('')
  }

  const draw = () => {
    dropPrediction()
    setPoints([])
    setHistory([])
    setMode('drawing')
  }
  const finish = () => setMode('locked')
  // Undo reverts the last operation: removes an added point, or moves a dragged point back.
  const undo = () => {
    setPoints(history[history.length - 1])
    setHistory((h) => h.slice(0, -1))
  }
  const clear = () => {
    dropPrediction()
    setPoints([])
    setHistory([])
    setMode('idle')
  }

  const area = areaKm2(points)

  // Wind at the middle of the polygon for the label (none without a polygon, or if it is outside the loaded area).
  const windHere = (() => {
    if (!wind || points.length < 3) return null
    const lng = points.reduce((a, p) => a + p[0], 0) / points.length
    const lat = points.reduce((a, p) => a + p[1], 0) / points.length
    const uv = sampleWind(wind, lng, lat)
    if (!uv) return null
    const dir = (((Math.atan2(-uv[0], -uv[1]) * 180) / Math.PI) + 360) % 360 // direction the wind comes from
    return { speed: Math.hypot(uv[0], uv[1]), dir }
  })()

  // One-line hint for the next step, shown under the toolbar.
  const hint =
    mode === 'idle'
      ? 'Click Draw to start a polygon.'
      : mode === 'locked'
        ? 'Set the start time (UTC) and press Predict to see where the fire is likely to spread, or Clear to start over.'
        : points.length < 3
          ? 'Click the map to add points (at least 3). Drag a point to move it.'
          : 'Click to add more points, drag a point to move it, or press Finish to lock the polygon.'

  // Send the finished polygon to the Python server, which prints its points.
  const extract = async () => {
    const feature: GeoJSON.Feature = {
      type: 'Feature',
      properties: { area_km2: area },
      geometry: { type: 'Polygon', coordinates: [[...points, points[0]]] },
    }
    if (NO_SERVER) return // nothing to send it to
    try {
      const res = await api('/api/polygon', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feature),
      })
      if (res.status === 503) throw new Error('The Python server is not running. Start everything with: npm run dev')
      if (!res.ok) throw new Error(`The server answered ${res.status}: ${await res.text()}`)
    } catch (err) {
      alert(`Could not send the polygon.
${err instanceof Error ? err.message : err}`)
    }
  }

  // Run the spread model on the finished polygon: the server fetches only the data around it.
  const predict = async () => {
    const id = ++requestId.current
    setPrediction(null)
    setError('')
    setPredicting(true)
    try {
      const res = await api('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          polygon: toGeoJSON(points, 'locked'),
          start: start.trim(),
          weather_source: weatherSource,
          hours: forecastHours,
        }),
      })
      if (res.status === 503) throw new Error('The Python server is not running. Start everything with: npm run dev')
      const data = await res.json().catch(() => null)
      if (!res.ok) throw new Error(data?.error ?? `The server answered ${res.status}`)
      if (id === requestId.current) setPrediction(data as Prediction)
    } catch (err) {
      if (id === requestId.current) setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (id === requestId.current) setPredicting(false)
    }
  }

  const s = prediction?.summary

  return (
    <div className="page">
      <header className="topbar">
        <div>Psapyro</div>
        <div className="tagline">Predict the spread, protect the people.</div>
      </header>
      <div className="toolbar">
        <button className="primary" onClick={draw} disabled={mode === 'drawing'}><Pencil />Draw</button>
        <button onClick={finish} disabled={mode !== 'drawing' || points.length < 3}><Check />Finish</button>
        <button onClick={undo} disabled={mode !== 'drawing' || history.length === 0}><Undo2 />Undo</button>
        <button onClick={clear} disabled={points.length === 0 && mode === 'idle'}><Trash2 />Clear</button>
        <button onClick={extract} disabled={mode !== 'locked'}><Download />Extract Data</button>
        <StartTimePicker value={start} onChange={setStart} />
        <label className="start">
          Weather
          <select value={weatherSource} onChange={(e) => setWeatherSource(e.target.value as typeof weatherSource)}>
            <option value="auto">Auto (Meteomatics / RTMA / GFS)</option>
            <option value="meteomatics">Meteomatics (1 km Forecast)</option>
            <option value="rtma">NOAA RTMA (US analysis)</option>
            <option value="gfs">NOAA GFS (0.25° forecast)</option>
            <option value="era5_land">ECMWF ERA5-Land</option>
          </select>
        </label>
        <label className="start">
          Forecast (h)
          <input
            type="number"
            min={1}
            max={168}
            value={forecastHours}
            onChange={(e) => setForecastHours(Math.max(1, Math.min(168, Number(e.target.value) || 6)))}
            style={{ width: '3.5rem' }}
          />
        </label>
        <button
          className="primary"
          onClick={predict}
          disabled={mode !== 'locked' || area > MAX_AREA_KM2 || predicting || !start.trim()}
        >
          <Flame />
          {predicting ? 'Predicting…' : 'Predict'}
        </button>
        {mode !== 'idle' && (
          <span className={`area ${area > MAX_AREA_KM2 ? 'too-big' : 'ok'}`}>
            <Ruler />
            {area.toLocaleString('en-US', { maximumFractionDigits: 1 })} km²
          </span>
        )}
      </div>
      <div className="hint">{hint}</div>
      <div className="map-wrap">
        <div ref={container} className="map" />
        <LocationSearch map={loaded ? mapRef.current : null} />
        <a className="credit" href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">
          © OpenStreetMap contributors
        </a>
      </div>
      <div className="options">
        <label className="regions">
          <input type="checkbox" checked={showRegions} onChange={(e) => setShowRegions(e.target.checked)} />
          regions with full US fuel data (LANDFIRE + gridMET); Predict also runs elsewhere, with global data
        </label>
        <label className="regions wind">
          <input type="checkbox" checked={showWind} onChange={(e) => setShowWind(e.target.checked)} />
          {!showWind && 'wind (MERRA-2)'}
          {showWind && windStatus.state === 'loading' && 'wind (loading MERRA-2…)'}
          {showWind && windStatus.state === 'error' && (
            <span className="wind-error" title={windStatus.message}>
              wind: {windStatus.message}
            </span>
          )}
          {showWind && windStatus.state === 'ok' && wind && (
            <span
              title={`MERRA-2 wind at 10 m, mean of ${hourRange(wind.hour_start_utc)} UTC on ${wind.time_utc.slice(0, 10)} (the Start time), ${wind.resolution}. Colours: speed 0-${LEGEND_MAX_MS} m/s.`}
            >
              wind (MERRA-2, {hourRange(wind.hour_start_utc)} UTC)
              {windHere && `: ${windHere.speed.toFixed(1)} m/s from ${compass(windHere.dir)} at the polygon`}
              <span className="wind-ramp" style={{ background: legendGradient() }} />
            </span>
          )}
        </label>
      </div>
      <div className="panel">
        {predicting && <p>Fetching Earth Engine and NISAR data around the polygon and running the model. This can take a few minutes…</p>}
        {error && <p className="error">Prediction failed: {error}</p>}
        {s && (
          <>
            <p>
              <b>Most likely spread:</b>{' '}
              {s.n_cells === 0
                ? 'no spread expected (nothing around the polygon can burn)'
                : `${s.main_direction} (${s.main_direction_deg?.toFixed(0)}°), strength ${s.direction_strength?.toFixed(2)}`}
              {' · '}
              <b>Expected new burned area:</b> {s.sim_expected_area_km2?.toFixed(1)} km² within {s.sim_steps} steps
              {' '}({s.n_cells} cells at risk; {s.expected_new_area_km2?.toFixed(2)} km² in the next step)
            </p>
            <p>
              <b>Weather:</b> {s.weather_time_utc || 'none'} UTC{s.weather_source ? ` (${s.weather_source})` : ''}
              {s.sim_time_developing
                ? ` · time-developing over ${s.sim_forecast_hours?.toFixed(0) ?? s.sim_weather_steps} h (${s.sim_weather_steps} hourly steps)`
                : ''}
              {' · '}
              <b>NISAR:</b>{' '}
              {s.nisar_passes_used ? `${s.nisar_passes_used}, covers ${s.nisar_coverage_pct?.toFixed(0)}%` : 'not used'}
              {' · '}
              <b>Burn scars (Sentinel-2):</b>{' '}
              {s.s2_clear_pct !== undefined ? `${s.s2_burned_pct}% burned, ${s.s2_clear_pct}% clear view` : 'not used'}
            </p>
            {s.arrival && (
              <p>
                <b>Arrival time (survival model):</b> expected burned area within{' '}
                {(s.arrival_horizons_h ?? []).map((h, i) => (
                  <span key={h}>
                    {i > 0 && ', '}
                    {h} h {(s[`arrival_area_${h}h_km2` as keyof typeof s] as number | undefined)?.toFixed(1)} km²
                  </span>
                ))}
                {'. '}
<b>Active front:</b>{' '}
                {s.active_detections
                  ? `${s.active_front_km2?.toFixed(1)} km² (${s.active_detections} VIIRS detections in the last 12 h)`
                  : 'the whole polygon'}
                {'. '}
                <span className="model-name">{s.arrival_model}</span>
              </p>
            )}
            {s.weather_timeline && s.weather_timeline.length > 1 && (
              <p className="note">
                <b>Forecast evolution ({s.weather_timeline.length}h):</b>{' '}
                {s.weather_timeline.slice(0, 6).map((w) => {
                  const hh = w.time_utc.slice(11, 16)
                  const ws = w.wind_speed_ms != null ? `${w.wind_speed_ms.toFixed(1)} m/s` : ''
                  const wd = w.wind_dir_deg != null ? `@${w.wind_dir_deg.toFixed(0)}°` : ''
                  const tc = w.temp_c != null ? `, ${w.temp_c.toFixed(0)}°C` : ''
                  const rh = w.rh_pct != null ? `/${w.rh_pct.toFixed(0)}%RH` : ''
                  return `${hh}Z (${ws}${wd}${tc}${rh})`
                }).join(' → ')}
                {s.weather_timeline.length > 6 ? ' …' : ''}
              </p>
            )}
            {[s.weather_note, ...s.notes].filter(Boolean).map((note, i) => (
              <p key={i} className="note">Note: {note}</p>
            ))}
            <p className="legend">
              {s.arrival && (
                <label className="view">
                  Show{' '}
                  <select value={view} onChange={(e) => setView(e.target.value as View)}>
                    {(Object.keys(VIEW_LABELS) as View[]).map((v) => (
                      <option key={v} value={v}>{VIEW_LABELS[v]}</option>
                    ))}
                  </select>
                </label>
              )}
              {view === 'p' && (
                <>Chance of burning within {s.sim_steps} steps (one step = up to one 100 m cell) <span className="ramp" /> 0 → 1</>
              )}
              {view === 't50' && (
                <>Median hours until the fire arrives, if it does (cells it reaches with ≥ 10% chance) 0 h <span className="ramp hours" /> 4 days</>
              )}
              {view !== 'p' && view !== 't50' && (
                <>{VIEW_LABELS[view]} <span className="ramp" /> 0 → 1</>
              )}
              {' · click a cell for its numbers'}
            </p>
          </>
        )}
      </div>
    </div>
  )
}

export default App
