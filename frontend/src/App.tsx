import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import './App.css'

type Point = [number, number] // [lng, lat]
type Mode = 'idle' | 'drawing' | 'locked'

const COLOR = '#e4572e'
const MAX_AREA_KM2 = 400 // above this the area is shown in red
const DEFAULT_START = '2026-08-09T21:00Z' // Bug Fire, first afternoon: easy to test with
const EMPTY: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] }

// Answer of POST /api/predict (spread_model/src/firespread/web.py).
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
    weather_note?: string
    nisar_passes_used?: string
    nisar_coverage_pct?: number
    s2_clear_pct?: string
    s2_burned_pct?: string
    sim_steps?: number
    sim_expected_area_km2?: number
    sim_reached_grid_edge?: number
    n_cells: number
    max_prob: number | null
    notes: string[]
  }
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
    if (points.length >= 2) {
      features.push({
        type: 'Feature',
        properties: {},
        geometry: { type: 'LineString', coordinates: [...points, points[0]] },
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
  const modeRef = useRef(mode)
  modeRef.current = mode
  const pointsRef = useRef(points)
  pointsRef.current = points
  const [start, setStart] = useState(DEFAULT_START) // UTC time the prediction starts
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState('')
  const requestId = useRef(0) // answers of older requests (polygon cleared or redrawn meanwhile) are dropped

  useEffect(() => {
    const map = new maplibregl.Map({
      container: container.current!,
      style: 'https://demotiles.maplibre.org/style.json',
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

    map.on('load', () => {
      map.addSource('polygon', { type: 'geojson', data: toGeoJSON([], 'idle') })
      map.addLayer({
        id: 'polygon-fill',
        type: 'fill',
        source: 'polygon',
        filter: ['==', ['geometry-type'], 'Polygon'],
        paint: { 'fill-color': COLOR, 'fill-opacity': 0.25 },
      })
      map.addLayer({
        id: 'polygon-line',
        type: 'line',
        source: 'polygon',
        filter: ['!=', ['geometry-type'], 'Point'],
        paint: { 'line-color': COLOR, 'line-width': 2 },
      })
      map.addLayer({
        id: 'polygon-points',
        type: 'circle',
        source: 'polygon',
        filter: ['==', ['geometry-type'], 'Point'],
        paint: { 'circle-color': COLOR, 'circle-radius': 5 },
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
      setLoaded(true)
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

  // Send the finished polygon to the Python server, which prints its points.
  const extract = async () => {
    const feature: GeoJSON.Feature = {
      type: 'Feature',
      properties: { area_km2: area },
      geometry: { type: 'Polygon', coordinates: [[...points, points[0]]] },
    }
    try {
      const res = await fetch('/api/polygon', {
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
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ polygon: toGeoJSON(points, 'locked'), start: start.trim() }),
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
      <div className="toolbar">
        <button onClick={draw} disabled={mode === 'drawing'}>Draw</button>
        <button onClick={finish} disabled={mode !== 'drawing' || points.length < 3}>Finish</button>
        <button onClick={undo} disabled={mode !== 'drawing' || history.length === 0}>Undo</button>
        <button onClick={clear} disabled={points.length === 0 && mode === 'idle'}>Clear</button>
        <button onClick={extract} disabled={mode !== 'locked'}>Extract Data</button>
        <label className="start">
          Start (UTC)
          <input value={start} onChange={(e) => setStart(e.target.value)} placeholder={DEFAULT_START} size={18} />
        </label>
        <button
          className="predict"
          onClick={predict}
          disabled={mode !== 'locked' || area > MAX_AREA_KM2 || predicting || !start.trim()}
        >
          {predicting ? 'Predicting…' : 'Predict'}
        </button>
        {mode !== 'idle' && (
          <span className={`area ${area > MAX_AREA_KM2 ? 'too-big' : 'ok'}`}>
            {area.toLocaleString('en-US', { maximumFractionDigits: 1 })} km²
          </span>
        )}
      </div>
      <div ref={container} className="map" />
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
              {' · '}
              <b>NISAR:</b>{' '}
              {s.nisar_passes_used ? `${s.nisar_passes_used}, covers ${s.nisar_coverage_pct?.toFixed(0)}%` : 'not used'}
              {' · '}
              <b>Burn scars (Sentinel-2):</b>{' '}
              {s.s2_clear_pct !== undefined ? `${s.s2_burned_pct}% burned, ${s.s2_clear_pct}% clear view` : 'not used'}
            </p>
            {[s.weather_note, ...s.notes].filter(Boolean).map((note, i) => (
              <p key={i} className="note">Note: {note}</p>
            ))}
            <p className="legend">
              Chance of burning within {s.sim_steps} steps (one step = up to one 100 m cell) <span className="ramp" /> 0 → 1
            </p>
          </>
        )}
      </div>
    </div>
  )
}

export default App
