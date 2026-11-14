import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import './App.css'

type Point = [number, number] // [lng, lat]
type Mode = 'idle' | 'drawing' | 'locked'

const COLOR = '#e4572e'
const MAX_AREA_KM2 = 400 // above this the area is shown in red

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

  const draw = () => {
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
    setPoints([])
    setHistory([])
    setMode('idle')
  }

  const area = areaKm2(points)

  return (
    <div className="page">
      <div className="toolbar">
        <button onClick={draw} disabled={mode === 'drawing'}>Draw</button>
        <button onClick={finish} disabled={mode !== 'drawing' || points.length < 3}>Finish</button>
        <button onClick={undo} disabled={mode !== 'drawing' || history.length === 0}>Undo</button>
        <button onClick={clear} disabled={points.length === 0 && mode === 'idle'}>Clear</button>
        {mode !== 'idle' && (
          <span className={`area ${area > MAX_AREA_KM2 ? 'too-big' : 'ok'}`}>
            {area.toLocaleString('en-US', { maximumFractionDigits: 1 })} km²
          </span>
        )}
      </div>
      <div ref={container} className="map" />
    </div>
  )
}

export default App
