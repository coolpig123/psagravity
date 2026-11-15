import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import { LoaderCircle, LocateFixed, Search, X } from 'lucide-react'

// Search bar and "my location" button on top of the map.
// Places come from OpenStreetMap Nominatim (https://operations.osmfoundation.org/policies/nominatim/):
// at most one request per second and no search-as-you-type, so we search only when Enter / the button is pressed.
const NOMINATIM = 'https://nominatim.openstreetmap.org/search'
const MAX_ZOOM = 15 // a house or a street: zoomed in no further than this
const ZOOM_OUT = 1 // after fitting the place, zoom out by this much to show its surroundings
const MY_LOCATION_ZOOM = 12

type Place = {
  display_name: string
  lat: string
  lon: string
  boundingbox: [string, string, string, string] // south, north, west, east
}

type Status = { state: 'idle' | 'busy' | 'error'; message?: string }

// Fly to the place, framing its whole extent (a city, a country...), then a bit zoomed out.
function flyToBounds(map: maplibregl.Map, bounds: maplibregl.LngLatBoundsLike, maxZoom = MAX_ZOOM) {
  const cam = map.cameraForBounds(bounds, { padding: 40 })
  if (!cam?.center) return
  const zoom = Math.max(1, Math.min(cam.zoom ?? maxZoom, maxZoom) - ZOOM_OUT)
  map.flyTo({ center: cam.center, zoom, duration: 2500, essential: true })
}

// Blue dot where the user is.
function locationDot(): HTMLElement {
  const el = document.createElement('div')
  el.className = 'my-location-dot'
  el.title = 'Your location'
  return el
}

export default function LocationSearch({ map }: { map: maplibregl.Map | null }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<Place[]>([])
  const [status, setStatus] = useState<Status>({ state: 'idle' })
  const [locating, setLocating] = useState(false)
  const marker = useRef<maplibregl.Marker | null>(null)
  const searchId = useRef(0) // answers of older searches are dropped

  // Ask the browser for the user's position (it asks permission the first time) and fly there.
  // quiet: on page load, a refusal is not an error, the search bar is still there.
  const locate = (quiet: boolean) => {
    if (!map) return
    if (!('geolocation' in navigator)) {
      if (!quiet) setStatus({ state: 'error', message: 'This browser cannot give your location: search for a place.' })
      return
    }
    setLocating(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocating(false)
        setStatus({ state: 'idle' })
        const { longitude: lng, latitude: lat, accuracy } = pos.coords
        if (!marker.current) marker.current = new maplibregl.Marker({ element: locationDot() })
        marker.current.setLngLat([lng, lat]).addTo(map)
        // Frame the accuracy circle (city-level from Wi-Fi / IP), never closer than MY_LOCATION_ZOOM.
        const d = Math.max(accuracy, 500) / 111_320 // metres -> degrees of latitude
        const dLng = d / Math.max(Math.cos((lat * Math.PI) / 180), 0.01)
        flyToBounds(map, [[lng - dLng, lat - d], [lng + dLng, lat + d]], MY_LOCATION_ZOOM + ZOOM_OUT)
      },
      (err) => {
        setLocating(false)
        if (quiet) return
        const message =
          err.code === err.PERMISSION_DENIED
            ? 'Location access is blocked: allow it in the browser (icon in the address bar), or search for a place.'
            : 'Could not find your location: search for a place.'
        setStatus({ state: 'error', message })
      },
      { enableHighAccuracy: false, timeout: 15_000, maximumAge: 10 * 60_000 },
    )
  }

  // On page load: try the user's location once the map is ready.
  useEffect(() => {
    locate(true)
    return () => {
      marker.current?.remove()
      marker.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map])

  const goTo = (place: Place) => {
    if (!map) return
    const [s, n, w, e] = place.boundingbox.map(Number)
    flyToBounds(map, [[w, s], [e, n]])
    setQuery(place.display_name)
    setResults([])
  }

  const search = async (e: React.FormEvent) => {
    e.preventDefault()
    const q = query.trim()
    if (!q || !map) return
    const id = ++searchId.current
    setResults([])
    setStatus({ state: 'busy' })
    try {
      const params = { q, format: 'jsonv2', limit: '5', 'accept-language': navigator.language || 'en' }
      const res = await fetch(`${NOMINATIM}?${new URLSearchParams(params)}`)
      if (!res.ok) throw new Error(`the place search answered ${res.status}`)
      const places = (await res.json()) as Place[]
      if (id !== searchId.current) return
      if (!places.length) {
        setStatus({ state: 'error', message: `No place found for "${q}".` })
        return
      }
      setStatus({ state: 'idle' })
      const [first, ...others] = places
      goTo(first)
      // The other matches, in case the first one is not the right place (a city and its district can share a name).
      const seen = new Set([first.display_name])
      setResults(others.filter((p) => !seen.has(p.display_name) && seen.add(p.display_name)))
    } catch (err) {
      if (id !== searchId.current) return
      setStatus({ state: 'error', message: `Search failed: ${err instanceof Error ? err.message : err}` })
    }
  }

  return (
    <div className="location-search">
      <form className="search-row" onSubmit={search}>
        <Search className="search-icon" />
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Escape' && setResults([])}
          placeholder="Search a place…"
          aria-label="Search a place"
          disabled={!map}
        />
        {query && (
          <button
            type="button"
            className="icon-button"
            title="Clear the search"
            onClick={() => {
              setQuery('')
              setResults([])
              setStatus({ state: 'idle' })
            }}
          >
            <X />
          </button>
        )}
        <button type="submit" className="icon-button" title="Search" disabled={!map || status.state === 'busy'}>
          {status.state === 'busy' ? <LoaderCircle className="spin" /> : <Search />}
        </button>
        <button
          type="button"
          className="icon-button"
          title="Go to my location"
          onClick={() => locate(false)}
          disabled={!map || locating}
        >
          {locating ? <LoaderCircle className="spin" /> : <LocateFixed />}
        </button>
      </form>
      {results.length > 0 && (
        <ul className="search-results">
          <li className="search-results-title">Other matches</li>
          {results.map((p, i) => (
            <li key={i}>
              <button type="button" onClick={() => goTo(p)}>{p.display_name}</button>
            </li>
          ))}
        </ul>
      )}
      {status.state === 'error' && <div className="search-message">{status.message}</div>}
    </div>
  )
}
