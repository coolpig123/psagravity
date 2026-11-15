"""The website's Python server, without a server: the same spread model on files downloaded in advance.

Runs in the browser (Pyodide, started by frontend/src/staticWorker.ts) and in plain Python (tests). It stands in
for backend/server.py on the published site: firespread.web.predict_payload is called with

    predict_fn         inputs cut out of the files of israel_static/scripts/build_haifa_data.py, instead of
                       geelayers.get_fire_inputs (Earth Engine) and firespread.burn_scars (Sentinel-2)
    nisar_factory      the NISAR snapshot of the latest pass before the start, instead of firedata.nisar_provider
    detections_factory the downloaded VIIRS detections, instead of NASA FIRMS

so the model, the simulation and the arrival times are the repo's own code, unchanged. Only places inside the
downloaded area (index.json "bbox") and start times inside the downloaded period work.

    files_for(body)          data files (relative paths) a request needs: fetch them into DATA_DIR first
    handle(path, body_json)  "/api/predict" or "/api/polygon" -> "<status>\n<JSON body>"
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from geelayers.derive import slope_aspect, wind_speed_dir
from geelayers.grid import Grid, _extract_geojson_polygons
from geelayers.perimeter import resolve_perimeter

DATA_DIR = Path("/data")          # set_data_dir() in tests
_index: dict | None = None
_cache: dict[str, dict] = {}      # relative path -> arrays of an .npz file
_notes: list[str] = []            # what the static data changed for the current request, shown under the map

WEATHER_UNITS = {"temp_c": "degC", "rh_pct": "%", "wind_u_ms": "m/s", "wind_v_ms": "m/s", "wind_speed_ms": "m/s",
                 "wind_dir_deg": "degrees", "gust_ms": "m/s", "precip_mm": "mm", "soil_moisture": "m3/m3"}
UNITS = {"elevation_m": "m", "slope_deg": "degrees", "aspect_deg": "degrees", "landcover": "class", "evc": "class",
         "fuel_lifeform": "class", "cover_pct": "%", "nisar_cover_pct": "%", "nisar_soil_moisture": "m3/m3",
         "nisar_vwc_kg_m2": "kg/m2", "nisar_burned": "0/1", "s2_burned": "0/1", **WEATHER_UNITS}
EXTRACT_HOURS = 3.0


def set_data_dir(path) -> None:
    global DATA_DIR, _index
    DATA_DIR, _index = Path(path), None
    _cache.clear()


def index() -> dict:
    global _index
    if _index is None:
        _index = json.loads((DATA_DIR / "index.json").read_text(encoding="utf-8"))
    return _index


def _npz(rel: str) -> dict:
    if rel not in _cache:
        with np.load(DATA_DIR / rel) as z:
            _cache[rel] = {k: z[k] for k in z.files}
    return _cache[rel]


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    if t is pd.NaT:
        raise ValueError("start: a UTC time like 2026-08-09T21:00Z is required")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def master_grid() -> Grid:
    g = index()["grid"]
    return Grid(int(g["epsg"]), float(g["x0"]), float(g["y0"]), float(g["res"]), int(g["width"]), int(g["height"]))


# ---- which files a request needs ----------------------------------------------------------------

def _months(first: pd.Timestamp, last: pd.Timestamp) -> list[str]:
    return [str(p) for p in pd.period_range(first.tz_localize(None), last.tz_localize(None), freq="M")]


def _weather_order(requested: str, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Downloaded weather sources to try, in order (as geelayers.api._weather_candidates: the reanalysis for the
    past it covers, the forecast model otherwise). ValueError for an unknown name."""
    have = index().get("weather") or {}
    if requested not in ("auto", "rtma", "meteomatics", "era5_land", "gfs"):
        raise ValueError(f"unknown weather source {requested!r}")
    if requested in ("era5_land", "gfs"):
        order = [requested]
    else:
        order = ["era5_land", "gfs"]
        if "era5_land" in have and end > _utc(have["era5_land"]["last_utc"]) + pd.Timedelta(hours=1):
            order = ["gfs", "era5_land"]
    # only sources with at least one hour in [start's hour, end)
    return [s for s in order if s in have
            and _utc(have[s]["first_utc"]) < end and _utc(have[s]["last_utc"]) >= start.floor("h")]


def _weather_files(source: str, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    meta = index()["weather"][source]
    return [f"weather/{source}/{m}.npz" for m in _months(start.floor("h"), end) if m in meta["months"]]


def _snapshot(kind: str, start: pd.Timestamp, strict: bool) -> dict | None:
    """The latest snapshot of index()[kind] at or before `start` (before it, if strict)."""
    snaps = (index().get(kind) or {}).get("snapshots") or []
    ok = [s for s in snaps if (_utc(s["time_utc"]) < start if strict else _utc(s["time_utc"]) <= start)]
    return max(ok, key=lambda s: _utc(s["time_utc"])) if ok else None


def _nisar_snapshot(start: pd.Timestamp) -> dict | None:
    snap = _snapshot("nisar", start, strict=True)
    lookback = pd.Timedelta(days=float((index().get("nisar") or {}).get("lookback_days", 30.0)))
    return snap if snap is not None and _utc(snap["time_utc"]) >= start - lookback else None


def files_for(body_json: str) -> str:
    """JSON list of the data files (relative to the data folder) the request needs, static.npz included.
    Never fails: a bad request is reported by handle()."""
    files = ["static.npz"]
    try:
        body = json.loads(body_json)
        props = body.get("properties") or {}
        start = _utc(body.get("start") or props.get("start"))
        hours = float(body.get("hours") or props.get("hours") or (EXTRACT_HOURS if "geometry" in body else 6.0))
        end = start + pd.Timedelta(hours=hours)
        requested = str(body.get("weather_source") or props.get("weather_source") or "auto").lower()
        for source in _weather_order(requested, start, end)[:1]:      # the first one is the one used
            files += _weather_files(source, start, end)
        for snap in (_snapshot("s2", start, strict=False), _nisar_snapshot(start)):
            if snap:
                files.append(snap["file"])
        if index().get("firms"):
            files.append("firms.json")
    except Exception:
        pass
    return json.dumps(files)


# ---- inputs from the files ----------------------------------------------------------------------

def _window(grid: Grid) -> tuple[slice, slice]:
    """Rows and columns of the master grid that `grid` covers (it must nest in it: same CRS, cells, snapping)."""
    m = master_grid()
    c0, r0 = round((grid.x0 - m.x0) / m.res), round((m.y0 - grid.y0) / m.res)
    return slice(r0, r0 + grid.height), slice(c0, c0 + grid.width)


def _fire_grid(polygon, radius_km: float, res_m: float) -> Grid:
    """The model grid: polygon + radius_km, as Grid.from_polygon, cut to the downloaded area."""
    m = master_grid()
    if abs(res_m - m.res) > 1e-6:
        raise ValueError(f"res_m: the downloaded data has {m.res:g} m cells only")
    pts = np.vstack([ring for poly in _extract_geojson_polygons(polygon) for ring in poly])
    w, s, e, n = index()["bbox"]
    if (pts[:, 0] < w).any() or (pts[:, 0] > e).any() or (pts[:, 1] < s).any() or (pts[:, 1] > n).any():
        raise ValueError(f"the polygon is outside the downloaded area: this site only has data around "
                         f"{index()['name']} (longitude {w} to {e}, latitude {s} to {n}; tick the area checkbox "
                         "under the map to see it)")
    g = Grid.from_polygon(polygon, buffer_km=radius_km, res_m=m.res, epsg=m.epsg)
    rows, cols = _window(g)
    r0, r1, c0, c1 = max(rows.start, 0), min(rows.stop, m.height), max(cols.start, 0), min(cols.stop, m.width)
    if (r0, r1, c0, c1) != (rows.start, rows.stop, cols.start, cols.stop):
        _notes.append(f"the {radius_km:g} km area around the polygon reaches past the downloaded area around "
                      f"{index()['name']}: the model stops at its edge")
    return Grid(m.epsg, m.x0 + c0 * m.res, m.y0 - r0 * m.res, m.res, c1 - c0, r1 - r0)


def _bilinear(a: np.ndarray, sx: np.ndarray, sy: np.ndarray, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    """(time, sy, sx) -> (time, ty, tx): bilinear, source x west -> east and y north -> south, regular."""
    fx = np.clip((tx - sx[0]) / (sx[1] - sx[0]), 0, len(sx) - 1)
    fy = np.clip((sy[0] - ty) / (sy[0] - sy[1]), 0, len(sy) - 1)
    ix, iy = np.minimum(fx.astype(int), len(sx) - 2), np.minimum(fy.astype(int), len(sy) - 2)
    wx, wy = (fx - ix).astype(np.float32)[None, None, :], (fy - iy).astype(np.float32)[None, :, None]
    top, bottom = a[:, iy], a[:, iy + 1]
    return ((top[:, :, ix] * (1 - wx) + top[:, :, ix + 1] * wx) * (1 - wy)
            + (bottom[:, :, ix] * (1 - wx) + bottom[:, :, ix + 1] * wx) * wy).astype(np.float32)


def _period(meta: dict) -> str:
    return f"{meta['first_utc'][:16]}Z to {meta['last_utc'][:16]}Z"


def _weather(grid: Grid, start: pd.Timestamp, end: pd.Timestamp, requested: str) -> tuple[dict, pd.DatetimeIndex, dict]:
    """Hourly weather on `grid` for hours in [start's hour, end): ({variable: (time, y, x)}, times, attrs)."""
    have = index().get("weather") or {}
    order = _weather_order(requested, start, end)
    first = start.floor("h").tz_localize(None)
    last = end.tz_localize(None)
    for source in order[:1]:          # files_for() fetched this one's files
        meta = have[source]
        parts = [_npz(f) for f in _weather_files(source, start, end)]
        if not parts:
            continue
        t = pd.to_datetime(np.concatenate([p["time_s"] for p in parts]), unit="s").as_unit("ns")
        keep = (t >= first) & (t < last)
        if not keep.any():
            continue
        sx, sy = np.asarray(meta["grid"]["x"]), np.asarray(meta["grid"]["y"])
        out = {v: _bilinear(np.concatenate([p[v] for p in parts])[keep], sx, sy, grid.x, grid.y)
               for v in meta["variables"]}
        out["wind_speed_ms"], out["wind_dir_deg"] = wind_speed_dir(out["wind_u_ms"], out["wind_v_ms"])
        attrs = {"weather_source": source, "weather_dataset": meta["dataset"]}
        if requested == "meteomatics":
            _notes.append("Meteomatics needs an account, so it is not part of the downloaded data: "
                          f"{meta['dataset']} was used instead")
        elif requested == "rtma":
            _notes.append(f"RTMA covers the United States only: {meta['dataset']} was used instead")
        if requested not in ("auto", source):
            attrs["weather_source_requested"] = requested
        return out, t[keep], attrs
    periods = "; ".join(f"{s}: {_period(m)}" for s, m in have.items()
                        if requested not in ("era5_land", "gfs") or s == requested) or "none"
    raise LookupError(f"no downloaded weather between {start:%Y-%m-%d %H:%M} and {end:%Y-%m-%d %H:%M} UTC. This "
                      f"site has weather for these times only ({periods}): choose a Start inside them")


def fire_inputs(polygon, start, *, hours: float = 6.0, radius_km: float = 5.0, res_m: float = 100.0,
                weather_source: str = "auto") -> xr.Dataset:
    """What geelayers.get_fire_inputs(layers=firespread.MODEL_LAYERS) returns, from the downloaded files."""
    t_start = _utc(start)
    t_end = t_start + pd.Timedelta(hours=hours)
    if t_end <= t_start:
        raise ValueError(f"end ({t_end}) must be after start ({t_start})")
    polygon, perimeter_meta = resolve_perimeter(polygon, t_start, None)
    grid = _fire_grid(polygon, radius_km, res_m)
    mask = grid.polygon_mask(polygon)
    if not mask.any():
        raise ValueError("the fire is outside the grid: no burning cell. Check lon/lat order ([lon, lat] in GeoJSON)")
    rows, cols = _window(grid)
    static = _npz("static.npz")
    elev = static["elevation_m"][rows, cols].astype(np.float32)
    landcover = static["landcover"][rows, cols].astype(np.int16)
    slope, aspect = slope_aspect(elev, grid.res)
    lat, lon = grid.latlon()
    dims = ("y", "x")
    ds = xr.Dataset(
        {"elevation_m": (dims, elev), "landcover": (dims, landcover),
         # LANDFIRE (US only) has nothing here, as on the live site outside the US: land cover decides the fuel
         "evc": (dims, np.full(grid.shape, -1, np.int16)),
         "fuel_lifeform": (dims, np.full(grid.shape, -1, np.int16)),
         "cover_pct": (dims, np.full(grid.shape, np.nan, np.float32)),
         "slope_deg": (dims, slope), "aspect_deg": (dims, aspect), "initial_burned": (dims, mask)},
        coords={"y": ("y", grid.y, {"units": "m"}), "x": ("x", grid.x, {"units": "m"}),
                "lat": (dims, lat.astype(np.float32)), "lon": (dims, lon.astype(np.float32))},
        attrs={"crs": grid.crs, "res_m": grid.res, "transform": list(grid.transform),
               "time_utc": t_start.isoformat(), "end_utc": t_end.isoformat(),
               **{f"dataset_{k}": v for k, v in index().get("datasets", {}).items()},
               "dataset_vegetation_cover": "LANDFIRE/Vegetation/EVC/v1_4_0 (skipped: outside CONUS)",
               **perimeter_meta})
    weather, times, attrs = _weather(grid, t_start, t_end, weather_source)
    ds = ds.assign_coords(time=times)
    for v, a in weather.items():
        ds[v] = (("time", "y", "x"), a, {"units": WEATHER_UNITS.get(v, "")})
    ds.attrs.update(attrs)
    return ds


def attach_burn_scars(inputs: xr.Dataset, start) -> xr.Dataset:
    """s2_burned (Sentinel-2 dNBR burn scars, firespread.burn_scars) from the latest snapshot at or before start."""
    snap = _snapshot("s2", _utc(start), strict=False)
    out = inputs.copy()
    if snap is None:
        out.attrs["s2_burn_scars"] = "skipped: no downloaded Sentinel-2 snapshot at or before the start"
        return out
    rows, cols = _window(_grid_of(inputs))
    burned = _npz(snap["file"])["s2_burned"][rows, cols].astype(np.int16)
    out["s2_burned"] = (("y", "x"), burned,
                        {"long_name": "Sentinel-2 burn scar (1), not (0), no clear observation (-1)"})
    out.attrs.update({**snap.get("attrs", {}), "s2_snapshot_utc": snap["time_utc"],
                      "s2_clear_pct": f"{100 * (burned != -1).mean():.0f}",
                      "s2_burned_pct": f"{100 * (burned == 1).mean():.1f}"})
    return out


def _grid_of(ds) -> Grid:
    m = master_grid()
    x, y = np.asarray(ds.x), np.asarray(ds.y)
    return Grid(m.epsg, float(x[0] - m.res / 2), float(y[0] + m.res / 2), m.res, x.size, y.size)


def nisar_provider(request) -> xr.Dataset | None:
    """firespread NISAR provider: the snapshot of the latest pass before the start (within the lookback), cut to
    the request grid. None when there is none or it does not cover the grid."""
    snap = _nisar_snapshot(_utc(request.time_utc))
    if snap is None:
        return None
    meta = index()["nisar"]
    rows, cols = _window(_grid_of(request))
    z = _npz(snap["file"])
    burned = z["nisar_burned"][rows, cols].astype(np.int16)
    if not (burned != -1).any():
        return None
    data = {"nisar_burned": (("y", "x"), burned)}
    for v, scale in meta["scale"].items():
        q = z[v][rows, cols]
        data[v] = (("y", "x"), np.where(q == meta["nodata"], np.nan, q / scale).astype(np.float32))
    passes = "; ".join(f"{p['time_utc'][:16]}Z {p['track']}" for p in snap.get("passes", [])) \
        or f"{snap['time_utc'][:16]}Z {snap['track']}"
    attrs = {**snap.get("attrs", {}), "crs": request.crs, "nisar_passes_used": passes,
             "nisar_coverage_pct": float(100 * (burned != -1).mean())}
    return xr.Dataset(data, coords={"y": ("y", np.asarray(request.y)), "x": ("x", np.asarray(request.x))},
                      attrs=attrs)


def detections_provider():
    """firespread detections provider: the downloaded VIIRS detections inside the box."""
    if not index().get("firms"):
        raise RuntimeError("no VIIRS detections were downloaded for this site (no FIRMS_MAP_KEY when its data "
                           "was built)")
    firms = json.loads((DATA_DIR / "firms.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(firms["rows"], columns=firms["columns"])

    def get(bbox, start):
        w, s, e, n = bbox
        return df[(df["lon"] >= w) & (df["lon"] <= e) & (df["lat"] >= s) & (df["lat"] <= n)]
    return get


def viirs_points(bbox, start, lookback_h: float) -> pd.DataFrame:
    """Stands in for firespread.active_front.firms_viirs_points: the downloaded VIIRS detections in the box over
    (start - lookback_h, start]. RuntimeError when none were downloaded (reported as a note, not a failure)."""
    pts = detections_provider()(bbox, start)
    t1 = _utc(start)
    when = pd.to_datetime(pts["time_utc"], utc=True)
    return pts[(when > t1 - pd.Timedelta(hours=lookback_h)) & (when <= t1)].reset_index(drop=True)


def static_predict(polygon=None, start=None, *, nisar=None, burn_scars: bool = True, radius_km: float = 5.0,
                   res_m: float = 100.0, weather_source: str = "auto", hours: float = 6.0,
                   active_front: bool | None = True, lookback_h: float | None = None,
                   notes: list[str] | None = None, **_newer) -> xr.Dataset:
    """firespread.predict() with the inputs from the downloaded files.

    `_newer`: arguments a later firespread.web may add; they are ignored here instead of failing the prediction."""
    from firespread import attach_nisar, spread_probabilities
    inputs = fire_inputs(polygon, start, hours=hours, radius_km=radius_km, res_m=res_m, weather_source=weather_source)
    if burn_scars:
        inputs = attach_burn_scars(inputs, start)
    inputs = attach_nisar(inputs, nisar)
    if active_front is not False:
        try:      # the CA spreads from the cells VIIRS saw burning (firespread.active_front), when there are any
            from firespread.active_front import DEFAULT_LOOKBACK_H, attach_active_front
        except ImportError:                 # an older firespread without it: the whole polygon spreads
            pass
        else:
            inputs = attach_active_front(inputs, start, lookback_h=lookback_h or DEFAULT_LOOKBACK_H, notes=notes,
                                         points_fn=viirs_points)
    return spread_probabilities(inputs, time=start)


# ---- the two endpoints --------------------------------------------------------------------------

def predict(body: dict) -> dict:
    """POST /api/predict (same answer as backend/server.py)."""
    from firespread.web import predict_payload
    _notes.clear()
    out = predict_payload(body, predict_fn=static_predict, nisar_factory=lambda: nisar_provider,
                          detections_factory=detections_provider)
    out["summary"]["notes"] = [*_notes, *out["summary"]["notes"]]
    return out


def describe_inputs(ds: xr.Dataset) -> str:
    """The table backend/server.py prints for Extract Data: min / mean / max / % missing per variable."""
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    res = float(ds.attrs["res_m"])
    burning = int(ds["initial_burned"].values.sum())
    lines = [f"grid      {ny} x {nx} cells at {res:g} m ({ny * res / 1000:.1f} x {nx * res / 1000:.1f} km), "
             f"{ds.attrs['crs']}",
             f"time      {ds.attrs['time_utc']} -> {ds.attrs['end_utc']}, {ds.sizes.get('time', 0)} weather hours"
             f" ({ds.attrs.get('weather_dataset', 'no weather')})",
             f"burning   {burning} cells ({burning * res * res / 1e6:,.1f} km²) at start",
             f"{'variable':<20} {'units':<8} {'min':>9} {'mean':>9} {'max':>9}  missing"]
    for name, var in ds.data_vars.items():
        if name == "initial_burned":
            continue
        v = var.values.astype(np.float64)
        if np.issubdtype(var.dtype, np.integer):
            v = np.where(v < 0, np.nan, v)          # -1 = no data for categorical layers
        ok = np.isfinite(v)
        missing = f"{100 * (1 - ok.mean()):.0f}%"
        units = var.attrs.get("units") or UNITS.get(name, "")
        if not ok.any():
            lines.append(f"{name:<20} {units:<8} {'-':>9} {'-':>9} {'-':>9}  {missing}")
        else:
            lines.append(f"{name:<20} {units:<8} {np.nanmin(v):>9.2f} {np.nanmean(v):>9.2f} {np.nanmax(v):>9.2f}"
                         f"  {missing}")
    for key in ("nisar_passes_used", "s2_snapshot_utc"):
        if key in ds.attrs:
            lines.append(f"{key}: {ds.attrs[key]}")
    return "\n".join(lines)


def extract(feature: dict) -> dict:
    """POST /api/polygon (Extract Data): the model inputs around the polygon, as a table."""
    from firespread import attach_nisar
    _notes.clear()
    if (feature.get("geometry") or {}).get("type") != "Polygon":
        raise ValueError("geometry must be a Polygon")
    props = feature.get("properties") or {}
    start = props.get("start")
    if not start:
        raise ValueError("start: a UTC time like 2026-08-09T21:00Z is required")
    ds = fire_inputs(feature, start, hours=float(props.get("hours") or EXTRACT_HOURS),
                     weather_source=str(props.get("weather_source") or "auto"))
    ds = attach_nisar(attach_burn_scars(ds, start), nisar_provider)
    ring = feature["geometry"]["coordinates"][0]
    text = f"Polygon with {len(ring) - 1} points" + "".join(f"\nNote: {n}" for n in _notes) + "\n" + describe_inputs(ds)
    return {"ok": True, "text": text}


def handle(path: str, body_json: str) -> str:
    """One request of the website -> the status code, a newline, then the JSON body; with the status codes of
    backend/server.py."""
    def answer(status: int, body: dict) -> str:
        return str(status) + chr(10) + json.dumps(body)
    try:
        body = json.loads(body_json)
        if path == "/api/predict":
            return answer(200, _finite(predict(body)))
        if path == "/api/polygon":
            return answer(200, extract(body))
        return answer(404, {"error": "not found"})
    except ValueError as e:
        return answer(400, {"error": f"bad request: {e}"})
    except Exception as e:
        if type(e) is LookupError:       # no downloaded data for that time (not KeyError / IndexError: those are bugs)
            return answer(404, {"error": str(e)})
        what = "predict" if path == "/api/predict" else "get_fire_inputs"
        return answer(500, {"error": f"{what} failed: {type(e).__name__}: {e}"})


def _finite(obj):
    """NaN / inf -> None anywhere in a JSON-ready structure (json.dumps would write NaN, which is not JSON)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj
