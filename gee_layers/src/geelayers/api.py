"""Public functions: fetch layers onto a Grid and get them back as xarray Datasets.

Output contract (same for every function):
- dims ("y", "x") or ("time", "y", "x"); row 0 = north, column 0 = west
- coords x, y (cell centres, metres in ds.attrs["crs"]), lat / lon (2-D, degrees), time (UTC, naive)
- one float32 variable per quantity, named and in the units listed in layers.VARIABLES
  (categorical ones are int16 with -1 = no data); missing data = NaN
- attrs: crs, res_m, transform, datasets used, and the dates actually used
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from . import fetch
from ._ee_compat import ee
from .auth import init
from .derive import decode_evc, slope_aspect, wind_speed_dir
from .grid import Grid
from .layers import LAYERS, VARIABLES, WEATHER
from .meteomatics import fetch_meteomatics_weather, has_credentials as has_meteomatics_credentials
from .perimeter import resolve_perimeter

DEFAULT_LAYERS = ("terrain", "landcover", "vegetation_cover", "vegetation", "fuel_dryness", "weather")
CONUS_BBOX = (-125.0, 24.0, -66.0, 50.0)   # west, south, east, north (contiguous US)
ERA5_LATENCY = pd.Timedelta(days=6)         # ERA5-Land reanalysis lags real time by ~5-6 days


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def in_conus(grid: Grid) -> bool:
    """True if `grid` overlaps the contiguous United States bounding box."""
    w, s, e, n = grid.bbox_lonlat()
    cw, cs, ce, cn = CONUS_BBOX
    return e > cw and w < ce and n > cs and s < cn


def _weather_candidates(grid: Grid, start: pd.Timestamp, end: pd.Timestamp, source: str,
                        now: pd.Timestamp | None = None) -> list[str]:
    """Ordered weather sources to try.

    When `source` is "auto" or "rtma", falls back to global sources (`era5_land` for historical
    windows older than ERA5_LATENCY, `gfs` for recent / forecast windows) whenever outside CONUS
    or when RTMA has no coverage for the requested time/location.
    """
    if source not in WEATHER and source != "auto":
        raise ValueError(f"unknown weather source {source!r}; choose from {list(WEATHER) + ['auto']}")
    if source not in ("auto", "rtma"):
        return [source]
    now_utc = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    global_order = ["era5_land", "gfs"] if end <= now_utc - ERA5_LATENCY else ["gfs", "era5_land"]
    if in_conus(grid) and end <= now_utc:
        return ["rtma", *global_order]
    return global_order


def _empty_dataset(grid: Grid) -> xr.Dataset:
    lat, lon = grid.latlon()
    return xr.Dataset(
        coords={"y": ("y", grid.y, {"units": "m"}), "x": ("x", grid.x, {"units": "m"}),
                "lat": (("y", "x"), lat.astype(np.float32), {"units": "degrees_north"}),
                "lon": (("y", "x"), lon.astype(np.float32), {"units": "degrees_east"})},
        attrs={"crs": grid.crs, "res_m": grid.res, "transform": list(grid.transform),
               "created_utc": pd.Timestamp.now(tz="UTC").isoformat()})


def _attrs(name: str) -> dict:
    units, description = VARIABLES[name]
    return {"units": units, "description": description}


def _add_wind(ds: xr.Dataset) -> xr.Dataset:
    speed, direction = wind_speed_dir(ds["wind_u_ms"].values, ds["wind_v_ms"].values)
    dims = ds["wind_u_ms"].dims
    ds["wind_speed_ms"] = (dims, speed, _attrs("wind_speed_ms"))
    ds["wind_dir_deg"] = (dims, direction, _attrs("wind_dir_deg"))
    return ds


def get_weather(grid: Grid, start, end, source: str = "rtma", step_hours: int = 1,
                issued=None) -> xr.Dataset:
    """Hourly weather on `grid` for valid times in [start, end), dims (time, y, x).

    source: "rtma" (US, observed analysis, best for hindcasts; auto-falls back outside CONUS),
    "era5_land" (global reanalysis, a few days behind), "gfs" (forecast: use for live predictions;
    `issued` picks the run, default = latest run at or before `start`), "meteomatics" (1 km
    historical, real-time and forecast weather via Meteomatics API), or "auto".
    Weather is smooth, so a coarse grid is enough and much faster: get_weather(grid.with_resolution(1000), ...),
    then regrid(weather, grid) to put it on the model grid.
    """
    start, end = _utc(start), _utc(end)
    candidates = _weather_candidates(grid, start, end, source)
    if candidates == ["meteomatics"]:
        return fetch_meteomatics_weather(
            grid, start, end, step_hours=step_hours, var_attrs={v: _attrs(v) for v in VARIABLES}
        )
    if source in ("auto", "rtma") and has_meteomatics_credentials():
        # Prefer high-resolution Meteomatics for future forecasts when credentials are configured,
        # and keep it as a fallback for historical windows.
        now_utc = pd.Timestamp.now(tz="UTC")
        if end > now_utc:
            candidates = ["meteomatics", *candidates]
        else:
            candidates = [*candidates, "meteomatics"]

    init()
    reg = fetch.region(grid)
    chosen = None
    times_ms: list[int] = []
    col = None
    meta: dict[str, str] = {}
    last_err: Exception | None = None
    for cand in candidates:
        if cand == "meteomatics":
            try:
                ds_mm = fetch_meteomatics_weather(
                    grid, start, end, step_hours=step_hours, var_attrs={v: _attrs(v) for v in VARIABLES}
                )
                if source != "meteomatics" and source != "auto":
                    ds_mm.attrs["weather_source_requested"] = source
                return ds_mm
            except Exception as exc:
                last_err = exc
                continue
        src = WEATHER[cand]
        try:
            c, m = src.collection(reg, start, end, issued)
            if step_hours > 1:
                wanted = pd.date_range(start.ceil("h"), end, freq=f"{step_hours}h", inclusive="left")
                c = c.filter(ee.Filter.inList("valid_time", [int(t.timestamp() * 1000) for t in wanted]))
            c = ee.ImageCollection(c.distinct("valid_time")).sort("valid_time")
            t_ms = c.aggregate_array("valid_time").getInfo()
        except LookupError as exc:
            last_err = exc
            continue
        if t_ms:
            chosen, col, meta, times_ms = cand, c, m, t_ms
            break
    if chosen is None or col is None:
        if last_err is not None and len(candidates) == 1:
            raise last_err
        tried = ", ".join(f"{c} ({WEATHER[c].coverage})" for c in candidates)
        raise LookupError(f"{source}: no weather between {start} and {end} here (tried {tried}).")

    src = WEATHER[chosen]
    names = [f"t{i}_{v}" for i in range(len(times_ms)) for v in src.variables]
    stack = col.toBands().rename(names)
    arr = fetch.download(stack, grid, names).reshape(len(times_ms), len(src.variables), *grid.shape)

    ds = _empty_dataset(grid)
    ds = ds.assign_coords(time=pd.to_datetime(times_ms, unit="ms"))   # UTC, tz-naive (netCDF-friendly)
    for k, v in enumerate(src.variables):
        ds[v] = (("time", "y", "x"), arr[:, k], _attrs(v))
    ds = _add_wind(ds)
    ds.attrs.update({"weather_source": chosen, "weather_dataset": src.dataset, **meta})
    if chosen != source and source != "auto":
        ds.attrs["weather_source_requested"] = source
    return ds


def get_layers(grid: Grid, time=None, layers: Iterable[str] = DEFAULT_LAYERS,
               weather_source: str = "rtma") -> xr.Dataset:
    """All requested layers on `grid` as of `time` (UTC), dims (y, x).

    `time` is the moment the prediction starts (e.g. the ignition time). Time-dependent layers
    only use data from before it: vegetation = 30 days before, fuel_dryness = latest day before,
    weather = latest hour at or before `time`. Static layers (terrain, landcover, vegetation_cover)
    don't need a time. CONUS-only layers (`vegetation_cover`, `fuel_dryness`) are skipped cleanly
    with NaN / -1 placeholders when `grid` is outside the US.
    """
    init()
    layers = list(layers)
    unknown = set(layers) - set(LAYERS) - {"weather"}
    if unknown:
        raise ValueError(f"unknown layers {sorted(unknown)}; choose from {list(LAYERS) + ['weather']}")
    t = _utc(time) if time is not None else None
    if t is None and any(name == "weather" or LAYERS[name].needs_time for name in layers):
        raise ValueError("these layers need a time: get_layers(grid, time='2026-08-08T22:30Z', ...)")

    ds = _empty_dataset(grid)
    if t is not None:
        ds.attrs["time_utc"] = t.isoformat()
    region = fetch.region(grid)
    conus = in_conus(grid)
    images, names, categorical, skipped = [], [], set(), []
    for name in layers:
        if name == "weather":
            continue
        layer = LAYERS[name]
        if layer.coverage == "CONUS" and not conus:
            skipped.append((name, layer, "outside CONUS"))
            continue
        try:
            img, meta = layer.build(region, t)
        except LookupError:
            if layer.coverage == "CONUS":
                skipped.append((name, layer, "no CONUS data at location"))
                continue
            raise
        images.append(img.select(list(layer.variables)))
        names += layer.variables
        if layer.categorical:
            categorical |= set(layer.variables)
        ds.attrs.update(meta)
        ds.attrs[f"dataset_{name}"] = layer.dataset

    if images:
        arr = fetch.download(ee.Image.cat(images), grid, names)
        for k, v in enumerate(names):
            if v in categorical:
                ds[v] = (("y", "x"), np.where(np.isnan(arr[k]), -1, np.round(arr[k])).astype(np.int16), _attrs(v))
            else:
                ds[v] = (("y", "x"), arr[k], _attrs(v))
    for name, layer, reason in skipped:
        for v in layer.variables:
            if layer.categorical:
                ds[v] = (("y", "x"), np.full(grid.shape, -1, np.int16), _attrs(v))
            else:
                ds[v] = (("y", "x"), np.full(grid.shape, np.nan, np.float32), _attrs(v))
        ds.attrs[f"dataset_{name}"] = f"{layer.dataset} (skipped: {reason})"
    if "evc" in ds:
        lifeform, cover = decode_evc(ds["evc"].values)
        ds["fuel_lifeform"] = (("y", "x"), lifeform, _attrs("fuel_lifeform"))
        ds["cover_pct"] = (("y", "x"), cover, _attrs("cover_pct"))
    if "elevation_m" in ds:
        slope, aspect = slope_aspect(ds["elevation_m"].values, grid.res)
        ds["slope_deg"] = (("y", "x"), slope, _attrs("slope_deg"))
        ds["aspect_deg"] = (("y", "x"), aspect, _attrs("aspect_deg"))

    if "weather" in layers:
        w = get_weather(grid, t - pd.Timedelta(hours=3), t + pd.Timedelta(seconds=1), source=weather_source)
        for v in w.data_vars:
            ds[v] = w[v].isel(time=-1, drop=True)
        ds.attrs.update({k: v for k, v in w.attrs.items() if k.startswith(("weather_", "gfs_"))})
        ds.attrs["weather_time_utc"] = pd.Timestamp(w.time.values[-1]).tz_localize("UTC").isoformat()
    return ds


def regrid(ds: xr.Dataset, grid: Grid) -> xr.Dataset:
    """Bilinear-interpolate a dataset (e.g. coarse weather) onto another grid in the same CRS.

    Wind is interpolated as u, v and speed / direction recomputed (interpolating angles is wrong).
    """
    if ds.attrs.get("crs") != grid.crs:
        raise ValueError(f"CRS differs: {ds.attrs.get('crs')} vs {grid.crs}; build both grids from the same one "
                         "with grid.with_resolution(...)")
    src = ds.drop_vars(["lat", "lon", "wind_speed_ms", "wind_dir_deg"], errors="ignore")
    out = src.interp(x=grid.x, y=grid.y, method="linear", kwargs={"fill_value": "extrapolate"})   # edge cells
    lat, lon = grid.latlon()
    out = out.assign_coords(lat=(("y", "x"), lat.astype(np.float32)), lon=(("y", "x"), lon.astype(np.float32)))
    for v in out.data_vars:
        out[v] = out[v].astype(np.float32)
    out.attrs.update({"res_m": grid.res, "transform": list(grid.transform)})
    return _add_wind(out) if "wind_u_ms" in out else out


def get_fire_inputs(
    grid: Grid | None = None,
    start=None,
    end=None,
    *,
    hours: float = 24.0,
    polygon=None,
    point: tuple[float, float] | None = None,
    ignition_radius_m: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
    radius_km: float = 15.0,
    res_m: float = 100.0,
    weather_res_m: float = 1000.0,
    step_hours: int = 1,
    layers: Iterable[str] = DEFAULT_LAYERS,
    weather_source: str = "rtma",
    issued=None,
    perimeter_time_key: str | None = None,
) -> xr.Dataset:
    """One aligned Dataset for a spread run: layers at `start`, `initial_burned` mask, hourly weather.

    The fire (exactly one): `polygon` = the area burning at `start` (list of (lon, lat), GeoJSON
    dict / string / file, or anything with __geo_interface__, e.g. a shape drawn in the UI), or
    `point=(lat, lon)` = an ignition, optionally with `ignition_radius_m`.
    Several untimed features are all burning (merged). A perimeter time series (every feature has a
    time property, e.g. real_fires FEDS 12 h perimeters) gives the last perimeter observed at or
    before `start` (see perimeter.resolve_perimeter); which one is recorded in attrs.

    The area fetched is only the fire plus `radius_km` around it (or `grid` / `bbox` if given):
    nothing outside that box is downloaded from Earth Engine.
    """
    if start is None:
        raise ValueError("start time is required (e.g. start='2026-08-08T22:30Z')")
    if (polygon is None) == (point is None):
        raise ValueError("give exactly one fire: polygon=... (area burning at start) or point=(lat, lon) (ignition)")
    t_start = _utc(start)
    t_end = _utc(end) if end is not None else t_start + pd.Timedelta(hours=hours)
    if t_end <= t_start:
        raise ValueError(f"end ({t_end}) must be after start ({t_start})")

    perimeter_meta: dict[str, str] = {}
    if polygon is not None:
        polygon, perimeter_meta = resolve_perimeter(polygon, t_start, perimeter_time_key)

    if grid is None:
        if bbox is not None:
            grid = Grid.from_bbox(*bbox, res_m=res_m)
        elif polygon is not None:
            grid = Grid.from_polygon(polygon, buffer_km=radius_km, res_m=res_m)
        else:
            grid = Grid.from_point(point[0], point[1], radius_km=radius_km, res_m=res_m)

    if polygon is not None:
        mask = grid.polygon_mask(polygon)
    else:
        mask = grid.point_mask(point[0], point[1], radius_m=ignition_radius_m)
    if not mask.any():
        raise ValueError("the fire is outside the grid: no burning cell. Check lon/lat order ([lon, lat] in GeoJSON) "
                         "or the grid / bbox.")

    req_layers = list(layers)
    want_weather = "weather" in req_layers
    env_layers = [l for l in req_layers if l != "weather"]

    ds = get_layers(grid, time=t_start, layers=env_layers, weather_source=weather_source) if env_layers else _empty_dataset(grid)
    ds.attrs["time_utc"] = t_start.isoformat()
    ds.attrs["end_utc"] = t_end.isoformat()
    ds.attrs.update(perimeter_meta)
    ds["initial_burned"] = (("y", "x"), mask, _attrs("initial_burned"))

    if want_weather:
        wx_grid = grid.with_resolution(weather_res_m) if weather_res_m and weather_res_m != grid.res else grid
        # Start at the top of the hour so the first weather step covers `start` (22:00 for a 22:30 start).
        # That hour is observed before `start`, so nothing from the future leaks in. GFS keeps the run
        # issued at or before `start` itself.
        wx = get_weather(wx_grid, t_start.floor("h"), t_end, source=weather_source, step_hours=step_hours,
                         issued=issued if issued is not None else t_start)
        if wx_grid != grid:
            wx = regrid(wx, grid)
        ds = ds.assign_coords(time=wx.coords["time"])
        for v in wx.data_vars:
            ds[v] = wx[v]
        ds.attrs.update({k: v for k, v in wx.attrs.items() if k.startswith(("weather_", "gfs_"))})
    return ds


def describe() -> str:
    """Human-readable list of the available layers, weather sources and variables."""
    lines = ["Layers (get_layers):"]
    for l in LAYERS.values():
        lines.append(f"  {l.name:13s} {l.coverage:16s} ~{l.native_res_m:>6.0f} m  {l.dataset}  -> {', '.join(l.variables)}")
    lines.append(f"  {'weather':13s} see sources below (latest hour at or before `time`)")
    lines.append("Weather sources (get_weather / weather_source=):")
    for w in WEATHER.values():
        lines.append(f"  {w.name:13s} {w.coverage:16s} ~{w.native_res_m:>6.0f} m  {w.dataset}  -> {', '.join(w.variables)}")
    lines.append("Variables:")
    for name, (units, desc) in VARIABLES.items():
        lines.append(f"  {name:16s} [{units}] {desc}")
    return "\n".join(lines)
