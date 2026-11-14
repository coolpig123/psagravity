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

import ee
import numpy as np
import pandas as pd
import xarray as xr

from . import fetch
from .auth import init
from .derive import slope_aspect, wind_speed_dir
from .grid import Grid
from .layers import LAYERS, VARIABLES, WEATHER

DEFAULT_LAYERS = ("terrain", "landcover", "fuel_model", "vegetation", "fuel_dryness", "weather")


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


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

    source: "rtma" (US, observed analysis, best for hindcasts), "era5_land" (global reanalysis,
    a few days behind), "gfs" (forecast: use for live predictions; `issued` picks the run,
    default = latest run at or before `start`).
    Weather is smooth, so a coarse grid is enough and much faster: get_weather(grid.with_resolution(1000), ...),
    then regrid(weather, grid) to put it on the model grid.
    """
    init()
    if source not in WEATHER:
        raise ValueError(f"unknown weather source {source!r}; choose from {list(WEATHER)}")
    src = WEATHER[source]
    start, end = _utc(start), _utc(end)
    col, meta = src.collection(fetch.region(grid), start, end, issued)
    if step_hours > 1:
        wanted = pd.date_range(start.ceil("h"), end, freq=f"{step_hours}h", inclusive="left")
        col = col.filter(ee.Filter.inList("valid_time", [int(t.timestamp() * 1000) for t in wanted]))
    col = col.distinct("valid_time").sort("valid_time")
    times_ms = col.aggregate_array("valid_time").getInfo()
    if not times_ms:
        raise LookupError(f"{source}: no weather between {start} and {end} here. The dataset may not reach this "
                          f"date yet ({src.description}) or not cover this area ({src.coverage}).")
    names = [f"t{i}_{v}" for i in range(len(times_ms)) for v in src.variables]
    stack = col.toBands().rename(names)
    arr = fetch.download(stack, grid, names).reshape(len(times_ms), len(src.variables), *grid.shape)

    ds = _empty_dataset(grid)
    ds = ds.assign_coords(time=pd.to_datetime(times_ms, unit="ms"))   # UTC, tz-naive (netCDF-friendly)
    for k, v in enumerate(src.variables):
        ds[v] = (("time", "y", "x"), arr[:, k], _attrs(v))
    ds = _add_wind(ds)
    ds.attrs.update({"weather_source": source, "weather_dataset": src.dataset, **meta})
    return ds


def get_layers(grid: Grid, time=None, layers: Iterable[str] = DEFAULT_LAYERS,
               weather_source: str = "rtma") -> xr.Dataset:
    """All requested layers on `grid` as of `time` (UTC), dims (y, x).

    `time` is the moment the prediction starts (e.g. the ignition time). Time-dependent layers
    only use data from before it: vegetation = 30 days before, fuel_dryness = latest day before,
    weather = latest hour at or before `time`. Static layers (terrain, landcover, fuel_model)
    don't need a time.
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
    images, names, categorical = [], [], set()
    for name in layers:
        if name == "weather":
            continue
        layer = LAYERS[name]
        img, meta = layer.build(region, t)
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
