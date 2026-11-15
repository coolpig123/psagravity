"""Download everything the website needs around Haifa, once, into frontend/public/haifa/ (committed).

The published site (GitHub Pages) has no Python server and no credentials: the browser reads these files
and runs the same spread model (firespread) in Pyodide. See israel_static/README.md.

Run from the repo root with the backend venv (needs the logins of frontend/README.md):

    backend\\.venv\\Scripts\\python israel_static\\scripts\\build_haifa_data.py all
    backend\\.venv\\Scripts\\python israel_static\\scripts\\build_haifa_data.py static firms wind era5 manifest

Steps (each can be re-run; downloads are cached in israel_static/data/raw/, never committed):
    static    elevation (NASADEM) + land cover (ESA WorldCover) on the 100 m master grid      Earth Engine
    era5      hourly ERA5-Land weather on a 4 km grid                                          Earth Engine
    gfs       hourly GFS weather (first 6 h of each 6-hourly run, then the last run's forecast) Earth Engine
    wind      hourly MERRA-2 10 m wind around Israel, for the wind animation                   GES DISC
    firms     VIIRS active-fire detections (the active front)                                  NASA FIRMS
    s2        Sentinel-2 burn scars (dNBR), one snapshot every S2_EVERY_DAYS days              Earth Engine
    nisar     NISAR cover / soil moisture / vegetation water / burned, one snapshot per pass   ASF
    manifest  index.json: what is there, for the browser
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
for _pkg in ("spread_model", "real_fires", "gee_layers"):
    sys.path.append(str(ROOT / _pkg / "src"))

OUT = ROOT / "frontend" / "public" / "haifa"
RAW = ROOT / "israel_static" / "data" / "raw"
PARTS = ROOT / "israel_static" / "data" / "parts"      # what each step downloaded; the manifest step merges them

NAME = "Haifa"
BBOX = (34.85, 32.55, 35.25, 32.95)          # west, south, east, north: Haifa, the Carmel and the bay
CENTER = (35.0, 32.76)                        # lon, lat the map opens on
RES_M = 100.0
WEATHER_RES_M = 4000.0                        # ERA5-Land is ~11 km and GFS ~28 km: 4 km loses nothing
START = pd.Timestamp("2026-06-01T00:00Z")     # the 2026 fire season
WIND_BBOX = (31.25, 29.0, 38.75, 36.5)        # MERRA-2 cells (~55 km) around Israel, for the wind animation
S2_EVERY_DAYS = 5
NISAR_LOOKBACK_DAYS = 30.0                    # as firedata.nisar_provider


def master_grid():
    from geelayers.grid import Grid
    return Grid.from_bbox(*BBOX, res_m=RES_M)


def weather_grid():
    from geelayers.grid import Grid
    g = master_grid()
    xmin, ymin, xmax, ymax = g.bounds
    half = WEATHER_RES_M / 2      # cell centres reach past the master grid on every side: no extrapolation
    return Grid.from_bounds(xmin - half, ymin - half, xmax + half, ymax + half, WEATHER_RES_M, g.epsg)


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def write_json(name: str, data: dict, folder: Path = OUT) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"  wrote {name}")


def write_part(name: str, data: dict) -> None:
    write_json(name, data, PARTS)


def save_npz(path: Path, **arrays) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path.stat().st_size


def iso(t) -> str:
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- static ----------------------------------------------------------------------------------

def step_static() -> None:
    from geelayers import get_layers
    g = master_grid()
    print(f"static: {g.width} x {g.height} cells at {g.res:g} m, {g.crs}")
    ds = get_layers(g, layers=("terrain", "landcover"))
    elev, lc = ds["elevation_m"].values.astype(np.float32), ds["landcover"].values.astype(np.int16)
    size = save_npz(OUT / "static.npz", elevation_m=elev, landcover=lc)
    w, s, e, n = g.bbox_lonlat()
    write_part("static.json", {
        "grid": {"epsg": g.epsg, "x0": g.x0, "y0": g.y0, "res": g.res, "width": g.width, "height": g.height},
        "grid_bbox_lonlat": [round(v, 6) for v in (w, s, e, n)],
        "datasets": {"terrain": ds.attrs["dataset_terrain"], "landcover": ds.attrs["dataset_landcover"]},
        "bytes": size})


# ---- hourly weather (Earth Engine) --------------------------------------------------------------

def _fill_nearest(a: np.ndarray) -> np.ndarray:
    """(time, y, x): cells that are always NaN (ERA5-Land has no sea cells) take the nearest cell that has data,
    so the coast is not left without weather. Cells missing only sometimes stay NaN."""
    from scipy.ndimage import distance_transform_edt
    never = np.isnan(a).all(axis=0)
    if not never.any() or never.all():
        return a
    _, (iy, ix) = distance_transform_edt(never, return_indices=True)
    return a[:, iy, ix]


def _weather_chunk(source: str, start: pd.Timestamp, end: pd.Timestamp, issued=None) -> xr.Dataset | None:
    """One cached get_weather call on the weather grid; None if the source has nothing there."""
    from geelayers import get_weather
    cache = RAW / "weather" / source / f"{start:%Y%m%dT%H}_{end:%Y%m%dT%H}.nc"
    if cache.exists():
        return xr.load_dataset(cache, engine="h5netcdf")
    for attempt in range(4):
        try:
            ds = get_weather(weather_grid(), start, end, source=source, issued=issued)
            break
        except LookupError:
            return None
        except Exception as e:   # transient Earth Engine errors
            if attempt == 3:
                raise
            print(f"    retry {source} {start}: {type(e).__name__}: {e}")
            time.sleep(5 * (attempt + 1))
    ds = ds.drop_vars(["lat", "lon"])
    ds.attrs = {k: v for k, v in ds.attrs.items() if isinstance(v, (str, int, float))}
    cache.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(cache, engine="h5netcdf")
    return ds


def _write_weather(source: str, parts: list[xr.Dataset], extra: dict) -> None:
    ds = xr.concat(parts, dim="time").sortby("time")
    ds = ds.isel(time=~pd.DatetimeIndex(ds.time.values).duplicated())
    ds = ds.drop_vars(["wind_speed_ms", "wind_dir_deg"], errors="ignore")     # recomputed from u, v in the browser
    names = list(ds.data_vars)
    times = pd.DatetimeIndex(ds.time.values)
    months, total = [], 0
    for month, idx in pd.Series(np.arange(len(times)), index=times).groupby(times.strftime("%Y-%m")):
        sel = ds.isel(time=idx.values)
        arrays = {v: np.round(_fill_nearest(sel[v].values.astype(np.float32)), 3) for v in names}
        size = save_npz(OUT / "weather" / source / f"{month}.npz",
                        time_s=pd.DatetimeIndex(sel.time.values).as_unit("s").asi8.astype(np.int64), **arrays)
        months.append(month)
        total += size
    g = weather_grid()
    write_part(f"weather_{source}.json", {
        "source": source, "dataset": parts[0].attrs.get("weather_dataset", ""), "variables": names,
        "first_utc": iso(times[0]), "last_utc": iso(times[-1]), "hours": int(len(times)), "months": months,
        "grid": {"epsg": g.epsg, "x": g.x.tolist(), "y": g.y.tolist(), "res": g.res}, "bytes": total, **extra})
    print(f"  {source}: {len(times)} hours {times[0]} .. {times[-1]}, {total / 1e6:.1f} MB")


def step_era5() -> None:
    edges = pd.date_range(START, now_utc().ceil("D"), freq="5D")
    spans = list(zip(edges[:-1], edges[1:])) + [(edges[-1], edges[-1] + pd.Timedelta(days=5))]
    print(f"era5: {len(spans)} chunks of 5 days from {START}")
    with ThreadPoolExecutor(4) as pool:
        parts = list(pool.map(lambda se: _weather_chunk("era5_land", *se), spans))
    parts = [p for p in parts if p is not None and p.sizes["time"]]
    _write_weather("era5_land", parts, {"note": "ECMWF ERA5-Land hourly reanalysis; sea cells filled from the "
                                                "nearest land cell"})


def step_gfs() -> None:
    runs = pd.date_range(START, now_utc().floor("6h"), freq="6h")
    print(f"gfs: {len(runs)} runs of 6 hours from {START}")
    with ThreadPoolExecutor(8) as pool:
        parts = list(pool.map(lambda r: _weather_chunk("gfs", r, r + pd.Timedelta(hours=6), issued=r), runs))
    got = [(r, p) for r, p in zip(runs, parts) if p is not None and p.sizes["time"]]
    last_run = got[-1][0]
    # the forecast of the last run (hours 6 .. 120), so start times up to a few days ahead also work
    fc = _weather_chunk("gfs", last_run + pd.Timedelta(hours=6), last_run + pd.Timedelta(hours=121), issued=last_run)
    parts = [p for _, p in got] + ([fc] if fc is not None and fc.sizes["time"] else [])
    _write_weather("gfs", parts, {
        "last_run_utc": iso(last_run),
        "note": "NOAA GFS 0.25 deg: hours 0-5 of each 6-hourly run (close to an analysis), then the forecast of "
                f"the last run downloaded ({iso(last_run)})"})


# ---- MERRA-2 wind for the animation ---------------------------------------------------------

def step_wind() -> None:
    from firedata import merra2
    cache = RAW / "merra2"
    granules = merra2.search_granules(START, now_utc())
    lat_idx, lon_idx = merra2.grid_window(WIND_BBOX)
    print(f"wind: {len(granules)} MERRA-2 days from {START:%Y-%m-%d}")
    merra2._token()   # log in once, before the threads
    with ThreadPoolExecutor(6) as pool:
        files = list(pool.map(lambda g: merra2.fetch_day(g, lat_idx, lon_idx, merra2.WIND_VARIABLES, cache),
                              [g for _, g in granules.iterrows()]))
    days = [xr.load_dataset(f, engine="h5netcdf") for f in files]
    ds = xr.concat(days, dim="time").sortby("time").sortby("lat").sortby("lon")
    hours = pd.DatetimeIndex(ds.time.values).floor("h")           # MERRA-2 times are the middle of the hour
    full = pd.date_range(hours[0], hours[-1], freq="h")
    if len(full) != len(hours):
        raise SystemExit(f"wind: {len(full) - len(hours)} hours are missing between {hours[0]} and {hours[-1]}")
    u = np.round(np.nan_to_num(ds.U10M.values) * 100).astype(np.int16)     # cm/s
    v = np.round(np.nan_to_num(ds.V10M.values) * 100).astype(np.int16)
    months, total = [], 0
    for month in sorted(set(hours.strftime("%Y-%m"))):
        sel = hours.strftime("%Y-%m") == month
        # one file per month: int16 u then v, each (hours, lat south->north, lon west->east), little endian
        path = OUT / "wind" / f"{month}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(u[sel].astype("<i2").tobytes() + v[sel].astype("<i2").tobytes())
        months.append({"month": month, "first_hour_utc": iso(hours[sel][0]), "hours": int(sel.sum())})
        total += path.stat().st_size
    write_part("wind.json", {
        "source": f"MERRA-2 {merra2.SHORT_NAME} 10 m wind (NASA GMAO)",
        "resolution": "0.5 deg x 0.625 deg (~55 km), hourly mean",
        "lat": np.round(ds.lat.values, 4).tolist(), "lon": np.round(ds.lon.values, 4).tolist(),
        "first_hour_utc": iso(hours[0]), "last_hour_utc": iso(hours[-1]), "months": months,
        "scale": 0.01, "bytes": total})
    print(f"  wind: {len(hours)} hours {hours[0]} .. {hours[-1]}, {total / 1e6:.1f} MB")


# ---- VIIRS detections ------------------------------------------------------------------------

def step_firms() -> None:
    from firedata import firms
    g = master_grid()
    bbox = [round(v, 4) for v in g.bbox_lonlat()]
    end = now_utc().normalize() + pd.Timedelta(days=1)
    raw = firms.download_viirs(bbox, START - pd.Timedelta(days=1), end, RAW / "firms")
    df = firms.tidy_viirs(raw)
    rows = [[iso(t), round(float(la), 5), round(float(lo), 5), None if pd.isna(f) else round(float(f), 1), s]
            for t, la, lo, f, s in zip(df["time_utc"], df["lat"], df["lon"], df["frp_mw"], df["source"])] \
        if len(df) else []
    meta = {"source": "NASA FIRMS VIIRS 375 m active fire (S-NPP, NOAA-20, NOAA-21)", "bbox": bbox,
            "first_utc": iso(START - pd.Timedelta(days=1)), "last_utc": iso(end), "detections": len(rows)}
    write_json("firms.json", {**meta, "columns": ["time_utc", "lat", "lon", "frp_mw", "source"], "rows": rows})
    write_part("firms.json", meta)
    print(f"  firms: {len(rows)} detections")


# ---- Sentinel-2 burn scars -------------------------------------------------------------------

def _master_inputs(t: pd.Timestamp) -> xr.Dataset:
    g = master_grid()
    return xr.Dataset(coords={"y": ("y", g.y), "x": ("x", g.x)},
                      attrs={"crs": g.crs, "res_m": g.res, "time_utc": t.isoformat()})


def step_s2() -> None:
    from firespread.burn_scars import optical_burn_scars
    times = pd.date_range(START, now_utc(), freq=f"{S2_EVERY_DAYS}D")
    print(f"s2: {len(times)} snapshots, one every {S2_EVERY_DAYS} days")
    snapshots = []

    def one(t):
        cache = RAW / "s2" / f"{t:%Y%m%d}.npz"
        if not cache.exists():
            out = optical_burn_scars(_master_inputs(t), time=t)
            burned = out["s2_burned"].values.astype(np.int8) if "s2_burned" in out else None
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, burned=burned if burned is not None else np.zeros((0, 0), np.int8),
                                attrs=json.dumps({k: str(v) for k, v in out.attrs.items() if k.startswith("s2_")}))
            print(f"    s2 {t:%Y-%m-%d} done", flush=True)
        z = np.load(cache)
        return t, z["burned"], json.loads(str(z["attrs"]))

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(one, times))
    total = 0
    for t, burned, attrs in results:
        if burned.size == 0:
            continue
        name = f"s2/{t:%Y-%m-%d}.npz"
        total += save_npz(OUT / name, s2_burned=burned)
        snapshots.append({"time_utc": iso(t), "file": name, "attrs": attrs})
    write_part("s2.json", {"dataset": "COPERNICUS/S2_SR_HARMONIZED, dNBR > 0.15 (firespread.burn_scars)",
                           "every_days": S2_EVERY_DAYS, "snapshots": snapshots, "bytes": total})
    print(f"  s2: {len(snapshots)} snapshots, {total / 1e6:.1f} MB")


# ---- NISAR ------------------------------------------------------------------------------------

NISAR_SCALE = {"nisar_cover_pct": 100.0, "nisar_soil_moisture": 10000.0, "nisar_vwc_kg_m2": 1000.0}   # -> uint16
NISAR_NODATA = 65535


def step_nisar(limit: int | None = None) -> None:
    from firedata.nisar_access import group_passes, search_granules
    from firedata.nisar_provider import nisar_provider
    from firespread.nisar import NisarRequest
    g = master_grid()
    bbox = g.bbox_lonlat()
    passes = [p for p in group_passes(search_granules("GCOV", bbox, START - pd.Timedelta(days=NISAR_LOOKBACK_DAYS),
                                                      now_utc()))]
    # a snapshot per pass from the last one before START on: the state of NISAR right after that pass
    before = [p for p in passes if p.time_utc < START]
    wanted = before[-1:] + [p for p in passes if p.time_utc >= START]
    if limit:
        wanted = wanted[-limit:]
    print(f"nisar: {len(wanted)} snapshots (one per pass), {wanted[0].time_utc} .. {wanted[-1].time_utc}")
    provider = nisar_provider()
    snapshots, total = [], 0
    for p in wanted:
        t = p.time_utc + pd.Timedelta(minutes=1)
        cache = RAW / "nisar" / f"{p.time_utc:%Y%m%dT%H%M}_{p.track_id}.nc"
        if not cache.exists():
            t0 = time.time()
            req = NisarRequest(crs=g.crs, x=g.x, y=g.y, res_m=g.res, bbox_lonlat=bbox, time_utc=t.isoformat())
            ds = None
            for attempt in range(3):
                try:
                    ds = provider(req)
                    break
                except Exception as e:
                    print(f"    retry nisar {p.time_utc}: {type(e).__name__}: {e}", flush=True)
                    time.sleep(20)
            if ds is None:
                print(f"    nisar {p.time_utc} {p.track_id}: nothing", flush=True)
                continue
            cache.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(cache, engine="h5netcdf")
            print(f"    nisar {p.time_utc} {p.track_id}: {ds.attrs['nisar_coverage_pct']:.0f}% covered, "
                  f"{time.time() - t0:.0f} s", flush=True)
        ds = xr.load_dataset(cache, engine="h5netcdf")
        arrays = {"nisar_burned": ds["nisar_burned"].values.astype(np.int8)}
        for v, scale in NISAR_SCALE.items():
            a = ds[v].values
            q = np.clip(np.round(a * scale), 0, NISAR_NODATA - 1)
            arrays[v] = np.where(np.isfinite(a), q, NISAR_NODATA).astype(np.uint16)
        name = f"nisar/{p.time_utc:%Y-%m-%dT%H%M}_{p.track_id}.npz"
        total += save_npz(OUT / name, **arrays)
        detail = json.loads(ds.attrs.get("nisar_passes_detail", "[]"))
        snapshots.append({"time_utc": iso(p.time_utc), "track": p.track_id, "file": name,
                          "passes": [{"time_utc": iso(d["time"]), "track": d["track"]} for d in detail],
                          "attrs": {k: (float(v) if isinstance(v, (float, np.floating)) else str(v))
                                    for k, v in ds.attrs.items()
                                    if k.startswith("nisar_") and k != "nisar_passes_detail"}})
        write_part("nisar.json", {"scale": NISAR_SCALE, "nodata": NISAR_NODATA, "lookback_days": NISAR_LOOKBACK_DAYS,
                                  "snapshots": snapshots, "bytes": total})
    print(f"  nisar: {len(snapshots)} snapshots, {total / 1e6:.1f} MB")


# ---- manifest ---------------------------------------------------------------------------------

def _default_start(wind: dict | None) -> str | None:
    """A start time worth trying first: the hour with the worst fire weather (strong wind, hot, dry air) in
    ERA5-Land, among the hours that also have MERRA-2 wind for the animation."""
    files = sorted((OUT / "weather" / "era5_land").glob("*.npz"))
    if not files:
        return None
    parts = [np.load(f) for f in files]
    t = pd.to_datetime(np.concatenate([p["time_s"] for p in parts]), unit="s", utc=True)
    mean = {v: np.concatenate([np.nanmean(p[v], axis=(1, 2)) for p in parts])
            for v in ("temp_c", "rh_pct", "wind_u_ms", "wind_v_ms")}
    score = np.hypot(mean["wind_u_ms"], mean["wind_v_ms"]) * (100 - mean["rh_pct"]) * np.maximum(mean["temp_c"], 0)
    if wind:
        score[(t < pd.Timestamp(wind["first_hour_utc"])) | (t > pd.Timestamp(wind["last_hour_utc"]))] = -np.inf
    k = int(np.nanargmax(score))
    print(f"  default start {t[k]}: wind {np.hypot(mean['wind_u_ms'][k], mean['wind_v_ms'][k]):.1f} m/s, "
          f"{mean['temp_c'][k]:.0f} degC, RH {mean['rh_pct'][k]:.0f}%")
    return t[k].strftime("%Y-%m-%dT%H:%MZ")


def step_manifest() -> None:
    def part(name):
        f = PARTS / name
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    static = part("static.json")
    if static is None:
        raise SystemExit("run the static step first")
    w, s, e, n = BBOX
    wind = part("wind.json")
    index = {
        "name": NAME, "built_utc": iso(now_utc()), "center": list(CENTER), "bbox": list(BBOX),
        "start_utc": iso(START), "default_start_utc": _default_start(wind), **static,
        "weather": {src: part(f"weather_{src}.json") for src in ("era5_land", "gfs") if part(f"weather_{src}.json")},
        "wind": wind, "firms": part("firms.json"), "s2": part("s2.json"), "nisar": part("nisar.json"),
    }
    write_json("index.json", index)
    # the area with data, for the map: green inside, red outside (same format as regions.geojson)
    box = [[w, s], [e, s], [e, n], [w, n], [w, s]]
    world = [[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]
    write_json("regions.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"available": True}, "geometry": {"type": "Polygon", "coordinates": [box]}},
        {"type": "Feature", "properties": {"available": False},
         "geometry": {"type": "Polygon", "coordinates": [world, box[::-1]]}}]})
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"  total {size / 1e6:.1f} MB in {OUT}")


STEPS = {"static": step_static, "firms": step_firms, "wind": step_wind, "era5": step_era5, "gfs": step_gfs,
         "s2": step_s2, "nisar": step_nisar, "manifest": step_manifest}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", choices=[*STEPS, "all"])
    ap.add_argument("--nisar-limit", type=int, default=None, help="only the last N NISAR passes (to try it out)")
    args = ap.parse_args(argv)
    steps = list(STEPS) if "all" in args.steps else args.steps
    for name in steps:
        t0 = time.time()
        print(f"== {name}", flush=True)
        if name == "nisar":
            step_nisar(args.nisar_limit)
        else:
            STEPS[name]()
        print(f"== {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
