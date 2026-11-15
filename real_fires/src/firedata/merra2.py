"""MERRA-2 hourly near-surface weather (wind first) from NASA GES DISC, found through Earthdata Search (CMR).

Collection: M2T1NXSLV v5.12.4, "tavg1_2d_slv_Nx" (CMR concept id C1276812863-GES_DISC)
    https://search.earthdata.nasa.gov/search/granules?p=C1276812863-GES_DISC
    One granule (netCDF4) per UTC day, 24 hourly means, global grid 0.5 deg lat x 0.625 deg lon (~55 km).
    Times are the middle of each averaging hour: 00:30, 01:30, ... (value at 13:30 = mean over 13:00-14:00 UTC).
    Published ~2-6 weeks after the month ends (checked 2026-11-15: last granule 2026-10-01), so it serves
    hindcasts of past fires, not live predictions.

Access: nothing global is downloaded. For each day only the fire box is read:
    1. method="opendap" (default): NASA Cloud OPeNDAP subsets on the server (DAP4 constraint), a few KB per day.
    2. method="earthaccess": earthaccess opens the full granule remotely and reads only the box (fallback,
       slower: more HTTP requests).
    Both need an Earthdata login (EARTHDATA_USERNAME / EARTHDATA_PASSWORD in real_fires/.env, or ~/.netrc)
    and, once, approval of the "NASA GESDISC DATA ARCHIVE" application in the Earthdata profile.
    Each day is cached in data/fires/<fire>/raw/merra2/ (git-ignored), or data/merra2_cache/raw/ for a free box.

Usage:
    from firedata import merra2
    ds = merra2.open_box((-122.6, 37.2, -117.4, 42.3), "2026-08-08", "2026-08-14")   # xarray (time, lat, lon)
    ds = merra2.for_fire("bug_fire")                         # the fire's bbox + 2 deg, its data window
    df = merra2.point_series(ds, 39.727, -120.037)           # hourly table at one point (bilinear)

    python -m firedata.merra2 search --start 2026-08-08 --end 2026-08-10
    python -m firedata.merra2 fire bug_fire --start 2026-08-08 --end 2026-08-14    # writes processed/merra2_*.{nc,csv}
"""
from __future__ import annotations

import argparse
import functools
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .config import FIRES, REPO_ROOT, FireConfig

log = logging.getLogger(__name__)

SHORT_NAME = "M2T1NXSLV"
VERSION = "5.12.4"
CONCEPT_ID = "C1276812863-GES_DISC"
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
CACHE_DIR = REPO_ROOT / "data" / "merra2_cache" / "raw"     # data/**/raw is git-ignored

# Native MERRA-2 grid (cell centres): lat -90..90 step 0.5 (361), lon -180..179.375 step 0.625 (576).
LAT0, DLAT, NLAT = -90.0, 0.5, 361
LON0, DLON, NLON = -180.0, 0.625, 576

# MERRA-2 variable -> (units, description). All are in M2T1NXSLV.
VARIABLES = {
    "U10M": ("m/s", "Eastward wind at 10 m"),
    "V10M": ("m/s", "Northward wind at 10 m"),
    "U2M": ("m/s", "Eastward wind at 2 m"),
    "V2M": ("m/s", "Northward wind at 2 m"),
    "U50M": ("m/s", "Eastward wind at 50 m"),
    "V50M": ("m/s", "Northward wind at 50 m"),
    "T2M": ("K", "Air temperature at 2 m"),
    "T2MDEW": ("K", "Dew point temperature at 2 m"),
    "QV2M": ("kg/kg", "Specific humidity at 2 m"),
    "PS": ("Pa", "Surface pressure"),
}
DEFAULT_VARIABLES = ("U10M", "V10M", "U2M", "V2M", "U50M", "V50M", "T2M", "T2MDEW", "PS")
WIND_LEVELS = {"10m": ("U10M", "V10M"), "2m": ("U2M", "V2M"), "50m": ("U50M", "V50M")}


# ---- grid ------------------------------------------------------------------------------------

def grid_window(bbox: tuple[float, float, float, float], margin_deg: float = 0.0) -> tuple[slice, slice]:
    """Index slices (lat, lon) of the MERRA-2 cells whose centres cover bbox (west, south, east, north) + margin.

    One extra cell on each side, so bilinear interpolation anywhere inside the box has neighbours.
    """
    w, s, e, n = bbox
    if w > e or s > n:
        raise ValueError(f"bbox must be (west, south, east, north) without crossing the antimeridian, got {bbox}")
    w, s, e, n = w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg
    i0 = max(0, math.floor((s - LAT0) / DLAT))
    i1 = min(NLAT - 1, math.ceil((n - LAT0) / DLAT))
    j0 = max(0, math.floor((w - LON0) / DLON))
    j1 = min(NLON - 1, math.ceil((e - LON0) / DLON))
    return slice(i0, i1 + 1), slice(j0, j1 + 1)


def cell_centres(lat_idx: slice, lon_idx: slice) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes (degrees) of the cells in the window."""
    return (LAT0 + DLAT * np.arange(lat_idx.start, lat_idx.stop),
            LON0 + DLON * np.arange(lon_idx.start, lon_idx.stop))


# ---- search (Earthdata Search / CMR, no login needed) -----------------------------------------

def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def search_granules(start, end) -> pd.DataFrame:
    """Daily M2T1NXSLV granules overlapping [start, end) UTC: date, granule id, data / OPeNDAP / S3 URLs.

    The same query as Earthdata Search; days not yet published are simply missing (see missing_days()).
    """
    start, end = _utc(start), _utc(end)
    r = requests.get(CMR_GRANULES, params={
        "collection_concept_id": CONCEPT_ID,
        "temporal": f"{start:%Y-%m-%dT%H:%M:%SZ},{(end - pd.Timedelta(seconds=1)):%Y-%m-%dT%H:%M:%SZ}",
        "page_size": 2000, "sort_key": "start_date"}, timeout=120)
    r.raise_for_status()
    rows = []
    for e in r.json()["feed"]["entry"]:
        links = {"data": None, "opendap": None, "s3": None}
        for link in e.get("links", []):
            href, rel = link.get("href", ""), link.get("rel", "")
            if href.startswith("s3://") and links["s3"] is None:
                links["s3"] = href
            elif "opendap.earthdata.nasa.gov" in href and links["opendap"] is None:
                links["opendap"] = href
            elif rel.endswith("/data#") and href.endswith(".nc4") and links["data"] is None:
                links["data"] = href
        rows.append({"date": pd.Timestamp(e["time_start"]).tz_convert("UTC").normalize(),
                     "granule": e["title"], "file": e["title"].split(":")[-1], **links})
    df = pd.DataFrame(rows, columns=["date", "granule", "file", "data", "opendap", "s3"])
    # a reprocessed day can appear twice (stream 400 -> 401): keep the last one listed
    return df.drop_duplicates("date", keep="last").reset_index(drop=True)


def missing_days(granules: pd.DataFrame, start, end) -> list[pd.Timestamp]:
    """UTC days in [start, end) with no granule (not published yet, or a gap)."""
    days = pd.date_range(_utc(start).normalize(), _utc(end) - pd.Timedelta(seconds=1), freq="D")
    have = set(granules["date"]) if len(granules) else set()
    return [d for d in days if d not in have]


# ---- login -----------------------------------------------------------------------------------

class Merra2AccessError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def _token() -> str:
    """Earthdata bearer token (login once per process; .env first, then ~/.netrc)."""
    import earthaccess
    strategy = "environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc"
    auth = earthaccess.login(strategy=strategy)
    if not auth or not getattr(auth, "authenticated", False):
        raise Merra2AccessError(
            f"Earthdata login failed (strategy {strategy}). Put EARTHDATA_USERNAME / EARTHDATA_PASSWORD in "
            f"{REPO_ROOT / '.env'} (see .env.example).")
    return earthaccess.__store__.auth.token["access_token"]


# ---- read one day ----------------------------------------------------------------------------

def opendap_constraint(variables, lat_idx: slice, lon_idx: slice) -> str:
    """DAP4 constraint expression for `variables` (time, lat, lon) in the window, plus the coordinates."""
    la = f"[{lat_idx.start}:1:{lat_idx.stop - 1}]"
    lo = f"[{lon_idx.start}:1:{lon_idx.stop - 1}]"
    parts = [f"/{v}[0:1:23]{la}{lo}" for v in variables] + [f"/lat{la}", f"/lon{lo}", "/time"]
    return ";".join(parts)


def _read_opendap(granule: pd.Series, variables, lat_idx: slice, lon_idx: slice, out: Path) -> None:
    if not granule.get("opendap"):
        raise Merra2AccessError(f"{granule['file']}: no OPeNDAP link in CMR")
    r = requests.get(granule["opendap"] + ".dap.nc4",
                     params={"dap4.ce": opendap_constraint(variables, lat_idx, lon_idx)},
                     headers={"Authorization": f"Bearer {_token()}"}, timeout=300)
    if r.status_code in (401, 403):
        raise Merra2AccessError(
            f"OPeNDAP refused the Earthdata login (HTTP {r.status_code}). In https://urs.earthdata.nasa.gov/profile "
            "-> Applications -> Authorized Apps, approve 'NASA GESDISC DATA ARCHIVE', then retry.")
    r.raise_for_status()
    tmp = out.with_suffix(".part")
    tmp.write_bytes(r.content)
    with xr.open_dataset(tmp, engine="h5netcdf") as ds:   # check it parses before caching
        ds = _tidy(ds, variables).load()
    tmp.unlink()
    ds.to_netcdf(out, engine="h5netcdf")


def _read_earthaccess(granule: pd.Series, variables, lat_idx: slice, lon_idx: slice, out: Path) -> None:
    import earthaccess
    _token()
    results = earthaccess.search_data(short_name=SHORT_NAME, version=VERSION,
                                      granule_name=granule["file"], count=1)
    if not results:
        raise Merra2AccessError(f"earthaccess found no granule {granule['file']}")
    (f,) = earthaccess.open(results)
    with xr.open_dataset(f, engine="h5netcdf") as ds:
        ds = _tidy(ds[list(variables)].isel(lat=lat_idx, lon=lon_idx), variables).load()
    ds.to_netcdf(out, engine="h5netcdf")


def _tidy(ds: xr.Dataset, variables) -> xr.Dataset:
    """Keep the variables, float32, fill values as NaN, UTC times without tz, units attrs."""
    ds = ds[list(variables)]
    for v in variables:
        ds[v] = ds[v].astype("float32")
        ds[v].attrs = {"units": VARIABLES.get(v, ("", ""))[0], "long_name": VARIABLES.get(v, ("", v))[1]}
    return ds


def _cache_name(granule: pd.Series, lat_idx: slice, lon_idx: slice, variables) -> str:
    tag = "-".join(sorted(variables))
    return (f"{Path(granule['file']).stem}_lat{lat_idx.start}-{lat_idx.stop - 1}"
            f"_lon{lon_idx.start}-{lon_idx.stop - 1}_{tag}.nc")


def fetch_day(granule: pd.Series, lat_idx: slice, lon_idx: slice, variables=DEFAULT_VARIABLES,
              cache_dir: Path | None = None, method: str = "opendap") -> Path:
    """The window of one daily granule as a small netCDF file in the cache (read remotely once)."""
    unknown = set(variables) - set(VARIABLES)
    if unknown:
        raise ValueError(f"unknown MERRA-2 variables {sorted(unknown)}; choose from {list(VARIABLES)}")
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / _cache_name(granule, lat_idx, lon_idx, variables)
    if out.exists():
        return out
    readers = {"opendap": _read_opendap, "earthaccess": _read_earthaccess}
    if method not in readers:
        raise ValueError(f"method must be one of {list(readers)}")
    log.info("MERRA-2 %s via %s", granule["file"], method)
    readers[method](granule, variables, lat_idx, lon_idx, out)
    return out


# ---- public API ------------------------------------------------------------------------------

def open_box(bbox: tuple[float, float, float, float], start, end, variables=DEFAULT_VARIABLES, *,
             margin_deg: float = 0.0, cache_dir: Path | None = None, method: str = "opendap",
             derived: bool = True) -> xr.Dataset:
    """Hourly MERRA-2 (time, lat, lon) over bbox + margin, for averaging hours centred in [start, end) UTC.

    Missing (unpublished) days are skipped with a warning and listed in attrs["missing_days"]; if no day is
    available a LookupError is raised. With derived=True, wind speed / direction and RH are added.
    """
    start, end = _utc(start), _utc(end)
    lat_idx, lon_idx = grid_window(bbox, margin_deg)
    granules = search_granules(start, end)
    missing = missing_days(granules, start, end)
    if missing:
        log.warning("MERRA-2: no granule for %d day(s): %s", len(missing), ", ".join(f"{d:%Y-%m-%d}" for d in missing))
    if granules.empty:
        raise LookupError(f"MERRA-2 {SHORT_NAME}: no granule between {start} and {end} (not published yet?)")
    files = [fetch_day(g, lat_idx, lon_idx, variables, cache_dir, method) for _, g in granules.iterrows()]
    days = []
    for f in files:
        with xr.open_dataset(f, engine="h5netcdf") as d:
            days.append(d.load())
    ds = xr.concat(days, dim="time").sortby("time") if len(days) > 1 else days[0]
    t = pd.DatetimeIndex(ds.time.values)
    ds = ds.isel(time=(t >= start.tz_localize(None)) & (t < end.tz_localize(None)))
    if derived:
        ds = add_derived(ds)
    ds.attrs.update({
        "source": f"MERRA-2 {SHORT_NAME} v{VERSION} (NASA GMAO, GES DISC), CMR {CONCEPT_ID}",
        "resolution": "0.5 deg lat x 0.625 deg lon, hourly means; time = middle of the averaging hour (UTC)",
        "bbox_requested": list(map(float, bbox)), "margin_deg": float(margin_deg),
        "start_utc": start.isoformat(), "end_utc": end.isoformat(),
        "missing_days": ",".join(f"{d:%Y-%m-%d}" for d in missing),
    })
    return ds


def wind_speed_dir(u, v):
    """Speed (m/s) and meteorological direction (degrees the wind blows FROM, clockwise from north).

    Same convention as geelayers.derive.wind_speed_dir: u > 0 (blowing east) -> from the west, 270.
    """
    u, v = np.asarray(u), np.asarray(v)
    return np.hypot(u, v), np.mod(np.degrees(np.arctan2(-u, -v)), 360.0)


def rh_from_dewpoint(t_k, td_k):
    """Relative humidity (%) from temperature and dew point (K), Magnus formula over water."""
    t, td = np.asarray(t_k) - 273.15, np.asarray(td_k) - 273.15
    return np.clip(100.0 * np.exp(17.625 * td / (243.04 + td) - 17.625 * t / (243.04 + t)), 0, 100)


def add_derived(ds: xr.Dataset) -> xr.Dataset:
    """wind_speed_<lvl>_ms / wind_dir_<lvl>_deg for each wind level present, temp_c, rh_pct."""
    ds = ds.copy()
    for lvl, (u, v) in WIND_LEVELS.items():
        if u in ds and v in ds:
            spd, dirn = wind_speed_dir(ds[u].values, ds[v].values)
            ds[f"wind_speed_{lvl}_ms"] = (ds[u].dims, spd.astype("float32"),
                                          {"units": "m/s", "long_name": f"Wind speed at {lvl}"})
            ds[f"wind_dir_{lvl}_deg"] = (ds[u].dims, dirn.astype("float32"),
                                         {"units": "deg", "long_name": f"Wind direction at {lvl} (from, clockwise from N)"})
    if "T2M" in ds:
        ds["temp_c"] = (ds.T2M.dims, (ds.T2M.values - 273.15).astype("float32"),
                        {"units": "degC", "long_name": "Air temperature at 2 m"})
        if "T2MDEW" in ds:
            ds["rh_pct"] = (ds.T2M.dims, rh_from_dewpoint(ds.T2M.values, ds.T2MDEW.values).astype("float32"),
                            {"units": "%", "long_name": "Relative humidity at 2 m (from T2M, T2MDEW)"})
    return ds


def point_series(ds: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
    """Hourly table at one point: raw variables bilinear-interpolated, then speed / direction / RH recomputed
    (interpolating angles is wrong). Index: time_utc (tz-aware)."""
    raw = [v for v in ds.data_vars if v in VARIABLES]
    p = add_derived(ds[raw].interp(lat=lat, lon=lon))
    df = p.reset_coords(drop=True).to_dataframe()
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC").rename("time_utc")
    return df


WIND_VARIABLES = ("U10M", "V10M")


def wind_grid(bbox: tuple[float, float, float, float], time, margin_deg: float = 1.5,
              cache_dir: Path | None = None, method: str = "opendap") -> dict:
    """10 m wind of the MERRA-2 hour that contains `time`, on the cells around bbox + margin, as plain JSON.

    {"time_utc", "hour_start_utc", "lat": [south -> north], "lon": [west -> east], "u", "v": [lat][lon] m/s,
     "speed_max", "mean_speed_ms", "mean_dir_deg", "source", "resolution"}. LookupError if MERRA-2 has no data
    for that day yet (published ~2-6 weeks after the month ends).
    """
    t = _utc(time)
    hour = t.floor("h")
    ds = open_box(bbox, hour, hour + pd.Timedelta(hours=1), WIND_VARIABLES, margin_deg=margin_deg,
                  cache_dir=cache_dir, method=method, derived=False)
    if ds.sizes.get("time", 0) == 0:
        raise LookupError(f"MERRA-2 has no data for {hour:%Y-%m-%d %H:%M} UTC yet")
    one = ds.isel(time=0).sortby("lat").sortby("lon")
    u, v = one.U10M.values.astype(float), one.V10M.values.astype(float)
    ok = np.isfinite(u) & np.isfinite(v)
    if not ok.any():
        raise LookupError(f"MERRA-2 wind is missing at {hour:%Y-%m-%d %H:%M} UTC")
    speed = np.hypot(u, v)
    _, mean_dir = wind_speed_dir(u[ok].mean(), v[ok].mean())
    r = lambda a: np.where(np.isfinite(a), np.round(a, 2), 0.0).tolist()  # noqa: E731
    return {
        "time_utc": pd.Timestamp(one.time.values).tz_localize("UTC").isoformat(),
        "hour_start_utc": hour.isoformat(),
        "lat": np.round(one.lat.values.astype(float), 4).tolist(),
        "lon": np.round(one.lon.values.astype(float), 4).tolist(),
        "u": r(u), "v": r(v),
        "speed_max": round(float(np.nanmax(speed)), 2),
        "mean_speed_ms": round(float(np.hypot(u[ok].mean(), v[ok].mean())), 2),
        "mean_dir_deg": round(float(mean_dir), 1),
        "source": f"MERRA-2 {SHORT_NAME} 10 m wind (NASA GMAO)",
        "resolution": "0.5 deg x 0.625 deg (~55 km), hourly mean",
    }


def for_fire(fire: str | FireConfig, start=None, end=None, variables=DEFAULT_VARIABLES, *,
             margin_deg: float = 2.0, method: str = "opendap") -> xr.Dataset:
    """MERRA-2 over a study fire's bbox + margin (default: the fire's data window), cached in its raw/merra2/."""
    F = FIRES[fire] if isinstance(fire, str) else fire
    return open_box(F.bbox, start or F.start_utc, end or F.end_utc, variables, margin_deg=margin_deg,
                    cache_dir=F.dir("raw") / "merra2", method=method)


# ---- CLI -------------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m firedata.merra2", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("search", help="list the daily granules (no login)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p = sub.add_parser("fire", help="read a fire's box, write processed/merra2_grid.nc + merra2_point.csv")
    p.add_argument("fire", choices=FIRES)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--margin-deg", type=float, default=2.0)
    p.add_argument("--vars", default=",".join(DEFAULT_VARIABLES))
    p.add_argument("--method", default="opendap", choices=["opendap", "earthaccess"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if a.cmd == "search":
        g = search_granules(a.start, a.end)
        print(g[["date", "file"]].to_string(index=False) if len(g) else "no granules")
        miss = missing_days(g, a.start, a.end)
        if miss:
            print("missing days:", ", ".join(f"{d:%Y-%m-%d}" for d in miss))
        return

    F = FIRES[a.fire]
    ds = for_fire(F, a.start, a.end, [v.strip() for v in a.vars.split(",") if v.strip()],
                  margin_deg=a.margin_deg, method=a.method)
    proc = F.dir("processed")
    ds.to_netcdf(proc / "merra2_grid.nc", engine="h5netcdf",
                 encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    df = point_series(ds, F.ignition_lat, F.ignition_lon)
    df.round(3).to_csv(proc / "merra2_point.csv")
    print(f"{F.name}: {ds.sizes['time']} hours, {ds.sizes['lat']} x {ds.sizes['lon']} cells "
          f"(lat {float(ds.lat.min())}..{float(ds.lat.max())}, lon {float(ds.lon.min())}..{float(ds.lon.max())})")
    print(f"  -> {proc / 'merra2_grid.nc'}\n  -> {proc / 'merra2_point.csv'}")
    if ds.attrs.get("missing_days"):
        print("  missing days:", ds.attrs["missing_days"])


if __name__ == "__main__":
    main()
