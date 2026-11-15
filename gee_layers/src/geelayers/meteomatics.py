"""Meteomatics Weather API client (historical, real-time, and forecast weather at down to 1 km).

URL format:
    https://api.meteomatics.com/<validdatetime>/<parameters>/<location>/json?<optionals>

Authentication uses HTTP Basic Auth with METEOMATICS_USERNAME and METEOMATICS_PASSWORD from the
environment or .env file (in gee_layers/.env, real_fires/.env, or repository root .env).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests
import xarray as xr
from dotenv import load_dotenv
from scipy.interpolate import RegularGridInterpolator

from .derive import wind_speed_dir
from .grid import Grid

# Load .env from gee_layers/, real_fires/, or repo root if present (never overrides existing env vars)
_GEE_DIR = Path(__file__).resolve().parents[2]
_REPO_DIR = _GEE_DIR.parent
for _env_candidate in (_GEE_DIR / ".env", _REPO_DIR / "real_fires" / ".env", _REPO_DIR / ".env"):
    if _env_candidate.exists():
        load_dotenv(_env_candidate, override=False)

BASE_URL = "https://api.meteomatics.com"

# Shared NASA Space Apps Hackathon Meteomatics credentials (can be overridden in .env)
DEFAULT_USERNAME = "hackathon_nasa_space_apps_fueredi_alon"
DEFAULT_PASSWORD = "cnS0A6ESYDCydZN8W9z8"

# Mapping from internal geelayers variable name -> Meteomatics parameter code
CORE_PARAMETERS: dict[str, str] = {
    "wind_u_ms": "wind_speed_u_10m:ms",
    "wind_v_ms": "wind_speed_v_10m:ms",
    "temp_c": "t_2m:C",
    "rh_pct": "relative_humidity_2m:p",
}

STANDARD_PARAMETERS: dict[str, str] = {
    **CORE_PARAMETERS,
    "gust_ms": "wind_gusts_10m_1h:ms",
    "precip_mm": "precip_1h:mm",
}

EXTENDED_PARAMETERS: dict[str, str] = {
    **STANDARD_PARAMETERS,
    "soil_moisture": "soil_moisture_index_-5cm:idx",
    "fosberg_fwi": "fosberg_fire_weather_index:idx",
}

PARAM_TO_VAR: dict[str, str] = {v: k for k, v in EXTENDED_PARAMETERS.items()}


def get_credentials(username: str | None = None, password: str | None = None) -> tuple[str, str]:
    """Return (username, password) from arguments, environment variables, or team hackathon defaults."""
    user = username or os.environ.get("METEOMATICS_USERNAME", "").strip() or DEFAULT_USERNAME
    pwd = password or os.environ.get("METEOMATICS_PASSWORD", "").strip() or DEFAULT_PASSWORD
    if not user or not pwd:
        raise RuntimeError(
            "Meteomatics credentials missing. Set METEOMATICS_USERNAME and METEOMATICS_PASSWORD "
            "in gee_layers/.env (or environment variables) to query the Meteomatics Weather API."
        )
    return user, pwd


def has_credentials() -> bool:
    """True if Meteomatics credentials are available (from environment or team defaults)."""
    return bool(
        (os.environ.get("METEOMATICS_USERNAME", "").strip() or DEFAULT_USERNAME)
        and (os.environ.get("METEOMATICS_PASSWORD", "").strip() or DEFAULT_PASSWORD)
    )


def _to_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _format_utc(ts: str | pd.Timestamp) -> str:
    return _to_utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _grid_dims(grid: Grid, max_side: int = 12) -> tuple[int, int]:
    """Choose (ny, nx) lat/lon sampling counts for a Meteomatics bounding-box query.

    Uses ~2 km spacing by default, clamped to [2, max_side] per axis so a single request stays
    well within hackathon API rate/point limits while resolving local wind & terrain gradients.
    """
    km_x = grid.width * grid.res / 1000.0
    km_y = grid.height * grid.res / 1000.0
    step_km = max(1.0, grid.res / 1000.0, 2.0)
    nx = max(2, min(max_side, int(round(km_x / step_km)) + 1))
    ny = max(2, min(max_side, int(round(km_y / step_km)) + 1))
    return ny, nx


def _build_url(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
    step_hours: int,
    params: Sequence[str],
    bbox_lonlat: tuple[float, float, float, float],
    ny: int,
    nx: int,
) -> str:
    west, south, east, north = bbox_lonlat
    # Pad bbox slightly (0.005 deg ~ 500 m) so target grid cells are strictly inside the query box
    pad = 0.005
    north_p, south_p = min(90.0, north + pad), max(-90.0, south - pad)
    west_p, east_p = max(-180.0, west - pad), min(180.0, east + pad)

    t_start = _to_utc(start)
    if end is None:
        time_spec = _format_utc(t_start.floor("h"))
    else:
        t_end = _to_utc(end)
        if t_end <= t_start:
            time_spec = _format_utc(t_start.floor("h"))
        else:
            # Align to top-of-the-hour steps in [start, end), matching Earth Engine hourly collections
            t0_aligned = t_start.ceil("h")
            # Subtract 1 second before floor("h") if t_end is on an exact hour or has a +1s epsilon (as in get_layers)
            t1_aligned = (t_end - pd.Timedelta(seconds=1)).floor("h")
            if t1_aligned < t0_aligned:
                t0_aligned = t1_aligned = t_start.floor("h")
            if t1_aligned <= t0_aligned:
                time_spec = _format_utc(t0_aligned)
            else:
                time_spec = f"{_format_utc(t0_aligned)}--{_format_utc(t1_aligned)}:PT{max(1, int(step_hours))}H"

    param_spec = ",".join(params)
    loc_spec = f"{north_p:.5f},{west_p:.5f}_{south_p:.5f},{east_p:.5f}:{ny}x{nx}"
    return f"{BASE_URL}/{time_spec}/{param_spec}/{loc_spec}/json?model=mix"


def _parse_response(
    payload: dict,
    grid: Grid,
    var_attrs: dict[str, dict[str, str]],
) -> xr.Dataset:
    """Parse Meteomatics JSON response and bilinearly interpolate onto `grid`."""
    data_list = payload.get("data", [])
    if not data_list:
        raise ValueError("Meteomatics response contained no 'data' entries")

    # Collect unique sorted coordinates and timestamps from the first parameter block
    first_coords = data_list[0].get("coordinates", [])
    if not first_coords:
        raise ValueError("Meteomatics response contained no coordinates")

    lats = np.array(sorted({ float(c["lat"]) for c in first_coords }), dtype=np.float64)
    lons = np.array(sorted({ float(c["lon"]) for c in first_coords }), dtype=np.float64)
    raw_dates = [d["date"] for d in first_coords[0].get("dates", [])]
    times = pd.to_datetime(raw_dates, utc=True).tz_localize(None)
    n_t = len(times)

    lat_to_idx = {float(lat): i for i, lat in enumerate(lats)}
    lon_to_idx = {float(lon): j for j, lon in enumerate(lons)}

    target_lat, target_lon = grid.latlon()
    pts = np.stack([target_lat.ravel(), target_lon.ravel()], axis=-1)

    lat_c = np.round(target_lat, 6).astype(np.float32)
    lon_c = np.round(target_lon, 6).astype(np.float32)
    ds = xr.Dataset(
        coords={
            "time": times,
            "y": ("y", grid.y, {"units": "m", "standard_name": "projection_y_coordinate"}),
            "x": ("x", grid.x, {"units": "m", "standard_name": "projection_x_coordinate"}),
            "lat": (("y", "x"), lat_c, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": (("y", "x"), lon_c, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "crs": grid.crs,
            "res_m": float(grid.res),
            "transform": list(grid.transform),
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "weather_source": "meteomatics",
            "weather_dataset": "Meteomatics Weather API (mix: US1k / EURO1k / global high-res)",
        },
    )

    for entry in data_list:
        param_name = entry.get("parameter", "")
        var_name = PARAM_TO_VAR.get(param_name)
        if not var_name:
            continue

        arr_ll = np.full((n_t, len(lats), len(lons)), np.nan, dtype=np.float64)
        for c in entry.get("coordinates", []):
            i = lat_to_idx.get(float(c["lat"]))
            j = lon_to_idx.get(float(c["lon"]))
            if i is None or j is None:
                continue
            vals = [d.get("value", np.nan) for d in c.get("dates", [])]
            arr_ll[:, i, j] = np.asarray(vals, dtype=np.float64)

        # Meteomatics uses -666.0 or -999.0 for missing values in some indices
        arr_ll[(arr_ll <= -600.0)] = np.nan

        out = np.empty((n_t, grid.height, grid.width), dtype=np.float32)
        for t_idx in range(n_t):
            slab = arr_ll[t_idx]
            if np.isnan(slab).all():
                out[t_idx] = np.nan
                continue
            if np.isnan(slab).any():
                slab = np.where(np.isnan(slab), np.nanmean(slab), slab)

            if len(lats) >= 2 and len(lons) >= 2:
                interp = RegularGridInterpolator(
                    (lats, lons), slab, method="linear", bounds_error=False, fill_value=None
                )
                out[t_idx] = interp(pts).reshape(grid.shape).astype(np.float32)
            else:
                out[t_idx] = float(np.nanmean(slab))

        ds[var_name] = (("time", "y", "x"), out, var_attrs.get(var_name, {}))

    if "wind_u_ms" in ds and "wind_v_ms" in ds:
        speed, direction = wind_speed_dir(ds["wind_u_ms"].values, ds["wind_v_ms"].values)
        ds["wind_speed_ms"] = (("time", "y", "x"), speed, var_attrs.get("wind_speed_ms", {}))
        ds["wind_dir_deg"] = (("time", "y", "x"), direction, var_attrs.get("wind_dir_deg", {}))

    return ds


def fetch_meteomatics_weather(
    grid: Grid,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    step_hours: int = 1,
    *,
    var_attrs: dict[str, dict[str, str]] | None = None,
    username: str | None = None,
    password: str | None = None,
    max_side: int = 12,
    timeout: int = 45,
) -> xr.Dataset:
    """Fetch hourly weather time series (historical, analysis, or future forecast) from Meteomatics.

    Automatically falls back from extended fire-weather parameters to standard/core meteorological
    parameters if the account license restricts certain specialized indices.
    """
    user, pwd = get_credentials(username, password)
    bbox = grid.bbox_lonlat()
    ny, nx = _grid_dims(grid, max_side=max_side)
    attrs = var_attrs or {}

    param_tiers = [
        list(EXTENDED_PARAMETERS.values()),
        list(STANDARD_PARAMETERS.values()),
        list(CORE_PARAMETERS.values()),
    ]

    last_err: Exception | None = None
    for params in param_tiers:
        url = _build_url(start, end, step_hours, params, bbox, ny, nx)
        try:
            resp = requests.get(url, auth=(user, pwd), timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"Meteomatics request failed: {exc}") from exc

        if resp.status_code == 200:
            return _parse_response(resp.json(), grid, attrs)

        if resp.status_code in (401, 402):
            raise RuntimeError(
                f"Meteomatics authentication failed (HTTP {resp.status_code}): check METEOMATICS_USERNAME "
                f"and METEOMATICS_PASSWORD. Response: {resp.text[:300]}"
            )

        # If a parameter is not enabled on the current license tier, try the next tier
        last_err = RuntimeError(f"Meteomatics HTTP {resp.status_code}: {resp.text[:400]}")
        if resp.status_code not in (400, 403, 404):
            break

    assert last_err is not None
    raise last_err
