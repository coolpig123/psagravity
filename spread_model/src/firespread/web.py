"""What the web app needs from a prediction: request body in, JSON-ready dict out (no web framework here).

backend/server.py serves it as POST /api/predict:

    body = {"polygon": <GeoJSON>, "start": "2026-08-09T21:00Z",
            "radius_km": 5.0, "res_m": 100.0, "use_nisar": true, "burn_scars": true, "steps": 30}

    -> {"cells":   GeoJSON FeatureCollection, one square per cell with burn_prob > MIN_PROB: the chance the
                   cell burns within `steps` CA steps (simulate.py; properties.p), ready for a MapLibre fill layer,
        "bounds":  [west, south, east, north] of the model grid,
        "summary": {main_direction, main_direction_deg, direction_strength, expected_new_area_km2,
                    weather_time_utc, weather_source, nisar_passes_used, nisar_coverage_pct, s2_clear_pct,
                    s2_burned_pct, sim_steps, sim_expected_area_km2, n_cells, max_prob, text, notes}}
                   (main_direction / expected_new_area_km2 are for the next single step)

Only the polygon + radius_km is fetched from Earth Engine and NISAR (see predict()).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xarray as xr

from .alexandridis import DIRECTIONS, MASK_REASONS, shift, summary
from .predict import predict
from .simulate import DEFAULT_STEPS, simulate_spread

MIN_PROB = 0.01
MAX_STEPS = 200
MAX_HOURS = 168.0
MAX_RADIUS_KM = 50.0
RES_RANGE_M = (20.0, 1000.0)
VALID_WEATHER_SOURCES = ("auto", "meteomatics", "rtma", "era5_land", "gfs")
SUMMARY_KEYS = ("main_direction", "main_direction_deg", "direction_strength", "expected_new_area_km2",
                "expected_new_cells", "time_utc", "weather_time_utc", "weather_source", "weather_dataset",
                "weather_steps", "weather_note", "fosberg_fwi_mean", "nisar_passes_used", "nisar_coverage_pct",
                "nisar_alignment", "s2_clear_pct", "s2_burned_pct", "fire_touches_grid_edge", "sim_steps", "sim_runs",
                "sim_expected_area_km2", "sim_reached_grid_edge", "sim_time_developing", "sim_weather_steps",
                "sim_forecast_hours")


def parse_request(body: dict) -> dict:
    """Checked predict() arguments from the request body. ValueError = the client's mistake (HTTP 400)."""
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    polygon = body.get("polygon")
    if not isinstance(polygon, dict):
        raise ValueError("polygon: a GeoJSON Polygon, Feature or FeatureCollection is required")
    try:
        start = pd.Timestamp(body.get("start"))
        if start is pd.NaT:
            raise ValueError("empty")
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    except (ValueError, TypeError) as e:
        raise ValueError(f"start: a UTC time like 2026-08-09T21:00Z is required ({e})") from None

    def number(key, default, low, high):
        try:
            v = float(body.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number") from None
        if not (math.isfinite(v) and low <= v <= high):
            raise ValueError(f"{key} must be between {low:g} and {high:g}")
        return v

    out = {"polygon": polygon, "start": start.isoformat(),
           "radius_km": number("radius_km", 5.0, 0.1, MAX_RADIUS_KM),
           "res_m": number("res_m", 100.0, *RES_RANGE_M),
           "steps": int(number("steps", DEFAULT_STEPS, 1, MAX_STEPS)),
           "use_nisar": bool(body.get("use_nisar", True)), "burn_scars": bool(body.get("burn_scars", True))}
    if "weather_source" in body and body["weather_source"] is not None:
        ws = str(body["weather_source"]).strip().lower()
        if ws not in VALID_WEATHER_SOURCES:
            raise ValueError(f"weather_source must be one of {list(VALID_WEATHER_SOURCES)}")
        out["weather_source"] = ws
    if "hours" in body and body["hours"] is not None:
        out["hours"] = number("hours", 6.0, 1.0, MAX_HOURS)
    return out


def default_nisar_provider():
    """The live NISAR provider from real_fires (firedata.nisar_provider). ImportError if it is not installed."""
    from firedata.nisar_provider import nisar_provider
    return nisar_provider()


def _guarded(provider, notes: list[str]):
    """A provider that never breaks the prediction: on any error, run without NISAR and say so."""
    def get_nisar(request):
        try:
            return provider(request)
        except Exception as e:  # no Earthdata login, ASF down, ...
            notes.append(f"NISAR skipped: {type(e).__name__}: {e}")
            return None
    return get_nisar


def cells_geojson(result: xr.Dataset, min_prob: float = MIN_PROB, var: str = "ignition_prob") -> dict:
    """Cells with `var` > min_prob as lon/lat squares (properties.p), for a MapLibre fill layer."""
    from pyproj import Transformer
    prob = result[var].values
    rows, cols = np.nonzero(np.nan_to_num(prob) > min_prob)
    if rows.size == 0:
        return {"type": "FeatureCollection", "features": []}
    half = float(result.attrs["res_m"]) / 2
    x, y = result.x.values[cols], result.y.values[rows]
    corners_x = np.stack([x - half, x + half, x + half, x - half, x - half], axis=1)
    corners_y = np.stack([y + half, y + half, y - half, y - half, y + half], axis=1)
    lon, lat = Transformer.from_crs(result.attrs["crs"], 4326, always_xy=True).transform(corners_x, corners_y)
    ring = np.round(np.stack([lon, lat], axis=2), 6).tolist()
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"p": round(float(p), 4)},
         "geometry": {"type": "Polygon", "coordinates": [r]}}
        for r, p in zip(ring, prob[rows, cols])]}


def _bounds(result: xr.Dataset) -> list[float]:
    from pyproj import Transformer
    half = float(result.attrs["res_m"]) / 2
    x, y = result.x.values, result.y.values
    xs = np.array([x.min() - half, x.max() + half, x.max() + half, x.min() - half])
    ys = np.array([y.max() + half, y.max() + half, y.min() - half, y.min() - half])
    lon, lat = Transformer.from_crs(result.attrs["crs"], 4326, always_xy=True).transform(xs, ys)
    return [round(float(v), 6) for v in (lon.min(), lat.min(), lon.max(), lat.max())]


def no_spread_reason(result: xr.Dataset, min_prob: float = MIN_PROB) -> str:
    """Why nothing shows on the map: what the cells right next to the fire are, or how small their chances are."""
    burning = result["burning"].values.astype(bool)
    ring = np.zeros_like(burning)
    for _, dr, dc, _ in DIRECTIONS:
        ring |= shift(burning.astype(np.float32), dr, dc, 0.0) > 0
    ring &= ~burning
    if not ring.any():
        return "the polygon fills the whole computed area; draw a smaller polygon"
    reasons = result["mask_reason"].values[ring]
    if (reasons > 0).all():
        codes, counts = np.unique(reasons, return_counts=True)
        parts = [f"{100 * n / reasons.size:.0f}% {MASK_REASONS[c]}" for c, n in sorted(zip(codes, counts),
                                                                                     key=lambda t: -t[1])]
        return "nothing next to the polygon can catch fire: " + ", ".join(parts)
    ok = ring & (result["mask_reason"].values == 0)
    var = "burn_prob" if "burn_prob" in result else "ignition_prob"
    best = float(np.nanmax(result[var].values[ok]))
    why = ""
    if "emc_pct" in result and np.isfinite(result["emc_pct"].values[ok]).any():
        emc_med = float(np.nanmedian(result["emc_pct"].values[ok]))
        if emc_med > 15.0:
            why = f"; the air keeps fuel moist (equilibrium moisture {emc_med:.0f}%)"
    return (f"the cells next to the polygon can burn, but every chance of spreading is below {min_prob:.0%} "
            f"(highest {best:.1%}){why}")


def _plain(v):
    """Attribute value -> something json.dumps accepts (NaN / inf -> None)."""
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


def _weather_timeline(result: xr.Dataset) -> list[dict]:
    """Extract hourly weather forecast timeline when multi-step weather is present on `result`."""
    if "time" not in result.coords or result.sizes.get("time", 0) < 1:
        return []
    times = pd.to_datetime(result.coords["time"].values)
    ws = result["wind_speed_time_ms"].values if "wind_speed_time_ms" in result else None
    wd = result["wind_dir_time_deg"].values if "wind_dir_time_deg" in result else None
    tc = result["temp_time_c"].values if "temp_time_c" in result else None
    rh = result["rh_time_pct"].values if "rh_time_pct" in result else None
    timeline = []
    for i, t in enumerate(times):
        entry = {"time_utc": pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if ws is not None and i < len(ws):
            entry["wind_speed_ms"] = _plain(round(float(ws[i]), 2))
        if wd is not None and i < len(wd):
            entry["wind_dir_deg"] = _plain(round(float(wd[i]), 1))
        if tc is not None and i < len(tc):
            entry["temp_c"] = _plain(round(float(tc[i]), 1))
        if rh is not None and i < len(rh):
            entry["rh_pct"] = _plain(round(float(rh[i]), 1))
        timeline.append(entry)
    return timeline


def predict_payload(body: dict, *, predict_fn=predict, nisar_factory=default_nisar_provider) -> dict:
    """POST /api/predict: request body -> response dict. ValueError for a bad request."""
    req = parse_request(body)
    notes: list[str] = []
    nisar = None
    if req["use_nisar"]:
        try:
            nisar = _guarded(nisar_factory(), notes)
        except ImportError as e:
            notes.append(f"NISAR skipped: the NISAR reader is not installed (pip install -e real_fires): {e}")
    extra_kw = {k: req[k] for k in ("weather_source", "hours") if k in req}
    result = predict_fn(polygon=req["polygon"], start=req["start"], nisar=nisar, burn_scars=req["burn_scars"],
                        radius_km=req["radius_km"], res_m=req["res_m"], **extra_kw)
    result = simulate_spread(result, steps=req["steps"])
    if result.attrs.get("sim_reached_grid_edge"):
        notes.append(f"the simulated fire reached the edge of the {req['radius_km']:g} km area around the polygon; "
                     "spread beyond it is not computed (use fewer steps or a larger radius)")
    cells = cells_geojson(result, var="burn_prob")
    if not cells["features"]:
        notes.append("No spread on the map: " + no_spread_reason(result))
    prob = result["burn_prob"].values
    info = {k: _plain(result.attrs[k]) for k in SUMMARY_KEYS if k in result.attrs}
    timeline = _weather_timeline(result)
    if timeline:
        info["weather_timeline"] = timeline
    info.update({"n_cells": len(cells["features"]), "max_prob": _plain(np.nanmax(prob)) if prob.size else None,
                 "text": summary(result), "notes": notes})
    return {"cells": cells, "bounds": _bounds(result), "summary": info}
