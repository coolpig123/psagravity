"""What the web app needs from a prediction: request body in, JSON-ready dict out (no web framework here).

backend/server.py serves it as POST /api/predict:

    body = {"polygon": <GeoJSON>, "start": "2026-08-09T21:00Z",
            "radius_km": 5.0, "res_m": 100.0, "use_nisar": true, "burn_scars": true}

    -> {"cells":   GeoJSON FeatureCollection, one square per cell with ignition_prob > MIN_PROB
                   (properties.p = probability), ready for a MapLibre fill layer,
        "bounds":  [west, south, east, north] of the model grid,
        "summary": {main_direction, main_direction_deg, direction_strength, expected_new_area_km2,
                    weather_time_utc, weather_source, nisar_passes_used, nisar_coverage_pct, s2_clear_pct,
                    s2_burned_pct, n_cells, max_prob, text, notes}}

Only the polygon + radius_km is fetched from Earth Engine and NISAR (see predict()).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xarray as xr

from .alexandridis import summary
from .predict import predict

MIN_PROB = 0.01
MAX_RADIUS_KM = 50.0
RES_RANGE_M = (20.0, 1000.0)
SUMMARY_KEYS = ("main_direction", "main_direction_deg", "direction_strength", "expected_new_area_km2",
                "expected_new_cells", "time_utc", "weather_time_utc", "weather_source", "weather_note",
                "nisar_passes_used", "nisar_coverage_pct", "nisar_alignment", "s2_clear_pct", "s2_burned_pct",
                "fire_touches_grid_edge")


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

    return {"polygon": polygon, "start": start.isoformat(),
            "radius_km": number("radius_km", 5.0, 0.1, MAX_RADIUS_KM),
            "res_m": number("res_m", 100.0, *RES_RANGE_M),
            "use_nisar": bool(body.get("use_nisar", True)), "burn_scars": bool(body.get("burn_scars", True))}


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


def cells_geojson(result: xr.Dataset, min_prob: float = MIN_PROB) -> dict:
    """Cells with ignition_prob > min_prob as lon/lat squares (properties.p), for a MapLibre fill layer."""
    from pyproj import Transformer
    prob = result["ignition_prob"].values
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


def _plain(v):
    """Attribute value -> something json.dumps accepts (NaN / inf -> None)."""
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


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
    result = predict_fn(polygon=req["polygon"], start=req["start"], nisar=nisar, burn_scars=req["burn_scars"],
                        radius_km=req["radius_km"], res_m=req["res_m"])
    cells = cells_geojson(result)
    prob = result["ignition_prob"].values
    info = {k: _plain(result.attrs[k]) for k in SUMMARY_KEYS if k in result.attrs}
    info.update({"n_cells": len(cells["features"]), "max_prob": _plain(np.nanmax(prob)) if prob.size else None,
                 "text": summary(result), "notes": notes})
    return {"cells": cells, "bounds": _bounds(result), "summary": info}
