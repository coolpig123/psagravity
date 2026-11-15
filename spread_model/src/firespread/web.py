"""What the web app needs from a prediction: request body in, JSON-ready dict out (no web framework here).

backend/server.py serves it as POST /api/predict:

    body = {"polygon": <GeoJSON>, "start": "2026-08-09T21:00Z",
            "radius_km": 5.0, "res_m": 100.0, "use_nisar": true, "burn_scars": true, "steps": 30, "arrival": true,
            "use_firms": true}     # use_firms: VIIRS detections of the last 12 h (NASA FIRMS) mark the active front

    -> {"cells":   GeoJSON FeatureCollection, one square per cell with burn_prob > MIN_PROB: the chance the
                   cell burns within `steps` CA steps (simulate.py; properties.p), ready for a MapLibre fill layer.
                   With "arrival" (and a trained model, survival.py) also the cells with p_arrive_48h > MIN_PROB,
                   and per cell pe = chance the fire ever gets there, t50 = median hours until it arrives if it
                   does (t25 / t75: its 25-75% range), p6 / p12 / p24 / p48 = chance within 6 / 12 / 24 / 48 h (null for cells already burning),
        "bounds":  [west, south, east, north] of the model grid,
        "summary": {main_direction, main_direction_deg, direction_strength, expected_new_area_km2,
                    weather_time_utc, weather_source, nisar_passes_used, nisar_coverage_pct, s2_clear_pct,
                    s2_burned_pct, sim_steps, sim_expected_area_km2, n_cells, n_map_cells, max_prob, text, notes,
                    arrival, arrival_horizons_h, arrival_model, arrival_area_<h>h_km2,
                    active_detections, active_front_km2}}
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
from .survival import ACTIVE_HOURS, HORIZONS_H, attach_active_front, predict_arrival

MIN_PROB = 0.01
MAX_STEPS = 200
MAX_RADIUS_KM = 50.0
RES_RANGE_M = (20.0, 1000.0)
SUMMARY_KEYS = ("main_direction", "main_direction_deg", "direction_strength", "expected_new_area_km2",
                "expected_new_cells", "time_utc", "weather_time_utc", "weather_source", "weather_note",
                "nisar_passes_used", "nisar_coverage_pct", "nisar_alignment", "s2_clear_pct", "s2_burned_pct",
                "fire_touches_grid_edge", "sim_steps", "sim_runs", "sim_expected_area_km2", "sim_reached_grid_edge",
                "arrival_model", *(f"arrival_area_{h}h_km2" for h in HORIZONS_H),
                "active_detections", "active_front_km2")
# cell properties for the arrival-time view: median hours + chance within each horizon (survival.py)
ARRIVAL_PROPS = {"pe": "p_ever", "t50": "arrival_h", "t25": "arrival_h_q25", "t75": "arrival_h_q75", **{f"p{h}": f"p_arrive_{h}h" for h in HORIZONS_H}}


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
            "steps": int(number("steps", DEFAULT_STEPS, 1, MAX_STEPS)),
            "use_nisar": bool(body.get("use_nisar", True)), "burn_scars": bool(body.get("burn_scars", True)),
            "arrival": bool(body.get("arrival", True)), "use_firms": bool(body.get("use_firms", True))}


def default_nisar_provider():
    """The live NISAR provider from real_fires (firedata.nisar_provider). ImportError if it is not installed."""
    from firedata.nisar_provider import nisar_provider
    return nisar_provider()


def default_detections_provider():
    """Recent VIIRS detections from NASA FIRMS (firedata.firms, needs FIRMS_MAP_KEY in real_fires/.env), for the
    active front: a function (bbox [west, south, east, north], start) -> DataFrame(time_utc, lat, lon, ...).
    ImportError if real_fires is not installed."""
    import tempfile
    from pathlib import Path
    from firedata import firms
    from firedata.config import REPO_ROOT

    def get(bbox, start):
        t0 = pd.Timestamp(start)
        t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
        first = (t0 - pd.Timedelta(hours=ACTIVE_HOURS)).normalize()
        last = t0.normalize() + pd.Timedelta(days=1)
        name = "_".join(f"{v:.3f}" for v in bbox)
        if last < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3):       # past days no longer change: cache
            raw = firms.download_viirs(bbox, first, last, Path(REPO_ROOT) / "data" / "firms_cache" / "raw" / name)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                raw = firms.download_viirs(bbox, first, last, Path(tmp))
        return firms.tidy_viirs(raw)
    return get


def _guarded(provider, notes: list[str]):
    """A provider that never breaks the prediction: on any error, run without NISAR and say so."""
    def get_nisar(request):
        try:
            return provider(request)
        except Exception as e:  # no Earthdata login, ASF down, ...
            notes.append(f"NISAR skipped: {type(e).__name__}: {e}")
            return None
    return get_nisar


def _prop(v) -> float | None:
    v = float(v)
    return round(v, 4) if math.isfinite(v) else None


def cells_geojson(result: xr.Dataset, min_prob: float = MIN_PROB, var: str = "ignition_prob",
                  extra: dict[str, str] | None = None, also_select: tuple[str, ...] = ()) -> dict:
    """Cells with `var` (or any `also_select` variable) > min_prob as lon/lat squares, for a MapLibre fill layer.
    properties.p = `var`; `extra` adds more properties {property name: result variable} (NaN -> null)."""
    from pyproj import Transformer
    prob = result[var].values
    select = np.nan_to_num(prob) > min_prob
    for name in also_select:
        select |= np.nan_to_num(result[name].values) > min_prob
    rows, cols = np.nonzero(select)
    extra = {k: result[v].values[rows, cols] for k, v in (extra or {}).items()}
    if rows.size == 0:
        return {"type": "FeatureCollection", "features": []}
    half = float(result.attrs["res_m"]) / 2
    x, y = result.x.values[cols], result.y.values[rows]
    corners_x = np.stack([x - half, x + half, x + half, x - half, x - half], axis=1)
    corners_y = np.stack([y + half, y + half, y - half, y - half, y + half], axis=1)
    lon, lat = Transformer.from_crs(result.attrs["crs"], 4326, always_xy=True).transform(corners_x, corners_y)
    ring = np.round(np.stack([lon, lat], axis=2), 6).tolist()
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"p": round(float(p), 4), **{k: _prop(v[i]) for k, v in extra.items()}},
         "geometry": {"type": "Polygon", "coordinates": [r]}}
        for i, (r, p) in enumerate(zip(ring, np.nan_to_num(prob[rows, cols])))]}


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


def predict_payload(body: dict, *, predict_fn=predict, nisar_factory=default_nisar_provider,
                    arrival_fn=predict_arrival, detections_factory=default_detections_provider) -> dict:
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
    result = simulate_spread(result, steps=req["steps"])
    if result.attrs.get("sim_reached_grid_edge"):
        notes.append(f"the simulated fire reached the edge of the {req['radius_km']:g} km area around the polygon; "
                     "spread beyond it is not computed (use fewer steps or a larger radius)")
    arrival = False
    if req["arrival"] and req["use_firms"]:      # the active front: where VIIRS saw fire in the last hours
        try:
            result = attach_active_front(result, detections_factory()(_bounds(result), req["start"]), req["start"])
            if not result.attrs.get("active_detections"):
                notes.append(f"no VIIRS detection on the polygon in the {ACTIVE_HOURS:g} h before the start: the whole "
                             "polygon is taken as the active front")
        except ImportError as e:
            notes.append(f"active front skipped: the FIRMS reader is not installed (pip install -e real_fires): {e}")
        except Exception as e:                   # no FIRMS_MAP_KEY, FIRMS down, no source for that date, ...
            notes.append(f"active front skipped ({type(e).__name__}: {e}): the whole polygon is taken as the front")
    if req["arrival"]:
        try:
            result = arrival_fn(result, req["start"])
            arrival = True
        except (FileNotFoundError, ImportError, ValueError) as e:   # no trained model, xgboost not installed
            notes.append(f"arrival times skipped: {e}")
    cells = (cells_geojson(result, var="burn_prob", extra=ARRIVAL_PROPS, also_select=(f"p_arrive_{HORIZONS_H[-1]}h",))
             if arrival else cells_geojson(result, var="burn_prob"))
    if not cells["features"]:
        notes.append("No spread on the map: " + no_spread_reason(result))
    prob = result["burn_prob"].values
    info = {k: _plain(result.attrs[k]) for k in SUMMARY_KEYS if k in result.attrs}
    # n_cells: cells at risk within the CA steps (matches sim_expected_area_km2); n_map_cells: squares on the map
    info.update({"n_cells": int((np.nan_to_num(prob) > MIN_PROB).sum()), "n_map_cells": len(cells["features"]),
                 "max_prob": _plain(np.nanmax(prob)) if prob.size else None,
                 "text": summary(result), "notes": notes, "arrival": arrival,
                 "arrival_horizons_h": list(HORIZONS_H) if arrival else []})
    return {"cells": cells, "bounds": _bounds(result), "summary": info}
