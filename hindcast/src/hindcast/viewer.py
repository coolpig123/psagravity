"""Self-contained hindcast viewer: hindcast/results/viewer.html (data embedded, no server needed).

    python -m hindcast.viewer bug_fire biscar

Uses only what the hindcast already produced: cached inputs (hindcast/data/, no Earth Engine calls), the calibrated
parameters (results/calibration.json) and the scores (results/<fire>_calibration.csv, <fire>_scores.csv). For each
window the model is re-run (same seed) for the +24 h horizon to get arrival_step and burn_prob. No new analyses.
Basemap and look follow real_fires/src/firedata/web.py (shaded relief, same template style).
"""
from __future__ import annotations

import argparse
import base64
import json
from importlib.resources import files

import numpy as np
import pandas as pd
from pyproj import Transformer

from . import run
from .windows import build_windows

SCORE_COLS = ["iou", "precision", "recall", "base_iou", "base_precision", "base_recall", "oracle_iou",
              "pred_new_km2", "obs_new_km2", "steps", "reached_grid_edge"]


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def _rings(feature: dict, grid) -> list[list[list[float]]]:
    """Polygon rings of a GeoJSON feature in grid cell coordinates (col, row; floats, row 0 = north)."""
    tr = Transformer.from_crs(4326, grid.epsg, always_xy=True)
    geom = feature["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    out = []
    for poly in polys:
        for ring in poly:
            xs, ys = tr.transform(*np.asarray(ring, float).T[:2])
            out.append([[round((x - grid.x0) / grid.res, 2), round((grid.y0 - y) / grid.res, 2)]
                        for x, y in zip(xs, ys)])
    return out


def _basemap(grid) -> str:
    try:
        from firedata.web import _basemap as relief
    except ImportError:
        return ""
    tr = Transformer.from_crs(grid.epsg, 3857, always_xy=True)
    xmin, ymin, xmax, ymax = grid.bounds
    xs, ys = tr.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    return relief(min(xs), min(ys), max(xs), max(ys), zoom=12, max_px=1000)


def _scores(fire: str, window_id: str, hps: float, thr: float) -> list[dict]:
    rows = []
    for name in (f"{fire}_calibration.csv", f"{fire}_scores.csv"):
        path = run.RESULTS_DIR / name
        if path.exists():
            t = pd.read_csv(path)
            t = t[(t.window == window_id) & np.isclose(t.hours_per_step, hps) & np.isclose(t.threshold, thr)]
            rows += [{"horizon_h": int(r.horizon_h), "target_utc": str(r.target_utc), "hours": float(r.hours),
                      **{c: (None if pd.isna(r[c]) else round(float(r[c]), 4)) for c in SCORE_COLS}}
                     for _, r in t.iterrows()]
    return list({r["horizon_h"]: r for r in rows}.values())


def window_data(window, hps: float, thr: float) -> dict:
    from firespread.simulate import simulate_spread
    inputs = run.fetch_inputs(window)                 # cached: no Earth Engine call
    result = run.spread_result(inputs, window)
    grid = run.grid_of(result)
    hours = max((t.time - window.start).total_seconds() / 3600 for t in window.targets if t.feature is not None)
    steps = max(1, round(hours / hps))
    sim = simulate_spread(result, steps=steps, runs=run.RUNS, seed=0)
    prob = sim["burn_prob"].values
    arrival = np.nan_to_num(sim["arrival_step"].values, nan=255).clip(0, 255)
    # the wind the model used: the weather step at or before the start (spread_probabilities' choice)
    times = pd.to_datetime(inputs.time.values)
    k = max(int((times <= window.start.tz_localize(None)).sum()) - 1, 0)
    u = float(np.nanmean(inputs.wind_u_ms.isel(time=k)))
    v = float(np.nanmean(inputs.wind_v_ms.isel(time=k)))
    return {
        "id": window.id, "fire": window.fire, "start_utc": window.start.isoformat(),
        "start_km2": round(window.start_area_km2, 2), "w": grid.width, "h": grid.height, "res_m": grid.res,
        "steps": steps, "hours_per_step": hps, "threshold": thr,
        "prob": _b64(np.round(prob * 100)), "arrival": _b64(np.round(arrival)),
        "start_rings": _rings(window.start_feature, grid),
        "targets": [{"horizon_h": t.horizon_h, "time_utc": t.time.isoformat(), "area_km2": round(t.area_km2, 2),
                     "stale": t.stale, "rings": _rings(t.feature, grid)}
                    for t in window.targets if t.feature is not None],
        "wind": {"u": round(u, 2), "v": round(v, 2), "speed": round(float(np.hypot(u, v)), 1),
                 "from_deg": round(float((np.degrees(np.arctan2(-u, -v)) + 360) % 360)),
                 "time_utc": str(times[k])},
        "scores": _scores(window.fire, window.id, hps, thr),
        "basemap": _basemap(grid),
    }


def build(fires: list[str]) -> str:
    cal = json.loads((run.RESULTS_DIR / "calibration.json").read_text())
    hps, thr = float(cal["hours_per_step"]), float(cal["threshold"])
    data = {"calibration": cal, "windows": []}
    for fire in fires:
        for w in build_windows(fire):
            print(f"  {w.id} ...", flush=True)
            data["windows"].append(window_data(w, hps, thr))
    block = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = (files("hindcast") / "templates" / "viewer.html").read_text()
    out = run.RESULTS_DIR / "viewer.html"
    out.write_text(page.replace("__DATA__", block))
    print(f"viewer written to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fires", nargs="+")
    build(ap.parse_args().fires)
