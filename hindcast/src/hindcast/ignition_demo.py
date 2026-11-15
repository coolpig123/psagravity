"""Ignition demo: one website-style prediction from a fire's ignition point and time, against what happened.

The request is the website's (frontend App.tsx -> backend -> firespread.web.predict_payload), with its defaults:
a 200 m square at the official ignition point, start = ignition time, weather_source "auto", hours 6, radius 5 km,
100 m cells, 30 CA steps, NISAR + Sentinel-2 burn scars + FIRMS active front + arrival model on. Nothing re-fitted.
The model's Dataset is captured from inside predict_payload (arrival_fn hook) to score it:
  timed output   p_arrive_<h>h >= 0.5 vs the first 2 FEDS perimeters within 60 h of ignition, h = the model horizon
                 (6 / 12 / 24 / 48 h) nearest to each perimeter's time (Biscar's first FEDS is +50 h, Colwash's +18 h)
  baseline       a circle around the ignition square with the same new area
The CA burn_prob (30 steps, no time unit on the website) is drawn, not scored.

    python -m hindcast.ignition_demo bug_fire
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .metrics import circular_growth, scores
from .replay import CACHE_DIR, OUT_DIR, RES_M, _bald_feds
from .run import grid_of
from .windows import FIRES_DIR, _utc, load_perimeters
from firespread.survival import HORIZONS_H

HALF_M = 100.0            # ignition square: 200 m side (2 x 2 cells)
MAX_TARGET_H = 60.0      # FEDS targets: the first 2 perimeters with new data within 60 h of ignition
ARRIVAL_P = 0.5


def ignition(fire: str) -> tuple[float, float, pd.Timestamp]:
    if fire == "bald_range":
        from firedata.fire_state import FIRES
        f = FIRES[fire]
        return f.ignition_lat, f.ignition_lon, pd.Timestamp("2026-08-07T23:20Z")     # BCWS (README)
    if fire == "biscar":     # config.py decision 2026-11-13: start from the first detections (WFIGS origin 6.6 km off)
        return 40.618, -120.300, pd.Timestamp("2026-07-18T19:57Z")
    from firedata.config import FIRES
    f = FIRES[fire]
    return f.ignition_lat, f.ignition_lon, _utc(f.ignition_utc)


def square(lat: float, lon: float) -> dict:
    dlat = HALF_M / 111_320
    dlon = HALF_M / (111_320 * np.cos(np.radians(lat)))
    ring = [[lon - dlon, lat - dlat], [lon + dlon, lat - dlat], [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
            [lon - dlon, lat - dlat]]
    return {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def perimeters(fire: str) -> list[tuple[pd.Timestamp, dict]]:
    from shapely.geometry import mapping
    if fire == "bald_range":
        g = _bald_feds()
        return [(t, {"type": "Feature", "geometry": mapping(geom), "properties": {}}) for t, geom in zip(g.t, g.geometry)]
    return [(_utc(f["properties"]["obs_time_utc"]), f) for f in load_perimeters(fire)
            if f["properties"].get("is_new", True)]


def run(fire: str) -> pd.DataFrame:
    from firespread.survival import predict_arrival
    from firespread.web import predict_payload
    lat, lon, t0 = ignition(fire)
    body = {"polygon": square(lat, lon), "start": f"{t0:%Y-%m-%dT%H:%MZ}", "weather_source": "auto", "hours": 6}
    got = {}

    def capture(result, start):
        got["ds"] = predict_arrival(result, start)
        return got["ds"]

    payload = predict_payload(body, arrival_fn=capture)
    ds = got["ds"]
    grid = grid_of(ds)
    start = ds["burning"].values.astype(bool)
    obs, rows = {}, []
    later = [(t, f) for t, f in perimeters(fire) if t0 < t <= t0 + pd.Timedelta(hours=MAX_TARGET_H)][:2]
    for t, feat in later:            # first 2 FEDS perimeters after ignition, model horizon nearest in time
        hours = (t - t0).total_seconds() / 3600
        h = min(HORIZONS_H, key=lambda x: abs(x - hours))
        if h in obs:
            continue
        o = grid.polygon_mask(feat).astype(bool)
        pred = (np.nan_to_num(ds[f"p_arrive_{h}h"].values) >= ARRIVAL_P) | start
        s = scores(pred, o, start, RES_M)
        b = scores(circular_growth(start, int((pred & ~start).sum())), o, start, RES_M)
        obs[h] = (t, o, pred)
        rows.append({"fire": fire, "ignition_utc": f"{t0:%Y-%m-%d %H:%M}", "horizon_h": h,
                     "target_utc": f"{t:%Y-%m-%d %H:%M}", "hours": round((t - t0).total_seconds() / 3600, 1),
                     "obs_new_km2": s["obs_new_km2"], "pred_new_km2": s["pred_new_km2"],
                     "iou": s["iou"], "precision": s["precision"], "recall": s["recall"],
                     "circle_iou": b["iou"], "circle_precision": b["precision"], "circle_recall": b["recall"],
                     "expected_new_km2": float(np.nansum(ds[f"p_arrive_{h}h"].values)) * RES_M ** 2 / 1e6,
                     "feds_km2_in_grid": float(o.sum()) * RES_M ** 2 / 1e6,
                     "weather": ds.attrs.get("weather_source"), "reached_grid_edge": int(ds.attrs["sim_reached_grid_edge"]),
                     "active_detections": ds.attrs.get("active_detections", 0),
                     "notes": " | ".join(payload["summary"]["notes"])})
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{fire}_ignition.csv", index=False, float_format="%.4f")
    (CACHE_DIR / fire).mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / fire / "ignition_payload_summary.json").write_text(json.dumps(payload["summary"], default=str, indent=1))
    figure(fire, ds, start, obs, t0)
    return df


def figure(fire, ds, start, obs, t0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bp = ds["burn_prob"].values
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ext = [(float(ds.x[0]) - float(ds.x[0])) / 1e3, (float(ds.x[-1]) - float(ds.x[0])) / 1e3,
           (float(ds.y[-1]) - float(ds.y[0])) / 1e3, 0]
    kw = dict(extent=ext, origin="upper")
    ax[0].imshow(ds["elevation_m"].values, cmap="terrain", **kw)
    ax[0].contour(start, [0.5], colors="r", linewidths=2, **kw)
    ax[0].set_title(f"ignition {t0:%m-%d %H:%M} UTC (red square), terrain", fontsize=9)
    ax[1].imshow(np.where(start, np.nan, bp), cmap="Oranges", vmin=0, vmax=1, **kw)
    cols = dict(zip(sorted(obs), ("tab:blue", "tab:purple")))
    for h, c in cols.items():
        p = np.nan_to_num(ds[f"p_arrive_{h}h"].values) >= ARRIVAL_P
        if p.any():
            ax[1].contour(p.astype(float), [0.5], colors=c, linewidths=1.3, **kw)
    ax[1].set_title(f"website: CA burn prob (30 steps); arrival p≥{ARRIVAL_P:g} "
                    + ", ".join(f"{c[4:]} {h} h" for h, c in cols.items()) + "\n"
                    f"weather {ds.attrs.get('weather_source')}", fontsize=9)
    ax[2].imshow(np.zeros_like(bp), cmap="Greys", vmin=0, vmax=1, **kw)
    fills = dict(zip(sorted(obs), ("tab:orange", "tab:red")))
    for h in sorted(obs, reverse=True):
        t, o, pred = obs[h]
        ax[2].contourf(o.astype(float), [0.5, 1.5], colors=[fills[h]], alpha=0.45, **kw)
        ax[2].contour(pred.astype(float), [0.5], colors=cols[h], linewidths=1, **kw)
    ax[2].set_title("observed FEDS: " + ", ".join(f"{fills[h][4:]} {obs[h][0]:%m-%d %H:%M} (+{(obs[h][0] - t0).total_seconds() / 3600:.0f} h)"
                                                for h in sorted(obs)) if obs else "no FEDS within 60 h", fontsize=9)
    for a in ax:
        a.set_xlabel("km")
    fig.suptitle(f"{fire}: website-style prediction from the ignition point vs what happened "
                 f"(grid = website default, 5 km around the square)", fontsize=10)
    fig.tight_layout()
    path = FIRES_DIR / fire / "figures" / "ignition_demo.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fires", nargs="+")
    for fire in ap.parse_args(argv).fires:
        df = run(fire)
        if df.empty:
            print(fire, "no FEDS target"); continue
        print(df.drop(columns=["notes"]).round(3).to_string(index=False))
        print("notes:", df["notes"].iloc[0] if len(df) else "")


if __name__ == "__main__":
    main()
