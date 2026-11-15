"""Demo: radar fire state as the spread model's starting perimeter (Bald Range, BC, 2026).

Start: fire_state("bald_range", 08-10 02:00 UTC) = RCM SC30MCPA 08-10 01:34 (2 days after ignition; no NISAR pass
since 07-27 and no FEDS used). Model: firespread.predict (Earth Engine inputs, no NISAR, no optical scars) +
simulate_spread, with the hindcast calibration from the Bug Fire held fixed (2 h per CA step, burn_prob >= 0.9;
hindcast/results/calibration.json). Targets: FEDS perimeters (fireid 11710), the radar state at 08-22
(RCM 08-22 + NISAR A77 08-18) and the BCWS final perimeter.
Scores on the NEWLY burned area (target minus start), as in hindcast/metrics.py, vs the same-area circle baseline
(start grown by the same distance everywhere until it adds as many cells as the model).

Run from real_fires/ (Earth Engine login, Earthdata in .env):  python scripts/fire_state_demo.py
"""
from __future__ import annotations

import json

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import requests
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from firedata.config import REPO_ROOT
from firedata.fire_state import fire_state
from firespread import predict
from firespread.simulate import simulate_spread

D = REPO_ROOT / "data" / "fires" / "bald_range"
T0 = pd.Timestamp("2026-08-10T02:00Z")
# Hindcast calibration on the Bug Fire, held fixed (hindcast/results/calibration.json: 2 h per step, >= 0.9).
CALIBRATION = json.loads((REPO_ROOT.parent / "hindcast" / "results" / "calibration.json").read_text())
HOURS_PER_STEP, THRESHOLD, RUNS = float(CALIBRATION["hours_per_step"]), float(CALIBRATION["threshold"]), 200
FEDS_ID = 11710.0
FEDS_TARGET_HOURS = (24, 48, 96)                      # FEDS perimeters nearest to T0 + these hours


def scores(pred, obs, start):
    """IoU / precision / recall on the newly burned area (copied from hindcast/metrics.py)."""
    p, o = pred & ~start, obs & ~start
    tp, union = float((p & o).sum()), float((p | o).sum())
    return {"iou": tp / union if union else np.nan, "precision": tp / p.sum() if p.sum() else np.nan,
            "recall": tp / o.sum() if o.sum() else np.nan, "pred_new_ha": float(p.sum()), "obs_new_ha": float(o.sum())}


def circular_growth(start, n_new):
    """Same-area circle baseline (copied from hindcast/metrics.py)."""
    if n_new <= 0 or not start.any():
        return start.copy()
    dist = distance_transform_edt(~start)
    out = dist[~start]
    n = min(int(n_new), out.size)
    return start | (dist <= np.partition(out, n - 1)[n - 1])


def feds_perimeters():
    r = requests.get("https://openveda.cloud/api/features/collections/public.eis_fire_lf_perimeter_nrt/items",
                     params={"bbox": "-120.1,49.45,-119.6,49.8", "datetime": "2026-08-08T00:00:00Z/2026-08-30T00:00:00Z",
                             "limit": 200}, timeout=120)
    r.raise_for_status()
    g = gpd.GeoDataFrame.from_features(r.json()["features"], crs=4326)
    g = g[g.fireid == FEDS_ID].copy()
    g["t"] = pd.to_datetime(g.t, utc=True)
    return g.sort_values("t")


def main():
    state = fire_state("bald_range", T0)
    print("start state:", state.area_ha, "ha from", state.scenes)
    result = predict(polygon=state.polygon_lonlat, start=T0, radius_km=15, res_m=100, nisar=None, burn_scars=False)
    x, y = result["x"].values, result["y"].values
    res = float(abs(x[1] - x[0]))
    crs = int(str(result.attrs["crs"]).split(":")[-1])
    tr = from_origin(x[0] - res / 2, y[0] + res / 2, res, res)
    sh = (y.size, x.size)
    raster = lambda geom: rasterize([geom], out_shape=sh, transform=tr).astype(bool)
    start = result["burning"].values.astype(bool)

    targets = []
    feds = feds_perimeters()
    for h in FEDS_TARGET_HOURS:
        row = feds.iloc[(feds.t - (T0 + pd.Timedelta(hours=h))).abs().argsort().iloc[0]]
        if row.t > T0:
            targets.append((f"FEDS {row.t:%m-%d %H:%M}", row.t, raster(gpd.GeoSeries([row.geometry], crs=4326).to_crs(crs).iloc[0])))
    later = fire_state("bald_range", "2026-08-22T02:00Z")
    targets.append(("Radar state 08-22 (RCM + NISAR)", pd.Timestamp("2026-08-22T01:34Z"),
                    raster(gpd.GeoSeries([later.polygon_lonlat], crs=4326).to_crs(crs).iloc[0])))
    bcws = gpd.read_file(D / "processed" / "bcws_final_perimeter.geojson")
    targets.append(("BCWS final", pd.Timestamp(bcws.poly_PolygonDateTime.iloc[0]), raster(bcws.to_crs(crs).geometry.iloc[0])))

    rows, sims = [], {}
    for name, t, obs in targets:
        steps = max(1, round((t - T0).total_seconds() / 3600 / HOURS_PER_STEP))
        if steps not in sims:
            sims[steps] = simulate_spread(result, steps=steps, runs=RUNS, seed=0)["burn_prob"].values
        pred = (sims[steps] >= THRESHOLD) | start
        circ = circular_growth(start, int((pred & ~start).sum()))
        for method, m in (("model", pred), ("circle", circ)):
            rows.append({"target": name, "target_utc": f"{t:%Y-%m-%d %H:%M}", "steps": steps, "method": method,
                         **{k: round(v, 3) if k in ("iou", "precision", "recall") else v
                            for k, v in scores(m, obs, start).items()}})
    df = pd.DataFrame(rows)
    df.to_csv(D / "processed" / "fire_state_demo_scores.csv", index=False)
    (D / "processed" / "fire_state_demo.json").write_text(json.dumps(
        {"start_utc": str(T0), "start_area_ha": state.area_ha, "start_scenes": state.scenes,
         "hours_per_step": HOURS_PER_STEP, "threshold": THRESHOLD, "runs": RUNS}, indent=1))
    gpd.GeoSeries([state.polygon_lonlat], crs=4326).to_file(D / "processed" / "fire_state_0810.geojson", driver="GeoJSON")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    # Figure: start state | prediction (to the BCWS final date) | observed later
    name, t, obs = targets[1]          # FEDS ~48 h later: long horizons burn the whole window (see README)
    steps = max(1, round((t - T0).total_seconds() / 3600 / HOURS_PER_STEP))
    ext = (tr.c, tr.c + sh[1] * res, tr.f - sh[0] * res, tr.f)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    ax[0].imshow(np.where(start, 1, np.nan), cmap="Reds", vmin=0, vmax=1.2, extent=ext)
    ax[0].set_title(f"Start: radar fire state {T0:%m-%d %H:%M} UTC\n{state.area_ha:,.0f} ha · {state.scenes[0][:40]}", fontsize=9)
    im = ax[1].imshow(np.where(sims[steps] > 0, sims[steps], np.nan), cmap="magma_r", vmin=0, vmax=1, extent=ext)
    ax[1].contour(np.flipud((sims[steps] >= THRESHOLD) | start), levels=[0.5], colors="b", linewidths=0.8,
                  extent=ext, origin="lower")
    r = df[(df.target == name) & (df.method == "model")].iloc[0]
    c = df[(df.target == name) & (df.method == "circle")].iloc[0]
    ax[1].set_title(f"Model burn probability, {steps} steps to {t:%m-%d} (blue: ≥ {THRESHOLD})\n"
                    f"new area IoU {r.iou:.2f} (circle {c.iou:.2f}) · precision {r.precision:.2f} · recall {r.recall:.2f}", fontsize=9)
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="burn probability")
    ax[2].imshow(np.where(obs, 1, np.nan), cmap="Greys", vmin=0, vmax=2, extent=ext)
    ax[2].contour(np.flipud(start), levels=[0.5], colors="r", linewidths=0.8, extent=ext, origin="lower")
    ax[2].set_title(f"Observed later: {name} ({t:%Y-%m-%d})\nred = start state", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Bald Range (BC, 2026): spread model started from the RCM radar fire state", fontsize=11)
    fig.savefig(D / "figures" / "fire_state_demo.png", dpi=100)


if __name__ == "__main__":
    main()
