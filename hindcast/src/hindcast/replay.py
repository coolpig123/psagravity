"""Replay the CURRENT model (master) on past fires, with the weather they really had, and score it.

Model, as on the website (firespread.web.predict_payload), with nothing re-fitted:
  predict(inputs)                     one-step Alexandridis spread probabilities, p_spread_time over every weather hour
  simulate_spread(time_developing)    Monte Carlo CA with the hourly weather interpolated along the run
  predict_arrival(attach_active_front) survival arrival-time model, active front = VIIRS of the 12 h before the start
Two prediction masks per target, each with its same-area circle baseline (hindcast.metrics.circular_growth):
  CA      : burn_prob >= threshold after round(hours / hours_per_step) steps. Both knobs from results/calibration.json
            (fitted on Bug 08-09 09:30 only, before this replay; that window is left out here).
  arrival : p_arrive_<12|24>h >= 0.5 (no knob fitted).
Weather: MERRA-2 (GES DISC, hourly, 0.5 x 0.625 deg) put in place of the RTMA fields of the cached Earth Engine
inputs, bilinear in space and linear in time. `--weather rtma` keeps the RTMA fields (sensitivity run).

Starts: FEDS perimeters with new data (hindcast.windows); Bald Range (Canada, no FEDS start): the radar fire state
(firedata.fire_state, RCM / NISAR) scored against the FEDS NRT perimeters (fireid 11710).

    python -m hindcast.replay                    # all fires, MERRA-2
    python -m hindcast.replay --weather rtma     # same windows, RTMA
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .metrics import circular_growth, scores
from .run import CACHE_DIR, RESULTS_DIR, RES_M, RUNS, fetch_inputs, grid_of
from .windows import FIRES_DIR, Target, Window, build_windows

log = logging.getLogger("replay")
CAL = json.loads((RESULTS_DIR / "calibration.json").read_text())
HOURS_PER_STEP, THRESHOLD = float(CAL["hours_per_step"]), float(CAL["threshold"])
ARRIVAL_P = 0.5
OUT_DIR = RESULTS_DIR / "replay"

# Window choice, fixed before any score: the first N windows with new data, the calibration window left out.
PLAN = {"bug_fire": 3, "biscar": 3, "colwash": 3, "crosswhite": 2}
SKIP = {CAL["window"]}

# Bald Range: radar starts (RCM 08-10 01:34; RCM 08-10 + NISAR D172 08-13 03:12), FEDS NRT targets
BALD_STARTS = ("2026-08-10T02:00Z", "2026-08-13T04:00Z")
BALD_FEDS_ID = 11710.0
BALD_TOLERANCE_H = 4.5        # FEDS NRT is at 00 / 12 UTC; the 08-13 start has targets at +8 h / +20 h


# ---- windows -----------------------------------------------------------------------------------

def feds_windows(fire: str) -> list[Window]:
    ws = [w for w in build_windows(fire) if w.id not in SKIP]
    return ws[:PLAN[fire]]


def _bald_feds() -> "pd.DataFrame":
    import geopandas as gpd
    import requests
    path = CACHE_DIR / "bald_range" / "feds_nrt_11710.geojson"
    if not path.exists():
        r = requests.get("https://openveda.cloud/api/features/collections/public.eis_fire_lf_perimeter_nrt/items",
                         params={"bbox": "-120.1,49.45,-119.6,49.8",
                                 "datetime": "2026-08-08T00:00:00Z/2026-08-30T00:00:00Z", "limit": 200}, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    g = gpd.read_file(path)
    g = g[g.fireid == BALD_FEDS_ID].copy()
    g["t"] = pd.to_datetime(g.t, utc=True)
    return g.sort_values("t").drop_duplicates("t")


def bald_windows() -> list[Window]:
    from shapely.geometry import mapping
    from firedata.fire_state import fire_state
    feds = _bald_feds()
    out = []
    for s in BALD_STARTS:
        t0 = pd.Timestamp(s)
        cache = CACHE_DIR / "bald_range" / f"state_{t0:%m%d_%H%M}.geojson"
        if cache.exists():
            feat = json.loads(cache.read_text())
        else:
            st = fire_state("bald_range", t0)
            feat = {"type": "Feature", "geometry": mapping(st.polygon_lonlat),
                    "properties": {"area_ha": st.area_ha, "scenes": "; ".join(st.scenes)}}
            cache.write_text(json.dumps(feat))
        w = Window("bald_range", t0, feat, feat["properties"]["area_ha"] / 100)
        for h in (12, 24):
            d = (feds.t - (t0 + pd.Timedelta(hours=h))).abs()
            row = feds.loc[d.idxmin()]
            ok = d.min() <= pd.Timedelta(hours=BALD_TOLERANCE_H) and row.t > t0
            w.targets.append(Target(h, row.t, {"type": "Feature", "geometry": mapping(row.geometry), "properties": {}}
                                    if ok else None, float(row.farea), stale=False))
        out.append(w)
    return out


# ---- weather -----------------------------------------------------------------------------------

def with_merra2(ds: xr.Dataset, fire: str) -> xr.Dataset:
    """Replace the gridded weather of the inputs by MERRA-2 at the same hours (cached per fire)."""
    from firedata import merra2
    lat, lon = ds["lat"].values, ds["lon"].values
    times = pd.DatetimeIndex(ds.time.values)
    bbox = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    m = merra2.open_box(bbox, times[0] - pd.Timedelta(hours=1), times[-1] + pd.Timedelta(hours=1),
                        margin_deg=1.0, cache_dir=CACHE_DIR / fire / "merra2")
    m = m[["U10M", "V10M", "temp_c", "rh_pct"]].interp(time=times.values, kwargs={"fill_value": "extrapolate"})
    m = m.interp(lat=xr.DataArray(lat, dims=("y", "x")), lon=xr.DataArray(lon, dims=("y", "x")))
    out = ds.drop_vars([v for v in ("gust_ms", "precip_mm") if v in ds])
    for name, src in (("wind_u_ms", "U10M"), ("wind_v_ms", "V10M"), ("temp_c", "temp_c"), ("rh_pct", "rh_pct")):
        out[name] = (("time", "y", "x"), m[src].transpose("time", "y", "x").values.astype(np.float32))
    u, v = out["wind_u_ms"], out["wind_v_ms"]
    out["wind_speed_ms"] = np.hypot(u, v).astype(np.float32)
    out["wind_dir_deg"] = ((np.degrees(np.arctan2(-u, -v)) + 360) % 360).astype(np.float32)
    out.attrs.update(weather_source="merra2", weather_dataset="NASA GMAO MERRA-2 M2T1NXSLV (GES DISC), bilinear")
    return out


# ---- one window --------------------------------------------------------------------------------

def viirs_before(fire: str, t0: pd.Timestamp) -> pd.DataFrame | None:
    path = FIRES_DIR / fire / "processed" / "fire_points.parquet"
    if not path.exists():
        return None
    p = pd.read_parquet(path, columns=["time_utc", "lat", "lon", "source"])
    p = p[p.source.str.startswith("VIIRS")]
    return p[pd.to_datetime(p.time_utc, utc=True) <= t0]      # attach_active_front keeps the last 12 h


@dataclass
class Run:
    window: Window
    start: np.ndarray
    burn_prob: dict          # horizon -> (y, x)
    p_arrive: dict           # horizon -> (y, x)
    obs: dict                # horizon -> (y, x) bool
    extent: tuple
    rows: list


def run_window(w: Window, weather: str) -> Run:
    from firespread import predict
    from firespread.simulate import simulate_spread
    from firespread.survival import attach_active_front, predict_arrival
    ds = fetch_inputs(w)                                   # Earth Engine, cached in hindcast/data/<fire>/
    if weather == "merra2":
        ds = with_merra2(ds, w.fire)
    result = predict(inputs=ds, time=w.start, nisar=None, burn_scars=False)
    result = attach_active_front(result, viirs_before(w.fire, w.start), w.start)
    arr = predict_arrival(result, w.start)
    grid = grid_of(result)
    start = result["burning"].values.astype(bool)
    run = Run(w, start, {}, {}, {}, (float(result.x[0]), float(result.x[-1]), float(result.y[-1]), float(result.y[0])), [])
    for t in w.targets:
        if t.feature is None or t.stale:
            continue
        hours = (t.time - w.start).total_seconds() / 3600
        steps = max(1, round(hours / HOURS_PER_STEP))
        sim = simulate_spread(result, steps=steps, runs=RUNS, seed=0, time_developing=True, step_hours=HOURS_PER_STEP)
        obs = grid.polygon_mask(t.feature).astype(bool)
        bp = sim["burn_prob"].values
        pa = np.nan_to_num(arr[f"p_arrive_{t.horizon_h}h"].values)
        run.burn_prob[t.horizon_h], run.p_arrive[t.horizon_h], run.obs[t.horizon_h] = bp, pa, obs
        n_obs = int((obs & ~start).sum())
        oracle = scores(circular_growth(start, n_obs), obs, start, RES_M)["iou"]
        for method, pred in (("CA", (bp >= THRESHOLD) | start), ("arrival", (pa >= ARRIVAL_P) | start)):
            s = scores(pred, obs, start, RES_M)
            b = scores(circular_growth(start, int((pred & ~start).sum())), obs, start, RES_M)
            run.rows.append({
                "fire": w.fire, "window": w.id, "start_utc": f"{w.start:%Y-%m-%d %H:%M}",
                "start_km2": round(float(start.sum()) * RES_M ** 2 / 1e6, 1), "horizon_h": t.horizon_h,
                "target_utc": f"{t.time:%Y-%m-%d %H:%M}", "hours": round(hours, 1), "method": method,
                "iou": s["iou"], "precision": s["precision"], "recall": s["recall"],
                "pred_new_km2": s["pred_new_km2"], "obs_new_km2": s["obs_new_km2"],
                "circle_iou": b["iou"], "circle_precision": b["precision"], "circle_recall": b["recall"],
                "beats_circle": s["iou"] > b["iou"], "oracle_circle_iou": oracle,
                "weather": result.attrs.get("weather_source", weather),
                "time_developing": int(sim.attrs["sim_time_developing"]),
                "reached_grid_edge": int(sim.attrs["sim_reached_grid_edge"]),
                "active_front_km2": result.attrs.get("active_front_km2", np.nan)})
    return run


# ---- outputs -----------------------------------------------------------------------------------

def figure(fire: str, runs: list[Run], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(r, h) for r in runs for h in sorted(r.obs)]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.8 * len(rows)), squeeze=False)
    for i, (r, h) in enumerate(rows):
        start, obs = r.start, r.obs[h]
        ca, ar = (r.burn_prob[h] >= THRESHOLD) | start, (r.p_arrive[h] >= ARRIVAL_P) | start
        box = start | obs | ca | ar
        ys, xs = np.nonzero(box)
        pad = 15
        sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
        km = lambda a: a[sl]
        t = next(t for t in r.window.targets if t.horizon_h == h)
        ax = axes[i]
        ax[0].imshow(km(start), cmap="Greys", vmin=0, vmax=1.5)
        ax[0].set_title(f"start {r.window.start:%m-%d %H:%M} UTC ({start.sum() * RES_M ** 2 / 1e6:.1f} km²)", fontsize=9)
        ax[1].imshow(np.where(km(start), np.nan, km(r.burn_prob[h])), cmap="Oranges", vmin=0, vmax=1)
        ax[1].contour(km(start), [0.5], colors="k", linewidths=0.8)
        ax[1].contour(km(ar).astype(float), [0.5], colors="tab:blue", linewidths=1.2)
        ax[1].contour(km(ca).astype(float), [0.5], colors="tab:red", linewidths=1.2)
        ax[1].set_title(f"+{h} h: CA burn prob; red CA ≥ {THRESHOLD:g}, blue arrival p ≥ {ARRIVAL_P:g}", fontsize=9)
        ax[2].imshow(km(obs & ~start), cmap="Reds", vmin=0, vmax=1.5)
        ax[2].contour(km(start), [0.5], colors="k", linewidths=0.8)
        ax[2].contour(km(ca).astype(float), [0.5], colors="tab:red", linewidths=0.8, linestyles="--")
        ax[2].contour(km(ar).astype(float), [0.5], colors="tab:blue", linewidths=0.8, linestyles="--")
        ax[2].set_title(f"observed {t.time:%m-%d %H:%M} (new area red, {(obs & ~start).sum() * RES_M ** 2 / 1e6:.1f} km²)",
                        fontsize=9)
        for a in ax:
            a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"{fire}: replay of the current model (100 m cells; black = start perimeter)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    cols = ["fire", "start_utc", "start_km2", "horizon_h", "hours", "obs_new_km2", "method", "pred_new_km2",
            "iou", "precision", "recall", "circle_iou", "oracle_circle_iou"]
    d = df[cols].copy()
    for c in ("iou", "precision", "recall", "circle_iou", "oracle_circle_iou"):
        d[c] = (100 * d[c]).round(1)
    for c in ("obs_new_km2", "pred_new_km2"):
        d[c] = d[c].round(1)
    d["iou"] = [f"**{v}**" if b else f"{v}" for v, b in zip(d.iou, df.beats_circle)]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fires", nargs="*", default=[*PLAN, "bald_range"])
    ap.add_argument("--weather", choices=("merra2", "rtma"), default="merra2")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for fire in a.fires:
        windows = bald_windows() if fire == "bald_range" else feds_windows(fire)
        runs = []
        for w in windows:
            log.info("%s (%s)", w.id, a.weather)
            runs.append(run_window(w, a.weather))
            rows += runs[-1].rows
        if not a.no_figures and a.weather == "merra2":
            figure(fire, runs, FIRES_DIR / fire / "figures" / "replay_current_model.png")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"replay_{a.weather}.csv", index=False, float_format="%.4f")
    (OUT_DIR / f"replay_{a.weather}.md").write_text(markdown_table(df) + "\n")
    print(markdown_table(df))


if __name__ == "__main__":
    main()
