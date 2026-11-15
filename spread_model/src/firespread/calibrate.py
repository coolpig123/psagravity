"""First calibration of the Alexandridis parameters on real fires (FEDS 12 h perimeters).

Each training example is one FEDS segment: start from the perimeter seen at t0, predict the perimeter at t1
(12 or 24 h later), score the overlap. Inputs (Earth Engine only: terrain, land cover, weather near the start of
the segment) are fetched once per segment and cached in spread_model/data/calibration/. NISAR and Sentinel-2 are
off (the burned area so far is the start perimeter itself).

Fitted (grid search): p0, a wind multiplier on c1 and c2, and steps_per_12h (how many CA steps one 12 h period
is). Slope and moisture keep the paper values. Score: soft IoU between P(burned) (start perimeter = 1) and the next
FEDS perimeter. Baseline: "no growth" (the next perimeter = the start one).

    python -m firespread.calibrate fetch bug_fire colwash biscar
    python -m firespread.calibrate fit --train bug_fire --test colwash biscar
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .alexandridis import DIRECTIONS, AlexandridisParams, spread_probabilities

ROOT = Path(__file__).resolve().parents[3]                     # repo root
FIRES_DIR = ROOT / "real_fires" / "data" / "fires"
OUT_DIR = ROOT / "spread_model" / "data" / "calibration"
RES_M = 400.0
RADIUS_KM = 12.0
RUNS = 40
MIN_GROWTH_KM2 = 0.5          # skip segments where FEDS saw (almost) no growth: nothing to learn from
AFTERNOON_UTC_HOUR = 21       # weather at the afternoon peak of spread (21 UTC = 14:00 US Pacific)

GRID = {"p0": [0.05, 0.1, 0.2, 0.35, 0.58, 0.8],      # 0.58 = the paper's value
        "wind": [0.5, 1.0, 2.0, 3.0],
        "steps_per_12h": [1, 2, 4, 8, 16, 32]}
USE_ACTIVE_FRONT = True  # spread only from cells VIIRS saw burning (else from the whole start perimeter)
METRIC = "growth_iou"   # soft IoU on the new burned area only (outside the start perimeter); "no growth" scores 0


def segments(fire: str) -> list[dict]:
    """Consecutive FEDS perimeters that grew: [{fire, i, t0, t1, hours, start_geom, end_geom, growth_km2}]."""
    import geopandas as gpd
    per = gpd.read_file(FIRES_DIR / fire / "processed" / "feds_perimeters_12h.geojson")
    per = per[per["is_new"].astype(bool)].sort_values("obs_time_utc").reset_index(drop=True)
    per["obs_time_utc"] = pd.to_datetime(per["obs_time_utc"], utc=True)
    out = []
    for i in range(len(per) - 1):
        a, b = per.iloc[i], per.iloc[i + 1]
        growth = float(b["area_km2"] - a["area_km2"])
        if growth < MIN_GROWTH_KM2:
            continue
        out.append({"fire": fire, "i": i, "t0": a["obs_time_utc"], "t1": b["obs_time_utc"],
                    "hours": (b["obs_time_utc"] - a["obs_time_utc"]).total_seconds() / 3600,
                    "start_geom": a.geometry, "end_geom": b.geometry, "growth_km2": growth,
                    "area0_km2": float(a["area_km2"]), "area1_km2": float(b["area_km2"])})
    return out


def _cache_path(seg: dict) -> Path:
    return OUT_DIR / seg["fire"] / f"seg{seg['i']:02d}_{seg['t0']:%Y%m%dT%H%M}.nc"


def weather_time(seg: dict) -> pd.Timestamp:
    """The afternoon peak inside the segment (AFTERNOON_UTC_HOUR; US West = 14:00 local), else just before t1."""
    t = seg["t0"].normalize() + pd.Timedelta(hours=AFTERNOON_UTC_HOUR)
    while t <= seg["t0"]:
        t += pd.Timedelta(days=1)
    return t if t < seg["t1"] else seg["t1"] - pd.Timedelta(hours=1)


def _target(ds: xr.Dataset, geom) -> np.ndarray:
    """The end perimeter rasterised on the model grid (1 inside)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    res = float(ds.attrs["res_m"])
    x, y = ds.x.values, ds.y.values
    transform = from_origin(x.min() - res / 2, y.max() + res / 2, res, res)
    g = gpd.GeoSeries([geom], crs=4326).to_crs(ds.attrs["crs"]).iloc[0]
    arr = rasterize([(g, 1)], out_shape=(y.size, x.size), transform=transform, all_touched=False, dtype="uint8")
    if y[0] < y[-1]:                      # grid stored south -> north
        arr = arr[::-1]
    return arr.astype(bool)


def fetch(fire: str, max_segments: int | None = None) -> list[Path]:
    """Earth Engine inputs + target for every segment of a fire (cached)."""
    from shapely.geometry import mapping

    from geelayers import get_fire_inputs
    from .predict import MODEL_LAYERS
    paths = []
    for seg in segments(fire)[:max_segments]:
        path = _cache_path(seg)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            when = weather_time(seg).isoformat()
            ds = get_fire_inputs(polygon=mapping(seg["start_geom"]), start=when, hours=1.0,
                                 radius_km=RADIUS_KM, res_m=RES_M, layers=MODEL_LAYERS)
            ds["target"] = (("y", "x"), _target(ds, seg["end_geom"]).astype(np.int8))
            ds.attrs.update({"seg_hours": seg["hours"], "seg_t0": str(seg["t0"]), "seg_t1": str(seg["t1"]),
                             "seg_growth_km2": seg["growth_km2"], "fire": fire})
            for k, v in list(ds.attrs.items()):
                if not isinstance(v, (str, int, float, np.number)):
                    ds.attrs[k] = str(v)
            ds.to_netcdf(path)
            print(f"  {fire} seg {seg['i']}: {seg['t0']:%m-%d %H:%M} -> {seg['t1']:%m-%d %H:%M} "
                  f"(+{seg['growth_km2']:.1f} km2), grid {ds.sizes['y']}x{ds.sizes['x']}", flush=True)
        paths.append(path)
    return paths


ACTIVE_HOURS = 12.0           # VIIRS detections this long before the segment start = the active front
_FIRE_POINTS: dict[str, pd.DataFrame] = {}


def _viirs_points(fire: str) -> pd.DataFrame:
    if fire not in _FIRE_POINTS:
        proc = FIRES_DIR / fire / "processed"
        pts = pd.read_parquet(proc / "fire_points.parquet") if (proc / "fire_points.parquet").exists() \
            else pd.read_csv(proc / "fire_points.csv")
        pts["time_utc"] = pd.to_datetime(pts["time_utc"], utc=True)
        pts = pts[pts["source"].astype(str).str.contains("VIIRS", case=False)]
        if "ambiguous" in pts:
            pts = pts[~pts["ambiguous"].astype(bool)]
        _FIRE_POINTS[fire] = pts
    return _FIRE_POINTS[fire]


def active_front(ds: xr.Dataset, fire: str, t0: pd.Timestamp) -> np.ndarray | None:
    """Cells with a VIIRS detection in the ACTIVE_HOURS before t0 (where the fire was really burning), or None."""
    from .active_front import points_to_grid, project_front
    pts = _viirs_points(fire)
    pts = pts[(pts["time_utc"] > t0 - pd.Timedelta(hours=ACTIVE_HOURS)) & (pts["time_utc"] <= t0)]
    if pts.empty:
        return None
    out, _ = points_to_grid(ds, pts["lat"].values, pts["lon"].values)
    if out.any() and "initial_burned" in ds:
        # FEDS buffers VIIRS detections by ~500 m when building the start polygon, leaving ~2/3 of VIIRS cell centres
        # 1 cell inside `start` with no unburned neighbours. Activate adjacent outer-rim cells of `start` as well.
        out = project_front(ds["initial_burned"].values, out)
    return out if out.any() and float(ds.attrs["res_m"]) > 0 else None


def first_burn_step(p_spread: np.ndarray, start: np.ndarray, max_steps: int, runs: int = RUNS,
                    seed: int = 0, front: np.ndarray | None = None, spot: dict | None = None) -> np.ndarray:
    """(runs, h, w) int16: CA step at which each cell first burned in each run (0 = start, 32767 = never).

    `front`: the cells that spread at step 1 (default: all of `start`); the rest of `start` is burned out.
    `spot`: spotting as in simulate.py: {launch, ignite, dist_m, dir_deg (y, x arrays), min_m, max_m, cone_deg,
    res_m}. A newly burning cell launches one firebrand with probability launch; it lands min_m + Exp(dist_m - min_m)
    (at most max_m) downwind within a normal cone, and ignites the landing cell with probability ignite."""
    never = np.int16(32767)
    h, w = start.shape
    rng = np.random.default_rng(seed)
    first = np.full((runs, h, w), never, np.int16)
    first[:, start] = 0
    lit = start if front is None else front
    first[:, lit] = 0
    fr, fy, fx = np.nonzero(np.broadcast_to(lit, (runs, h, w)))
    for step in range(1, max_steps + 1):
        if fr.size == 0:
            break
        nr, ny, nx = [], [], []
        for k, (_, dr, dc, _) in enumerate(DIRECTIONS):
            ty, tx = fy + dr, fx + dc
            ok = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
            r, sy, sx, ty, tx = fr[ok], fy[ok], fx[ok], ty[ok], tx[ok]
            go = rng.random(r.size, dtype=np.float32) < p_spread[k, sy, sx]
            nr.append(r[go]), ny.append(ty[go]), nx.append(tx[go])
        if spot is not None:
            out = rng.random(fr.size, dtype=np.float32) < spot["launch"][fy, fx]
            if out.any():
                sr, sy, sx = fr[out], fy[out], fx[out]
                bearing = np.radians(spot["dir_deg"][sy, sx] + rng.normal(0.0, spot["cone_deg"], sr.size))
                scale = np.maximum(spot["dist_m"][sy, sx] - spot["min_m"], 1.0)
                dist = np.minimum(spot["min_m"] + rng.exponential(scale), spot["max_m"])
                ty = sy + np.rint(-dist * np.cos(bearing) / spot["res_m"]).astype(np.int64)   # rows run north -> south
                tx = sx + np.rint(dist * np.sin(bearing) / spot["res_m"]).astype(np.int64)
                ok = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
                sr, ty, tx = sr[ok], ty[ok], tx[ok]
                lit_spot = rng.random(sr.size, dtype=np.float32) < spot["ignite"][ty, tx]
                nr.append(sr[lit_spot]), ny.append(ty[lit_spot]), nx.append(tx[lit_spot])
        r, y, x = np.concatenate(nr), np.concatenate(ny), np.concatenate(nx)
        keep = first[r, y, x] == never
        flat = np.unique(((r[keep].astype(np.int64) * h + y[keep]) * w + x[keep]))
        r, rest = np.divmod(flat, h * w)
        y, x = np.divmod(rest, w)
        first[r, y, x] = step
        fr, fy, fx = r, y, x
    return first


def soft_iou(prob: np.ndarray, target: np.ndarray) -> float:
    t = target.astype(np.float32)
    return float(np.minimum(prob, t).sum() / max(np.maximum(prob, t).sum(), 1e-9))


def evaluate(paths: list[Path], combos: list[dict]) -> pd.DataFrame:
    """Soft IoU of every parameter combination on every segment (one simulation per (segment, p0, wind))."""
    rows = []
    max_s12 = max(c["steps_per_12h"] for c in combos)
    for path in paths:
        ds = xr.load_dataset(path)
        target = ds["target"].values.astype(bool)
        start = ds["initial_burned"].values.astype(bool)
        hours = float(ds.attrs["seg_hours"])
        base = AlexandridisParams()
        new = ~start                                            # the growth region
        front = active_front(ds, ds.attrs["fire"], pd.Timestamp(ds.attrs["seg_t0"]))
        n_front = 0 if front is None else int(front.sum())
        if front is None:
            front = start

        def score(prob):
            return {"iou": soft_iou(prob, target), "growth_iou": soft_iou(prob[new], target[new])}
        rows.append({"segment": path.stem, "fire": ds.attrs["fire"], "hours": hours, "p0": np.nan,
                     "wind": np.nan, "steps_per_12h": 0, **score(start.astype(np.float32))})
        for p0, wind in sorted({(c["p0"], c["wind"]) for c in combos}):
            params = dataclasses.replace(base, p0=p0, c1=base.c1 * wind, c2=base.c2 * wind)
            res = spread_probabilities(ds.drop_vars("target"), params=params)
            p = np.nan_to_num(res["p_spread"].values.astype(np.float32))
            max_steps = int(round(max_s12 * hours / 12))
            first = first_burn_step(p, start | front, max_steps, front=front if USE_ACTIVE_FRONT else None)
            for s12 in sorted({c["steps_per_12h"] for c in combos}):
                steps = max(1, int(round(s12 * hours / 12)))
                prob = (first <= steps).mean(axis=0)
                rows.append({"segment": path.stem, "fire": ds.attrs["fire"], "hours": hours, "p0": p0, "wind": wind,
                             "steps_per_12h": s12, **score(prob)})
        print(f"  scored {ds.attrs['fire']} {path.stem}: active front {n_front} of {int(start.sum())} cells",
              flush=True)
    return pd.DataFrame(rows)


def fit(train: list[str], test: list[str]) -> dict:
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    paths = {f: sorted((OUT_DIR / f).glob("seg*.nc")) for f in train + test}
    scores = evaluate([p for f in train + test for p in paths[f]], combos)
    scores.to_csv(OUT_DIR / "scores.csv", index=False)
    key = ["p0", "wind", "steps_per_12h"]
    on_train = scores[scores["fire"].isin(train) & (scores["steps_per_12h"] > 0)]
    best = dict(zip(key, on_train.groupby(key)[METRIC].mean().idxmax()))
    paper = on_train[(on_train["p0"] == AlexandridisParams().p0) & (on_train["wind"] == 1.0)]
    paper = {"p0": AlexandridisParams().p0, "wind": 1.0,    # paper p0 and wind; steps per 12 h still fitted on train
             "steps_per_12h": paper.groupby("steps_per_12h")[METRIC].mean().idxmax()}

    def pick(df, k):
        if k is None:
            return df[df["steps_per_12h"] == 0]
        return df[(df["p0"] == k["p0"]) & (df["wind"] == k["wind"]) & (df["steps_per_12h"] == k["steps_per_12h"])]

    def table(fires):
        sel = scores[scores["fire"].isin(fires)]
        return {name: {m: round(float(pick(sel, k)[m].mean()), 3) for m in ("iou", "growth_iou")}
                for name, k in (("no_growth", None), ("paper", paper), ("fitted", best))} | {
            "n_segments": int((sel["steps_per_12h"] == 0).sum())}

    report = {"metric": METRIC, "grid": GRID, "res_m": RES_M, "runs": RUNS, "train": train, "test": test,
              "best_params": {"p0": float(best["p0"]), "c1": AlexandridisParams().c1 * best["wind"],
                              "c2": AlexandridisParams().c2 * best["wind"], "wind_multiplier": float(best["wind"]),
                              "steps_per_12h": int(best["steps_per_12h"])},
              "paper_steps_per_12h": int(paper["steps_per_12h"]),
              "train_scores": table(train), "test_scores": table(test) if test else None,
              "per_segment": [{"segment": s, "fire": f,
                               **{f"{n}_{m}": round(float(pick(g, k)[m].iloc[0]), 3)
                                  for n, k in (("no_growth", None), ("fitted", best)) for m in ("iou", "growth_iou")}}
                              for (s, f), g in scores.groupby(["segment", "fire"])]}
    (OUT_DIR / "calibration.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))
    return report


def figure(path: Path, out: Path, p0: float, wind: float, steps_per_12h: int) -> None:
    """Start perimeter, VIIRS active front, next FEDS perimeter and the predicted P(burned) for one segment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = xr.load_dataset(path)
    start = ds["initial_burned"].values.astype(bool)
    target = ds["target"].values.astype(bool)
    front = active_front(ds, ds.attrs["fire"], pd.Timestamp(ds.attrs["seg_t0"]))
    front = start if front is None else front
    base = AlexandridisParams()
    params = dataclasses.replace(base, p0=p0, c1=base.c1 * wind, c2=base.c2 * wind)
    p = np.nan_to_num(spread_probabilities(ds.drop_vars("target"), params=params)["p_spread"].values.astype(np.float32))
    steps = max(1, int(round(steps_per_12h * float(ds.attrs["seg_hours"]) / 12)))
    prob = (first_burn_step(p, start | front, steps, runs=100, front=front) <= steps).mean(axis=0)
    prob[start] = np.nan
    flip = ds.y.values[0] < ds.y.values[-1]
    show = (lambda a: a[::-1]) if flip else (lambda a: a)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(show(prob), cmap="YlOrRd", vmin=0, vmax=1)
    ax.contour(show(target.astype(float)), [0.5], colors="blue", linewidths=1.5)
    ax.contour(show(start.astype(float)), [0.5], colors="black", linewidths=1)
    fy, fx = np.nonzero(show(front))
    ax.scatter(fx, fy, s=4, c="magenta", label="VIIRS active (12 h before)")
    u, v = float(np.nanmean(ds["wind_u_ms"])), float(np.nanmean(ds["wind_v_ms"]))
    h, w = start.shape
    ax.annotate("", xy=(w * 0.12 + 4 * u, h * 0.12 - 4 * v), xytext=(w * 0.12, h * 0.12),
                arrowprops={"arrowstyle": "->", "color": "green", "lw": 2})
    ax.set_title(f"{ds.attrs['fire']} {ds.attrs['seg_t0'][:16]} -> {ds.attrs['seg_t1'][5:16]} UTC\n"
                 f"black: start perimeter, blue: FEDS next, colour: P(burned), green: wind ({math.hypot(u, v):.1f} m/s)",
                 fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticks([]), ax.set_yticks([])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m firespread.calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("fires", nargs="+")
    t = sub.add_parser("fit")
    t.add_argument("--train", nargs="+", required=True)
    t.add_argument("--test", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        for fire in a.fires:
            print(fire, flush=True)
            fetch(fire)
    else:
        fit(a.train, a.test)


if __name__ == "__main__":
    main()
