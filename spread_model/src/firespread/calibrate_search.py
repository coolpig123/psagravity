"""Joint calibration of the Alexandridis and spotting parameters on the cached FEDS segments (see calibrate.py).

Random search over SPACE, then local refinement around the best sets. Every parameter set is scored on every
segment (growth IoU: soft IoU on the new burned area), so any train/test split is just an aggregation of the same
table. Accuracy is reported honestly with leave-one-fire-out: for each fire, the best set on the segments of the
other 3 fires (weighted per segment so 1-segment Colwash does not dominate 47-segment Crosswhite) is scored on
the held-out fire.

    python -m firespread.calibrate_search --n 200 --refine 100
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import numpy as np
import pandas as pd
import xarray as xr

from .alexandridis import (DIRECTIONS, AlexandridisParams, mask_reasons, p_moisture, p_veg, shift,
                           slope_angle_deg)
from .calibrate import OUT_DIR, active_front, first_burn_step, soft_iou

RUNS = 50
SPACE = {   # name: (low, high) for continuous, list for discrete
    "p0": (0.15, 0.98),
    "c1": (0.01, 0.16),                  # downwind wind speed term (s/m)
    "c2": (0.02, 0.30),                  # wind directional concentration term (s/m)
    "a": (0.02, 0.14),                   # slope term (1/deg)
    "b_emc": (0.0, 0.10),                # Simard EMC moisture sensitivity (1/%)
    "v_shrub": (-0.1, 0.35),             # P_veg for shrub lifeform (code 2)
    "v_grass": (-0.3, 0.25),             # P_veg for grass/herbaceous lifeform (code 3)
    "diagonal_factor": [round(1 / math.sqrt(2), 6), 1.0],
    "steps_per_12h": [1, 2, 3, 4],
    "spot_prob": (0.0, 0.65),
    "spot_min_wind_ms": (5.0, 7.5),
    "spot_min_m": (200.0, 800.0),
    "spot_wind_dist_m_per_ms": (80.0, 360.0),
    "spot_max_m": [3000.0, 5000.0, 7000.0],
    "spot_cone_deg": (8.0, 22.0),
    "spot_gust": [0, 1],                 # 1 = firebrands lofted by the gust speed (RTMA gust_ms), 0 = mean wind
}
PAPER = {"p0": 0.58, "c1": 0.045, "c2": 0.131, "a": 0.078, "b_emc": 0.05, "v_shrub": 0.0, "v_grass": -0.3,
         "diagonal_factor": round(1 / math.sqrt(2), 6), "steps_per_12h": 2, "spot_prob": 0.0,
         "spot_min_wind_ms": 6.0, "spot_min_m": 200.0, "spot_wind_dist_m_per_ms": 120.0, "spot_max_m": 3000.0,
         "spot_cone_deg": 15.0, "spot_gust": 0}
GEMINI = {**PAPER, "spot_prob": 0.12}    # paper values + initial spotting defaults
STAGE1 = {**PAPER, "p0": 0.8, "c1": 0.0225, "c2": 0.0655}
CALIBRATED = {"p0": 0.93, "c1": 0.095, "c2": 0.057, "a": 0.056, "b_emc": 0.006, "v_shrub": 0.21, "v_grass": 0.16,
              "diagonal_factor": 1.0, "steps_per_12h": 1, "spot_prob": 0.45, "spot_min_wind_ms": 5.8,
              "spot_min_m": 260.0, "spot_wind_dist_m_per_ms": 237.0, "spot_max_m": 7000.0,
              "spot_cone_deg": 9.5, "spot_gust": 1}


def load_segments() -> list[dict]:
    base = AlexandridisParams()
    segs = []
    for path in sorted(OUT_DIR.glob("*/seg*.nc")):
        ds = xr.load_dataset(path)
        assert float(ds.y[0]) > float(ds.y[-1]), "spotting assumes rows run north -> south"
        start = ds["initial_burned"].values.astype(bool)
        target = ds["target"].values.astype(bool)
        front = active_front(ds, ds.attrs["fire"], pd.Timestamp(ds.attrs["seg_t0"]))
        snap = ds.drop_vars("target").isel(time=0)
        res_m = float(ds.attrs["res_m"])

        veg_default, _ = p_veg(snap, base)
        lf = snap["fuel_lifeform"].values if "fuel_lifeform" in snap else np.zeros_like(start, np.int16)
        _, moist_parts, _ = p_moisture(snap, base)
        emc = np.nan_to_num(moist_parts["emc_pct"], nan=base.emc_ref_pct).astype(np.float32)
        reason, _ = mask_reasons(snap, start)
        mask = (reason == 0).astype(np.float32)

        u = np.nan_to_num(snap["wind_u_ms"].values.astype(np.float32))
        v = np.nan_to_num(snap["wind_v_ms"].values.astype(np.float32))
        speed = np.hypot(u, v).astype(np.float32)
        gust = (np.maximum(np.nan_to_num(snap["gust_ms"].values.astype(np.float32)), speed)
                if "gust_ms" in snap else speed)
        toward = np.degrees(np.arctan2(u, v)).astype(np.float32)
        dir_deg = (toward % 360.0).astype(np.float32)
        elev = snap["elevation_m"].values.astype(np.float32)

        h, w = start.shape
        cos_m1 = np.empty((8, h, w), np.float32)
        theta_s = np.empty((8, h, w), np.float32)
        for k, (_, dr, dc, bearing) in enumerate(DIRECTIONS):
            dist = res_m * (math.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0)
            cos_m1[k] = np.cos(np.radians(bearing - toward)) - 1.0
            theta_s[k] = np.nan_to_num(slope_angle_deg(elev, shift(elev, dr, dc, np.nan), dist))

        segs.append({"name": f"{ds.attrs['fire']}/{path.stem}", "fire": ds.attrs["fire"],
                     "start": start, "new": ~start, "front": start if front is None else front,
                     "target": target, "target_new": target[~start],
                     "hours": float(ds.attrs["seg_hours"]), "res_m": res_m,
                     "veg_default": veg_default, "lf": lf, "emc": emc, "mask": mask,
                     "speed": speed, "gust": gust, "dir_deg": dir_deg, "cos_m1": cos_m1, "theta_s": theta_s})
    return segs


def params_of(theta: dict) -> AlexandridisParams:
    b = AlexandridisParams()
    return dataclasses.replace(
        b, p0=theta["p0"], c1=theta["c1"], c2=theta["c2"], a=theta["a"], b_emc=theta["b_emc"],
        veg_by_lifeform={1: 0.4, 2: theta["v_shrub"], 3: theta["v_grass"]},
        diagonal_factor=theta["diagonal_factor"], spot_prob=max(theta["spot_prob"], 0.0),
        spot_use_gust=bool(theta["spot_gust"]), spot_min_wind_ms=theta["spot_min_wind_ms"],
        spot_min_m=theta["spot_min_m"], spot_wind_dist_m_per_ms=theta["spot_wind_dist_m_per_ms"],
        spot_max_m=theta["spot_max_m"], spot_cone_deg=theta["spot_cone_deg"])


def score(theta: dict, seg: dict, runs: int = RUNS) -> tuple[float, float]:
    """(growth IoU, IoU) of one parameter set on one segment."""
    if theta["v_shrub"] == 0.0 and theta["v_grass"] == -0.3:
        veg = seg["veg_default"]
    else:
        veg = seg["veg_default"].copy()
        veg[seg["lf"] == 2] = theta["v_shrub"]
        veg[seg["lf"] == 3] = theta["v_grass"]

    moist = np.exp(np.clip(-theta["b_emc"] * (seg["emc"] - 8.0), -15.0, 15.0))
    target_fac = (theta["p0"] * np.maximum(1.0 + veg, 0.05) * moist * seg["mask"]).astype(np.float32)

    h, w = seg["start"].shape
    p = np.empty((8, h, w), np.float32)
    spd = seg["speed"]
    c1, c2, a, dfac_val = theta["c1"], theta["c2"], theta["a"], theta["diagonal_factor"]
    for k, (_, dr, dc, _) in enumerate(DIRECTIONS):
        dfac = dfac_val if (dr != 0 and dc != 0) else 1.0
        tf_shift = shift(target_fac, dr, dc, 0.0) * dfac
        log_ws = np.clip(c1 * spd + c2 * spd * seg["cos_m1"][k] + a * seg["theta_s"][k], -15.0, 15.0)
        p[k] = np.clip(tf_shift * np.exp(log_ws), 0.0, 1.0)

    spot = None
    if theta["spot_prob"] > 0:
        sp_spd = seg["gust"] if theta["spot_gust"] else spd
        vmin = theta["spot_min_wind_ms"]
        above = np.maximum(sp_spd - vmin, 0.0)
        wind_factor = np.where(sp_spd > vmin, 1.0 - np.exp(-above / 4.0), 0.0)
        launch = np.clip(theta["spot_prob"] * np.maximum(1.0 + veg, 0.0) * wind_factor, 0.0, 1.0).astype(np.float32)
        min_m, max_m = theta["spot_min_m"], theta["spot_max_m"]
        dist_m = np.clip(min_m + theta["spot_wind_dist_m_per_ms"] * above, min_m, max_m).astype(np.float32)
        spot = {"launch": launch, "ignite": np.clip(target_fac, 0.0, 1.0), "dist_m": dist_m,
                "dir_deg": seg["dir_deg"], "min_m": min_m, "max_m": max_m,
                "cone_deg": theta["spot_cone_deg"], "res_m": seg["res_m"]}

    steps = max(1, int(round(theta["steps_per_12h"] * seg["hours"] / 12)))
    first = first_burn_step(p, seg["start"] | seg["front"], steps, runs=runs, front=seg["front"], spot=spot)
    prob = (first <= steps).mean(axis=0)
    new = seg["new"]
    return soft_iou(prob[new], seg["target_new"]), soft_iou(prob, seg["target"])


def sample(rng) -> dict:
    return {k: (rng.choice(v).item() if isinstance(v, list) else float(rng.uniform(*v))) for k, v in SPACE.items()}


def perturb(theta: dict, rng, scale: float = 0.15) -> dict:
    out = dict(theta)
    for k, v in SPACE.items():
        if rng.random() < 0.5:
            continue
        if isinstance(v, list):
            i = v.index(out[k]) if out[k] in v else 0
            out[k] = v[int(np.clip(i + rng.choice([-1, 1]), 0, len(v) - 1))]
        else:
            out[k] = float(np.clip(out[k] + rng.normal(0, scale * (v[1] - v[0])), *v))
    return out


def by_fire(table: pd.DataFrame, metric: str = "growth_iou") -> pd.DataFrame:
    """Mean score per (set, fire): rows = set id, columns = fire."""
    return table.groupby(["set", "fire"])[metric].mean().unstack("fire")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m firespread.calibrate_search")
    ap.add_argument("--n", type=int, default=180, help="random parameter sets")
    ap.add_argument("--refine", type=int, default=80, help="local refinement sets around the best")
    ap.add_argument("--minutes", type=float, default=15.0, help="stop the search after this long")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(1)
    t0 = time.time()
    segs = load_segments()
    fires = sorted({s["fire"] for s in segs})
    print(f"{len(segs)} segments ({time.time() - t0:.0f} s): "
          + ", ".join(f"{f} {sum(s['fire'] == f for s in segs)}" for f in fires), flush=True)

    sets, rows = [], []
    seg_means, fire_means = [], []

    def run(theta, label):
        i = len(sets)
        sets.append({"set": i, "label": label, **theta})
        gs = []
        by_f = {f: [] for f in fires}
        for s in segs:
            g, iou = score(theta, s)
            rows.append({"set": i, "fire": s["fire"], "segment": s["name"], "growth_iou": g, "iou": iou})
            gs.append(g)
            by_f[s["fire"]].append(g)
        seg_means.append(float(np.mean(gs)))
        fire_means.append(float(np.mean([np.mean(by_f[f]) for f in fires])))

    def rank_score(i):        # balanced objective: equal-fire mean + segment mean
        return 0.5 * fire_means[i] + 0.5 * seg_means[i]

    baselines = (("paper", PAPER), ("gemini_defaults", GEMINI), ("stage1_fit", STAGE1), ("calibrated", CALIBRATED))
    for label, theta in baselines:
        run(theta, label)
        print(f"  {label}: fire_mean={fire_means[-1]:.4f}, seg_mean={seg_means[-1]:.4f}  ({time.time() - t0:.0f} s)",
              flush=True)
    for n in range(a.n):
        if time.time() - t0 > a.minutes * 60 * 0.6:
            break
        run(sample(rng), "random")
        if (n + 1) % 20 == 0:
            best = max(range(len(sets)), key=rank_score)
            print(f"  random {n + 1}: best fire_mean={fire_means[best]:.4f}, seg_mean={seg_means[best]:.4f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    for n in range(a.refine):
        if time.time() - t0 > a.minutes * 60:
            break
        ranked = sorted(range(len(sets)), key=rank_score, reverse=True)[:5]
        parent = sets[ranked[n % len(ranked)]]
        run(perturb({k: parent[k] for k in SPACE}, rng, scale=0.12 if n < a.refine / 2 else 0.05), "refine")
        if (n + 1) % 20 == 0:
            best = max(range(len(sets)), key=rank_score)
            print(f"  refine {n + 1}: best fire_mean={fire_means[best]:.4f}, seg_mean={seg_means[best]:.4f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "search_scores.csv", index=False)
    pd.DataFrame(sets).to_csv(OUT_DIR / "search_sets.csv", index=False)
    per_fire = by_fire(table)
    per_fire_iou = by_fire(table, "iou")
    overall = per_fire.mean(axis=1)
    seg_overall = table.groupby("set")["growth_iou"].mean()
    best = int((0.5 * overall + 0.5 * seg_overall).idxmax())

    lofo = {}
    for f in fires:   # leave one fire out: choose on all training segments of the other 3 fires, score on f
        train_segs = table[table["fire"] != f].groupby("set")["growth_iou"].mean()
        pick = int(train_segs.idxmax())
        lofo[f] = {"chosen_set": pick, "growth_iou": round(float(per_fire.loc[pick, f]), 4),
                   "paper_growth_iou": round(float(per_fire.loc[0, f]), 4)}
    label = {s["set"]: s["label"] for s in sets}
    report = {
        "n_sets": len(sets), "n_segments": len(segs), "runs": RUNS, "space": SPACE,
        "best_set": best, "best_params": {k: sets[best][k] for k in SPACE},
        "all_fires_growth_iou": {"best": round(float(overall[best]), 4),
                                 **{label[i]: round(float(overall[i]), 4) for i in range(len(baselines))}},
        "all_segments_growth_iou": {"best": round(float(seg_overall[best]), 4),
                                    **{label[i]: round(float(seg_overall[i]), 4) for i in range(len(baselines))}},
        "per_fire_growth_iou": {name: {f: round(float(per_fire.loc[i, f]), 4) for f in fires}
                                for name, i in ((*( (label[j], j) for j in range(len(baselines)) ), ("best", best)))},
        "per_fire_iou": {name: {f: round(float(per_fire_iou.loc[i, f]), 4) for f in fires}
                         for name, i in (("paper", 0), ("calibrated", 3), ("best", best))},
        "leave_one_fire_out": lofo,
        "lofo_mean_growth_iou": round(float(np.mean([v["growth_iou"] for v in lofo.values()])), 4),
        "lofo_mean_paper_growth_iou": round(float(np.mean([v["paper_growth_iou"] for v in lofo.values()])), 4),
    }
    # how much spotting adds: the best set with spotting switched off
    no_spot = {**report["best_params"], "spot_prob": 0.0}
    g = [score(no_spot, s)[0] for s in segs]
    t = pd.DataFrame({"fire": [s["fire"] for s in segs], "g": g})
    report["best_without_spotting_growth_iou"] = round(float(t.groupby("fire")["g"].mean().mean()), 4)
    (OUT_DIR / "search_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
