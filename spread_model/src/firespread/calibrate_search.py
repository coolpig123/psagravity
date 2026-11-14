"""Joint calibration of the Alexandridis and spotting parameters on the cached FEDS segments (see calibrate.py).

Random search over SPACE, then local refinement around the best sets. Every parameter set is scored on every
segment (growth IoU: soft IoU on the new burned area), so any train/test split is just an aggregation of the same
table. Accuracy is reported honestly with leave-one-fire-out: for each fire, the best set on the other 3 fires is
scored on that fire.

    python -m firespread.calibrate_search --n 300 --refine 120
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import pandas as pd
import xarray as xr

from .alexandridis import AlexandridisParams, spread_probabilities
from .calibrate import OUT_DIR, active_front, first_burn_step, soft_iou

RUNS = 30
SPACE = {   # name: (low, high) for continuous, list for discrete
    "p0": (0.1, 1.0),
    "wind": (0.25, 3.0),                 # multiplier on c1 and c2
    "slope": (0.5, 2.0),                 # multiplier on a
    "b_emc": (0.0, 0.15),
    "steps_per_12h": [1, 2, 3, 4, 6, 8],
    "spot_prob": (0.0, 1.0),
    "spot_min_wind_ms": (0.0, 8.0),
    "spot_min_m": (200.0, 1500.0),
    "spot_wind_dist_m_per_ms": (0.0, 600.0),
    "spot_cone_deg": (5.0, 30.0),
    "spot_gust": [0, 1],                 # 1 = firebrands lofted by the gust speed (RTMA gust_ms), 0 = mean wind
}
STAGE1 = {"p0": 0.8, "wind": 0.5, "slope": 1.0, "b_emc": 0.05, "steps_per_12h": 2, "spot_prob": 0.0,
          "spot_min_wind_ms": 6.0, "spot_min_m": 200.0, "spot_wind_dist_m_per_ms": 120.0, "spot_cone_deg": 15.0,
          "spot_gust": 0}
PAPER = {**STAGE1, "p0": 0.58, "wind": 1.0}
GEMINI = {**PAPER, "spot_prob": 0.12}    # paper values + Gemini's spotting defaults


def load_segments() -> list[dict]:
    segs = []
    for path in sorted(OUT_DIR.glob("*/seg*.nc")):
        ds = xr.load_dataset(path)
        assert float(ds.y[0]) > float(ds.y[-1]), "spotting assumes rows run north -> south"
        start = ds["initial_burned"].values.astype(bool)
        front = active_front(ds, ds.attrs["fire"], pd.Timestamp(ds.attrs["seg_t0"]))
        def first_time(name):     # the weather hour spread_probabilities uses (the segment's afternoon time)
            da = ds[name]
            da = da.isel({d: 0 for d in da.dims if d not in ("y", "x")})
            return np.nan_to_num(da.values.astype(np.float32))
        u, v = first_time("wind_u_ms"), first_time("wind_v_ms")
        speed = np.hypot(u, v)
        gust = first_time("gust_ms") if "gust_ms" in ds else speed
        segs.append({"name": f"{ds.attrs['fire']}/{path.stem}", "fire": ds.attrs["fire"], "ds": ds.drop_vars("target"),
                     "start": start, "front": start if front is None else front, "target": ds["target"].values.astype(bool),
                     "hours": float(ds.attrs["seg_hours"]), "speed": speed, "gust": np.maximum(gust, speed),
                     "dir_deg": (np.degrees(np.arctan2(u, v)) % 360).astype(np.float32),
                     "res_m": float(ds.attrs["res_m"])})
    return segs


def params_of(theta: dict) -> AlexandridisParams:
    b = AlexandridisParams()
    return dataclasses.replace(b, p0=theta["p0"], c1=b.c1 * theta["wind"], c2=b.c2 * theta["wind"],
                               a=b.a * theta["slope"], b_emc=theta["b_emc"], spot_prob=max(theta["spot_prob"], 1e-6),
                               spot_min_wind_ms=theta["spot_min_wind_ms"], spot_min_m=theta["spot_min_m"],
                               spot_wind_dist_m_per_ms=theta["spot_wind_dist_m_per_ms"],
                               spot_cone_deg=theta["spot_cone_deg"])


def score(theta: dict, seg: dict, runs: int = RUNS) -> tuple[float, float]:
    """(growth IoU, IoU) of one parameter set on one segment."""
    params = params_of(theta)
    res = spread_probabilities(seg["ds"], params=params)
    p = np.nan_to_num(res["p_spread"].values.astype(np.float32))
    spot = None
    if theta["spot_prob"] > 0:
        speed = seg["gust"] if theta["spot_gust"] else seg["speed"]
        above = np.maximum(speed - params.spot_min_wind_ms, 0.0)
        wind_factor = np.where(speed > params.spot_min_wind_ms, 1 - np.exp(-above / params.spot_wind_scale_ms), 0.0)
        veg, den = res["p_veg"].values, res["p_den"].values           # same launch formula as alexandridis.p_spot
        launch = np.clip(theta["spot_prob"] * np.maximum(1 + veg, 0) * np.maximum(1 + den, 0) * wind_factor, 0, 1)
        spot = {"launch": np.nan_to_num(launch).astype(np.float32),
                "ignite": np.nan_to_num(res["p_spot_ignite"].values.astype(np.float32)),
                "dist_m": np.clip(params.spot_min_m + params.spot_wind_dist_m_per_ms * above, params.spot_min_m,
                                  params.spot_max_m).astype(np.float32),
                "dir_deg": seg["dir_deg"], "min_m": params.spot_min_m, "max_m": params.spot_max_m,
                "cone_deg": params.spot_cone_deg, "res_m": seg["res_m"]}
    steps = max(1, int(round(theta["steps_per_12h"] * seg["hours"] / 12)))
    first = first_burn_step(p, seg["start"] | seg["front"], steps, runs=runs, front=seg["front"], spot=spot)
    prob = (first <= steps).mean(axis=0)
    new = ~seg["start"]
    return soft_iou(prob[new], seg["target"][new]), soft_iou(prob, seg["target"])


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
    ap.add_argument("--n", type=int, default=300, help="random parameter sets")
    ap.add_argument("--refine", type=int, default=120, help="local refinement sets around the best")
    ap.add_argument("--minutes", type=float, default=40.0, help="stop the search after this long")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(1)
    segs = load_segments()
    fires = sorted({s["fire"] for s in segs})
    print(f"{len(segs)} segments: " + ", ".join(f"{f} {sum(s['fire'] == f for s in segs)}" for f in fires), flush=True)

    sets, rows = [], []
    t0 = time.time()

    def run(theta, label):
        i = len(sets)
        sets.append({"set": i, "label": label, **theta})
        for s in segs:
            g, iou = score(theta, s)
            rows.append({"set": i, "fire": s["fire"], "segment": s["name"], "growth_iou": g, "iou": iou})

    def fire_mean(i):         # equal weight per fire
        t = pd.DataFrame(rows)
        return float(t[t["set"] == i].groupby("fire")["growth_iou"].mean().mean())

    for label, theta in (("paper", PAPER), ("gemini_defaults", GEMINI), ("stage1_fit", STAGE1)):
        run(theta, label)
        print(f"  {label}: {fire_mean(len(sets) - 1):.4f}  ({time.time() - t0:.0f} s)", flush=True)
    for n in range(a.n):
        if time.time() - t0 > a.minutes * 60 * 0.6:
            break
        run(sample(rng), "random")
        if n % 20 == 0:
            best = max(range(len(sets)), key=fire_mean)
            print(f"  random {n}: best so far {fire_mean(best):.4f}  ({time.time() - t0:.0f} s)", flush=True)
    for n in range(a.refine):
        if time.time() - t0 > a.minutes * 60:
            break
        ranked = sorted(range(len(sets)), key=fire_mean, reverse=True)[:5]
        parent = sets[ranked[n % len(ranked)]]
        run(perturb({k: parent[k] for k in SPACE}, rng, scale=0.15 if n < a.refine / 2 else 0.06), "refine")
        if n % 20 == 0:
            print(f"  refine {n}: best so far {fire_mean(ranked[0]):.4f}  ({time.time() - t0:.0f} s)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "search_scores.csv", index=False)
    pd.DataFrame(sets).to_csv(OUT_DIR / "search_sets.csv", index=False)
    per_fire = by_fire(table)
    per_fire_iou = by_fire(table, "iou")
    overall = per_fire.mean(axis=1)
    best = int(overall.idxmax())

    lofo = {}
    for f in fires:                               # leave one fire out: choose on the others, score on f
        others = [g for g in fires if g != f]
        pick = int(per_fire[others].mean(axis=1).idxmax())
        lofo[f] = {"chosen_set": pick, "growth_iou": round(float(per_fire.loc[pick, f]), 4),
                   "paper_growth_iou": round(float(per_fire.loc[0, f]), 4)}
    label = {s["set"]: s["label"] for s in sets}
    report = {
        "n_sets": len(sets), "n_segments": len(segs), "runs": RUNS, "space": SPACE,
        "best_set": best, "best_params": {k: sets[best][k] for k in SPACE},
        "all_fires_growth_iou": {"best": round(float(overall[best]), 4),
                                 **{label[i]: round(float(overall[i]), 4) for i in (0, 1, 2)}},
        "per_fire_growth_iou": {name: {f: round(float(per_fire.loc[i, f]), 4) for f in fires}
                                for name, i in (("paper", 0), ("gemini_defaults", 1), ("stage1_fit", 2), ("best", best))},
        "per_fire_iou": {name: {f: round(float(per_fire_iou.loc[i, f]), 4) for f in fires}
                         for name, i in (("paper", 0), ("best", best))},
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
