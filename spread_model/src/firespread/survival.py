"""When does the fire reach each cell? A survival (time-to-event) model on top of the cellular automaton.

The CA (alexandridis.py + simulate.py) says *whether* a cell is likely to burn, in CA steps that have no unit
of time. This module learns *when*, in hours, from the real fires in real_fires/data/fires/, with an
XGBoost accelerated-failure-time (AFT) model:

    ln T = f(x) + sigma * Z,   Z ~ Normal(0, 1)        T = hours from the start until the fire reaches the cell

f is a gradient-boosted tree ensemble over per-cell features x (cell_features). Then for every cell:

    median arrival   = exp(f(x))
    P(T <= h hours)  = Phi((ln h - f(x)) / sigma)

Labels (arrival_bounds): one interval [lower, upper] per cell, in UTC, from the ground truth decided for the
study fires (real_fires/data/fires/<fire>/README.md):
  - first VIIRS 375 m detection covering the cell -> upper bound; the VIIRS pass before it (no detection
    there yet) -> lower bound. VIIRS passes are a few hours apart, so most intervals are 1-12 h wide;
  - FEDS 12 h perimeters -> another upper bound (the cell was inside the perimeter by then);
  - inside the WFIGS final perimeter but never detected -> burned some time between ignition and the end of
    growth (wide interval);
  - outside the final perimeter -> never burned: right-censored at the end of growth (upper = infinity).
XGBoost AFT is trained on these intervals directly (interval censoring), so nothing is guessed in between.

Training samples (fire_samples): several start times per fire (VIIRS passes while the fire grows); at each,
the area already detected is the burning fire, the CA runs from it, and every other cell becomes one sample
with its features and its interval relative to the start. Inputs come from Earth Engine (terrain, land cover,
vegetation, hourly weather at the start), downloaded once per fire and cached in spread_model/data/survival/.

    python -m firespread.survival train         # downloads what is missing, trains, writes the model + metrics
    python -m firespread.survival evaluate      # metrics of the saved model on every fire

Validation: leave-one-fire-out over the training fires (the fire being scored is never seen in training),
and Crosswhite held out completely (merged lightning fires: validation only, see its README).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .simulate import simulate_spread

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parents[2]                       # the psagravity checkout
FIRES_DIR = REPO_DIR / "real_fires" / "data" / "fires"
CACHE_DIR = REPO_DIR / "spread_model" / "data" / "survival"
MODEL_PATH = PACKAGE_DIR / "models" / "arrival_aft.json"

TRAIN_FIRES = ("bug_fire", "colwash", "biscar")
VALID_FIRES = ("crosswhite",)
HORIZONS_H = (6, 12, 24, 48)

# features: same code for training and for the website, so they always match
REACH_M = 12_000.0       # the feature CA runs until it could have travelled this far (one cell per step)
FEATURE_RUNS = 100       # Monte Carlo runs of the feature CA
FEATURES = ("dist_front_m", "ca_arrival_m", "ca_burn_prob", "wind_speed_ms", "wind_along_ms", "wind_cross_ms",
            "slope_path_deg", "p_veg", "p_m", "emc_pct", "can_burn")
# Not used: features that are the same for every cell of a start (time of day, fire area). With ~13 start times
# in total the trees used them to recognise the start instead of learning spread (worse on held-out fires).
# Arrival can only get later with distance and earlier with the CA's burn chance.
MONOTONE = {"dist_front_m": 1, "ca_burn_prob": -1}

# labels / samples
RES_M = 100.0             # same cell size as the website
BUFFER_KM = 5.0           # grid = WFIGS final perimeter + this
VIIRS_FOOTPRINT_M = 300.0  # a VIIRS detection covers the cells within this distance of its centre
PASS_GAP_MIN = 20          # detections closer in time than this belong to the same satellite pass
MAX_STARTS = 6
MIN_START_GAP_H = 6.0
START_FRACTION = (0.03, 0.85)  # start times: between 3% and 85% of the final detected area already burned
CENSORED_KEEP = 0.25       # share of never-burned cells kept (with weight 1 / CENSORED_KEEP)
MIN_HOURS = 0.05           # AFT needs lower bounds > 0


# ---- features ---------------------------------------------------------------------------------

def cell_features(result: xr.Dataset, time_utc=None) -> dict[str, np.ndarray]:
    """Per-cell features (y, x) from a spread_probabilities() result (the weather in it is the start time's).

    Geometry is measured from the nearest burning cell: distance, the wind component along / across that path,
    and the mean slope along it. The CA runs forward until it could reach REACH_M (ca_burn_prob, and the mean
    distance it took to get there, ca_arrival_m = arrival step x cell size; NaN where it never got).
    """
    from scipy.ndimage import distance_transform_edt

    burning = result["burning"].values.astype(bool)
    res = float(result.attrs.get("res_m", abs(float(result.x[1] - result.x[0]))))
    h, w = burning.shape
    dist, (iy, ix) = distance_transform_edt(~burning, sampling=res, return_indices=True)
    rows, cols = np.indices((h, w))
    east, north = (cols - ix) * res, (iy - rows) * res                 # row 0 = north
    safe = np.where(dist > 0, dist, 1.0)
    u = np.nan_to_num(result["wind_u_ms"].values) if "wind_u_ms" in result else np.zeros((h, w))
    v = np.nan_to_num(result["wind_v_ms"].values) if "wind_v_ms" in result else np.zeros((h, w))
    elev = result["elevation_m"].values if "elevation_m" in result else np.zeros((h, w))
    rise = elev - elev[iy, ix]

    steps = max(1, min(int(math.ceil(REACH_M / res)), 2 * (h + w)))
    sim = simulate_spread(result, steps=steps, runs=FEATURE_RUNS, seed=0)
    f = {
        "dist_front_m": dist,
        "ca_arrival_m": sim["arrival_step"].values * res,
        "ca_burn_prob": sim["burn_prob"].values,
        "wind_speed_ms": np.hypot(u, v),
        "wind_along_ms": np.where(dist > 0, (u * east + v * north) / safe, 0.0),   # > 0: wind pushes fire here
        "wind_cross_ms": np.where(dist > 0, np.abs(u * north - v * east) / safe, 0.0),
        "slope_path_deg": np.where(dist > 0, np.degrees(np.arctan(rise / safe)), 0.0),  # > 0: uphill
        "p_veg": result["p_veg"].values,
        "p_m": result["p_m"].values,
        "emc_pct": result["emc_pct"].values,
        "can_burn": (result["mask_reason"].values == 0).astype(np.float32),
    }
    return {k: np.asarray(f[k], np.float32) for k in FEATURES}


def feature_matrix(features: dict[str, np.ndarray], select: np.ndarray | None = None) -> np.ndarray:
    cols = [features[k] if select is None else features[k][select] for k in FEATURES]
    return np.stack([c.ravel() for c in cols], axis=1)


# ---- the model --------------------------------------------------------------------------------

@dataclass
class ArrivalModel:
    """XGBoost AFT booster + the noise scale sigma it was trained with (normal distribution of ln T)."""
    booster: object
    sigma: float
    features: tuple[str, ...] = FEATURES
    meta: dict = field(default_factory=dict)

    def median_hours(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(X, feature_names=list(self.features)))

    def p_within(self, median_h: np.ndarray, hours: float) -> np.ndarray:
        from scipy.stats import norm
        return norm.cdf((math.log(hours) - np.log(np.maximum(median_h, 1e-6))) / self.sigma)

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"sigma": self.sigma, "features": list(self.features), "distribution": "normal", **self.meta},
            indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ArrivalModel":
        import xgboost as xgb
        if not path.exists():
            raise FileNotFoundError(f"no arrival-time model at {path}; train it: python -m firespread.survival train")
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if tuple(meta["features"]) != FEATURES:
            raise ValueError(f"model {path} was trained on other features ({meta['features']}); retrain it")
        booster = xgb.Booster()
        booster.load_model(str(path))
        return cls(booster, float(meta.pop("sigma")), tuple(meta.pop("features")), meta)


_LOADED: dict[Path, ArrivalModel] = {}


def load_model(path: Path = MODEL_PATH) -> ArrivalModel:
    """The saved model, loaded once per process."""
    if path not in _LOADED:
        _LOADED[path] = ArrivalModel.load(path)
    return _LOADED[path]


def predict_arrival(result: xr.Dataset, time_utc=None, model: ArrivalModel | None = None) -> xr.Dataset:
    """Add arrival_h (median hours until the fire reaches the cell) and p_arrive_<h>h (chance it does within
    h hours, h in HORIZONS_H) to a spread_probabilities() / simulate_spread() result. Burning cells -> NaN.
    `time_utc` (the start) is not a feature: the weather already in `result` is the start time's."""
    model = model or load_model()
    burning = result["burning"].values.astype(bool)
    feats = cell_features(result, time_utc)
    median = model.median_hours(feature_matrix(feats)).reshape(burning.shape).astype(np.float32)
    median[burning] = np.nan
    out = result.copy()
    out["arrival_h"] = (("y", "x"), median, {"units": "h", "description": "median time until the fire arrives "
                                                                          "(XGBoost AFT, survival.py)"})
    res = float(result.attrs.get("res_m", abs(float(result.x[1] - result.x[0]))))
    for hours in HORIZONS_H:
        p = model.p_within(median, hours).astype(np.float32)
        p[burning] = np.nan
        out[f"p_arrive_{hours}h"] = (("y", "x"), p, {"description": f"chance the fire arrives within {hours} h"})
        out.attrs[f"arrival_area_{hours}h_km2"] = float(np.nansum(p)) * res * res / 1e6
    out.attrs["arrival_model"] = (f"XGBoost AFT (normal, sigma {model.sigma:g}), trained "
                                  f"{model.meta.get('trained_utc', '?')} on {', '.join(model.meta.get('fires', []))}")
    return out


# ---- labels from the real fires ---------------------------------------------------------------

def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _sec(t) -> float:
    return _utc(t).timestamp()


@dataclass
class FireTruth:
    slug: str
    final_perimeter: dict          # WFIGS GeoJSON
    feds: dict                     # FEDS 12 h perimeters GeoJSON
    points: pd.DataFrame           # fire_points, ambiguous == False
    ignition_utc: pd.Timestamp


def load_truth(slug: str) -> FireTruth:
    d = FIRES_DIR / slug / "processed"
    if not d.exists():
        raise FileNotFoundError(f"{d} not found: build the ground truth first (real_fires/scripts/build_ground_truth.py)")
    parquet, csv = d / "fire_points.parquet", d / "fire_points.csv"
    pts = pd.read_csv(csv) if csv.exists() else pd.read_parquet(parquet)
    pts["time_utc"] = pd.to_datetime(pts["time_utc"], utc=True)
    pts = pts[~pts["ambiguous"].astype(bool)].reset_index(drop=True)
    final = json.loads((d / "wfigs_final_perimeter.geojson").read_text(encoding="utf-8"))
    props = final["features"][0]["properties"]
    discovery = props.get("attr_FireDiscoveryDateTime")
    ignition = pd.to_datetime(discovery, utc=True, unit="ms" if isinstance(discovery, (int, float)) else None) \
        if discovery else pts["time_utc"].min()
    feds = json.loads((d / "feds_perimeters_12h.geojson").read_text(encoding="utf-8"))
    return FireTruth(slug, final, feds, pts, min(_utc(ignition), pts["time_utc"].min()))


def arrival_bounds(grid, truth: FireTruth) -> dict:
    """Per cell: lower / upper bound of the arrival time (UTC seconds; upper = inf = never burned), and the
    satellite-pass times and end of growth used. See the module docstring for the rules."""
    from pyproj import Transformer
    shape = grid.shape
    final = grid.polygon_mask(truth.final_perimeter).astype(bool)
    viirs = truth.points[truth.points["source"].str.startswith("VIIRS")]
    t_ign = _sec(truth.ignition_utc)

    first = np.full(shape, np.inf)
    xs, ys = Transformer.from_crs(4326, grid.epsg, always_xy=True).transform(viirs["lon"].values, viirs["lat"].values)
    r0 = np.floor((grid.y0 - np.asarray(ys)) / grid.res).astype(int)
    c0 = np.floor((np.asarray(xs) - grid.x0) / grid.res).astype(int)
    t = viirs["time_utc"].map(lambda v: v.timestamp()).to_numpy(float)
    k = int(math.ceil(VIIRS_FOOTPRINT_M / grid.res))
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if math.hypot(dr, dc) * grid.res > VIIRS_FOOTPRINT_M:
                continue
            r, c = r0 + dr, c0 + dc
            ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
            np.minimum.at(first, (r[ok], c[ok]), t[ok])
    first[~final] = np.inf                  # detections outside the official perimeter: other fires / spill-over

    # satellite passes: groups of VIIRS detection times
    times = np.sort(np.unique(t))
    passes = times[np.concatenate([[True], np.diff(times) > PASS_GAP_MIN * 60])] if times.size else times
    detected = np.isfinite(first)
    t_end = float(np.quantile(first[detected], 0.99)) if detected.any() else t_ign + 48 * 3600

    upper = first.copy()
    for feat in truth.feds["features"]:
        when = _sec(feat["properties"]["obs_time_utc"])
        inside = grid.polygon_mask({"type": "FeatureCollection", "features": [feat]}).astype(bool) & final
        upper[inside] = np.minimum(upper[inside], when)
    upper[final & ~np.isfinite(upper)] = max(t_end, t_ign + 3600)   # burned, time unknown: by the end of growth

    lower = np.full(shape, t_ign)
    if passes.size:
        idx = np.searchsorted(passes, first - PASS_GAP_MIN * 60) - 1          # last pass before the detection
        prev = np.where(idx >= 0, passes[np.clip(idx, 0, passes.size - 1)], t_ign)
        lower = np.where(detected, np.maximum(prev, t_ign), lower)
    lower[~final] = t_end
    lower = np.minimum(lower, upper)
    return {"lower": lower, "upper": upper, "final": final, "passes": passes, "t_end": t_end, "t_ign": t_ign}


def start_times(bounds: dict) -> list[pd.Timestamp]:
    """VIIRS passes while the fire grows (START_FRACTION of the final burned area), at least MIN_START_GAP_H apart."""
    burned = bounds["upper"][bounds["final"]]
    burned = burned[burned < bounds["t_end"]]
    if burned.size == 0:
        return []
    chosen: list[float] = []
    for p in bounds["passes"]:
        frac = float((burned <= p).mean())
        if START_FRACTION[0] <= frac <= START_FRACTION[1] and (not chosen or p - chosen[-1] >= MIN_START_GAP_H * 3600):
            chosen.append(float(p))
    if len(chosen) > MAX_STARTS:                                        # spread them over the growth
        chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, MAX_STARTS).round().astype(int)]
    return [pd.Timestamp(c, unit="s", tz="UTC") for c in chosen]


def burning_at(bounds: dict, t0: pd.Timestamp) -> np.ndarray:
    """Cells certainly burned by t0, as one solid area (gaps between VIIRS footprints closed, holes filled)."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    burned = (bounds["upper"] <= _sec(t0)) & bounds["final"]
    disk = np.hypot(*np.mgrid[-2:3, -2:3]) <= 2
    return binary_fill_holes(binary_closing(burned, structure=disk)) & bounds["final"]


# ---- inputs from Earth Engine (cached) --------------------------------------------------------

STATIC_LAYERS = ("terrain", "landcover", "vegetation_cover")


def fire_grid(truth: FireTruth, res_m: float = RES_M):
    from geelayers.grid import Grid
    return Grid.from_polygon(truth.final_perimeter, buffer_km=BUFFER_KM, res_m=res_m)


def fire_inputs(truth: FireTruth, grid, t0: pd.Timestamp, cache: Path = CACHE_DIR) -> xr.Dataset:
    """Static layers (downloaded once per fire) + the weather of the hour at or before t0 (once per start)."""
    from geelayers.api import get_layers, get_weather, regrid
    d = cache / truth.slug
    d.mkdir(parents=True, exist_ok=True)
    static_path = d / f"static_{grid.res:g}m.nc"
    if static_path.exists():
        static = xr.load_dataset(static_path)
    else:
        print(f"  {truth.slug}: downloading terrain, land cover, vegetation ({grid.shape[0]} x {grid.shape[1]} cells)",
              flush=True)
        static = get_layers(grid, time=truth.ignition_utc, layers=STATIC_LAYERS)
        static.to_netcdf(static_path, engine="h5netcdf")
    wx_path = d / f"weather_{t0:%Y%m%dT%H%M}_{grid.res:g}m.nc"
    if wx_path.exists():
        wx = xr.load_dataset(wx_path)
    else:
        print(f"  {truth.slug}: downloading weather at {t0:%Y-%m-%d %H:%M} UTC", flush=True)
        coarse = get_weather(grid.with_resolution(1000.0), t0.floor("h") - pd.Timedelta(hours=3),
                             t0 + pd.Timedelta(seconds=1), source="rtma")
        wx = regrid(coarse.isel(time=[-1]), grid).isel(time=0, drop=True)
        wx.attrs["weather_time_utc"] = str(pd.Timestamp(coarse.time.values[-1]))
        wx.to_netcdf(wx_path, engine="h5netcdf")
    ds = static.copy()
    for v in wx.data_vars:
        ds[v] = (("y", "x"), wx[v].values.astype(np.float32))
    ds.attrs.update({k: v for k, v in wx.attrs.items() if k.startswith("weather_")})
    return ds


# ---- training samples -------------------------------------------------------------------------

def fire_samples(slug: str, cache: Path = CACHE_DIR, seed: int = 0) -> pd.DataFrame:
    """One row per (start time, cell not yet burning): FEATURES + lower_h / upper_h (inf = censored) + weight."""
    from .alexandridis import spread_probabilities
    path = cache / slug / f"samples_{RES_M:g}m.parquet"
    if path.exists():
        return pd.read_parquet(path)
    truth = load_truth(slug)
    grid = fire_grid(truth)
    bounds = arrival_bounds(grid, truth)
    starts = start_times(bounds)
    print(f"{slug}: grid {grid.shape[0]} x {grid.shape[1]}, {len(starts)} start times: "
          + ", ".join(f"{t:%m-%d %H:%M}" for t in starts), flush=True)
    rng = np.random.default_rng(seed)
    frames = []
    for t0 in starts:
        burning = burning_at(bounds, t0)
        ds = fire_inputs(truth, grid, t0, cache)
        result = spread_probabilities(ds, burning=burning)
        feats = cell_features(result, t0)
        s0 = _sec(t0)
        lo = (bounds["lower"] - s0) / 3600
        hi = (bounds["upper"] - s0) / 3600
        # Inside the final perimeter, not burning at t0, but never timed by VIIRS (lower = ignition < t0): it burned
        # between t0 and its upper bound. Dropping these cells instead (half of Bug, 94% of Colwash) leaves mostly
        # never-burned cells near the fire, and the model then predicts arrival 3-10x too late.
        lo = np.where(bounds["final"] & (lo < 0), 0.0, lo)
        censored = ~np.isfinite(hi)
        keep = ~burning & (hi > 0)
        keep &= ~censored | (rng.random(keep.shape) < CENSORED_KEEP)
        df = pd.DataFrame(feature_matrix(feats, keep), columns=list(FEATURES))
        df["lower_h"] = np.maximum(lo[keep], MIN_HOURS)
        df["upper_h"] = np.maximum(hi[keep], MIN_HOURS)
        df["weight"] = np.where(censored[keep], 1 / CENSORED_KEEP, 1.0)
        df["fire"], df["start_utc"] = slug, t0.isoformat()
        frames.append(df)
        print(f"  {t0:%m-%d %H:%M}: {burning.sum() * grid.res ** 2 / 1e6:.1f} km2 burning, {keep.sum()} samples "
              f"({(~censored[keep]).sum()} burned later)", flush=True)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


# ---- training and evaluation ------------------------------------------------------------------

PARAMS = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
          "tree_method": "hist", "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 500,
          "subsample": 0.8, "colsample_bytree": 0.8, "seed": 0,
          "monotone_constraints": "(" + ",".join(str(MONOTONE.get(f, 0)) for f in FEATURES) + ")"}
ROUNDS = 400
SIGMAS = (0.6, 1.0, 1.5)


def _dmatrix(df: pd.DataFrame):
    import xgboost as xgb
    m = xgb.DMatrix(df[list(FEATURES)].to_numpy(np.float32), feature_names=list(FEATURES))
    m.set_float_info("label_lower_bound", df["lower_h"].to_numpy(np.float32))
    m.set_float_info("label_upper_bound", df["upper_h"].to_numpy(np.float32))
    m.set_weight(df["weight"].to_numpy(np.float32))
    return m


def fit(df: pd.DataFrame, sigma: float, rounds: int = ROUNDS) -> ArrivalModel:
    import xgboost as xgb
    booster = xgb.train({**PARAMS, "aft_loss_distribution_scale": sigma}, _dmatrix(df), num_boost_round=rounds)
    return ArrivalModel(booster, sigma)


def _auc(score: np.ndarray, label: np.ndarray) -> float | None:
    from scipy.stats import rankdata
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata(score)
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def evaluate(model: ArrivalModel, df: pd.DataFrame) -> dict:
    """Scores on samples the model has not seen. For each horizon h, a cell counts as positive if it certainly
    burned within h hours (upper <= h), negative if certainly not (lower >= h), and is skipped otherwise.
      auc_<h>h       AFT P(T <= h) ranks positives above negatives (0.5 = chance, 1 = perfect)
      brier_<h>h     mean squared error of that probability (lower is better)
      auc_ca_<h>h    the same AUC for the CA's own burn probability (the baseline the AFT has to beat)
      auc_dist_<h>h  ... and for plain distance to the fire (closer = sooner)
      interval_hit   share of burned cells (interval at most 13 h wide) whose median falls in their interval
      median_abs_log_err  |ln median - ln mid-interval| for those cells (0.69 = a factor of 2)
    """
    median = model.median_hours(df[list(FEATURES)].to_numpy(np.float32))
    lo, hi = df["lower_h"].to_numpy(), df["upper_h"].to_numpy()
    w = df["weight"].to_numpy()
    out: dict = {"n_samples": int(len(df)), "n_burned": int(np.isfinite(hi).sum())}
    for h in HORIZONS_H:
        pos, neg = hi <= h, lo >= h
        known = pos | neg
        if not known.any():
            continue
        p = model.p_within(median, h)
        label = pos[known]
        out[f"auc_{h}h"] = _auc(p[known], label)
        out[f"brier_{h}h"] = float(np.average((p[known] - label) ** 2, weights=w[known]))
        out[f"auc_ca_{h}h"] = _auc(df["ca_burn_prob"].to_numpy()[known], label)
        out[f"auc_dist_{h}h"] = _auc(-df["dist_front_m"].to_numpy()[known], label)
        out[f"n_known_{h}h"] = int(known.sum())
    tight = np.isfinite(hi) & (hi - lo <= 13)
    if tight.any():
        m = median[tight]
        out["interval_hit"] = float(((m >= lo[tight]) & (m <= hi[tight])).mean())
        out["median_abs_log_err"] = float(np.median(np.abs(np.log(m) - np.log((lo[tight] + hi[tight]) / 2))))
        out["n_tight"] = int(tight.sum())
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def _nloglik(model: ArrivalModel, df: pd.DataFrame) -> float:
    return float(model.booster.eval(_dmatrix(df)).split(":")[-1])


def train(fires=TRAIN_FIRES, valid=VALID_FIRES, save: bool = True) -> dict:
    """Leave-one-fire-out to pick sigma and score the method, then fit on all training fires and save."""
    data = {s: fire_samples(s) for s in (*fires, *valid)}
    data = {s: d for s, d in data.items() if len(d)}
    train_fires = [s for s in fires if s in data]
    report: dict = {"lofo": {}, "sigma_nloglik": {}}
    for sigma in SIGMAS:
        losses = []
        for held in train_fires:
            rest = pd.concat([data[s] for s in train_fires if s != held])
            losses.append(_nloglik(fit(rest, sigma), data[held]))
        report["sigma_nloglik"][str(sigma)] = round(float(np.mean(losses)), 4)
    sigma = float(min(SIGMAS, key=lambda s: report["sigma_nloglik"][str(s)]))
    report["sigma"] = sigma
    for held in train_fires:
        rest = pd.concat([data[s] for s in train_fires if s != held])
        report["lofo"][held] = evaluate(fit(rest, sigma), data[held])
    model = fit(pd.concat([data[s] for s in train_fires]), sigma)
    report["validation"] = {s: evaluate(model, data[s]) for s in valid if s in data}
    gain = model.booster.get_score(importance_type="gain")
    report["feature_gain"] = {k: round(v, 2) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}
    model.meta = {"trained_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "fires": train_fires,
                  "validation_fires": list(report["validation"]), "res_m": RES_M, "params": PARAMS, "rounds": ROUNDS,
                  "n_samples": {s: int(len(d)) for s, d in data.items()}, "report": report}
    if save:
        model.save()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_report(report: dict) -> None:
    print(f"\nsigma = {report['sigma']} (mean held-out -loglik by sigma: {report['sigma_nloglik']})")
    rows = {**{f"{k} (held out)": v for k, v in report["lofo"].items()},
            **{f"{k} (validation)": v for k, v in report.get("validation", {}).items()}}
    for name, m in rows.items():
        print(f"\n{name}: {m['n_samples']} samples, {m['n_burned']} burned later")
        for h in HORIZONS_H:
            if f"auc_{h}h" in m:
                print(f"  within {h:>2} h: AUC {m[f'auc_{h}h']}  (CA {m[f'auc_ca_{h}h']}, distance "
                      f"{m[f'auc_dist_{h}h']})  Brier {m[f'brier_{h}h']}  [{m[f'n_known_{h}h']} cells]")
        if "interval_hit" in m:
            print(f"  median inside the observed interval: {100 * m['interval_hit']:.0f}%, median |ln error| "
                  f"{m['median_abs_log_err']} ({m['n_tight']} cells)")
    print("\nfeature gain:", report["feature_gain"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m firespread.survival", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="download missing inputs, train, save the model and the report")
    sub.add_parser("evaluate", help="print the report saved with the model")
    args = ap.parse_args(argv)
    if args.cmd == "train":
        report = train()
        _print_report(report)
        print(f"\nmodel: {MODEL_PATH}\nreport: {CACHE_DIR / 'report.json'}")
    else:
        _print_report(load_model().meta["report"])


if __name__ == "__main__":
    main()
