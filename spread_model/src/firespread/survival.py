"""When does the fire reach each cell? A cure-fraction survival model on top of the cellular automaton.

The CA (alexandridis.py) gives a per-step spread probability; its steps have no unit of time, and many cells
around a fire never burn at all (containment, weather, fuel breaks). So the arrival time T of a cell is modelled
as a **mixture cure model**:

    P(T <= h) = pi(x) * Phi((ln h - f(x)) / sigma)

    pi(x)  = P(the fire ever reaches the cell)          XGBoost binary:logistic   ("cure" = never burns)
    f(x)   = ln of the median arrival, if it does         XGBoost AFT (survival:aft, normal), interval-censored
    sigma  = spread of ln T around f(x)                   chosen by held-out Brier score

A plain AFT (P(T < inf) = 1) has to put cells that burn after 3 h and never-burned cells at the same distance on
one distribution, and the never-burned ones drag every median late (3-10x in the first version). Here the AFT only
learns *when*, from cells that did burn; the classifier learns *whether*.

**Features** (cell_features): the fastest path from the fire to each cell over the 8-neighbour grid, where crossing
from cell i to neighbour j costs 1 / p_spread(i -> j) CA steps: the expected number of steps until a burning cell
that keeps trying with probability p succeeds (geometric waiting time). So a backing or flanking fire is slow, not
extinguished (no percolation die-off as in a Monte Carlo run with one-step burnout), unburnable cells (water,
rock, built-up) are walls the path goes around, and the source is whichever burning cell is fastest, not the
nearest. Along that path: its cost (path_steps), length, total climb and descent, and the length-weighted mean wind
along it and P_veg. Weather enters only through the path cost and the along-path wind: per-cell weather is nearly
the same everywhere at one start hour, and with 13 training starts the trees used it to recognise the start.
Every feature has a sign constraint matching the physics (longer / costlier path -> later and less likely).
The path probabilities leave out the NISAR and Sentinel-2 terms (P_den, P_dry, P_vwc, their burn-scar masks),
which training has no data for, so the website (which has them) sees the same features as training; cells those
masks rule out are set to "never" afterwards.

**Labels** (arrival_bounds), one interval [lower, upper] per cell, UTC, from real_fires/data/fires/<fire>/:
  - first VIIRS 375 m detection covering the cell (or an earlier FEDS 12 h perimeter containing it) -> upper;
    the VIIRS pass before that upper bound -> lower (the cell was not detected yet then);
  - inside the WFIGS final perimeter but never timed -> burned between the start and the end of growth;
  - outside the final perimeter -> never burned (the "cured" class; upper = inf).
FEDS perimeters only give upper bounds: their area is a lower bound of the real one (38% for Bug), so a cell
outside a FEDS perimeter may well have burned already.

**Samples** (fire_samples): 2-6 start times per fire at VIIRS passes while 3-85% of the final area has burned;
at each, the detected area is the burning fire and every other cell is one sample. Inputs from Earth Engine
(terrain, land cover, LANDFIRE, RTMA weather at the start hour), cached in spread_model/data/survival/.

    python -m firespread.survival train         # downloads what is missing, trains, writes the model + report
    python -m firespread.survival evaluate      # prints the report saved with the model

**Validation**: leave-one-fire-out over the training fires, Crosswhite held out completely (merged lightning
fires: validation only). Scores (evaluate) are computed per start time and averaged, so cells from different
starts (with different remaining growth time) are never compared with each other.
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

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parents[2]                       # the psagravity checkout
FIRES_DIR = REPO_DIR / "real_fires" / "data" / "fires"
CACHE_DIR = REPO_DIR / "spread_model" / "data" / "survival"
MODEL_PATH = PACKAGE_DIR / "models" / "arrival_aft.json"        # when (AFT); + arrival_aft.cure.json, .meta.json

TRAIN_FIRES = ("bug_fire", "colwash", "biscar")
VALID_FIRES = ("crosswhite",)
HORIZONS_H = (6, 12, 24, 48)

# features: same code for training and for the website, so they always match
FEATURES = ("path_steps", "path_len_m", "path_up_m", "path_down_m", "path_wind_along_ms", "path_p_veg")
# sign of the effect on the arrival time (+1 = later); the "ever burns" classifier gets the opposite sign
MONOTONE_TIME = {"path_steps": 1, "path_len_m": 1, "path_up_m": -1, "path_down_m": 1,
                 "path_wind_along_ms": -1, "path_p_veg": -1}
P_MIN = 1e-4              # spread probabilities below this are no path (cost > 10,000 steps)

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
REACH_M = 12_000.0         # baseline only: the Monte Carlo CA runs until it could travel this far


# ---- features ---------------------------------------------------------------------------------

def path_spread_probabilities(result: xr.Dataset) -> np.ndarray:
    """(direction, y, x) per-step spread probability as in alexandridis.py, without the NISAR / Sentinel-2 terms:
    P0 (1 + P_veg) P_emc P_w P_s D, with only the land-cover / fuel / burning masks."""
    from .alexandridis import DIRECTIONS, AlexandridisParams, shift
    params = AlexandridisParams()
    reason = result["mask_reason"].values
    ok = np.isin(reason, (0, 4, 5)).astype(np.float32)        # 4 = NISAR, 5 = Sentinel-2 burn scar: not in training
    p_emc = np.nan_to_num(result["p_emc"].values, nan=1.0) if "p_emc" in result else 1.0
    target = params.p0 * (1 + result["p_veg"].values) * p_emc * ok
    p = np.empty(result["p_wind"].shape, np.float32)
    for k, (_, dr, dc, _) in enumerate(DIRECTIONS):
        diag = params.diagonal_factor if dr and dc else 1.0
        p[k] = shift(target, dr, dc, 0.0) * result["p_wind"].values[k] * result["p_slope"].values[k] * diag
    return np.clip(np.nan_to_num(p), 0.0, 1.0)


def _sum_along_paths(pred: np.ndarray, contrib: np.ndarray) -> np.ndarray:
    """For a shortest-path forest (pred = predecessor, < 0 for roots / unreached), the sum of `contrib` (the value
    of the edge into each node) along the path from the root to every node. Pointer jumping: O(n log depth)."""
    acc = contrib.astype(np.float64).copy()
    anc = pred.astype(np.int64).copy()
    live = anc >= 0
    while live.any():
        i = np.nonzero(live)[0]
        acc[i] += acc[anc[i]]                 # right side is read before the update: all nodes jump together
        anc[i] = anc[anc[i]]
        live = anc >= 0
    return acc


def cell_features(result: xr.Dataset, time_utc=None) -> dict[str, np.ndarray]:
    """Per-cell features (y, x) from a spread_probabilities() result (the weather in it is the start time's):
    the fastest path from the burning cells (module docstring). Unreachable cells and burning cells -> NaN."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from .alexandridis import DIRECTIONS

    burning = result["burning"].values.astype(bool)
    h, w = burning.shape
    n = h * w
    res = float(result.attrs.get("res_m", abs(float(result.x[1] - result.x[0]))))
    p = path_spread_probabilities(result)
    idx = np.arange(n).reshape(h, w)
    src, dst, cost = [], [], []
    for k, (_, dr, dc, _) in enumerate(DIRECTIONS):
        r0, r1, c0, c1 = max(0, -dr), h - max(0, dr), max(0, -dc), w - max(0, dc)
        pk = p[k, r0:r1, c0:c1]
        m = pk > P_MIN
        src.append(idx[r0:r1, c0:c1][m])
        dst.append(idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc][m])
        cost.append(1.0 / pk[m])               # expected CA steps until this neighbour catches fire
    graph = csr_matrix((np.concatenate(cost), (np.concatenate(src), np.concatenate(dst))), shape=(n, n))

    nan = np.full((h, w), np.nan, np.float32)
    front = idx[burning].ravel()
    if front.size == 0 or graph.nnz == 0:
        return {k: nan.copy() for k in FEATURES}
    steps, pred, _ = dijkstra(graph, directed=True, indices=front, min_only=True, return_predecessors=True)

    # the edge into every node: from pred[j] to j
    has = pred >= 0
    j = np.nonzero(has)[0]
    q = pred[j]
    drow, dcol = j // w - q // w, j % w - q % w
    length = np.zeros(n)
    length[j] = res * np.hypot(drow, dcol)
    elev = result["elevation_m"].values.astype(np.float64) if "elevation_m" in result else np.zeros((h, w))
    elev = np.nan_to_num(elev, nan=float(np.nanmean(elev)) if np.isfinite(elev).any() else 0.0).ravel()
    rise = np.zeros(n)
    rise[j] = elev[j] - elev[q]
    u = np.nan_to_num(result["wind_u_ms"].values).ravel() if "wind_u_ms" in result else np.zeros(n)
    v = np.nan_to_num(result["wind_v_ms"].values).ravel() if "wind_v_ms" in result else np.zeros(n)
    along = np.zeros(n)                        # wind component along the edge (east = +col, north = -row) x length
    along[j] = (u[q] * dcol - v[q] * drow) / np.hypot(drow, dcol) * length[j]
    veg = np.zeros(n)
    veg[j] = result["p_veg"].values.ravel()[j] * length[j]

    total_len = _sum_along_paths(pred, length)
    safe = np.where(total_len > 0, total_len, np.nan)
    reached = np.isfinite(steps) & ~burning.ravel()
    out = {
        "path_steps": steps,
        "path_len_m": total_len,
        "path_up_m": _sum_along_paths(pred, np.maximum(rise, 0)),
        "path_down_m": _sum_along_paths(pred, np.maximum(-rise, 0)),
        "path_wind_along_ms": _sum_along_paths(pred, along) / safe,
        "path_p_veg": _sum_along_paths(pred, veg) / safe,
    }
    return {k: np.where(reached, out[k], np.nan).reshape(h, w).astype(np.float32) for k in FEATURES}


def feature_matrix(features: dict[str, np.ndarray], select: np.ndarray | None = None) -> np.ndarray:
    cols = [features[k] if select is None else features[k][select] for k in FEATURES]
    return np.stack([c.ravel() for c in cols], axis=1)


# ---- the model --------------------------------------------------------------------------------

@dataclass
class ArrivalModel:
    """Cure-fraction mixture: `cure` = XGBoost classifier of P(ever burns), `time` = XGBoost AFT of the arrival
    time of cells that burn, `sigma` = its noise scale (normal distribution of ln T)."""
    cure: object
    time: object
    sigma: float
    features: tuple[str, ...] = FEATURES
    meta: dict = field(default_factory=dict)
    # isotonic maps raw -> calibrated probability, learned from held-out fires: {"ever" | "<h>": [xs, ys]}
    calibration: dict | None = None

    def probabilities(self, p_ever: np.ndarray, median_h: np.ndarray, raw: bool = False) -> dict:
        """{"ever": P(ever burns), h: P(T <= h) for h in HORIZONS_H}, calibrated unless raw (or no calibration),
        and consistent: P(T <= 6) <= P(T <= 12) <= ... <= P(ever)."""
        out = {"ever": np.asarray(p_ever, np.float64)}
        for h in HORIZONS_H:
            out[h] = self.p_within(out["ever"], median_h, h)
        if self.calibration and not raw:
            # isotonic maps have flat steps; + 1e-6 x raw keeps the raw order inside a step (no ties, same ranking)
            out = {k: np.clip(np.interp(v, *self.calibration[str(k)]) + 1e-6 * v, 0.0, 1.0) for k, v in out.items()}
            prev = out["ever"]
            for h in reversed(HORIZONS_H):          # keep the order the calibration of each horizon may break
                out[h] = np.minimum(out[h], prev)
                prev = out[h]
        return out

    def _dm(self, X):
        import xgboost as xgb
        return xgb.DMatrix(X, feature_names=list(self.features))

    def p_ever(self, X: np.ndarray) -> np.ndarray:
        return self.cure.predict(self._dm(X))

    def median_if_burns(self, X: np.ndarray) -> np.ndarray:
        return self.time.predict(self._dm(X))

    def p_within(self, p_ever: np.ndarray, median_h: np.ndarray, hours: float) -> np.ndarray:
        from scipy.stats import norm
        return p_ever * norm.cdf((math.log(hours) - np.log(np.maximum(median_h, 1e-6))) / self.sigma)

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.time.save_model(str(path))
        self.cure.save_model(str(path.with_suffix(".cure.json")))
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"sigma": self.sigma, "features": list(self.features), "distribution": "normal",
             "calibration": self.calibration, **self.meta}, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ArrivalModel":
        import xgboost as xgb
        cure_path = path.with_suffix(".cure.json")
        if not path.exists() or not cure_path.exists():
            raise FileNotFoundError(f"no arrival-time model at {path}; train it: python -m firespread.survival train")
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if tuple(meta["features"]) != FEATURES:
            raise ValueError(f"model {path} was trained on other features ({meta['features']}); retrain it")
        time, cure = xgb.Booster(), xgb.Booster()
        time.load_model(str(path))
        cure.load_model(str(cure_path))
        return cls(cure, time, float(meta.pop("sigma")), tuple(meta.pop("features")), meta,
                   meta.pop("calibration", None))


_LOADED: dict[Path, ArrivalModel] = {}


def load_model(path: Path = MODEL_PATH) -> ArrivalModel:
    """The saved model, loaded once per process."""
    if path not in _LOADED:
        _LOADED[path] = ArrivalModel.load(path)
    return _LOADED[path]


def predict_arrival(result: xr.Dataset, time_utc=None, model: ArrivalModel | None = None) -> xr.Dataset:
    """Add to a spread_probabilities() / simulate_spread() result:
      p_ever         chance the fire ever reaches the cell
      arrival_h      median hours until it arrives, if it does
      p_arrive_<h>h  chance it arrives within h hours (h in HORIZONS_H) = p_ever x P(T <= h | it burns)
    Burning cells -> NaN. Cells no path reaches, or that cannot burn (any mask, NISAR / Sentinel-2 included) ->
    p_ever 0, arrival inf. `time_utc` is not a feature: the weather in `result` is the start time's."""
    model = model or load_model()
    burning = result["burning"].values.astype(bool)
    feats = cell_features(result, time_utc)
    X = feature_matrix(feats)
    shape = burning.shape
    median = model.median_if_burns(X).reshape(shape).astype(np.float32)
    probs = {k: v.reshape(shape).astype(np.float32) for k, v in model.probabilities(model.p_ever(X), median.ravel()).items()}
    never = ~np.isfinite(feats["path_steps"]) | (result["mask_reason"].values != 0)
    median[never] = np.inf
    median[burning] = np.nan
    for v in probs.values():
        v[never], v[burning] = 0.0, np.nan
    pe = probs["ever"]
    out = result.copy()
    out["p_ever"] = (("y", "x"), pe, {"description": "chance the fire ever reaches the cell (survival.py)"})
    out["arrival_h"] = (("y", "x"), median, {"units": "h", "description": "median hours until the fire arrives, "
                                                                          "if it does (survival.py)"})
    res = float(result.attrs.get("res_m", abs(float(result.x[1] - result.x[0]))))
    for hours in HORIZONS_H:
        p = probs[hours]
        out[f"p_arrive_{hours}h"] = (("y", "x"), p, {"description": f"chance the fire arrives within {hours} h"})
        out.attrs[f"arrival_area_{hours}h_km2"] = float(np.nansum(p)) * res * res / 1e6
    out.attrs["arrival_model"] = (f"cure-fraction XGBoost AFT (normal, sigma {model.sigma:g}"
                                  f"{', calibrated on held-out fires' if model.calibration else ''}), trained "
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

    # lower bound: the VIIRS pass before the (final) upper bound, for cells VIIRS saw. Taken from `upper`, not from
    # the first detection: when a FEDS perimeter contains the cell before VIIRS detects it, the pass before the
    # detection can be after the FEDS time and would make a zero-width interval.
    lower = np.full(shape, t_ign)
    if passes.size:
        i = np.searchsorted(passes, upper - PASS_GAP_MIN * 60) - 1
        prev = np.where(i >= 0, passes[np.clip(i, 0, passes.size - 1)], t_ign)
        lower = np.where(detected, np.maximum(prev, t_ign), lower)
    upper[final & ~np.isfinite(upper)] = max(t_end, t_ign + 3600)   # burned, time unknown: by the end of growth
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

BASELINES = ("euclid_dist_m", "ca_burn_prob")   # stored with the samples to score against, not model features


def fire_samples(slug: str, cache: Path = CACHE_DIR, seed: int = 0) -> pd.DataFrame:
    """One row per (start time, cell not yet burning): FEATURES, BASELINES, lower_h / upper_h (inf = never burned),
    weight, fire, start_utc."""
    from scipy.ndimage import distance_transform_edt
    from .alexandridis import spread_probabilities
    from .simulate import simulate_spread
    path = cache / slug / f"samples_v2_{RES_M:g}m.parquet"
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
        result = spread_probabilities(fire_inputs(truth, grid, t0, cache), burning=burning)
        feats = cell_features(result)
        sim = simulate_spread(result, steps=int(math.ceil(REACH_M / grid.res)), runs=100, seed=0)
        s0 = _sec(t0)
        lo = (bounds["lower"] - s0) / 3600
        hi = (bounds["upper"] - s0) / 3600
        # Inside the final perimeter, not burning at t0, never timed by VIIRS (lower = ignition < t0): it burned
        # between t0 and its upper bound.
        lo = np.where(bounds["final"] & (lo < 0), 0.0, lo)
        never = ~np.isfinite(hi)
        keep = ~burning & (hi > 0)
        keep &= ~never | (rng.random(keep.shape) < CENSORED_KEEP)
        df = pd.DataFrame(feature_matrix(feats, keep), columns=list(FEATURES))
        df["euclid_dist_m"] = distance_transform_edt(~burning, sampling=grid.res)[keep]
        df["ca_burn_prob"] = sim["burn_prob"].values[keep]
        df["lower_h"] = np.maximum(lo[keep], MIN_HOURS)
        df["upper_h"] = np.maximum(hi[keep], MIN_HOURS)
        df["weight"] = np.where(never[keep], 1 / CENSORED_KEEP, 1.0)
        df["fire"], df["start_utc"] = slug, t0.isoformat()
        frames.append(df)
        print(f"  {t0:%m-%d %H:%M}: {burning.sum() * grid.res ** 2 / 1e6:.1f} km2 burning, {keep.sum()} samples "
              f"({(~never[keep]).sum()} burned later)", flush=True)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


# ---- training and evaluation ------------------------------------------------------------------

def _constraints(sign: int) -> str:
    return "(" + ",".join(str(sign * MONOTONE_TIME.get(f, 0)) for f in FEATURES) + ")"


COMMON = {"tree_method": "hist", "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 500,
          "subsample": 0.8, "colsample_bytree": 0.8, "seed": 0}
TIME_PARAMS = {**COMMON, "objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
               "monotone_constraints": _constraints(+1)}
# min_child_weight counts the hessian: p (1 - p) <= 0.25 per sample for the classifier, ~1 for the AFT
CURE_PARAMS = {**COMMON, "objective": "binary:logistic", "eval_metric": "logloss", "min_child_weight": 100,
               "monotone_constraints": _constraints(-1)}
ROUNDS = 300
SIGMAS = (0.3, 0.5, 0.7, 1.0, 1.4)


def _X(df: pd.DataFrame) -> np.ndarray:
    return df[list(FEATURES)].to_numpy(np.float32)


def fit(df: pd.DataFrame, sigma: float, rounds: int = ROUNDS, cure=None) -> ArrivalModel:
    """Fit both parts. `cure`: an already fitted classifier to reuse (it does not depend on sigma)."""
    import xgboost as xgb
    burned = np.isfinite(df["upper_h"].to_numpy())
    if cure is None:
        dc = xgb.DMatrix(_X(df), label=burned.astype(np.float32), weight=df["weight"].to_numpy(np.float32),
                         feature_names=list(FEATURES))
        cure = xgb.train(CURE_PARAMS, dc, num_boost_round=rounds)
    b = df[burned]
    dt = xgb.DMatrix(_X(b), feature_names=list(FEATURES))
    dt.set_float_info("label_lower_bound", b["lower_h"].to_numpy(np.float32))
    dt.set_float_info("label_upper_bound", b["upper_h"].to_numpy(np.float32))
    dt.set_weight(b["weight"].to_numpy(np.float32))
    time = xgb.train({**TIME_PARAMS, "aft_loss_distribution_scale": sigma}, dt, num_boost_round=rounds)
    return ArrivalModel(cure, time, sigma)


def weighted_auc(score: np.ndarray, label: np.ndarray, weight: np.ndarray) -> float | None:
    """P(score of a random positive > score of a random negative), with sample weights; ties count half.
    NaN scores rank lowest."""
    score = np.nan_to_num(np.asarray(score, np.float64), nan=-np.inf)
    label = np.asarray(label, bool)
    _, inv = np.unique(score, return_inverse=True)
    wp = np.bincount(inv, weights=weight * label)
    wn = np.bincount(inv, weights=weight * ~label)
    if wp.sum() == 0 or wn.sum() == 0:
        return None
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def _labels(lo, hi, h):
    """Certainly burned within h / certainly not (never burned counts as not, at every horizon)."""
    pos = hi <= h
    neg = (lo >= h) | ~np.isfinite(hi)
    return pos, neg


def evaluate(model: ArrivalModel, df: pd.DataFrame) -> dict:
    """Scores on samples the model has not seen, per start time and then averaged (weighted by the cells each start
    contributes), so cells of different starts are never ranked against each other. For each horizon h, a cell is
    positive if it certainly burned within h hours (upper <= h), negative if it certainly did not (lower >= h, or
    never burned), else skipped. All scores use the sample weights (never-burned cells were subsampled).
      auc_<h>h            AFT mixture P(T <= h) (0.5 = chance, 1 = perfect)
      auc_<base>_<h>h     the same for the baselines: path (physics only: -path_steps), ca (Monte Carlo CA burn
                          probability), dist (-straight-line distance to the fire)
      brier_<h>h          weighted mean squared error of P(T <= h) (brier_raw_<h>h: before calibration)
      auc_ever            the classifier: burned at all vs never
      interval_hit        burned cells with an interval at most 13 h wide whose median-if-burns falls in it
      median_abs_log_err  |ln median - ln mid-interval| for those cells (0.69 = a factor of 2)
      calibration_24h     [predicted bin, mean predicted, observed share, weight] for P(T <= 24 h)
    """
    probs, raw, med = _predict_samples(model, df)
    pe = probs["ever"]
    lo, hi, w = df["lower_h"].to_numpy(), df["upper_h"].to_numpy(), df["weight"].to_numpy()
    starts = df["start_utc"].to_numpy()
    scores = {"aft": None, "path": -df["path_steps"].to_numpy(), "ca": df["ca_burn_prob"].to_numpy(),
              "dist": -df["euclid_dist_m"].to_numpy()}
    out: dict = {"n_samples": int(len(df)), "n_burned": int(np.isfinite(hi).sum()), "n_starts": int(len(set(starts)))}

    def per_start(score, label, known):
        num = den = 0.0
        for s in np.unique(starts):
            m = known & (starts == s)
            auc = weighted_auc(score[m], label[m], w[m]) if m.any() else None
            if auc is not None:
                num, den = num + auc * w[m].sum(), den + w[m].sum()
        return round(num / den, 4) if den else None

    for h in HORIZONS_H:
        pos, neg = _labels(lo, hi, h)
        known = pos | neg
        p = probs[h]
        for name, sc in scores.items():
            key = f"auc_{h}h" if name == "aft" else f"auc_{name}_{h}h"
            out[key] = per_start(p if sc is None else sc, pos, known)
        out[f"brier_{h}h"] = round(float(np.average((p[known] - pos[known]) ** 2, weights=w[known])), 4)
        out[f"brier_raw_{h}h"] = round(float(np.average((raw[h][known] - pos[known]) ** 2, weights=w[known])), 4)
        out[f"n_pos_{h}h"], out[f"n_known_{h}h"] = int(pos.sum()), int(known.sum())
    ever = np.isfinite(hi)
    out["auc_ever"] = per_start(pe, ever, np.ones_like(ever))
    tight = ever & (hi - lo <= 13)
    if tight.any():
        m = med[tight]
        out["interval_hit"] = round(float(((m >= lo[tight]) & (m <= hi[tight])).mean()), 4)
        out["median_abs_log_err"] = round(float(np.median(np.abs(np.log(m) - np.log((lo[tight] + hi[tight]) / 2)))), 4)
        out["n_tight"] = int(tight.sum())
    pos, neg = _labels(lo, hi, 24)
    known = pos | neg
    p, y, ww = probs[24][known], pos[known], w[known]
    calib = []
    for a, b in ((0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        m = (p >= a) & (p < b)
        if m.any():
            calib.append([f"{a:.2f}-{min(b, 1):.2f}", round(float(np.average(p[m], weights=ww[m])), 3),
                          round(float(np.average(y[m], weights=ww[m])), 3), round(float(ww[m].sum()))])
    out["calibration_24h"] = calib
    return out


def _predict_samples(model: ArrivalModel, df: pd.DataFrame) -> tuple[dict, dict, np.ndarray]:
    """(calibrated probabilities, raw probabilities, median if it burns) for samples; unreachable -> never."""
    X = _X(df)
    pe = np.where(np.isfinite(df["path_steps"].to_numpy()), model.p_ever(X), 0.0)  # as on the website
    med = model.median_if_burns(X)
    return model.probabilities(pe, med), model.probabilities(pe, med, raw=True), med


def _mean_brier(model: ArrivalModel, df: pd.DataFrame) -> float:
    _, raw, _ = _predict_samples(model, df)
    lo, hi, w = df["lower_h"].to_numpy(), df["upper_h"].to_numpy(), df["weight"].to_numpy()
    total = []
    for h in HORIZONS_H:
        pos, neg = _labels(lo, hi, h)
        known = pos | neg
        total.append(np.average((raw[h][known] - pos[known]) ** 2, weights=w[known]))
    return float(np.mean(total))


CALIBRATION_BINS = 200


def _isotonic(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> list[list[float]]:
    """Increasing map raw p -> observed share, as [xs, ys] for np.interp. Weighted quantile bins, then isotonic
    regression over the bins (keeps the map small)."""
    from scipy.optimize import isotonic_regression
    order = np.argsort(p, kind="mergesort")
    p, y, w = p[order], y[order].astype(np.float64), w[order]
    edges = np.searchsorted(np.cumsum(w) / w.sum(), np.linspace(0, 1, CALIBRATION_BINS + 1)[1:-1])
    xs, ys, ws = [], [], []
    for part in np.split(np.arange(p.size), np.unique(edges)):
        if part.size:
            ww = w[part]
            xs.append(float(np.average(p[part], weights=ww)))
            ys.append(float(np.average(y[part], weights=ww)))
            ws.append(float(ww.sum()))
    fitted = isotonic_regression(np.array(ys), weights=np.array(ws), increasing=True).x
    xs, fitted = [0.0, *xs, 1.0], [0.0, *fitted, float(fitted[-1])]
    return [[round(x, 6) for x in xs], [round(float(v), 6) for v in fitted]]


def fit_calibration(oof: list[tuple[ArrivalModel, pd.DataFrame]]) -> dict:
    """Calibration maps from out-of-fold predictions: each (model, samples of a fire that model never saw)."""
    raw: dict = {k: [] for k in ("ever", *HORIZONS_H)}
    lab: dict = {k: [] for k in raw}
    wts: dict = {k: [] for k in raw}
    for model, df in oof:
        _, r, _ = _predict_samples(ArrivalModel(model.cure, model.time, model.sigma), df)
        lo, hi, w = df["lower_h"].to_numpy(), df["upper_h"].to_numpy(), df["weight"].to_numpy()
        raw["ever"].append(r["ever"]), lab["ever"].append(np.isfinite(hi)), wts["ever"].append(w)
        for h in HORIZONS_H:
            pos, neg = _labels(lo, hi, h)
            known = pos | neg
            raw[h].append(r[h][known]), lab[h].append(pos[known]), wts[h].append(w[known])
    return {str(k): _isotonic(np.concatenate(raw[k]), np.concatenate(lab[k]), np.concatenate(wts[k])) for k in raw}


def train(fires=TRAIN_FIRES, valid=VALID_FIRES, save: bool = True) -> dict:
    """Leave-one-fire-out to pick sigma (held-out Brier score, which does not reward a too-small sigma the way the
    censored likelihood does), to calibrate the probabilities and to score the method; then fit on all training
    fires and save."""
    data = {s: fire_samples(s) for s in (*fires, *valid)}
    data = {s: d for s, d in data.items() if len(d)}
    train_fires = [s for s in fires if s in data]
    folds = {held: pd.concat([data[s] for s in train_fires if s != held]) for held in train_fires}
    cures = {held: fit(rest, 1.0).cure for held, rest in folds.items()}
    report: dict = {"lofo": {}, "sigma_brier": {}}
    for sigma in SIGMAS:
        losses = [_mean_brier(fit(folds[held], sigma, cure=cures[held]), data[held]) for held in train_fires]
        report["sigma_brier"][str(sigma)] = round(float(np.mean(losses)), 5)
    sigma = float(min(SIGMAS, key=lambda s: report["sigma_brier"][str(s)]))
    report["sigma"] = sigma
    # Calibration from out-of-fold predictions: the maps used to score fire F are learned from the other training
    # fires' held-out predictions only, so F never informs its own calibration.
    fold_models = {held: fit(folds[held], sigma, cure=cures[held]) for held in train_fires}
    for held in train_fires:
        m = fold_models[held]
        m.calibration = fit_calibration([(fold_models[o], data[o]) for o in train_fires if o != held])
        report["lofo"][held] = evaluate(m, data[held])
    model = fit(pd.concat([data[s] for s in train_fires]), sigma)
    model.calibration = fit_calibration([(fold_models[o], data[o]) for o in train_fires])
    report["validation"] = {s: evaluate(model, data[s]) for s in valid if s in data}
    report["feature_gain"] = {
        part: {k: round(v, 2) for k, v in sorted(b.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])}
        for part, b in (("time", model.time), ("ever", model.cure))}
    model.meta = {"trained_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "fires": train_fires,
                  "validation_fires": list(report["validation"]), "res_m": RES_M, "time_params": TIME_PARAMS,
                  "cure_params": CURE_PARAMS, "rounds": ROUNDS,
                  "n_samples": {s: int(len(d)) for s, d in data.items()}, "report": report}
    if save:
        model.save()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_report(report: dict) -> None:
    print(f"\nsigma = {report['sigma']} (mean held-out Brier by sigma: {report['sigma_brier']})")
    rows = {**{f"{k} (held out)": v for k, v in report["lofo"].items()},
            **{f"{k} (validation)": v for k, v in report.get("validation", {}).items()}}
    for name, m in rows.items():
        print(f"\n{name}: {m['n_samples']} samples, {m['n_burned']} burned later, {m['n_starts']} starts; "
              f"ever-burns AUC {m['auc_ever']}")
        for h in HORIZONS_H:
            print(f"  within {h:>2} h: AUC {m[f'auc_{h}h']}  (path {m[f'auc_path_{h}h']}, CA {m[f'auc_ca_{h}h']}, "
                  f"distance {m[f'auc_dist_{h}h']})  Brier {m[f'brier_{h}h']} (raw {m[f'brier_raw_{h}h']})  "
                  f"[{m[f'n_pos_{h}h']} of {m[f'n_known_{h}h']} cells burned]")
        if "interval_hit" in m:
            print(f"  median-if-burns inside the observed interval: {100 * m['interval_hit']:.0f}%, median |ln error| "
                  f"{m['median_abs_log_err']} ({m['n_tight']} cells)")
        print("  calibration 24 h [bin, predicted, observed, weight]: " + "  ".join(map(str, m["calibration_24h"])))
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
