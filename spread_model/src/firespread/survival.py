"""When does the fire reach each cell? A cure-fraction survival model on top of the cellular automaton.

The CA (alexandridis.py) gives a per-step spread probability; its steps have no unit of time, and many cells
around a fire never burn at all (containment, weather, fuel breaks). So the arrival time T of a cell is modelled
as a **mixture cure model**:

    P(T <= h) = pi(x) * Phi((ln h - f(x)) / sigma)

    pi(x)  = P(the fire ever reaches the cell)   logit pi = c0 + c1 ln(path_steps) [+ XGBoost binary:logistic]
    f(x)   = ln of the median arrival, if it does  f = t0 + t1 ln(path_steps) [+ XGBoost AFT, interval-censored]
    sigma  = spread of ln T around f(x)            chosen by held-out Brier score

Both parts are anchored on the physics: a straight line in the log of the fastest-path cost, fitted by maximum
likelihood, which the trees only correct (base_margin). The number of tree rounds, 0 included, is chosen by
held-out AUC: on 3 training fires the trees can learn fire-specific corrections that rank a new fire worse than
the path cost alone, and trees cannot extrapolate (arrival times beyond the training fires' 2-5 days of growth
would flatten out). Then the probabilities are calibrated (isotonic) on predictions for fires the model did not
see.

A plain AFT (P(T < inf) = 1) has to put cells that burn after 3 h and never-burned cells at the same distance on
one distribution, and the never-burned ones drag every median late (3-10x in the first version). Here the AFT only
learns *when*, from cells that did burn; the classifier learns *whether*.

**Features** (cell_features): the fastest path from the fire to each cell over the 8-neighbour grid, where crossing
from cell i to neighbour j costs 1 / rate(i -> j) CA steps (path_spread_rate: the Alexandridis product, unclipped):
for rates below 1, the expected number of steps until a burning cell that keeps trying with probability p
succeeds (geometric waiting time). So a backing or flanking fire is slow, not
extinguished (no percolation die-off as in a Monte Carlo run with one-step burnout), unburnable cells (water,
rock, built-up, burn scars) are walls the path goes around, and the source is whichever burning cell is fastest, not the
nearest. Along that path: its cost (path_steps), length, total climb and descent, and the length-weighted mean wind
along it and P_veg. Weather enters only through the path cost and the along-path wind: per-cell weather is nearly
the same everywhere at one start hour, and with 13 training starts the trees used it to recognise the start.
Every feature has a sign constraint matching the physics (longer / costlier path -> later and less likely).
The path rates leave out the continuous NISAR terms (P_den, P_dry, P_vwc), which training has no data for, so the
website (which has them) sees the same features as training.

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
FEATURES = ("path_steps", "path_len_m", "path_up_m", "path_down_m", "path_wind_along_ms", "path_p_veg",
            "active_steps")
# active_steps: the fastest-path cost from the active front only (result["active_front"]: burning cells with a
# detection in the last ACTIVE_HOURS). That path may cross the already burned area at the calm, flat reference rate
# (the fire creeps along its own edge). Equal to path_steps when no active front is given.
# (time terms, ever-burns terms) the two anchor lines may use (ln of each); train() picks the pair held out.
ANCHOR_CHOICES = ((("path_steps",), ("path_steps",)), (("path_steps",), ("active_steps",)),
                  (("active_steps",), ("active_steps",)))
# Tested held out and not kept (2026-11-15, see spread_model/README.md): straight-line distance to the fire and to
# the active front, the distance round barriers, the fire's growth rate over the previous 12 h and its recent FRP
# (as terms of either line), and GOES lower bounds for the labels (fire_samples still stores them for experiments).
# sign of the effect on the arrival time (+1 = later); the "ever burns" classifier gets the opposite sign
MONOTONE_TIME = {"path_steps": 1, "path_len_m": 1, "path_up_m": -1, "path_down_m": 1,
                 "path_wind_along_ms": -1, "path_p_veg": -1, "active_steps": 1}
ACTIVE_HOURS = 12.0       # a detection this recent makes a burning cell part of the active front
GROWTH_HOURS = 12.0       # fire growth (a stored sample column, not a feature) is measured over this long
GROWTH_MIN_M_H = 1.0      # floor of growth_m_h (a fire that did not grow in GROWTH_HOURS)
NOFUEL_RATE = 0.1         # rate factor of LANDFIRE no-fuel cells in the fastest path (0 = wall); chosen held out
P_MIN = 1e-4              # spread rates below this are no path (cost > 10,000 steps)

# labels / samples
RES_M = 100.0             # same cell size as the website
BUFFER_KM = 5.0           # grid = WFIGS final perimeter + this
VIIRS_FOOTPRINT_M = 300.0  # a VIIRS detection covers the cells within this distance of its centre
PASS_GAP_MIN = 20          # detections closer in time than this belong to the same satellite pass
GOES_HALF_M = 1300.0       # half the side of a GOES pixel here (~6.8 km2, real_fires metrics)
MAX_STARTS = 6
MIN_START_GAP_H = 6.0
START_FRACTION = (0.03, 0.85)  # start times: between 3% and 85% of the final detected area already burned
CENSORED_KEEP = 0.25       # share of never-burned cells kept (with weight 1 / CENSORED_KEEP)
MIN_HOURS = 0.05           # AFT needs lower bounds > 0
REACH_M = 12_000.0         # baseline only: the Monte Carlo CA runs until it could travel this far


# ---- features ---------------------------------------------------------------------------------

def path_spread_rate(result: xr.Dataset) -> np.ndarray:
    """(direction, y, x) spread rate per CA step for the fastest path: the Alexandridis product
    P0 (1 + P_veg) P_emc P_w P_s D P_mask without the continuous NISAR terms (P_den, P_dry, P_vwc: training has no
    NISAR data) and NOT clipped to 1. The CA clips it because a probability cannot exceed 1, but as a rate the
    clipping would make 5 m/s and 20 m/s of wind equally fast, and once both exceed 1, a diagonal step (141 m) would
    cost the same as a straight one (100 m): fire fronts would run 1.41x faster along 45 degrees.
    P_mask keeps every barrier, burn scars (NISAR, Sentinel-2) included: an old scar has no fuel, so the path goes
    around it just like around water or rock."""
    from .alexandridis import DIRECTIONS, AlexandridisParams, shift
    params = AlexandridisParams()
    reason = result["mask_reason"].values
    # LANDFIRE "no fuel" (reason 2) is not a wall: 17-25% of those cells ended inside the Bug and Biscar perimeters
    # (WorldCover calls most of them grassland). They can be crossed at NOFUEL_RATE of the normal rate.
    ok = np.where(reason == 0, 1.0, np.where(reason == 2, NOFUEL_RATE, 0.0)).astype(np.float32)
    p_emc = np.nan_to_num(result["p_emc"].values, nan=1.0) if "p_emc" in result else 1.0
    target = params.p0 * (1 + result["p_veg"].values) * p_emc * ok
    p = np.empty(result["p_wind"].shape, np.float32)
    for k, (_, dr, dc, _) in enumerate(DIRECTIONS):
        diag = params.diagonal_factor if dr and dc else 1.0
        p[k] = shift(target, dr, dc, 0.0) * result["p_wind"].values[k] * result["p_slope"].values[k] * diag
    return np.maximum(np.nan_to_num(p), 0.0)


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
    p = path_spread_rate(result)
    idx = np.arange(n).reshape(h, w)
    src, dst, cost = [], [], []
    for k, (_, dr, dc, _) in enumerate(DIRECTIONS):
        r0, r1, c0, c1 = max(0, -dr), h - max(0, dr), max(0, -dc), w - max(0, dc)
        pk = p[k, r0:r1, c0:c1]
        m = pk > P_MIN
        src.append(idx[r0:r1, c0:c1][m])
        dst.append(idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc][m])
        cost.append(1.0 / pk[m])               # CA steps to cross: the expected wait 1/p of a cell that keeps trying
    src, dst = np.concatenate(src), np.concatenate(dst)
    graph = csr_matrix((np.concatenate(cost), (src, dst)), shape=(n, n))

    nan = np.full((h, w), np.nan, np.float32)
    front = idx[burning].ravel()
    if front.size == 0 or graph.nnz == 0:
        return {k: nan.copy() for k in FEATURES}   # no fire, or nothing around it can burn
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
    active = result["active_front"].values.astype(bool) & burning if "active_front" in result else burning
    if active.any() and not np.array_equal(active, burning):
        from .alexandridis import AlexandridisParams
        prm = AlexandridisParams()
        bs, bd, bc = [], [], []
        for _, dr, dc, _ in DIRECTIONS:                              # edges inside the burned area
            r0, r1, c0, c1 = max(0, -dr), h - max(0, dr), max(0, -dc), w - max(0, dc)
            both = burning[r0:r1, c0:c1] & burning[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            bs.append(idx[r0:r1, c0:c1][both])
            bd.append(idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc][both])
            bc.append(np.full(int(both.sum()), 1.0 / (prm.p0 * (prm.diagonal_factor if dr and dc else 1.0))))
        g2 = graph + csr_matrix((np.concatenate(bc), (np.concatenate(bs), np.concatenate(bd))), shape=(n, n))
        out["active_steps"] = dijkstra(g2, directed=True, indices=idx[active].ravel(), min_only=True)
    else:
        out["active_steps"] = steps
    return {k: np.where(reached, out[k], np.nan).reshape(h, w).astype(np.float32) for k in FEATURES}


def feature_matrix(features: dict[str, np.ndarray], select: np.ndarray | None = None) -> np.ndarray:
    cols = [features[k] if select is None else features[k][select] for k in FEATURES]
    X = np.stack([c.ravel() for c in cols], axis=1)
    return np.where(np.isinf(X), np.nan, X)


# ---- the model --------------------------------------------------------------------------------

def _log_terms(X: np.ndarray, terms) -> np.ndarray:
    """(n, len(terms)): ln of each anchor term (columns of the feature matrix)."""
    X = np.asarray(X, np.float64)
    return np.stack([np.log(np.maximum(X[:, FEATURES.index(k)], 1e-3)) for k in terms], axis=1)


def _line(coef, X: np.ndarray, terms, fill: dict | None = None) -> np.ndarray:
    """coef[0] + sum_k coef[k] ln(term_k). A missing (NaN) term other than path_steps uses fill[term] (the
    training median), so an unknown fire growth means an average fire."""
    L = _log_terms(X, terms)
    for i, k in enumerate(terms):
        if fill and k in fill and k != "path_steps":
            L[:, i] = np.where(np.isnan(L[:, i]), math.log(max(fill[k], 1e-3)), L[:, i])
    return coef[0] + L @ np.asarray(coef[1:], np.float64)


@dataclass
class ArrivalModel:
    """Cure-fraction mixture, both parts anchored on the physics (the fastest-path cost):

        logit P(ever burns) = c0 + sum_k c_k ln(term_k) + cure trees(x)
        ln median arrival   = t0 + sum_k t_k ln(term_k) + time trees(x)      (AFT, normal, scale sigma)

    The terms of each line are anchor["time_terms"] / anchor["cure_terms"] (ANCHOR_CHOICES, chosen held out).

    The two straight lines are fitted by maximum likelihood (fit_anchor); the trees (XGBoost, started from that
    margin with base_margin) only correct them, and may be None (0 rounds: physics curve only). Far outside the
    training range the line keeps growing with path cost, where trees alone would flatten out."""
    cure: object | None
    time: object | None
    sigma: float
    anchor: dict = field(default_factory=lambda: {"terms": ["path_steps"], "cure": [0.0, 0.0], "time": [0.0, 1.0]})
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
            out = {k: np.where(v > 0, np.clip(np.interp(v, *self.calibration[str(k)]) + 1e-6 * v, 0.0, 1.0), 0.0)
                   for k, v in out.items()}                 # an exact 0 (no path) stays 0
            prev = out["ever"]
            for h in reversed(HORIZONS_H):          # keep the order the calibration of each horizon may break
                out[h] = np.minimum(out[h], prev)
                prev = out[h]
        return out

    def _margin(self, part: str, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        base = _line(self.anchor[part], X, self.anchor.get(f"{part}_terms", self.anchor["terms"]),
                     self.anchor.get("fill"))
        booster = self.cure if part == "cure" else self.time
        if booster is None:
            return base
        dm = xgb.DMatrix(X, feature_names=list(self.features), base_margin=base)
        return booster.predict(dm, output_margin=True).astype(np.float64)

    @staticmethod
    def _reached(X: np.ndarray) -> np.ndarray:
        return np.isfinite(np.asarray(X)[:, FEATURES.index("path_steps")])

    def p_ever(self, X: np.ndarray) -> np.ndarray:
        """Chance the fire ever reaches the cell; exactly 0 where no path does."""
        from scipy.special import expit
        return np.where(self._reached(X), expit(self._margin("cure", X)), 0.0)

    def median_if_burns(self, X: np.ndarray) -> np.ndarray:
        """Median hours until arrival if it burns; inf where no path reaches the cell."""
        return np.where(self._reached(X), np.exp(np.clip(self._margin("time", X), -20, 20)), np.inf)

    def p_within(self, p_ever: np.ndarray, median_h: np.ndarray, hours: float) -> np.ndarray:
        from scipy.stats import norm
        return p_ever * norm.cdf((math.log(hours) - np.log(np.maximum(median_h, 1e-6))) / self.sigma)

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for booster, file in ((self.time, path), (self.cure, path.with_suffix(".cure.json"))):
            if booster is not None:
                booster.save_model(str(file))
            elif file.exists():
                file.unlink()                       # no trees for this part: no stale file left behind
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"sigma": self.sigma, "anchor": self.anchor, "features": list(self.features), "distribution": "normal",
             "trees": {"time": self.time is not None, "cure": self.cure is not None},
             "calibration": self.calibration, **self.meta}, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ArrivalModel":
        import xgboost as xgb
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"no arrival-time model at {path}; train it: python -m firespread.survival train")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if tuple(meta["features"]) != FEATURES or "anchor" not in meta:
            raise ValueError(f"model {path} was trained with other features or an older version; retrain it")
        trees = meta.pop("trees")
        boosters = {}
        for part, file in (("time", path), ("cure", path.with_suffix(".cure.json"))):
            boosters[part] = None
            if trees[part]:
                if not file.exists():
                    raise FileNotFoundError(f"{file} is missing; retrain: python -m firespread.survival train")
                boosters[part] = xgb.Booster()
                boosters[part].load_model(str(file))
        return cls(boosters["cure"], boosters["time"], float(meta.pop("sigma")), meta.pop("anchor"),
                   tuple(meta.pop("features")), meta, meta.pop("calibration", None))


_LOADED: dict[Path, ArrivalModel] = {}


def load_model(path: Path = MODEL_PATH) -> ArrivalModel:
    """The saved model, loaded once per process."""
    if path not in _LOADED:
        _LOADED[path] = ArrivalModel.load(path)
    return _LOADED[path]


def arrival_quantiles(probs: dict, qs=(0.25, 0.5, 0.75), sigma: float = 1.0) -> dict:
    """Quantiles (hours) of the arrival time if the cell burns, from the (calibrated) probabilities, so the times
    shown agree with the chances shown. F(h) = P(T <= h) / P(ever) at the HORIZONS_H; Phi^-1(F) is interpolated
    linearly in ln h (exact for a log-normal). Before 6 h and after 48 h it is extended with the steepest of: the end
    segment, the overall 6-48 h slope, and half the model's own slope 1 / sigma. The isotonic calibration is a
    staircase, so an end segment is often flat or nearly so; extending a flat one gave "never" (inf) or 6 h for
    everything, a nearly flat one months. Arrays of any shape; P(ever) = 0 -> inf; NaN in -> NaN out."""
    from scipy.special import ndtri
    pe = np.asarray(probs["ever"], np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.stack([np.clip(np.asarray(probs[h], np.float64) / pe, 1e-4, 1 - 1e-4) for h in HORIZONS_H])
    Z = ndtri(F)                                                   # (horizon, ...)
    L = np.log(np.asarray(HORIZONS_H, np.float64)).reshape(-1, *([1] * pe.ndim))
    n = len(HORIZONS_H)
    overall = (Z[n - 1] - Z[0]) / (L[n - 1] - L[0])
    floor = 0.5 / max(float(sigma), 1e-3)
    s_first = np.maximum.reduce([(Z[1] - Z[0]) / (L[1] - L[0]), overall, np.full(pe.shape, floor)])
    s_last = np.maximum.reduce([(Z[n - 1] - Z[n - 2]) / (L[n - 1] - L[n - 2]), overall, np.full(pe.shape, floor)])
    return {q: _quantile(float(ndtri(q)), pe, Z, L, s_first, s_last, n) for q in qs}


def _quantile(zq, pe, Z, L, s_first, s_last, n):
    """The hours at which Phi^-1(F) reaches zq (see arrival_quantiles); s_first, s_last > 0."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):   # np.where evaluates both branches
        done = zq <= Z[0]                                          # reached by the first horizon: extend downwards
        res = np.where(done, np.exp(L[0] + (zq - Z[0]) / s_first), np.inf)
        for k in range(n - 1):                                     # between horizons k and k+1
            hit = ~done & (zq <= Z[k + 1])
            frac = (zq - Z[k]) / np.maximum(Z[k + 1] - Z[k], 1e-12)
            res = np.where(hit, np.exp(L[k] + frac * (L[k + 1] - L[k])), res)
            done |= hit
        res = np.where(done, res, np.exp(L[n - 1] + (zq - Z[n - 1]) / s_last))   # beyond the last: extend upwards
        res = np.clip(res, 0.05, 24 * 60.0)                        # 3 minutes .. 60 days
    res = res.reshape(pe.shape)
    res[~(pe > 0)] = np.inf
    res[np.isnan(pe)] = np.nan
    return res


def predict_arrival(result: xr.Dataset, time_utc=None, model: ArrivalModel | None = None) -> xr.Dataset:
    """Add to a spread_probabilities() / simulate_spread() result:
      p_ever         chance the fire ever reaches the cell
      p_arrive_<h>h  chance it arrives within h hours (h in HORIZONS_H)
      arrival_h      median hours until it arrives, if it does (arrival_h_q25 / _q75: its 25-75% range), read off
                     the same calibrated chances (arrival_quantiles), so times and chances agree
    Burning cells -> NaN. Cells no path reaches, or that cannot burn (WorldCover built-up / bare / water, burn scars)
    -> p_ever 0, arrival inf. With result["active_front"] (attach_active_front) the chance of ever burning is
    measured from the active part of the fire. `time_utc` is not a feature: the weather in `result` is the start time's."""
    model = model or load_model()
    burning = result["burning"].values.astype(bool)
    feats = cell_features(result, time_utc)
    X = feature_matrix(feats)
    shape = burning.shape
    probs = {k: v.reshape(shape).astype(np.float64)
             for k, v in model.probabilities(model.p_ever(X), model.median_if_burns(X)).items()}
    # never: no path, or the cell cannot burn. LANDFIRE "no fuel" (2) can (NOFUEL_RATE, see path_spread_rate).
    never = ~np.isfinite(feats["path_steps"]) | ~np.isin(result["mask_reason"].values, (0, 2))
    for v in probs.values():
        v[never], v[burning] = 0.0, np.nan
    quant = arrival_quantiles(probs, sigma=model.sigma)
    out = result.copy()
    out["p_ever"] = (("y", "x"), probs["ever"].astype(np.float32),
                     {"description": "chance the fire ever reaches the cell (survival.py)"})
    for name, q, what in (("arrival_h", 0.5, "median"), ("arrival_h_q25", 0.25, "25% quantile of the"),
                          ("arrival_h_q75", 0.75, "75% quantile of the")):
        out[name] = (("y", "x"), quant[q].astype(np.float32),
                     {"units": "h", "description": f"{what} hours until the fire arrives, if it does (survival.py)"})
    res = float(result.attrs.get("res_m", abs(float(result.x[1] - result.x[0]))))
    for hours in HORIZONS_H:
        p = probs[hours].astype(np.float32)
        out[f"p_arrive_{hours}h"] = (("y", "x"), p, {"description": f"chance the fire arrives within {hours} h"})
        out.attrs[f"arrival_area_{hours}h_km2"] = float(np.nansum(p)) * res * res / 1e6
    trees = [part for part, b in (("ever", model.cure), ("time", model.time)) if b is not None]
    out.attrs["arrival_model"] = (f"cure-fraction AFT on the fastest-path cost (normal, sigma {model.sigma:.2f}"
                                  f"{', trees: ' + ', '.join(trees) if trees else ', physics curve only'}"
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

    # GOES (5 min, ~6.8 km2 pixels): a detection means fire somewhere in the pixel, not at every cell in it (the
    # first GOES detection is before the VIIRS lower bound for 44-70% of the timed cells), so it cannot be an upper
    # bound. But a cell cannot burn before its pixel is first detected: a lower bound, where it does not contradict
    # the upper one.
    goes = truth.points[truth.points["source"].str.startswith("GOES")]
    goes_first = np.full(shape, np.inf)
    if len(goes):
        px = goes.groupby(["lat", "lon"])["time_utc"].min().reset_index()
        gx, gy = Transformer.from_crs(4326, grid.epsg, always_xy=True).transform(px["lon"].values, px["lat"].values)
        kk = int(GOES_HALF_M // grid.res)
        for x, y, when in zip(gx, gy, px["time_utc"].map(lambda v: v.timestamp())):
            r, c = int((grid.y0 - y) // grid.res), int((x - grid.x0) // grid.res)
            r0, r1, c0, c1 = max(r - kk, 0), min(r + kk + 1, shape[0]), max(c - kk, 0), min(c + kk + 1, shape[1])
            if r0 < r1 and c0 < c1:
                goes_first[r0:r1, c0:c1] = np.minimum(goes_first[r0:r1, c0:c1], when)
    ok = final & np.isfinite(goes_first) & (goes_first <= upper)
    lower_goes = np.where(ok, np.maximum(lower, goes_first), lower)
    return {"lower": lower, "upper": upper, "final": final, "passes": passes, "t_end": t_end, "t_ign": t_ign,
            "lower_goes": lower_goes, "viirs_first": first}


def detections_mask(result_or_grid, lon, lat, radius_m: float = VIIRS_FOOTPRINT_M) -> np.ndarray:
    """Cells within radius_m of any (lon, lat) point, on the grid of a spread result (coords x, y in attrs["crs"])
    or of a geelayers Grid."""
    from pyproj import Transformer
    if hasattr(result_or_grid, "epsg"):
        g = result_or_grid
        crs, x0, y0, res, shape = g.epsg, g.x0, g.y0, g.res, g.shape
    else:
        r = result_or_grid
        res = float(r.attrs.get("res_m", abs(float(r.x[1] - r.x[0]))))
        crs, x0, y0 = r.attrs["crs"], float(r.x.values[0]) - res / 2, float(r.y.values[0]) + res / 2
        shape = (r.sizes["y"], r.sizes["x"])
    out = np.zeros(shape, bool)
    lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
    if lon.size == 0:
        return out
    xs, ys = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    r0 = np.floor((y0 - np.asarray(ys)) / res).astype(int)
    c0 = np.floor((np.asarray(xs) - x0) / res).astype(int)
    k = int(math.ceil(radius_m / res))
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if math.hypot(dr, dc) * res <= radius_m:
                r, c = r0 + dr, c0 + dc
                ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
                out[r[ok], c[ok]] = True
    return out


def active_front(grid, truth: FireTruth, bounds: dict, t0: pd.Timestamp, burning: np.ndarray) -> np.ndarray:
    """Training: burning cells with a VIIRS detection (footprint VIIRS_FOOTPRINT_M) in the ACTIVE_HOURS before t0.
    None at all -> the whole burning area."""
    v = truth.points[truth.points["source"].str.startswith("VIIRS")]
    v = v[(v["time_utc"] > t0 - pd.Timedelta(hours=ACTIVE_HOURS)) & (v["time_utc"] <= t0)]
    active = detections_mask(grid, v["lon"].values, v["lat"].values) & burning
    return active if active.any() else burning.copy()


def attach_active_front(result: xr.Dataset, detections: pd.DataFrame | None, start) -> xr.Dataset:
    """Website: mark the active front from fire detections (columns time_utc, lat, lon; e.g. VIIRS from FIRMS) in the
    ACTIVE_HOURS before `start`. No detection on the fire -> unchanged (the whole polygon is the front).
    Sets attrs active_front_km2 and active_detections."""
    out = result.copy()
    out.attrs["active_detections"] = 0
    if detections is None or len(detections) == 0:
        return out
    t0 = _utc(start)
    when = pd.to_datetime(detections["time_utc"], utc=True)
    d = detections[(when > t0 - pd.Timedelta(hours=ACTIVE_HOURS)) & (when <= t0)]
    burning = result["burning"].values.astype(bool)
    active = detections_mask(result, d["lon"].values, d["lat"].values) & burning
    if not active.any():
        return out
    res = float(result.attrs.get("res_m", abs(float(result.x[1] - result.x[0]))))
    out["active_front"] = (("y", "x"), active)
    out.attrs.update({"active_detections": int(len(d)), "active_front_km2": float(active.sum()) * res * res / 1e6})
    return out


def growth_rate(area_now_km2: float, area_before_km2: float, hours: float) -> float:
    """How fast the fire has been growing, as metres per hour of the radius of a circle with the same area:
    (sqrt(A_now / pi) - sqrt(A_before / pi)) / hours, at least GROWTH_MIN_M_H. Works for any two outlines of the
    same fire `hours` apart (training: the detected area GROWTH_HOURS before the start; website: an earlier
    polygon)."""
    if not (hours > 0):
        return float("nan")
    r = lambda a: math.sqrt(max(a, 0.0) * 1e6 / math.pi)   # noqa: E731
    return max((r(area_now_km2) - r(area_before_km2)) / hours, GROWTH_MIN_M_H)


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
# also stored, for experiments: lower_goes_h (label variant), growth_m_h (growth_rate over GROWTH_HOURS before the
# start), frp_12h_mw (VIIRS FRP in the 12 h before the start)


def fire_samples(slug: str, cache: Path = CACHE_DIR, seed: int = 0) -> pd.DataFrame:
    """One row per (start time, cell not yet burning): FEATURES, BASELINES, lower_h / upper_h (inf = never burned),
    weight, fire, start_utc."""
    from scipy.ndimage import distance_transform_edt
    from .alexandridis import spread_probabilities
    from .simulate import simulate_spread
    path = cache / slug / f"samples_v6_{RES_M:g}m.parquet"
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
        # what was known at t0 only: detections up to t0, and the area detected GROWTH_HOURS earlier
        result["active_front"] = (("y", "x"), active_front(grid, truth, bounds, t0, burning))
        back = min(GROWTH_HOURS, (_sec(t0) - bounds["t_ign"]) / 3600)
        cell_km2 = grid.res ** 2 / 1e6
        before = burning_at(bounds, t0 - pd.Timedelta(hours=back)).sum() * cell_km2
        growth = growth_rate(burning.sum() * cell_km2, before, back)
        recent = truth.points[truth.points["source"].str.startswith("VIIRS")
                              & (truth.points["time_utc"] > t0 - pd.Timedelta(hours=12))
                              & (truth.points["time_utc"] <= t0)]
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
        lo_goes = np.where(bounds["final"], np.maximum((bounds["lower_goes"] - s0) / 3600, lo), lo)
        df["lower_goes_h"] = np.minimum(np.maximum(lo_goes[keep], MIN_HOURS), df["upper_h"].to_numpy())
        df["frp_12h_mw"], df["growth_m_h"] = float(recent["frp_mw"].sum()), growth
        df["weight"] = np.where(never[keep], 1 / CENSORED_KEEP, 1.0)
        df["fire"], df["start_utc"] = slug, t0.isoformat()
        frames.append(df)
        print(f"  {t0:%m-%d %H:%M}: {burning.sum() * grid.res ** 2 / 1e6:.1f} km2 burning "
              f"({result['active_front'].values.sum() * cell_km2:.1f} active, growing {growth:.0f} m/h, "
              f"FRP {recent['frp_mw'].sum():.0f} MW), {keep.sum()} samples ({(~never[keep]).sum()} burned later)",
              flush=True)
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
TRUNCATE = False             # right-truncated time likelihood (fit_anchor); train() chooses it held out
ROUNDS_GRID = (0, 100, 300)   # tree rounds on top of the anchor lines; chosen by held-out AUC


def _X(df: pd.DataFrame) -> np.ndarray:
    X = df[list(FEATURES)].to_numpy(np.float32)
    return np.where(np.isinf(X), np.nan, X)        # e.g. no way round a barrier: missing, not infinite


def _reachable(df: pd.DataFrame) -> np.ndarray:
    """Cells a fastest path reaches. The others are "never" by construction (the website sets them to 0), so they
    are left out of fitting and calibration; they stay in the scores."""
    return np.isfinite(df["path_steps"].to_numpy())


def fit_anchor(df: pd.DataFrame, sigma: float | None = None, truncate: bool | None = None,
               terms=("path_steps",), cure_terms=None) -> dict:
    """Maximum-likelihood straight lines in the logs of the anchor terms, on reachable cells:
      cure: logit P(ever burns) = c0 + sum c_k ln(cure_term_k)                 (weighted logistic likelihood)
      time: ln T ~ Normal(t0 + sum t_k ln(term_k), sigma), T in [lower, upper]  (interval-censored, burned cells)
    `cure_terms` defaults to `terms`. sigma is estimated with the line when not given (the probability scale is set
    by the calibration afterwards). `truncate` (default TRUNCATE): right-truncated likelihood
    P(T in [lower, upper] | T < t_end - start), since a cell only counts as burned if it burned before its fire
    stopped growing; train() keeps it only if it times held-out fires better.
    Returns {"terms", "time_terms", "cure_terms", "cure": [c0, ...], "time": [t0, ...], "fill": {term: training
    median, used when a term is missing at prediction time}, "sigma"}."""
    from scipy.optimize import minimize
    from scipy.special import expit, log_ndtr
    terms = list(terms)
    cure_terms = list(cure_terms) if cure_terms is not None else terms
    d = df[_reachable(df)]
    X = _X(d)
    fill = {k: float(np.nanmedian(X[:, FEATURES.index(k)])) for k in {*terms, *cure_terms} if k != "path_steps"}
    w = d["weight"].to_numpy(np.float64)
    y = np.isfinite(d["upper_h"].to_numpy())
    opts = {"xatol": 1e-5, "fatol": 1e-9, "maxiter": 8000, "maxfev": 16000}

    xc = _log_terms(X, cure_terms)
    kc = len(cure_terms)

    def nll_cure(c):
        z = c[0] + xc @ c[1:]
        return -np.sum(w * (y * np.log(expit(z) + 1e-12) + (~y) * np.log(expit(-z) + 1e-12))) / w.sum()
    cure = minimize(nll_cure, np.array([0.0, *([-0.5 / kc] * kc)]), method="Nelder-Mead", options=opts).x

    k = len(terms)
    xb, wb = _log_terms(X, terms)[y], w[y]
    llo, lhi = np.log(d["lower_h"].to_numpy()[y]), np.log(d["upper_h"].to_numpy()[y])
    # growth left at each start = lower bound of its never-burned cells (t_end - start); no truncation if unknown
    never = d.loc[~y]
    truncate = TRUNCATE if truncate is None else truncate
    left = never.groupby("start_utc")["lower_h"].max() if truncate and len(never) else pd.Series(dtype=float)
    ltr = np.log(d.loc[y, "start_utc"].map(left).to_numpy(np.float64)) if len(left) else np.full(y.sum(), np.inf)
    ltr = np.where(np.isfinite(ltr), np.maximum(ltr, lhi), np.inf)        # never below a cell's own upper bound

    def nll_time(c):
        mu, sg = c[0] + xb @ c[1:k + 1], (sigma if sigma is not None else math.exp(c[k + 1]))
        lu, ll = log_ndtr((lhi - mu) / sg), log_ndtr((llo - mu) / sg)
        logp = lu + np.log1p(-np.exp(np.minimum(ll - lu, -1e-12)))       # ln(Phi(z_U) - Phi(z_L)), stable
        logp -= np.where(np.isfinite(ltr), log_ndtr((ltr - mu) / sg), 0.0)      # / P(T < end of growth)
        return -np.sum(wb * logp) / wb.sum()
    start = np.array([0.0, *([1.0 / k] * k)] + ([] if sigma is not None else [0.0]))
    c = minimize(nll_time, start, method="Nelder-Mead", options=opts).x
    return {"terms": terms, "time_terms": terms, "cure_terms": cure_terms, "cure": [float(v) for v in cure],
            "time": [float(v) for v in c[:k + 1]], "fill": fill,
            "sigma": float(sigma if sigma is not None else math.exp(c[k + 1]))}


def fit(df: pd.DataFrame, sigma: float | None = None, rounds: int = ROUNDS, cure_part: tuple | None = None,
        truncate: bool | None = None, terms=("path_steps",), cure_terms=None) -> ArrivalModel:
    """Anchor lines + `rounds` of trees on top of them (0 = physics curve only), on reachable cells. sigma: None =
    maximum likelihood with the anchor line. `cure_part`: (cure coefficients, booster) of a fitted "ever burns"
    part to reuse (same cure terms)."""
    import xgboost as xgb
    d = df[_reachable(df)]
    X = _X(d)
    burned = np.isfinite(d["upper_h"].to_numpy())
    anchor = fit_anchor(d, sigma, truncate, terms, cure_terms)
    sigma = anchor.pop("sigma")
    if cure_part is not None:
        anchor["cure"], cure = cure_part
    else:
        cure = None
        if rounds:
            dc = xgb.DMatrix(X, label=burned.astype(np.float32), weight=d["weight"].to_numpy(np.float32),
                             feature_names=list(FEATURES),
                             base_margin=_line(anchor["cure"], X, anchor["cure_terms"], anchor["fill"]))
            cure = xgb.train(CURE_PARAMS, dc, num_boost_round=rounds)
    time = None
    if rounds:
        b = d[burned]
        dt = xgb.DMatrix(X[burned], feature_names=list(FEATURES),
                         base_margin=_line(anchor["time"], X[burned], anchor["time_terms"], anchor["fill"]))
        dt.set_float_info("label_lower_bound", b["lower_h"].to_numpy(np.float32))
        dt.set_float_info("label_upper_bound", b["upper_h"].to_numpy(np.float32))
        dt.set_weight(b["weight"].to_numpy(np.float32))
        time = xgb.train({**TIME_PARAMS, "aft_loss_distribution_scale": sigma}, dt, num_boost_round=rounds)
    return ArrivalModel(cure, time, sigma, anchor)


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
      interval_hit        burned cells with an interval at most 13 h wide whose median arrival falls in it: the
                          median the website shows (arrival_quantiles of the calibrated chances)
      median_abs_log_err  |ln median - ln mid-interval| for those cells (0.69 = a factor of 2)
      ..._raw             the same for the uncalibrated median of the time part (median_if_burns)
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
        mid = np.log((lo[tight] + hi[tight]) / 2)
        shown = arrival_quantiles({k: v[tight] for k, v in probs.items()}, qs=(0.5,), sigma=model.sigma)[0.5]
        for suffix, m in (("", shown), ("_raw", med[tight])):
            out[f"interval_hit{suffix}"] = round(float(((m >= lo[tight]) & (m <= hi[tight])).mean()), 4)
            out[f"median_abs_log_err{suffix}"] = round(float(np.median(np.abs(np.log(m) - mid))), 4)
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
    pe, med = model.p_ever(X), model.median_if_burns(X)     # unreachable: 0 and inf, as on the website
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


def _isotonic(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> list[list[float]]:
    """Increasing map raw p -> observed share, as [xs, ys] for np.interp: weighted isotonic regression on every cell
    (equal raw values merged first), kept as the two ends of each constant block. No binning: quantile bins put
    almost all bins below 5% and left the high-probability tail to a handful of them."""
    from scipy.optimize import isotonic_regression
    ux, inv = np.unique(np.round(p, 6), return_inverse=True)
    wsum = np.bincount(inv, weights=w)
    ymean = np.bincount(inv, weights=w * y) / wsum
    fitted = isotonic_regression(ymean, weights=wsum, increasing=True).x
    change = np.flatnonzero(np.diff(fitted) != 0)
    keep = np.unique(np.concatenate([[0, ux.size - 1], change, change + 1]))   # both ends of every block
    return [[round(float(v), 6) for v in ux[keep]], [round(float(v), 6) for v in fitted[keep]]]


def fit_calibration(oof: list[tuple[ArrivalModel, pd.DataFrame]]) -> dict:
    """Calibration maps from out-of-fold predictions: each (model, samples of a fire that model never saw).
    Only reachable cells: the unreachable ones are set to "never" anyway, and their raw 0 would pull the bottom of
    the map up (some did burn: spotting, cut-off cells inside the final perimeter)."""
    raw: dict = {k: [] for k in ("ever", *HORIZONS_H)}
    lab: dict = {k: [] for k in raw}
    wts: dict = {k: [] for k in raw}
    for model, df in oof:
        df = df[_reachable(df)]
        bare = ArrivalModel(model.cure, model.time, model.sigma, model.anchor)    # uncalibrated copy
        _, r, _ = _predict_samples(bare, df)
        lo, hi, w = df["lower_h"].to_numpy(), df["upper_h"].to_numpy(), df["weight"].to_numpy()
        raw["ever"].append(r["ever"]), lab["ever"].append(np.isfinite(hi)), wts["ever"].append(w)
        for h in HORIZONS_H:
            pos, neg = _labels(lo, hi, h)
            known = pos | neg
            raw[h].append(r[h][known]), lab[h].append(pos[known]), wts[h].append(w[known])
    return {str(k): _isotonic(np.concatenate(raw[k]), np.concatenate(lab[k]), np.concatenate(wts[k])) for k in raw}


def _mean_auc(scores: dict) -> float:
    vals = [scores[f"auc_{h}h"] for h in HORIZONS_H if scores.get(f"auc_{h}h") is not None]
    return float(np.mean(vals)) if vals else 0.0


def train(fires=TRAIN_FIRES, valid=VALID_FIRES, save: bool = True) -> dict:
    """Leave-one-fire-out over the training fires (Crosswhite plays no part in any choice):
      1. the anchor terms (ANCHOR_CHOICES, a pair for the time and the ever-burns line): best held-out AUC with the
         plain likelihood and no trees;
         then for each time likelihood (plain / right-truncated) and tree rounds on top of the anchor lines
         (ROUNDS_GRID, 0 = physics curve only): held-out AUC and timing error. Rounds: best AUC (ranking is the map's
         job); then the likelihood whose models time held-out fires best (median |ln error| of the median shown);
      2. sigma is estimated by maximum likelihood with the anchor line (the calibration sets the probability scale,
         so it need not be tuned on the Brier score);
      3. calibration from out-of-fold predictions, and the held-out scores;
    then fit on all training fires and save."""
    data = {s: fire_samples(s) for s in (*fires, *valid)}
    data = {s: d for s, d in data.items() if len(d)}
    train_fires = [s for s in fires if s in data]
    folds = {held: pd.concat([data[s] for s in train_fires if s != held]) for held in train_fires}
    report: dict = {"lofo": {}, "anchor_selection": {}, "selection": {}}
    name = lambda c: f"time {'+'.join(c[0])} / ever {'+'.join(c[1])}"   # noqa: E731
    for choice in ANCHOR_CHOICES:
        scores = [evaluate(fit(folds[held], None, 0, truncate=False, terms=choice[0], cure_terms=choice[1]),
                           data[held]) for held in train_fires]
        report["anchor_selection"][name(choice)] = {
            "auc": round(float(np.mean([_mean_auc(s) for s in scores])), 4),
            "brier24": round(float(np.mean([s["brier_raw_24h"] for s in scores])), 5),
            "log_err": round(float(np.mean([s.get("median_abs_log_err", np.inf) for s in scores])), 4)}
    terms, cure_terms = max(ANCHOR_CHOICES, key=lambda c: report["anchor_selection"][name(c)]["auc"])
    report["terms"], report["cure_terms"] = list(terms), list(cure_terms)
    best: dict = {}
    for truncate in (False, True):
        for rounds in ROUNDS_GRID:
            scores = [evaluate(fit(folds[held], None, rounds, truncate=truncate, terms=terms,
                                   cure_terms=cure_terms), data[held]) for held in train_fires]
            auc = float(np.mean([_mean_auc(s) for s in scores]))
            err = float(np.mean([s.get("median_abs_log_err", np.inf) for s in scores]))
            report["selection"][f"truncate={truncate} rounds={rounds}"] = {"auc": round(auc, 4), "log_err": round(err, 4)}
            if truncate not in best or auc > best[truncate][1]:
                best[truncate] = (rounds, auc, err)
    truncate = min(best, key=lambda k: best[k][2])
    rounds = best[truncate][0]
    report.update({"truncate": truncate, "rounds": rounds})
    # Calibration from out-of-fold predictions: the maps used to score fire F are learned from the other training
    # fires' held-out predictions only, so F never informs its own calibration.
    fold_models = {held: fit(folds[held], None, rounds, truncate=truncate, terms=terms, cure_terms=cure_terms)
                   for held in train_fires}
    for held in train_fires:
        m = fold_models[held]
        m.calibration = fit_calibration([(fold_models[o], data[o]) for o in train_fires if o != held])
        report["lofo"][held] = evaluate(m, data[held])
    model = fit(pd.concat([data[s] for s in train_fires]), None, rounds, truncate=truncate, terms=terms,
                cure_terms=cure_terms)
    model.calibration = fit_calibration([(fold_models[o], data[o]) for o in train_fires])
    report["validation"] = {s: evaluate(model, data[s]) for s in valid if s in data}
    report["anchor"], report["sigma"] = model.anchor, model.sigma
    report["feature_gain"] = {
        part: {k: round(v, 2) for k, v in sorted(b.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])}
        for part, b in (("time", model.time), ("ever", model.cure)) if b is not None}
    model.meta = {"trained_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "fires": train_fires,
                  "validation_fires": list(report["validation"]), "res_m": RES_M, "time_params": TIME_PARAMS,
                  "cure_params": CURE_PARAMS, "rounds": rounds, "truncate": truncate,
                  "n_samples": {s: int(len(d)) for s, d in data.items()}, "report": report}
    if save:
        model.save()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_report(report: dict) -> None:
    print(f"\nchosen: truncated likelihood {report['truncate']}, tree rounds {report['rounds']}, "
          f"sigma {report['sigma']:.3f} (maximum likelihood), anchor {report['anchor']}")
    print("anchor terms, held-out mean AUC / raw Brier 24 h / timing |ln error|: " + "; ".join(
        f"{k}: {v['auc']} / {v['brier24']} / {v['log_err']}" for k, v in report["anchor_selection"].items()))
    print("held-out mean AUC / timing |ln error| by choice: " + "; ".join(
        f"{k}: {v['auc']} / {v['log_err']}" for k, v in report["selection"].items()))
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
            print(f"  median arrival (as shown) inside the observed interval: {100 * m['interval_hit']:.0f}%, median "
                  f"|ln error| {m['median_abs_log_err']} (uncalibrated {m['median_abs_log_err_raw']}; "
                  f"{m['n_tight']} cells)")
        print("  calibration 24 h [bin, predicted, observed, weight]: " + "  ".join(map(str, m["calibration_24h"])))
    if report["feature_gain"]:
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
