"""Run the spread model from a real FEDS perimeter and score it against the later FEDS perimeters.

Model: firespread.predict (one-step spread probabilities) + firespread.simulate.simulate_spread (Monte Carlo,
`steps` CA steps). Two free knobs, calibrated on Bug's first window and then held fixed:
  hours_per_step : steps = round(hours to target / hours_per_step)
  threshold      : burn_prob >= threshold -> predicted burned

No answer leaks into the inputs: Earth Engine layers and Sentinel-2 burn scars use only data from before the
start time, and NISAR is left out (the post-fire NISAR passes show the final burned area).

Scores are against FEDS, which itself under-maps the official (WFIGS) area: Bug FEDS covers 37.8 % of the
official area, Biscar 11.2 %. A prediction outside FEDS is not necessarily wrong.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from .metrics import circular_growth, scores
from .windows import Window

HINDCAST_DIR = Path(__file__).resolve().parents[2]      # hindcast/
CACHE_DIR = HINDCAST_DIR / "data"                         # not committed
RESULTS_DIR = HINDCAST_DIR / "results"

RES_M = 100.0
RADIUS_KM = 15.0          # grid = start perimeter bbox + 15 km (Bug grew ~3 km in its fastest 24 h)
WEATHER_HOURS = 26        # fetched now for later hourly-weather runs; simulate_spread uses the start hour only
RUNS = 200
HOURS_PER_STEP = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0)
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def grid_of(ds: xr.Dataset):
    """The geelayers Grid of a Dataset, to rasterize the observed perimeters onto the same cells."""
    from geelayers import Grid
    x, y = ds["x"].values, ds["y"].values
    res = float(abs(x[1] - x[0]))
    epsg = int(str(ds.attrs["crs"]).split(":")[-1])
    return Grid(epsg, float(x[0] - res / 2), float(y[0] + res / 2), res, x.size, y.size)


def fetch_inputs(window: Window, cache_dir: Path = CACHE_DIR, refresh: bool = False) -> xr.Dataset:
    """Model inputs at the window start (Earth Engine), cached as NetCDF so calibration costs no extra calls."""
    path = cache_dir / window.fire / f"inputs_{window.start:%Y%m%dT%H%M}.nc"
    if path.exists() and not refresh:
        return xr.load_dataset(path, engine="h5netcdf")
    from firespread import MODEL_LAYERS
    from firespread.burn_scars import optical_burn_scars
    from geelayers import get_fire_inputs
    ds = get_fire_inputs(polygon=window.start_feature, start=window.start, hours=WEATHER_HOURS,
                         radius_km=RADIUS_KM, res_m=RES_M, layers=MODEL_LAYERS)
    ds = optical_burn_scars(ds, time=window.start)     # scars from the 30 days BEFORE the start only
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.attrs = {k: (str(v) if not isinstance(v, (int, float, str)) else v) for k, v in ds.attrs.items()}
    ds.to_netcdf(path, engine="h5netcdf")
    return ds


def spread_result(inputs: xr.Dataset, window: Window) -> xr.Dataset:
    from firespread import predict
    return predict(inputs=inputs, time=window.start, nisar=None, burn_scars=False)


def evaluate_window(window: Window, result: xr.Dataset, hours_per_step=HOURS_PER_STEP, thresholds=THRESHOLDS,
                    runs: int = RUNS) -> pd.DataFrame:
    """Scores for every (hours_per_step, threshold, target) of one window, with both baselines."""
    from firespread.simulate import simulate_spread
    grid = grid_of(result)
    start = result["burning"].values.astype(bool)
    sims: dict[int, xr.Dataset] = {}
    rows = []
    for target in window.scored_targets():
        obs = grid.polygon_mask(target.feature).astype(bool)
        hours = (target.time - window.start).total_seconds() / 3600
        oracle = scores(circular_growth(start, int((obs & ~start).sum())), obs, start, RES_M)
        for hps in hours_per_step:
            steps = max(1, round(hours / hps))
            if steps not in sims:
                sims[steps] = simulate_spread(result, steps=steps, runs=runs, seed=0)
            sim = sims[steps]
            for thr in thresholds:
                pred = (sim["burn_prob"].values >= thr) | start
                s = scores(pred, obs, start, RES_M)
                base = scores(circular_growth(start, int((pred & ~start).sum())), obs, start, RES_M)
                rows.append({"window": window.id, "start_utc": window.start, "horizon_h": target.horizon_h,
                             "target_utc": target.time, "hours": round(hours, 1),
                             "hours_per_step": hps, "steps": steps, "threshold": thr,
                             "reached_grid_edge": int(sim.attrs["sim_reached_grid_edge"]),
                             **s, **{f"base_{k}": v for k, v in base.items() if not k.endswith("_km2")},
                             **{f"oracle_{k}": v for k, v in oracle.items() if k.startswith("iou")}})
    return pd.DataFrame(rows)


def best_params(table: pd.DataFrame, metric: str = "iou") -> pd.Series:
    """(hours_per_step, threshold) with the best mean `metric` over the targets in `table`."""
    mean = table.groupby(["hours_per_step", "threshold"])[metric].mean()
    hps, thr = mean.idxmax()
    return pd.Series({"hours_per_step": hps, "threshold": thr, metric: mean.max()})

