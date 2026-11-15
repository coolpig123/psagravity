"""Run the Alexandridis cellular automaton for several steps (Monte Carlo) -> burn probability per cell.

spread_probabilities() gives, for one step, P(burning cell -> each of its 8 neighbours) (`p_spread`). Here the
automaton is run `steps` times in `runs` independent random realisations, as in Alexandridis et al. (2008):

  - at step 0 only the active front burns (result["active_front"], e.g. VIIRS hotspots; default the whole burned
    area); the rest of the burned area is burned out;
  - a burning cell tries to ignite each neighbour once, with probability p_spread[direction, cell];
  - with spotting, a burning cell may also throw a firebrand downwind (p_spot_launch, spot_dist_m, spot_dir_deg);
  - a cell burns for one step, then is burned out (it cannot spread again or be re-ignited);
  - newly ignited cells are the burning cells of the next step.

burn_prob[y, x] = share of runs in which the cell has burned within `steps` steps (the initial fire excluded),
arrival_step[y, x] = mean step at which it ignited, over the runs where it did (NaN where it never did).

The spread probabilities are held fixed for all steps (the weather at the start time). One step moves the fire at
most one cell (res_m), so 30 steps at 100 m reach at most 3 km (diagonals 4.2 km). How long a step takes in hours is
not calibrated yet.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .alexandridis import DIRECTIONS

DEFAULT_STEPS = 30
DEFAULT_RUNS = 200


def simulate_spread(result: xr.Dataset, steps: int = DEFAULT_STEPS, runs: int = DEFAULT_RUNS,
                    seed: int | None = 0, spotting: bool = True, front: np.ndarray | None = None) -> xr.Dataset:
    """Add burn_prob and arrival_step (y, x) to a spread_probabilities() result. Deterministic for a fixed seed.

    front: the cells burning at step 0 (the active front, e.g. VIIRS); default result["active_front"] if present,
    else all of result["burning"]. The rest of the burned area is burned out: it never spreads nor re-burns.
    An empty front falls back to the whole burned area."""
    if steps < 1 or runs < 1:
        raise ValueError("steps and runs must be >= 1")
    p = np.nan_to_num(result["p_spread"].values.astype(np.float32))
    start = result["burning"].values.astype(bool)
    h, w = start.shape
    if front is None and "active_front" in result:
        front = result["active_front"].values
    front = start if front is None else np.asarray(front).astype(bool)
    if front.shape != start.shape:
        raise ValueError(f"front has shape {front.shape}, the grid is {start.shape}")
    if not front.any():
        front = start
    start = start | front
    res = float(result.attrs.get("res_m", abs(float(result.x[1] - result.x[0]))))
    rng = np.random.default_rng(seed)

    use_spot = (spotting and "p_spot_launch" in result and "p_spot_ignite" in result
                and float(np.nanmax(result["p_spot_launch"].values)) > 0)
    if use_spot:
        p_launch = np.nan_to_num(result["p_spot_launch"].values.astype(np.float32))
        p_ignite = np.nan_to_num(result["p_spot_ignite"].values.astype(np.float32))
        spot_dist = np.nan_to_num(result["spot_dist_m"].values.astype(np.float32))
        spot_dir = np.nan_to_num(result["spot_dir_deg"].values.astype(np.float32))
        spot_min_m = float(result.attrs.get("spot_min_m", 200.0))
        spot_max_m = float(result.attrs.get("spot_max_m", 3000.0))
        spot_cone_deg = float(result.attrs.get("spot_cone_deg", 15.0))

    burned = np.broadcast_to(start, (runs, h, w)).copy()
    front_r, front_y, front_x = np.nonzero(np.broadcast_to(front, (runs, h, w)))   # burning cells (run, row, col)
    hits = np.zeros((h, w), np.int32)
    step_sum = np.zeros((h, w), np.int64)
    for step in range(1, steps + 1):
        if front_r.size == 0:
            break
        new_r, new_y, new_x = [], [], []
        for k, (_, dr, dc, _) in enumerate(DIRECTIONS):
            ty, tx = front_y + dr, front_x + dc
            inside = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
            r, sy, sx, ty, tx = front_r[inside], front_y[inside], front_x[inside], ty[inside], tx[inside]
            go = rng.random(r.size, dtype=np.float32) < p[k, sy, sx]
            new_r.append(r[go]), new_y.append(ty[go]), new_x.append(tx[go])
        if use_spot:
            launched = rng.random(front_r.size, dtype=np.float32) < p_launch[front_y, front_x]
            if launched.any():
                sr, sy, sx = front_r[launched], front_y[launched], front_x[launched]
                bearing = np.radians(spot_dir[sy, sx] + rng.normal(0.0, spot_cone_deg, sr.size))
                scale = np.maximum(spot_dist[sy, sx] - spot_min_m, 1.0)
                dist = np.minimum(spot_min_m + rng.exponential(scale), spot_max_m)
                ty = sy + np.rint(-dist * np.cos(bearing) / res).astype(np.int64)
                tx = sx + np.rint(dist * np.sin(bearing) / res).astype(np.int64)
                inside = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
                sr, ty, tx = sr[inside], ty[inside], tx[inside]
                lit_spot = rng.random(sr.size, dtype=np.float32) < p_ignite[ty, tx]
                new_r.append(sr[lit_spot]), new_y.append(ty[lit_spot]), new_x.append(tx[lit_spot])
        r, y, x = np.concatenate(new_r), np.concatenate(new_y), np.concatenate(new_x)
        fresh = ~burned[r, y, x]
        r, y, x = r[fresh], y[fresh], x[fresh]
        flat = np.unique((r.astype(np.int64) * h + y) * w + x)  # a cell lit from two sides ignites once
        r, rest = np.divmod(flat, h * w)
        y, x = np.divmod(rest, w)
        burned[r, y, x] = True
        np.add.at(hits, (y, x), 1)
        np.add.at(step_sum, (y, x), step)
        front_r, front_y, front_x = r, y, x

    out = result.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        arrival = np.where(hits > 0, step_sum / np.maximum(hits, 1), np.nan).astype(np.float32)
    out["burn_prob"] = (("y", "x"), (hits / runs).astype(np.float32),
                        {"description": f"share of {runs} runs in which the cell burned within {steps} CA steps"})
    out["arrival_step"] = (("y", "x"), arrival, {"description": "mean CA step at which the cell ignited"})
    lit = hits > 0
    reached_edge = lit[0].any() or lit[-1].any() or lit[:, 0].any() or lit[:, -1].any()
    out.attrs.update({"sim_steps": steps, "sim_runs": runs, "sim_spotting": int(bool(spotting)),
                      "sim_front_cells": int(front.sum()),
                      "sim_expected_area_km2": float(hits.sum() / runs) * res * res / 1e6,
                      "sim_reached_grid_edge": int(reached_edge)})
    return out
