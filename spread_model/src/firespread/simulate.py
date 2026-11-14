"""Run the Alexandridis cellular automaton for several steps (Monte Carlo) -> burn probability per cell.

spread_probabilities() gives, for one step, P(burning cell -> each of its 8 neighbours) (`p_spread`). Here the
automaton is run `steps` times in `runs` independent random realisations, as in Alexandridis et al. (2008):

  - a burning cell tries to ignite each neighbour once, with probability p_spread[direction, cell];
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
                    seed: int | None = 0) -> xr.Dataset:
    """Add burn_prob and arrival_step (y, x) to a spread_probabilities() result. Deterministic for a fixed seed."""
    if steps < 1 or runs < 1:
        raise ValueError("steps and runs must be >= 1")
    p = np.nan_to_num(result["p_spread"].values.astype(np.float32))
    start = result["burning"].values.astype(bool)
    h, w = start.shape
    rng = np.random.default_rng(seed)

    burned = np.broadcast_to(start, (runs, h, w)).copy()
    front_r, front_y, front_x = np.nonzero(burned)            # burning cells (run, row, col)
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
    res = float(result.attrs.get("res_m", abs(float(result.x[1] - result.x[0]))))
    lit = hits > 0
    reached_edge = lit[0].any() or lit[-1].any() or lit[:, 0].any() or lit[:, -1].any()
    out.attrs.update({"sim_steps": steps, "sim_runs": runs,
                      "sim_expected_area_km2": float(hits.sum() / runs) * res * res / 1e6,
                      "sim_reached_grid_edge": int(reached_edge)})
    return out
