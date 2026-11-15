"""Run the Alexandridis cellular automaton for several steps (Monte Carlo) -> burn probability per cell.

spread_probabilities() gives, for one step, P(burning cell -> each of its 8 neighbours) (`p_spread`), and when
forecast weather time steps are available in the input dataset, `p_spread_time` `(time, direction, y, x)`.
Here the automaton is run `steps` times in `runs` independent random realisations, as in Alexandridis et al. (2008):

  - a burning cell tries to ignite each neighbour once, with probability p_spread(t)[direction, cell];
  - a cell burns for one step, then is burned out (it cannot spread again or be re-ignited);
  - newly ignited cells are the burning cells of the next step.

When `p_spread_time` is present (e.g. from Meteomatics or GFS multi-hour weather forecasts), the spread
probabilities evolve over time across the simulation steps (`time_developing=True`), reflecting changing wind
speed, wind direction, temperature, humidity, and rainfall hour by hour.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xarray as xr

from .alexandridis import DIRECTIONS

DEFAULT_STEPS = 30
DEFAULT_RUNS = 200


def _step_time_weights(
    step: int,
    steps: int,
    n_t: int,
    times: np.ndarray | None,
    step_hours: float | None,
) -> tuple[int, int, float]:
    """Return (i0, i1, alpha) for linear interpolation in `p_spread_time` at 1-based CA `step`."""
    if n_t <= 1:
        return 0, 0, 0.0
    if step_hours is not None and times is not None and len(times) == n_t:
        t0 = pd.Timestamp(times[0])
        t_cur = t0 + pd.Timedelta(hours=(step - 1) * float(step_hours))
        ts = pd.to_datetime(times)
        if t_cur <= ts[0]:
            return 0, 0, 0.0
        if t_cur >= ts[-1]:
            return n_t - 1, n_t - 1, 0.0
        idx = int(np.searchsorted(ts.values, t_cur.to_datetime64(), side="right") - 1)
        idx = max(0, min(n_t - 2, idx))
        dt_total = (ts[idx + 1] - ts[idx]).total_seconds()
        alpha = float((t_cur - ts[idx]).total_seconds() / dt_total) if dt_total > 0 else 0.0
        return idx, idx + 1, max(0.0, min(1.0, alpha))

    pos = ((step - 1) / max(steps - 1, 1)) * (n_t - 1)
    i0 = int(math.floor(pos))
    i1 = min(n_t - 1, i0 + 1)
    alpha = float(pos - i0)
    return i0, i1, alpha


def simulate_spread(
    result: xr.Dataset,
    steps: int = DEFAULT_STEPS,
    runs: int = DEFAULT_RUNS,
    seed: int | None = 0,
    *,
    time_developing: bool = True,
    step_hours: float | None = None,
) -> xr.Dataset:
    """Add burn_prob and arrival_step (y, x) to a spread_probabilities() result. Deterministic for a fixed seed.

    When `result` contains `p_spread_time` (from multi-hour weather forecasts such as Meteomatics) and
    `time_developing=True`, each CA step interpolates the time-evolving directional spread probabilities
    at that step's forecast time instead of holding weather fixed at t=0.
    """
    if steps < 1 or runs < 1:
        raise ValueError("steps and runs must be >= 1")
    p = np.nan_to_num(result["p_spread"].values.astype(np.float32))
    has_time_series = (
        bool(time_developing)
        and "p_spread_time" in result
        and result["p_spread_time"].sizes.get("time", 1) > 1
    )
    p_time = (
        np.nan_to_num(result["p_spread_time"].values.astype(np.float32))
        if has_time_series
        else None
    )
    # If the caller manually mutated result["p_spread"] (e.g. setting p_spread = 1 in a test),
    # respect result["p_spread"] rather than overriding it with the unmutated p_spread_time.
    use_time = bool(p_time is not None and np.allclose(p, p_time[0], atol=1e-5))
    n_t = int(p_time.shape[0]) if p_time is not None else 1
    times = result.coords["time"].values if "time" in result.coords else None

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
        if use_time and p_time is not None:
            i0, i1, alpha = _step_time_weights(step, steps, n_t, times, step_hours)
        else:
            i0 = i1 = 0
            alpha = 0.0

        new_r, new_y, new_x = [], [], []
        for k, (_, dr, dc, _) in enumerate(DIRECTIONS):
            ty, tx = front_y + dr, front_x + dc
            inside = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
            r, sy, sx, ty, tx = front_r[inside], front_y[inside], front_x[inside], ty[inside], tx[inside]
            if use_time and p_time is not None:
                if i0 == i1 or alpha <= 0.0:
                    prob = p_time[i0, k, sy, sx]
                else:
                    prob = (1.0 - alpha) * p_time[i0, k, sy, sx] + alpha * p_time[i1, k, sy, sx]
            else:
                prob = p[k, sy, sx]
            go = rng.random(r.size, dtype=np.float32) < prob
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

    forecast_hours = 0.0
    if times is not None and len(times) > 1:
        forecast_hours = float((pd.Timestamp(times[-1]) - pd.Timestamp(times[0])).total_seconds() / 3600.0)
    if step_hours is not None:
        hrs_per_step = float(step_hours)
    elif forecast_hours > 0 and steps > 0:
        hrs_per_step = forecast_hours / float(steps)
    else:
        hrs_per_step = 0.0
    if hrs_per_step > 0:
        out["arrival_hours"] = (
            ("y", "x"),
            (arrival * hrs_per_step).astype(np.float32),
            {"units": "hours", "description": "mean time from start (hours) at which the cell ignited"},
        )

    res = float(result.attrs.get("res_m", abs(float(result.x[1] - result.x[0]))))
    lit = hits > 0
    reached_edge = lit[0].any() or lit[-1].any() or lit[:, 0].any() or lit[:, -1].any()
    out.attrs.update({
        "sim_steps": steps,
        "sim_runs": runs,
        "sim_expected_area_km2": float(hits.sum() / runs) * res * res / 1e6,
        "sim_reached_grid_edge": int(reached_edge),
        "sim_time_developing": int(use_time),
        "sim_weather_steps": int(n_t if use_time else 1),
        "sim_forecast_hours": float(forecast_hours),
    })
    return out
