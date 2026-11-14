"""Scores of a predicted burned mask against an observed one, and the circular-growth baseline.

Scores are on the NEWLY burned area: (mask minus the start perimeter). Counting the start perimeter, which every
method gets for free, would inflate all scores. Totals including it are reported too (`*_total`).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def scores(pred: np.ndarray, obs: np.ndarray, start: np.ndarray, res_m: float) -> dict[str, float]:
    pred, obs, start = (np.asarray(a).astype(bool) for a in (pred, obs, start))
    out = {}
    for tag, p, o in (("", pred & ~start, obs & ~start), ("_total", pred | start, obs | start)):
        tp = float((p & o).sum())
        union = float((p | o).sum())
        out[f"iou{tag}"] = tp / union if union else float("nan")
        out[f"precision{tag}"] = tp / p.sum() if p.sum() else float("nan")
        out[f"recall{tag}"] = tp / o.sum() if o.sum() else float("nan")
    cell_km2 = res_m * res_m / 1e6
    out["pred_new_km2"] = float((pred & ~start).sum()) * cell_km2
    out["obs_new_km2"] = float((obs & ~start).sum()) * cell_km2
    return out


def circular_growth(start: np.ndarray, n_new_cells: int) -> np.ndarray:
    """Grow the start perimeter by the same distance in every direction until it adds `n_new_cells` cells.

    The dumb baseline: same new area as the model, no wind / slope / fuel. Returns start | grown ring.
    """
    start = np.asarray(start).astype(bool)
    if n_new_cells <= 0 or not start.any():
        return start.copy()
    dist = distance_transform_edt(~start)           # cells; 0 inside the start perimeter
    outside = dist[~start]
    n = min(int(n_new_cells), outside.size)
    radius = np.partition(outside, n - 1)[n - 1]     # n-th smallest distance
    return start | (dist <= radius)
