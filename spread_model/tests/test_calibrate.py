"""Offline tests of the calibration helpers (no Earth Engine)."""
import numpy as np

from firespread.calibrate import _target, first_burn_step, soft_iou
from firespread import spread_probabilities
from test_alexandridis import C, N, make_ds


def test_first_burn_step_rings_and_no_spread():
    start = np.zeros((N, N), bool)
    start[C, C] = True
    sure = np.ones((8, N, N), np.float32)
    first = first_burn_step(sure, start, max_steps=3, runs=2)
    yy, xx = np.mgrid[:N, :N]
    ring = np.maximum(abs(yy - C), abs(xx - C))               # Chebyshev distance = step with p = 1
    expected = np.where(ring <= 3, ring, 32767)
    assert (first == expected).all()
    none = first_burn_step(np.zeros((8, N, N), np.float32), start, max_steps=5, runs=3)
    assert (none[:, start] == 0).all() and (none[:, ~start] == 32767).all()


def test_soft_iou():
    t = np.zeros((4, 4), bool)
    t[:2] = True
    assert soft_iou(t.astype(np.float32), t) == 1.0
    assert soft_iou(np.zeros((4, 4), np.float32), t) == 0.0
    assert abs(soft_iou(np.full((4, 4), 0.5, np.float32), t) - 4 / 12) < 1e-6   # sum min = 8*0.5, sum max = 8 + 8*0.5


def test_target_rasterises_on_the_grid():
    from pyproj import Transformer
    from shapely.geometry import box
    res = spread_probabilities(make_ds())
    to_ll = Transformer.from_crs(res.attrs["crs"], 4326, always_xy=True)
    r = float(res.attrs["res_m"])
    x, y = float(res.x[C]), float(res.y[C])
    (w, e), (s, n) = to_ll.transform([x - 0.4 * r, x + 1.4 * r], [y - 0.4 * r, y + 0.4 * r])
    t = _target(res, box(w, s, e, n))                      # covers the fire cell and the one east of it
    assert t[C, C] and t[C, C + 1] and t.sum() == 2
