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


def test_first_burn_step_spotting_jumps_downwind():
    start = np.zeros((N, N), bool)
    start[C, C] = True
    never = np.zeros((8, N, N), np.float32)                # no neighbour spread: only firebrands
    ones = np.ones((N, N), np.float32)
    spot = {"launch": ones, "ignite": ones, "dist_m": ones * 300.0, "dir_deg": ones * 90.0,   # toward the east
            "min_m": 300.0, "max_m": 300.0, "cone_deg": 0.0, "res_m": 100.0}
    first = first_burn_step(never, start, max_steps=1, runs=4, spot=spot)
    assert (first[:, C, C + 3] == 1).all()                 # 300 m east, in every run
    assert ((first == 1).sum(axis=(1, 2)) == 1).all()      # and nowhere else
    spot["dir_deg"] = ones * 0.0                           # toward the north = up the rows
    assert (first_burn_step(never, start, 1, runs=2, spot=spot)[:, C - 3, C] == 1).all()


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


def test_calibrated_preset_and_search_params():
    from firespread import AlexandridisParams
    from firespread.calibrate_search import CALIBRATED, params_of
    cal = AlexandridisParams.calibrated()
    p = params_of(CALIBRATED)
    assert cal.p0 == p.p0 and cal.c1 == p.c1 and cal.c2 == p.c2
    assert cal.veg_by_lifeform[2] > 0 and cal.veg_by_lifeform[3] > 0
    res = spread_probabilities(make_ds(u=7.0), params=cal)
    assert float(res["p_spot_launch"].max()) > 0.0
