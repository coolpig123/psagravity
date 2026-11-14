"""Offline tests: windows from the real FEDS files, metrics, baseline."""
import numpy as np

from hindcast.metrics import circular_growth, scores
from hindcast.windows import build_windows, describe


def test_bug_windows_skip_stale_steps():
    table = describe(build_windows("bug_fire"))
    first = table[table.window == "bug_fire_0809_0930"]
    assert list(first.status) == ["scored", "scored"]
    assert list(first.target_km2) == [10.8, 37.0]
    # 08-10 20:30 repeats 08-10 09:30 (FEDS is_new = False): never a start, stale as a target
    assert "bug_fire_0810_2030" not in set(table.window)
    assert table[(table.window == "bug_fire_0810_0930") & (table.horizon_h == 12)].status.item() == "stale"


def test_scores_count_only_new_area():
    start = np.zeros((10, 10), bool); start[4:6, 4:6] = True
    obs = start.copy(); obs[4:6, 6:8] = True                  # grew 4 cells east
    s = scores(obs, obs, start, 100.0)
    assert s["iou"] == s["precision"] == s["recall"] == 1.0
    s = scores(start, obs, start, 100.0)                      # predicted no growth
    assert np.isnan(s["precision"]) and s["recall"] == 0 and s["iou_total"] == 0.5


def test_circular_growth_same_area_and_symmetric():
    start = np.zeros((41, 41), bool); start[20, 20] = True
    grown = circular_growth(start, 100)
    assert abs((grown & ~start).sum() - 100) <= 8              # ties at the same distance are kept together
    assert (grown == grown[::-1]).all() and (grown == grown[:, ::-1]).all()
    assert circular_growth(start, 0).sum() == 1


def test_evaluate_window_end_to_end_synthetic():
    """Wind blowing east, observed growth to the east: the model must beat the same-area circle."""
    import pandas as pd
    import xarray as xr
    from firespread import predict
    from pyproj import Transformer

    from hindcast.run import best_params, evaluate_window
    from hindcast.windows import Target, Window

    n, res, x0, y0 = 41, 100.0, 700_000.0, 4_400_000.0       # UTM 10N, near the Bug Fire
    burning = np.zeros((n, n), np.int16); burning[19:22, 19:22] = 1
    shape, t0 = (n, n), pd.Timestamp("2026-08-09T09:30Z")
    ds = xr.Dataset({
        "elevation_m": (("y", "x"), np.zeros(shape, np.float32)),
        "fuel_lifeform": (("y", "x"), np.full(shape, 2, np.int16)),
        "landcover": (("y", "x"), np.full(shape, 20, np.int16)),
        "wind_u_ms": (("time", "y", "x"), np.full((1, *shape), 8.0, np.float32)),
        "wind_v_ms": (("time", "y", "x"), np.zeros((1, *shape), np.float32)),
        "initial_burned": (("y", "x"), burning),
    }, coords={"y": y0 - res * (np.arange(n) + 0.5), "x": x0 + res * (np.arange(n) + 0.5),
               "time": np.array([t0.tz_localize(None)], dtype="datetime64[ns]")},
        attrs={"res_m": res, "crs": "EPSG:32610"})
    to_ll = Transformer.from_crs(32610, 4326, always_xy=True)
    def box(c0, c1, r0, r1):                                   # cell columns / rows -> lon/lat polygon
        xs = [x0 + c0 * res, x0 + c1 * res, x0 + c1 * res, x0 + c0 * res, x0 + c0 * res]
        ys = [y0 - r0 * res, y0 - r0 * res, y0 - r1 * res, y0 - r1 * res, y0 - r0 * res]
        return {"type": "Feature", "geometry": {"type": "Polygon",
                "coordinates": [[list(p) for p in zip(*to_ll.transform(xs, ys))]]}, "properties": {}}
    w = Window("synthetic", t0, box(19, 22, 19, 22), 0.09,
               [Target(12, t0 + pd.Timedelta(hours=12), box(19, 30, 18, 23), 0.55, stale=False)])
    result = predict(inputs=ds, time=t0, burn_scars=False)
    table = evaluate_window(w, result, hours_per_step=(1.0, 2.0), thresholds=(0.3, 0.5), runs=50)
    assert len(table) == 4
    best = best_params(table)
    row = table[(table.hours_per_step == best.hours_per_step) & (table.threshold == best.threshold)].iloc[0]
    assert row.iou > row.base_iou
