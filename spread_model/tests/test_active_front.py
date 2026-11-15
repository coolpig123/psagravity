"""Offline tests of the VIIRS active front (active_front.py): FIRMS is mocked, nothing is fetched."""
import numpy as np
import pandas as pd
from pyproj import Transformer

from firespread import AlexandridisParams, spread_probabilities
from firespread.active_front import (attach_active_front, fetch_active_front, front_from_points, outer_rim,
                                     project_front)
from test_alexandridis import C, N, make_ds


def square(half=2):
    sq = np.zeros((N, N), np.int16)
    sq[C - half:C + half + 1, C - half:C + half + 1] = 1
    return sq


def lat_lon(ds, rows, cols):
    lon, lat = Transformer.from_crs(ds.attrs["crs"], 4326, always_xy=True).transform(
        ds.x.values[np.asarray(cols)], ds.y.values[np.asarray(rows)])
    return np.atleast_1d(lat), np.atleast_1d(lon)


def test_outer_rim_and_projection():
    sq = square().astype(bool)
    rim = outer_rim(sq)
    assert rim.sum() == 16 and not rim[C, C] and rim[C - 2, C]
    detected = np.zeros((N, N), bool)
    detected[C, C + 1] = True                               # a hotspot one cell inside the east edge
    front = project_front(sq, detected)
    assert front[C, C + 1] and front[C, C + 2]               # ... activates the rim cell next to it
    assert front.sum() == 2


def test_front_from_points_frp_and_far_points():
    ds = make_ds(burning=square())
    lat, lon = lat_lon(ds, [C], [C + 1])
    front, frp = front_from_points(ds, lat, lon, [80.0])
    assert front[C, C + 2] and frp[C, C + 2] == 80.0 and frp[C, C + 1] == 80.0 and frp[~front].sum() == 0
    lat, lon = lat_lon(ds, [0], [0])                         # far from the fire: another fire, ignored
    assert front_from_points(ds, lat, lon, [10.0]) is None


def test_fetch_active_front_never_raises():
    ds = make_ds(burning=square())
    notes = []

    def no_key(bbox, start, lookback_h):
        raise RuntimeError("FIRMS_MAP_KEY is not set")

    def not_installed(bbox, start, lookback_h):
        raise ImportError("No module named 'firedata'")

    assert fetch_active_front(ds, "2026-08-09T21:00Z", notes=notes, points_fn=no_key) is None
    assert fetch_active_front(ds, "2026-08-09T21:00Z", notes=notes, points_fn=not_installed) is None
    empty = lambda bbox, start, lookback_h: pd.DataFrame(columns=["lat", "lon", "frp_mw"])  # noqa: E731
    assert fetch_active_front(ds, "2026-08-09T21:00Z", notes=notes, points_fn=empty) is None
    assert "FIRMS_MAP_KEY" in notes[0] and "real_fires is not installed" in notes[1] and "no VIIRS" in notes[2]
    out = attach_active_front(ds, "2026-08-09T21:00Z", points_fn=no_key)
    assert "active_front" not in out and "FIRMS_MAP_KEY" in out.attrs["active_front_note"]
    assert spread_probabilities(out).attrs["active_front_source"] == "whole burned area"


def test_attach_active_front_and_frp_spotting_boost():
    ds = make_ds(burning=square(), u=9.0)
    lat, lon = lat_lon(ds, [C - 1, C, C + 1], [C + 2] * 3)
    seen = {}

    def points(bbox, start, lookback_h):
        seen.update(bbox=bbox, start=start, lookback_h=lookback_h)
        return pd.DataFrame({"lat": lat, "lon": lon, "frp_mw": [5.0, 200.0, 5.0]})

    out = attach_active_front(ds, "2026-08-09T21:00Z", lookback_h=12, points_fn=points)
    w, s, e, n = seen["bbox"]
    assert w < lon.min() < e and s < lat.min() < n and seen["lookback_h"] == 12
    assert int(out["active_front"].sum()) == 5 and "5 of the polygon's 25 cells" in out.attrs["active_front_note"]
    res = spread_probabilities(out)
    assert "frp_mw" in res and res.attrs["active_front_cells"] == 5   # + the rim cells next to the end hotspots
    launch = res["p_spot_launch"].values
    plain = spread_probabilities(ds)["p_spot_launch"].values
    assert launch[C, C + 2] > launch[C - 1, C + 2] > plain[C - 1, C + 2] > 0      # hotter cell lofts more
    assert np.allclose(launch[0], plain[0])                                         # no FRP -> unchanged
    off = spread_probabilities(out, params=AlexandridisParams(spot_frp_boost=0.0))["p_spot_launch"].values
    assert np.allclose(off, plain)
