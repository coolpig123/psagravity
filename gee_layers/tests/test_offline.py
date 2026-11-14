"""Tests that need no Earth Engine account: grid maths, derived variables, tiling, regrid."""
import numpy as np
import pandas as pd
import pytest

from geelayers import fetch
from geelayers.api import _add_wind, _empty_dataset, regrid
from geelayers.derive import slope_aspect, wind_speed_dir
from geelayers.grid import Grid, utm_epsg
from geelayers.layers import LAYERS, VARIABLES, WEATHER

BUG = (39.727395, -120.0372941)


def test_utm_zone():
    assert utm_epsg(-120.04, 39.73) == 32610
    assert utm_epsg(-4.7, 40.6) == 32630          # Avila, Spain
    assert utm_epsg(151.2, -33.9) == 32756        # southern hemisphere


def test_grid_from_point_contains_point_and_snaps():
    g = Grid.from_point(*BUG, radius_km=10, res_m=100)
    assert g.crs == "EPSG:32610" and g.res == 100
    assert 200 <= g.width <= 201 and 200 <= g.height <= 201
    assert g.x0 % 100 == 0 and g.y0 % 100 == 0
    r, c = g.cell_of(*BUG)
    assert abs(r - g.height / 2) <= 1 and abs(c - g.width / 2) <= 1
    lat, lon = g.latlon()
    assert lat.shape == g.shape
    assert abs(lat[r, c] - BUG[0]) < 0.002 and abs(lon[r, c] - BUG[1]) < 0.002
    assert lat[0, 0] > lat[-1, 0]                   # row 0 is north


def test_grid_from_bbox_covers_box():
    g = Grid.from_bbox(-120.6, 39.2, -119.4, 40.3, res_m=500)
    for lat, lon in [(39.2, -120.6), (40.3, -119.4), (39.2, -119.4), (40.3, -120.6)]:
        r, c = g.cell_of(lat, lon)
        assert 0 <= r < g.height and 0 <= c < g.width


def test_with_resolution_nests():
    g = Grid.from_point(*BUG, radius_km=10, res_m=100)
    coarse = g.with_resolution(1000)
    xmin, ymin, xmax, ymax = g.bounds
    cxmin, cymin, cxmax, cymax = coarse.bounds
    assert cxmin <= xmin and cymin <= ymin and cxmax >= xmax and cymax >= ymax
    assert coarse.epsg == g.epsg


def test_tiles_cover_grid_exactly_once():
    g = Grid(32610, 0.0, 1000.0, 10.0, 1000, 730)
    seen = np.zeros(g.shape, int)
    for r0, c0, sub in g.tiles(256):
        assert sub.x0 == g.x0 + c0 * g.res and sub.y0 == g.y0 - r0 * g.res
        seen[r0:r0 + sub.height, c0:c0 + sub.width] += 1
    assert (seen == 1).all()


def test_ee_grid_spec():
    spec = Grid(32610, 700000.0, 4400000.0, 100.0, 3, 2).ee_grid()
    assert spec["dimensions"] == {"width": 3, "height": 2}
    assert spec["affineTransform"]["scaleY"] == -100.0 and spec["crsCode"] == "EPSG:32610"


def test_slope_aspect_plane():
    res = 100.0
    rows, cols = np.mgrid[0:20, 0:20]
    elev = cols * res * np.tan(np.radians(30))     # rises toward the east at 30 degrees
    slope, aspect = slope_aspect(elev, res)
    assert np.allclose(slope, 30, atol=1e-4)
    assert np.allclose(aspect, 270)                # faces west (downhill to the west)
    elev = rows * 10.0                             # rises toward the south (rows go south)
    _, aspect = slope_aspect(elev, res)
    assert np.allclose(aspect, 0)                  # faces north
    _, aspect = slope_aspect(np.zeros((5, 5)), res)
    assert np.isnan(aspect).all()


def test_wind_conventions():
    speed, d = wind_speed_dir(np.array([5.0, 0.0, -3.0, 0.0]), np.array([0.0, 5.0, 0.0, -4.0]))
    assert np.allclose(speed, [5, 5, 3, 4])
    assert np.allclose(d, [270, 180, 90, 0])       # from west, from south, from east, from north


def test_download_assembles_tiles_and_band_groups(monkeypatch):
    g = Grid(32610, 0.0, 5000.0, 10.0, 600, 500)
    bands = [f"b{i}" for i in range(5)]

    def fake_compute(image, names, sub):
        out = np.zeros((sub.height, sub.width), dtype=[(n, "f4") for n in names])
        for n in names:
            out[n] = int(n[1:]) * 1e6 + sub.y[:, None] * 0 + sub.x[None, :]
            out[n][0, 0] = fetch.NODATA if n == "b2" and sub.x0 == 0 and sub.y0 == 5000 else out[n][0, 0]
        return out

    monkeypatch.setattr(fetch, "_prepare", lambda image: image)
    monkeypatch.setattr(fetch, "_compute", fake_compute)
    arr = fetch.download(None, g, bands, max_bytes=2 * 4 * 256 * 256)   # forces 2 bands/request, 256-px tiles
    assert arr.shape == (5, 500, 600)
    expected = np.arange(5)[:, None, None] * 1e6 + g.x[None, None, :] + np.zeros((5, 500, 600))
    assert np.isnan(arr[2, 0, 0])
    arr[2, 0, 0] = expected[2, 0, 0]
    assert np.allclose(arr, expected)


def test_plan_stays_under_limit():
    for n in (1, 5, 40, 700):
        per, side = fetch.plan(n)
        assert per * side * side * fetch.BYTES_PER_PX <= fetch.MAX_BYTES or side == 256


def test_regrid_interpolates_and_recomputes_wind():
    fine = Grid.from_point(*BUG, radius_km=5, res_m=100)
    coarse = fine.with_resolution(1000)
    ds = _empty_dataset(coarse).assign_coords(time=pd.to_datetime(["2026-08-09T00:00"]))
    xx = np.broadcast_to(coarse.x[None, None, :], (1, *coarse.shape)).astype(np.float32)
    ds["temp_c"] = (("time", "y", "x"), (xx - coarse.x0) / 1000)  # linear in x: interpolation is exact
    ds["wind_u_ms"] = (("time", "y", "x"), np.full((1, *coarse.shape), 3.0, np.float32))
    ds["wind_v_ms"] = (("time", "y", "x"), np.full((1, *coarse.shape), 0.0, np.float32))
    ds = _add_wind(ds)
    out = regrid(ds, fine)
    assert out["temp_c"].shape == (1, *fine.shape)
    assert np.allclose(out["temp_c"][0].values, ((fine.x - coarse.x0) / 1000)[None, :], atol=1e-4)
    assert not np.isnan(out["temp_c"]).any()        # edges extrapolated, not NaN
    assert np.allclose(out["wind_dir_deg"], 270) and np.allclose(out["wind_speed_ms"], 3)


def test_every_variable_is_documented():
    names = {v for l in LAYERS.values() for v in l.variables} | {v for w in WEATHER.values() for v in w.variables}
    names |= {"slope_deg", "aspect_deg", "wind_speed_ms", "wind_dir_deg"}
    assert names <= set(VARIABLES)


def test_cli_list(capsys):
    from geelayers.__main__ import main
    main(["list"])
    out = capsys.readouterr().out
    assert "terrain" in out and "rtma" in out and "wind_dir_deg" in out


def test_init_without_project_explains(monkeypatch):
    from geelayers import auth
    monkeypatch.delenv("EE_PROJECT", raising=False)
    monkeypatch.setattr(auth, "_initialized", False)
    with pytest.raises(RuntimeError, match="EE_PROJECT"):
        auth.init()
