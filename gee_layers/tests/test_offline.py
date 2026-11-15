"""Tests that need no Earth Engine account: grid maths, derived variables, tiling, regrid."""
import numpy as np
import pandas as pd
import pytest

from geelayers import fetch
from geelayers.api import _add_wind, _empty_dataset, regrid
from geelayers.derive import decode_evc, slope_aspect, wind_speed_dir
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


def test_decode_evc():
    lifeform, cover = decode_evc(np.array([101, 109, 111, 125, 129, 11, 31, 100, -1]))
    assert lifeform.tolist() == [1, 1, 2, 3, 3, 0, 0, 0, -1]
    assert cover[:5].tolist() == [15, 95, 15, 55, 95] and np.isnan(cover[5:]).all()


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
    names |= {"slope_deg", "aspect_deg", "wind_speed_ms", "wind_dir_deg", "fuel_lifeform", "cover_pct",
              "initial_burned"}
    assert names <= set(VARIABLES)


def test_grid_from_polygon_and_masks(tmp_path):
    ring = [[-120.05, 39.72], [-120.02, 39.72], [-120.02, 39.74], [-120.05, 39.74], [-120.05, 39.72]]
    hole = [[-120.04, 39.725], [-120.03, 39.725], [-120.03, 39.735], [-120.04, 39.735], [-120.04, 39.725]]
    geojson = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring, hole]}, "properties": {}}
    ]}
    gj_file = tmp_path / "fire.geojson"
    import json
    gj_file.write_text(json.dumps(geojson), encoding="utf-8")

    g = Grid.from_polygon(gj_file, buffer_km=2, res_m=100)
    assert g.crs == "EPSG:32610"
    mask = g.polygon_mask(gj_file)
    assert mask.shape == g.shape and mask.dtype == np.int16
    assert mask.sum() > 20
    # Center of the hole is unburned (0), while inside outer ring away from hole is burned (1)
    r_hole, c_hole = g.cell_of(39.73, -120.035)
    r_ring, c_ring = g.cell_of(39.722, -120.045)
    assert mask[r_hole, c_hole] == 0
    assert mask[r_ring, c_ring] == 1

    # Sub-cell tiny polygon still ignites its centroid cell
    tiny = [[-120.0373, 39.7274], [-120.03729, 39.7274], [-120.03729, 39.72741], [-120.0373, 39.72741]]
    tmask = g.polygon_mask(tiny)
    assert tmask.sum() == 1
    rt, ct = g.cell_of(39.727405, -120.037295)
    assert tmask[rt, ct] == 1

    # Point mask (single cell and radius)
    pm1 = g.point_mask(*BUG)
    assert pm1.sum() == 1 and pm1[g.cell_of(*BUG)] == 1
    pm_rad = g.point_mask(*BUG, radius_m=250.0)
    assert pm_rad.sum() > 10


def test_get_fire_inputs_aligns_layers_weather_and_mask(monkeypatch, tmp_path):
    from geelayers import api

    def fake_get_layers(grid, time=None, layers=api.DEFAULT_LAYERS, weather_source="rtma"):
        ds = _empty_dataset(grid)
        ds["slope_deg"] = (("y", "x"), np.full(grid.shape, 12.0, np.float32), api._attrs("slope_deg"))
        return ds

    calls = {}

    def fake_get_weather(grid, start, end, source="rtma", step_hours=1, issued=None):
        calls.update(start=start, issued=issued)
        ds = _empty_dataset(grid).assign_coords(time=pd.to_datetime(["2026-08-08T22:00", "2026-08-08T23:00"]))
        ds["wind_u_ms"] = (("time", "y", "x"), np.full((2, *grid.shape), 4.0, np.float32), api._attrs("wind_u_ms"))
        ds["wind_v_ms"] = (("time", "y", "x"), np.zeros((2, *grid.shape), np.float32), api._attrs("wind_v_ms"))
        return _add_wind(ds)

    monkeypatch.setattr(api, "get_layers", fake_get_layers)
    monkeypatch.setattr(api, "get_weather", fake_get_weather)

    poly = [[-120.04, 39.72], [-120.03, 39.72], [-120.03, 39.73], [-120.04, 39.73]]
    ds = api.get_fire_inputs(polygon=poly, start="2026-08-08T22:30Z", hours=2, radius_km=3, res_m=100, weather_res_m=1000)
    assert set(ds.dims) == {"time", "y", "x"}
    assert ds["slope_deg"].shape == ds["initial_burned"].shape == (ds.sizes["y"], ds.sizes["x"])
    assert ds["wind_u_ms"].shape == (2, ds.sizes["y"], ds.sizes["x"])
    assert ds["initial_burned"].values.sum() > 0
    assert np.allclose(ds["wind_speed_ms"].values, 4.0)
    # weather starts at the top of the hour so it covers the start; GFS run chosen at start itself
    assert calls["start"] == pd.Timestamp("2026-08-08T22:00Z")
    assert calls["issued"] == pd.Timestamp("2026-08-08T22:30Z")

    # exactly one fire must be given
    with pytest.raises(ValueError, match="exactly one fire"):
        api.get_fire_inputs(polygon=poly, point=BUG, start="2026-08-08T22:30Z")
    with pytest.raises(ValueError, match="exactly one fire"):
        api.get_fire_inputs(bbox=(-120.1, 39.7, -120.0, 39.8), start="2026-08-08T22:30Z")
    # lat / lon swapped
    with pytest.raises(ValueError, match="swapped"):
        api.get_fire_inputs(point=BUG[::-1], start="2026-08-08T22:30Z")
    with pytest.raises(ValueError, match="swapped"):
        api.get_fire_inputs(polygon=[[39.72, -120.04], [39.72, -120.03], [39.73, -120.03]], start="2026-08-08T22:30Z")
    # valid fire, but outside the given area
    with pytest.raises(ValueError, match="outside the grid"):
        api.get_fire_inputs(point=(40.5, -121.0), bbox=(-120.1, 39.7, -120.0, 39.8), start="2026-08-08T22:30Z")

    # Also test CLI bundle command writes NetCDF
    from geelayers.__main__ import main
    out_nc = tmp_path / "bundle.nc"
    main(["bundle", "--point", str(BUG[0]), str(BUG[1]), "--radius-km", "2",
          "--start", "2026-08-08T22:30Z", "--hours", "2", "--out", str(out_nc)])
    assert out_nc.exists()


def _square(lon, lat, d):
    return [[lon, lat], [lon + d, lat], [lon + d, lat + d], [lon, lat + d], [lon, lat]]


def _feature(ring, **props):
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": props}


def test_perimeter_time_series_uses_perimeter_at_start():
    from geelayers.perimeter import resolve_perimeter
    # FEDS-like: growing perimeters, one per step, each with obs_time_utc
    fc = {"type": "FeatureCollection", "features": [
        _feature(_square(-120.04, 39.72, 0.01), step_utc="2026-08-09T00:00:00Z", obs_time_utc="2026-08-09T09:30:00Z"),
        _feature(_square(-120.05, 39.71, 0.03), step_utc="2026-08-09T12:00:00Z", obs_time_utc="2026-08-09T20:30:00Z"),
        _feature(_square(-120.10, 39.65, 0.20), step_utc="2026-08-10T00:00:00Z", obs_time_utc="2026-08-10T09:30:00Z"),
    ]}
    geom, meta = resolve_perimeter(fc, "2026-08-09T21:00Z")
    assert geom is fc["features"][1]
    assert meta["initial_perimeter_time_utc"] == "2026-08-09T20:30:00+00:00"
    assert "obs_time_utc" in meta["initial_perimeter"]
    with pytest.raises(ValueError, match="first perimeter"):
        resolve_perimeter(fc, "2026-08-09T05:00Z")
    geom, _ = resolve_perimeter(fc, "2026-08-09T21:00Z", time_key="step_utc")   # explicit key
    assert geom is fc["features"][1]

    # The mask follows the chosen perimeter, not the union (the union would be the final area)
    g = Grid.from_polygon(fc, buffer_km=1, res_m=100)
    assert g.polygon_mask(geom).sum() < g.polygon_mask(fc).sum()


def test_untimed_features_are_all_burning():
    from geelayers.perimeter import resolve_perimeter
    fc = {"type": "FeatureCollection", "features": [
        _feature(_square(-120.04, 39.72, 0.01), name="a"), _feature(_square(-120.00, 39.72, 0.01), name="b")]}
    geom, meta = resolve_perimeter(fc, "2026-08-09T21:00Z")
    assert geom is fc and "union of 2" in meta["initial_perimeter"]
    single = _feature(_square(-120.04, 39.72, 0.01))
    assert resolve_perimeter(single, "2026-08-09T21:00Z") == (single, {})


def test_area_guard_and_bbox_lonlat(monkeypatch):
    g = Grid.from_polygon(_square(-120.04, 39.72, 0.05), buffer_km=5, res_m=100)
    w, s, e, n = g.bbox_lonlat()
    assert w < -120.04 and s < 39.72 and e > -119.99 and n > 39.77      # polygon + buffer
    assert e - w < 0.3 and n - s < 0.3                                   # and not much more
    huge = Grid(32610, 0.0, 1e6, 10.0, 5000, 5000)
    monkeypatch.setattr(fetch, "_compute", lambda *a: pytest.fail("must not download"))
    with pytest.raises(ValueError, match="area too large"):
        fetch.download(None, huge, ["b"])


def test_cli_list(capsys):
    from geelayers.__main__ import main
    main(["list"])
    out = capsys.readouterr().out
    assert "terrain" in out and "rtma" in out and "wind_dir_deg" in out


def test_init_defaults_to_team_project_and_logs_in_once(monkeypatch):
    from geelayers import auth
    calls = []
    monkeypatch.delenv("EE_PROJECT", raising=False)
    monkeypatch.delenv("EE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setattr(auth, "_initialized", False)
    monkeypatch.setattr(auth, "_has_saved_login", lambda: False)
    monkeypatch.setattr(auth.ee, "Authenticate", lambda: calls.append("login"))
    monkeypatch.setattr(auth.ee, "Initialize", lambda cred, project, opt_url: calls.append((cred, project)))
    auth.init()
    assert calls == ["login", ("persistent", auth.DEFAULT_PROJECT)]

    def denied(*a, **k):
        raise Exception("Caller does not have required permission to use project x")
    monkeypatch.setattr(auth.ee, "Initialize", denied)
    monkeypatch.setattr(auth, "_has_saved_login", lambda: True)
    with pytest.raises(RuntimeError, match="IAM"):
        auth.init(force=True)
    monkeypatch.setattr(auth, "_initialized", False)


def test_in_conus_and_weather_candidates():
    from geelayers.api import _weather_candidates, in_conus
    g_us = Grid.from_point(*BUG, radius_km=10, res_m=100)
    g_spain = Grid.from_point(40.6, -4.7, radius_km=10, res_m=100)
    g_aus = Grid.from_point(-33.9, 151.2, radius_km=10, res_m=100)
    g_ak = Grid.from_point(64.8, -147.7, radius_km=10, res_m=100)
    assert in_conus(g_us)
    assert not in_conus(g_spain) and not in_conus(g_aus) and not in_conus(g_ak)

    now = pd.Timestamp("2026-11-14T12:00Z")
    # Historical US: tries rtma first, then era5_land, then gfs
    assert _weather_candidates(g_us, pd.Timestamp("2026-08-08T00:00Z"), pd.Timestamp("2026-08-09T00:00Z"),
                               "rtma", now=now) == ["rtma", "era5_land", "gfs"]
    # Historical non-US (Spain): skips rtma, tries era5_land then gfs
    assert _weather_candidates(g_spain, pd.Timestamp("2026-08-08T00:00Z"), pd.Timestamp("2026-08-09T00:00Z"),
                               "rtma", now=now) == ["era5_land", "gfs"]
    # Recent / live non-US: tries gfs first, then era5_land
    assert _weather_candidates(g_spain, pd.Timestamp("2026-11-13T00:00Z"), pd.Timestamp("2026-11-14T00:00Z"),
                               "auto", now=now) == ["gfs", "era5_land"]
    # Future forecast in US: rtma has no future hours, so gfs first
    assert _weather_candidates(g_us, pd.Timestamp("2026-11-14T12:00Z"), pd.Timestamp("2026-11-15T12:00Z"),
                               "rtma", now=now) == ["gfs", "era5_land"]
    # Explicit global source is respected
    assert _weather_candidates(g_us, pd.Timestamp("2026-08-08T00:00Z"), pd.Timestamp("2026-08-09T00:00Z"),
                               "era5_land", now=now) == ["era5_land"]


def test_global_layers_and_weather_fallback_outside_conus(monkeypatch):
    from geelayers import api, layers

    monkeypatch.setattr(api, "init", lambda: None)
    monkeypatch.setattr(fetch, "region", lambda grid: None)

    g_spain = Grid.from_point(40.6, -4.7, radius_km=2, res_m=500)

    # CONUS-only builders must not be called for Spain
    def fail_conus(region, t):
        pytest.fail("CONUS-only layer should be skipped outside CONUS")

    monkeypatch.setitem(layers.LAYERS, "vegetation_cover",
                        layers.Layer("vegetation_cover", "", "LANDFIRE", "CONUS", 30, False, True, ("evc",), fail_conus))
    monkeypatch.setitem(layers.LAYERS, "fuel_dryness",
                        layers.Layer("fuel_dryness", "", "GRIDMET", "CONUS", 4000, True, False, ("fm100_pct",), fail_conus))
    monkeypatch.setitem(layers.LAYERS, "terrain",
                        layers.Layer("terrain", "", "NASADEM", "global", 30, False, False,
                                     ("elevation_m",), lambda r, t: (_FakeImg(), {})))
    monkeypatch.setitem(layers.LAYERS, "landcover",
                        layers.Layer("landcover", "", "WorldCover", "global", 10, False, True,
                                     ("landcover",), lambda r, t: (_FakeImg(), {})))

    monkeypatch.setattr(api.ee.Image, "cat", staticmethod(lambda imgs: _FakeImg()))
    monkeypatch.setattr(fetch, "download",
                        lambda img, grid, names: np.stack([
                            np.full(grid.shape, 500.0 if n == "elevation_m" else 10.0, np.float32) for n in names
                        ]))

    ds = api.get_layers(g_spain, time="2026-08-08T22:30Z",
                        layers=("terrain", "landcover", "vegetation_cover", "fuel_dryness"))
    assert (ds["evc"].values == -1).all()
    assert (ds["fuel_lifeform"].values == -1).all()
    assert np.isnan(ds["cover_pct"].values).all()
    assert np.isnan(ds["fm100_pct"].values).all()
    assert (ds["landcover"].values == 10).all()
    assert "skipped: outside CONUS" in ds.attrs["dataset_vegetation_cover"]
    assert "skipped: outside CONUS" in ds.attrs["dataset_fuel_dryness"]


class _FakeImg:
    def select(self, bands):
        return self


def test_meteomatics_weather_parsing_and_license_fallback(monkeypatch):
    from geelayers import api, meteomatics

    g = Grid.from_point(*BUG, radius_km=2, res_m=500)
    w, s, e, n = g.bbox_lonlat()
    lats = [round(s, 4), round(n, 4)]
    lons = [round(w, 4), round(e, 4)]
    dates = ["2026-08-09T21:00:00Z", "2026-08-09T22:00:00Z"]

    def make_coords(val_t0, val_t1):
        return [
            {"lat": lat, "lon": lon, "dates": [{"date": dates[0], "value": val_t0}, {"date": dates[1], "value": val_t1}]}
            for lat in lats for lon in lons
        ]

    fake_json = {
        "data": [
            {"parameter": "wind_speed_u_10m:ms", "coordinates": make_coords(4.0, 0.0)},
            {"parameter": "wind_speed_v_10m:ms", "coordinates": make_coords(0.0, 6.0)},
            {"parameter": "t_2m:C", "coordinates": make_coords(31.0, 27.0)},
            {"parameter": "relative_humidity_2m:p", "coordinates": make_coords(18.0, 35.0)},
            {"parameter": "wind_gusts_10m_1h:ms", "coordinates": make_coords(7.5, 9.0)},
            {"parameter": "precip_1h:mm", "coordinates": make_coords(0.0, 1.2)},
        ]
    }

    calls = []

    class _FakeResp:
        def __init__(self, status_code, payload, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    def fake_get(url, auth=None, timeout=None):
        calls.append((url, auth))
        if "fosberg_fire_weather_index" in url:
            return _FakeResp(400, {}, "parameter fosberg_fire_weather_index:idx not available in Basic")
        return _FakeResp(200, fake_json)

    monkeypatch.setenv("METEOMATICS_USERNAME", "demo_user")
    monkeypatch.setenv("METEOMATICS_PASSWORD", "demo_pass")
    monkeypatch.setattr(meteomatics.requests, "get", fake_get)

    wx = api.get_weather(g, "2026-08-09T21:00Z", "2026-08-09T23:00Z", source="meteomatics")
    assert len(calls) == 2  # first tried extended, fell back to standard on HTTP 400
    assert calls[0][1] == ("demo_user", "demo_pass")
    assert wx.attrs["weather_source"] == "meteomatics"
    assert set(wx.dims) == {"time", "y", "x"}
    assert wx.sizes["time"] == 2 and wx.sizes["y"] == g.height and wx.sizes["x"] == g.width
    assert np.allclose(wx["wind_speed_ms"].isel(time=0).values, 4.0, atol=1e-4)
    assert np.allclose(wx["wind_dir_deg"].isel(time=0).values, 270.0, atol=1e-3)
    assert np.allclose(wx["wind_speed_ms"].isel(time=1).values, 6.0, atol=1e-4)
    assert np.allclose(wx["wind_dir_deg"].isel(time=1).values, 180.0, atol=1e-3)
    assert np.allclose(wx["temp_c"].isel(time=0).values, 31.0, atol=1e-4)
    assert np.allclose(wx["precip_mm"].isel(time=1).values, 1.2, atol=1e-4)


