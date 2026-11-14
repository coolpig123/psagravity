"""Offline tests: each factor against its formula, and whole-grid cases with known answers."""
import math

import numpy as np
import pytest
import xarray as xr

from firespread import (AlexandridisParams, NisarRequest, nisar_template, p_den, p_dry, p_emc, p_mask,
                        p_moisture, p_slope, p_veg, p_wind, predict, simard_emc, slope_angle_deg,
                        spread_probabilities, summary)
from firespread.alexandridis import DIR_NAMES, shift

P = AlexandridisParams()
N = 11                   # grid size; the fire is the centre cell
C = N // 2


def make_ds(n=N, res=100.0, elev=None, u=0.0, v=0.0, lifeform=2, cover=45.0, landcover=20, burning=None,
            nisar_cover=None, nisar_sm=None, temp=None, rh=None):
    """Synthetic geelayers-like Dataset: shrub, flat, calm, centre cell burning, no NISAR unless given.
    `cover` is LANDFIRE cover_pct, which the model must ignore (density comes from NISAR only).
    temp / rh (hourly weather) only added when given, so P_emc is neutral by default."""
    shape = (n, n)
    if burning is None:
        burning = np.zeros(shape, np.int16)
        burning[n // 2, n // 2] = 1
    full = lambda val, dt=np.float32: np.full(shape, val, dt)  # noqa: E731
    ds = xr.Dataset({
        "elevation_m": (("y", "x"), full(0.0) if elev is None else elev.astype(np.float32)),
        "fuel_lifeform": (("y", "x"), full(lifeform, np.int16)),
        "cover_pct": (("y", "x"), full(cover)),
        "landcover": (("y", "x"), full(landcover, np.int16)),
        "wind_u_ms": (("time", "y", "x"), np.full((1, *shape), u, np.float32)),
        "wind_v_ms": (("time", "y", "x"), np.full((1, *shape), v, np.float32)),
        "initial_burned": (("y", "x"), burning),
    }, coords={"y": 1000 - res * (np.arange(n) + 0.5), "x": res * (np.arange(n) + 0.5),
               "time": np.array(["2026-08-09T21:00"], dtype="datetime64[ns]")},
        attrs={"res_m": res, "crs": "EPSG:32610"})
    if nisar_cover is not None:
        ds["nisar_cover_pct"] = (("y", "x"), full(nisar_cover))
    if nisar_sm is not None:
        ds["nisar_soil_moisture"] = (("y", "x"), full(nisar_sm))
    if temp is not None:
        ds["temp_c"] = (("time", "y", "x"), np.full((1, *shape), temp, np.float32))
    if rh is not None:
        ds["rh_pct"] = (("time", "y", "x"), np.full((1, *shape), rh, np.float32))
    return ds


def nisar_on(ds, **variables):
    """NISAR Dataset on the same grid as ds (what the NISAR module should hand over)."""
    return xr.Dataset({k: (("y", "x"), v) for k, v in variables.items()},
                      coords={"y": ds.y, "x": ds.x}, attrs={"crs": ds.attrs["crs"]})


# ---- factors ---------------------------------------------------------------------------------

def test_shift():
    a = np.arange(9.0).reshape(3, 3)
    assert shift(a, 0, 1)[0].tolist() == [1, 2, 0]        # neighbour to the east
    assert shift(a, -1, 0)[:, 0].tolist() == [0, 0, 3]     # neighbour to the north
    assert shift(a, 1, 1, np.nan)[2, 2] != shift(a, 1, 1, np.nan)[2, 2]   # outside -> fill (NaN)


def test_p_wind_formula():
    V = 8.0
    # wind blowing toward the east (u > 0): theta = 0 for spread east, 180 for west, 90 for north
    assert p_wind(np.array(V), np.array(0.0), 90.0, P) == pytest.approx(math.exp(P.c1 * V))
    assert p_wind(np.array(V), np.array(0.0), 270.0, P) == pytest.approx(math.exp(P.c1 * V) * math.exp(-2 * P.c2 * V))
    assert p_wind(np.array(V), np.array(0.0), 0.0, P) == pytest.approx(math.exp(P.c1 * V) * math.exp(-P.c2 * V))
    assert p_wind(np.array(0.0), np.array(0.0), 123.0, P) == pytest.approx(1.0)


def test_p_slope_formula():
    assert slope_angle_deg(np.array(0.0), np.array(100.0), 100.0) == pytest.approx(45.0)
    assert p_slope(np.array(45.0), P) == pytest.approx(math.exp(P.a * 45))
    assert p_slope(np.array(0.0), P) == 1.0
    assert p_slope(np.array(-10.0), P) < 1.0                # downhill slows spread


def test_p_veg_and_fallback():
    ds = make_ds()
    lf = np.array([[1, 2, 3, 0, -1, -1]], np.int16)
    lc = np.array([[10, 10, 10, 10, 30, 80]], np.int16)
    small = xr.Dataset({"fuel_lifeform": (("y", "x"), lf), "landcover": (("y", "x"), lc)})
    veg, _ = p_veg(small, P)
    assert veg.tolist()[0] == pytest.approx([0.4, 0.0, -0.3, 0.0, -0.3, 0.0])   # last: water, removed by P_mask
    assert ds is not None


def test_p_den_from_nisar_only():
    # Earth Engine cover / NDVI present but no NISAR: neutral, GEE ignored
    ds = xr.Dataset({"cover_pct": (("y", "x"), np.array([[10, 90, 50, 70, 5]], np.float32)),
                     "ndvi": (("y", "x"), np.array([[0.9] * 5], np.float32))})
    den, src = p_den(ds, P)
    assert (den == 0).all() and "not provided" in src
    ds["nisar_cover_pct"] = (("y", "x"), np.array([[10, 30, 59, 60, np.nan]], np.float32))
    den, src = p_den(ds, P)
    assert den.tolist()[0] == pytest.approx([-0.4, 0.0, 0.0, 0.3, 0.0])          # NaN (no NISAR data) -> 0
    assert src.startswith("nisar_cover_pct") and "1 cells without NISAR data" in src


def test_p_dry_from_nisar_only():
    ds = xr.Dataset({"fm100_pct": (("y", "x"), np.array([[3.0] * 5], np.float32))})   # gridMET: ignored
    dry, src = p_dry(ds, P)
    assert (dry == 0).all() and "not provided" in src
    ds["nisar_soil_moisture"] = (("y", "x"), np.array([[0.05, 0.10, 0.2, 0.25, np.nan]], np.float32))
    dry, _ = p_dry(ds, P)
    assert dry.tolist()[0] == pytest.approx([0.3, 0.0, 0.0, -0.3, 0.0])


def test_simard_emc_three_regimes():
    t_c = 30.0
    tf = t_c * 1.8 + 32                                       # 86 F
    for h, expected in [(5, 0.03229 + 0.281073 * 5 - 0.000578 * 5 * tf),
                        (30, 2.22749 + 0.160107 * 30 - 0.014784 * tf),
                        (70, 21.0606 + 0.005565 * 70 ** 2 - 0.00035 * 70 * tf - 0.483199 * 70)]:
        assert simard_emc(t_c, h) == pytest.approx(expected)
    # regime boundaries: 10 and 50 belong to the upper piece
    assert simard_emc(t_c, 10) == pytest.approx(2.22749 + 0.160107 * 10 - 0.014784 * tf)
    assert simard_emc(t_c, 50) == pytest.approx(21.0606 + 0.005565 * 2500 - 0.00035 * 50 * tf - 0.483199 * 50)
    # wetter air -> higher EMC; RH clipped to [0, 100]; never negative; NaN passes through
    emc = simard_emc(np.full(5, 25.0), np.array([5, 20, 40, 60, 95]))
    assert (np.diff(emc) > 0).all()
    assert simard_emc(t_c, 120) == pytest.approx(simard_emc(t_c, 100))
    assert simard_emc(45.0, 0) >= 0
    assert np.isnan(simard_emc(np.nan, 30))


def test_p_emc_direction_and_neutral_defaults():
    hot_dry = make_ds(temp=35, rh=12)
    cool_wet = make_ds(temp=12, rh=85)
    f_hot, emc_hot, _ = p_emc(hot_dry.isel(time=0), P)
    f_cool, emc_cool, _ = p_emc(cool_wet.isel(time=0), P)
    assert emc_hot[0, 0] < P.emc_ref_pct < emc_cool[0, 0]
    assert f_hot[0, 0] > 1 > f_cool[0, 0]
    assert f_hot[0, 0] == pytest.approx(math.exp(-P.b_emc * (float(emc_hot[0, 0]) - P.emc_ref_pct)), rel=1e-5)
    # no weather -> neutral; NaN cells -> neutral
    f, emc, src = p_emc(make_ds().isel(time=0), P)
    assert (f == 1).all() and np.isnan(emc).all() and "not in inputs" in src
    ds = hot_dry.isel(time=0)
    ds["rh_pct"].values[0, 0] = np.nan
    f, _, src = p_emc(ds, P)
    assert f[0, 0] == 1 and f[1, 1] > 1 and "1 cells without weather" in src


def test_hot_dry_air_spreads_more_than_cool_humid():
    hot = spread_probabilities(make_ds(temp=35, rh=12))
    cool = spread_probabilities(make_ds(temp=12, rh=85))
    assert float(hot["p_m"][0, 0]) > 1 > float(cool["p_m"][0, 0])
    assert float(hot["ignition_prob"].sum()) > float(cool["ignition_prob"].sum())
    assert float(hot["ignition_prob"][C - 1, C]) > float(cool["ignition_prob"][C - 1, C])


def test_p_m_combines_emc_and_nisar_soil_moisture():
    base = make_ds(temp=30, rh=25)
    f_emc, emc, _ = p_emc(base.isel(time=0), P)
    for sm, p_dry_expected in [(0.05, 0.3), (0.18, 0.0), (0.30, -0.3)]:
        ds = make_ds(temp=30, rh=25, nisar_sm=sm)
        p_m, parts, src = p_moisture(ds.isel(time=0), P)
        assert np.allclose(parts["p_dry"], p_dry_expected)
        assert np.allclose(p_m, f_emc * (1 + p_dry_expected))
        assert "Simard" in src and "nisar_soil_moisture" in src
    # same through predict(): NISAR passed separately, both parts in the output
    ds = make_ds(temp=30, rh=25)
    res = predict(inputs=ds, nisar=nisar_on(ds, nisar_soil_moisture=np.full((N, N), 0.05, np.float32)))
    assert float(res["p_m"][0, 0]) == pytest.approx(float(f_emc[0, 0]) * 1.3, rel=1e-5)
    assert float(res["p_emc"][0, 0]) == pytest.approx(float(f_emc[0, 0]), rel=1e-5)
    assert float(res["emc_pct"][0, 0]) == pytest.approx(float(emc[0, 0]), rel=1e-5)
    assert "Moisture P_m" in summary(res)


def test_p_vwc_and_three_part_p_m():
    from firespread import p_vwc
    ds = make_ds()
    v, src = p_vwc(ds, P)
    assert (v == 0).all() and "not provided" in src
    ds["nisar_vwc_kg_m2"] = (("y", "x"), np.full((N, N), 0.2, np.float32))
    ds["nisar_vwc_kg_m2"].values[0, :4] = [0.5, 3.9, 4.0, np.nan]
    v, _ = p_vwc(ds, P)
    assert v[1, 1] == pytest.approx(0.2) and v[0, :4].tolist() == pytest.approx([0.0, 0.0, -0.2, 0.0])
    # P_m = P_emc (1 + P_dry)(1 + P_vwc), all from one call
    ds = make_ds(temp=30, rh=25, nisar_sm=0.05)
    ds["nisar_vwc_kg_m2"] = (("y", "x"), np.full((N, N), 6.0, np.float32))
    p_m, parts, src = p_moisture(ds.isel(time=0), P)
    assert np.allclose(p_m, parts["p_emc"] * 1.3 * 0.8)
    assert "nisar_vwc_kg_m2" in src
    res = spread_probabilities(ds)
    assert float(res["p_vwc"][0, 0]) == pytest.approx(-0.2)


def test_p_mask():
    burning = np.array([[1, 0, 0, 0, 0]], bool)
    ds = xr.Dataset({"fuel_lifeform": (("y", "x"), np.array([[2, 2, 0, 2, 2]], np.int16)),
                     "landcover": (("y", "x"), np.array([[20, 20, 20, 80, 20]], np.int16)),
                     "nisar_burned": (("y", "x"), np.array([[0, 0, 0, 0, 1]], np.int16))})
    mask, _ = p_mask(ds, burning)
    assert mask.tolist()[0] == [0, 1, 0, 0, 0]               # burning, ok, no fuel, water, NISAR burn scar


# ---- whole grid ------------------------------------------------------------------------------

def neighbours(res, name):
    k = DIR_NAMES.index(name)
    return float(res["p_spread"][k, C, C])


def test_flat_calm_reference_fuel_is_p0_with_diagonal_correction():
    res = spread_probabilities(make_ds())
    for d in ("N", "E", "S", "W"):
        assert neighbours(res, d) == pytest.approx(P.p0)
    for d in ("NE", "SE", "SW", "NW"):
        assert neighbours(res, d) == pytest.approx(P.p0 / math.sqrt(2))
        # expected advance per step is the same: P * distance
        assert neighbours(res, d) * math.sqrt(2) == pytest.approx(neighbours(res, "N"))
    ign = res["ignition_prob"].values
    assert ign.sum() == pytest.approx(4 * P.p0 + 4 * P.p0 / math.sqrt(2))   # only the 8 neighbours
    assert ign[C, C] == 0                                    # the burning cell itself
    assert res.attrs["direction_strength"] == pytest.approx(0, abs=1e-6)
    assert "not provided" in res.attrs["source_p_den"]
    assert "nisar_soil_moisture not provided" in res.attrs["source_p_m"]
    assert "temp_c / rh_pct not in inputs" in res.attrs["source_p_m"]
    # diagonal_factor = 1 reproduces the published model
    res = spread_probabilities(make_ds(), params=AlexandridisParams(diagonal_factor=1.0))
    assert neighbours(res, "NE") == pytest.approx(P.p0)


def test_full_product_one_neighbour():
    """Grass, sparse (NISAR), dry (NISAR), wind toward the east 2 m/s, 2 m rise to the east over 100 m.
    LANDFIRE cover_pct says dense (90 %): must be ignored."""
    elev = np.zeros((N, N))
    elev[:, C + 1:] = 2.0
    res = spread_probabilities(make_ds(lifeform=3, cover=90, nisar_cover=10, nisar_sm=0.05, u=2.0, elev=elev))
    expected = (P.p0 * (1 - 0.3) * (1 - 0.4) * (1 + 0.3) * math.exp(P.c1 * 2)
                * math.exp(P.a * math.degrees(math.atan(2 / 100))))
    assert expected < 1   # so no clipping hides an error
    assert neighbours(res, "E") == pytest.approx(expected, rel=1e-5)
    assert float(res["p_dry"][C, C + 1]) == pytest.approx(0.3)


def test_wind_pushes_fire_downwind():
    res = spread_probabilities(make_ds(u=6.0))               # wind blowing toward the east
    assert res.attrs["main_direction"] == "E"
    assert neighbours(res, "E") > neighbours(res, "N") > neighbours(res, "W")
    assert res.attrs["direction_strength"] > 0.3


def test_fire_runs_uphill():
    elev = np.tile(((N - np.arange(N)) * 20.0)[:, None], (1, N))  # rises toward the north (row 0)
    res = spread_probabilities(make_ds(elev=elev))
    assert res.attrs["main_direction"] == "N"
    assert neighbours(res, "N") > P.p0 > neighbours(res, "S")


def test_two_burning_neighbours_combine():
    burning = np.zeros((N, N), np.int16)
    burning[C, C - 1] = burning[C, C + 1] = 1                # target (C, C) between two fires
    res = spread_probabilities(make_ds(burning=burning))
    assert float(res["ignition_prob"][C, C]) == pytest.approx(1 - (1 - P.p0) ** 2)


def test_non_burnable_and_edges():
    ds = make_ds()
    ds["landcover"].values[C, C + 1] = 80                    # water east of the fire
    res = spread_probabilities(ds)
    assert neighbours(res, "E") == 0 and float(res["ignition_prob"][C, C + 1]) == 0
    burning = np.zeros((N, N), np.int16)
    burning[0, 0] = 1                                        # fire in the corner
    res = spread_probabilities(make_ds(burning=burning))
    assert res.attrs["fire_touches_grid_edge"] == 1
    assert neighbours(res, "N") >= 0                         # no wrap-around
    assert float(res["ignition_prob"][-1, -1]) == 0


def test_clipping_at_one_and_summary():
    elev = np.tile(((N - np.arange(N)) * 200.0)[:, None], (1, N))   # 63 deg slope uphill north
    res = spread_probabilities(make_ds(lifeform=1, nisar_cover=90, elev=elev, v=15.0))
    assert float(res["p_spread"].max()) == 1.0 and res.attrs["n_clipped"] > 0
    text = summary(res)
    assert "Most likely spread direction: N" in text and "P > 1" in text


def test_weather_time_selection():
    ds = make_ds()
    ds = xr.concat([ds, ds.assign_coords(time=[np.datetime64("2026-08-09T22:00")])], dim="time",
                   data_vars="minimal")
    ds["wind_u_ms"].values[1] = 5.0                          # second hour windy
    r = spread_probabilities(ds, time="2026-08-09T21:30Z")
    assert r.attrs["weather_time_utc"].startswith("2026-08-09 21:00") and r.attrs["weather_note"] == ""
    r = spread_probabilities(ds, time="2026-08-09T22:10Z")
    assert r.attrs["weather_time_utc"].startswith("2026-08-09 22:00") and r.attrs["main_direction"] == "E"
    r = spread_probabilities(ds, time="2026-08-09T20:59Z")   # just before the first step: no KeyError
    assert r.attrs["weather_time_utc"].startswith("2026-08-09 21:00") and "no weather at or before" in r.attrs["weather_note"]
    r = spread_probabilities(ds, time="2026-08-10T05:00Z")   # long after the last step: flagged as stale
    assert "older than the requested time" in r.attrs["weather_note"]
    assert "WARNING" in summary(r)


def test_predict_with_nisar_same_grid():
    ds = make_ds()
    burned = np.zeros((N, N), np.int16)
    burned[C - 1, C] = 1                                     # NISAR says north neighbour already burned
    res = predict(inputs=ds, nisar=nisar_on(ds, nisar_burned=burned,
                                            nisar_cover_pct=np.full((N, N), 80, np.float32)))
    assert float(res["ignition_prob"][C - 1, C]) == 0
    assert res.attrs["nisar_alignment"] == "same grid"
    assert float(res["p_den"][C, C + 1]) == pytest.approx(0.3)


def test_nisar_aligned_by_coordinates_not_index():
    ds = make_ds()
    burned = np.zeros((N, N), np.int16)
    burned[C - 1, C] = 1                                     # north of the fire
    nisar = nisar_on(ds, nisar_burned=burned)
    flipped = nisar.isel(y=slice(None, None, -1))            # y ascending (south first): same data, other order
    res = predict(inputs=ds, nisar=flipped)
    assert float(res["ignition_prob"][C - 1, C]) == 0        # still the north cell, not the south one
    assert float(res["ignition_prob"][C + 1, C]) > 0
    # coarser NISAR (200 m) over the same area: continuous interpolated, classes nearest
    coarse = xr.Dataset({"nisar_soil_moisture": (("y", "x"), np.full((6, 6), 0.05, np.float32)),
                         "nisar_burned": (("y", "x"), np.zeros((6, 6), np.int16))},
                        coords={"y": 1000 - 200 * (np.arange(6) + 0.5), "x": 200 * (np.arange(6) + 0.5)},
                        attrs={"crs": "EPSG:32610"})
    res = predict(inputs=ds, nisar=coarse)
    assert "regridded" in res.attrs["nisar_alignment"]
    inner = res["p_dry"].values[1:-1, 1:-1]
    assert np.allclose(inner, 0.3)                           # dry everywhere NISAR covers


def test_nisar_provider_and_template(tmp_path):
    """The NISAR module can plug in as a function: it gets the grid / area / time and fills a template."""
    ds = make_ds().assign_coords(lat=(("y", "x"), np.full((N, N), 39.7)), lon=(("y", "x"), np.full((N, N), -120.0)))
    ds.attrs["time_utc"] = "2026-08-09T21:00:00+00:00"
    seen = {}

    def get_nisar(request: NisarRequest):              # what the teammates' module would expose
        seen["request"] = request
        out = nisar_template(request)
        out["nisar_soil_moisture"] = (("y", "x"), np.full((len(request.y), len(request.x)), 0.05, np.float32))
        burned = np.zeros((len(request.y), len(request.x)), np.int16)
        burned[C - 1, C] = 1
        out["nisar_burned"] = (("y", "x"), burned)
        return out

    res = predict(inputs=ds, nisar=get_nisar)
    req = seen["request"]
    assert req.crs == "EPSG:32610" and req.res_m == 100 and req.time_utc.startswith("2026-08-09T21")
    w, s, e, n = req.bbox_lonlat
    assert w < -120.0 < e and s < 39.7 < n
    assert res.attrs["nisar_alignment"] == "same grid"
    assert float(res["p_dry"][0, 0]) == pytest.approx(0.3) and float(res["ignition_prob"][C - 1, C]) == 0
    # provider with nothing for this place / time -> neutral, recorded
    res = predict(inputs=ds, nisar=lambda request: None)
    assert res.attrs["nisar_alignment"] == "provider returned no NISAR data" and float(res["p_dry"].max()) == 0
    # a NetCDF file written from the template works too
    path = tmp_path / "nisar.nc"
    get_nisar(NisarRequest.from_inputs(ds)).to_netcdf(path, engine="h5netcdf")
    res = predict(inputs=ds, nisar=path)
    assert float(res["p_dry"][0, 0]) == pytest.approx(0.3)


def test_nisar_bad_inputs_rejected():
    ds = make_ds()
    ok = nisar_on(ds, nisar_burned=np.zeros((N, N), np.int16))
    with pytest.raises(ValueError, match="crs"):
        predict(inputs=ds, nisar=ok.assign_attrs(crs=None))
    with pytest.raises(ValueError, match="differs"):
        predict(inputs=ds, nisar=ok.assign_attrs(crs="EPSG:4326"))
    with pytest.raises(ValueError, match="unknown NISAR variables"):
        predict(inputs=ds, nisar=ok.rename(nisar_burned="burned"))


def test_predict_fetches_only_model_layers(monkeypatch):
    import geelayers
    from firespread.predict import MODEL_LAYERS
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return make_ds()
    monkeypatch.setattr(geelayers, "get_fire_inputs", fake)
    predict(point=(39.7, -120.0), start="2026-08-09T21:00Z")
    assert tuple(seen["layers"]) == MODEL_LAYERS
    assert "vegetation" not in MODEL_LAYERS and "fuel_dryness" not in MODEL_LAYERS
