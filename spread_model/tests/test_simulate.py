"""Multi-step Monte Carlo cellular automaton (simulate.py)."""
import numpy as np
import pytest

from firespread import spread_probabilities
from firespread.simulate import simulate_spread
from test_alexandridis import C, N, make_ds


def certain(result):
    """The same result with every spread certain (P = 1): the automaton becomes deterministic."""
    r = result.copy()
    r["p_spread"] = r["p_spread"] * 0 + 1
    return r


def test_one_step_matches_ignition_prob():
    res = spread_probabilities(make_ds(u=3.0))
    sim = simulate_spread(res, steps=1, runs=4000, seed=1)
    assert np.allclose(sim["burn_prob"].values, res["ignition_prob"].values, atol=0.03)


def test_certain_spread_grows_one_ring_per_step():
    sim = simulate_spread(certain(spread_probabilities(make_ds())), steps=3, runs=5)
    yy, xx = np.mgrid[:N, :N]
    ring = np.maximum(abs(yy - C), abs(xx - C))         # Chebyshev distance: 8 neighbours per step
    assert (sim["burn_prob"].values == ((ring >= 1) & (ring <= 3))).all()   # initial fire not counted
    reached = (ring >= 1) & (ring <= 3)
    assert (sim["arrival_step"].values[reached] == ring[reached]).all()
    assert np.isnan(sim["arrival_step"].values[~reached]).all()
    assert sim.attrs["sim_expected_area_km2"] == pytest.approx((7 * 7 - 1) * 0.01)
    assert sim.attrs["sim_reached_grid_edge"] == 0
    assert simulate_spread(certain(spread_probabilities(make_ds())), steps=C + 1, runs=2).attrs["sim_reached_grid_edge"] == 1


def test_fire_does_not_cross_water_and_stops_when_out():
    landcover = np.full((N, N), 20, np.int16)
    landcover[:, C + 2] = 80                              # a river two cells east of the fire
    ds = make_ds(u=8.0)                                   # strong wind toward it
    ds["landcover"] = (("y", "x"), landcover)
    res = spread_probabilities(ds)
    sim_no_spot = simulate_spread(res, steps=40, runs=100, spotting=False)
    bp = sim_no_spot["burn_prob"].values
    assert (bp[:, C + 2:] == 0).all() and bp[:, :C + 2].max() > 0.5

    # With spotting enabled, embers in strong wind (8 m/s) can jump across the narrow river to the eastern bank,
    # while the water cells themselves never burn.
    sim_spot = simulate_spread(res, steps=40, runs=200, spotting=True)
    bp_spot = sim_spot["burn_prob"].values
    assert (bp_spot[:, C + 2] == 0).all()
    assert bp_spot[:, C + 3:].max() > 0.05

    # Wide water (all the way to the east edge) or light wind (u = 2 m/s) cannot be crossed even with spotting.
    calm = make_ds(u=2.0)
    calm["landcover"] = (("y", "x"), landcover)
    assert (simulate_spread(spread_probabilities(calm), steps=40, runs=100)["burn_prob"].values[:, C + 2:] == 0).all()


def test_reproducible_and_wind_pushes_downwind():
    res = spread_probabilities(make_ds(u=6.0))            # wind blowing east
    a = simulate_spread(res, steps=4, runs=300, seed=7)["burn_prob"].values
    b = simulate_spread(res, steps=4, runs=300, seed=7)["burn_prob"].values
    assert (a == b).all()
    assert a[C, C + 3] > a[C, C - 3]


def test_spotting_fields_and_moisture_dependence():
    calm = spread_probabilities(make_ds(u=3.0))
    windy_dry = spread_probabilities(make_ds(u=10.0, temp=35.0, rh=10.0))
    windy_wet = spread_probabilities(make_ds(u=10.0, temp=10.0, rh=95.0))
    assert float(calm["p_spot_launch"].max()) == 0.0
    assert float(windy_dry["p_spot_launch"].max()) > 0.0
    assert float(windy_dry["p_spot_ignite"][C, C + 1]) > float(windy_wet["p_spot_ignite"][C, C + 1])
    assert float(windy_dry["p_spot_ignite"][C, C]) == 0.0     # already burning -> P_mask = 0
    # gusts loft firebrands even when 10 m mean wind is below spot_min_wind_ms
    gusty = make_ds(u=4.0)
    gusty["gust_ms"] = (("time", "y", "x"), np.full((1, N, N), 11.0, np.float32))
    assert float(spread_probabilities(gusty)["p_spot_launch"].max()) > 0.0


def test_bad_arguments():
    with pytest.raises(ValueError):
        simulate_spread(spread_probabilities(make_ds()), steps=0)


def _square(half=2):
    sq = np.zeros((N, N), np.int16)
    sq[C - half:C + half + 1, C - half:C + half + 1] = 1
    return sq


def test_front_only_on_east_edge_spreads_east_not_west():
    sq = _square()
    front = np.zeros((N, N), bool)
    front[C - 2:C + 3, C + 2] = True                      # only the east edge of the square is burning
    res = certain(spread_probabilities(make_ds(burning=sq)))
    sim = simulate_spread(res, steps=2, runs=3, spotting=False, front=front)
    bp = sim["burn_prob"].values
    assert (bp[C - 2:C + 3, C + 3:C + 5] == 1).all()       # east: 2 columns in 2 steps
    assert (bp[:, :C - 2] == 0).all()                      # nothing west of the cold west edge
    assert (bp[sq.astype(bool)] == 0).all()                # the interior never re-burns
    assert sim.attrs["sim_front_cells"] == 5
    whole = simulate_spread(res, steps=2, runs=3, spotting=False)    # no front: every edge spreads
    assert (whole["burn_prob"].values[C - 2:C + 3, C - 3] == 1).all()


def test_front_from_dataset_and_empty_front_fallback():
    sq = _square()
    ds = make_ds(burning=sq)
    front = np.zeros((N, N), np.int16)
    front[C - 2:C + 3, C + 2] = 1
    ds["active_front"] = (("y", "x"), front)
    res = spread_probabilities(ds)                         # one-step summary uses the front too
    ign = res["ignition_prob"].values
    assert (ign[:, :C - 2] == 0).all() and (ign[C, C + 3] > 0)
    assert res.attrs["main_direction"] == "E" and res.attrs["active_front_cells"] == 5
    assert res.attrs["active_front_source"] == "VIIRS active front"
    sim = simulate_spread(certain(res), steps=1, runs=2, spotting=False)       # picks result["active_front"]
    assert (sim["burn_prob"].values[:, :C - 2] == 0).all()
    empty = simulate_spread(certain(res), steps=1, runs=2, spotting=False, front=np.zeros((N, N), bool))
    assert (empty["burn_prob"].values[C, C - 3] == 1)      # empty front -> the whole burned area spreads
    with pytest.raises(ValueError):
        simulate_spread(res, steps=1, front=np.zeros((3, 3), bool))


def test_time_developing_weather_shifts_fire_direction():
    import xarray as xr

    # 3-hour forecast: hour 0 blows east (u=8, v=0), hours 1 and 2 shift strongly north (u=0, v=10)
    ds0 = make_ds(u=8.0, v=0.0, temp=32.0, rh=15.0)
    ds1 = make_ds(u=0.0, v=10.0, temp=30.0, rh=20.0).assign_coords(
        time=np.array(["2026-08-09T22:00"], dtype="datetime64[ns]")
    )
    ds2 = make_ds(u=0.0, v=10.0, temp=28.0, rh=25.0).assign_coords(
        time=np.array(["2026-08-09T23:00"], dtype="datetime64[ns]")
    )
    ds = xr.concat([ds0, ds1, ds2], dim="time", data_vars="minimal")

    res = spread_probabilities(ds)
    assert "p_spread_time" in res
    assert res["p_spread_time"].shape == (3, 8, N, N)
    assert np.allclose(res["wind_speed_time_ms"].values, [8.0, 10.0, 10.0])

    sim_dynamic = simulate_spread(res, steps=5, runs=400, seed=42, time_developing=True)
    sim_static = simulate_spread(res, steps=5, runs=400, seed=42, time_developing=False)

    assert sim_dynamic.attrs["sim_time_developing"] == 1
    assert sim_dynamic.attrs["sim_weather_steps"] == 3
    assert sim_dynamic.attrs["sim_forecast_hours"] == pytest.approx(2.0)
    assert "arrival_hours" in sim_dynamic

    # North of ignition (row < C) must burn with much higher probability when the northward wind shift is simulated
    north_dynamic = float(sim_dynamic["burn_prob"].values[:C - 1, C].mean())
    north_static = float(sim_static["burn_prob"].values[:C - 1, C].mean())
    assert north_dynamic > north_static + 0.15


def test_meteomatics_weather_enhancements_soil_moisture_precip_and_gusts():
    ds = make_ds(u=4.0, v=0.0, temp=30.0, rh=20.0)
    base_res = spread_probabilities(ds)

    # Weather soil_moisture (e.g. from Meteomatics) acts as fallback when nisar_soil_moisture is absent
    ds_dry_sm = ds.copy()
    ds_dry_sm["soil_moisture"] = (("time", "y", "x"), np.full((1, N, N), 0.05, np.float32))
    dry_res = spread_probabilities(ds_dry_sm)
    assert float(dry_res["p_dry"][C, C + 1]) == pytest.approx(0.3)
    assert float(dry_res["ignition_prob"][C, C + 1]) > float(base_res["ignition_prob"][C, C + 1])

    # Wind gusts boost downwind spread probability
    ds_gust = ds.copy()
    ds_gust["gust_ms"] = (("time", "y", "x"), np.full((1, N, N), 12.0, np.float32))
    gust_res = spread_probabilities(ds_gust)
    assert float(gust_res["ignition_prob"][C, C + 1]) > float(base_res["ignition_prob"][C, C + 1])

    # Active precipitation damps moisture factor and ignition probability
    ds_rain = ds.copy()
    ds_rain["precip_mm"] = (("time", "y", "x"), np.full((1, N, N), 5.0, np.float32))
    rain_res = spread_probabilities(ds_rain)
    assert float(rain_res["p_m"][C, C + 1]) < float(base_res["p_m"][C, C + 1])
    assert float(rain_res["ignition_prob"][C, C + 1]) < float(base_res["ignition_prob"][C, C + 1])

