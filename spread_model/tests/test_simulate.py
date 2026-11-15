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
    sim = simulate_spread(spread_probabilities(ds), steps=40, runs=100)
    bp = sim["burn_prob"].values
    assert (bp[:, C + 2:] == 0).all() and bp[:, :C + 2].max() > 0.5


def test_reproducible_and_wind_pushes_downwind():
    res = spread_probabilities(make_ds(u=6.0))            # wind blowing east
    a = simulate_spread(res, steps=4, runs=300, seed=7)["burn_prob"].values
    b = simulate_spread(res, steps=4, runs=300, seed=7)["burn_prob"].values
    assert (a == b).all()
    assert a[C, C + 3] > a[C, C - 3]


def test_bad_arguments():
    with pytest.raises(ValueError):
        simulate_spread(spread_probabilities(make_ds()), steps=0)


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

