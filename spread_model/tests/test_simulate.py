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
