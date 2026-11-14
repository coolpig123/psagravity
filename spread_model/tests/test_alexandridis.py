"""Offline tests: each factor against its formula, and whole-grid cases with known answers."""
import math

import numpy as np
import pytest
import xarray as xr

from firespread import (AlexandridisParams, p_den, p_mask, p_slope, p_veg, p_wind, predict, slope_angle_deg,
                        spread_probabilities, summary)
from firespread.alexandridis import DIR_NAMES, shift

P = AlexandridisParams()
N = 11                   # grid size; the fire is the centre cell
C = N // 2


def make_ds(n=N, res=100.0, elev=None, u=0.0, v=0.0, lifeform=2, cover=45.0, landcover=20, burning=None):
    """Synthetic geelayers-like Dataset: shrub, normal density, flat, calm, centre cell burning."""
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
    return ds


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


def test_p_den_thresholds_and_nisar_priority():
    cover = np.array([[10, 30, 59, 60, np.nan]], np.float32)
    ds = xr.Dataset({"cover_pct": (("y", "x"), cover),
                     "ndvi": (("y", "x"), np.array([[0.9] * 5], np.float32))})
    den, _ = p_den(ds, P)
    assert den.tolist()[0] == pytest.approx([-0.4, 0.0, 0.0, 0.3, 0.3])          # last from NDVI 0.9 -> dense
    ds["nisar_cover_pct"] = (("y", "x"), np.array([[80, 80, 80, 80, 80]], np.float32))
    den, src = p_den(ds, P)
    assert (den == 0.3).all() and src.startswith("nisar_cover_pct")


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


def test_flat_calm_reference_fuel_is_p0_all_round():
    res = spread_probabilities(make_ds())
    for d in DIR_NAMES:
        assert neighbours(res, d) == pytest.approx(P.p0)
    ign = res["ignition_prob"].values
    assert ign[C - 1:C + 2, C - 1:C + 2].sum() == pytest.approx(8 * P.p0)
    assert ign.sum() == pytest.approx(8 * P.p0)              # only the 8 neighbours can catch fire
    assert ign[C, C] == 0                                    # the burning cell itself
    assert res.attrs["direction_strength"] == pytest.approx(0, abs=1e-6)


def test_full_product_one_neighbour():
    """Forest, dense, wind toward the east 5 m/s, 10 m rise to the east over 100 m."""
    elev = np.zeros((N, N))
    elev[:, C + 1:] = 10.0
    res = spread_probabilities(make_ds(lifeform=1, cover=70, u=5.0, elev=elev))
    expected = (P.p0 * (1 + 0.4) * (1 + 0.3) * math.exp(P.c1 * 5) * math.exp(P.a * math.degrees(math.atan(10 / 100))))
    assert neighbours(res, "E") == pytest.approx(min(expected, 1.0), rel=1e-5)


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
    res = spread_probabilities(make_ds(lifeform=1, cover=90, elev=elev, v=15.0))
    assert float(res["p_spread"].max()) == 1.0 and res.attrs["n_clipped"] > 0
    text = summary(res)
    assert "Most likely spread direction: N" in text and "P > 1" in text


def test_predict_with_inputs_and_nisar():
    ds = make_ds()
    nisar = xr.Dataset({"nisar_burned": (("y", "x"), np.zeros((N, N), np.int16))})
    nisar["nisar_burned"].values[C - 1, C] = 1               # NISAR says north neighbour already burned
    res = predict(inputs=ds, nisar=nisar)
    assert float(res["ignition_prob"][C - 1, C]) == 0
    with pytest.raises(ValueError, match="same grid"):
        predict(inputs=ds, nisar=xr.Dataset({"nisar_burned": (("y", "x"), np.zeros((3, 3)))}))
