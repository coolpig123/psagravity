"""Offline tests of the arrival-time (survival) model: features, labels, fit / save / load, web payload."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from firespread import spread_probabilities, survival
from firespread.survival import FEATURES, HORIZONS_H, ArrivalModel, cell_features, fit, predict_arrival
from firespread.web import predict_payload
from test_alexandridis import C, N, make_ds
from test_web import BODY, fake_predict

T0 = "2026-08-09T21:00Z"


def test_cell_features_geometry():
    elev = np.tile(np.arange(N, dtype=np.float32) * 10.0, (N, 1))       # rises 10 m per cell toward the east
    f = cell_features(spread_probabilities(make_ds(u=5.0, elev=elev)), T0)   # wind blowing toward the east
    assert set(f) == set(FEATURES)
    assert f["dist_front_m"][C, C] == 0 and f["dist_front_m"][C, C + 3] == pytest.approx(300.0)
    assert f["wind_along_ms"][C, C + 3] == pytest.approx(5.0) and f["wind_along_ms"][C, C - 3] == pytest.approx(-5.0)
    assert f["wind_cross_ms"][C - 3, C] == pytest.approx(5.0)            # north of the fire: wind blows across
    assert f["slope_path_deg"][C, C + 3] > 0 > f["slope_path_deg"][C, C - 3]   # uphill east, downhill west
    assert f["ca_burn_prob"][C, C + 1] > f["ca_burn_prob"][C, C - 1]      # downwind burns more often
    assert np.all(f["fire_area_km2"] == pytest.approx(0.01))
    assert np.all(np.abs(f["solar_hour_sin"] ** 2 + f["solar_hour_cos"] ** 2 - 1) < 1e-5)


def test_arrival_bounds_from_viirs_and_perimeter():
    from geelayers.grid import Grid
    square = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[-120.06, 39.71], [-120.02, 39.71], [-120.02, 39.74], [-120.06, 39.74],
                                            [-120.06, 39.71]]]}}]}
    grid = Grid.from_polygon(square, buffer_km=1.0, res_m=100.0)
    t = pd.to_datetime(["2026-08-09 09:00", "2026-08-09 12:00", "2026-08-09 21:00", "2026-08-09 21:05"], utc=True)
    pts = pd.DataFrame({"time_utc": t, "lat": [39.72, 39.73, 39.725, 39.90], "lon": [-120.05, -120.03, -120.04, -120.04],
                        "source": ["VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21", "VIIRS_SNPP"], "ambiguous": False})
    truth = survival.FireTruth("test", square, {"features": []}, pts, pd.Timestamp("2026-08-09 06:00", tz="UTC"))
    b = survival.arrival_bounds(grid, truth)
    hour = lambda s: (s - pd.Timestamp("2026-08-09", tz="UTC").timestamp()) / 3600   # noqa: E731
    r, c = grid.cell_of(39.73, -120.03)
    assert hour(b["upper"][r, c]) == pytest.approx(12.0) and hour(b["lower"][r, c]) == pytest.approx(9.0)
    r, c = grid.cell_of(39.72, -120.05)
    assert hour(b["upper"][r, c]) == pytest.approx(9.0) and hour(b["lower"][r, c]) == pytest.approx(6.0)  # ignition
    assert len(b["passes"]) == 3                                        # 21:00 and 21:05 are one pass
    outside = ~b["final"]
    assert np.isinf(b["upper"][outside]).all() and (b["lower"][outside] == b["t_end"]).all()   # never burned
    assert (b["lower"] <= b["upper"]).all()
    burning = survival.burning_at(b, pd.Timestamp("2026-08-09 10:00", tz="UTC"))
    assert burning[grid.cell_of(39.72, -120.05)] and not burning[grid.cell_of(39.73, -120.03)]


def toy_samples(n=3000, seed=0):
    """Fire moving 1 km/h downwind and 0.25 km/h upwind; arrival observed in 2 h intervals, far cells censored."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({k: rng.normal(size=n).astype(np.float32) for k in FEATURES})
    df["dist_front_m"] = rng.uniform(100, 20_000, n)
    df["wind_along_ms"] = rng.uniform(-5, 5, n)
    speed_kmh = np.where(df["wind_along_ms"] > 0, 1.0, 0.25)
    t = df["dist_front_m"] / 1000 / speed_kmh
    burned = t < 40
    df["lower_h"] = np.where(burned, np.maximum(np.floor(t / 2) * 2, survival.MIN_HOURS), 40.0)
    df["upper_h"] = np.where(burned, np.floor(t / 2) * 2 + 2, np.inf)
    df["weight"] = 1.0
    return df


def test_fit_learns_speed_and_round_trips(tmp_path):
    df = toy_samples()
    model = fit(df, sigma=0.5, rounds=150)
    X = df[list(FEATURES)].to_numpy(np.float32)
    med = model.median_hours(X)
    true_t = df["dist_front_m"] / 1000 / np.where(df["wind_along_ms"] > 0, 1.0, 0.25)
    ok = true_t < 30
    assert np.median(np.abs(np.log(med[ok]) - np.log(true_t[ok]))) < 0.5
    p6, p48 = model.p_within(med, 6), model.p_within(med, 48)
    assert (p48 >= p6).all() and ((p6 >= 0) & (p48 <= 1)).all()
    scores = survival.evaluate(model, df)
    assert scores["auc_12h"] > 0.9 and scores["auc_12h"] >= scores["auc_dist_12h"] - 0.02

    path = tmp_path / "m.json"
    model.meta = {"trained_utc": "test", "fires": ["toy"]}
    model.save(path)
    again = ArrivalModel.load(path)
    assert again.sigma == 0.5 and np.allclose(again.median_hours(X[:50]), med[:50])
    with pytest.raises(FileNotFoundError):
        ArrivalModel.load(tmp_path / "missing.json")


@pytest.fixture(scope="module")
def toy_model():
    m = fit(toy_samples(), sigma=0.5, rounds=60)
    m.meta = {"trained_utc": "test", "fires": ["toy"]}
    return m


def test_predict_arrival_adds_layers(toy_model):
    res = predict_arrival(spread_probabilities(make_ds(u=5.0)), T0, model=toy_model)
    assert np.isnan(res["arrival_h"].values[C, C])                      # burning already
    assert res["arrival_h"].values[C, C + 4] < res["arrival_h"].values[C, C - 4]   # downwind arrives first
    for h in HORIZONS_H:
        assert f"p_arrive_{h}h" in res and res.attrs[f"arrival_area_{h}h_km2"] >= 0
    assert res.attrs["arrival_area_48h_km2"] >= res.attrs["arrival_area_6h_km2"]
    assert "XGBoost AFT" in res.attrs["arrival_model"]


def test_payload_with_and_without_arrival_model(toy_model):
    out = predict_payload({**BODY, "use_nisar": False, "steps": 3}, predict_fn=fake_predict({}),
                          arrival_fn=lambda r, t: predict_arrival(r, t, model=toy_model))
    s = out["summary"]
    assert s["arrival"] is True and s["arrival_horizons_h"] == list(HORIZONS_H)
    props = out["cells"]["features"][0]["properties"]
    assert {"p", "t50", "p6", "p12", "p24", "p48"} <= set(props)
    json.dumps(out, allow_nan=False)

    def missing(r, t):
        raise FileNotFoundError("no arrival-time model")
    out = predict_payload({**BODY, "use_nisar": False, "steps": 3}, predict_fn=fake_predict({}), arrival_fn=missing)
    assert out["summary"]["arrival"] is False and "t50" not in out["cells"]["features"][0]["properties"]
    assert any(n.startswith("arrival times skipped") for n in out["summary"]["notes"])
