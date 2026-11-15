"""Offline tests of the arrival-time (cure-fraction survival) model: features, labels, metrics, fit / save / load,
web payload."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from firespread import spread_probabilities, survival
from firespread.survival import (FEATURES, HORIZONS_H, ArrivalModel, cell_features, fit, predict_arrival,
                                 weighted_auc)
from firespread.web import predict_payload
from test_alexandridis import C, N, make_ds
from test_web import BODY, fake_predict

T0 = "2026-08-09T21:00Z"


def test_fastest_path_features():
    f = cell_features(spread_probabilities(make_ds(u=5.0)))            # wind blowing toward the east
    assert set(f) == set(FEATURES) and np.isnan(f["path_steps"][C, C])  # burning cell: no feature
    assert f["path_steps"][C, C + 4] < f["path_steps"][C - 4, C] < f["path_steps"][C, C - 4]   # down < cross < upwind
    assert f["path_len_m"][C, C + 4] == pytest.approx(400.0)
    assert f["path_wind_along_ms"][C, C + 4] == pytest.approx(5.0) and f["path_wind_along_ms"][C, C - 4] == pytest.approx(-5.0)

    elev = np.tile(np.arange(N, dtype=np.float32) * 10.0, (N, 1))       # rises 10 m per cell toward the east
    f = cell_features(spread_probabilities(make_ds(elev=elev)))
    assert f["path_up_m"][C, C + 3] == pytest.approx(30.0) and f["path_down_m"][C, C + 3] == 0
    assert f["path_down_m"][C, C - 3] == pytest.approx(30.0) and f["path_up_m"][C, C - 3] == 0


def test_barriers_are_walls_not_shortcuts():
    ds = make_ds()
    lc = np.full((N, N), 20, np.int16)
    lc[:, C + 2] = 80                                                   # a river of water east of the fire
    ds["landcover"] = (("y", "x"), lc)
    f = cell_features(spread_probabilities(ds))
    assert np.isnan(f["path_steps"][C, C + 3]) and np.isfinite(f["path_steps"][C, C + 1])   # unreachable beyond
    lc[0, C + 2] = 20                                                   # a crossing at the top edge
    ds["landcover"] = (("y", "x"), lc)
    f = cell_features(spread_probabilities(ds))
    assert f["path_len_m"][C, C + 3] > 1000                             # goes round, not 300 m straight across


def test_neutral_path_ignores_nisar_terms():
    from test_alexandridis import nisar_on
    from firespread import attach_nisar
    ds = make_ds(u=3.0)
    base = cell_features(spread_probabilities(ds))
    wet = attach_nisar(ds, nisar_on(ds, nisar_soil_moisture=np.full((N, N), 0.4, np.float32)))
    assert np.allclose(cell_features(spread_probabilities(wet))["path_steps"], base["path_steps"], equal_nan=True)


def test_arrival_bounds_from_viirs_and_perimeter():
    from geelayers.grid import Grid
    square = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[-120.06, 39.71], [-120.02, 39.71], [-120.02, 39.74], [-120.06, 39.74],
                                            [-120.06, 39.71]]]}}]}
    grid = Grid.from_polygon(square, buffer_km=1.0, res_m=100.0)
    t = pd.to_datetime(["2026-08-09 09:00", "2026-08-09 12:00", "2026-08-09 21:00", "2026-08-09 21:05",
                        "2026-08-10 03:00"], utc=True)
    pts = pd.DataFrame({"time_utc": t, "lat": [39.72, 39.73, 39.725, 39.90, 39.735],
                        "lon": [-120.05, -120.03, -120.04, -120.04, -120.055],
                        "source": ["VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21", "VIIRS_SNPP", "VIIRS_SNPP"],
                        "ambiguous": False})
    # a FEDS perimeter at 13:00 already contains the cell VIIRS only detects at 03:00 the next day
    feds_poly = [[[-120.058, 39.733], [-120.052, 39.733], [-120.052, 39.737], [-120.058, 39.737], [-120.058, 39.733]]]
    feds = {"features": [{"type": "Feature", "properties": {"obs_time_utc": "2026-08-09T13:00:00Z"},
                          "geometry": {"type": "Polygon", "coordinates": feds_poly}}]}
    truth = survival.FireTruth("test", square, feds, pts, pd.Timestamp("2026-08-09 06:00", tz="UTC"))
    b = survival.arrival_bounds(grid, truth)
    hour = lambda s: (s - pd.Timestamp("2026-08-09", tz="UTC").timestamp()) / 3600   # noqa: E731
    r, c = grid.cell_of(39.73, -120.03)
    assert hour(b["upper"][r, c]) == pytest.approx(12.0) and hour(b["lower"][r, c]) == pytest.approx(9.0)
    r, c = grid.cell_of(39.72, -120.05)
    assert hour(b["upper"][r, c]) == pytest.approx(9.0) and hour(b["lower"][r, c]) == pytest.approx(6.0)  # ignition
    r, c = grid.cell_of(39.735, -120.055)
    assert hour(b["upper"][r, c]) == pytest.approx(13.0) and hour(b["lower"][r, c]) == pytest.approx(12.0)  # not 13-13
    assert len(b["passes"]) == 4                                        # 21:00 and 21:05 are one pass
    outside = ~b["final"]
    assert np.isinf(b["upper"][outside]).all() and (b["lower"][outside] == b["t_end"]).all()   # never burned
    assert (b["lower"] <= b["upper"]).all()
    burning = survival.burning_at(b, pd.Timestamp("2026-08-09 10:00", tz="UTC"))
    assert burning[grid.cell_of(39.72, -120.05)] and not burning[grid.cell_of(39.73, -120.03)]


def test_weighted_auc():
    s = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([False, False, True, True])
    assert weighted_auc(s, y, np.ones(4)) == pytest.approx(0.75)
    # negative at 0.4 weighs 3: positive 0.35 beats weight 1 of 4, positive 0.8 beats all 4 -> (1 + 4) / (2 * 4)
    assert weighted_auc(s, y, np.array([1.0, 3.0, 1.0, 1.0])) == pytest.approx(0.625)
    assert weighted_auc(np.array([1.0, 1.0]), np.array([True, False]), np.ones(2)) == 0.5     # ties count half
    assert weighted_auc(s, np.zeros(4, bool), np.ones(4)) is None


def toy_samples(n=4000, seed=0, starts=2):
    """Fire moving 1 km/h downwind, 0.25 km/h upwind, stopping at 12 km (beyond: never burns). Path steps are the
    distance in cells; 2 h observation intervals."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({k: np.zeros(n, np.float32) for k in FEATURES})
    d = rng.uniform(100, 20_000, n)
    along = rng.uniform(-5, 5, n)
    df["path_len_m"], df["path_steps"], df["path_wind_along_ms"] = d, d / 100, along
    df["barrier_dist_m"] = d * rng.uniform(0.9, 1.0, n)            # straight-ish: a little shorter than the path
    t = d / 1000 / np.where(along > 0, 1.0, 0.25)
    burned = d < 12_000
    df["lower_h"] = np.where(burned, np.maximum(np.floor(t / 2) * 2, survival.MIN_HOURS), 40.0)
    df["upper_h"] = np.where(burned, np.floor(t / 2) * 2 + 2, np.inf)
    df["weight"] = 1.0
    df["euclid_dist_m"], df["ca_burn_prob"] = d, np.exp(-d / 3000)
    df["start_utc"] = rng.integers(0, starts, n).astype(str)
    return df, t, burned


def test_fit_learns_whether_and_when_and_round_trips(tmp_path):
    df, t, burned = toy_samples()
    model = fit(df, sigma=0.4, rounds=150)
    X = df[list(FEATURES)].to_numpy(np.float32)
    pe, med = model.p_ever(X), model.median_if_burns(X)
    assert pe[df["path_len_m"] < 8000].mean() > 0.9 and pe[df["path_len_m"] > 16000].mean() < 0.1
    ok = burned & (t < 30)
    assert np.median(np.abs(np.log(med[ok]) - np.log(t[ok]))) < 0.4    # when, for cells that burn: not dragged late
    p6, p48 = model.p_within(pe, med, 6), model.p_within(pe, med, 48)
    assert (p48 >= p6).all() and (p48 <= pe + 1e-6).all()
    scores = survival.evaluate(model, df)
    assert scores["auc_12h"] > 0.9 and scores["auc_ever"] > 0.95 and scores["interval_hit"] > 0.3
    assert scores["n_known_48h"] > scores["n_pos_48h"]                  # never-burned cells count as negatives
    assert scores["calibration_24h"]

    path = tmp_path / "m.json"
    model.meta = {"trained_utc": "test", "fires": ["toy"]}
    model.save(path)
    again = ArrivalModel.load(path)
    assert again.sigma == 0.4 and np.allclose(again.p_ever(X[:50]), pe[:50])
    assert np.allclose(again.median_if_burns(X[:50]), med[:50])
    with pytest.raises(FileNotFoundError):
        ArrivalModel.load(tmp_path / "missing.json")


@pytest.fixture(scope="module")
def toy_model():
    m = fit(toy_samples()[0], sigma=0.4, rounds=60)
    m.meta = {"trained_utc": "test", "fires": ["toy"]}
    return m


def test_predict_arrival_adds_layers(toy_model):
    res = predict_arrival(spread_probabilities(make_ds(u=5.0)), T0, model=toy_model)
    assert np.isnan(res["arrival_h"].values[C, C]) and np.isnan(res["p_ever"].values[C, C])   # burning already
    assert res["arrival_h"].values[C, C + 4] < res["arrival_h"].values[C, C - 4]   # downwind arrives first
    for h in HORIZONS_H:
        assert f"p_arrive_{h}h" in res and res.attrs[f"arrival_area_{h}h_km2"] >= 0
    assert (np.nan_to_num(res["p_arrive_48h"].values) <= np.nan_to_num(res["p_ever"].values) + 1e-6).all()
    assert res.attrs["arrival_area_48h_km2"] >= res.attrs["arrival_area_6h_km2"]
    assert "cure-fraction" in res.attrs["arrival_model"]

    island = make_ds(u=5.0)
    lc = np.full((N, N), 20, np.int16)
    lc[:, C + 2] = 80                                                   # water wall: land beyond is unreachable
    lc[:, 0] = 80                                                       # and a lake cell west
    island["landcover"] = (("y", "x"), lc)
    res = predict_arrival(spread_probabilities(island), T0, model=toy_model)
    for cell in ((C, C + 4), (C, C + 2), (C, 0)):
        assert res["p_ever"].values[cell] == 0 and np.isinf(res["arrival_h"].values[cell])
        assert res["p_arrive_48h"].values[cell] == 0
    assert res["p_arrive_48h"].values[C, C - 3] > 0


def test_payload_with_and_without_arrival_model(toy_model):
    out = predict_payload({**BODY, "use_nisar": False, "steps": 3}, predict_fn=fake_predict({}),
                          arrival_fn=lambda r, t: predict_arrival(r, t, model=toy_model))
    s = out["summary"]
    assert s["arrival"] is True and s["arrival_horizons_h"] == list(HORIZONS_H)
    assert s["n_cells"] <= s["n_map_cells"] == len(out["cells"]["features"])   # CA count vs squares on the map
    props = out["cells"]["features"][0]["properties"]
    assert {"p", "pe", "t50", "t25", "t75", "p6", "p12", "p24", "p48"} <= set(props)
    json.dumps(out, allow_nan=False)

    def missing(r, t):
        raise FileNotFoundError("no arrival-time model")
    out = predict_payload({**BODY, "use_nisar": False, "steps": 3}, predict_fn=fake_predict({}), arrival_fn=missing)
    assert out["summary"]["arrival"] is False and "t50" not in out["cells"]["features"][0]["properties"]
    assert out["summary"]["n_cells"] == out["summary"]["n_map_cells"]
    assert any(n.startswith("arrival times skipped") for n in out["summary"]["notes"])


def test_calibration_fixes_overconfidence_and_keeps_order():
    df, _, _ = toy_samples(seed=1)
    df.loc[df.sample(frac=0.5, random_state=0).index, ["lower_h", "upper_h"]] = [40.0, np.inf]  # half never burn
    model = fit(df, sigma=0.4, rounds=60)
    before = survival.evaluate(model, df)
    model.calibration = survival.fit_calibration([(model, df)])
    after = survival.evaluate(model, df)
    assert after["brier_24h"] < before["brier_24h"]
    top = after["calibration_24h"][-1]                                  # [bin, predicted, observed, weight]
    assert abs(top[1] - top[2]) < 0.1
    X = df[list(FEATURES)].to_numpy(np.float32)
    probs = model.probabilities(model.p_ever(X), model.median_if_burns(X))
    for a, b in zip(HORIZONS_H, HORIZONS_H[1:]):
        assert (probs[a] <= probs[b] + 1e-9).all()
    assert (probs[HORIZONS_H[-1]] <= probs["ever"] + 1e-9).all()


def test_burn_scars_are_walls():
    ds = make_ds()
    scar = np.zeros((N, N), np.int16)
    scar[:, C + 2] = 1                                                  # an old Sentinel-2 burn scar east of the fire
    ds["s2_burned"] = (("y", "x"), scar)
    f = cell_features(spread_probabilities(ds))
    assert np.isnan(f["path_steps"][C, C + 3])                          # no fuel in the scar: nothing beyond it


def test_path_rate_is_not_clipped():
    slow = cell_features(spread_probabilities(make_ds(u=10.0)))["path_steps"][C, C + 4]
    fast = cell_features(spread_probabilities(make_ds(u=20.0)))["path_steps"][C, C + 4]
    assert fast < slow                                                  # clipped to 1, both would be 4 steps
    f = cell_features(spread_probabilities(make_ds(u=20.0)))
    diag, straight = f["path_steps"][C - 3, C + 3] / (300 * np.sqrt(2)), f["path_steps"][C, C + 3] / 300
    # cost per metre: a 45-degree front must not be faster than a straight one along the wind's own direction
    assert diag >= straight - 1e-9


def test_physics_only_model_round_trips(tmp_path):
    df, t, burned = toy_samples()
    model = fit(df, sigma=0.4, rounds=0)                                # anchor lines, no trees
    assert model.cure is None and model.time is None and model.anchor["time"][1] > 0 > model.anchor["cure"][1]
    X = df[list(FEATURES)].to_numpy(np.float32)
    med = model.median_if_burns(X)
    far, near = df["path_len_m"] > 15000, df["path_len_m"] < 2000
    assert med[far].min() > med[near].max() * 0.5                       # keeps growing with path cost
    model.meta = {"trained_utc": "test", "fires": ["toy"]}
    model.save(tmp_path / "m.json")
    assert not (tmp_path / "m.json").exists()                            # no tree files for a physics-only model
    again = ArrivalModel.load(tmp_path / "m.json")
    assert again.cure is None and np.allclose(again.median_if_burns(X[:20]), med[:20])


def test_calibration_has_no_floor_from_unreachable_cells():
    df, _, _ = toy_samples(seed=2)
    df.loc[df.index[:400], "path_steps"] = np.nan                        # unreachable, some of them burned
    model = fit(df, sigma=0.4, rounds=30)
    cal = survival.fit_calibration([(model, df)])
    xs, ys = cal["ever"]
    assert xs == sorted(xs) and ys == sorted(ys) and xs.count(0.0) <= 1
    assert np.interp(1e-4, xs, ys) < 0.05                                # a raw ~0 stays ~0


def test_unreachable_cells_are_exactly_never():
    df, _, _ = toy_samples(seed=3)
    df.loc[df.index[:100], "path_steps"] = np.nan
    for rounds in (0, 20):
        model = fit(df, sigma=0.4, rounds=rounds)
        model.calibration = survival.fit_calibration([(model, df)])
        X = df[list(FEATURES)].to_numpy(np.float32)
        probs = model.probabilities(model.p_ever(X), model.median_if_burns(X))
        for v in probs.values():
            assert np.isfinite(v).all() and (v[:100] == 0).all()
        s = survival.evaluate(model, df)
        assert all(np.isfinite(s[f"brier_{h}h"]) for h in HORIZONS_H)
        assert np.isfinite(survival._mean_brier(model, df))


def test_arrival_quantiles_agree_with_the_chances(toy_model):
    from scipy.stats import norm
    pe = np.array([0.4, 0.4, 0.0, np.nan])
    med = np.array([10.0, 200.0, 10.0, 10.0])
    probs = {"ever": pe, **{h: pe * norm.cdf((np.log(h) - np.log(med)) / 1.0) for h in HORIZONS_H}}
    q = survival.arrival_quantiles(probs)
    assert q[0.5][:2] == pytest.approx([10.0, 200.0], rel=1e-3)        # exact for a log-normal, inside and beyond
    assert q[0.25][0] == pytest.approx(10 * np.exp(-0.6745), rel=1e-3)
    assert np.isinf(q[0.5][2]) and np.isnan(q[0.5][3])                   # never reached / burning

    model = fit(toy_samples()[0], sigma=0.4, rounds=0)
    model.calibration = survival.fit_calibration([(model, toy_samples(seed=4)[0])])
    res = predict_arrival(spread_probabilities(make_ds(u=5.0)), T0, model=model)
    m = np.isfinite(res["arrival_h"].values) & (res["p_ever"].values > 0)
    # where the median is under 6 h, at least half of P(ever) has arrived by 6 h, and the reverse
    p6, pe_ = res["p_arrive_6h"].values[m], res["p_ever"].values[m]
    t50 = res["arrival_h"].values[m]
    assert np.all((p6 >= 0.5 * pe_ - 1e-4) == (t50 <= 6 + 1e-3)) or np.allclose(p6 / pe_, 0.5, atol=1e-3)
    assert (res["arrival_h_q25"].values[m] <= t50 + 1e-6).all() and (t50 <= res["arrival_h_q75"].values[m] + 1e-6).all()


def test_barrier_distance_goes_round_walls():
    ds = make_ds(u=8.0)                                                 # strong wind east: path cost is lopsided
    f = cell_features(spread_probabilities(ds))
    assert f["barrier_dist_m"][C, C + 3] == pytest.approx(300.0) and f["barrier_dist_m"][C, C - 3] == pytest.approx(300.0)
    lc = np.full((N, N), 20, np.int16)
    lc[:, C + 2] = 80
    lc[0, C + 2] = 20                                                   # a river with one crossing at the top
    ds["landcover"] = (("y", "x"), lc)
    f = cell_features(spread_probabilities(ds))
    assert f["barrier_dist_m"][C, C + 3] > 1000                         # round the river, not across it


def test_anchor_on_two_terms():
    df, t, burned = toy_samples()
    model = fit(df, None, 0, terms=("path_steps", "barrier_dist_m"))
    assert model.anchor["terms"] == ["path_steps", "barrier_dist_m"] and len(model.anchor["time"]) == 3
    X = df[list(FEATURES)].to_numpy(np.float32)
    ok = burned & (t < 30)
    assert np.median(np.abs(np.log(model.median_if_burns(X)[ok]) - np.log(t[ok]))) < 1.0


def test_quantiles_survive_flat_calibration_steps():
    # calibrated staircases: 24 h and 48 h on the same step, 6 h and 12 h on the same step
    pe = np.array([0.5, 0.5])
    probs = {"ever": pe, 6: np.array([0.30, 0.05]), 12: np.array([0.30, 0.10]), 24: np.array([0.40, 0.12]),
             48: np.array([0.45, 0.12])}
    q = survival.arrival_quantiles(probs, sigma=1.0)
    assert np.isfinite(q[0.5]).all() and np.isfinite(q[0.75]).all()    # not "no estimate" on a flat 24-48 h step
    assert (q[0.25] < 6).all() or q[0.25][0] < q[0.5][0]                # no collapse onto 6 h
    assert q[0.25][0] < q[0.5][0] < q[0.75][0] and q[0.25][1] < q[0.5][1] < q[0.75][1]
    assert (q[0.75] < 24 * 60).all()                                    # no months from a microscopic slope
