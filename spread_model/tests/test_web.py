"""Offline tests of the web payload (POST /api/predict): predict is mocked, nothing is fetched."""
import json

import numpy as np
import pytest

from firespread import spread_probabilities
from firespread.web import cells_geojson, parse_request, predict_payload
from test_alexandridis import C, N, make_ds

POLYGON = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
    "type": "Polygon", "coordinates": [[[-120.05, 39.72], [-120.03, 39.72], [-120.03, 39.74], [-120.05, 39.72]]]}}]}
BODY = {"polygon": POLYGON, "start": "2026-08-09T21:00Z", "use_firms": False}   # offline: no FIRMS request


def fake_predict(seen):
    def predict(**kw):
        seen.update(kw)
        ds = make_ds(u=5.0)                                   # wind blowing east
        ds.attrs.update({"weather_source": "rtma", "s2_clear_pct": "100", "s2_burned_pct": "0.0"})
        if kw["nisar"] is not None:
            kw["nisar"](None)                                 # what attach_nisar does with a provider
            ds.attrs.update({"nisar_passes_used": "2026-08-04 A34", "nisar_coverage_pct": 97.0})
        return spread_probabilities(ds)
    return predict


def test_parse_request_defaults_and_errors():
    req = parse_request({k: v for k, v in BODY.items() if k != "use_firms"})
    assert req == {"polygon": POLYGON, "start": "2026-08-09T21:00:00+00:00", "radius_km": 5.0, "res_m": 100.0,
                   "steps": 30, "use_nisar": True, "burn_scars": True, "arrival": True, "use_firms": True}
    assert parse_request({**BODY, "steps": 5})["steps"] == 5
    with pytest.raises(ValueError):
        parse_request({**BODY, "steps": 0})
    for bad in ({"start": "2026-08-09T21:00Z"},                              # no polygon
                {"polygon": POLYGON},                                         # no start
                {**BODY, "start": "yesterday-ish"},
                {**BODY, "radius_km": 500}, {**BODY, "res_m": "fine"}, []):
        with pytest.raises(ValueError):
            parse_request(bad)


def test_predict_payload_calls_predict_and_returns_map_cells():
    seen = {}
    out = predict_payload({**BODY, "radius_km": 3, "use_nisar": True, "burn_scars": False, "steps": 3, "arrival": False},
                          predict_fn=fake_predict(seen), nisar_factory=lambda: (lambda request: None))
    assert seen["polygon"] == POLYGON and seen["start"] == "2026-08-09T21:00:00+00:00"
    assert seen["radius_km"] == 3.0 and seen["res_m"] == 100.0 and seen["burn_scars"] is False
    assert callable(seen["nisar"])

    cells = out["cells"]["features"]
    assert 8 < len(cells) <= 48 and len(cells) == out["summary"]["n_cells"]   # beyond the first ring, within 3 steps
    assert out["summary"]["sim_steps"] == 3 and out["summary"]["sim_expected_area_km2"] > 0
    ring = cells[0]["geometry"]["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]
    assert all(-180 <= lon <= 180 and -90 <= lat <= 90 for lon, lat in ring)
    assert all(0.01 < f["properties"]["p"] <= 1 for f in cells)
    w, s, e, n = out["bounds"]
    assert w < min(p[0] for p in ring) and max(p[0] for p in ring) < e and s < n

    info = out["summary"]
    assert info["main_direction"] == "E" and 0 < info["direction_strength"] <= 1
    assert info["expected_new_area_km2"] > 0 and info["weather_source"] == "rtma"
    assert info["nisar_passes_used"] == "2026-08-04 A34" and info["nisar_coverage_pct"] == 97.0
    assert info["s2_clear_pct"] == "100" and info["notes"] == []
    assert "Most likely spread direction" in info["text"]
    json.dumps(out, allow_nan=False)                               # everything is plain JSON


def test_predict_payload_without_nisar_and_when_nisar_fails():
    seen = {}
    out = predict_payload({**BODY, "use_nisar": False}, predict_fn=fake_predict(seen),
                          nisar_factory=lambda: pytest.fail("NISAR must not be loaded"))
    assert seen["nisar"] is None and "nisar_passes_used" not in out["summary"]

    def not_installed():
        raise ImportError("No module named 'firedata'")
    out = predict_payload(BODY, predict_fn=fake_predict(seen), nisar_factory=not_installed)
    assert seen["nisar"] is None and "not installed" in out["summary"]["notes"][0]

    def broken(request):
        raise RuntimeError("Earthdata login failed")
    out = predict_payload(BODY, predict_fn=fake_predict(seen), nisar_factory=lambda: broken)
    assert out["summary"]["notes"][0] == "NISAR skipped: RuntimeError: Earthdata login failed"   # still predicts
    assert out["summary"]["n_cells"] > 0
    assert any("edge" in n for n in out["summary"]["notes"])        # 30 steps on an 11-cell grid reach the edge


def test_empty_map_says_why():
    from firespread.web import no_spread_reason
    ds = make_ds()
    lc = np.full((N, N), 80, np.int16)                              # water all round...
    lc[C - 1:C + 2, C + 1] = 20
    ds["landcover"] = (("y", "x"), lc)
    scar = np.zeros((N, N), np.int16)
    scar[C - 1:C + 2, C + 1] = 1                                    # ...except a burn scar to the east
    ds["s2_burned"] = (("y", "x"), scar)
    msg = no_spread_reason(spread_probabilities(ds))
    assert msg == ("nothing next to the polygon can catch fire: 62% built-up, bare, snow or water (ESA WorldCover), "
                   "38% burn scar (Sentinel-2)")
    wet = make_ds(temp=15.0, rh=100.0)                              # can burn, but very wet air
    wet_res = spread_probabilities(wet)
    msg = no_spread_reason(wet_res, min_prob=1.0)
    assert "every chance of spreading is below" in msg and "equilibrium moisture" in msg

    # Missing weather (emc_pct = NaN) or dry air must not print "equilibrium moisture nan%" or blame moist air
    no_wx = spread_probabilities(make_ds())
    assert "nan%" not in no_spread_reason(no_wx, min_prob=1.0)
    dry_res = spread_probabilities(make_ds(temp=35.0, rh=10.0))
    assert "keeps fuel moist" not in no_spread_reason(dry_res, min_prob=1.0)

    # After the simulation the map uses burn_prob, so the note must quote burn_prob too (not ignition_prob)
    sim = spread_probabilities(make_ds()).copy()
    sim["ignition_prob"] = sim["ignition_prob"] * 0 + 0.012          # one step: 1.2%
    sim["burn_prob"] = sim["ignition_prob"] * 0 + 0.005              # 200 runs: 0.5% -> nothing on the map
    assert "(highest 0.5%)" in no_spread_reason(sim)

    def fake(**kw):
        d = make_ds()
        d["landcover"] = (("y", "x"), np.full((N, N), 80, np.int16))
        return spread_probabilities(d)
    out = predict_payload({**BODY, "use_nisar": False}, predict_fn=fake)
    assert out["summary"]["n_cells"] == 0 and out["summary"]["notes"][-1].startswith("No spread on the map: nothing")


def test_cells_geojson_positions_and_empty():
    res = spread_probabilities(make_ds())
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(res.attrs["crs"], 4326, always_xy=True).transform(
        float(res.x[C + 1]), float(res.y[C]))                     # centre of the cell east of the fire
    hits = [f for f in cells_geojson(res)["features"]
            if min(p[0] for p in f["geometry"]["coordinates"][0]) < lon < max(p[0] for p in f["geometry"]["coordinates"][0])
            and min(p[1] for p in f["geometry"]["coordinates"][0]) < lat < max(p[1] for p in f["geometry"]["coordinates"][0])]
    assert len(hits) == 1 and hits[0]["properties"]["p"] == pytest.approx(float(res["ignition_prob"][C, C + 1]), abs=1e-4)
    nothing = spread_probabilities(make_ds(landcover=80))          # water all round: nothing can catch fire
    assert cells_geojson(nothing) == {"type": "FeatureCollection", "features": []}


def test_nisar_attrs_reach_the_result():
    from firespread import predict
    from test_alexandridis import nisar_on
    ds = make_ds()
    nisar = nisar_on(ds, nisar_burned=np.zeros((N, N), np.int16))
    nisar.attrs.update({"nisar_passes_used": "2026-08-04 A34", "nisar_coverage_pct": 100.0, "other": "x"})
    res = predict(inputs=ds, nisar=nisar)
    assert res.attrs["nisar_passes_used"] == "2026-08-04 A34" and res.attrs["nisar_coverage_pct"] == 100.0
    assert "other" not in res.attrs and "nisar_alignment" in res.attrs


def test_parse_request_and_payload_with_meteomatics_and_timeline():
    import xarray as xr

    req = parse_request({**BODY, "weather_source": "meteomatics", "hours": 12})
    assert req["weather_source"] == "meteomatics" and req["hours"] == 12.0
    with pytest.raises(ValueError, match="weather_source"):
        parse_request({**BODY, "weather_source": "invalid_source"})

    seen = {}

    def fake_forecast_predict(**kw):
        seen.update(kw)
        ds0 = make_ds(u=4.0, v=0.0, temp=29.0, rh=22.0)
        ds1 = make_ds(u=2.0, v=5.0, temp=26.0, rh=30.0).assign_coords(
            time=np.array(["2026-08-09T22:00"], dtype="datetime64[ns]")
        )
        ds = xr.concat([ds0, ds1], dim="time", data_vars="minimal")
        ds.attrs.update({"weather_source": "meteomatics", "weather_dataset": "Meteomatics API (mix)"})
        return spread_probabilities(ds)

    out = predict_payload(
        {**BODY, "weather_source": "meteomatics", "hours": 12, "use_nisar": False, "steps": 6},
        predict_fn=fake_forecast_predict,
    )
    assert seen["weather_source"] == "meteomatics" and seen["hours"] == 12.0
    summary = out["summary"]
    assert summary["weather_source"] == "meteomatics"
    assert summary["sim_time_developing"] == 1
    assert summary["sim_weather_steps"] == 2
    assert len(summary["weather_timeline"]) == 2
    assert summary["weather_timeline"][0]["wind_speed_ms"] == pytest.approx(4.0)
    json.dumps(out, allow_nan=False)

