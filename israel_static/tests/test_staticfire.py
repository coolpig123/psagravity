"""Offline tests of the static site's Python (israel_static/browser), on the committed data in frontend/public/haifa.

    backend\\.venv\\Scripts\\python -m pytest israel_static/tests/test_staticfire.py

No login and no network: exactly what the browser runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "frontend" / "public" / "haifa"
sys.path[:0] = [str(ROOT / "israel_static" / "browser"), str(ROOT / "spread_model" / "src"),
                str(ROOT / "gee_layers" / "src")]

if not (DATA / "index.json").exists():
    pytest.skip("frontend/public/haifa has no data (run build_haifa_data.py)", allow_module_level=True)

real_xgboost = sys.modules.pop("xgboost", None)     # israel_static/browser/xgboost.py must be the one imported
import staticfire  # noqa: E402
import xgboost as shim  # noqa: E402

assert shim.__version__.endswith("shim")
staticfire.set_data_dir(DATA)
INDEX = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
START = INDEX["default_start_utc"]
RING = [[35.03, 32.72], [35.045, 32.72], [35.045, 32.732], [35.03, 32.732], [35.03, 32.72]]   # on the Carmel
POLYGON = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [RING]}}]}


def call(path: str, body: dict) -> tuple[int, dict]:
    status, _, text = staticfire.handle(path, json.dumps(body)).partition("\n")
    return int(status), json.loads(text)


@pytest.fixture(scope="module")
def prediction() -> dict:
    status, out = call("/api/predict", {"polygon": POLYGON, "start": START, "weather_source": "auto", "hours": 6})
    assert status == 200, out
    return out


def test_predict_gives_cells_and_arrival_times(prediction):
    s = prediction["summary"]
    cells = prediction["cells"]["features"]
    assert len(cells) == s["n_map_cells"] > 100
    assert s["arrival"] and s["arrival_horizons_h"] == [6, 12, 24, 48]
    assert 0 < s["arrival_area_6h_km2"] <= s["arrival_area_12h_km2"] <= s["arrival_area_48h_km2"]
    assert s["weather_source"] == "era5_land" and s["sim_weather_steps"] == 6
    props = cells[0]["properties"]
    assert {"p", "pe", "t50", "p6", "p48"} <= set(props)
    w, south, e, n = prediction["bounds"]
    assert INDEX["grid_bbox_lonlat"][0] <= w < e <= INDEX["grid_bbox_lonlat"][2]
    assert INDEX["grid_bbox_lonlat"][1] <= south < n <= INDEX["grid_bbox_lonlat"][3]
    json.dumps(prediction, allow_nan=False)      # real JSON: no NaN


def test_predict_uses_nisar_and_burn_scars(prediction):
    s = prediction["summary"]
    assert s["nisar_passes_used"] and s["nisar_coverage_pct"] > 50
    assert "s2_clear_pct" in s and "s2_burned_pct" in s


def test_files_for_lists_only_what_the_request_needs():
    files = json.loads(staticfire.files_for(json.dumps({"polygon": POLYGON, "start": START, "hours": 6})))
    month = START[:7]
    assert "static.npz" in files and f"weather/era5_land/{month}.npz" in files
    assert sum(f.startswith("weather/") for f in files) == 1
    assert sum(f.startswith("nisar/") for f in files) == 1 and sum(f.startswith("s2/") for f in files) == 1
    assert all((DATA / f).exists() for f in files)


@pytest.mark.parametrize("source, used", [("gfs", "gfs"), ("era5_land", "era5_land"), ("meteomatics", "era5_land"),
                                           ("rtma", "era5_land")])
def test_weather_sources(source, used):
    status, out = call("/api/predict", {"polygon": POLYGON, "start": START, "weather_source": source, "hours": 3,
                                        "arrival": False})
    assert status == 200, out
    assert out["summary"]["weather_source"] == used
    notes = " ".join(out["summary"]["notes"])
    if source == "meteomatics":
        assert "Meteomatics needs an account" in notes
    if source == "rtma":
        assert "United States only" in notes


def test_auto_falls_back_to_gfs_after_the_reanalysis_ends():
    gfs = INDEX["weather"]["gfs"]
    status, out = call("/api/predict", {"polygon": POLYGON, "start": gfs["last_run_utc"], "hours": 6, "arrival": False})
    assert status == 200, out
    assert out["summary"]["weather_source"] == "gfs"


def test_start_outside_the_downloaded_period_is_a_clear_404():
    status, out = call("/api/predict", {"polygon": POLYGON, "start": "2020-01-01T12:00Z"})
    assert status == 404 and "no downloaded weather" in out["error"] and "2026-06-01" in out["error"]


def test_polygon_outside_the_area_is_a_bad_request():
    far = [[lon + 1.0, lat] for lon, lat in RING]
    status, out = call("/api/predict", {"polygon": {"type": "Polygon", "coordinates": [far]}, "start": START})
    assert status == 400 and "outside the downloaded area" in out["error"]


def test_bad_requests():
    assert call("/api/predict", {"start": START})[0] == 400
    assert call("/api/predict", {"polygon": POLYGON, "start": "not a time"})[0] == 400
    assert call("/api/nothing", {})[0] == 404


def test_grid_near_the_edge_is_cut_to_the_downloaded_area():
    w, s, e, n = INDEX["bbox"]
    ring = [[w + 0.005, s + 0.3], [w + 0.015, s + 0.3], [w + 0.015, s + 0.31], [w + 0.005, s + 0.31],
            [w + 0.005, s + 0.3]]
    status, out = call("/api/predict", {"polygon": {"type": "Polygon", "coordinates": [ring]}, "start": START,
                                        "arrival": False})
    assert status == 200, out
    assert any("reaches past the downloaded area" in note for note in out["summary"]["notes"])


def test_extract_data_table():
    feature = {"type": "Feature", "properties": {"start": START},
               "geometry": {"type": "Polygon", "coordinates": [RING]}}
    status, out = call("/api/polygon", feature)
    assert status == 200 and out["ok"], out
    for word in ("elevation_m", "landcover", "wind_speed_ms", "nisar_cover_pct", "s2_burned", "3 weather hours"):
        assert word in out["text"]


def test_inputs_match_the_geelayers_contract():
    ds = staticfire.fire_inputs(POLYGON, START, hours=4)
    assert ds.sizes["time"] == 4 and ds.attrs["crs"] == "EPSG:32636" and ds.attrs["res_m"] == 100.0
    assert ds["elevation_m"].dims == ("y", "x") and ds["temp_c"].dims == ("time", "y", "x")
    assert float(ds.y[0]) > float(ds.y[-1]) and float(ds.x[0]) < float(ds.x[-1])       # row 0 north, column 0 west
    assert int(ds["initial_burned"].sum()) > 0
    assert (ds["fuel_lifeform"].values == -1).all()                                    # LANDFIRE is US only
    assert np.isfinite(ds["wind_u_ms"].values).all() and np.isfinite(ds["rh_pct"].values).all()
    # the grid is polygon + 5 km, snapped to 100 m: 5 km of cells on each side of the polygon
    assert 100 <= ds.sizes["x"] <= 125 and 100 <= ds.sizes["y"] <= 125


def test_xgboost_stand_in_matches_xgboost():
    """The numpy tree evaluator gives the margins of the real xgboost on the repo's arrival model."""
    if real_xgboost is None:
        try:
            del sys.modules["xgboost"]
            sys.path.remove(str(ROOT / "israel_static" / "browser"))
            import xgboost as real
        except ImportError:
            pytest.skip("xgboost is not installed")
        finally:
            sys.path.insert(0, str(ROOT / "israel_static" / "browser"))
            sys.modules["xgboost"] = shim
    else:
        real = real_xgboost
    from firespread.survival import FEATURES, MODEL_PATH
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, len(FEATURES))).astype(np.float32) * [50, 5000, 200, 200, 5, 0.3, 50][:len(FEATURES)]
    X[rng.random(X.shape) < 0.05] = np.nan
    base = rng.normal(size=500)
    for file in (MODEL_PATH, MODEL_PATH.with_suffix(".cure.json")):
        if not file.exists():
            continue
        a, b = shim.Booster(), real.Booster()
        a.load_model(str(file))
        b.load_model(str(file))
        got = a.predict(shim.DMatrix(X, feature_names=list(FEATURES), base_margin=base), output_margin=True)
        want = b.predict(real.DMatrix(X, feature_names=list(FEATURES), base_margin=base), output_margin=True)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
