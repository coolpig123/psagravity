"""Offline test of backend/server.py: a real HTTP request to POST /api/predict, with the model mocked.

Run from the repo root in a venv that has gee_layers and spread_model:  python -m pytest backend/tests
"""
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("backend_server", Path(__file__).resolve().parents[1] / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


@pytest.fixture
def post(monkeypatch):
    calls = []

    def fake_run_predict(body):
        calls.append(body)
        if "polygon" not in body:
            raise ValueError("polygon: a GeoJSON Polygon, Feature or FeatureCollection is required")
        if body.get("start") == "boom":
            raise RuntimeError("Earth Engine is down")
        return {"cells": {"type": "FeatureCollection", "features": []}, "bounds": [0, 0, 1, 1],
                "summary": {"n_cells": 0, "notes": [], "text": ""}}
    monkeypatch.setattr(server, "run_predict", fake_run_predict)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def send(path, payload):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    send.calls = calls
    send.port = httpd.server_address[1]
    yield send
    httpd.shutdown()
    httpd.server_close()


def test_predict_endpoint(post):
    body = {"polygon": {"type": "Polygon", "coordinates": []}, "start": "2026-08-09T21:00Z"}
    status, out = post("/api/predict", body)
    assert status == 200 and out["summary"]["n_cells"] == 0 and out["bounds"] == [0, 0, 1, 1]
    assert post.calls == [body]


def test_browser_visit_points_to_the_website(post):
    port = post.port
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as r:
        assert r.status == 200 and "localhost:5173" in r.read().decode()


def test_predict_endpoint_errors(post):
    status, out = post("/api/predict", {"start": "2026-08-09T21:00Z"})
    assert status == 400 and "polygon" in out["error"]
    status, out = post("/api/predict", b"not json")
    assert status == 400
    status, out = post("/api/predict", {"polygon": {}, "start": "boom"})
    assert status == 500 and "Earth Engine is down" in out["error"]
    status, out = post("/api/nothing", {})
    assert status == 404



def test_wind_view_bbox():
    # snapped out to 5 deg tiles, so small pans and zooms reuse the same data
    assert server.wind_view_bbox([-120.3, 39.5, -119.8, 39.9]) == (-125.0, 35.0, -115.0, 40.0)
    assert server.wind_view_bbox([-121, 37, -118, 41]) == (-125.0, 35.0, -115.0, 45.0)
    assert server.wind_view_bbox([-150, 0, -100, 10]) == (-150.0, 0.0, -100.0, 10.0)        # any size
    assert server.wind_view_bbox([-540, -95, 540, 95]) == (-180.0, -90.0, 180.0, 90.0)      # world copies
    assert server.wind_view_bbox([170, -20, 190, -10]) == (-180.0, -20.0, 180.0, -10.0)     # across the date line
    for bad in (None, [1, 2, 3], ["a", 0, 1, 1], [10, 0, -10, 1], [0, 5, 1, 5]):
        with pytest.raises(ValueError):
            server.wind_view_bbox(bad)


def test_wind_endpoint(post, monkeypatch):
    """/api/wind reads the map view (snapped to tiles) at the start time via firedata.merra2.wind_grid (mocked)."""
    merra2 = pytest.importorskip("firedata.merra2")
    seen = []

    def fake_wind_grid(bbox, time):
        seen.append((bbox, str(time)))
        if time.year == 2030:
            raise LookupError("MERRA-2 M2T1NXSLV: no granule (not published yet?)")
        return {"time_utc": "2026-08-09T21:30:00+00:00", "lat": [39.5, 40.0], "lon": [-120.625, -120.0],
                "u": [[1, 1], [1, 1]], "v": [[0, 0], [0, 0]], "mean_speed_ms": 1.0, "mean_dir_deg": 270.0,
                "stride": 1}
    monkeypatch.setattr(merra2, "wind_grid", fake_wind_grid)

    view = [-120.3, 39.5, -119.8, 39.9]
    status, out = post("/api/wind", {"bbox": view, "start": "2026-08-09T21:00Z"})
    assert status == 200 and out["mean_dir_deg"] == 270.0 and out["bbox"] == [-125.0, 35.0, -115.0, 40.0]
    assert seen == [((-125.0, 35.0, -115.0, 40.0), "2026-08-09 21:00:00+00:00")]
    status, out = post("/api/wind", {"bbox": view, "start": "2030-01-01T00:00Z"})
    assert status == 404 and out["no_data"] is True
    status, out = post("/api/wind", {"bbox": view})
    assert status == 400 and "start" in out["error"]
    status, out = post("/api/wind", {"start": "2026-08-09T21:00Z"})
    assert status == 400 and "bbox" in out["error"]
