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
    yield send
    httpd.shutdown()
    httpd.server_close()


def test_predict_endpoint(post):
    body = {"polygon": {"type": "Polygon", "coordinates": []}, "start": "2026-08-09T21:00Z"}
    status, out = post("/api/predict", body)
    assert status == 200 and out["summary"]["n_cells"] == 0 and out["bounds"] == [0, 0, 1, 1]
    assert post.calls == [body]


def test_predict_endpoint_errors(post):
    status, out = post("/api/predict", {"start": "2026-08-09T21:00Z"})
    assert status == 400 and "polygon" in out["error"]
    status, out = post("/api/predict", b"not json")
    assert status == 400
    status, out = post("/api/predict", {"polygon": {}, "start": "boom"})
    assert status == 500 and "Earth Engine is down" in out["error"]
    status, out = post("/api/nothing", {})
    assert status == 404
