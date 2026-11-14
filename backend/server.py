"""Small API server for the frontend.

Run from the repo root:  python backend/server.py   (listens on http://localhost:8000)
It runs itself inside backend/.venv (pip install -r gee_layers/requirements.txt -e gee_layers), which has gee_layers.

POST /api/polygon   body: GeoJSON Feature with a Polygon geometry ([lon, lat] order).
                    Prints the polygon's points, runs gee_layers.get_fire_inputs on it
                    and prints a short summary of the result.
                    Optional properties: "start" (ISO time, UTC) and "hours" (default: see below).

POST /api/predict   body: {"polygon": GeoJSON, "start": "2026-08-09T21:00Z", "radius_km": 5, "res_m": 100,
                    "use_nisar": true, "burn_scars": true}
                    Runs the spread model (firespread.predict) on the polygon + radius_km only and returns the
                    ignition-probability cells as GeoJSON plus a summary; see spread_model/src/firespread/web.py.
                    Needs spread_model (and real_fires for NISAR) in the venv; see frontend/README.md.
"""

import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_PY = "Scripts/python.exe" if os.name == "nt" else "bin/python"
_ROOT = Path(__file__).resolve().parents[1]
# backend/.venv, or real_fires/.venv (which has gee_layers, the spread model and the NISAR reader) if there is none
VENV = next((v for v in (_ROOT / "backend" / ".venv", _ROOT / "real_fires" / ".venv") if (v / _PY).exists()),
            _ROOT / "backend" / ".venv")
VENV_PYTHON = VENV / _PY

# Re-run inside that venv if started with another Python (e.g. plain `python` from npm run dev).
if __name__ == "__main__" and VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV.resolve():
    sys.exit(subprocess.call([str(VENV_PYTHON), __file__, *sys.argv[1:]]))

import numpy as np  # noqa: E402  (after the venv switch)
import pandas as pd  # noqa: E402
from geelayers import get_fire_inputs  # noqa: E402

# The spread model and the NISAR reader, straight from the repo when they are not pip-installed in this venv.
for _pkg in ("spread_model", "real_fires"):
    sys.path.append(str(_ROOT / _pkg / "src"))

DEFAULT_PORT = 8000
# The UI has no time picker yet: default to a recent window whose data already exists on Earth Engine
# (RTMA weather can be ~13 h late, gridMET 2-3 days; gridMET looks back up to 7 days).
DEFAULT_LAG_HOURS = 48
DEFAULT_HOURS = 3


def print_polygon(feature: dict) -> None:
    ring = feature["geometry"]["coordinates"][0]
    points = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring  # drop the closing point
    area = feature.get("properties", {}).get("area_km2")
    header = f"Polygon with {len(points)} points"
    if area is not None:
        header += f", area {area:,.1f} km²"
    print(f"\n{header}")
    print(f"{'#':>3}  {'lat':>11}  {'lon':>12}")
    for i, (lon, lat) in enumerate(points, 1):
        print(f"{i:>3}  {lat:>11.6f}  {lon:>12.6f}")
    print("GeoJSON:", json.dumps(feature["geometry"]), flush=True)


def print_summary(ds, seconds: float) -> None:
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    res = ds.attrs["res_m"]
    burning = int(ds["initial_burned"].values.sum())
    print(f"\nget_fire_inputs done in {seconds:.0f} s")
    print(f"  grid      {ny} x {nx} cells at {res:g} m ({ny * res / 1000:.1f} x {nx * res / 1000:.1f} km), {ds.attrs['crs']}")
    print(f"  time      {ds.attrs['time_utc']} -> {ds.attrs['end_utc']}, {ds.sizes.get('time', 0)} weather hours")
    print(f"  burning   {burning} cells ({burning * res * res / 1e6:,.1f} km²) at start")
    print(f"  {'variable':<16} {'units':<8} {'min':>9} {'mean':>9} {'max':>9}  missing")
    for name, var in ds.data_vars.items():
        if name == "initial_burned":
            continue
        v = var.values.astype(np.float64)
        if np.issubdtype(var.dtype, np.integer):
            v = np.where(v < 0, np.nan, v)  # -1 = no data for categorical layers
        ok = np.isfinite(v)
        missing = f"{100 * (1 - ok.mean()):.0f}%"
        if not ok.any():
            print(f"  {name:<16} {var.attrs.get('units', ''):<8} {'-':>9} {'-':>9} {'-':>9}  {missing}")
            continue
        print(f"  {name:<16} {var.attrs.get('units', ''):<8} {np.nanmin(v):>9.2f} {np.nanmean(v):>9.2f} "
              f"{np.nanmax(v):>9.2f}  {missing}")
    sys.stdout.flush()


def run_fire_inputs(feature: dict):
    props = feature.get("properties") or {}
    start = props.get("start") or (pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=DEFAULT_LAG_HOURS))
    hours = float(props.get("hours") or DEFAULT_HOURS)
    print(f"\nRunning get_fire_inputs(polygon=..., start={pd.Timestamp(start).isoformat()}, hours={hours:g}) ...",
          flush=True)
    t0 = time.time()
    ds = get_fire_inputs(polygon=feature, start=start, hours=hours)
    print_summary(ds, time.time() - t0)
    return ds


def run_predict(body: dict) -> dict:
    """POST /api/predict: the spread model on the drawn polygon. ValueError = bad request."""
    from firespread.web import predict_payload   # imported here so /api/polygon works without the spread model
    print(f"\nRunning predict(start={body.get('start')}, radius_km={body.get('radius_km', 5)}, "
          f"use_nisar={body.get('use_nisar', True)}, burn_scars={body.get('burn_scars', True)}) ...", flush=True)
    t0 = time.time()
    out = predict_payload(body)
    print(f"predict done in {time.time() - t0:.0f} s: {out['summary']['n_cells']} cells on the map")
    print(out["summary"]["text"])
    for note in out["summary"]["notes"]:
        print(f"NOTE: {note}")
    sys.stdout.flush()
    return out


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path == "/api/predict":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._reply(200, run_predict(body))
            except ValueError as e:      # bad JSON, missing polygon / start, numbers out of range
                self._reply(400, {"error": f"bad request: {e}"})
            except Exception as e:       # Earth Engine login, data not available yet, area too large, ...
                print(f"\npredict failed: {type(e).__name__}: {e}", flush=True)
                self._reply(500, {"error": f"predict failed: {type(e).__name__}: {e}"})
            return
        if self.path != "/api/polygon":
            self._reply(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if body["geometry"]["type"] != "Polygon":
                raise ValueError("geometry must be a Polygon")
            print_polygon(body)
        except (ValueError, KeyError, TypeError, IndexError) as e:
            self._reply(400, {"error": f"bad polygon: {e}"})
            return
        try:
            run_fire_inputs(body)
        except Exception as e:  # Earth Engine login, data not available yet, area too large, ...
            print(f"\nget_fire_inputs failed: {type(e).__name__}: {e}", flush=True)
            self._reply(500, {"error": f"get_fire_inputs failed: {e}"})
            return
        self._reply(200, {"ok": True})

    def _reply(self, status: int, data: dict) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # keep the terminal for the polygons only
        pass


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # so "km²" prints on Windows terminals
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    print(f"API server on http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    try:
        ThreadingHTTPServer(("localhost", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
