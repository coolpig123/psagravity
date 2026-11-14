"""Small API server for the frontend.

Run from the repo root:  python backend/server.py   (listens on http://localhost:8000)
It runs itself inside backend/.venv (pip install -r gee_layers/requirements.txt -e gee_layers), which has gee_layers.

POST /api/polygon   body: GeoJSON Feature with a Polygon geometry ([lon, lat] order).
                    Prints the polygon's points, runs gee_layers.get_fire_inputs on it
                    and prints a short summary of the result.
                    Optional properties: "start" (ISO time, UTC) and "hours" (default: see below).
"""

import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VENV = Path(__file__).resolve().parent / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# Re-run inside backend/.venv if started with another Python (e.g. plain `python` from npm run dev).
if __name__ == "__main__" and VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV.resolve():
    sys.exit(subprocess.call([str(VENV_PYTHON), __file__, *sys.argv[1:]]))

import numpy as np  # noqa: E402  (after the venv switch)
import pandas as pd  # noqa: E402
from geelayers import get_fire_inputs  # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
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


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
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
    print(f"API server on http://localhost:{PORT}  (Ctrl+C to stop)", flush=True)
    try:
        ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
