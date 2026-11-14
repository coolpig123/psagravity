"""Build the NISAR inputs NetCDF (.nc) of a study fire for the spread model.

Uses firedata.nisar_provider: picks the latest NISAR passes before the chosen time automatically (any track,
older passes fill swath gaps), reads only the area around the fire from ASF (no 7 GB downloads), and writes
the spread-model variables:
  - nisar_cover_pct     (%)     : vegetation cover -> density (P_den)
  - nisar_soil_moisture (m3/m3) : SME2 soil moisture -> dryness (P_dry)
  - nisar_vwc_kg_m2     (kg/m2) : SME2 vegetation water content (P_vwc)
  - nisar_burned        (0/1, -1 = no data): burn scars, water, bare rock -> can't burn

Usage (from real_fires/, Earthdata login in .env):
    python scripts/build_nisar_inputs.py bug_fire                        # as of the ignition
    python scripts/build_nisar_inputs.py bug_fire --phase post           # as of the end of the data window
    python scripts/build_nisar_inputs.py bug_fire --time 2026-08-16T00:00Z --radius-km 25
Then:  python -m firespread --polygon ... --start ... --nisar <out.nc>   (or --nisar auto to skip the file)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running without installing firedata / geelayers
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "real_fires" / "src"))
sys.path.insert(0, str(repo_root / "gee_layers" / "src"))

from firedata.config import FIRES  # noqa: E402
from firedata.nisar_provider import nisar_provider  # noqa: E402


class _Request:
    """The request a provider gets from the spread model (firespread.NisarRequest), built from a geelayers Grid."""
    def __init__(self, grid, time_utc: pd.Timestamp):
        self.crs, self.x, self.y, self.res_m = grid.crs, grid.x, grid.y, grid.res
        self.bbox_lonlat, self.time_utc = grid.bbox_lonlat(), time_utc.isoformat()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build NISAR NetCDF inputs for the fire spread model.")
    ap.add_argument("fire", choices=list(FIRES), help="Fire name, e.g. bug_fire")
    ap.add_argument("--phase", default="pre", choices=["pre", "post"],
                    help="pre = NISAR as of the ignition; post = as of the end of the fire's data window")
    ap.add_argument("--time", help="UTC time to use instead of --phase (latest passes before it)")
    ap.add_argument("--res", type=float, default=100.0, help="Grid resolution in metres (default: 100 m)")
    ap.add_argument("--radius-km", type=float, default=20.0, help="Grid radius around the ignition (default: 20 km)")
    ap.add_argument("--lookback-days", type=float, default=30.0, help="Ignore passes older than this (default: 30)")
    ap.add_argument("--out", default=None, help="Output NetCDF file path (.nc)")
    args = ap.parse_args()

    from geelayers import Grid
    F = FIRES[args.fire]
    t = pd.Timestamp(args.time) if args.time else (F.ignition_utc if args.phase == "pre" else F.end_utc)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    grid = Grid.from_point(F.ignition_lat, F.ignition_lon, radius_km=args.radius_km, res_m=args.res)
    out_nc = Path(args.out) if args.out else F.root / "processed" / f"{F.slug}_nisar_{t:%Y%m%dT%H%M}.nc"

    print(f"Building NISAR inputs for {F.name} as of {t} ({grid.width}x{grid.height} cells of {grid.res:g} m)...")
    ds = nisar_provider(lookback_days=args.lookback_days)(_Request(grid, t))
    if ds is None:
        raise SystemExit(f"No NISAR pass covers this area in the {args.lookback_days:g} days before {t}.")
    lat, lon = grid.latlon()
    ds = ds.assign_coords(lat=(("y", "x"), lat.astype("float32")), lon=(("y", "x"), lon.astype("float32")))
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_nc, engine="h5netcdf")

    print(f"\nPasses used: {ds.attrs['nisar_passes_used']}")
    print(f"Coverage: {ds.attrs['nisar_coverage_pct']:.0f}% of cells")
    print(ds)
    print(f"\nSaved NetCDF file -> {out_nc}")
    print(f"Pass to the model as:  predict(..., nisar='{out_nc}')  or  python -m firespread ... --nisar {out_nc}")


if __name__ == "__main__":
    main()
