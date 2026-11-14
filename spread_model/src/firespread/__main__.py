"""Command line.

    python -m firespread --polygon fire.geojson --start 2026-08-09T21:00Z --out data/bug_spread.nc
    python -m firespread --point 39.7274 -120.0373 --start 2026-08-08T22:30Z --out data/bug_spread.nc
    python -m firespread --inputs ../gee_layers/data/bug_bundle.nc --out data/bug_spread.nc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from .alexandridis import summary
from .predict import predict


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m firespread", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--polygon", help="GeoJSON file / string: the area burning at --start")
    ap.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"), help="ignition point")
    ap.add_argument("--inputs", type=Path, help="existing geelayers NetCDF (skips fetching)")
    ap.add_argument("--nisar", type=Path, help="NetCDF with NISAR layers on the same grid (optional)")
    ap.add_argument("--start", help="UTC time, e.g. 2026-08-09T21:00Z")
    ap.add_argument("--radius-km", type=float, default=5.0)
    ap.add_argument("--res", type=float, default=100.0)
    ap.add_argument("--out", type=Path, help="output NetCDF")
    a = ap.parse_args(argv)

    inputs = xr.load_dataset(a.inputs, engine="h5netcdf") if a.inputs else None
    nisar = xr.load_dataset(a.nisar, engine="h5netcdf") if a.nisar else None
    if inputs is None and not a.start:
        raise SystemExit("--start is required unless --inputs is given")
    result = predict(polygon=a.polygon, point=tuple(a.point) if a.point else None, start=a.start,
                     inputs=inputs, nisar=nisar, radius_km=a.radius_km, res_m=a.res)
    print(summary(result))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(a.out, engine="h5netcdf")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
