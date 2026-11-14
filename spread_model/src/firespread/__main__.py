"""Command line.

    python -m firespread --polygon fire.geojson --start 2026-08-09T21:00Z --out data/bug_spread.nc
    python -m firespread --point 39.7274 -120.0373 --start 2026-08-08T22:30Z --out data/bug_spread.nc
    python -m firespread --inputs ../gee_layers/data/bug_bundle.nc --out data/bug_spread.nc
    python -m firespread --polygon fire.geojson --start 2026-08-09T21:00Z --nisar auto     # live NISAR from ASF
    python -m firespread --polygon fire.geojson --start 2026-08-09T21:00Z --nisar nisar.nc # saved NISAR layers

--nisar auto reads NISAR over the fire area with firedata.nisar_provider (pip install -e ../real_fires,
Earthdata login in real_fires/.env).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from .alexandridis import summary
from .predict import predict


def _nisar_source(value: str | None, lookback_days: float):
    """--nisar value -> what predict(nisar=...) takes: None, a provider function, or a loaded Dataset."""
    if not value:
        return None
    if value.lower() == "auto":
        try:
            from firedata.nisar_provider import nisar_provider
        except ImportError as e:
            raise SystemExit("--nisar auto needs the NISAR reader from real_fires: pip install -e ../real_fires "
                             f"(and Earthdata login in real_fires/.env). Import failed: {e}")
        return nisar_provider(lookback_days=lookback_days)
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"--nisar: no such file {path} (use 'auto' for live NISAR)")
    return xr.load_dataset(path, engine="h5netcdf")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m firespread", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--polygon", help="GeoJSON file / string: the area burning at --start")
    ap.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"), help="ignition point")
    ap.add_argument("--inputs", type=Path, help="existing geelayers NetCDF (skips fetching)")
    ap.add_argument("--nisar", help="'auto' = live NISAR from ASF over the fire area (firedata.nisar_provider), "
                                    "or a NetCDF with NISAR layers (crs attr + x/y coords; see README)")
    ap.add_argument("--nisar-lookback-days", type=float, default=30.0,
                    help="with --nisar auto: use passes at most this many days before --start")
    ap.add_argument("--start", help="UTC time, e.g. 2026-08-09T21:00Z")
    ap.add_argument("--radius-km", type=float, default=5.0)
    ap.add_argument("--res", type=float, default=100.0)
    ap.add_argument("--out", type=Path, help="output NetCDF")
    a = ap.parse_args(argv)

    inputs = xr.load_dataset(a.inputs, engine="h5netcdf") if a.inputs else None
    nisar = _nisar_source(a.nisar, a.nisar_lookback_days)
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
