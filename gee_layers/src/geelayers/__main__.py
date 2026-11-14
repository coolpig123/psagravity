"""Command line: fetch layers to NetCDF.

    python -m geelayers list
    python -m geelayers layers  --point 39.7274 -120.0373 --radius-km 20 --time 2026-08-08T22:30Z --out data/bug_env.nc
    python -m geelayers weather --point 39.7274 -120.0373 --radius-km 20 --res 1000 \
        --start 2026-08-08T22:00Z --end 2026-08-13T00:00Z --source rtma --out data/bug_weather.nc

Area: --point LAT LON [--radius-km R]  or  --bbox WEST SOUTH EAST NORTH.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .api import DEFAULT_LAYERS, describe, get_layers, get_weather
from .grid import Grid
from .layers import WEATHER


def _grid(a) -> Grid:
    if a.point:
        return Grid.from_point(a.point[0], a.point[1], a.radius_km, a.res)
    if a.bbox:
        return Grid.from_bbox(*a.bbox, res_m=a.res)
    raise SystemExit("give an area: --point LAT LON or --bbox WEST SOUTH EAST NORTH")


def _area_args(p: argparse.ArgumentParser, default_res: float) -> None:
    p.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"))
    p.add_argument("--radius-km", type=float, default=15.0)
    p.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--res", type=float, default=default_res, help="cell size in metres")
    p.add_argument("--out", type=Path, required=True, help="output NetCDF file")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m geelayers", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show layers, weather sources and variables")

    p = sub.add_parser("layers", help="all layers at one time, on one grid")
    _area_args(p, 100.0)
    p.add_argument("--time", help="UTC time the prediction starts, e.g. 2026-08-08T22:30Z")
    p.add_argument("--layers", default=",".join(DEFAULT_LAYERS))
    p.add_argument("--weather-source", default="rtma", choices=list(WEATHER))

    p = sub.add_parser("weather", help="hourly weather time series")
    _area_args(p, 1000.0)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--source", default="rtma", choices=list(WEATHER))
    p.add_argument("--step-hours", type=int, default=1)
    p.add_argument("--issued", help="gfs only: use the latest run issued at or before this time")

    a = ap.parse_args(argv)
    if a.cmd == "list":
        print(describe())
        return
    grid = _grid(a)
    print(f"grid {grid.crs} {grid.width}x{grid.height} cells of {grid.res:g} m")
    if a.cmd == "layers":
        ds = get_layers(grid, a.time, [s.strip() for s in a.layers.split(",") if s.strip()], a.weather_source)
    else:
        ds = get_weather(grid, a.start, a.end, a.source, a.step_hours, a.issued)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(a.out, engine="h5netcdf")
    print(f"wrote {a.out}: {', '.join(ds.data_vars)}")


if __name__ == "__main__":
    main()
