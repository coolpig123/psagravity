"""Command line: fetch layers to NetCDF.

    python -m geelayers list
    python -m geelayers layers  --point 39.7274 -120.0373 --radius-km 20 --time 2026-08-08T22:30Z --out data/bug_env.nc
    python -m geelayers weather --point 39.7274 -120.0373 --radius-km 20 --res 1000 \
        --start 2026-08-08T22:00Z --end 2026-08-13T00:00Z --source rtma --out data/bug_weather.nc
    python -m geelayers bundle  --polygon fire.geojson --radius-km 15 \
        --start 2026-08-08T22:30Z --hours 24 --out data/bug_bundle.nc

Area: --polygon FILE_OR_GEOJSON [--radius-km R]  or  --point LAT LON [--radius-km R]  or  --bbox WEST SOUTH EAST NORTH.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .api import DEFAULT_LAYERS, describe, get_fire_inputs, get_layers, get_weather
from .grid import Grid
from .layers import WEATHER


def _grid(a) -> Grid:
    if getattr(a, "polygon", None):
        return Grid.from_polygon(a.polygon, a.radius_km, a.res)
    if a.point:
        return Grid.from_point(a.point[0], a.point[1], a.radius_km, a.res)
    if a.bbox:
        return Grid.from_bbox(*a.bbox, res_m=a.res)
    raise SystemExit("give an area: --polygon FILE_OR_GEOJSON, --point LAT LON, or --bbox WEST SOUTH EAST NORTH")


def _area_args(p: argparse.ArgumentParser, default_res: float) -> None:
    p.add_argument("--polygon", help="GeoJSON file path or inline GeoJSON string of initial fire perimeter")
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
    p.add_argument("--weather-source", default="rtma", choices=[*WEATHER, "auto"])

    p = sub.add_parser("weather", help="hourly weather time series")
    _area_args(p, 1000.0)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--source", default="rtma", choices=[*WEATHER, "auto"])
    p.add_argument("--step-hours", type=int, default=1)
    p.add_argument("--issued", help="gfs only: use the latest run issued at or before this time")

    p = sub.add_parser("bundle", help="single aligned dataset: static layers + regridded hourly weather + initial burn mask")
    _area_args(p, 100.0)
    p.add_argument("--start", required=True, help="UTC time the prediction starts, e.g. 2026-08-08T22:30Z")
    p.add_argument("--end", help="UTC end time (default: start + --hours)")
    p.add_argument("--hours", type=float, default=24.0, help="hours of weather if --end is omitted")
    p.add_argument("--ignition-radius-m", type=float, default=0.0, help="initial fire radius (m) when using --point")
    p.add_argument("--weather-res", type=float, default=1000.0, help="coarse cell size (m) for weather before regridding")
    p.add_argument("--step-hours", type=int, default=1)
    p.add_argument("--layers", default=",".join(DEFAULT_LAYERS))
    p.add_argument("--weather-source", default="rtma", choices=[*WEATHER, "auto"])
    p.add_argument("--issued", help="gfs only: use the latest run issued at or before this time")
    p.add_argument("--perimeter-time-key", help="property dating each feature of a perimeter time series "
                                                "(default: auto, e.g. obs_time_utc for FEDS)")

    a = ap.parse_args(argv)
    if a.cmd == "list":
        print(describe())
        return
    if a.cmd == "bundle":
        # the grid is built inside get_fire_inputs, around the perimeter actually used at --start
        ds = get_fire_inputs(
            None, a.start, a.end, hours=a.hours, polygon=a.polygon,
            point=tuple(a.point) if a.point else None, ignition_radius_m=a.ignition_radius_m,
            bbox=tuple(a.bbox) if a.bbox else None, radius_km=a.radius_km, res_m=a.res,
            weather_res_m=a.weather_res, step_hours=a.step_hours,
            layers=[s.strip() for s in a.layers.split(",") if s.strip()],
            weather_source=a.weather_source, issued=a.issued, perimeter_time_key=a.perimeter_time_key,
        )
        print(f"grid {ds.attrs['crs']} {ds.sizes['x']}x{ds.sizes['y']} cells of {ds.attrs['res_m']:g} m"
              + (f"; {ds.attrs['initial_perimeter']}" if "initial_perimeter" in ds.attrs else ""))
    else:
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
