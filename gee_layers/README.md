# gee_layers: model inputs from Google Earth Engine

The `geelayers` package fetches what the spread model needs to know about the ground and the air:
terrain (elevation, slope, aspect), land cover and fuel models, vegetation greenness and moisture,
fuel dryness and fire-danger indices, and hourly weather (temperature, humidity, wind, rain).

Every layer comes back as an `xarray.Dataset` on **one shared grid** (UTM, square cells, metres).
Cell `[i, j]` is the same piece of ground in every variable, so the cellular automaton can read
`ds["slope_deg"][i, j]` and `ds["wind_u_ms"][i, j]` without any reprojection.

## Setup (once)

```bash
cd gee_layers
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -e .
earthengine authenticate                            # opens a browser, Google login
cp .env.example .env                                # then set EE_PROJECT=<your Cloud project id>
pytest                                              # offline tests, no account needed
```

`EE_PROJECT` is a Google Cloud project registered for Earth Engine
(free for non-commercial use: https://code.earthengine.google.com/register).
For a server, set `EE_SERVICE_ACCOUNT_KEY` to a service-account JSON key instead of logging in.

From another part of the repo (e.g. the model code), install it into your own venv:
`pip install -e ../gee_layers`.

## Use it

```python
from geelayers import Grid, get_layers, get_weather, regrid

# 1. The model grid: 40 x 40 km around the Bug Fire ignition, 100 m cells
grid = Grid.from_point(lat=39.727395, lon=-120.0372941, radius_km=20, res_m=100)
# or Grid.from_bbox(west, south, east, north, res_m=100)

# 2. Everything the model needs at the start time (only data from BEFORE that time is used)
env = get_layers(grid, time="2026-08-08T22:30Z")
env["slope_deg"].values          # numpy (rows, cols), row 0 = north
env["fuel_model"].values         # int16 class codes, -1 = no data

# 3. Hourly weather while the fire runs. Fetch it coarse (fast), then interpolate to the model grid
wx = get_weather(grid.with_resolution(1000), "2026-08-08T22:00Z", "2026-08-13T00:00Z", source="rtma")
wx = regrid(wx, grid)            # dims (time, y, x)
wind_u_at_hour_5 = wx["wind_u_ms"].isel(time=5).values

# 4. Save / load
env.to_netcdf("data/bug_env.nc", engine="h5netcdf")    # gee_layers/data/ is not committed
```

Command line (same thing, writes NetCDF):

```bash
python -m geelayers list        # layers, sources, variables, units
python -m geelayers layers  --point 39.7274 -120.0373 --radius-km 20 --time 2026-08-08T22:30Z --out data/bug_env.nc
python -m geelayers weather --point 39.7274 -120.0373 --radius-km 20 --res 1000 \
    --start 2026-08-08T22:00Z --end 2026-08-13T00:00Z --out data/bug_weather.nc
```

## What you get

Output contract (all functions):

- dims `(y, x)` or `(time, y, x)`. Row 0 is north and column 0 is west.
- coords: `x`, `y` (cell centres in metres, CRS in `ds.attrs["crs"]`), `lat`, `lon` (2-D), and `time` (UTC, no timezone attached).
- variables are float32 with NaN for missing data. Categorical ones are int16 with -1 for missing.
- `ds.attrs`: `crs`, `res_m`, `transform` (rasterio-style affine), the datasets used, and the dates actually
  used (`fuel_dryness_day_utc`, `weather_time_utc`, `gfs_issued_utc`, ...).

| Layer (`get_layers`) | Dataset | Coverage | Variables |
|---|---|---|---|
| `terrain` | NASADEM 30 m | global | `elevation_m`, `slope_deg`, `aspect_deg` |
| `landcover` | ESA WorldCover 2021, 10 m | global | `landcover` (class) |
| `fuel_model` | LANDFIRE FBFM40 (LF 1.4.0, 2016), 30 m | US | `fuel_model` (Scott & Burgan 40) |
| `vegetation` | Sentinel-2 SR, 30-day median before `time` | global | `ndvi`, `ndmi` |
| `fuel_dryness` | gridMET daily, 4 km | US | `fm100_pct`, `fm1000_pct`, `erc`, `burning_index`, `vpd_kpa`, `rh_min_pct`, `tmax_c`, `precip_daily_mm` |
| `weather` | see sources below, latest hour at or before `time` | | as below |

| Weather source (`get_weather`) | Dataset | Coverage | Use for |
|---|---|---|---|
| `rtma` (default) | NOAA RTMA hourly analysis, 2.5 km | US | hindcasts, "now" |
| `era5_land` | ECMWF ERA5-Land hourly, ~11 km, a few days behind | global | Europe (Ávila, Le Porge) |
| `gfs` | NOAA GFS 0.25° forecast | global | live predictions (future hours) |

Every weather source gives `temp_c`, `rh_pct`, `wind_u_ms`, `wind_v_ms`, `wind_speed_ms`, `wind_dir_deg`.
`rtma` adds `gust_ms` and `precip_mm`. `era5_land` adds `precip_mm` and `soil_moisture`.
Switching source therefore needs no change in the model code.

Conventions worth knowing:
- `wind_dir_deg` is the direction the wind blows **from**, clockwise from north (meteorological).
  The fire is pushed toward `(wind_dir_deg + 180) % 360`; or use `wind_u_ms` / `wind_v_ms` (toward east / north).
- `slope_deg` / `aspect_deg` are computed from `elevation_m` at the grid's own cell size, which is
  the scale the automaton sees between neighbours. Aspect is the direction the slope faces.
- Units for every variable: `python -m geelayers list` or `geelayers.VARIABLES`.

## Adding a layer

Write a builder in [src/geelayers/layers.py](src/geelayers/layers.py) that returns an `ee.Image` whose bands
are named after your variables. Then register it in `LAYERS` and describe each variable in `VARIABLES` (a test
checks this). The download, tiling, grid and NetCDF output work without any other change.

## Limits

- Requests are split into tiles and band groups of at most 32 MB and run in parallel. A 40 x 40 km grid at
  100 m (400 x 400 cells) is a few requests, and a whole-fire box at 30 m is many but still works.
- `fuel_model` and `fuel_dryness` cover the US only. For Europe, use `layers=("terrain", "landcover",
  "vegetation", "weather")` with `weather_source="era5_land"`.
- Each dataset has its own latency: RTMA about 1 h, gridMET 1-2 days, ERA5-Land about 5 days. If data for a
  time does not exist yet, the call raises a `LookupError` that says so instead of returning stale data.
