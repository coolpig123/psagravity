"""geelayers: environmental layers for fire-spread modelling, from Google Earth Engine.

    from geelayers import Grid, get_layers, get_weather, regrid

    grid = Grid.from_point(lat=39.7274, lon=-120.0373, radius_km=20, res_m=100)
    env = get_layers(grid, time="2026-08-08T22:30Z")          # terrain, fuels, dryness, weather at t
    wx = get_weather(grid.with_resolution(1000), "2026-08-08T22:00Z", "2026-08-13T00:00Z")
    wx_fine = regrid(wx, grid)                                 # hourly weather on the model grid

See gee_layers/README.md for the variable list and units.
"""
from .api import DEFAULT_LAYERS, describe, get_layers, get_weather, regrid
from .auth import init
from .grid import Grid, utm_epsg
from .layers import LAYERS, VARIABLES, WEATHER

__all__ = ["Grid", "utm_epsg", "init", "get_layers", "get_weather", "regrid", "describe",
           "DEFAULT_LAYERS", "LAYERS", "WEATHER", "VARIABLES"]
