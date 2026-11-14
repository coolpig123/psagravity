"""Quantities computed locally from fetched layers (no Earth Engine needed, easy to test)."""
from __future__ import annotations

import numpy as np


def slope_aspect(elevation: np.ndarray, res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (degrees clockwise from north, the direction the slope faces).

    Computed at the grid's own cell size, which is the scale the spread model sees between
    neighbouring cells. Rows run north to south, columns west to east. Aspect is NaN on flat cells.
    """
    dz_drow, dz_dx = np.gradient(elevation.astype(np.float64), res_m)
    dz_dnorth = -dz_drow                      # row index grows southward
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dnorth)))
    # downhill vector is (-dz_dx east, -dz_dnorth north); bearing = atan2(east, north)
    aspect = np.degrees(np.arctan2(-dz_dx, -dz_dnorth)) % 360
    aspect[np.hypot(dz_dx, dz_dnorth) < 1e-9] = np.nan
    return slope.astype(np.float32), aspect.astype(np.float32)


def wind_speed_dir(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wind speed (m/s) and meteorological direction (degrees the wind blows FROM, clockwise from north).

    u > 0 blows toward the east, v > 0 toward the north. A west wind (u > 0, v = 0) gives 270.
    The fire is pushed toward (dir + 180) % 360.
    """
    speed = np.hypot(u, v)
    direction = np.degrees(np.arctan2(-u, -v)) % 360
    return speed.astype(np.float32), direction.astype(np.float32)
