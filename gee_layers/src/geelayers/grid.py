"""A regular map grid in UTM (square cells, metres) that every layer is sampled onto.

All layers fetched for one Grid line up cell for cell, so the spread model can use
`ds["slope_deg"][i, j]` and `ds["wind_u_ms"][i, j]` for the same piece of ground.
Row 0 is the northern edge; column 0 is the western edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone that contains (lon, lat)."""
    zone = min(max(int((lon + 180) // 6) + 1, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


@dataclass(frozen=True)
class Grid:
    epsg: int       # projected CRS, metres (UTM by default)
    x0: float       # western edge of column 0
    y0: float       # northern edge of row 0
    res: float      # cell size in metres
    width: int      # columns
    height: int     # rows

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def from_bounds(cls, xmin: float, ymin: float, xmax: float, ymax: float, res_m: float, epsg: int) -> "Grid":
        """Smallest grid with cells snapped to multiples of res_m that covers the box (in epsg metres).

        Snapping means grids of different resolutions over the same area share edges
        (e.g. a 100 m grid nests inside a 1000 m one).
        """
        x0, x1 = math.floor(xmin / res_m) * res_m, math.ceil(xmax / res_m) * res_m
        y0, y1 = math.floor(ymin / res_m) * res_m, math.ceil(ymax / res_m) * res_m
        return cls(epsg, x0, y1, float(res_m), max(1, round((x1 - x0) / res_m)), max(1, round((y1 - y0) / res_m)))

    @classmethod
    def from_bbox(cls, west: float, south: float, east: float, north: float,
                  res_m: float = 100.0, epsg: int | None = None) -> "Grid":
        """Grid covering a lon/lat box (degrees)."""
        epsg = epsg or utm_epsg((west + east) / 2, (south + north) / 2)
        # densify the edges: a lon/lat box is curved once projected
        n = 50
        lons = np.concatenate([np.linspace(west, east, n), np.full(n, east), np.linspace(east, west, n), np.full(n, west)])
        lats = np.concatenate([np.full(n, south), np.linspace(south, north, n), np.full(n, north), np.linspace(north, south, n)])
        xs, ys = Transformer.from_crs(4326, epsg, always_xy=True).transform(lons, lats)
        return cls.from_bounds(xs.min(), ys.min(), xs.max(), ys.max(), res_m, epsg)

    @classmethod
    def from_point(cls, lat: float, lon: float, radius_km: float = 15.0,
                   res_m: float = 100.0, epsg: int | None = None) -> "Grid":
        """Square grid centred on a point (e.g. an ignition), radius_km from centre to each edge."""
        epsg = epsg or utm_epsg(lon, lat)
        x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        r = radius_km * 1000.0
        return cls.from_bounds(x - r, y - r, x + r, y + r, res_m, epsg)

    def with_resolution(self, res_m: float) -> "Grid":
        """Same area at another cell size (e.g. a coarse grid for weather)."""
        return Grid.from_bounds(*self.bounds, res_m, self.epsg)

    # ---- geometry ---------------------------------------------------------------------
    @property
    def crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """xmin, ymin, xmax, ymax in metres."""
        return self.x0, self.y0 - self.height * self.res, self.x0 + self.width * self.res, self.y0

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """Affine transform (a, b, c, d, e, f) as used by rasterio: x = a*col + b*row + c, y = d*col + e*row + f."""
        return self.res, 0.0, self.x0, 0.0, -self.res, self.y0

    @property
    def x(self) -> np.ndarray:
        """Cell-centre x of each column (metres)."""
        return self.x0 + self.res * (np.arange(self.width) + 0.5)

    @property
    def y(self) -> np.ndarray:
        """Cell-centre y of each row (metres), north to south."""
        return self.y0 - self.res * (np.arange(self.height) + 0.5)

    def latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """2-D arrays (lat, lon) of every cell centre."""
        xx, yy = np.meshgrid(self.x, self.y)
        lon, lat = Transformer.from_crs(self.epsg, 4326, always_xy=True).transform(xx, yy)
        return lat, lon

    def cell_of(self, lat: float, lon: float) -> tuple[int, int]:
        """(row, col) of the cell containing a lon/lat point (may be outside the grid)."""
        x, y = Transformer.from_crs(4326, self.epsg, always_xy=True).transform(lon, lat)
        return int((self.y0 - y) // self.res), int((x - self.x0) // self.res)

    # ---- Earth Engine helpers ----------------------------------------------------------
    def tiles(self, max_side: int) -> Iterator[tuple[int, int, "Grid"]]:
        """Split into sub-grids of at most max_side x max_side cells: (row0, col0, sub_grid)."""
        for r0 in range(0, self.height, max_side):
            for c0 in range(0, self.width, max_side):
                yield r0, c0, Grid(self.epsg, self.x0 + c0 * self.res, self.y0 - r0 * self.res, self.res,
                                   min(max_side, self.width - c0), min(max_side, self.height - r0))

    def ee_grid(self) -> dict:
        """PixelGrid for ee.data.computePixels."""
        return {"dimensions": {"width": self.width, "height": self.height},
                "affineTransform": {"scaleX": self.res, "shearX": 0, "translateX": self.x0,
                                    "shearY": 0, "scaleY": -self.res, "translateY": self.y0},
                "crsCode": self.crs}
