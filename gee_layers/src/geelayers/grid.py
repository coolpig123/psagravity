"""A regular map grid in UTM (square cells, metres) that every layer is sampled onto.

All layers fetched for one Grid line up cell for cell, so the spread model can use
`ds["slope_deg"][i, j]` and `ds["wind_u_ms"][i, j]` for the same piece of ground.
Row 0 is the northern edge; column 0 is the western edge.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone that contains (lon, lat)."""
    zone = min(max(int((lon + 180) // 6) + 1, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def check_lonlat(lon, lat) -> None:
    """Raise a clear error for impossible coordinates (usually lat / lon swapped)."""
    lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()
            and (np.abs(lat) <= 90).all() and (np.abs(lon) <= 180).all()):
        raise ValueError("coordinates out of range (|lat| <= 90, |lon| <= 180): lat / lon swapped? "
                         "Points are (lat, lon); GeoJSON rings are [lon, lat].")


def _extract_geojson_polygons(obj: Any) -> list[list[np.ndarray]]:
    """Parse GeoJSON-like object into a list of polygons, each a list of (N, 2) [lon, lat] rings."""
    if hasattr(obj, "__geo_interface__"):
        obj = obj.__geo_interface__
    if isinstance(obj, (str, Path)):
        text = str(obj).strip()
        data = json.loads(text) if text.startswith(("{", "[")) else json.loads(Path(obj).read_text(encoding="utf-8"))
        return _extract_geojson_polygons(data)
    if isinstance(obj, dict):
        t = obj.get("type")
        if t == "FeatureCollection":
            out: list[list[np.ndarray]] = []
            for feat in obj.get("features", []):
                out.extend(_extract_geojson_polygons(feat))
            return out
        if t == "Feature":
            return _extract_geojson_polygons(obj.get("geometry"))
        if t == "Polygon":
            return [[np.asarray(ring, dtype=np.float64)[:, :2] for ring in obj["coordinates"] if len(ring) >= 3]]
        if t == "MultiPolygon":
            return [
                [np.asarray(ring, dtype=np.float64)[:, :2] for ring in poly if len(ring) >= 3]
                for poly in obj["coordinates"]
            ]
        raise ValueError(f"unsupported GeoJSON type: {t!r}")
    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
        return [[arr[:, :2]]]
    raise ValueError("polygon must be a sequence of (lon, lat) points, a GeoJSON dict/file, or have __geo_interface__")


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
        check_lonlat(lon, lat)
        epsg =epsg or utm_epsg(lon, lat)
        x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        r = radius_km * 1000.0
        return cls.from_bounds(x - r, y - r, x + r, y + r, res_m, epsg)

    @classmethod
    def from_polygon(cls, polygon: Any, buffer_km: float = 15.0,
                     res_m: float = 100.0, epsg: int | None = None) -> "Grid":
        """Grid covering a polygon (sequence of (lon, lat), GeoJSON dict/file, or __geo_interface__) plus buffer_km."""
        polys = _extract_geojson_polygons(polygon)
        pts = np.vstack([ring for poly in polys for ring in poly])
        lons, lats = pts[:, 0], pts[:, 1]
        check_lonlat(lons, lats)
        epsg = epsg or utm_epsg(float(lons.mean()), float(lats.mean()))
        xs, ys = Transformer.from_crs(4326, epsg, always_xy=True).transform(lons, lats)
        buf = buffer_km * 1000.0
        return cls.from_bounds(xs.min() - buf, ys.min() - buf, xs.max() + buf, ys.max() + buf, res_m, epsg)

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

    def bbox_lonlat(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) in degrees enclosing the grid, e.g. to search other archives
        (NISAR on ASF, FIRMS, ...) for exactly this area."""
        xmin, ymin, xmax, ymax = self.bounds
        n = 50
        xs = np.concatenate([np.linspace(xmin, xmax, n), np.full(n, xmax), np.linspace(xmax, xmin, n), np.full(n, xmin)])
        ys = np.concatenate([np.full(n, ymin), np.linspace(ymin, ymax, n), np.full(n, ymax), np.linspace(ymax, ymin, n)])
        lon, lat = Transformer.from_crs(self.epsg, 4326, always_xy=True).transform(xs, ys)
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())

    def cell_of(self, lat: float, lon: float) -> tuple[int, int]:
        """(row, col) of the cell containing a lon/lat point (may be outside the grid)."""
        x, y = Transformer.from_crs(4326, self.epsg, always_xy=True).transform(lon, lat)
        return int((self.y0 - y) // self.res), int((x - self.x0) // self.res)

    def _ring_mask(self, rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
        """Even-odd ray-cast mask (bool, shape (height, width)) for a single projected ring (rx, ry)."""
        if rx[0] != rx[-1] or ry[0] != ry[-1]:
            rx, ry = np.append(rx, rx[0]), np.append(ry, ry[0])
        x1, y1, x2, y2 = rx[:-1], ry[:-1], rx[1:], ry[1:]
        mask = np.zeros(self.shape, dtype=bool)
        gx, gy = self.x, self.y
        r_min = max(0, int((self.y0 - ry.max()) // self.res))
        r_max = min(self.height, int((self.y0 - ry.min()) // self.res) + 1)
        for r in range(r_min, r_max):
            yr = gy[r]
            cross = (y1 > yr) != (y2 > yr)
            if not cross.any():
                continue
            x_int = x1[cross] + (yr - y1[cross]) * (x2[cross] - x1[cross]) / (y2[cross] - y1[cross])
            c_idx = np.searchsorted(gx, x_int, side="right")
            counts = np.bincount(c_idx, minlength=self.width + 1)[:self.width]
            mask[r] = (np.cumsum(counts) & 1).astype(bool)
        return mask

    def polygon_mask(self, polygon: Any) -> np.ndarray:
        """Rasterize a polygon onto the grid as int16 (1 = inside/burning, 0 = outside)."""
        polys = _extract_geojson_polygons(polygon)
        tr = Transformer.from_crs(4326, self.epsg, always_xy=True)
        out = np.zeros(self.shape, dtype=bool)
        for rings in polys:
            pmask = np.zeros(self.shape, dtype=bool)
            outer_xy = None
            for idx, ring in enumerate(rings):
                rx, ry = tr.transform(ring[:, 0], ring[:, 1])
                rx, ry = np.asarray(rx, dtype=np.float64), np.asarray(ry, dtype=np.float64)
                if idx == 0:
                    outer_xy = (rx, ry)
                pmask ^= self._ring_mask(rx, ry)
            if not pmask.any() and outer_xy is not None:
                # Sub-cell polygon: ensure the cell containing its centroid is marked if inside grid bounds
                cx, cy = float(outer_xy[0].mean()), float(outer_xy[1].mean())
                r, c = int((self.y0 - cy) // self.res), int((cx - self.x0) // self.res)
                if 0 <= r < self.height and 0 <= c < self.width:
                    pmask[r, c] = True
            out |= pmask
        return out.astype(np.int16)

    def point_mask(self, lat: float, lon: float, radius_m: float = 0.0) -> np.ndarray:
        """Initial burn mask (int16: 1 = burning, 0 = unburned) around an ignition point."""
        check_lonlat(lon, lat)
        x, y =Transformer.from_crs(4326, self.epsg, always_xy=True).transform(lon, lat)
        mask = np.zeros(self.shape, dtype=bool)
        if radius_m > 0:
            xx, yy = np.meshgrid(self.x, self.y)
            mask = np.hypot(xx - x, yy - y) <= radius_m
        r, c = int((self.y0 - y) // self.res), int((x - self.x0) // self.res)
        if 0 <= r < self.height and 0 <= c < self.width:
            mask[r, c] = True
        return mask.astype(np.int16)

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
