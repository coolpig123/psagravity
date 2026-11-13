"""GOES ABI L2 Fire Detection and Characterization (FDC) from NOAA's public AWS buckets.

Product ABI-L2-FDCC (CONUS sector, 5 min, 2 km at nadir). Files are streamed anonymously
and only fire pixels inside the bbox are kept (one parquet per day in raw/goes/), so the
~85 MB/day of NetCDF is never stored.

Note: FDC geolocation is not terrain-corrected (parallax). Use a few-km tolerance when
matching GOES pixels to a fire.
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import s3fs

PRODUCT = "ABI-L2-FDCC"
# Fire mask categories (Mask flag values). 10-15 = fire, 30-35 = temporally filtered fire.
FIRE_CODES = {
    10: "good", 11: "saturated", 12: "cloud_contaminated", 13: "high_prob", 14: "medium_prob", 15: "low_prob",
    30: "good", 31: "saturated", 32: "cloud_contaminated", 33: "high_prob", 34: "medium_prob", 35: "low_prob",
}
# Categories counted in FRP totals (low-probability pixels excluded as noisy).
FRP_CODES = [10, 11, 12, 13, 14, 30, 31, 32, 33, 34]
# Schema of the per-day pixel files (used for days with no fire pixels at all).
PIXEL_DTYPES = {"time_utc": "datetime64[ns, UTC]", "lat": "float64", "lon": "float64", "frp_mw": "float64",
                "fire_temp_k": "float64", "fire_area_m2": "float64", "mask": "int16", "dqf": "int16"}


def _fs():
    return s3fs.S3FileSystem(anon=True)


def list_files(satellite: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[str]:
    """`satellite` like 'goes18'. Lists files whose hour directory overlaps [start, end)."""
    fs = _fs()
    files = []
    for hour in pd.date_range(start_utc.floor("h"), end_utc, freq="h", inclusive="left"):
        prefix = f"noaa-{satellite}/{PRODUCT}/{hour:%Y}/{hour.dayofyear:03d}/{hour:%H}/"
        try:
            files += fs.ls(prefix)
        except FileNotFoundError:
            print(f"  ! missing GOES hour {prefix}")
    return sorted(files)


def _scan_start(path: str) -> pd.Timestamp:
    s = path.split("_s")[1][:13]          # sYYYYJJJHHMMSSs
    return pd.to_datetime(s[:13], format="%Y%j%H%M%S", utc=True)


def _scaled(f: h5py.File, name: str) -> np.ndarray:
    v = f[name]
    return v[:] * v.attrs["scale_factor"][0] + v.attrs["add_offset"][0]


def _latlon_grid(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre lat/lon of the fixed grid."""
    return _project(f, *np.meshgrid(_scaled(f, "x"), _scaled(f, "y")))


def _project(f: h5py.File, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-grid scan angles (x, y radians) -> lat/lon, GOES-R PUG vol. 3 sec. 5.1.2.8."""
    p = f["goes_imager_projection"].attrs
    H = float(p["perspective_point_height"][0]) + float(p["semi_major_axis"][0])
    a, b = float(p["semi_major_axis"][0]), float(p["semi_minor_axis"][0])
    lon0 = np.deg2rad(float(p["longitude_of_projection_origin"][0]))
    with np.errstate(invalid="ignore"):
        A = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (a / b) ** 2 * np.sin(y) ** 2)
        B = -2 * H * np.cos(x) * np.cos(y)
        C = H ** 2 - a ** 2
        rs = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        sx, sy, sz = rs * np.cos(x) * np.cos(y), -rs * np.sin(x), rs * np.cos(x) * np.sin(y)
        lat = np.degrees(np.arctan((a / b) ** 2 * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
        lon = np.degrees(lon0 - np.arctan(sy / (H - sx)))
    return lat, lon


class _Subsetter:
    """Caches the bbox window of the fixed grid (the CONUS grid is constant per satellite)."""

    def __init__(self, bbox):
        self.bbox, self.window, self.lat, self.lon = bbox, None, None, None

    def init(self, f: h5py.File):
        lat, lon = _latlon_grid(f)
        w, s, e, n = self.bbox
        inside = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
        rows, cols = np.where(inside)
        if rows.size == 0:
            raise ValueError("bbox is outside the GOES CONUS sector")
        self.window = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        self.lat, self.lon = lat[self.window], lon[self.window]

    def extract(self, f: h5py.File, scan_start: pd.Timestamp) -> pd.DataFrame:
        if self.window is None:
            self.init(f)
        mask = f["Mask"][self.window]
        fire = np.isin(mask, list(FIRE_CODES))
        if not fire.any():
            return pd.DataFrame()
        def var(name):
            v = f[name]
            raw = v[self.window][fire].astype("float64")
            fill = v.attrs.get("_FillValue")
            if fill is not None:
                raw[raw == fill[0]] = np.nan
            sf = v.attrs.get("scale_factor", [1.0])[0]
            off = v.attrs.get("add_offset", [0.0])[0]
            return raw * sf + off
        w, s, e, n = self.bbox
        df = pd.DataFrame({
            "time_utc": scan_start, "lat": self.lat[fire], "lon": self.lon[fire],
            "frp_mw": var("Power"), "fire_temp_k": var("Temp"), "fire_area_m2": var("Area"),
            "mask": mask[fire], "dqf": f["DQF"][self.window][fire],
        })
        return df[(df.lat >= s) & (df.lat <= n) & (df.lon >= w) & (df.lon <= e)]


def _read_one(fs, path, sub: _Subsetter) -> tuple[pd.Timestamp, pd.DataFrame]:
    t = _scan_start(path)
    data = fs.cat_file(path)  # one GET per file; h5py on a remote file handle is much slower
    with h5py.File(io.BytesIO(data), "r") as f:
        return t, sub.extract(f, t)


def download_goes(bbox, start_utc, end_utc, raw_dir: Path, satellite: str = "goes18",
                  workers: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fire pixels in bbox for every scan in [start, end).

    Returns (pixels, scans): `scans` lists every scan processed, so scans with zero fire
    pixels (FRP = 0, or cloud) are distinguishable from missing data.
    """
    raw_dir = Path(raw_dir) / "goes"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fs = _fs()
    sub = _Subsetter(bbox)
    pix_all, scans_all = [], []
    for day in pd.date_range(start_utc.normalize(), end_utc, freq="D", inclusive="left"):
        pix_path = raw_dir / f"{satellite}_fdcc_{day:%Y%m%d}.parquet"
        scan_path = raw_dir / f"{satellite}_fdcc_{day:%Y%m%d}_scans.parquet"
        if not (pix_path.exists() and scan_path.exists()):
            d0, d1 = max(day, start_utc), min(day + pd.Timedelta(days=1), end_utc)
            files = [p for p in list_files(satellite, d0, d1) if d0 <= _scan_start(p) < d1]
            if sub.window is None and files:
                _read_one(fs, files[0], sub)
            with ThreadPoolExecutor(workers) as ex:
                res = list(ex.map(lambda p: _read_one(fs, p, sub), files))
            found = [d for _, d in res if not d.empty]
            pix = (pd.concat(found, ignore_index=True) if found else
                   pd.DataFrame({c: pd.Series(dtype=t) for c, t in PIXEL_DTYPES.items()}))
            scans = pd.DataFrame({"time_utc": [t for t, _ in res], "file": files})
            pix.to_parquet(pix_path)
            scans.to_parquet(scan_path)
            print(f"  GOES {day:%F}: {len(files)} scans, {len(pix)} fire pixels")
        pix_all.append(pd.read_parquet(pix_path))
        scans_all.append(pd.read_parquet(scan_path))
    pix = pd.concat(pix_all, ignore_index=True)
    scans = pd.concat(scans_all, ignore_index=True)
    return pix, scans


def pixel_footprints(points: pd.DataFrame, satellite: str = "goes18",
                     when: pd.Timestamp | None = None) -> "gpd.GeoSeries":
    """Ground footprint polygon (EPSG:4326) of each GOES fire pixel, from the fixed-grid
    pixel corners. Pixels near 40N, 120W are ~2.4 x 3.6 km, larger than the nominal 2 km.
    Not terrain-corrected (no parallax correction)."""
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely.geometry import Polygon

    fs = _fs()
    # The CONUS fixed grid is constant, so any file of the period gives the geometry.
    t = when if when is not None else (points.time_utc.min() if "time_utc" in points else None)
    if t is None:
        raise ValueError("pass `when` (any time in the data period) or points with a time_utc column")
    path = next(p for p in list_files(satellite, t.floor("h"), t.floor("h") + pd.Timedelta(hours=1)))
    with h5py.File(io.BytesIO(fs.cat_file(path)), "r") as f:
        x, y = _scaled(f, "x"), _scaled(f, "y")
        lat_c, lon_c = _latlon_grid(f)
        ok = np.isfinite(lat_c)
        rows, cols = np.where(ok & (np.abs(lat_c - points.lat.mean()) < 2) & (np.abs(lon_c - points.lon.mean()) < 2))
        tree = cKDTree(np.c_[lat_c[rows, cols], lon_c[rows, cols]])
        _, idx = tree.query(np.c_[points.lat, points.lon])
        r, c = rows[idx], cols[idx]
        dx, dy = (x[1] - x[0]) / 2, (y[1] - y[0]) / 2
        corners = [_project(f, x[c] + sx * dx, y[r] + sy * dy) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    polys = [Polygon([(corners[k][1][i], corners[k][0][i]) for k in range(4)]) for i in range(len(points))]
    return gpd.GeoSeries(polys, index=points.index, crs=4326)


def tidy_goes(pix: pd.DataFrame, satellite: str = "goes18") -> pd.DataFrame:
    out = pix.copy()
    out["source"] = f"GOES{satellite[-2:]}_FDC"
    out["confidence"] = out["mask"].map(FIRE_CODES)
    out["in_frp_total"] = out["mask"].isin(FRP_CODES)
    return out.sort_values("time_utc").reset_index(drop=True)
