"""Find NISAR granules over an area and read only that area onto a model grid.

- search_granules(): ASF search by the area's lon/lat box (all frames, latest product version)
- group_passes():    granules of one acquisition (same track, a few minutes apart) = one pass
- read_layers():     open a granule remotely (HTTP range reads via earthaccess, no full download) or
                     locally, read the window around the grid, and put it on the grid:
                       finer source (GCOV 10 m -> 100 m): mean of all source pixels in each cell
                       coarser source (SME2 200 m -> 100 m): bilinear sample at cell centres
                     results are cached on disk per granule + grid
- read_pass():       read_layers over all frames of a pass, mosaicked

The grid is anything with crs ("EPSG:xxxx"), x, y (cell centres, metres; y north -> south), res_m:
the spread model's NisarRequest qualifies.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import h5py
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.ndimage import map_coordinates

from .config import REPO_ROOT

log = logging.getLogger(__name__)

ASF_SEARCH = "https://api.daac.asf.alaska.edu/services/search/param"
CACHE_DIR = REPO_ROOT / "data" / "nisar_cache" / "raw"     # data/**/raw is git-ignored
GCOV_GROUP = "/science/LSAR/GCOV/grids/frequencyA"
SME2_GROUP = "/science/LSAR/SME2/grids"

Prepare = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


# ---- search ----------------------------------------------------------------------------------

def search_granules(product: str, bbox_lonlat: tuple[float, float, float, float], start, end) -> pd.DataFrame:
    """Standard-production NISAR granules of `product` (GCOV, SME2, ...) intersecting the box, in [start, end)."""
    w, s, e, n = bbox_lonlat
    r = requests.get(ASF_SEARCH, params={
        "dataset": "NISAR", "processingLevel": product,
        "intersectsWith": f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))",
        "start": pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": pd.Timestamp(end).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output": "geojson", "maxResults": 2000}, timeout=300)
    r.raise_for_status()
    rows = []
    for feat in r.json().get("features", []):
        p = feat["properties"]
        fid = p.get("fileID") or ""
        if "_PR_" not in fid:          # urgent-response duplicates
            continue
        rows.append({"time_utc": pd.Timestamp(p["startTime"]).tz_convert("UTC") if pd.Timestamp(p["startTime"]).tzinfo
                     else pd.Timestamp(p["startTime"], tz="UTC"),
                     "track_id": (p.get("flightDirection") or "?")[0] + str(p.get("pathNumber")),
                     "frame": int(p.get("frameNumber") or 0), "file_id": fid, "url": p.get("url"),
                     "version": int(fid[-3:]) if fid[-3:].isdigit() else 0})
    cols = ["time_utc", "track_id", "frame", "file_id", "url", "version"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    # several versions of the same granule (_001, _002): keep the latest
    df["key"] = df.file_id.str[:-4]
    df = df.sort_values("version").drop_duplicates("key", keep="last").drop(columns="key")
    return df.sort_values("time_utc").reset_index(drop=True)


@dataclass
class Pass:
    track_id: str
    time_utc: pd.Timestamp
    granules: list[dict] = field(default_factory=list)   # rows of search_granules (one per frame)

    @property
    def file_ids(self) -> list[str]:
        return [g["file_id"] for g in self.granules]


def group_passes(df: pd.DataFrame, gap: pd.Timedelta = pd.Timedelta(minutes=5)) -> list[Pass]:
    """Frames of the same track acquired within `gap` form one pass. Sorted by time."""
    passes: list[Pass] = []
    for track, g in df.sort_values("time_utc").groupby("track_id"):
        cur: Pass | None = None
        for row in g.to_dict("records"):
            if cur is None or row["time_utc"] - cur.time_utc > gap:
                cur = Pass(track, row["time_utc"])
                passes.append(cur)
            cur.granules.append(row)
    return sorted(passes, key=lambda p: p.time_utc)


# ---- grid ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetGrid:
    crs: str
    x: np.ndarray       # cell centres, west -> east
    y: np.ndarray       # cell centres, north -> south
    res: float

    @classmethod
    def from_request(cls, request: Any) -> "TargetGrid":
        x, y = np.asarray(request.x, np.float64), np.asarray(request.y, np.float64)
        res = float(getattr(request, "res_m", None) or abs(x[1] - x[0]))
        return cls(str(request.crs), x, y, res)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.y), len(self.x)

    @property
    def x0(self) -> float:          # west edge
        return float(self.x[0] - self.res / 2)

    @property
    def y0(self) -> float:          # north edge
        return float(self.y[0] + self.res / 2)

    def key(self) -> str:
        return f"{self.crs}|{self.x0:.3f}|{self.y0:.3f}|{self.res:.3f}|{self.shape}"

    def edge_points(self, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Densified outline of the grid (with one cell margin), in the grid CRS."""
        xmin, xmax = self.x0 - self.res, self.x0 + (len(self.x) + 1) * self.res
        ymax, ymin = self.y0 + self.res, self.y0 - (len(self.y) + 1) * self.res
        xs = np.concatenate([np.linspace(xmin, xmax, n), np.full(n, xmax), np.linspace(xmax, xmin, n), np.full(n, xmin)])
        ys = np.concatenate([np.full(n, ymin), np.linspace(ymin, ymax, n), np.full(n, ymax), np.linspace(ymax, ymin, n)])
        return xs, ys


# ---- reading ---------------------------------------------------------------------------------

def granule_url(file_id: str) -> str:
    r = requests.get(ASF_SEARCH, params={"granule_list": file_id, "output": "geojson"}, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        raise FileNotFoundError(f"granule not found on ASF: {file_id}")
    return feats[0]["properties"]["url"]


READ_TIMEOUT_S = 120      # a byte-range request with no data for this long is abandoned (then retried)
READ_ATTEMPTS = 3


def _https_filesystem():
    """fsspec HTTPS filesystem with the Earthdata bearer token (as earthaccess builds it) plus timeouts.

    earthaccess.get_fsspec_https_session() has no timeout, so one stalled connection hangs forever.
    """
    import aiohttp
    import earthaccess
    import fsspec
    earthaccess.login(strategy="environment")
    token = earthaccess.__store__.auth.token["access_token"]
    return fsspec.filesystem("https", skip_instance_cache=True, client_kwargs={
        "headers": {"Authorization": f"Bearer {token}"},
        "trust_env": False,       # as earthaccess: env proxies + bearer token break the login
        "timeout": aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=READ_TIMEOUT_S)})


@contextlib.contextmanager
def open_granule(source: str | Path) -> Iterator[h5py.File]:
    """h5py.File for a local path, or for a URL read remotely by byte ranges (Earthdata login from .env)."""
    if Path(str(source)).exists():
        with h5py.File(source, "r") as f:
            yield f
        return
    with _https_filesystem().open(str(source), block_size=8 * 2**20) as fh, h5py.File(fh, "r") as f:
        yield f


def _epsg(group: h5py.Group) -> int:
    proj = group["projection"]
    code = proj.attrs.get("epsg_code")
    return int(code) if code is not None else int(proj[()])


def _window(coords: np.ndarray, lo: float, hi: float) -> slice:
    """Index slice of 1-D cell centres (ascending or descending) covering [lo, hi]."""
    if coords[-1] >= coords[0]:
        return slice(max(0, int(np.searchsorted(coords, lo)) - 1), int(np.searchsorted(coords, hi)) + 1)
    return slice(max(0, int(np.searchsorted(-coords, -hi)) - 1), int(np.searchsorted(-coords, -lo)) + 1)


def _fill_to_nan(a: np.ndarray, ds: h5py.Dataset) -> np.ndarray:
    a = a.astype(np.float32)
    fill = ds.attrs.get("_FillValue")
    if fill is not None and np.isfinite(fill):
        a[a == fill] = np.nan
    return a


def _mean_to_grid(arrays: dict[str, np.ndarray], sx: np.ndarray, sy: np.ndarray, src_epsg: int,
                  grid: TargetGrid, chunk_rows: int = 512) -> dict[str, np.ndarray]:
    """Mean of all source pixels (centres sx, sy) falling in each grid cell, NaN-aware."""
    tr = Transformer.from_crs(src_epsg, grid.crs, always_xy=True)
    h, w = grid.shape
    sums = {k: np.zeros(h * w) for k in arrays}
    counts = {k: np.zeros(h * w) for k in arrays}
    for r0 in range(0, len(sy), chunk_rows):
        xx, yy = np.meshgrid(sx, sy[r0:r0 + chunk_rows])
        tx, ty = tr.transform(xx, yy)
        col = np.floor((tx - grid.x0) / grid.res).astype(np.int64)
        row = np.floor((grid.y0 - ty) / grid.res).astype(np.int64)
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        idx = (row * w + col)[inside]
        for k, a in arrays.items():
            v = a[r0:r0 + chunk_rows][inside]
            ok = np.isfinite(v)
            sums[k] += np.bincount(idx[ok], weights=v[ok], minlength=h * w)
            counts[k] += np.bincount(idx[ok], minlength=h * w)
    out = {}
    for k in arrays:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.where(counts[k] > 0, sums[k] / counts[k], np.nan).reshape(h, w).astype(np.float32)
    return out


def _sample_to_grid(arrays: dict[str, np.ndarray], sx: np.ndarray, sy: np.ndarray, src_epsg: int,
                    grid: TargetGrid) -> dict[str, np.ndarray]:
    """Bilinear sample of a coarser source at the grid cell centres (NaN where any neighbour is NaN)."""
    tr = Transformer.from_crs(grid.crs, src_epsg, always_xy=True)
    xx, yy = np.meshgrid(grid.x, grid.y)
    fx, fy = tr.transform(xx, yy)
    cols = (fx - sx[0]) / (sx[1] - sx[0])
    rows = (fy - sy[0]) / (sy[1] - sy[0])
    return {k: map_coordinates(a, [rows, cols], order=1, cval=np.nan).astype(np.float32) for k, a in arrays.items()}


def read_layers(source: str | Path, group: str, datasets: list[str], grid: TargetGrid, *,
                prepare: Prepare | None = None, how: str = "auto", cache_dir: Path | None = CACHE_DIR,
                cache_key: str | None = None) -> dict[str, np.ndarray]:
    """Read `datasets` (paths relative to `group`) around `grid` and put them on it.

    prepare: applied to the native-resolution window first (fill values are already NaN), e.g. to
             apply quality masks; returns the arrays to regrid (may rename / drop / derive).
    how:     "mean" (source finer than the grid), "sample" (coarser), or "auto".
    Returns {name: (ny, nx) float32}; cells outside the granule are NaN.
    """
    key = None
    if cache_dir is not None:
        ident = json.dumps([cache_key or Path(str(source)).name, group, datasets, grid.key(), how,
                            getattr(prepare, "__name__", None)])
        key = Path(cache_dir) / (hashlib.sha1(ident.encode()).hexdigest()[:20] + ".npz")
        if key.exists():
            with np.load(key) as z:
                return {k: z[k] for k in z.files}

    for attempt in range(1, READ_ATTEMPTS + 1):
        try:
            out = _read_window(source, group, datasets, grid, prepare, how)
            break
        except Exception as e:          # stalled / dropped connection: retry; anything else: raise
            if Path(str(source)).exists() or not _is_network_error(e) or attempt == READ_ATTEMPTS:
                raise
            log.warning("NISAR read of %s failed (%s), retry %d/%d", Path(str(source)).name, e, attempt,
                        READ_ATTEMPTS - 1)
    if key is not None:
        key.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(key, **out)
    return out


def _is_network_error(e: BaseException) -> bool:
    try:
        import aiohttp
        if isinstance(e, aiohttp.ClientError):
            return True
    except ImportError:
        pass
    return isinstance(e, (TimeoutError, ConnectionError, OSError))


def _read_window(source, group: str, datasets: list[str], grid: TargetGrid, prepare: Prepare | None,
                 how: str) -> dict[str, np.ndarray]:
    with open_granule(source) as f:
        g = f[group]
        epsg = _epsg(g)
        x, y = g["xCoordinates"][:].astype(np.float64), g["yCoordinates"][:].astype(np.float64)
        ex, ey = grid.edge_points()
        bx, by = Transformer.from_crs(grid.crs, epsg, always_xy=True).transform(ex, ey)
        cs, rs = _window(x, bx.min(), bx.max()), _window(y, by.min(), by.max())
        if cs.stop - cs.start < 2 or rs.stop - rs.start < 2:      # grid outside this granule
            dummy = {k: np.full((2, 2), np.nan, np.float32) for k in datasets}
            names = (prepare(dummy) if prepare is not None else dummy).keys()
            out = {k: np.full(grid.shape, np.nan, np.float32) for k in names}
        else:
            arrays = {k: _fill_to_nan(g[k][rs, cs], g[k]) for k in datasets}
            if prepare is not None:
                arrays = prepare(arrays)
            sx, sy = x[cs], y[rs]
            src_res = abs(float(sx[1] - sx[0]))
            mode = how if how != "auto" else ("mean" if src_res * 1.5 <= grid.res else "sample")
            out = (_mean_to_grid if mode == "mean" else _sample_to_grid)(arrays, sx, sy, epsg, grid)
    return out


def read_pass(p: Pass, group: str, datasets: list[str], grid: TargetGrid, *, source_of: Callable[[dict], str] | None = None,
              **kwargs) -> dict[str, np.ndarray]:
    """read_layers over every frame of a pass; the first frame with data wins in each cell."""
    source_of = source_of or (lambda g: g.get("url") or granule_url(g["file_id"]))
    out: dict[str, np.ndarray] | None = None
    for gr in p.granules:
        part = read_layers(source_of(gr), group, datasets, grid, cache_key=gr["file_id"], **kwargs)
        if out is None:
            out = part
        else:
            for k, v in part.items():
                out[k] = np.where(np.isnan(out[k]), v, out[k])
    if out is None:
        raise ValueError("pass has no granules")
    return out
