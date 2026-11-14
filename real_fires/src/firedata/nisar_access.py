"""Find NISAR granules over an area and read only that area onto a model grid.

- search_granules(): ASF search by the area's lon/lat box (all frames, latest product version)
- group_passes():    granules of one acquisition (same track, a few minutes apart) = one pass
- read_layers():     read the window around the grid from a granule, locally or remotely (no full download),
                     and put it on the grid:
                       finer source (GCOV 10 m -> 100 m): mean of all source pixels in each cell
                       coarser source (SME2 200 m -> 100 m): bilinear sample at cell centres
                     results are cached on disk per granule + grid
                     Remote reads fetch all HDF5 chunks of the window in parallel (one batch of HTTP range
                     requests) and decode them locally (deflate + shuffle). The raw chunks and the granule's
                     layout are cached per granule, so another grid over the same area reads nothing remotely.
- read_pass():       read_layers over all frames of a pass, mosaicked

The grid is anything with crs ("EPSG:xxxx"), x, y (cell centres, metres; y north -> south), res_m:
the spread model's NisarRequest qualifies.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import re
import time
import zlib
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


@functools.lru_cache(maxsize=1)
def _https_filesystem():
    """fsspec HTTPS filesystem with the Earthdata bearer token (as earthaccess builds it) plus timeouts.

    Built once per process (one Earthdata login); read_layers drops it after a network error so the retry
    starts on a fresh session. earthaccess.get_fsspec_https_session() has no timeout, so one stalled
    connection would hang forever.
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


@functools.lru_cache(maxsize=1)
def _plain_filesystem():
    """fsspec HTTPS filesystem without credentials, for the signed download URLs ASF redirects to."""
    import aiohttp
    import fsspec
    return fsspec.filesystem("https", skip_instance_cache=True, client_kwargs={
        "trust_env": False, "timeout": aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=READ_TIMEOUT_S)})


SIGNED_URL_MAX_AGE_S = 30 * 60      # re-resolve after this (the signed URLs ASF hands out expire)
_signed: dict[str, tuple[float, str, int]] = {}


def _remote(url: str) -> tuple[Any, str, int | None]:
    """(filesystem, URL to read, size) for a remote granule.

    Every request to the ASF URL is redirected (with an Earthdata check) to a signed CloudFront URL, which
    makes each small read take ~1.4 s instead of ~0.6 s. So the redirect is followed once per granule and the
    signed URL is read directly; if that does not work, the ASF URL is read with the bearer token.
    """
    hit = _signed.get(url)
    if hit is not None and time.time() - hit[0] < SIGNED_URL_MAX_AGE_S:
        return _plain_filesystem(), hit[1], hit[2]
    fs = _https_filesystem()                      # logs in once per process
    try:
        import earthaccess
        token = earthaccess.__store__.auth.token["access_token"]
        r = requests.get(url, headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-0"},
                         allow_redirects=False, timeout=60)
        if r.status_code in (301, 302, 303, 307, 308):
            signed = r.headers["Location"]
            r = requests.get(signed, headers={"Range": "bytes=0-0"}, timeout=60)
            if r.status_code == 206:
                size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                _signed[url] = (time.time(), signed, size)
                return _plain_filesystem(), signed, size
    except (requests.RequestException, KeyError, ValueError, AttributeError, TypeError) as e:
        log.debug("could not resolve the signed URL of %s (%s); reading via ASF", url, e)
    return fs, url, None


def _reset_remote() -> None:
    """Fresh sessions and signed URLs (before retrying a failed read)."""
    _https_filesystem.cache_clear()
    _plain_filesystem.cache_clear()
    _signed.clear()


@contextlib.contextmanager
def open_granule(source: str | Path) -> Iterator[h5py.File]:
    """h5py.File for a local path, or for a URL read remotely by byte ranges (Earthdata login from .env)."""
    if Path(str(source)).exists():
        with h5py.File(source, "r") as f:
            yield f
        return
    fs, url, size = _remote(str(source))
    # Through the file object h5py only reads metadata (chunk data comes from _remote_window). Metadata is
    # scattered small reads that revisit the same places, so keep the last 64 x 1 MB blocks (the default
    # read-ahead cache keeps one block and re-downloads on every jump: 5x slower here).
    with fs.open(url, block_size=2**20, cache_type="blockcache", cache_options={"maxblocks": 64},
                 **({"size": size} if size else {})) as fh, h5py.File(fh, "r") as f:
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

    chunk_dir = None
    if cache_dir is not None:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", cache_key or Path(str(source)).name)
        chunk_dir = Path(cache_dir) / "chunks" / name
    for attempt in range(1, READ_ATTEMPTS + 1):
        try:
            out = _read_window(source, group, datasets, grid, prepare, how, chunk_dir)
            break
        except Exception as e:          # stalled / dropped connection: retry; anything else: raise
            if Path(str(source)).exists() or not _is_network_error(e) or attempt == READ_ATTEMPTS:
                raise
            log.warning("NISAR read of %s failed (%s), retry %d/%d", Path(str(source)).name, e, attempt,
                        READ_ATTEMPTS - 1)
            _reset_remote()                     # fresh sessions, token and signed URLs for the retry
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


def _grid_window(x: np.ndarray, y: np.ndarray, epsg: int, grid: TargetGrid) -> tuple[slice, slice] | None:
    """(row slice, col slice) of the source covering the grid, or None if the grid is outside the source."""
    ex, ey = grid.edge_points()
    bx, by = Transformer.from_crs(grid.crs, epsg, always_xy=True).transform(ex, ey)
    cs, rs = _window(x, bx.min(), bx.max()), _window(y, by.min(), by.max())
    if cs.stop - cs.start < 2 or rs.stop - rs.start < 2:
        return None
    return rs, cs


def _read_window(source, group: str, datasets: list[str], grid: TargetGrid, prepare: Prepare | None,
                 how: str, chunk_dir: Path | None = None) -> dict[str, np.ndarray]:
    if Path(str(source)).exists():                      # local file: h5py reads the window directly
        with open_granule(source) as f:
            g = f[group]
            epsg = _epsg(g)
            x, y = g["xCoordinates"][:].astype(np.float64), g["yCoordinates"][:].astype(np.float64)
            win = _grid_window(x, y, epsg, grid)
            arrays = None if win is None else {k: _fill_to_nan(g[k][win], g[k]) for k in datasets}
    else:
        got = _remote_window(str(source), group, datasets, grid, chunk_dir)
        arrays, x, y, epsg, win = got if got is not None else (None, None, None, None, None)
    if arrays is None:                                  # grid outside this granule
        dummy = {k: np.full((2, 2), np.nan, np.float32) for k in datasets}
        names = (prepare(dummy) if prepare is not None else dummy).keys()
        return {k: np.full(grid.shape, np.nan, np.float32) for k in names}
    if prepare is not None:
        arrays = prepare(arrays)
    rs, cs = win
    sx, sy = x[cs], y[rs]
    src_res = abs(float(sx[1] - sx[0]))
    mode = how if how != "auto" else ("mean" if src_res * 1.5 <= grid.res else "sample")
    return (_mean_to_grid if mode == "mean" else _sample_to_grid)(arrays, sx, sy, epsg, grid)


# ---- remote reads: all chunks of the window in parallel, cached per granule -----------------------

_DEFLATE, _SHUFFLE = 1, 2              # HDF5 filter ids (h5py.h5z.FILTER_DEFLATE / FILTER_SHUFFLE)
_UNALLOCATED = 0xFFFFFFFF              # chunk never written: the dataset's fill value


def _coords(d: h5py.Dataset) -> list[float]:
    """First and last value of a regular coordinate axis (2 small reads instead of the whole axis)."""
    return [float(d[0]), float(d[-1]), int(d.shape[0])]


def _dataset_meta(d: h5py.Dataset) -> dict:
    plist = d.id.get_create_plist()
    fill = d.attrs.get("_FillValue")
    return {"dtype": d.dtype.str, "shape": list(d.shape), "chunks": list(d.chunks) if d.chunks else None,
            "filters": [int(plist.get_filter(i)[0]) for i in range(plist.get_nfilters())],
            "fillvalue": float(d.fillvalue) if np.isscalar(d.fillvalue) else 0.0,
            "_FillValue": None if fill is None else float(np.asarray(fill).ravel()[0])}


def _meta(f_open: Callable[[], h5py.File], group: str, datasets: list[str], chunk_dir: Path | None) -> dict:
    """Layout of the granule (projection, regular axes, per-dataset dtype / chunks / filters), cached per granule."""
    path = chunk_dir / "meta.json" if chunk_dir is not None else None
    meta = json.loads(path.read_text()) if path is not None and path.exists() else {}
    m = meta.get(group)
    if m is None or any(k not in m["datasets"] for k in datasets):
        g = f_open()[group]
        m = m or {"epsg": _epsg(g), "x": _coords(g["xCoordinates"]), "y": _coords(g["yCoordinates"]), "datasets": {}}
        for k in datasets:
            m["datasets"].setdefault(k, _dataset_meta(g[k]))
        meta[group] = m
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(meta))
    return m


def _decode(raw: bytes, filter_mask: int, dm: dict) -> np.ndarray:
    """One stored chunk -> array of the chunk shape (undoing deflate / shuffle in reverse pipeline order)."""
    dtype, shape = np.dtype(dm["dtype"]), tuple(dm["chunks"])
    if filter_mask == _UNALLOCATED:
        return np.full(shape, dm["fillvalue"], dtype)
    buf = raw
    for i in reversed(range(len(dm["filters"]))):
        if filter_mask & (1 << i):
            continue
        if dm["filters"][i] == _DEFLATE:
            buf = zlib.decompress(buf)
        elif dm["filters"][i] == _SHUFFLE:
            buf = np.frombuffer(buf, np.uint8).reshape(dtype.itemsize, -1).T.tobytes()
        else:
            raise NotImplementedError(f"HDF5 filter {dm['filters'][i]}")
    return np.frombuffer(buf, dtype).reshape(shape)


def _chunk_starts(dm: dict, rs: slice, cs: slice) -> list[tuple[int, int]]:
    ch, cw = dm["chunks"]
    return [(r, c) for r in range((rs.start // ch) * ch, min(rs.stop, dm["shape"][0]), ch)
            for c in range((cs.start // cw) * cw, min(cs.stop, dm["shape"][1]), cw)]


def _remote_window(url: str, group: str, datasets: list[str], grid: TargetGrid, chunk_dir: Path | None):
    """(arrays, x, y, epsg, (rs, cs)) of the window around the grid, or None if the grid is outside.

    The granule is opened only for what the per-granule cache does not have yet: its layout (once) and the
    byte offsets of missing chunks. Missing chunks are then fetched in parallel in one batch.
    """
    t0 = time.perf_counter()
    with contextlib.ExitStack() as stack:
        opened: list[h5py.File] = []

        def f_open() -> h5py.File:
            if not opened:
                opened.append(stack.enter_context(open_granule(url)))
                log.debug("%s: opened in %.1f s", Path(url).name[:40], time.perf_counter() - t0)
            return opened[0]

        m = _meta(f_open, group, datasets, chunk_dir)
        x, y = np.linspace(*m["x"]), np.linspace(*m["y"])
        win = _grid_window(x, y, m["epsg"], grid)
        if win is None:
            return None
        rs, cs = win
        chunks: dict[tuple[str, int, int], tuple[int, bytes]] = {}   # (dataset, row, col) -> (filter mask, bytes)
        missing, direct = [], []
        for k in datasets:
            dm = m["datasets"][k]
            if dm["chunks"] is None or len(dm["shape"]) != 2 or set(dm["filters"]) - {_DEFLATE, _SHUFFLE}:
                direct.append(k)                    # contiguous / other compression: let h5py read it
                continue
            for r, c in _chunk_starts(dm, rs, cs):
                p = chunk_dir / k.replace("/", "__") / f"{r}_{c}.bin" if chunk_dir is not None else None
                if p is not None and p.exists():
                    b = p.read_bytes()
                    chunks[(k, r, c)] = (int.from_bytes(b[:4], "little"), b[4:])
                else:
                    missing.append((k, r, c, p))
        log.debug("%s: layout ready %.1f s, %d chunks cached, %d missing", Path(url).name[:40],
                  time.perf_counter() - t0, len(chunks), len(missing))
        if missing:
            g = f_open()[group]
            infos = [g[k].id.get_chunk_info_by_coord((r, c)) for k, r, c, _ in missing]
            log.debug("%s: chunk offsets %.1f s", Path(url).name[:40], time.perf_counter() - t0)
            stored = [i for i, info in enumerate(infos) if info.byte_offset is not None and info.size]
            fs, read_url, _ = _remote(url)
            blobs = fs.cat_ranges(
                [read_url] * len(stored), [infos[i].byte_offset for i in stored],
                [infos[i].byte_offset + infos[i].size for i in stored], on_error="raise") if stored else []
            log.debug("%s: fetched %.1f MB %.1f s", Path(url).name[:40], sum(map(len, blobs)) / 2**20,
                      time.perf_counter() - t0)
            data = dict(zip(stored, blobs))
            for i, (k, r, c, p) in enumerate(missing):
                mask, raw = (int(infos[i].filter_mask), bytes(data[i])) if i in data else (_UNALLOCATED, b"")
                if i in data and len(raw) != infos[i].size:
                    raise ConnectionError(f"short read of a NISAR chunk ({len(raw)} of {infos[i].size} bytes)")
                chunks[(k, r, c)] = (mask, raw)
                if p is not None:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(mask.to_bytes(4, "little") + raw)
        arrays = {}
        for k in datasets:
            dm = m["datasets"][k]
            if k in direct:
                a = f_open()[group][k][rs, cs]
            else:
                ch, cw = dm["chunks"]
                a = np.empty((min(rs.stop, dm["shape"][0]) - rs.start, min(cs.stop, dm["shape"][1]) - cs.start),
                             np.dtype(dm["dtype"]))
                for r, c in _chunk_starts(dm, rs, cs):
                    block = _decode(chunks[(k, r, c)][1], chunks[(k, r, c)][0], dm)
                    r0, c0 = max(r, rs.start), max(c, cs.start)
                    r1, c1 = min(r + ch, rs.stop, dm["shape"][0]), min(c + cw, cs.stop, dm["shape"][1])
                    a[r0 - rs.start:r1 - rs.start, c0 - cs.start:c1 - cs.start] = block[r0 - r:r1 - r, c0 - c:c1 - c]
            a = a.astype(np.float32)
            if dm["_FillValue"] is not None and np.isfinite(dm["_FillValue"]):
                a[a == dm["_FillValue"]] = np.nan
            arrays[k] = a
        return arrays, x, y, m["epsg"], (rs, cs)


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
