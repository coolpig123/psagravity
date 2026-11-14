"""Download an ee.Image onto a Grid as a numpy array, using ee.data.computePixels.

Large requests are split into tiles and band groups (each request stays under Earth Engine's
~48 MB limit) and run in parallel, with retries on rate-limit / transient errors.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np

from .grid import Grid

NODATA = -9999.0              # what masked pixels are filled with before download; becomes NaN
MAX_BYTES = 32 * 2**20        # per request, below the 48 MB hard limit
MAX_CELLS = 16_000_000        # per grid (4000 x 4000 = 400 km square at 100 m); guards against huge drawn areas
BYTES_PER_PX = 4              # float32


def region(grid: Grid) -> ee.Geometry:
    """The grid's footprint as an ee.Geometry (for filterBounds)."""
    return ee.Geometry.Rectangle(list(grid.bounds), grid.crs, False)


def _prepare(image: ee.Image) -> ee.Image:
    return ee.Image(image).toFloat().unmask(NODATA, False)


def _compute(image: ee.Image, bands: list[str], grid: Grid) -> np.ndarray:
    """One computePixels call. Returns a structured array with one field per band."""
    return ee.data.computePixels({"expression": image.select(bands), "fileFormat": "NUMPY_NDARRAY",
                                  "grid": grid.ee_grid()})


def _with_retry(fn, tries: int = 6):
    for k in range(tries):
        try:
            return fn()
        except ee.EEException as e:
            msg = str(e).lower()
            transient = any(s in msg for s in ("too many requests", "quota", "rate", "timeout", "timed out",
                                                "internal error", "unavailable", "deadline"))
            if not transient or k == tries - 1:
                raise
            time.sleep(min(60, 2 ** k))


def plan(n_bands: int, max_bytes: int = MAX_BYTES) -> tuple[int, int]:
    """(bands per request, tile side in cells) so each request stays under max_bytes."""
    per_request = max(1, min(n_bands, max_bytes // (BYTES_PER_PX * 256 * 256)))
    side = int(math.sqrt(max_bytes / (BYTES_PER_PX * per_request)))
    return per_request, max(256, min(2048, side))


def download(image: ee.Image, grid: Grid, bands: list[str], workers: int = 8,
             max_bytes: int = MAX_BYTES) -> np.ndarray:
    """Pixels of `bands` of `image` on `grid`, as float32 (n_bands, height, width); masked = NaN.

    Only the grid's cells are computed and downloaded, never the whole dataset.
    """
    if grid.width * grid.height > MAX_CELLS:
        km_w, km_h = grid.width * grid.res / 1000, grid.height * grid.res / 1000
        raise ValueError(f"area too large: {grid.width} x {grid.height} cells ({km_w:.0f} x {km_h:.0f} km at "
                         f"{grid.res:g} m) > {MAX_CELLS:,} cells. Draw a smaller area, reduce the buffer, or use "
                         "coarser cells (res_m).")
    image = _prepare(image)
    out = np.full((len(bands), grid.height, grid.width), np.nan, np.float32)
    per_request, side = plan(len(bands), max_bytes)
    jobs = [(b0, bands[b0:b0 + per_request], r0, c0, sub)
            for b0 in range(0, len(bands), per_request) for r0, c0, sub in grid.tiles(side)]

    def run(job):
        b0, names, r0, c0, sub = job
        raw = _with_retry(lambda: _compute(image, names, sub))
        for i, name in enumerate(names):
            out[b0 + i, r0:r0 + sub.height, c0:c0 + sub.width] = raw[name]

    with ThreadPoolExecutor(max(1, min(workers, len(jobs)))) as pool:
        list(pool.map(run, jobs))   # list() re-raises the first error
    out[out == NODATA] = np.nan
    return out
