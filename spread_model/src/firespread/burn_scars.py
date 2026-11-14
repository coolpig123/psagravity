"""Burn scars from Sentinel-2 (Earth Engine): dNBR = NBR a year earlier - latest clear NBR.

NBR = (NIR - SWIR2) / (NIR + SWIR2) (bands B8A, B12, 20 m). Burning removes green leaves (NIR down) and
exposes char / dry soil (SWIR2 up), so NBR drops; dNBR > 0.15 marks a burn scar.

  reference = median clear NBR over the same season one year earlier (t - 1 y +- 30 days)
              -> same phenology, and no need to know when a fire burned
  current   = latest cloud-free observation of each cell in the 30 days before t

Validated on the Bug Fire (2026, 100 m cells, vs the official WFIGS perimeter): as of 08-17 (first clear
image after the fire) recall 77%, precision 96%, false alarms 1.3% outside the perimeter; NISAR HV change
gave 41% / 90% / 2.0%. Smoke and cloud hide cells, so before a clear post-fire image exists recall is low;
NISAR's nisar_burned (radar, sees through smoke) still applies there.

Output variables (dims y, x, on the inputs grid):
  s2_burned  int16 : 1 = burn scar -> P_mask = 0, 0 = not, -1 = no clear observation
  s2_dnbr    float : the dNBR value
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DNBR_THRESHOLD = 0.15
CURRENT_DAYS = 30           # latest clear pixel within this many days before t
REFERENCE_HALF_DAYS = 30    # reference = t - 365 d +- this
NATIVE_RES_M = 20.0         # B8A / B12
_SCL_CLEAR = [4, 5, 6, 7, 11]   # vegetation, bare, water, unclassified, snow (drops cloud, shadow, saturated)


def scars_from_dnbr(dnbr: np.ndarray, threshold: float = DNBR_THRESHOLD) -> np.ndarray:
    """dNBR -> int16 1 burned / 0 not / -1 no data (NaN)."""
    out = np.where(dnbr > threshold, 1, 0).astype(np.int16)
    out[~np.isfinite(dnbr)] = -1
    return out


def _block_mean(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape[0] // k, a.shape[1] // k
    blocks = a.reshape(h, k, w, k)
    n = np.isfinite(blocks).sum(axis=(1, 3))
    s = np.nansum(blocks, axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan).astype(np.float32)


def _grid_of(inputs: xr.Dataset):
    """geelayers Grid of the inputs Dataset (cell centres x west->east, y north->south)."""
    from geelayers import Grid
    crs = str(inputs.attrs.get("crs", ""))
    if not crs.upper().startswith("EPSG:"):
        raise ValueError(f"inputs need attrs['crs'] like 'EPSG:32610', got {crs!r}")
    x, y = inputs.x.values, inputs.y.values
    res = float(inputs.attrs.get("res_m", abs(float(x[1] - x[0]))))
    if y[0] < y[-1]:
        raise ValueError("inputs y must run north -> south (geelayers order)")
    return Grid(int(crs.split(":")[1]), float(x[0] - res / 2), float(y[0] + res / 2), res, x.size, y.size)


def _fine_grid(grid):
    """The 20 m grid nested in `grid` (to average to the model cells), or None if not usable.

    None when the model cells are not a whole multiple (>= 2) of 20 m, or when the 20 m grid (k^2 times more
    cells) would exceed geelayers' MAX_CELLS, e.g. a polygon drawn > ~80 km wide at 100 m.
    """
    from geelayers import Grid
    from geelayers.fetch import MAX_CELLS
    k = int(round(grid.res / NATIVE_RES_M))
    if k < 2 or abs(grid.res / k - NATIVE_RES_M) > 1e-6 or grid.width * grid.height * k * k > MAX_CELLS:
        return None
    return Grid(grid.epsg, grid.x0, grid.y0, grid.res / k, grid.width * k, grid.height * k)


def _nbr_images(region, t: pd.Timestamp):
    import ee
    ms = lambda ts: int(ts.timestamp() * 1000)  # noqa: E731

    def nbr(img):
        clear = img.select("SCL").remap(_SCL_CLEAR, [1] * len(_SCL_CLEAR), 0)
        return img.normalizedDifference(["B8A", "B12"]).rename("nbr").updateMask(clear)

    s2 = ee.ImageCollection(S2_COLLECTION).filterBounds(region).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    cur_start = t - pd.Timedelta(days=CURRENT_DAYS)
    ref_mid = t - pd.Timedelta(days=365)
    ref_start, ref_end = ref_mid - pd.Timedelta(days=REFERENCE_HALF_DAYS), ref_mid + pd.Timedelta(days=REFERENCE_HALF_DAYS)
    cur = s2.filterDate(ms(cur_start), ms(t))
    ref = s2.filterDate(ms(ref_start), ms(ref_end))
    n_cur, n_ref = ee.List([cur.size(), ref.size()]).getInfo()
    meta = {"s2_current_window": f"{cur_start.isoformat()} / {t.isoformat()}", "s2_current_scenes": n_cur,
            "s2_reference_window": f"{ref_start.isoformat()} / {ref_end.isoformat()}", "s2_reference_scenes": n_ref}
    if not n_cur or not n_ref:
        return None, meta
    current = cur.map(nbr).sort("system:time_start").mosaic()    # newest clear pixel on top
    return ref.map(nbr).median().subtract(current).rename("dnbr"), meta


def optical_burn_scars(inputs: xr.Dataset, time=None, threshold: float = DNBR_THRESHOLD) -> xr.Dataset:
    """Add s2_burned / s2_dnbr (Sentinel-2 burn scars as of `time`) to a geelayers inputs Dataset.

    Downloads only the inputs grid's area. Coarser grids are computed at 20 m and averaged per cell.
    If Sentinel-2 has no scenes for the windows, the inputs are returned unchanged (attrs say why).
    """
    from geelayers.auth import init
    from geelayers.fetch import download, region

    t = pd.Timestamp(time or inputs.attrs.get("time_utc"))
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    grid = _grid_of(inputs)
    init()
    dnbr_img, meta = _nbr_images(region(grid), t)
    out = inputs.copy()
    out.attrs.update({k: str(v) for k, v in meta.items()})
    if dnbr_img is None:
        out.attrs["s2_burn_scars"] = "skipped: no Sentinel-2 scenes in the current or reference window"
        return out
    fine = _fine_grid(grid)
    if fine is not None:
        dnbr = _block_mean(download(dnbr_img, fine, ["dnbr"])[0], fine.width // grid.width)
    else:      # already ~20 m, not a multiple of 20 m, or the 20 m grid would be too big: sample at model cells
        dnbr = download(dnbr_img, grid, ["dnbr"])[0]
    out["s2_dnbr"] = (("y", "x"), dnbr.astype(np.float32), {"units": "1", "long_name": "Sentinel-2 dNBR"})
    out["s2_burned"] = (("y", "x"), scars_from_dnbr(dnbr, threshold),
                        {"long_name": "Sentinel-2 burn scar (1), not (0), no clear observation (-1)"})
    out.attrs.update({"s2_burn_scars": f"{S2_COLLECTION}, dNBR > {threshold}",
                      "s2_clear_pct": f"{100 * np.isfinite(dnbr).mean():.0f}",
                      "s2_burned_pct": f"{100 * (dnbr > threshold).mean():.1f}"})
    return out
