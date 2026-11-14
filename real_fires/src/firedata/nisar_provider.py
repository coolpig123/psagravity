"""Bridge: NISAR GCOV layers (nisar_layers.py) -> the spread model's NISAR contract (firespread/nisar.py).

    from firedata.nisar_provider import nisar_provider
    from firespread import predict
    result = predict(polygon=drawn_geojson, start="2026-08-09T21:00Z", nisar=nisar_provider())

The spread model calls the provider with a request (grid crs / x / y, lon-lat box, start time). The
provider picks the latest NISAR GCOV pass before the start time over that area, reads HH / HV straight
onto the model grid (nisar_layers.resample_gcov_to_grid) and computes the variables with
nisar_layers' formulas. Only the variables the spread model knows are returned:

    nisar_cover_pct (%), nisar_soil_moisture (m3/m3), nisar_burned (0/1, -1 = no NISAR data)

Differences from extract_nisar_dataset(), on purpose:
- cells without NISAR data (outside the swath / fill) are "no data" (NaN, -1), not "burned", so the
  fire is not stopped at the edge of a swath;
- `nisar_combustible` is not returned (it is 1 - nisar_burned; the model derives that itself);
- the grid is the model's, in the model's CRS, instead of a fixed 20 km box around the ignition.

No spread-model import is needed here: the request is used by attribute (crs, x, y, bbox_lonlat, time_utc).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from .config import FIRES, REPO_ROOT
from .nisar_layers import compute_burned_mask, compute_cover_pct, compute_soil_moisture, resample_gcov_to_grid

log = logging.getLogger(__name__)

SPREAD_MODEL_VARIABLES = ("nisar_cover_pct", "nisar_soil_moisture", "nisar_burned")


def layers_from_backscatter(hh: np.ndarray, hv: np.ndarray) -> dict[str, np.ndarray]:
    """nisar_layers' formulas on HH / HV (linear power) already on the model grid; no data kept as no data."""
    nodata = ~(np.isfinite(hh) & np.isfinite(hv) & (hh > 0) & (hv > 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rvi = np.where(nodata, 0.0, 4.0 * hv / (hh + hv)).astype(np.float32)
    cover = compute_cover_pct(hh, hv)
    soil = compute_soil_moisture(hh)
    burned, _ = compute_burned_mask(hh, hv, rvi=rvi)
    cover[nodata] = np.nan
    soil[nodata] = np.nan
    burned = burned.astype(np.int16)
    burned[nodata] = -1
    return {"nisar_cover_pct": cover, "nisar_soil_moisture": soil, "nisar_burned": burned}


def to_spread_model(ds: xr.Dataset) -> xr.Dataset:
    """Adapt an extract_nisar_dataset() result (e.g. a saved .nc) to the spread-model contract.

    Keeps only the contract variables. extract_nisar_dataset() marks no-data cells as burned; they
    cannot be told apart from real water / rock afterwards, so prefer nisar_provider() when possible.
    """
    out = ds[[v for v in SPREAD_MODEL_VARIABLES if v in ds]]
    return out.drop_vars([c for c in ("lat", "lon") if c in out.coords])


def _latest_pass_before(request: Any, fire: str | None, track: str | None, lookback_days: float) -> tuple[str, str] | None:
    """(file_id, raw cache folder) of the latest GCOV pass before the request time, or None."""
    t = pd.Timestamp(request.time_utc)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    if fire:
        cfg = FIRES[fire]
        passes = pd.read_csv(cfg.root / "processed" / "nisar_passes.csv", parse_dates=["time_utc"])
        raw = cfg.dir("raw") / "nisar"
    else:   # any place (e.g. a polygon drawn in the app): search ASF around the centre of the area
        from .nisar import search_passes
        w, s, e, n = request.bbox_lonlat
        passes = search_passes((w + e) / 2, (s + n) / 2, t - pd.Timedelta(days=lookback_days), t)
        raw = REPO_ROOT / "data" / "nisar_cache"
    if passes.empty:
        return None
    times = pd.to_datetime(passes["time_utc"], utc=True)
    sel = passes[(times < t) & (times >= t - pd.Timedelta(days=lookback_days))]
    if track:
        sel = sel[sel["track_id"] == track]
    if sel.empty:
        return None
    row = sel.assign(_t=pd.to_datetime(sel["time_utc"], utc=True)).sort_values("_t").iloc[-1]
    return str(row["file_id"]), str(raw)


def nisar_provider(fire: str | None = None, track: str | None = None, h5_path: str | Path | None = None,
                   lookback_days: float = 30.0) -> Callable[[Any], xr.Dataset | None]:
    """A provider for firespread.predict(nisar=...).

    fire:     use that fire's NISAR pass list (processed/nisar_passes.csv); None = search ASF around the area
    track:    only this track (e.g. "A34"); None = latest pass on any track
    h5_path:  use this GCOV file directly (no search, no download)
    lookback_days: ignore passes older than this before the start time (NISAR revisits every 12 days)
    Downloading needs EARTHDATA_USERNAME / EARTHDATA_PASSWORD in real_fires/.env; files are cached.
    """
    def get_nisar(request: Any) -> xr.Dataset | None:
        if h5_path is not None:
            path, granule = Path(h5_path), Path(h5_path).stem
        else:
            found = _latest_pass_before(request, fire, track, lookback_days)
            if found is None:
                log.info("no NISAR pass in the %s days before %s", lookback_days, request.time_utc)
                return None
            granule, raw = found
            from .vegetation import download_gcov
            path = download_gcov(granule, Path(raw))
        try:
            hh, hv = resample_gcov_to_grid(path, np.asarray(request.x), np.asarray(request.y), target_crs=request.crs)
        except (IndexError, ValueError):   # area outside the granule: resample_gcov_to_grid gets an empty window
            log.info("NISAR granule %s does not cover the area", granule)
            return None
        layers = layers_from_backscatter(hh, hv)
        if (layers["nisar_burned"] == -1).all():
            log.info("NISAR granule %s does not cover the area", granule)
            return None
        coords = {"y": ("y", np.asarray(request.y)), "x": ("x", np.asarray(request.x))}
        return xr.Dataset({k: (("y", "x"), v) for k, v in layers.items()}, coords=coords,
                          attrs={"crs": request.crs, "nisar_granule": granule,
                                 "nisar_coverage_pct": float(100 * (layers["nisar_burned"] != -1).mean())})

    return get_nisar
