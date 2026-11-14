"""NISAR -> the spread model's NISAR contract (firespread/nisar.py), reading only the area around the fire.

    from firedata.nisar_provider import nisar_provider
    from firespread import predict
    result = predict(polygon=drawn_geojson, start="2026-08-09T21:00Z", nisar=nisar_provider())

For a request (model grid crs / x / y, lon-lat box, start time) the provider:
1. finds the latest NISAR pass before the start time over the box (ASF search, all frames); cells that pass
   does not see (swath edges) are filled from the next-newest pass within lookback_days, and so on;
2. GCOV (10 m): reads HH / HV of the box only (remote byte-range reads, cached), averages them in linear
   power into each model cell, keeping only valid samples (GCOV `mask` 1..254);
     nisar_cover_pct  <- HV, calibrated to vegetation cover (see COVER_HV_DB below)
     nisar_burned = 1 <- water / bare rock (nisar_layers.compute_burned_mask) and burn scars: HV dropped by
                        >= scar_drop_db (averaged over 3 x 3 cells) since an earlier pass of the same track
                        (same viewing geometry); see SCAR_DROP_DB for how well this finds burned area
3. SME2 (L3, 200 m, same pass): NASA's soil-moisture retrieval, sampled at the cell centres
     nisar_soil_moisture <- soilMoisture where the retrieval is "recommended" (retrievalQualityFlag bit 0 = 0)
     nisar_vwc_kg_m2     <- PMI vegetationWaterContentRetrieved where the PMI quality flag is 0
     water bodies (waterbodyFraction >= 0.5) are added to nisar_burned
Cells NISAR does not cover are no data (NaN / -1), never "burned". Returns None when no pass is found.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from .nisar_access import (CACHE_DIR, GCOV_GROUP, SME2_GROUP, Pass, TargetGrid, group_passes, read_pass,
                           search_granules)
from .nisar_layers import compute_burned_mask

log = logging.getLogger(__name__)

SPREAD_MODEL_VARIABLES = ("nisar_cover_pct", "nisar_soil_moisture", "nisar_vwc_kg_m2", "nisar_burned")

# HV (dB, 100 m mean of gamma0) -> vegetation cover %. Calibrated 2026-11-14 on the Bug Fire area before the fire
# (GCOV A106 2026-07-27, 30 x 30 km, 100 m) against LANDFIRE EVC cover of tree / shrub cells: quantile matching,
# sparse / normal boundary (30 %) at the HV with LANDFIRE's share of sparse cells. Skill is modest: class agreement
# 0.68 vs 0.63 for always "sparse", mean abs error 10 vs 13 points. HV sees woody volume, not grass (Spearman with
# LANDFIRE cover: trees 0.55, shrubs 0.28, grass -0.06), so grass reads as sparse. Re-calibrate per region.
COVER_HV_DB = (-21.0, -17.6, -12.0, -9.0)
COVER_PCT = (10.0, 30.0, 55.0, 70.0)

SME2_DATASETS = ["soilMoisture", "retrievalQualityFlag", "algorithmCandidates/PMI/vegetationWaterContentRetrieved",
                 "algorithmCandidates/PMI/retrievalQualityFlag", "ancillaryData/waterbodyFraction"]


def cover_from_hv(hv_linear: np.ndarray) -> np.ndarray:
    """Vegetation cover % from HV gamma0 (linear power), using the calibrated COVER_HV_DB -> COVER_PCT curve."""
    with np.errstate(invalid="ignore", divide="ignore"):
        hv_db = 10 * np.log10(hv_linear)
    out = np.interp(hv_db, COVER_HV_DB, COVER_PCT).astype(np.float32)
    out[~np.isfinite(hv_db)] = np.nan
    return out


def hv_change_db(hv: np.ndarray, hv_ref: np.ndarray, smooth_cells: int = 3) -> np.ndarray:
    """HV change (dB) since the reference pass, averaged in linear ratio over smooth_cells x smooth_cells cells.

    Averaging over neighbours cuts radar speckle: on the Bug Fire it raised burn-scar recall at the same
    false-alarm rate (see SCAR_* below). NaN cells are left out of the average; no valid neighbour -> NaN.
    """
    from scipy.ndimage import uniform_filter
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = hv / hv_ref
    ok = np.isfinite(ratio) & (ratio > 0)
    if smooth_cells > 1:
        num = uniform_filter(np.where(ok, ratio, 0.0), size=smooth_cells, mode="nearest")
        den = uniform_filter(ok.astype(np.float64), size=smooth_cells, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(den > 0, num / den, np.nan)
        ok = np.isfinite(ratio) & (ratio > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, 10 * np.log10(ratio), np.nan)


# Burn-scar rule, chosen on the Bug Fire (08-15 A34 vs 05-11 A34, 100 m, vs the official WFIGS perimeter):
#   HV drop >= 3 dB, single cell (original)      recall  9 %  precision 95 %  false alarms 0.2 %
#   HV drop >= 2 dB, single cell                 recall 23 %  precision 90 %  false alarms 1.0 %
#   HV drop >= 1.5 dB, 3 x 3 average (default)   recall 29 %  precision 92 %  false alarms 1.0 %
#   HV drop >= 1 dB, 3 x 3 average               recall 42 %  precision 80 %  false alarms 4.6 %
# End to end through nisar_provider (reference 05-23, water / rock included): recall 41 %, precision 90 %,
# false alarms 2.0 % (the original 3 dB rule: 14 % / 93 % / 0.5 %).
# Precision is favoured: a false "burned" cell blocks spread in the model. Median HV change inside the burned
# area was only -0.8 dB (shrub / sagebrush has little volume to lose), so backscatter change cannot find most of
# this fire's area; interferometric coherence loss (NISAR GUNW) is the next thing to try.
SCAR_DROP_DB = 1.5
SCAR_SMOOTH_CELLS = 3


def layers_from_backscatter(hh: np.ndarray, hv: np.ndarray, hv_ref: np.ndarray | None = None,
                            scar_drop_db: float = SCAR_DROP_DB,
                            scar_smooth_cells: int = SCAR_SMOOTH_CELLS) -> dict[str, np.ndarray]:
    """Cover % and burned mask from HH / HV (linear power, already on the model grid).

    hv_ref: HV of an earlier pass of the same track; cells where the (neighbourhood-averaged) HV dropped by
    >= scar_drop_db are burn scars. No data stays no data: cover NaN, burned -1.
    """
    nodata = ~(np.isfinite(hh) & np.isfinite(hv) & (hh > 0) & (hv > 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rvi = np.where(nodata, 0.0, 4.0 * hv / (hh + hv)).astype(np.float32)
    delta = None
    if hv_ref is not None:
        delta = np.where(nodata, np.nan, hv_change_db(hv, hv_ref, scar_smooth_cells))
    burned, _ = compute_burned_mask(hh, hv, rvi=rvi)          # water / bare rock (nisar_layers thresholds)
    if delta is not None:                                     # burn scars, with the configured threshold
        burned = np.where(np.nan_to_num(delta, nan=0.0) <= -scar_drop_db, 1, burned)
    burned = burned.astype(np.int16)
    burned[nodata] = -1
    cover = cover_from_hv(hv)
    cover[nodata] = np.nan
    return {"nisar_cover_pct": cover, "nisar_burned": burned}


def _gcov_valid(a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep only fully focused samples: GCOV mask 1..254 (0 = invalid, 255 = outside the acquisition)."""
    m = a.pop("mask")
    bad = ~np.isfinite(m) | (m == 0) | (m == 255)
    for k in a:
        a[k][bad] = np.nan
    return a


def _sme2_valid(a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Soil moisture where recommended (flag bit 0 = 0); VWC where the PMI flag is 0 (clear)."""
    q = a["retrievalQualityFlag"]
    recommended = np.isfinite(q) & ((np.nan_to_num(q, nan=1).astype(np.int64) & 1) == 0)
    sm = np.where(recommended, a["soilMoisture"], np.nan)
    pmi = a["algorithmCandidates/PMI/retrievalQualityFlag"]
    vwc = np.where(pmi == 0, a["algorithmCandidates/PMI/vegetationWaterContentRetrieved"], np.nan)
    return {"soil_moisture": sm.astype(np.float32), "vwc": vwc.astype(np.float32),
            "water_fraction": a["ancillaryData/waterbodyFraction"]}


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def nisar_provider(*, lookback_days: float = 30.0, detect_scars: bool = True, scar_lookback_days: float = 90.0,
                   scar_min_gap_days: float = 11.0, scar_drop_db: float = SCAR_DROP_DB,
                   scar_smooth_cells: int = SCAR_SMOOTH_CELLS,
                   search: Callable[..., pd.DataFrame] = search_granules,
                   cache_dir: Path | None = CACHE_DIR) -> Callable[[Any], xr.Dataset | None]:
    """A provider for firespread.predict(nisar=...).

    lookback_days:      use passes at most this long before the start (NISAR revisits a place every few days
                        across tracks, 12 days per track); the newest wins, older ones fill uncovered cells
    detect_scars:       compare HV with the earliest same-track pass in [start - scar_lookback_days,
                        latest pass - scar_min_gap_days]; a drop of >= scar_drop_db dB, averaged over
                        scar_smooth_cells x scar_smooth_cells cells, = burn scar
    search:             granule search (default: ASF); injectable for tests / other archives
    cache_dir:          where regridded windows are cached (None = no cache)
    Needs EARTHDATA_USERNAME / EARTHDATA_PASSWORD in real_fires/.env for remote reads.
    """
    def one_pass(p: Pass, gcov_passes: list[Pass], grid: TargetGrid, bbox) -> tuple[dict[str, np.ndarray], dict] | None:
        """All contract variables from one pass (its own scar reference and SME2). None if it misses the grid."""
        gcov = read_pass(p, GCOV_GROUP, ["HHHH", "HVHV", "mask"], grid, prepare=_gcov_valid, cache_dir=cache_dir)
        hh, hv = gcov["HHHH"], gcov["HVHV"]
        if not np.isfinite(hv).any():
            return None
        info = {"time": p.time_utc.isoformat(), "track": p.track_id, "gcov": ",".join(p.file_ids)}

        hv_ref = None
        if detect_scars:
            refs = [r for r in gcov_passes if r.track_id == p.track_id
                    and r.time_utc <= p.time_utc - pd.Timedelta(days=scar_min_gap_days)]
            if refs:
                ref = refs[0]   # earliest in the window: scars from the whole season so far
                hv_ref = read_pass(ref, GCOV_GROUP, ["HVHV", "mask"], grid, prepare=_gcov_valid,
                                   cache_dir=cache_dir)["HVHV"]
                info["scar_reference"] = ref.time_utc.isoformat()
            else:
                info["scar_reference"] = "none (no earlier same-track pass): scars not detected"
        layers = layers_from_backscatter(hh, hv, hv_ref, scar_drop_db, scar_smooth_cells)

        sm = np.full(grid.shape, np.nan, np.float32)
        vwc = np.full(grid.shape, np.nan, np.float32)
        sme2 = [s for s in group_passes(search("SME2", bbox, p.time_utc - pd.Timedelta(minutes=10),
                                               p.time_utc + pd.Timedelta(minutes=10)))
                if s.track_id == p.track_id]
        if sme2:
            s = read_pass(sme2[0], SME2_GROUP, SME2_DATASETS, grid, prepare=_sme2_valid, cache_dir=cache_dir)
            sm, vwc = s["soil_moisture"], s["vwc"]
            water = np.nan_to_num(s["water_fraction"], nan=0.0) >= 0.5
            layers["nisar_burned"][water & (layers["nisar_burned"] >= 0)] = 1
            info["sme2"] = ",".join(sme2[0].file_ids)
        else:
            info["sme2"] = "none for this pass: soil moisture / VWC not provided"
        return {**layers, "nisar_soil_moisture": sm, "nisar_vwc_kg_m2": vwc}, info

    def get_nisar(request: Any) -> xr.Dataset | None:
        grid = TargetGrid.from_request(request)
        t = _utc(request.time_utc)
        bbox = tuple(request.bbox_lonlat)
        start_search = t - pd.Timedelta(days=max(lookback_days, scar_lookback_days if detect_scars else 0))
        gcov_passes = [p for p in group_passes(search("GCOV", bbox, start_search, t)) if p.time_utc < t]
        recent = [p for p in gcov_passes if p.time_utc >= t - pd.Timedelta(days=lookback_days)]
        if not recent:
            log.info("no NISAR GCOV pass in the %s days before %s", lookback_days, t)
            return None

        # Newest pass first; older passes in the lookback window only fill cells no newer pass saw (swath edges,
        # partial frames). A cell takes ALL its variables from one pass, so radar, scars and soil moisture of a
        # cell are always from the same date. Stops as soon as every cell is covered.
        data: dict[str, np.ndarray] | None = None
        used: list[dict] = []
        for p in reversed(recent):
            if data is not None and not (data["nisar_burned"] == -1).any():
                break
            got = one_pass(p, gcov_passes, grid, bbox)
            if got is None:
                continue
            layers, info = got
            if data is None:
                data, fill = layers, layers["nisar_burned"] != -1
            else:
                fill = (data["nisar_burned"] == -1) & (layers["nisar_burned"] != -1)
                for k in SPREAD_MODEL_VARIABLES:
                    data[k][fill] = layers[k][fill]
            info["cells_pct"] = round(float(100 * fill.mean()), 1)
            if info["cells_pct"] > 0:
                used.append(info)
        if data is None:
            log.info("no NISAR pass in the %s days before %s covers the area", lookback_days, t)
            return None

        newest = used[0]
        attrs = {"crs": grid.crs, "nisar_pass_utc": newest["time"], "nisar_track": newest["track"],
                 "nisar_gcov_granules": newest["gcov"], "nisar_scar_reference_utc": newest.get("scar_reference", ""),
                 "nisar_sme2_granules": newest["sme2"],
                 "nisar_passes_used": "; ".join(f"{u['time']} {u['track']} ({u['cells_pct']}% of cells)" for u in used),
                 "nisar_passes_detail": json.dumps(used),
                 "nisar_coverage_pct": float(100 * (data["nisar_burned"] != -1).mean())}
        coords = {"y": ("y", grid.y), "x": ("x", grid.x)}
        return xr.Dataset({k: (("y", "x"), data[k]) for k in SPREAD_MODEL_VARIABLES}, coords=coords, attrs=attrs)

    return get_nisar


def to_spread_model(ds: xr.Dataset) -> xr.Dataset:
    """Adapt an extract_nisar_dataset() result (e.g. a saved .nc) to the spread-model contract.

    Keeps only the contract variables. Prefer nisar_provider(): extract_nisar_dataset() marks no-data cells as
    burned, estimates soil moisture from HH and uses RVI x 100 as cover (see nisar_provider's docstring).
    """
    out = ds[[v for v in SPREAD_MODEL_VARIABLES if v in ds]]
    return out.drop_vars([c for c in ("lat", "lon") if c in out.coords])
