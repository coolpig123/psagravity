"""RADARSAT Constellation Mission (RCM) CEOS-ARD from the public AWS bucket (no account, no key).

Bucket s3://rcm-ceos-ard (ca-central-1), produced by NRCan / CCRS: 30 m compact-polarimetry (transmit
circular, receive H/V -> RR, RL, RRRL), radiometrically terrain-corrected gamma0 (linear power), tiled
Cloud-Optimised GeoTIFFs at 20 m pixels. Layout:
  MLC/YYYY/MM/DD/<scene>_CH_CV_MLC/<scene>_{RL,RR,RRRL,bitmask,localIncAngle,...}.tif + _product.xml
Licence: RCM public EULA (PDF in every scene folder); free to use.

Only a window around the fire is read over HTTP (GDAL /vsicurl/), never a full scene.
RL is the cross-polarised channel (volume scattering from canopy); burning removes canopy, so RL drops.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

BUCKET = "https://rcm-ceos-ard.s3.ca-central-1.amazonaws.com"
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
            "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES"}


def scene_url(scene: str, layer: str, product: str = "MLC") -> str:
    """URL of one layer (RL, RR, RRRL, bitmask, ...) of a scene, e.g.
    scene = 'RCM2_OK4114697_PK4292834_1_SC30MCPD_20260801_140821_CH_CV_MLC'."""
    date = re.search(r"_(\d{8})_\d{6}_", scene).group(1)
    stem = scene.replace(f"_CH_CV_{product}", "")
    return f"{BUCKET}/{product}/{date[:4]}/{date[4:6]}/{date[6:]}/{scene}/{stem}_{layer}.tif"


def grid(bounds_utm: tuple[float, float, float, float], res: float = 20.0) -> tuple:
    """(transform, (height, width)) of a north-up grid snapped to `res` covering bounds (W, S, E, N)."""
    w, s, e, n = bounds_utm
    w, s = np.floor(w / res) * res, np.floor(s / res) * res
    e, n = np.ceil(e / res) * res, np.ceil(n / res) * res
    return from_origin(w, n, res, res), (int(round((n - s) / res)), int(round((e - w) / res)))


def read_clipped(scene: str, layer: str, crs, transform, shape, cache_dir: Path | None = None) -> np.ndarray:
    """One layer of a scene resampled onto (crs, transform, shape); only the needed tiles are fetched.

    Backscatter is averaged (linear power), the bitmask uses nearest neighbour. Nodata -> NaN.
    With cache_dir the clip is kept as a small GeoTIFF and reused.
    """
    out = Path(cache_dir) / f"{scene.split('_CH_')[0]}_{layer}.tif" if cache_dir else None
    if out is not None and out.exists():
        with rasterio.open(out) as d:
            return d.read(1).astype(np.float32)
    mask = layer == "bitmask"
    with rasterio.Env(**GDAL_ENV), rasterio.open("/vsicurl/" + scene_url(scene, layer)) as src:
        with WarpedVRT(src, crs=crs, transform=transform, height=shape[0], width=shape[1],
                       resampling=Resampling.nearest if mask else Resampling.average,
                       src_nodata=src.nodata, nodata=np.nan if not mask else 255, dtype="float32") as vrt:
            a = vrt.read(1).astype(np.float32)
    if not mask:
        a[(a <= 0) | ~np.isfinite(a)] = np.nan
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="float32",
                           crs=crs, transform=transform, compress="deflate", nodata=np.nan) as d:
            d.write(a, 1)
    return a


def scores(burned: np.ndarray, official: np.ndarray, valid: np.ndarray, pixel_m2: float) -> dict:
    """Detected burned mask vs official perimeter raster, on valid pixels only.

    cover = recall = |B ∩ W| / |W|; precision = |B ∩ W| / |B|; IoU = |B ∩ W| / |B ∪ W|;
    false_alarm = share of valid pixels outside W flagged as burned.
    """
    b, w = burned & valid, official & valid
    inter, union = (b & w).sum(), (b | w).sum()
    out_w = valid & ~official
    return {"detected_ha": round(b.sum() * pixel_m2 / 1e4, 1),
            "detected_inside_ha": round(inter * pixel_m2 / 1e4, 1),
            "cover_pct": round(100 * inter / max(w.sum(), 1), 1),
            "precision_pct": round(100 * inter / max(b.sum(), 1), 1),
            "iou_pct": round(100 * inter / max(union, 1), 1),
            "false_alarm_pct": round(100 * (b & out_w).sum() / max(out_w.sum(), 1), 2),
            "valid_in_perimeter_pct": round(100 * w.sum() / max(official.sum(), 1), 1)}
