"""Vegetation / fuel density layer from NISAR L2 GCOV.

Scientific basis
----------------
NISAR L-band SAR (wavelength ~24 cm) penetrates smoke and cloud cover.
Inside the canopy the signal undergoes **volume scattering** off branches and
leaves.  The cross-polarised channel (HV) is almost entirely dominated by
this effect, so its intensity correlates directly with biomass / fuel load.
The co-polarised channel (HH) contains both volume scattering and surface /
double-bounce components, so it is less specific to vegetation density.

We use the **Dual-Polarisation Radar Vegetation Index** (DpRVI):

    RVI = 4 · γ⁰_HV / (γ⁰_HH + γ⁰_HV)

    RVI → 0  :  bare ground, sparse fuel
    RVI → 1  :  dense forest, heavy fuel load

Dividing by (HH + HV) normalises out incidence-angle and system-gain effects
and isolates the fraction of energy that ended up cross-polarised — a pure
proxy for the vegetated volume.  The normalised RVI becomes the **p_fuel**
layer in the Alexandridis spread model:

    p_burn = p_h · (1 + p_fuel) · (1 + p_dry) · p_wind · p_slope

We use the **L2 GCOV** product (not L1), which means NASA/ASF have already
applied noise filtering, radiometric calibration and terrain-flattening:
the values in the HDF5 file are terrain-corrected normalised radar cross-
sections (γ⁰) in linear power scale — ready to use with no further
pre-processing.

Outputs per fire × track
--------------------------
  processed/fuel_layer_<track>_<phase>.tif  : 3-band GeoTIFF in WGS84
        band 1  hv_db   – HV backscatter in dB (raw SAR signal)
        band 2  rvi     – Radar Vegetation Index [0, 1]
        band 3  p_fuel  – normalised RVI for the spread model [0, 1]
  figures/fuel_layer_<track>_<phase>.png    : two-panel heatmap figure

Usage
-----
    # from real_fires/
    python -m firedata.vegetation bug_fire A34
    python -m firedata.vegetation bug_fire A34 --phase pre

    # or via the script:
    python scripts/build_vegetation_layer.py bug_fire
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests
import earthaccess
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio

from .config import FIRES, FireConfig, get_secret

log = logging.getLogger(__name__)

# ── HDF5 paths inside a NISAR L2 GCOV file ─────────────────────────────────
#
# The GCOV is a hierarchical HDF5.  All SAR science data live under:
#   /science/LSAR/GCOV/grids/frequencyA/
# LSAR = L-band SAR (the main NISAR instrument).
# frequencyA = primary frequency (the only one in standard dual-pol DHDH mode).
#
# The covariance matrix diagonal elements are stored as real 2-D arrays:
#   HHHH  → γ⁰_HH  (HH·HH*, linear power)
#   HVHV  → γ⁰_HV  (HV·HV*, linear power)
#
FREQ_PATH = "/science/LSAR/GCOV/grids/frequencyA"
HH_DS     = f"{FREQ_PATH}/HHHH"
HV_DS     = f"{FREQ_PATH}/HVHV"
X_DS      = f"{FREQ_PATH}/xCoordinates"   # UTM easting  [m], 1-D
Y_DS      = f"{FREQ_PATH}/yCoordinates"   # UTM northing [m], 1-D (may be descending)
PROJ_DS   = f"{FREQ_PATH}/projection"     # WKT string of the native UTM CRS

WGS84 = "EPSG:4326"


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Download the GCOV granule from NASA Earthdata / ASF
# ═══════════════════════════════════════════════════════════════════════════

def download_gcov(file_id: str, raw_dir: Path) -> Path:
    """Download one NISAR L2 GCOV granule from ASF via earthaccess.

    ``file_id`` is the producer granule ID, e.g.:
        NISAR_L2_PR_GCOV_026_034_A_021_4005_DHDH_A_20260722T130932_..._001

    Authentication uses EARTHDATA_USERNAME / EARTHDATA_PASSWORD from the
    environment (.env).  The file is cached in ``raw_dir``; if it already
    exists on disk we skip the download entirely.

    The GCOV files are ~400–800 MB each (HDF5 with lossless compression).
    """
    local = raw_dir / (file_id + ".h5")
    if local.exists() and local.stat().st_size > 1000:
        log.info("Cache hit, skipping download: %s", local.name)
        return local

    raw_dir.mkdir(parents=True, exist_ok=True)
    get_secret("EARTHDATA_USERNAME")           # raises early if .env is incomplete
    earthaccess.login(strategy="environment")  # reads USERNAME + PASSWORD from env

    download_url = None

    # Step 1: Query ASF DAAC search API for the granule download URL
    log.info("Querying ASF DAAC for download URL of: %s", file_id)
    try:
        asf_resp = requests.get(
            "https://api.daac.asf.alaska.edu/services/search/param",
            params={"granule_list": file_id, "output": "geojson"},
            timeout=60,
        )
        if asf_resp.ok:
            data = asf_resp.json()
            features = data.get("features", []) if isinstance(data, dict) else []
            if features:
                props = features[0].get("properties", {})
                download_url = props.get("url") or props.get("downloadUrl") or props.get("fileUrl")
    except Exception as e:
        log.warning("ASF search query failed: %s", e)

    # Step 2: Fallback to ASF datapool standard URL if search did not return it
    if not download_url:
        download_url = f"https://datapool.asf.alaska.edu/GCOV/NISAR/{file_id}.h5"

    log.info("Downloading file from: %s", download_url)
    session = earthaccess.get_requests_https_session()
    with session.get(download_url, stream=True, timeout=1800) as r:
        if r.status_code == 404:
            raise FileNotFoundError(
                f"File not found on ASF: {download_url}\n"
                "Verify your Earthdata account has accepted the NISAR EULA."
            )
        r.raise_for_status()
        total_b = int(r.headers.get("content-length", 0))
        total_mb = total_b / (1024 * 1024) if total_b else 0
        downloaded = 0
        with open(local, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_mb:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Downloading: {mb:.1f} / {total_mb:.1f} MB ({mb/total_mb*100:.0f}%)", end="", flush=True)
        if total_mb:
            print()

    log.info("Saved to: %s", local)
    return local


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Open HDF5 and extract the two backscatter layers
# ═══════════════════════════════════════════════════════════════════════════

def _crs_wkt(h5: h5py.File) -> str:
    """Read the WKT CRS string from the GCOV projection dataset.

    ASF stores the WKT in several possible locations depending on the
    product version; we try them in order.
    """
    proj = h5[PROJ_DS]
    candidates = [
        proj.attrs.get("spatial_ref"),
        proj.attrs.get("crs_wkt"),
        proj[()],                    # the dataset value itself
    ]
    for src in candidates:
        if src is None:
            continue
        s = src if isinstance(src, str) else src.decode("utf-8", errors="replace")
        if s.strip():
            return s
    raise ValueError(
        f"Cannot find WKT CRS in GCOV at '{PROJ_DS}'. "
        "Check the product version against the NISAR-002237 spec."
    )


def _bbox_wgs84_to_crs(bbox_wgs84: tuple, target_wkt: str) -> tuple:
    """Transform a (west, south, east, north) bbox from WGS84 to ``target_wkt``.

    We project all four corners and take the bounding box of the result to
    handle any skew introduced by the UTM→WGS84 boundary.
    """
    t = Transformer.from_crs(WGS84, target_wkt, always_xy=True)
    west, south, east, north = bbox_wgs84
    xs, ys = t.transform(
        [west, east, west, east],
        [south, south, north, north],
    )
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _coord_window(coords: np.ndarray, lo: float, hi: float, pad: int = 2) -> slice:
    """Return an array index slice that covers the interval [lo, hi].

    Works for both ascending and descending coordinate arrays (GCOV y-coords
    are typically descending: north → south, matching image convention).
    ``pad`` adds a few extra pixels on each side to avoid boundary artefacts
    when reprojecting later.
    """
    if coords[-1] >= coords[0]:  # ascending (x-axis)
        i0 = max(0, int(np.searchsorted(coords, lo, "left")) - pad)
        i1 = min(len(coords), int(np.searchsorted(coords, hi, "right")) + pad)
    else:                         # descending (y-axis, north-first)
        i0 = max(0, int(np.searchsorted(-coords, -hi, "left")) - pad)
        i1 = min(len(coords), int(np.searchsorted(-coords, -lo, "right")) + pad)
    return slice(i0, i1)


def load_backscatter(h5_path: Path, bbox_wgs84: tuple) -> dict:
    """Open a GCOV HDF5 and return HH + HV arrays clipped to the fire bbox.

    The clip is done in the native UTM grid of the file (no resampling yet).
    Fill values and non-physical (≤ 0) pixels are replaced with NaN.

    Returns
    -------
    dict with keys:
      hh      : float32 array [nrows, ncols] – γ⁰_HH linear power
      hv      : float32 array [nrows, ncols] – γ⁰_HV linear power
      x       : float64 1-D easting  [m]
      y       : float64 1-D northing [m]  (may be descending)
      utm_wkt : WKT string of the GCOV's native UTM projection
    """
    with h5py.File(h5_path, "r") as h5:
        utm_wkt = _crs_wkt(h5)
        x_all   = h5[X_DS][:].astype(np.float64)
        y_all   = h5[Y_DS][:].astype(np.float64)

        # Convert the fire bbox from WGS84 degrees to UTM metres
        x_min, y_min, x_max, y_max = _bbox_wgs84_to_crs(bbox_wgs84, utm_wkt)
        col_sl = _coord_window(x_all, x_min, x_max)
        row_sl = _coord_window(y_all, y_min, y_max)

        def _read(ds_name: str) -> np.ndarray:
            ds   = h5[ds_name]
            arr  = ds[row_sl, col_sl].astype(np.float32)
            fill = ds.attrs.get("_FillValue", None)
            if fill is not None:
                arr[arr == float(fill)] = np.nan
            arr[arr <= 0] = np.nan   # zero / negative = unphysical in power scale
            return arr

        return {
            "hh":      _read(HH_DS),
            "hv":      _read(HV_DS),
            "x":       x_all[col_sl],
            "y":       y_all[row_sl],
            "utm_wkt": utm_wkt,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Compute RVI and normalise to p_fuel
# ═══════════════════════════════════════════════════════════════════════════

def compute_rvi(hh: np.ndarray, hv: np.ndarray) -> np.ndarray:
    """Dual-Polarisation Radar Vegetation Index.

    Formula (Kim & van Zyl 2009, adapted for HH+HV mode):

        RVI = 4·γ⁰_HV / (γ⁰_HH + γ⁰_HV)

    Why this works:
      - In uniform, bare ground the SAR pulse returns co-polarised (HH ≫ HV),
        so the denominator is almost entirely HH → RVI ≈ 0.
      - In dense canopy the pulse scatters off many branches at random
        orientations and depolarises: HV grows relative to HH → RVI → 1.
      - Dividing by the total power (HH + HV) makes RVI insensitive to
        overall signal strength (incidence angle, orbit-to-orbit system gain).

    Result is clipped to [0, 1] and masked (NaN) wherever either input is NaN.
    """
    denom = hh + hv
    with np.errstate(invalid="ignore", divide="ignore"):
        rvi = np.where(denom > 0, 4.0 * hv / denom, np.nan).astype(np.float32)
    return np.clip(rvi, 0.0, 1.0)


def normalise_percentile(arr: np.ndarray, plo: float = 2.0, phi: float = 98.0) -> np.ndarray:
    """Normalise to [0, 1] using percentile clipping.

    Why percentile clipping (not min/max)?
      SAR images always contain outlier pixels: open water (specular reflection
      → near-zero HV), urban double-bounce, cloud shadow.  Using the 2nd–98th
      percentile of *valid* pixels pulls the colour scale into the meaningful
      vegetation range and prevents one extreme pixel from washing out the map.

    The output is the p_fuel layer: a grid of [0, 1] values, one per NISAR
    pixel (~20–80 m spacing), ready to be bilinearly interpolated onto the
    cellular-automaton grid of the spread model.
    """
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return arr
    lo, hi = np.nanpercentile(valid, [plo, phi])
    normed = (arr - lo) / max(hi - lo, 1e-9)
    return np.clip(normed, 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Reproject to WGS84 and save as a multi-band GeoTIFF
# ═══════════════════════════════════════════════════════════════════════════

def _affine_from_xy(x: np.ndarray, y: np.ndarray) -> Affine:
    """Build a rasterio Affine transform from coordinate arrays.

    The Affine transform maps pixel indices → CRS coordinates.
    Convention: (col=0, row=0) is the **top-left corner** of the top-left pixel.
    For a descending y-array (north → south), dy is negative.
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])                      # negative if y is descending
    # top-left of pixel (0,0) = centre minus half a pixel
    x0 = float(x[0]) - dx / 2.0
    y0 = float(y[0]) - dy / 2.0
    return Affine(dx, 0.0, x0, 0.0, dy, y0)


def save_geotiff(
    bands: dict[str, np.ndarray],
    x: np.ndarray, y: np.ndarray,
    src_crs: str,
    dst_path: Path,
    dst_crs: str = WGS84,
) -> None:
    """Write several 2-D arrays to a multi-band GeoTIFF in WGS84.

    Why reproject to WGS84?
      All other layers in the project (WFIGS perimeters, FEDS polygons, VIIRS
      and GOES detections) use lat/lon WGS84.  Reprojecting the NISAR grid to
      the same CRS means the GeoTIFF overlays correctly in QGIS, the web map
      and the spread-model grid without any additional transformation at run time.

    Bands are labelled with the dict keys (hv_db, rvi, p_fuel) so any GIS tool
    can identify them.  Compression=LZW keeps the file manageable (~10–30 MB
    after reprojection to WGS84 at the original resolution).
    """
    names  = list(bands)
    arrays = [bands[n] for n in names]
    nrows, ncols = arrays[0].shape
    src_transform = _affine_from_xy(x, y)

    # Calculate bounding box coordinates
    left, right = float(min(x[0], x[-1])), float(max(x[0], x[-1]))
    bottom, top = float(min(y[0], y[-1])), float(max(y[0], y[-1]))
    dx = abs(float(x[1] - x[0])) / 2.0
    dy = abs(float(y[1] - y[0])) / 2.0
    left -= dx
    right += dx
    bottom -= dy
    top += dy

    # Calculate the destination grid: rasterio works out the best bounding box
    # and pixel size to minimise reprojection artefacts.
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, ncols, nrows, left=left, bottom=bottom, right=right, top=top,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        dst_path, "w", driver="GTiff",
        height=dst_h, width=dst_w, count=len(arrays),
        dtype="float32", crs=dst_crs, transform=dst_transform,
        nodata=float("nan"), compress="lzw",
    ) as dst:
        for band_i, (name, arr) in enumerate(zip(names, arrays), start=1):
            out = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
            reproject(
                source=arr,       destination=out,
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=float("nan"), dst_nodata=float("nan"),
            )
            dst.write(out, band_i)
            dst.update_tags(band_i, name=name)
    log.info("GeoTIFF saved → %s", dst_path)


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Diagnostic heatmap figure
# ═══════════════════════════════════════════════════════════════════════════

def plot_fuel_heatmap(
    rvi: np.ndarray, hv_db: np.ndarray,
    x: np.ndarray, y: np.ndarray,
    fire: FireConfig, track_id: str, phase: str,
    out_path: Path,
) -> None:
    """Two-panel diagnostic figure: HV backscatter (dB) and RVI heatmap.

    Left panel — raw HV in dB
        Shows the SAR signal as NASA delivers it.  Forests appear bright
        (-10 to -5 dB), shrubs/grassland medium (-15 to -12 dB),
        bare soil/rock dark (-22 to -18 dB).

    Right panel — RVI (= p_fuel candidate)
        The derived vegetation-density layer.  This is what goes into the
        spread model.  Yellow = sparse fuel, dark green = heavy fuel.

    Both panels share UTM coordinates (km) so the two images align exactly
    and the resolution of NISAR (~20–80 m) is visible.
    """
    import copy
    import geopandas as gpd

    # Convert UTM metres → km for readable axis labels
    ext_km = [x.min() / 1e3, x.max() / 1e3, y.min() / 1e3, y.max() / 1e3]
    origin = "upper" if y[0] > y[-1] else "lower"   # match the y-array direction

    # Downsample for fast display if resolution is very high (e.g. 10k x 10k)
    step = max(1, rvi.shape[0] // 1500)
    plot_hv = hv_db[::step, ::step] if step > 1 else hv_db
    plot_rvi = rvi[::step, ::step] if step > 1 else rvi

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="white")

    # Dark background for pixels outside the satellite swath
    cmap_hv = copy.copy(plt.cm.Greys_r)
    cmap_hv.set_bad("#222222")
    cmap_rvi = copy.copy(plt.cm.YlGn)
    cmap_rvi.set_bad("#222222")

    # Check if fire perimeter is available to overlay and frame
    perim_path = fire.root / "processed" / "wfigs_final_perimeter.geojson"
    perim_utm = None
    if perim_path.exists():
        try:
            gdf = gpd.read_file(perim_path)
            perim_utm = gdf.to_crs(fire.root / "processed" / "fuel_layer_A34_pre.tif" if False else 32610)
        except Exception:
            pass

    # --- Left: HV in dB ---
    im0 = axes[0].imshow(
        plot_hv, origin=origin, aspect="equal", extent=ext_km,
        cmap=cmap_hv, vmin=-25, vmax=-5,
    )
    cb0 = plt.colorbar(im0, ax=axes[0], label="γ⁰_HV  [dB]", shrink=0.78, pad=0.03)
    cb0.ax.tick_params(labelsize=8)
    axes[0].set_title(
        f"HV Backscatter (raw SAR, dB)\n{fire.name} | track {track_id} | {phase}-fire",
        fontsize=11, fontweight="bold", pad=10,
    )
    axes[0].set_xlabel("UTM Easting [km]", fontsize=9)
    axes[0].set_ylabel("UTM Northing [km]", fontsize=9)

    # --- Right: RVI / p_fuel ---
    im1 = axes[1].imshow(
        plot_rvi, origin=origin, aspect="equal", extent=ext_km,
        cmap=cmap_rvi, vmin=0, vmax=1,
    )
    cb1 = plt.colorbar(im1, ax=axes[1], label="RVI  (0 = bare soil, 1 = dense forest)", shrink=0.78, pad=0.03)
    cb1.ax.tick_params(labelsize=8)
    axes[1].set_title(
        f"Radar Vegetation Index → p_fuel\n{fire.name} | track {track_id} | {phase}-fire",
        fontsize=11, fontweight="bold", pad=10,
    )
    axes[1].set_xlabel("UTM Easting [km]", fontsize=9)
    axes[1].set_ylabel("UTM Northing [km]", fontsize=9)

    # Overlay perimeter if available
    if perim_utm is not None:
        for ax in axes:
            for geom in perim_utm.geometry:
                if hasattr(geom, "exterior") and geom.exterior is not None:
                    gx, gy = np.array(geom.exterior.xy)
                    ax.plot(gx / 1e3, gy / 1e3, color="red", linewidth=1.6, label="Official Perimeter")
                elif hasattr(geom, "geoms"):
                    for g in geom.geoms:
                        gx, gy = np.array(g.exterior.xy)
                        ax.plot(gx / 1e3, gy / 1e3, color="red", linewidth=1.6, label="Official Perimeter")
            # Zoom tightly to fire with 15 km padding if fire is within extent
            pb = perim_utm.total_bounds / 1e3
            margin = 15.0
            zoom_x = (max(ext_km[0], pb[0] - margin), min(ext_km[1], pb[2] + margin))
            zoom_y = (max(ext_km[2], pb[1] - margin), min(ext_km[3], pb[3] + margin))
            if zoom_x[1] > zoom_x[0] and zoom_y[1] > zoom_y[0]:
                ax.set_xlim(zoom_x)
                ax.set_ylim(zoom_y)
            ax.legend(loc="lower left", framealpha=0.9, fontsize=8)

    fig.suptitle(
        "NISAR L2 GCOV — vegetation / fuel density layer.\n"
        "RVI = 4·γ⁰_HV / (γ⁰_HH + γ⁰_HV). Volume scattering (HV) encodes biomass density.",
        fontsize=11, fontweight="bold", y=0.98,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Heatmap saved → %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator — build_fuel_layer
# ═══════════════════════════════════════════════════════════════════════════

def build_fuel_layer(fire: FireConfig, track_id: str, phase: str = "pre") -> Path:
    """Full pipeline for one fire × track × phase.

    1. Look up the granule ID from the already-processed nisar_passes.csv.
    2. Download the GCOV HDF5 (skipped if cached).
    3. Extract HH + HV layers clipped to the fire bbox.
    4. Compute RVI and normalise to p_fuel.
    5. Save a 3-band GeoTIFF (hv_db / rvi / p_fuel) in WGS84.
    6. Save a diagnostic heatmap PNG.

    Returns the path to the GeoTIFF.
    """
    # --- Look up which granule to use ---
    passes_csv = fire.root / "processed" / "nisar_passes.csv"
    if not passes_csv.exists():
        raise FileNotFoundError(
            f"nisar_passes.csv missing for {fire.slug}. "
            "Run  python scripts/build_ground_truth.py <fire>  first."
        )
    passes = pd.read_csv(passes_csv)
    sel = passes[(passes["track_id"] == track_id) & (passes["phase"] == phase)]
    if sel.empty:
        available = passes[["track_id", "phase"]].drop_duplicates().to_string(index=False)
        raise ValueError(
            f"No {phase!r} pass for track {track_id!r} in {fire.slug}.\n"
            f"Available:\n{available}"
        )

    # For pre-fire: use the LATEST available pass (closest to ignition → most
    # representative of the actual fuel state just before the fire started).
    row = sel.sort_values("time_utc").iloc[-1]
    file_id = row["file_id"]
    log.info(
        "Fire=%s  track=%s  phase=%s  pass=%s",
        fire.slug, track_id, phase,
        pd.Timestamp(row["time_utc"]).strftime("%Y-%m-%d %H:%M UTC"),
    )

    # --- Step 1: download ---
    h5_path = download_gcov(file_id, fire.dir("raw") / "nisar")

    # --- Step 2: load backscatter clipped to fire bbox ---
    layers = load_backscatter(h5_path, fire.bbox)
    hh, hv  = layers["hh"], layers["hv"]
    x, y    = layers["x"], layers["y"]
    utm_wkt = layers["utm_wkt"]

    # --- Step 3: compute indices ---
    rvi    = compute_rvi(hh, hv)
    p_fuel = normalise_percentile(rvi)

    # HV in dB for visualisation (log10 of linear power; NaN already set)
    with np.errstate(invalid="ignore"):
        hv_db = (10.0 * np.log10(hv)).astype(np.float32)

    # --- Step 4: save GeoTIFF ---
    stem = f"fuel_layer_{track_id}_{phase}"
    tif_path = fire.dir("processed") / (stem + ".tif")
    save_geotiff(
        {"hv_db": hv_db, "rvi": rvi, "p_fuel": p_fuel},
        x, y, utm_wkt, tif_path,
    )

    # --- Step 5: heatmap figure ---
    fig_path = fire.dir("figures") / (stem + ".png")
    plot_fuel_heatmap(rvi, hv_db, x, y, fire, track_id, phase, fig_path)

    return tif_path


# ═══════════════════════════════════════════════════════════════════════════
# CLI  (python -m firedata.vegetation <fire> <track> [--phase pre])
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    ap = argparse.ArgumentParser(
        description="Build NISAR vegetation/fuel layer for one fire + track."
    )
    ap.add_argument("fire",  choices=FIRES,
                    help="Fire slug, e.g. bug_fire")
    ap.add_argument("track",
                    help="Track ID from nisar_passes.csv, e.g. A34 or D42")
    ap.add_argument("--phase", default="pre", choices=["pre", "during", "post"],
                    help="NISAR pass phase (default: pre)")
    args = ap.parse_args()

    tif = build_fuel_layer(FIRES[args.fire], args.track, args.phase)
    print(f"\n✓  Fuel layer → {tif}")
