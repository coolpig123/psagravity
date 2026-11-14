"""NISAR L2 GCOV layer extractor for the Alexandridis wildfire spread model.

Generates an xarray.Dataset matching the exact model contract:
  - nisar_cover_pct     (%)     : vegetation canopy cover / density from HV volume scattering
  - nisar_soil_moisture (m3/m3) : volumetric soil moisture content from HH backscatter
  - nisar_burned        (0/1)   : 1 = already burned / non-combustible (cannot burn), 0 = combustible fuel
  - nisar_combustible   (0/1)   : 1 = ready-to-burn fuel, 0 = already burned or non-combustible

Can be passed directly into the model:
    predict(..., nisar=ds)
or saved to NetCDF:
    ds.to_netcdf("file.nc", engine="h5netcdf")
and used via CLI:
    --nisar file.nc

Grid alignment:
  Samples onto a shared UTM Grid (compatible with geelayers.Grid and the model grid).
  Coordinates: x (easting, m), y (northing, m, north-to-south), lat(y,x), lon(y,x).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import map_coordinates
import xarray as xr

from .config import FIRES, FireConfig
from .vegetation import FREQ_PATH, HH_DS, HV_DS, X_DS, Y_DS, _crs_wkt, download_gcov

log = logging.getLogger(__name__)


def compute_cover_pct(hh: np.ndarray, hv: np.ndarray) -> np.ndarray:
    """Vegetation canopy cover percentage (0-100%).

    Derived from Dual-pol Radar Vegetation Index (RVI):
        RVI = 4 * HV / (HH + HV)
        cover_pct = clip(RVI * 100, 0, 100)
    """
    denom = hh + hv
    with np.errstate(invalid="ignore", divide="ignore"):
        rvi = np.where(denom > 0, 4.0 * hv / denom, np.nan).astype(np.float32)
    return np.clip(rvi * 100.0, 0.0, 100.0).astype(np.float32)


def compute_soil_moisture(
    hh: np.ndarray,
    hh_min_db: float = -18.0,
    hh_max_db: float = -5.0,
    sm_min: float = 0.02,
    sm_max: float = 0.45,
) -> np.ndarray:
    """Volumetric soil moisture content in m3/m3 (0.02 to 0.45).

    L-band HH backscatter is strongly governed by soil surface dielectric constant.
    Over natural soil:
      HH <= -18 dB -> very dry soil (~0.02 - 0.03 m3/m3, near wilting point)
      HH >= -5 dB  -> saturated soil (~0.40 - 0.45 m3/m3, field capacity)
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        hh_db = np.where(hh > 0, 10.0 * np.log10(hh), np.nan).astype(np.float32)
        norm = np.clip((hh_db - hh_min_db) / (hh_max_db - hh_min_db), 0.0, 1.0)
        sm = sm_min + (sm_max - sm_min) * norm
    return sm.astype(np.float32)


def compute_burned_mask(
    hh: np.ndarray,
    hv: np.ndarray,
    rvi: np.ndarray | None = None,
    delta_hv_db: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantized 0 and 1 masks for burned / non-combustible ground.

    Identifies:
      1. Water bodies: specular reflection -> very low HH (< -20 dB) and low HV (< -26 dB)
      2. Bare rock / non-vegetated barren: RVI < 0.10 and HV < -22 dB
      3. Active / historical burn scars:
         - From pre/post change (if delta_hv_db provided): delta_HV <= -3.0 dB
         - Or very low volume scattering in scar areas

    Returns
    -------
    (burned, combustible):
      burned:      1 = already burned or non-combustible (cannot burn), 0 = combustible fuel
      combustible: 1 = combustable / ready-to-burn fuel, 0 = non-combustible or already burned
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        hh_db = 10.0 * np.log10(np.where(hh > 0, hh, np.nan))
        hv_db = 10.0 * np.log10(np.where(hv > 0, hv, np.nan))
        if rvi is None:
            denom = hh + hv
            rvi = np.where(denom > 0, 4.0 * hv / denom, 0.0)

    # Non-combustible / unburnable conditions
    is_water = (hh_db < -20.0) & (hv_db < -25.0)
    is_barren = (rvi < 0.10) & (hv_db < -22.0)
    is_scar = np.zeros_like(hh_db, dtype=bool)
    if delta_hv_db is not None:
        is_scar = delta_hv_db <= -3.0

    unburnable = is_water | is_barren | is_scar | np.isnan(hh_db)

    burned = np.where(unburnable, np.int16(1), np.int16(0))
    combustible = np.where(unburnable, np.int16(0), np.int16(1))
    return burned, combustible


def resample_gcov_to_grid(
    h5_path: Path,
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_crs: str | int = "EPSG:32610",
) -> tuple[np.ndarray, np.ndarray]:
    """Read HH and HV from a GCOV HDF5 and resample directly onto target grid coordinates.

    Parameters
    ----------
    h5_path: Path to NISAR L2 GCOV HDF5 file
    target_x: 1-D array of target easting (metres)
    target_y: 1-D array of target northing (metres, north-to-south)
    target_crs: EPSG code or CRS string of target_x, target_y

    Returns
    -------
    (hh_grid, hv_grid): 2-D float32 arrays of shape (len(target_y), len(target_x))
    """
    with h5py.File(h5_path, "r") as h5:
        grp = h5[FREQ_PATH]
        file_crs = _crs_wkt(h5)
        x_all = grp[X_DS][:].astype(np.float64)
        y_all = grp[Y_DS][:].astype(np.float64)

        dx = float(x_all[1] - x_all[0])
        dy = float(y_all[1] - y_all[0])

        # If CRS differs, project target coords to file native CRS
        if str(target_crs).upper() not in file_crs.upper():
            tr = Transformer.from_crs(target_crs, file_crs, always_xy=True)
            xx_tgt, yy_tgt = np.meshgrid(target_x, target_y)
            fx, fy = tr.transform(xx_tgt, yy_tgt)
        else:
            fx, fy = np.meshgrid(target_x, target_y)

        # Slice sub-region from HDF5 with safety margin
        margin = 10
        c0 = max(0, int((fx.min() - x_all[0]) / dx) - margin)
        c1 = min(len(x_all), int((fx.max() - x_all[0]) / dx) + margin)
        r0 = max(0, int((fy.max() - y_all[0]) / dy) - margin)
        r1 = min(len(y_all), int((fy.min() - y_all[0]) / dy) + margin)

        r_sl = slice(min(r0, r1), max(r0, r1))
        c_sl = slice(min(c0, c1), max(c0, c1))

        hh_sub = grp[HH_DS][r_sl, c_sl].astype(np.float32)
        hv_sub = grp[HV_DS][r_sl, c_sl].astype(np.float32)

        # Mask non-physical / fill values
        hh_sub[hh_sub <= 0] = np.nan
        hv_sub[hv_sub <= 0] = np.nan

        sub_x = x_all[c_sl]
        sub_y = y_all[r_sl]
        sub_dx = float(sub_x[1] - sub_x[0])
        sub_dy = float(sub_y[1] - sub_y[0])

        # Map target coordinates to array indices in the extracted sub-slice
        cols = (fx - sub_x[0]) / sub_dx
        rows = (fy - sub_y[0]) / sub_dy

        hh_grid = map_coordinates(hh_sub, [rows, cols], order=1, cval=np.nan).astype(np.float32)
        hv_grid = map_coordinates(hv_sub, [rows, cols], order=1, cval=np.nan).astype(np.float32)

        return hh_grid, hv_grid


def extract_nisar_dataset(
    fire: str | FireConfig = "bug_fire",
    track: str = "A34",
    phase: str = "pre",
    grid: Any = None,
    h5_path: Path | None = None,
    res_m: float = 100.0,
    radius_km: float = 20.0,
    out_nc: Path | str | None = None,
) -> xr.Dataset:
    """Build the NISAR model input Dataset for the Alexandridis spread model.

    Returns an xarray.Dataset containing:
      - nisar_cover_pct     (%):     vegetation cover -> density
      - nisar_soil_moisture (m3/m3): soil moisture -> dryness
      - nisar_burned        (0/1):   already burned -> can't burn again
      - nisar_combustible   (0/1):   combustible / ready-to-use fuel

    Parameters
    ----------
    fire: fire slug (e.g. 'bug_fire') or FireConfig
    track: track ID (e.g. 'A34')
    phase: pass phase ('pre', 'during', 'post')
    grid: optional geelayers.Grid instance. If None, builds a grid around ignition point.
    h5_path: optional path to already downloaded H5. If None, looks in raw cache or downloads.
    res_m: grid cell resolution in metres (default 100 m)
    radius_km: radius around fire ignition if grid is auto-constructed (default 20 km)
    out_nc: optional file path to save as NetCDF (.nc)
    """
    F = FIRES[fire] if isinstance(fire, str) else fire

    # 1. Determine or build the Grid
    if grid is None:
        try:
            import sys
            repo_root = Path(__file__).resolve().parents[3]
            sys.path.insert(0, str(repo_root / "gee_layers" / "src"))
            from geelayers import Grid
            grid = Grid.from_point(lat=F.ignition_lat, lon=F.ignition_lon, radius_km=radius_km, res_m=res_m)
        except Exception:
            # Fallback simple grid if geelayers Grid is unavailable
            from pyproj import Transformer
            zone = int((F.ignition_lon + 180) // 6) + 1
            epsg = 32600 + zone
            tr = Transformer.from_crs(4326, epsg, always_xy=True)
            cx, cy = tr.transform(F.ignition_lon, F.ignition_lat)
            half = radius_km * 1000.0
            x0, x1 = cx - half, cx + half
            y0, y1 = cy - half, cy + half
            xs = np.arange(x0, x1 + res_m, res_m)
            ys = np.arange(y1, y0 - res_m, -res_m)
            class SimpleGrid:
                pass
            grid = SimpleGrid()
            grid.epsg = epsg
            grid.res = res_m
            grid.x = xs
            grid.y = ys
            grid.width = len(xs)
            grid.height = len(ys)
            grid.crs = f"EPSG:{epsg}"
            xx, yy = np.meshgrid(xs, ys)
            inv_tr = Transformer.from_crs(epsg, 4326, always_xy=True)
            grid.latlon = lambda: inv_tr.transform(xx, yy)[::-1]

    # 2. Locate or download the H5 granule
    file_id = None
    if h5_path is None:
        passes_csv = F.root / "processed" / "nisar_passes.csv"
        if not passes_csv.exists():
            raise FileNotFoundError(f"nisar_passes.csv missing for {F.slug}.")
        passes = pd.read_csv(passes_csv)
        sel = passes[(passes["track_id"] == track) & (passes["phase"] == phase)]
        if sel.empty:
            raise ValueError(f"No {phase} pass found for track {track} in {F.slug}")
        row = sel.sort_values("time_utc").iloc[-1]
        file_id = row["file_id"]
        h5_path = download_gcov(file_id, F.dir("raw") / "nisar")

    log.info("Extracting NISAR layers from: %s", h5_path.name)

    # 3. Resample onto the Grid
    target_crs = getattr(grid, "crs", f"EPSG:{getattr(grid, 'epsg', 32610)}")
    hh, hv = resample_gcov_to_grid(h5_path, grid.x, grid.y, target_crs=target_crs)

    # 4. Compute the model variables
    rvi = np.where((hh + hv) > 0, 4.0 * hv / (hh + hv), 0.0).astype(np.float32)
    cover_pct = compute_cover_pct(hh, hv)
    soil_moisture = compute_soil_moisture(hh)
    burned, combustible = compute_burned_mask(hh, hv, rvi=rvi)

    # 5. Build coordinates
    lat, lon = grid.latlon()

    coords = {
        "y": ("y", grid.y.astype(np.float64), {"units": "m", "long_name": "UTM Northing"}),
        "x": ("x", grid.x.astype(np.float64), {"units": "m", "long_name": "UTM Easting"}),
        "lat": (("y", "x"), lat.astype(np.float32), {"units": "degrees_north", "long_name": "Latitude"}),
        "lon": (("y", "x"), lon.astype(np.float32), {"units": "degrees_east", "long_name": "Longitude"}),
    }

    data_vars = {
        "nisar_cover_pct": (
            ("y", "x"),
            cover_pct,
            {"units": "%", "long_name": "vegetation cover -> density", "description": "Canopy cover percentage from NISAR L-band HV volume scattering (0-100%)"},
        ),
        "nisar_soil_moisture": (
            ("y", "x"),
            soil_moisture,
            {"units": "m3/m3", "long_name": "soil moisture -> dryness", "description": "Volumetric soil water content from NISAR L-band HH backscatter (m3/m3)"},
        ),
        "nisar_burned": (
            ("y", "x"),
            burned,
            {"units": "1", "long_name": "already burned -> can't burn again", "description": "Binary mask: 1 = already burned or non-combustible (cannot burn), 0 = combustible fuel"},
        ),
        "nisar_combustible": (
            ("y", "x"),
            combustible,
            {"units": "1", "long_name": "combustible fuel mask", "description": "Binary mask: 1 = ready-to-burn combustible fuel, 0 = non-combustible or already burned"},
        ),
    }

    attrs = {
        "title": f"NISAR L2 Model Inputs for {F.name}",
        "crs": target_crs,
        "res_m": float(grid.res),
        "fire": F.slug,
        "track_id": track,
        "phase": phase,
        "granule_id": file_id or h5_path.stem,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_usage": "Pass to spread model via predict(..., nisar=ds) or CLI --nisar file.nc",
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    if out_nc is not None:
        out_path = Path(out_nc)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_path, engine="h5netcdf")
        log.info("Saved NISAR model dataset -> %s", out_path)

    return ds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Extract NISAR model inputs (cover %, soil moisture, burned mask).")
    ap.add_argument("fire", choices=list(FIRES), help="Fire name, e.g. bug_fire")
    ap.add_argument("--track", default="A34", help="Track ID, e.g. A34")
    ap.add_argument("--phase", default="pre", choices=["pre", "post"], help="NISAR pass phase")
    ap.add_argument("--res", type=float, default=100.0, help="Grid resolution in metres (default 100m)")
    ap.add_argument("--radius-km", type=float, default=20.0, help="Radius around ignition in km (default 20km)")
    ap.add_argument("--out", default=None, help="Output NetCDF file path (.nc)")
    args = ap.parse_args()

    ds = extract_nisar_dataset(
        fire=args.fire,
        track=args.track,
        phase=args.phase,
        res_m=args.res,
        radius_km=args.radius_km,
        out_nc=args.out,
    )
    print(ds)
