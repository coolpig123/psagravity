"""The actively burning edge of a fire, from VIIRS 375 m hotspots (NASA FIRMS).

A drawn or FEDS polygon treats every edge alike, but a fire usually burns on part of its perimeter only; the rest is
cold or contained. The active front is the cells of the burned area with a recent VIIRS detection, plus the outer-rim
cells next to one: FEDS (and people drawing polygons) buffer the hotspots by ~500 m, so a detection often sits 1-2
cells inside the polygon, where it has no unburned neighbours.

    front, frp = front_from_points(ds, lat, lon, frp_mw)          # offline: points you already have
    ds = attach_active_front(ds, start, lookback_h=24, notes=n)   # live: FIRMS -> ds["active_front"], ds["frp_mw"]

spread_probabilities() and simulate_spread() then spread from ds["active_front"] only (the rest of the burned area
cannot re-burn). Without a front (no FIRMS key, FIRMS down, no hotspot near the fire) the whole burned area spreads,
as before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_LOOKBACK_H = 24.0


def _neighbours_any(a: np.ndarray, diagonal: bool) -> np.ndarray:
    """b[r, c] = any of the 4 (or 8) neighbours of (r, c) is True; outside the grid = False."""
    p = np.pad(a, 1, constant_values=False)
    out = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    if diagonal:
        out = out | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    return out


def _neighbours_max(a: np.ndarray) -> np.ndarray:
    """Max of the cell and its 4 neighbours (outside the grid = 0)."""
    p = np.pad(a, 1, constant_values=0.0)
    return np.maximum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])


def outer_rim(burned: np.ndarray) -> np.ndarray:
    """Burned cells with at least one unburned (or off-grid) neighbour among their 8."""
    burned = np.asarray(burned, bool)
    return burned & _neighbours_any(np.pad(~burned, 1, constant_values=True), True)[1:-1, 1:-1]


def project_front(burned: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Active front: detected cells, plus outer-rim cells of `burned` next to (4-neighbourhood) a detected cell.
    Not restricted to `burned`: a detection outside it is a burning cell the polygon missed."""
    detected = np.asarray(detected, bool)
    return detected | (outer_rim(burned) & _neighbours_any(detected, False))


def points_to_grid(ds: xr.Dataset, lat, lon, frp_mw=None) -> tuple[np.ndarray, np.ndarray]:
    """VIIRS points -> (detected bool (y, x), frp float32 (y, x): max FRP in the cell, 0 where none)."""
    from pyproj import Transformer
    x, y = ds.x.values, ds.y.values
    detected = np.zeros((y.size, x.size), bool)
    frp = np.zeros((y.size, x.size), np.float32)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if lat.size == 0 or x.size < 2 or y.size < 2:
        return detected, frp
    px, py = Transformer.from_crs(4326, ds.attrs["crs"], always_xy=True).transform(lon, lat)
    col = np.round((px - x[0]) / (x[1] - x[0])).astype(int)
    row = np.round((py - y[0]) / (y[1] - y[0])).astype(int)
    ok = (row >= 0) & (row < y.size) & (col >= 0) & (col < x.size)
    detected[row[ok], col[ok]] = True
    if frp_mw is not None:
        np.maximum.at(frp, (row[ok], col[ok]), np.nan_to_num(np.asarray(frp_mw, np.float32)[ok]))
    return detected, frp


def front_from_points(ds: xr.Dataset, lat, lon, frp_mw=None,
                      burned: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """(active_front bool, frp_mw float32) on the grid of `ds`, or None if no detection touches the burned area.

    Only detections in, or next to, the burned area count (default ds["initial_burned"]): a hotspot elsewhere in the
    grid is another fire. A rim cell activated by a neighbouring detection gets that detection's FRP."""
    if burned is None:
        burned = ds["initial_burned"].values
    burned = np.asarray(burned, bool)
    detected, frp = points_to_grid(ds, lat, lon, frp_mw)
    detected &= burned | _neighbours_any(burned, True)
    front = project_front(burned, detected)
    if not front.any():
        return None
    own = np.where(detected, frp, 0.0)
    frp = np.where(detected, own, np.where(front, _neighbours_max(own), 0.0)).astype(np.float32)
    return front, frp


def grid_bbox(ds: xr.Dataset) -> tuple[float, float, float, float]:
    """(west, south, east, north) in degrees of the grid."""
    from pyproj import Transformer
    half = float(ds.attrs.get("res_m", abs(float(ds.x[1] - ds.x[0])))) / 2
    x, y = ds.x.values, ds.y.values
    xs = [x.min() - half, x.max() + half, x.max() + half, x.min() - half]
    ys = [y.max() + half, y.max() + half, y.min() - half, y.min() - half]
    lon, lat = Transformer.from_crs(ds.attrs["crs"], 4326, always_xy=True).transform(xs, ys)
    return float(min(lon)), float(min(lat)), float(max(lon)), float(max(lat))


def firms_viirs_points(bbox, start, lookback_h: float = DEFAULT_LOOKBACK_H) -> pd.DataFrame:
    """VIIRS SNPP / NOAA-20 / NOAA-21 points in bbox over (start - lookback_h, start], from the FIRMS area API.

    Needs real_fires (firedata) and FIRMS_MAP_KEY (real_fires/.env); raises otherwise."""
    from firedata.config import get_secret
    from firedata.firms import SATELLITES, data_availability, fetch_area, pick_source, tidy_viirs
    key = get_secret("FIRMS_MAP_KEY")
    t1 = pd.Timestamp(start)
    t1 = t1.tz_localize("UTC") if t1.tzinfo is None else t1.tz_convert("UTC")
    t0 = t1 - pd.Timedelta(hours=lookback_h)
    day0 = t0.tz_localize(None).normalize()
    days = int((t1.tz_localize(None).normalize() - day0).days) + 1
    avail = data_availability(key)
    frames = []
    for sat in SATELLITES:
        source = pick_source(sat, t1, avail) or pick_source(sat, day0, avail)
        if source is None:
            continue
        raw = fetch_area(source, bbox, day0, min(days, 5), key)
        if not raw.empty and "latitude" in raw:
            raw["firms_source"] = source
            frames.append(raw)
    if not frames:
        return pd.DataFrame(columns=["time_utc", "lat", "lon", "frp_mw"])
    pts = tidy_viirs(pd.concat(frames, ignore_index=True))
    return pts[(pts["time_utc"] > t0) & (pts["time_utc"] <= t1)].reset_index(drop=True)


def fetch_active_front(ds: xr.Dataset, start, lookback_h: float = DEFAULT_LOOKBACK_H, notes: list[str] | None = None,
                       points_fn=firms_viirs_points) -> tuple[np.ndarray, np.ndarray] | None:
    """(active_front, frp_mw) from VIIRS hotspots in the lookback_h before `start`, or None (never raises).

    None when real_fires is not installed, FIRMS_MAP_KEY is missing, FIRMS fails, or no hotspot touches the burned
    area (between satellite passes, under smoke, a contained fire); the reason is appended to `notes`."""
    notes = notes if notes is not None else []
    try:
        pts = points_fn(grid_bbox(ds), start, lookback_h)
    except ImportError as e:
        notes.append(f"active front skipped: real_fires is not installed (pip install -e real_fires): {e}")
        return None
    except Exception as e:  # no FIRMS_MAP_KEY, FIRMS down, ...
        notes.append(f"active front skipped: {type(e).__name__}: {e}")
        return None
    got = None if pts is None or len(pts) == 0 else front_from_points(
        ds, pts["lat"].values, pts["lon"].values, pts["frp_mw"].values if "frp_mw" in pts else None)
    if got is None:
        notes.append(f"active front: no VIIRS hotspot at the fire in the {lookback_h:g} h before the start; "
                     "spreading from the whole perimeter")
    return got


def attach_active_front(ds: xr.Dataset, start, lookback_h: float = DEFAULT_LOOKBACK_H,
                        notes: list[str] | None = None, points_fn=firms_viirs_points) -> xr.Dataset:
    """ds + active_front (y, x) int16 and frp_mw (y, x) when VIIRS shows where the fire burns; else ds unchanged.
    attrs["active_front_note"] says what was used or why not."""
    notes = notes if notes is not None else []
    n0 = len(notes)
    got = fetch_active_front(ds, start, lookback_h, notes, points_fn)
    out = ds.copy()
    if got is None:
        out.attrs["active_front_note"] = "; ".join(notes[n0:])
        return out
    front, frp = got
    out["active_front"] = (("y", "x"), front.astype(np.int16),
                           {"description": f"cells burning in VIIRS in the {lookback_h:g} h before the start"})
    out["frp_mw"] = (("y", "x"), frp, {"description": "max VIIRS fire radiative power (MW) on active-front cells"})
    burned = np.asarray(ds["initial_burned"].values, bool) if "initial_burned" in ds else np.zeros_like(front)
    inside, outside = int((front & burned).sum()), int((front & ~burned).sum())
    out.attrs["active_front_note"] = (f"active front from VIIRS ({lookback_h:g} h): {inside} of the polygon's "
                                      f"{int(burned.sum())} cells" + (f" + {outside} just outside it" if outside else ""))
    return out
