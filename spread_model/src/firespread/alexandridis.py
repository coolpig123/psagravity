"""Alexandridis et al. (2008) cellular-automaton spread probability.

For a burning cell i and each of its 8 neighbours j:

    P(i -> j) = P0 * (1 + P_veg[j]) * (1 + P_den[j]) * P_wind(i -> j) * P_slope(i -> j) * P_mask[j]

clipped to [0, 1]. An unburned cell j catches fire in the next step with probability
1 - prod_i (1 - P(i -> j)) over its burning neighbours i.

Inputs come from one xarray Dataset on one grid (geelayers.get_fire_inputs), optionally with
NISAR variables added by the NISAR module on the same grid (see NISAR_VARIABLES).
Everything is numpy array maths on whole grids (no loops over cells), so it can be ported to
torch / jax as-is to learn the parameters by gradient descent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

# 8 neighbours: name, row offset (row 0 = north), column offset (column 0 = west), compass bearing
DIRECTIONS = (("N", -1, 0, 0.0), ("NE", -1, 1, 45.0), ("E", 0, 1, 90.0), ("SE", 1, 1, 135.0),
              ("S", 1, 0, 180.0), ("SW", 1, -1, 225.0), ("W", 0, -1, 270.0), ("NW", -1, -1, 315.0))
DIR_NAMES = [d[0] for d in DIRECTIONS]

# Optional variables the NISAR module can add to the Dataset (same grid). Used first when present.
NISAR_VARIABLES = {
    "nisar_burned": "1 = already burned (burn scar from NISAR change detection) -> P_mask = 0",
    "nisar_cover_pct": "vegetation cover %, e.g. from HV backscatter / biomass -> used for P_den instead of LANDFIRE",
}

# ESA WorldCover classes (geelayers `landcover`)
WC_NOT_BURNABLE = (50, 60, 70, 80)   # built-up, bare, snow/ice, water


@dataclass
class AlexandridisParams:
    """Model constants. Defaults are the published values (Alexandridis et al. 2008, Spetses fire, 1990);
    they are the starting point for learning per-region values from real fires."""
    p0: float = 0.58          # P0 (the paper's p_h): spread probability on flat ground, no wind, reference fuel
    c1: float = 0.045         # wind speed term, s/m
    c2: float = 0.131         # wind direction term, s/m
    a: float = 0.078          # slope term, 1/degree
    # P_veg by fuel_lifeform (geelayers vegetation_cover): paper classes agricultural -0.3, thickets 0, forest 0.4
    veg_by_lifeform: dict[int, float] = field(default_factory=lambda: {1: 0.4, 2: 0.0, 3: -0.3})
    # fallback when LANDFIRE has no data (outside the US): ESA WorldCover class -> P_veg
    veg_by_worldcover: dict[int, float] = field(default_factory=lambda: {
        10: 0.4, 95: 0.4, 20: 0.0, 30: -0.3, 40: -0.3, 90: -0.3, 100: -0.3})
    # P_den: paper classes sparse -0.4, normal 0, dense 0.3; split by vegetation cover %
    den_sparse: float = -0.4
    den_normal: float = 0.0
    den_dense: float = 0.3
    sparse_below_pct: float = 30.0
    dense_from_pct: float = 60.0
    # fallback when no cover % at all: NDVI thresholds
    sparse_below_ndvi: float = 0.3
    dense_from_ndvi: float = 0.6


# ---- the five factors ----------------------------------------------------------------------

def p_veg(ds: xr.Dataset, params: AlexandridisParams) -> tuple[np.ndarray, str]:
    """P_veg per cell (the cell that would catch fire). Non-burnable cells get 0 here; P_mask removes them."""
    shape = (ds.sizes["y"], ds.sizes["x"])
    out = np.full(shape, np.nan, np.float32)
    used = []
    if "fuel_lifeform" in ds:
        lf = ds["fuel_lifeform"].values
        for code, value in params.veg_by_lifeform.items():
            out[lf == code] = value
        out[lf == 0] = 0.0
        used.append("fuel_lifeform (LANDFIRE EVC)")
    if "landcover" in ds and np.isnan(out).any():
        lc = ds["landcover"].values
        for code, value in params.veg_by_worldcover.items():
            out[np.isnan(out) & (lc == code)] = value
        used.append("landcover (WorldCover) where LANDFIRE is missing")
    n_default = int(np.isnan(out).sum())
    out[np.isnan(out)] = 0.0   # unknown vegetation -> reference fuel (thickets)
    if n_default:
        used.append(f"{n_default} cells unknown -> 0")
    return out, "; ".join(used) or "none -> 0"


def p_den(ds: xr.Dataset, params: AlexandridisParams) -> tuple[np.ndarray, str]:
    """P_den per cell from vegetation cover %: sparse / normal / dense."""
    shape = (ds.sizes["y"], ds.sizes["x"])
    cover = np.full(shape, np.nan, np.float32)
    used = []
    for name in ("nisar_cover_pct", "cover_pct"):
        if name in ds:
            v = ds[name].values.astype(np.float32)
            fill = np.isnan(cover) & ~np.isnan(v)
            cover[fill] = v[fill]
            used.append(name)
    out = np.full(shape, params.den_normal, np.float32)
    known = ~np.isnan(cover)
    out[known & (cover < params.sparse_below_pct)] = params.den_sparse
    out[known & (cover >= params.dense_from_pct)] = params.den_dense
    if "ndvi" in ds and (~known).any():
        ndvi = ds["ndvi"].values
        m = ~known & ~np.isnan(ndvi)
        out[m & (ndvi < params.sparse_below_ndvi)] = params.den_sparse
        out[m & (ndvi >= params.dense_from_ndvi)] = params.den_dense
        used.append("ndvi where no cover %")
    return out, ", ".join(used) or "none -> normal"


def p_mask(ds: xr.Dataset, burning: np.ndarray) -> tuple[np.ndarray, str]:
    """P_mask per cell: 1 = can catch fire, 0 = cannot (no fuel, water, built-up, burning or already burned)."""
    mask = np.ones(burning.shape, np.float32)
    used = ["burning now"]
    mask[burning.astype(bool)] = 0
    if "fuel_lifeform" in ds:
        mask[ds["fuel_lifeform"].values == 0] = 0
        used.append("fuel_lifeform = 0")
    if "landcover" in ds:
        mask[np.isin(ds["landcover"].values, WC_NOT_BURNABLE)] = 0
        used.append(f"landcover in {WC_NOT_BURNABLE}")
    if "nisar_burned" in ds:
        mask[ds["nisar_burned"].values == 1] = 0
        used.append("nisar_burned")
    return mask, ", ".join(used)


def p_wind(u: np.ndarray, v: np.ndarray, bearing_deg: float, params: AlexandridisParams) -> np.ndarray:
    """P_wind = exp(c1 V) * exp(c2 V (cos(theta) - 1)).

    V = wind speed (m/s); theta = angle between the direction the wind blows TOWARD and the
    direction of spread (bearing from the burning cell to the neighbour).
    """
    speed = np.hypot(u, v)
    toward = np.degrees(np.arctan2(u, v))            # bearing the wind blows toward (u east, v north)
    theta = np.radians(bearing_deg - toward)
    return np.exp(params.c1 * speed) * np.exp(params.c2 * speed * (np.cos(theta) - 1))


def slope_angle_deg(elev_from: np.ndarray, elev_to: np.ndarray, distance_m: float) -> np.ndarray:
    """theta_s = atan((E_to - E_from) / l) in degrees; > 0 uphill. l = cell size, or cell size * sqrt(2) diagonally."""
    return np.degrees(np.arctan((elev_to - elev_from) / distance_m))


def p_slope(theta_s_deg: np.ndarray, params: AlexandridisParams) -> np.ndarray:
    """P_slope = exp(a * theta_s)."""
    return np.exp(params.a * theta_s_deg)


# ---- grid helpers ---------------------------------------------------------------------------

def shift(a: np.ndarray, dr: int, dc: int, fill: float = 0.0) -> np.ndarray:
    """b[r, c] = a[r + dr, c + dc] (the neighbour at offset (dr, dc)); outside the grid -> fill."""
    h, w = a.shape[-2:]
    out = np.full(a.shape, fill, dtype=np.result_type(a.dtype, np.float32))
    out[..., max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)] = \
        a[..., max(0, dr):h - max(0, -dr), max(0, dc):w - max(0, -dc)]
    return out


def _at_time(ds: xr.Dataset, time) -> xr.Dataset:
    if "time" not in ds.dims:
        return ds
    if time is None:
        return ds.isel(time=0)
    t = pd.Timestamp(time)
    t = t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t
    return ds.sel(time=t, method="pad")   # last weather step at or before `time`


def _compass(bearing: float) -> str:
    return DIR_NAMES[int(((bearing % 360) + 22.5) // 45) % 8]


# ---- the model ------------------------------------------------------------------------------

def spread_probabilities(ds: xr.Dataset, burning: np.ndarray | None = None, time=None,
                         params: AlexandridisParams | None = None) -> xr.Dataset:
    """Spread probability from every burning cell to each of its 8 neighbours, for one CA step.

    ds:      inputs on one grid (geelayers.get_fire_inputs output, + optional NISAR variables).
    burning: bool/int (y, x) mask of burning cells; default ds["initial_burned"].
    time:    which weather step to use (last one at or before `time`); default the first.

    Returns a Dataset with
      p_veg, p_den, p_mask (y, x)               factors of the cell that would catch fire
      p_wind, p_slope, p_spread (direction, y, x) for the burning cell at (y, x) spreading toward `direction`
      ignition_prob (y, x)                       chance each unburned cell catches fire in the next step
    and attrs summarising the most likely spread direction.
    """
    params = params or AlexandridisParams()
    snap = _at_time(ds, time)
    if burning is None:
        if "initial_burned" not in ds:
            raise ValueError("no burning cells: pass burning=... or a Dataset with initial_burned")
        burning = ds["initial_burned"].values
    burning = np.asarray(burning).astype(bool)
    if not burning.any():
        raise ValueError("no burning cells")
    res = float(ds.attrs.get("res_m", abs(float(ds.x[1] - ds.x[0]))))
    shape = burning.shape

    veg, veg_src = p_veg(snap, params)
    den, den_src = p_den(snap, params)
    mask, mask_src = p_mask(snap, burning)
    if "wind_u_ms" in snap and "wind_v_ms" in snap:
        u = np.nan_to_num(snap["wind_u_ms"].values.astype(np.float32))
        v = np.nan_to_num(snap["wind_v_ms"].values.astype(np.float32))
        wind_src = "wind_u_ms, wind_v_ms at the burning cell"
    else:
        u = v = np.zeros(shape, np.float32)
        wind_src = "no wind in inputs -> P_wind = 1"
    if "elevation_m" in snap:
        elev = snap["elevation_m"].values.astype(np.float32)
        slope_src = "elevation_m difference to the neighbour"
    else:
        elev = np.zeros(shape, np.float32)
        slope_src = "no elevation in inputs -> P_slope = 1"

    # target-cell factor, combined once: P0 (1 + P_veg)(1 + P_den) P_mask
    target_factor = params.p0 * (1 + veg) * (1 + den) * mask

    n = len(DIRECTIONS)
    pw = np.empty((n, *shape), np.float32)
    ps = np.empty((n, *shape), np.float32)
    p = np.empty((n, *shape), np.float32)
    for k, (_, dr, dc, bearing) in enumerate(DIRECTIONS):
        dist = res * np.hypot(dr, dc)
        theta_s = np.nan_to_num(slope_angle_deg(elev, shift(elev, dr, dc, np.nan), dist))
        pw[k] = p_wind(u, v, bearing, params)
        ps[k] = p_slope(theta_s, params)
        p[k] = shift(target_factor, dr, dc, 0.0) * pw[k] * ps[k]   # neighbour outside the grid -> 0
    raw = p.copy()
    np.clip(p, 0.0, 1.0, out=p)

    # probability each cell catches fire: 1 - prod over burning neighbours of (1 - P(i -> j))
    no_fire = np.ones(shape, np.float64)
    from_burning = p * burning                       # only burning cells spread
    for k, (_, dr, dc, _) in enumerate(DIRECTIONS):
        no_fire *= 1 - shift(from_burning[k], -dr, -dc, 0.0)
    ignition = (1 - no_fire).astype(np.float32)

    # most likely direction: probability-weighted resultant of the 8 directions over the whole front
    expected = from_burning.reshape(n, -1).sum(axis=1)             # expected ignitions per direction
    bearings = np.radians([d[3] for d in DIRECTIONS])
    east, north = float((expected * np.sin(bearings)).sum()), float((expected * np.cos(bearings)).sum())
    total = float(expected.sum())
    main_bearing = float(np.degrees(np.arctan2(east, north)) % 360)
    strength = float(np.hypot(east, north) / total) if total > 0 else 0.0
    edge = burning[0].any() or burning[-1].any() or burning[:, 0].any() or burning[:, -1].any()

    dims2, dims3 = ("y", "x"), ("direction", "y", "x")
    coords = {k: ds.coords[k] for k in ("y", "x", "lat", "lon") if k in ds.coords}
    out = xr.Dataset(
        {"p_veg": (dims2, veg), "p_den": (dims2, den), "p_mask": (dims2, mask),
         "p_wind": (dims3, pw), "p_slope": (dims3, ps), "p_spread": (dims3, p),
         "ignition_prob": (dims2, ignition), "burning": (dims2, burning.astype(np.int16))},
        coords={**coords, "direction": DIR_NAMES})
    out["p_spread"].attrs["description"] = "P(burning cell at (y, x) ignites its neighbour in `direction`), clipped to 1"
    out["ignition_prob"].attrs["description"] = "probability the cell catches fire in the next CA step"
    weather_time = (str(pd.Timestamp(snap["time"].values)) if "time" in snap.coords else "")
    out.attrs.update({
        **{k: ds.attrs[k] for k in ("crs", "res_m", "transform", "time_utc", "initial_perimeter") if k in ds.attrs},
        "model": "Alexandridis et al. 2008: P = P0 (1+P_veg)(1+P_den) P_wind P_slope P_mask",
        "params": repr(params),
        "weather_time_utc": weather_time,
        "source_p_veg": veg_src, "source_p_den": den_src, "source_p_mask": mask_src,
        "source_p_wind": wind_src, "source_p_slope": slope_src,
        "main_direction_deg": main_bearing, "main_direction": _compass(main_bearing),
        "direction_strength": strength,   # 0 = spreads evenly all round, 1 = all in one direction
        "expected_new_cells": float(ignition.sum()),
        "expected_new_area_km2": float(ignition.sum()) * res * res / 1e6,
        "n_clipped": int(((raw > 1) & (from_burning > 0)).sum()),   # burning-cell spreads with P > 1 before clipping
        "fire_touches_grid_edge": int(edge),
        **{f"expected_{name}": float(e) for name, e in zip(DIR_NAMES, expected)},
    })
    return out


def summary(result: xr.Dataset) -> str:
    """Short text report of where the fire is most likely to go next."""
    a = result.attrs
    total = sum(a[f"expected_{d}"] for d in DIR_NAMES) or 1.0
    lines = [f"Most likely spread direction: {a['main_direction']} ({a['main_direction_deg']:.0f} deg), "
             f"strength {a['direction_strength']:.2f} (0 = even all round, 1 = one direction)",
             f"Expected new burning cells next step: {a['expected_new_cells']:.1f} "
             f"({a['expected_new_area_km2']:.2f} km2)",
             "Share of spread by direction: " + "  ".join(
                 f"{d} {100 * a[f'expected_{d}'] / total:.0f}%" for d in DIR_NAMES)]
    if a.get("weather_time_utc"):
        lines.append(f"Weather used: {a['weather_time_utc']} UTC")
    if a["fire_touches_grid_edge"]:
        lines.append("WARNING: the fire touches the grid edge; spread beyond it is not computed (use a larger radius).")
    if a["n_clipped"]:
        lines.append(f"Note: {a['n_clipped']} spreads had P > 1 before clipping (strong wind / steep uphill).")
    return "\n".join(lines)
