"""Alexandridis et al. (2008) cellular-automaton spread probability, with a fuel-moisture term.

For a burning cell i and each of its 8 neighbours j at time t:

    P_burn(i -> j, t) = clip(P0 * (1 + P_veg[j]) * (1 + P_den[j]) * P_w(i -> j, t) * P_s(i -> j)
                             * P_m(j, t) * P_mask[j] * D(i -> j), 0, 1)

    P_m(j, t) = P_emc(j, t) * (1 + P_dry[j]) * (1 + P_vwc[j])
    P_emc     = exp(-b_emc * (EMC(T, RH) - emc_ref))      fast: hourly air -> fine-fuel moisture (Simard 1968)
    P_dry     = NISAR soil-moisture class (+0.3 / 0 / -0.3) slow: ground moisture
    P_vwc     = NISAR vegetation-water class (+0.2 / 0 / -0.2) slow: water held in live vegetation

D = 1 for N/E/S/W neighbours and diagonal_factor (1/sqrt(2)) for diagonal ones. An unburned cell j
catches fire in the next step with probability 1 - prod_i (1 - P_burn(i -> j)) over its burning
neighbours i.

Data sources:
- P_veg, P_mask: Earth Engine (geelayers: LANDFIRE EVC fuel_lifeform, WorldCover landcover)
- P_w, P_s, P_emc: Earth Engine (geelayers: hourly wind, NASADEM elevation, hourly temp_c + rh_pct)
- P_den, P_dry (and burn scars in P_mask): NISAR only (see nisar.py). Until provided they are
  neutral (factor 1).

Everything is numpy array maths on whole grids (no loops over cells), so it can be ported to
torch / jax as-is to learn the parameters by gradient descent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

# 8 neighbours: name, row offset (row 0 = north), column offset (column 0 = west), compass bearing
DIRECTIONS = (("N", -1, 0, 0.0), ("NE", -1, 1, 45.0), ("E", 0, 1, 90.0), ("SE", 1, 1, 135.0),
              ("S", 1, 0, 180.0), ("SW", 1, -1, 225.0), ("W", 0, -1, 270.0), ("NW", -1, -1, 315.0))
DIR_NAMES = [d[0] for d in DIRECTIONS]

# ESA WorldCover classes (geelayers `landcover`)
WC_NOT_BURNABLE = (50, 60, 70, 80)   # built-up, bare, snow/ice, water

# Weather older than this (relative to the requested time) is flagged in the result
STALE_WEATHER = pd.Timedelta(hours=2)


@dataclass
class AlexandridisParams:
    """Model constants. P0, c1, c2, a, P_veg and P_den values are the published ones (Alexandridis et al.
    2008, Spetses fire, 1990); P_dry and the diagonal factor are our additions. All are starting points
    for learning per-region values from real fires."""
    p0: float = 0.58          # P0 (the paper's p_h): spread probability on flat ground, no wind, reference fuel
    c1: float = 0.045         # wind speed term, s/m
    c2: float = 0.131         # wind direction term, s/m
    a: float = 0.078          # slope term, 1/degree
    # P_veg by fuel_lifeform (geelayers vegetation_cover): paper classes agricultural -0.3, thickets 0, forest 0.4
    veg_by_lifeform: dict[int, float] = field(default_factory=lambda: {1: 0.4, 2: 0.0, 3: -0.3})
    # fallback when LANDFIRE has no data (outside the US): ESA WorldCover class -> P_veg
    veg_by_worldcover: dict[int, float] = field(default_factory=lambda: {
        10: 0.4, 95: 0.4, 20: 0.0, 30: -0.3, 40: -0.3, 90: -0.3, 100: -0.3})
    # P_den from nisar_cover_pct: paper classes sparse -0.4, normal 0, dense 0.3
    den_sparse: float = -0.4
    den_normal: float = 0.0
    den_dense: float = 0.3
    sparse_below_pct: float = 30.0
    dense_from_pct: float = 60.0
    # P_emc from hourly air temperature + humidity via Simard (1968) EMC (not in the 2008 paper; to be calibrated)
    b_emc: float = 0.05           # 1/% EMC: P_emc = exp(-b_emc * (EMC - emc_ref_pct))
    emc_ref_pct: float = 8.0      # EMC at which P_emc = 1 (reference fine-fuel moisture, %)
    # P_dry from nisar_soil_moisture (not in the 2008 paper; placeholder values, to be calibrated)
    dry_dry: float = 0.3          # soil moisture below dry_below_m3m3: drier fuel spreads more
    dry_normal: float = 0.0
    dry_wet: float = -0.3         # soil moisture at or above wet_from_m3m3
    dry_below_m3m3: float = 0.10
    wet_from_m3m3: float = 0.25
    # P_vwc from nisar_vwc_kg_m2 (SME2 vegetation water content; placeholder values, to be calibrated).
    # VWC also grows with biomass, so it partly overlaps P_veg / P_den: kept small until calibrated.
    vwc_low: float = 0.2          # VWC below vwc_low_below_kg_m2: little water in the vegetation (cured grass, dry shrub)
    vwc_normal: float = 0.0
    vwc_high: float = -0.2        # VWC at or above vwc_high_from_kg_m2: well-watered live canopy
    vwc_low_below_kg_m2: float = 0.5
    vwc_high_from_kg_m2: float = 4.0
    # diagonal neighbours are sqrt(2) cells away: same per-step probability would make flat, calm fires
    # advance sqrt(2) faster along diagonals. 1/sqrt(2) equalises the expected advance per step;
    # 1.0 reproduces the paper.
    diagonal_factor: float = 1 / math.sqrt(2)


# ---- the factors -----------------------------------------------------------------------------

def _classes(values: np.ndarray, low_below: float, high_from: float,
             low: float, normal: float, high: float) -> tuple[np.ndarray, int]:
    """Three-class lookup; NaN -> 0 (neutral). Returns (factor, number of NaN cells)."""
    out = np.full(values.shape, normal, np.float32)
    known = ~np.isnan(values)
    out[known & (values < low_below)] = low
    out[known & (values >= high_from)] = high
    out[~known] = 0.0
    return out, int((~known).sum())


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
    """P_den per cell from NISAR vegetation cover % only: sparse / normal / dense. No NISAR -> 0 (neutral)."""
    if "nisar_cover_pct" not in ds:
        return np.zeros((ds.sizes["y"], ds.sizes["x"]), np.float32), "nisar_cover_pct not provided -> 0 (neutral)"
    out, n_nan = _classes(ds["nisar_cover_pct"].values.astype(np.float32), params.sparse_below_pct,
                          params.dense_from_pct, params.den_sparse, params.den_normal, params.den_dense)
    return out, "nisar_cover_pct" + (f"; {n_nan} cells without NISAR data -> 0" if n_nan else "")


def p_dry(ds: xr.Dataset, params: AlexandridisParams) -> tuple[np.ndarray, str]:
    """P_dry per cell from NISAR soil moisture only: dry / normal / wet. No NISAR -> 0 (neutral)."""
    if "nisar_soil_moisture" not in ds:
        return np.zeros((ds.sizes["y"], ds.sizes["x"]), np.float32), "nisar_soil_moisture not provided -> 0 (neutral)"
    out, n_nan = _classes(ds["nisar_soil_moisture"].values.astype(np.float32), params.dry_below_m3m3,
                          params.wet_from_m3m3, params.dry_dry, params.dry_normal, params.dry_wet)
    return out, "nisar_soil_moisture" + (f"; {n_nan} cells without NISAR data -> 0" if n_nan else "")


def simard_emc(temp_c, rh_pct) -> np.ndarray:
    """Equilibrium moisture content of fine dead fuel, % (Simard 1968, as used in the US NFDRS).

    T_f = temp_c * 1.8 + 32, H = clip(rh_pct, 0, 100):
      H < 10:        0.03229 + 0.281073 H - 0.000578 H T_f
      10 <= H < 50:  2.22749 + 0.160107 H - 0.014784 T_f
      H >= 50:       21.0606 + 0.005565 H^2 - 0.00035 H T_f - 0.483199 H
    NaN in -> NaN out. Clipped at 0 (very hot, very dry air).
    """
    t_f = np.asarray(temp_c, dtype=np.float64) * 1.8 + 32.0
    h = np.clip(np.asarray(rh_pct, dtype=np.float64), 0.0, 100.0)
    emc = np.where(h < 10, 0.03229 + 0.281073 * h - 0.000578 * h * t_f,
          np.where(h < 50, 2.22749 + 0.160107 * h - 0.014784 * t_f,
                   21.0606 + 0.005565 * h ** 2 - 0.00035 * h * t_f - 0.483199 * h))
    return np.maximum(emc, 0.0)          # NaN stays NaN


def p_emc(ds: xr.Dataset, params: AlexandridisParams) -> tuple[np.ndarray, np.ndarray, str]:
    """P_emc = exp(-b_emc (EMC - emc_ref)): > 1 for hot / dry air, < 1 for cool / humid air.

    Uses temp_c and rh_pct of the cell that would catch fire, at the weather step in `ds`.
    Returns (P_emc, EMC %, source). Missing variables or NaN cells -> P_emc = 1 (neutral), EMC = NaN.
    """
    shape = (ds.sizes["y"], ds.sizes["x"])
    if "temp_c" not in ds or "rh_pct" not in ds:
        return (np.ones(shape, np.float32), np.full(shape, np.nan, np.float32),
                "EMC: temp_c / rh_pct not in inputs -> 1 (neutral)")
    emc = simard_emc(ds["temp_c"].values, ds["rh_pct"].values)
    factor = np.exp(-params.b_emc * (emc - params.emc_ref_pct))
    n_nan = int(np.isnan(factor).sum())
    factor[np.isnan(factor)] = 1.0
    return (factor.astype(np.float32), emc.astype(np.float32),
            "EMC: Simard 1968 from temp_c, rh_pct" + (f"; {n_nan} cells without weather -> 1" if n_nan else ""))


def p_vwc(ds: xr.Dataset, params: AlexandridisParams) -> tuple[np.ndarray, str]:
    """P_vwc per cell from NISAR vegetation water content only: low / normal / high. No NISAR -> 0 (neutral)."""
    if "nisar_vwc_kg_m2" not in ds:
        return np.zeros((ds.sizes["y"], ds.sizes["x"]), np.float32), "nisar_vwc_kg_m2 not provided -> 0 (neutral)"
    out, n_nan = _classes(ds["nisar_vwc_kg_m2"].values.astype(np.float32), params.vwc_low_below_kg_m2,
                          params.vwc_high_from_kg_m2, params.vwc_low, params.vwc_normal, params.vwc_high)
    return out, "nisar_vwc_kg_m2" + (f"; {n_nan} cells without NISAR data -> 0" if n_nan else "")


def p_moisture(ds: xr.Dataset, params: AlexandridisParams) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    """P_m = P_emc * (1 + P_dry) * (1 + P_vwc): fast air moisture (GEE hourly) x slow ground and vegetation
    moisture (NISAR).

    Returns (P_m, components {"emc_pct", "p_emc", "p_dry", "p_vwc"}, source).
    """
    emc_factor, emc, emc_src = p_emc(ds, params)
    dry, dry_src = p_dry(ds, params)
    vwc, vwc_src = p_vwc(ds, params)
    return (emc_factor * (1 + dry) * (1 + vwc), {"emc_pct": emc, "p_emc": emc_factor, "p_dry": dry, "p_vwc": vwc},
            f"{emc_src}; P_dry: {dry_src}; P_vwc: {vwc_src}")


# Why a cell cannot catch fire (result variable mask_reason); the first reason that applies is kept.
MASK_REASONS = ("can burn", "burning now", "no fuel (LANDFIRE)", "built-up, bare, snow or water (ESA WorldCover)",
                "already burned or water (NISAR)", "burn scar (Sentinel-2)", "no land cover data (open sea)")


def mask_reasons(ds: xr.Dataset, burning: np.ndarray) -> tuple[np.ndarray, str]:
    """Per cell, the index in MASK_REASONS (0 = can burn), and which inputs were used."""
    reason = np.zeros(burning.shape, np.int8)
    used = ["burning now"]

    def mark(cells: np.ndarray, code: int) -> None:
        reason[(reason == 0) & cells] = code
    mark(burning.astype(bool), 1)
    if "fuel_lifeform" in ds:
        mark(ds["fuel_lifeform"].values == 0, 2)
        used.append("fuel_lifeform = 0")
    if "landcover" in ds:
        lc = ds["landcover"].values
        mark(np.isin(lc, WC_NOT_BURNABLE), 3)
        # WorldCover maps land only: no class (-1 / NaN) is open sea, not unknown fuel (fire spread over the ocean)
        mark(~np.isfinite(lc) | (lc < 0) if lc.dtype.kind == "f" else lc < 0, 6)
        used.append(f"landcover in {WC_NOT_BURNABLE} or missing (open sea)")
    if "nisar_burned" in ds:
        mark(ds["nisar_burned"].values == 1, 4)
        used.append("nisar_burned")
    if "s2_burned" in ds:
        mark(ds["s2_burned"].values == 1, 5)
        used.append("s2_burned (Sentinel-2 dNBR)")
    return reason, ", ".join(used)


def p_mask(ds: xr.Dataset, burning: np.ndarray) -> tuple[np.ndarray, str]:
    """P_mask per cell: 1 = can catch fire, 0 = cannot (no fuel, water, built-up, burning or already burned)."""
    reason, used = mask_reasons(ds, burning)
    return (reason == 0).astype(np.float32), used


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


def _at_time(ds: xr.Dataset, time) -> tuple[xr.Dataset, str]:
    """Weather step for `time`: the last one at or before it. Returns (snapshot, note)."""
    if "time" not in ds.dims:
        return ds, ""
    if time is None:
        return ds.isel(time=0), ""
    t = pd.Timestamp(time)
    t = t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t
    first, last = pd.Timestamp(ds.time.values[0]), pd.Timestamp(ds.time.values[-1])
    if t < first:
        # no weather at or before t: use the first step (it is from after t; flagged)
        return ds.isel(time=0), f"no weather at or before {t}; used the first step {first} ({first - t} later)"
    snap = ds.sel(time=t, method="pad")
    used = pd.Timestamp(snap.time.values)
    note = f"weather is {t - used} older than the requested time" if t - used > STALE_WEATHER else ""
    return snap, note


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
      p_veg, p_den, p_m, p_mask (y, x)            factors of the cell that would catch fire
      emc_pct, p_emc, p_dry, p_vwc (y, x)         the parts of p_m (p_m = p_emc * (1 + p_dry) * (1 + p_vwc))
      p_wind, p_slope, p_spread (direction, y, x) for the burning cell at (y, x) spreading toward `direction`
      ignition_prob (y, x)                        chance each unburned cell catches fire in the next step
    and attrs summarising the most likely spread direction.
    """
    params = params or AlexandridisParams()
    snap, time_note = _at_time(ds, time)
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
    moist, moist_parts, moist_src = p_moisture(snap, params)
    reason, mask_src = mask_reasons(snap, burning)
    mask = (reason == 0).astype(np.float32)
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

    # target-cell factor, combined once: P0 (1 + P_veg)(1 + P_den) P_m P_mask
    target_factor = params.p0 * (1 + veg) * (1 + den) * moist * mask

    n = len(DIRECTIONS)
    pw = np.empty((n, *shape), np.float32)
    ps = np.empty((n, *shape), np.float32)
    p = np.empty((n, *shape), np.float32)
    for k, (_, dr, dc, bearing) in enumerate(DIRECTIONS):
        diagonal = dr != 0 and dc != 0
        dist = res * (math.sqrt(2) if diagonal else 1.0)
        theta_s = np.nan_to_num(slope_angle_deg(elev, shift(elev, dr, dc, np.nan), dist))
        pw[k] = p_wind(u, v, bearing, params)
        ps[k] = p_slope(theta_s, params)
        p[k] = (shift(target_factor, dr, dc, 0.0) * pw[k] * ps[k]       # neighbour outside the grid -> 0
                * (params.diagonal_factor if diagonal else 1.0))
    raw = p.copy()
    np.clip(p, 0.0, 1.0, out=p)

    # probability each cell catches fire: 1 - prod over burning neighbours of (1 - P(i -> j))
    no_fire = np.ones(shape, np.float64)
    from_burning = p * burning                       # only burning cells spread
    for k, (_, dr, dc, _) in enumerate(DIRECTIONS):
        no_fire *= 1 - shift(from_burning[k], -dr, -dc, 0.0)
    ignition = (1 - no_fire).astype(np.float32)

    # most likely direction: resultant of the expected ignitions in the 8 directions over the whole front.
    # Uses the clipped probabilities, i.e. what the CA actually does (P > 1 cannot ignite more than once).
    expected = from_burning.reshape(n, -1).sum(axis=1)
    bearings = np.radians([d[3] for d in DIRECTIONS])
    east, north = float((expected * np.sin(bearings)).sum()), float((expected * np.cos(bearings)).sum())
    total = float(expected.sum())
    main_bearing = float(np.degrees(np.arctan2(east, north)) % 360)
    strength = float(np.hypot(east, north) / total) if total > 0 else 0.0
    edge = burning[0].any() or burning[-1].any() or burning[:, 0].any() or burning[:, -1].any()
    n_spreads = int(((raw > 0) & burning).sum())
    n_clipped = int(((raw > 1) & burning).sum())

    dims2, dims3 = ("y", "x"), ("direction", "y", "x")
    coords = {k: ds.coords[k] for k in ("y", "x", "lat", "lon") if k in ds.coords}
    out = xr.Dataset(
        {"p_veg": (dims2, veg), "p_den": (dims2, den), "p_m": (dims2, moist),
         **{k: (dims2, v) for k, v in moist_parts.items()}, "p_mask": (dims2, mask),
         "mask_reason": (dims2, reason),
         "p_wind": (dims3, pw), "p_slope": (dims3, ps), "p_spread": (dims3, p),
         "ignition_prob": (dims2, ignition), "burning": (dims2, burning.astype(np.int16))},
        coords={**coords, "direction": DIR_NAMES})
    out["p_spread"].attrs["description"] = "P(burning cell at (y, x) ignites its neighbour in `direction`), clipped to 1"
    out["ignition_prob"].attrs["description"] = "probability the cell catches fire in the next CA step"
    out["mask_reason"].attrs["description"] = "why the cell cannot catch fire: " + "; ".join(
        f"{i} = {r}" for i, r in enumerate(MASK_REASONS))
    weather_time = (str(pd.Timestamp(snap["time"].values)) if "time" in snap.coords else "")
    out.attrs.update({
        **{k: ds.attrs[k] for k in ("crs", "res_m", "transform", "time_utc", "initial_perimeter", "weather_source")
           if k in ds.attrs},
        **{k: v for k, v in ds.attrs.items() if k.startswith(("s2_", "nisar_"))},
        "model":"Alexandridis et al. 2008 + moisture: P = P0 (1+P_veg)(1+P_den) P_w P_s P_m P_mask D; "
                 "P_m = exp(-b_emc (EMC_Simard(T, RH) - emc_ref)) (1+P_dry_NISAR) (1+P_vwc_NISAR); "
                 "D = 1/sqrt(2) diagonal",
        "params": repr(params),
        "weather_time_utc": weather_time,
        "weather_note": time_note,
        "source_p_veg": veg_src, "source_p_den": den_src, "source_p_m": moist_src, "source_p_mask": mask_src,
        "source_p_wind": wind_src, "source_p_slope": slope_src,
        "main_direction_deg": main_bearing, "main_direction": _compass(main_bearing),
        "direction_strength": strength,   # 0 = spreads evenly all round, 1 = all in one direction
        "expected_new_cells": float(ignition.sum()),
        "expected_new_area_km2": float(ignition.sum()) * res * res / 1e6,
        "n_spreads": n_spreads,            # burning-cell -> neighbour pairs with P > 0
        "n_clipped": n_clipped,            # ... of which P > 1 before clipping
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
    if a.get("weather_note"):
        lines.append(f"WARNING: {a['weather_note']}")
    lines.append(f"Moisture P_m: {a['source_p_m']}")
    lines.append(f"Density P_den: {a['source_p_den']}")
    if a["fire_touches_grid_edge"]:
        lines.append("WARNING: the fire touches the grid edge; spread beyond it is not computed (use a larger radius).")
    if a["n_clipped"]:
        lines.append(f"Note: {a['n_clipped']} of {a['n_spreads']} spreads had P > 1 before clipping "
                     "(strong wind / steep uphill); those directions are saturated.")
    return "\n".join(lines)
