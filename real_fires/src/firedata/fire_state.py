"""Current fire state from radar: burned-area polygon at a time, from the latest RCM and / or NISAR scene.

For when there is no fresh FEDS perimeter. Rules are the ones scored on Bald Range (data/fires/bald_range/README.md):
  RCM C-band  : RL rise  >= RCM_RISE_DB   vs a same-beam reference before ignition (CEOS-ARD, public AWS bucket)
  NISAR L-band: HV drop  >= NISAR_DROP_DB vs a same-track reference (threshold calibrated on the Bug Fire, issue #7)
The latest scene of each sensor acquired before `time` is used; sensors whose scene is within COMBINE_DAYS of the
newest one are OR-ed. Then specks < MIN_HA are dropped, holes < MIN_HA filled, and only the largest clump reaching
within SEED_M of the ignition is kept (RCM has many isolated false alarms). The official perimeter is NOT used: the window is
ignition +- radius_km, and each RCM scene is quantile-matched to its reference over the whole window.

Only the window is read (RCM COG tiles over HTTP, NISAR HDF5 byte ranges); clips are cached under the fire's raw/.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

from . import rcm
from .config import REPO_ROOT
from .nisar_access import GCOV_GROUP, TargetGrid, group_passes, read_pass, search_granules
from .nisar_provider import SCAR_SMOOTH_CELLS, _gcov_valid, hv_change_db

RCM_RISE_DB = 1.0
NISAR_DROP_DB = 0.25
COMBINE_DAYS = 4.0
MIN_HA = 5.0
LINK_M = 200.0            # clumps closer than this are joined when picking the one around the ignition
SEED_M = 2000.0           # the kept clump must reach within this distance of the ignition
RES_M = 100.0


@dataclass
class RadarFire:
    """Scenes available for one fire. RCM: (reference, [later scenes]) on one beam; NISAR: track -> (ref, [later])."""
    slug: str
    ignition_lat: float
    ignition_lon: float
    crs: int
    rcm_ref: str
    rcm_posts: list[str]
    nisar_pairs: dict[str, tuple[str, list[str]]] = field(default_factory=dict)   # dates as "YYYY-MM-DD"


FIRES = {
    "bald_range": RadarFire(
        slug="bald_range", ignition_lat=49.6262, ignition_lon=-119.8835, crs=32611,
        rcm_ref="RCM1_OK4114697_PK4293771_1_SC30MCPA_20260802_013425_CH_CV_MLC",
        rcm_posts=["RCM3_OK4114697_PK4308170_1_SC30MCPA_20260810_013442_CH_CV_MLC",
                   "RCM3_OK4114697_PK4329615_1_SC30MCPA_20260822_013440_CH_CV_MLC"],
        nisar_pairs={"D172": ("2026-07-20", ["2026-08-13", "2026-08-25"]),
                     "A77": ("2026-07-25", ["2026-08-18", "2026-08-30"]),
                     "D71": ("2026-07-25", ["2026-08-18", "2026-08-30"])}),
}


@dataclass
class FireState:
    polygon_lonlat: object          # shapely (Multi)Polygon in EPSG:4326, empty if nothing found
    mask: np.ndarray                # (ny, nx) bool on `transform`
    transform: object
    crs: int
    scenes: list[str]               # what was used, e.g. "RCM SC30MCPA 2026-08-10 01:34 vs 2026-08-02"
    area_ha: float


def _window(f: RadarFire, radius_km: float):
    x0, y0 = Transformer.from_crs(4326, f.crs, always_xy=True).transform(f.ignition_lon, f.ignition_lat)
    r = radius_km * 1000
    w, n = np.floor((x0 - r) / RES_M) * RES_M, np.ceil((y0 + r) / RES_M) * RES_M
    nx = ny = int(round(2 * r / RES_M))
    tr = from_origin(w, n, RES_M, RES_M)
    grid = TargetGrid(f"EPSG:{f.crs}", w + (np.arange(nx) + 0.5) * RES_M, n - (np.arange(ny) + 0.5) * RES_M, RES_M)
    return tr, grid, (ny, nx), (x0, y0)


def _quantile_map_db(x, ref, sel):
    q = np.linspace(0.5, 99.5, 199)
    xs, rs = np.nanpercentile(10 * np.log10(x[sel]), q), np.nanpercentile(10 * np.log10(ref[sel]), q)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10 ** (np.interp(10 * np.log10(x), xs, rs) / 10)


def _rcm_mask(f: RadarFire, scene: str, tr, shape_, cache) -> np.ndarray:
    def rl(s):  # RL averaged onto the 100 m grid (linear power), bitmask != 1 -> NaN
        a = rcm.read_clipped(s, "RL", f.crs, tr, shape_, cache)
        b = rcm.read_clipped(s, "bitmask", f.crs, tr, shape_, cache)
        a[b != 1] = np.nan
        return a
    r0, r1 = rl(f.rcm_ref), rl(scene)
    sel = np.isfinite(r0) & np.isfinite(r1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 10 * np.log10(_quantile_map_db(r1, r0, sel) / r0)
    return np.nan_to_num(d) >= RCM_RISE_DB


def _nisar_mask(track: str, ref: str, post: str, grid, bbox) -> np.ndarray:
    passes = group_passes(search_granules("GCOV", bbox, pd.Timestamp(ref) - pd.Timedelta(days=1),
                                          pd.Timestamp(post) + pd.Timedelta(days=1)))
    def hv(day):
        p = [q for q in passes if q.track_id == track and f"{q.time_utc:%Y-%m-%d}" == day]
        if len(p) != 1:
            raise RuntimeError(f"NISAR {track} {day}: {len(p)} passes")
        return read_pass(p[0], GCOV_GROUP, ["HVHV", "mask"], grid, prepare=_gcov_valid)["HVHV"]
    d = hv_change_db(hv(post), hv(ref), SCAR_SMOOTH_CELLS)
    return np.nan_to_num(d) <= -NISAR_DROP_DB


def _scene_time(scene: str) -> pd.Timestamp:
    p = scene.split("_")
    return pd.Timestamp(f"{p[5]}T{p[6]}", tz="UTC")


def _nisar_time(track: str, day: str) -> pd.Timestamp:
    # Pass times over Bald Range (ASF): ascending ~12:5x UTC, descending ~03:1x UTC.
    return pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=12.8 if track.startswith("A") else 3.2)


def clean(mask: np.ndarray, seed_rc: tuple[int, int], min_ha: float = MIN_HA, link_m: float = LINK_M,
          seed_m: float = SEED_M) -> np.ndarray:
    """Drop specks < min_ha, fill holes < min_ha, keep the largest clump (pieces < link_m apart joined)
    that has a burned cell within seed_m of the ignition."""
    min_cells = max(1, int(round(min_ha * 1e4 / RES_M ** 2)))
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    m = np.isin(lab, np.nonzero(sizes >= min_cells)[0] + 1)
    holes, nh = ndimage.label(~m)
    hs = ndimage.sum(~m, holes, range(1, nh + 1))
    m |= np.isin(holes, np.nonzero(hs < min_cells)[0] + 1)
    k = max(1, int(round(link_m / RES_M)))
    linked, n = ndimage.label(ndimage.binary_dilation(m, iterations=k), structure=np.ones((3, 3)))
    yy, xx = np.indices(m.shape)
    near = np.hypot(yy - seed_rc[0], xx - seed_rc[1]) * RES_M <= seed_m
    cand = np.unique(linked[near & m])
    cand = cand[cand > 0]
    if not cand.size:
        return np.zeros_like(m)
    sizes = ndimage.sum(m, linked, cand)            # burned cells per linked clump
    return m & (linked == cand[int(np.argmax(sizes))])


def fire_state(fire: str, time, radius_km: float = 15.0) -> FireState:
    """Burned area of `fire` at `time` from the latest radar scene(s) acquired before `time`."""
    f = FIRES[fire]
    time = pd.Timestamp(time).tz_convert("UTC") if pd.Timestamp(time).tzinfo else pd.Timestamp(time, tz="UTC")
    tr, grid, shape_, (x0, y0) = _window(f, radius_km)
    cache = REPO_ROOT / "data" / "fires" / f.slug / "raw" / f"rcm_state_{int(radius_km)}km"
    cands = [(_scene_time(s), "RCM", s) for s in f.rcm_posts]
    cands += [(_nisar_time(t, d), "NISAR", (t, ref, d)) for t, (ref, posts) in f.nisar_pairs.items() for d in posts]
    cands = [c for c in cands if c[0] < time]
    latest = {}
    for c in sorted(cands):
        latest[c[1]] = c                     # latest scene per sensor
    if not latest:
        raise ValueError(f"No radar scene of {fire} before {time}")
    newest = max(c[0] for c in latest.values())
    use = [c for c in latest.values() if (newest - c[0]).total_seconds() <= COMBINE_DAYS * 86400]
    bbox = tuple(gpd.GeoSeries.from_xy([x0 - 2e4, x0 + 2e4], [y0 - 2e4, y0 + 2e4], crs=f.crs).to_crs(4326).total_bounds)
    mask, used = np.zeros(shape_, bool), []
    for t, sensor, s in use:
        if sensor == "RCM":
            mask |= _rcm_mask(f, s, tr, shape_, cache)
            used.append(f"RCM {s.split('_')[4]} {t:%Y-%m-%d %H:%M} vs {_scene_time(f.rcm_ref):%Y-%m-%d} "
                        f"(RL rise >= {RCM_RISE_DB} dB)")
        else:
            mask |= _nisar_mask(*s, grid, bbox)
            used.append(f"NISAR {s[0]} {s[2]} vs {s[1]} (HV drop >= {NISAR_DROP_DB} dB)")
    seed = (int((tr.f - y0) // RES_M), int((x0 - tr.c) // RES_M))
    mask = clean(mask, seed)
    polys = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v == 1]
    poly = gpd.GeoSeries([unary_union(polys)] if polys else [], crs=f.crs).to_crs(4326)
    return FireState(poly.iloc[0] if len(poly) else None, mask, tr, f.crs, used,
                     round(float(mask.sum()) * RES_M ** 2 / 1e4, 1))
