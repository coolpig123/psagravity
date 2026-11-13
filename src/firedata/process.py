"""Assign detections to a fire and build the ground-truth tables."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

EQUAL_AREA = 5070  # CONUS Albers, metres

POINT_COLUMNS = ["time_utc", "lat", "lon", "frp_mw", "source", "confidence"]


def assign_to_fire(points: pd.DataFrame, fire: gpd.GeoDataFrame, others: gpd.GeoDataFrame,
                   buffer_m: float, t_min: pd.Timestamp, t_max: pd.Timestamp) -> pd.DataFrame:
    """Keep points within `buffer_m` of the fire's final perimeter, inside [t_min, t_max],
    and not closer to any other perimeter (neighbouring fires). Adds `dist_m`."""
    if points.empty:
        return points.assign(dist_m=pd.Series(dtype=float))
    pts = gpd.GeoSeries(gpd.points_from_xy(points.lon, points.lat), crs=4326).to_crs(EQUAL_AREA)
    target = fire.to_crs(EQUAL_AREA).geometry.union_all()
    d_target = pts.distance(target).to_numpy()
    d_other = np.full(len(points), np.inf)
    for geom in others.to_crs(EQUAL_AREA).geometry:
        d_other = np.minimum(d_other, pts.distance(geom).to_numpy())
    keep = ((d_target <= buffer_m) & (d_target <= d_other)
            & (points.time_utc >= t_min).to_numpy() & (points.time_utc <= t_max).to_numpy())
    return points.loc[keep].assign(dist_m=d_target[keep]).reset_index(drop=True)


def combined_points(viirs: pd.DataFrame, goes: pd.DataFrame) -> pd.DataFrame:
    """Single table: time_utc, lat, lon, frp_mw, source, confidence (+ source-specific extras)."""
    frames = [df for df in (viirs, goes) if not df.empty]
    out = pd.concat(frames, ignore_index=True, sort=False)
    extras = [c for c in out.columns if c not in POINT_COLUMNS]
    return out[POINT_COLUMNS + extras].sort_values(["time_utc", "source"]).reset_index(drop=True)


def viirs_overpasses(viirs: pd.DataFrame, gap: str = "20min") -> pd.DataFrame:
    """Total FRP per satellite overpass (points within `gap` of each other = one overpass)."""
    rows = []
    for src, g in viirs.sort_values("time_utc").groupby("source"):
        pass_id = (g.time_utc.diff() > pd.Timedelta(gap)).cumsum()
        for _, p in g.groupby(pass_id):
            rows.append({"time_utc": p.time_utc.min(), "source": src, "n_points": len(p),
                         "frp_total_mw": p.frp_mw.sum()})
    return pd.DataFrame(rows)


def goes_scans(goes: pd.DataFrame, scans: pd.DataFrame, source: str) -> pd.DataFrame:
    """Total FRP per GOES scan, including scans with zero fire pixels."""
    g = goes[goes.in_frp_total]
    agg = g.groupby("time_utc").agg(n_points=("frp_mw", "size"), frp_total_mw=("frp_mw", "sum"),
                                    n_without_frp=("frp_mw", lambda s: int(s.isna().sum())))
    out = scans[["time_utc"]].drop_duplicates().set_index("time_utc").join(agg).fillna(
        {"n_points": 0, "frp_total_mw": 0.0, "n_without_frp": 0})
    out["source"] = source
    return out.reset_index().astype({"n_points": int, "n_without_frp": int})


def frp_timeseries(viirs: pd.DataFrame, goes: pd.DataFrame, scans: pd.DataFrame) -> pd.DataFrame:
    parts = [viirs_overpasses(viirs)]
    if not goes.empty:
        parts.append(goes_scans(goes, scans, goes.source.iloc[0]))
    return pd.concat(parts, ignore_index=True).sort_values("time_utc").reset_index(drop=True)


def viirs_footprint(viirs: pd.DataFrame) -> gpd.GeoSeries:
    """Union of the VIIRS pixel footprints (scan x track km rectangles): the area ever seen burning."""
    from shapely.geometry import box
    pts = gpd.GeoSeries(gpd.points_from_xy(viirs.lon, viirs.lat), crs=4326).to_crs(EQUAL_AREA)
    hx, hy = viirs.scan_km.to_numpy() * 500, viirs.track_km.to_numpy() * 500
    boxes = [box(p.x - a, p.y - b, p.x + a, p.y + b) for p, a, b in zip(pts, hx, hy)]
    return gpd.GeoSeries([gpd.GeoSeries(boxes).union_all()], crs=EQUAL_AREA)


def area_comparison(final: gpd.GeoDataFrame, perims: gpd.GeoDataFrame, viirs: pd.DataFrame) -> dict:
    w = final.to_crs(EQUAL_AREA).geometry.iloc[0]
    f = perims.to_crs(EQUAL_AREA).geometry.iloc[perims.step_utc.argmax()]
    v = viirs_footprint(viirs).iloc[0]
    km2 = lambda g: round(g.area / 1e6, 2)
    return {"wfigs_final_km2": km2(w), "feds_final_km2": km2(f), "overlap_km2": km2(w.intersection(f)),
            "feds_outside_wfigs_km2": km2(f.difference(w)), "wfigs_not_in_feds_km2": km2(w.difference(f)),
            "viirs_footprint_km2": km2(v), "viirs_footprint_in_wfigs_km2": km2(v.intersection(w)),
            "feds_fraction_of_wfigs": round(f.area / w.area, 3)}


def latest_perimeter_at(perims: gpd.GeoDataFrame, when: pd.Timestamp) -> pd.Series | None:
    """Most recent FEDS perimeter observed at or before `when`."""
    before = perims[perims.obs_time_utc <= when]
    return None if before.empty else before.iloc[-1]
