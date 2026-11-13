"""FEDS "VIIRS Modeled Fire Perimeters" from the VEDA OGC API Features endpoint.

Docs: https://earth-information-system.github.io/fireatlas/docs/nrt.html
Collections used (large-fire tracking, ~12 h steps):
  public.eis_fire_lf_perimeter_nrt   perimeter polygons, farea (km2), meanfrp, n_pixels, ...
  public.eis_fire_lf_fireline_nrt    active fire front (lines)
  public.eis_fire_lf_newfirepix_nrt  new VIIRS pixels added at each step

Time convention (checked against VIIRS acquisition times for the Bug Fire):
  t = YYYY-MM-DDT00 -> night overpasses of that UTC date (~08:30-11:00 UTC)
  t = YYYY-MM-DDT12 -> afternoon overpasses of that UTC date (~20:00-21:30 UTC)
The `primarykey` carries the 12 h step label; once a fire stops growing FEDS repeats the
last perimeter with its old `t`, so we keep both.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

API = "https://openveda.cloud/api/features"
COLLECTIONS = {
    "perimeter": "public.eis_fire_lf_perimeter_nrt",
    "fireline": "public.eis_fire_lf_fireline_nrt",
    "newfirepix": "public.eis_fire_lf_newfirepix_nrt",
}
# Typical offset from the FEDS step label to the actual VIIRS overpass time.
OVERPASS_OFFSET = {0: pd.Timedelta(hours=9, minutes=30), 12: pd.Timedelta(hours=8, minutes=30)}


def _get_items(collection: str, params: dict) -> dict:
    # The server returns HTTP 500 for heavy pages over busy areas (many large polygons), so page
    # with limit/offset and halve the page size whenever a page keeps failing.
    url, items, offset, limit = f"{API}/collections/{collection}/items", [], 0, 100
    while True:
        for attempt in range(3):
            r = requests.get(url, params={**params, "limit": limit, "offset": offset}, timeout=180)
            if r.status_code < 500:
                break
            time.sleep(3 * (attempt + 1))
        if r.status_code >= 500 and limit > 5:
            limit = max(5, limit // 2)
            continue
        r.raise_for_status()
        feats = r.json().get("features", [])
        items += feats
        if len(feats) < limit:
            return {"type": "FeatureCollection", "features": items}
        offset += len(feats)


def find_fire_ids(bbox, start_utc, end_utc) -> pd.DataFrame:
    """List FEDS fire ids whose perimeters intersect bbox in the time window."""
    dt = f"{start_utc:%Y-%m-%dT%H:%M:%SZ}/{end_utc:%Y-%m-%dT%H:%M:%SZ}"
    fc = _get_items(COLLECTIONS["perimeter"], {"bbox": ",".join(map(str, bbox)), "datetime": dt})
    if not fc["features"]:
        return pd.DataFrame(columns=["fireid", "n", "t_first", "t_last", "max_farea_km2"])
    df = pd.DataFrame([f["properties"] for f in fc["features"]])
    return (df.groupby("fireid").agg(n=("t", "size"), t_first=("t", "min"), t_last=("t", "max"),
                                     max_farea_km2=("farea", "max")).reset_index())


def download_feds(fireid: float, raw_dir: Path) -> dict[str, gpd.GeoDataFrame]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, coll in COLLECTIONS.items():
        path = raw_dir / f"feds_{kind}_{int(fireid)}.geojson"
        if not path.exists():
            fc = _get_items(coll, {"fireid": fireid})
            path.write_text(json.dumps(fc))
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        out[kind] = gdf
    return out


def tidy_perimeters(perim: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One row per 12 h step, with the step label, observation time and area."""
    g = perim.copy()
    g["step_utc"] = pd.to_datetime(g["primarykey"].str.split("|").str[-1], utc=True)
    g["feds_t_utc"] = pd.to_datetime(g["t"], utc=True)
    g["obs_time_utc"] = g["step_utc"] + g["step_utc"].dt.hour.map(OVERPASS_OFFSET)
    g["is_new"] = g["step_utc"] == g["feds_t_utc"]   # False -> perimeter repeated from earlier step
    g["area_km2"] = g.to_crs(5070).area / 1e6
    g["area_acres"] = g["area_km2"] * 247.105
    cols = ["step_utc", "obs_time_utc", "feds_t_utc", "is_new", "area_km2", "area_acres", "farea",
            "fperim", "flinelen", "n_pixels", "n_newpixels", "meanfrp", "fireid", "geometry"]
    return g[cols].sort_values("step_utc").reset_index(drop=True)


def select_fire_ids(target: gpd.GeoDataFrame, start_utc, end_utc, min_frac: float = 0.2) -> pd.DataFrame:
    """FEDS fire ids that belong to the target (official) perimeter.

    FEDS can split one fire into several ids (e.g. Biscar) or track it as one (Bug). An id is
    kept when at least `min_frac` of its largest perimeter lies inside the official perimeter.
    """
    bbox = target.to_crs(4326).total_bounds
    dt = f"{start_utc:%Y-%m-%dT%H:%M:%SZ}/{end_utc:%Y-%m-%dT%H:%M:%SZ}"
    fc = _get_items(COLLECTIONS["perimeter"], {"bbox": ",".join(map(str, bbox)), "datetime": dt})
    cols = ["fireid", "n_steps", "max_area_km2", "overlap_km2", "frac_inside", "selected"]
    if not fc["features"]:
        return pd.DataFrame(columns=cols)
    g = gpd.GeoDataFrame.from_features(fc, crs=4326).to_crs(5070)
    g["a"] = g.area
    big = g.sort_values("a").groupby("fireid").tail(1).set_index("fireid")
    tgt = target.to_crs(5070).geometry.union_all()
    out = pd.DataFrame({
        "fireid": big.index,
        "n_steps": g.groupby("fireid").size().reindex(big.index).to_numpy(),
        "max_area_km2": (big.a / 1e6).round(2).to_numpy(),
        "overlap_km2": (big.geometry.intersection(tgt).area / 1e6).round(2).to_numpy(),
    })
    out["frac_inside"] = (out.overlap_km2 / out.max_area_km2.where(out.max_area_km2 > 0)).round(3)
    out["selected"] = out.frac_inside >= min_frac
    return out.sort_values("overlap_km2", ascending=False).reset_index(drop=True)[cols]


def combine_perimeters(per_id: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Merge tidy perimeters of several FEDS ids into one row per 12 h step: at each step the
    union of every id's latest perimeter so far."""
    if len(per_id) == 1:
        return per_id[0]
    steps = sorted(set().union(*[set(p.step_utc) for p in per_id]))
    rows = []
    for s in steps:
        cur = [p[p.step_utc <= s].iloc[-1] for p in per_id if (p.step_utc <= s).any()]
        new = [r for r in cur if r.step_utc == s and r.is_new]
        geom = gpd.GeoSeries([r.geometry for r in cur], crs=per_id[0].crs).union_all()
        rows.append({
            "step_utc": s, "obs_time_utc": s + OVERPASS_OFFSET[s.hour],
            "feds_t_utc": max(r.feds_t_utc for r in cur), "is_new": bool(new),
            "farea": sum(r.farea for r in cur), "fperim": sum(r.fperim for r in cur),
            "flinelen": sum((r.flinelen or 0) for r in cur), "n_pixels": sum(r.n_pixels for r in cur),
            "n_newpixels": sum(r.n_newpixels for r in new) if new else 0,
            "meanfrp": float(np.nanmean([r.meanfrp for r in new])) if new else np.nan,
            "fireid": ",".join(str(int(r.fireid)) for r in cur), "geometry": geom,
        })
    g = gpd.GeoDataFrame(rows, crs=per_id[0].crs)
    g["area_km2"] = g.to_crs(5070).area / 1e6
    g["area_acres"] = g["area_km2"] * 247.105
    return g[per_id[0].columns]
