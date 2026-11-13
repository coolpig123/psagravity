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
from pathlib import Path

import geopandas as gpd
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
    items, url, p = [], f"{API}/collections/{collection}/items", {"limit": 1000, **params}
    while url:
        r = requests.get(url, params=p, timeout=120)
        r.raise_for_status()
        j = r.json()
        items += j.get("features", [])
        url = next((l["href"] for l in j.get("links", []) if l.get("rel") == "next"), None)
        p = None
    return {"type": "FeatureCollection", "features": items}


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
