"""NIFC WFIGS official incident points and perimeters (public ArcGIS REST, no key).

WFIGS keeps only the *current* (final) perimeter per incident, not daily history.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

BASE = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
PERIMETERS = f"{BASE}/WFIGS_Interagency_Perimeters_YearToDate/FeatureServer/0/query"
INCIDENTS = f"{BASE}/WFIGS_Incident_Locations_YearToDate/FeatureServer/0/query"

PERIM_FIELDS = ["poly_IncidentName", "poly_GISAcres", "poly_PolygonDateTime", "poly_MapMethod",
                "attr_UniqueFireIdentifier", "attr_IncidentSize", "attr_FireDiscoveryDateTime",
                "attr_ContainmentDateTime", "attr_FireOutDateTime", "attr_InitialLatitude",
                "attr_InitialLongitude", "attr_POOState"]


def _query_geojson(url: str, params: dict) -> dict:
    r = requests.get(url, params={"outFields": "*", "outSR": 4326, "f": "geojson", **params}, timeout=120)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"])
    return j


def _ms_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if "DateTime" in c:
            df[c] = pd.to_datetime(df[c], unit="ms", utc=True)
    return df


def download_perimeters_in_bbox(bbox, start_utc, end_utc, raw_dir: Path) -> gpd.GeoDataFrame:
    """All WFIGS perimeters intersecting bbox with discovery in [start, end)."""
    path = Path(raw_dir) / "wfigs_perimeters.geojson"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        where = (f"attr_FireDiscoveryDateTime >= TIMESTAMP '{start_utc - pd.Timedelta(days=10):%Y-%m-%d %H:%M:%S}' "
                 f"AND attr_FireDiscoveryDateTime < TIMESTAMP '{end_utc:%Y-%m-%d %H:%M:%S}'")
        j = _query_geojson(PERIMETERS, {
            "where": where, "geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope",
            "inSR": 4326, "spatialRel": "esriSpatialRelIntersects"})
        path.write_text(json.dumps(j))
    g = gpd.read_file(path)
    g = g[[c for c in PERIM_FIELDS if c in g.columns] + ["geometry"]]
    return _ms_to_utc(g)


def download_incident(name: str, min_acres: float, raw_dir: Path) -> pd.DataFrame:
    path = Path(raw_dir) / "wfigs_incident.geojson"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        j = _query_geojson(INCIDENTS, {"where": f"IncidentName = '{name}' AND IncidentSize >= {min_acres}"})
        path.write_text(json.dumps(j))
    g = gpd.read_file(path)
    return _ms_to_utc(pd.DataFrame(g.drop(columns="geometry")))


def incidents_within(perimeter: gpd.GeoDataFrame, start_utc, end_utc, raw_dir: Path) -> pd.DataFrame:
    """Other WFIGS incidents whose point of origin lies inside `perimeter` and were discovered
    in [start, end): fires that merged into the target fire (e.g. Fred Mt into Bug)."""
    path = Path(raw_dir) / "wfigs_incidents_within.geojson"
    if not path.exists():
        where = (f"FireDiscoveryDateTime >= TIMESTAMP '{start_utc:%Y-%m-%d %H:%M:%S}' "
                 f"AND FireDiscoveryDateTime < TIMESTAMP '{end_utc:%Y-%m-%d %H:%M:%S}'")
        j = _query_geojson(INCIDENTS, {
            "where": where, "geometry": ",".join(map(str, perimeter.total_bounds)),
            "geometryType": "esriGeometryEnvelope", "inSR": 4326, "spatialRel": "esriSpatialRelIntersects"})
        path.write_text(json.dumps(j))
    g = gpd.read_file(path)
    if g.empty:
        return pd.DataFrame()
    pts = gpd.GeoSeries(gpd.points_from_xy(g.InitialLongitude, g.InitialLatitude), crs=4326)
    inside = pts.within(perimeter.to_crs(4326).geometry.union_all())
    target_id = perimeter.attr_UniqueFireIdentifier.iloc[0]
    g = g[inside.to_numpy() & (g.UniqueFireIdentifier != target_id).to_numpy()]
    cols = ["IncidentName", "UniqueFireIdentifier", "FireDiscoveryDateTime", "InitialLatitude",
            "InitialLongitude", "IncidentSize", "DiscoveryAcres"]
    return _ms_to_utc(pd.DataFrame(g[cols])).reset_index(drop=True)


def select_fire(perims: gpd.GeoDataFrame, name: str, min_acres: float) -> gpd.GeoDataFrame:
    sel = perims[(perims["poly_IncidentName"].str.lower() == name.lower())
                 & (perims["attr_IncidentSize"].fillna(perims["poly_GISAcres"]) >= min_acres)]
    if len(sel) != 1:
        raise ValueError(f"Expected exactly one WFIGS perimeter for {name!r}, found {len(sel)}")
    return sel
