"""BC Wildfire Service official fire perimeters and incident points (public ArcGIS REST, no key).

Canadian counterpart of wfigs.py (WFIGS is US only). Output columns use the same names as the
WFIGS perimeter (poly_* / attr_*) so downstream code can treat both alike; acres are converted
from hectares. Like WFIGS, BCWS keeps only the current (final) perimeter per fire.

Services (BC Wildfire Service open data, Open Government Licence - British Columbia):
  BCWS_FirePerimeters_PublicView  - FIRE_NUMBER, FIRE_SIZE_HECTARES, TRACK_DATE, FIRE_STATUS
  BCWS_ActiveFires_PublicView     - INCIDENT_NAME, IGNITION_DATE, FIRE_OUT_DATE, LATITUDE, LONGITUDE
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

BASE = "https://services6.arcgis.com/ubm4tcTYICKBpist/arcgis/rest/services"
PERIMETERS = f"{BASE}/BCWS_FirePerimeters_PublicView/FeatureServer/0/query"
INCIDENTS = f"{BASE}/BCWS_ActiveFires_PublicView/FeatureServer/0/query"
HA_TO_ACRES = 2.4710538
INCIDENT_URL = "https://wildfiresituation.nrs.gov.bc.ca/incidents?fireYear={year}&incidentNumber={number}"


def _query_geojson(url: str, where: str) -> dict:
    r = requests.get(url, params={"where": where, "outFields": "*", "outSR": 4326, "f": "geojson"}, timeout=120)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"])
    return j


def _cached(path: Path, url: str, where: str) -> gpd.GeoDataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_query_geojson(url, where)))
    return gpd.read_file(path)


def _utc(ms) -> pd.Timestamp | None:
    return None if ms is None or pd.isna(ms) else pd.to_datetime(ms, unit="ms", utc=True)


def download_fire(fire_number: str, year: int, raw_dir: Path) -> gpd.GeoDataFrame:
    """Official BCWS perimeter of one fire, joined with its incident record, in WFIGS column names.

    Raw responses are cached in raw_dir (bcws_perimeter.geojson, bcws_incident.geojson).
    """
    raw_dir = Path(raw_dir)
    where = f"FIRE_NUMBER = '{fire_number}' AND FIRE_YEAR = {int(year)}"
    perim = _cached(raw_dir / "bcws_perimeter.geojson", PERIMETERS, where)
    inc = _cached(raw_dir / "bcws_incident.geojson", INCIDENTS, where)
    if len(perim) != 1 or len(inc) != 1:
        raise ValueError(f"Expected one BCWS perimeter and one incident for {fire_number}/{year}, "
                         f"found {len(perim)} and {len(inc)}")
    p, i = perim.iloc[0], inc.iloc[0]
    # The GeoJSON reader may already have parsed the epoch-ms dates; normalise to UTC either way.
    as_utc = lambda v: (pd.Timestamp(v).tz_localize("UTC") if pd.Timestamp(v).tzinfo is None
                        else pd.Timestamp(v).tz_convert("UTC")) if isinstance(v, (pd.Timestamp, str)) else _utc(v)
    row = {
        "poly_IncidentName": i["INCIDENT_NAME"],
        "poly_GISAcres": float(p["FIRE_SIZE_HECTARES"]) * HA_TO_ACRES,
        "poly_PolygonDateTime": as_utc(p["TRACK_DATE"]),
        "poly_MapMethod": "BCWS",
        "attr_UniqueFireIdentifier": f"{int(year)}-BC-{fire_number}",
        "attr_IncidentSize": float(i["CURRENT_SIZE"]) * HA_TO_ACRES,
        "attr_FireDiscoveryDateTime": as_utc(i["IGNITION_DATE"]),   # BCWS "ignition date"
        "attr_ContainmentDateTime": None,                           # not published by BCWS
        "attr_FireOutDateTime": as_utc(i["FIRE_OUT_DATE"]) if pd.notna(i["FIRE_OUT_DATE"]) else None,
        "attr_InitialLatitude": float(i["LATITUDE"]),
        "attr_InitialLongitude": float(i["LONGITUDE"]),
        "attr_POOState": "CA-BC",
        "bcws_fire_status": p["FIRE_STATUS"],
        "bcws_fire_cause": i["FIRE_CAUSE"],
        "bcws_hectares": float(p["FIRE_SIZE_HECTARES"]),
        "bcws_url": INCIDENT_URL.format(year=int(year), number=fire_number),
    }
    return gpd.GeoDataFrame([row], geometry=[p.geometry], crs=4326)
