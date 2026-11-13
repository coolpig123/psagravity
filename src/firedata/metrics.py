"""Per-fire data inventory and agreement metrics against the official final perimeter.

    python -m firedata.metrics bug_fire      # -> data/fires/<fire>/processed/metrics.json

Every percentage is defined against the official (NIFC WFIGS) final perimeter W:
  feds_iou            area(FEDS_final ∩ W) / area(FEDS_final ∪ W)
  feds_area_ratio     area(FEDS_final) / area(W)
  viirs_cover         area(VIIRS pixel footprints ∩ W) / area(W)
  viirs_inside        area(VIIRS pixel footprints ∩ W) / area(VIIRS pixel footprints)
  goes_cover          area(GOES pixel footprints ∩ W) / area(W)
  goes_inside         area(GOES pixel footprints ∩ W) / area(GOES pixel footprints)
  all_cover           area((FEDS ∪ VIIRS ∪ GOES) ∩ W) / area(W)
  ignition_error_min  first detection time - official discovery time (minutes; > 0 = later)
  ignition_dist_km    distance from the first detection to the official point of origin
"""
from __future__ import annotations

import argparse
import json

import geopandas as gpd
import pandas as pd

from .config import FIRES, FireConfig
from .process import EQUAL_AREA, viirs_footprint

DEFINITIONS = {
    "feds_iou": "IoU of the final FEDS perimeter with the official perimeter W: area(FEDS ∩ W) / area(FEDS ∪ W).",
    "feds_area_ratio": "Final FEDS area / official area.",
    "viirs_cover": "Share of the official area inside at least one VIIRS 375 m pixel footprint.",
    "viirs_inside": "Share of the VIIRS pixel-footprint area that lies inside the official perimeter.",
    "goes_cover": "Share of the official area inside at least one GOES fire-pixel footprint (pixels are ~7 km²).",
    "goes_inside": "Share of the GOES fire-pixel footprint area that lies inside the official perimeter.",
    "all_cover": "Share of the official area covered by FEDS, VIIRS or GOES (union).",
    "ignition_error_min": "First detection time minus official discovery time, in minutes (positive = detected later).",
    "ignition_dist_km": "Distance from the first detection to the official point of origin, in km.",
}


def _pct(x):
    return None if x is None else round(100 * float(x), 1)


def compute(F: FireConfig) -> dict:
    P = F.root / "processed"
    s = json.loads((P / "summary.json").read_text())
    W = gpd.read_file(P / "wfigs_final_perimeter.geojson").to_crs(EQUAL_AREA).geometry.union_all()
    wa = W.area
    pts = pd.read_parquet(P / "fire_points.parquet")
    if "ambiguous" in pts:
        pts = pts[~pts.ambiguous]
    disc = pd.Timestamp(s["wfigs"]["discovery_utc"])
    origin = gpd.GeoSeries(gpd.points_from_xy([s["wfigs"]["origin_lon"]], [s["wfigs"]["origin_lat"]]),
                           crs=4326).to_crs(EQUAL_AREA).iloc[0]
    out: dict = {"fire": F.name, "official_area_km2": round(wa / 1e6, 1),
                 "official_acres": round(s["wfigs"]["final_acres"]), "items": {}}

    # FEDS
    fp = P / "feds_perimeters_12h.geojson"
    feds_geom = None
    if fp.exists():
        f = gpd.read_file(fp)
        f["step_utc"] = pd.to_datetime(f.step_utc, utc=True)
        feds_geom = f.to_crs(EQUAL_AREA).geometry.iloc[f.step_utc.argmax()]
        new = f[f.is_new.astype(bool)]
        out["items"]["FEDS"] = {
            "source": "NASA FEDS VIIRS modeled fire perimeters (VEDA OGC API, *_nrt)",
            "time_range": f"{pd.Timestamp(new.obs_time_utc.min()):%Y-%m-%d %H:%M} to {pd.Timestamp(new.obs_time_utc.max()):%Y-%m-%d %H:%M} UTC",
            "resolution": "polygons from 375 m VIIRS pixels", "update_rate": "12 h",
            "coverage": f"{len(new)} perimeters with new data ({len(f)} steps); fire id(s) {', '.join(str(int(float(x))) for x in s['feds']['fireids'])}",
            "metrics": {"feds_iou": _pct(feds_geom.intersection(W).area / feds_geom.union(W).area),
                        "feds_area_ratio": _pct(feds_geom.area / wa)},
        }
    else:
        out["items"]["FEDS"] = {"source": "NASA FEDS", "na": "no FEDS perimeter overlaps the official perimeter"}

    # VIIRS
    v = pts[pts.source.str.startswith("VIIRS")]
    vfp = viirs_footprint(v).iloc[0]
    first_v = v.loc[v.time_utc.idxmin()]
    fv = gpd.GeoSeries(gpd.points_from_xy([first_v.lon], [first_v.lat]), crs=4326).to_crs(EQUAL_AREA).iloc[0]
    out["items"]["VIIRS"] = {
        "source": "VIIRS 375 m active fire, S-NPP + NOAA-20 + NOAA-21 (NASA FIRMS, " + ", ".join(sorted(v["product"].unique())) + ")",
        "time_range": f"{v.time_utc.min():%Y-%m-%d %H:%M} to {v.time_utc.max():%Y-%m-%d %H:%M} UTC",
        "resolution": "375 m (pixel grows to ~0.8 km at scan edge)", "update_rate": "~6 overpasses/day (3 satellites × day/night)",
        "coverage": f"{len(v):,} detections",
        "metrics": {"viirs_cover": _pct(vfp.intersection(W).area / wa), "viirs_inside": _pct(vfp.intersection(W).area / vfp.area),
                    "ignition_error_min": round((first_v.time_utc - disc).total_seconds() / 60),
                    "ignition_dist_km": round(fv.distance(origin) / 1000, 2)},
    }

    # GOES
    g = pts[pts.source.str.startswith("GOES")]
    gfp = None
    if len(g):
        from . import goes
        sat = "goes" + g.source.iloc[0][4:6]
        pix = g.groupby(["lat", "lon"]).size().reset_index()[["lat", "lon"]]
        gfp = goes.pixel_footprints(pix, satellite=sat, when=g.time_utc.min()).to_crs(EQUAL_AREA).union_all()
        first_g = g.loc[g.time_utc.idxmin()]
        fg = gpd.GeoSeries(gpd.points_from_xy([first_g.lon], [first_g.lat]), crs=4326).to_crs(EQUAL_AREA).iloc[0]
        n_scans = int(pd.read_csv(P / "frp_timeseries.csv").source.str.startswith("GOES").sum())
        out["items"]["GOES"] = {
            "source": f"GOES-{g.source.iloc[0][4:6]} ABI L2 FDC, CONUS sector (NOAA, AWS)",
            "time_range": f"{g.time_utc.min():%Y-%m-%d %H:%M} to {g.time_utc.max():%Y-%m-%d %H:%M} UTC",
            "resolution": f"2 km nominal; ~{goes.pixel_footprints(pix.head(1), satellite=sat, when=g.time_utc.min()).to_crs(EQUAL_AREA).area.iloc[0] / 1e6:.1f} km² per pixel here",
            "update_rate": "5 min", "coverage": f"{len(g):,} fire pixels in {n_scans:,} scans",
            "metrics": {"goes_cover": _pct(gfp.intersection(W).area / wa), "goes_inside": _pct(gfp.intersection(W).area / gfp.area),
                        "ignition_error_min": round((first_g.time_utc - disc).total_seconds() / 60),
                        "ignition_dist_km": round(fg.distance(origin) / 1000, 2)},
        }
    else:
        out["items"]["GOES"] = {"source": "GOES ABI FDC", "na": "no GOES fire pixels assigned to this fire"}

    union = vfp
    for geom in (feds_geom, gfp):
        if geom is not None:
            union = union.union(geom)
    out["all_cover"] = _pct(union.intersection(W).area / wa)

    out["items"]["Final perimeter"] = {
        "source": "NIFC WFIGS interagency perimeter (" + s["wfigs"]["unique_fire_id"] + ")",
        "time_range": f"perimeter dated {pd.Timestamp(s['wfigs']['perimeter_date_utc']):%Y-%m-%d %H:%M} UTC",
        "resolution": "mapped polygon (Mixed Methods)", "update_rate": "final only (no history)",
        "coverage": f"{s['wfigs']['final_acres']:,.0f} acres ({wa / 1e6:,.1f} km²)",
        "metrics": {}, "reference": True,
    }

    pairs = pd.read_csv(P / "nisar_pairs.csv") if (P / "nisar_pairs.csv").exists() else pd.DataFrame()
    passes = pd.read_csv(P / "nisar_passes.csv") if (P / "nisar_passes.csv").exists() else pd.DataFrame()
    if len(pairs):
        out["items"]["NISAR GCOV"] = {
            "source": "NISAR L2 GCOV (ASF), standard production",
            "time_range": "; ".join(f"{r.track_id}: {pd.Timestamp(r.pre_utc):%m-%d} → {pd.Timestamp(r.post_utc):%m-%d}" for r in pairs.itertuples()),
            "resolution": "L-band HH/HV, geocoded grid", "update_rate": "12-day repeat per track; several tracks",
            "coverage": f"{len(passes)} passes, {len(pairs)} same-track pre/post pairs (shortest gap {pairs.gap_days.min():g} days)",
            "metrics": {}, "na": "burn-scar agreement not computed yet (issue #7)",
        }
    else:
        out["items"]["NISAR GCOV"] = {"source": "NISAR L2 GCOV (ASF)", "na": "no same-track pre/post pair found"}
    return out


def write(fire: str) -> dict:
    F = FIRES[fire]
    m = compute(F)
    (F.root / "processed" / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fire", choices=FIRES)
    print(json.dumps(write(ap.parse_args().fire), indent=2, ensure_ascii=False))
