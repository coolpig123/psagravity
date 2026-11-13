"""Build fire-progression ground truth for one fire.

    python scripts/build_ground_truth.py bug_fire            # everything
    python scripts/build_ground_truth.py bug_fire --no-goes  # skip GOES (slow, ~1.7 GB streamed)

Raw downloads are cached in data/fires/<fire>/raw/ so re-runs are fast.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from firedata import feds, firms, goes, plots, process, wfigs
from firedata.config import FIRES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fire", choices=FIRES)
    ap.add_argument("--no-goes", action="store_true")
    ap.add_argument("--goes-sat", default="goes18")
    ap.add_argument("--viirs-buffer-m", type=float, default=750)   # ~2 VIIRS I-band pixels
    ap.add_argument("--goes-buffer-m", type=float, default=4000)   # 2 km pixel + parallax
    args = ap.parse_args()
    F = FIRES[args.fire]
    raw, proc, figs = F.dir("raw"), F.dir("processed"), F.dir("figures")

    print("WFIGS ...")
    perims = wfigs.download_perimeters_in_bbox(F.bbox, F.start_utc, F.end_utc, raw)
    fire = wfigs.select_fire(perims, F.wfigs_name, F.wfigs_min_acres)
    others = perims.drop(index=fire.index)
    incident = wfigs.download_incident(F.wfigs_name, F.wfigs_min_acres, raw)
    fire.to_file(proc / "wfigs_final_perimeter.geojson", driver="GeoJSON")
    others.to_file(proc / "wfigs_neighbour_perimeters.geojson", driver="GeoJSON")
    discovery = fire.attr_FireDiscoveryDateTime.iloc[0]
    t_min = min(discovery, F.ignition_utc) - pd.Timedelta(hours=1)
    merged = wfigs.incidents_within(fire, F.start_utc, F.end_utc, raw)
    merged.to_csv(proc / "wfigs_merged_incidents.csv", index=False)
    if not merged.empty:
        print("  incidents with origin inside the final perimeter:\n" + merged.to_string(index=False))

    print("FEDS ...")
    feds_ids = feds.find_fire_ids(F.bbox, F.start_utc, F.end_utc)
    feds_ids.to_csv(proc / "feds_fire_ids_in_bbox.csv", index=False)
    print(feds_ids.to_string(index=False))
    if F.feds_fireid is not None:
        fid = F.feds_fireid
    else:  # FEDS id with the largest overlap with the official perimeter would be safer; use the largest for now
        fid = feds.find_fire_ids(fire.total_bounds, F.start_utc, F.end_utc).sort_values("max_farea_km2").fireid.iloc[-1]
    fd = feds.download_feds(fid, raw)
    perim12 = feds.tidy_perimeters(fd["perimeter"])
    perim12.to_file(proc / "feds_perimeters_12h.geojson", driver="GeoJSON")
    perim12.drop(columns="geometry").to_csv(proc / "feds_perimeters_12h.csv", index=False)
    fd["fireline"].to_file(proc / "feds_fireline_12h.geojson", driver="GeoJSON")

    print("VIIRS (FIRMS) ...")
    viirs_all = firms.tidy_viirs(firms.download_viirs(F.bbox, F.start_utc, F.end_utc, raw / "firms"))
    viirs = process.assign_to_fire(viirs_all, fire, others, args.viirs_buffer_m, t_min, F.end_utc)
    print(f"  {len(viirs)} of {len(viirs_all)} VIIRS points assigned to {F.name}")

    goes_pts, scans = pd.DataFrame(), pd.DataFrame(columns=["time_utc"])
    if not args.no_goes:
        print(f"GOES ({args.goes_sat}) ...")
        g_end = F.goes_end_utc or F.end_utc
        pix, scans = goes.download_goes(F.bbox, F.start_utc, g_end, raw, args.goes_sat)
        goes_all = goes.tidy_goes(pix, args.goes_sat)
        goes_pts = process.assign_to_fire(goes_all, fire, others, args.goes_buffer_m, t_min, g_end)
        print(f"  {len(goes_pts)} of {len(goes_all)} GOES pixels assigned, {len(scans)} scans")

    points = process.combined_points(viirs, goes_pts)
    points.to_parquet(proc / "fire_points.parquet")
    points.to_csv(proc / "fire_points.csv", index=False)
    frp = process.frp_timeseries(viirs, goes_pts, scans)
    frp.to_csv(proc / "frp_timeseries.csv", index=False)

    summary = {
        "fire": F.name,
        "wfigs": {
            "unique_fire_id": fire.attr_UniqueFireIdentifier.iloc[0],
            "discovery_utc": str(discovery),
            "origin_lat": float(fire.attr_InitialLatitude.iloc[0]),
            "origin_lon": float(fire.attr_InitialLongitude.iloc[0]),
            "containment_utc": str(fire.attr_ContainmentDateTime.iloc[0]),
            "fire_out_utc": str(fire.attr_FireOutDateTime.iloc[0]),
            "final_acres": float(fire.poly_GISAcres.iloc[0]),
            "perimeter_date_utc": str(fire.poly_PolygonDateTime.iloc[0]),
        },
        "feds": {"fireid": float(fid), "n_steps": len(perim12),
                 "max_area_km2": float(perim12.area_km2.max()),
                 "max_area_acres": float(perim12.area_acres.max()),
                 "last_growth_step_utc": str(perim12[perim12.is_new].step_utc.max())},
        "feds_fire_ids_in_bbox": feds_ids.astype(str).to_dict("records"),
        "merged_incidents": merged.astype(str).to_dict("records"),
        "area_comparison": process.area_comparison(fire, perim12, viirs),
        "points": points.groupby("source").size().to_dict(),
        "first_detection_utc": {s: str(g.time_utc.min()) for s, g in points.groupby("source")},
        "neighbour_fires": others[["poly_IncidentName", "poly_GISAcres", "attr_FireDiscoveryDateTime"]]
            .astype(str).to_dict("records"),
    }
    (proc / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    print("Figures ...")
    plots.make_all(F, points, perim12, fire, frp, figs, merged)


if __name__ == "__main__":
    main()
