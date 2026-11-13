# Bug Fire, CA/NV: fire-progression ground truth

Primary study fire. Built with `python scripts/build_ground_truth.py bug_fire`
(raw downloads cached in `raw/`, not committed).

## Key facts

| | Value | Source |
|---|---|---|
| Incident ID | 2026-NVCCD-030683 (point of origin in CA) | NIFC WFIGS |
| Ignition / discovery | **2026-08-08 22:30:49 UTC** (15:30 PDT) | NIFC WFIGS |
| Point of origin | 39.726833, -120.035167 | NIFC WFIGS |
| First satellite detection | **2026-08-08 22:31 UTC (GOES-18)**, 0.6-1 km from the origin, 28 s after WFIGS discovery; first VIIRS: 2026-08-09 08:58 UTC (S-NPP night pass) | GOES-18 FDC, FIRMS |
| Peak activity | Aug 10-11 (GOES total FRP peak 16.2 GW at 2026-08-10 23:06 UTC) | GOES-18 FDC |
| Growth ended | **~2026-08-13** (last FEDS growth step: night pass of Aug 13 ≈ 09:30 UTC; VIIRS detections almost stop after Aug 13) | FEDS, VIIRS |
| Final size | 93,733 acres (379.3 km²) | NIFC WFIGS final perimeter (dated 2026-10-01) |
| 100% contained | 2026-08-27 23:30 UTC | NIFC WFIGS |
| Fire out | 2026-09-21 01:00 UTC | NIFC WFIGS |
| Merged incident | **Fred Mt** (2026-NVTMFX-030694, 3,482 acres, discovered 2026-08-09 17:33 UTC, origin 39.7755, -119.8471, inside the Bug perimeter) | NIFC WFIGS |

NISAR GCOV same-track pre/post pairs (ASF, `processed/nisar_pairs.csv`). Compare only
within a track; the brief's 07-27 vs 08-15/16 mixes tracks (see issue #7):

| Track | Pre-fire | Post-fire |
|---|---|---|
| A34 | 2026-07-22 13:09 UTC | 2026-08-15 13:09 UTC |
| D42 | 2026-07-23 02:55 UTC | 2026-08-16 02:55 UTC |
| A106 | 2026-07-27 13:01 UTC | 2026-08-20 13:01 UTC |

All post passes are after growth ended, so they all see the final burn scar.

### Differences from the original brief

- Ignition: brief said 23:08 UTC (16:08 PDT); WFIGS discovery is **22:30 UTC**. The ignition
  point in the brief (39.727395, -120.0372941) is ~200 m from the WFIGS origin; the config
  keeps the brief's point and the pipeline uses the earlier of the two times.
- Growth: brief said the fire reached its size ~Aug 14-20. Satellite data show growth
  **ended ~Aug 13**. WFIGS has no acreage history to check when the official size was
  reported.
- FEDS area is much smaller than the official size (see below).

## FEDS vs official area

| | Area |
|---|---|
| NIFC WFIGS final perimeter | 379.3 km² (93,733 acres) |
| FEDS final perimeter (fireid 69872) | 143.5 km² (35,459 acres), 38% of WFIGS |
| Overlap | 129.0 km² |
| FEDS outside WFIGS | 14.5 km² |
| WFIGS not covered by FEDS | 250.4 km² |
| Union of all VIIRS 375 m pixel footprints | 164.2 km² (146.6 km² inside WFIGS) |
| Inside WFIGS, detected by VIIRS but not in FEDS | 49.0 km² (268 of 1,687 VIIRS points lie outside the FEDS final perimeter) |
| Inside WFIGS, never detected by VIIRS and not in FEDS | 201.4 km² |
| FEDS ∪ VIIRS footprints, inside WFIGS | 178.0 km² (47% of WFIGS) |

FEDS fire IDs in the search box, Aug 8-27 (`processed/feds_fire_ids_in_bbox.csv`):

| fireid | max area | matches | overlaps Bug |
|---|---|---|---|
| 69872 | 143.4 km² | Bug (+ Fred Mt) | yes |
| 70286 | 114.1 km² | Stallion (WFIGS 164.7 km²) | no |
| 73891 | 45.4 km² | Hawk (WFIGS 61.0 km²) | no |

Findings:
- FEDS did **not** split the Bug Fire: one FEDS ID (69872) covers it, and that ID already
  includes the merged Fred Mt fire. Summing all three IDs (74.8k acres) would add two
  unrelated fires.
- The 250 km² of WFIGS area missing from FEDS has two parts:
  - **49 km²: FEDS processing.** VIIRS detected fire there, but FEDS did not include it,
    mostly scattered detections in the north-centre and east of the fire (visible as dots
    outside the blue polygons in `figures/feds_vs_wfigs_final.png`).
  - **201 km²: no detection at all.** No VIIRS pixel ever fired there. Possible reasons
    (not verified): fast runs in grass/brush that burned out between overpasses (~12 h
    gaps), and unburned islands that the official perimeter does not exclude. GOES (2 km)
    cannot resolve this at the needed scale. The NISAR pre/post burn-scar change should
    answer it.
- FEDS also underestimates the neighbouring fires (Stallion 69%, Hawk 74% of WFIGS), but
  much less than Bug (38%).

See `figures/feds_vs_wfigs_final.png`.

### Does GOES cover the area VIIRS missed?

GOES-18 fire-pixel footprints come from `firedata.goes.pixel_footprints`, which uses the
fixed-grid pixel corners. Here a pixel is 6.8 km² (~2.2 x 3.1 km). Ambiguous pixels are excluded.

| GOES pixels used | Pixels | Footprint | Inside WFIGS | Covers the 201.4 km² gap |
|---|---|---|---|---|
| detected in >= 1 scan | 102 | 697.6 km² | 361.5 km² | 188.4 km² (94%) |
| detected in >= 2 scans | 97 | 663.5 km² | 360.4 km² | 187.7 km² (93%) |
| detected in >= 6 scans | 80 | 547.1 km² | 336.6 km² | 172.0 km² (85%) |

FEDS ∪ VIIRS ∪ GOES covers 366.4 km² (97%) of the WFIGS perimeter.

Interpretation: almost all of the no-VIIRS gap lies inside a GOES fire pixel, so it was next to
active fire. GOES cannot show that the gap itself burned: a 6.8 km² pixel fires when a small
part of it burns, and the GOES footprint (698 km²) is 1.8x the WFIGS area. 336 km² of it is
outside WFIGS. The NISAR burn-scar check (separate issue) has to settle this.

## Ground truth for calibration (decision)

Decided on 2026-11-13:
- **Final extent: NIFC WFIGS final perimeter** (`processed/wfigs_final_perimeter.geojson`).
- **Timing: arrival times from VIIRS + GOES detections** (`processed/fire_points.*`,
  rows with `ambiguous = False`). VIIRS gives 375 m locations; GOES gives 5 min timing.
- **FEDS 12 h perimeters: direction and shape only.** Their area is a lower bound (38% of the
  final size), not an area target.

The 12 h FEDS table is unchanged. Unburned islands inside the WFIGS perimeter and the
201 km² never detected by VIIRS are to be checked with NISAR (separate issue).

## Data sources

| Data | Access | Notes |
|---|---|---|
| VIIRS 375 m active fire (S-NPP, NOAA-20, NOAA-21) | FIRMS area API (`FIRMS_MAP_KEY`) | August 2026 is only in the NRT products (`VIIRS_*_NRT`); the standard (SP) archive ended 2026-07-31 when downloaded. Max 5 days per request. |
| FEDS perimeters, fire line, new pixels | VEDA OGC API Features, `https://openveda.cloud/api/features`, collections `public.eis_fire_lf_{perimeter,fireline,newfirepix}_nrt` | No key. The `_archive` collections have no 2026 data. |
| GOES-18 ABI L2 FDC (CONUS, 5 min) | `s3://noaa-goes18/ABI-L2-FDCC/` (anonymous) | Only in-bbox fire pixels are kept. |
| Official perimeter, incidents | NIFC WFIGS ArcGIS REST (`WFIGS_Interagency_Perimeters_YearToDate`, `WFIGS_Incident_Locations_YearToDate`) | No key. Only the final perimeter is published, no daily history. |

### FEDS time convention

A FEDS step `t` is a 12 h label, not an observation time. Matching FEDS new pixels to VIIRS
acquisition times shows: `T00` = night passes of that UTC date (~08:30-11:00 UTC),
`T12` = afternoon passes (~20:00-21:30 UTC). The tables keep the label (`step_utc`) and add
`obs_time_utc` = step + 9.5 h (T00) or + 8.5 h (T12). When the fire stops growing FEDS
repeats the last perimeter; `is_new = False` marks those rows.

### Assigning detections to the fire

The search box (-120.6, 39.2, -119.4, 40.3) also contains the Stallion (from Aug 10, east)
and Hawk (from Aug 22, south) fires and a few small ones (e.g. Hunter, which was burning before
the Bug ignition). A detection is kept if it is within 750 m (VIIRS) or 4 km (GOES, 2 km pixel
+ uncorrected parallax) of the Bug final perimeter, no closer to any other WFIGS perimeter,
and no earlier than 1 h before discovery. Perimeters lying mostly inside the Bug perimeter
(e.g. Ringo, 78 acres, Aug 2) are not treated as competing fires.

Points that are also within the buffer of a neighbouring fire get `ambiguous = True` and are
left out of the FRP totals. For Bug these are 18 GOES pixels on Aug 23 at the south edge,
most likely from the Hawk Fire, which was making its main run while Bug was quiet.

Result: 1,687 VIIRS points (S-NPP 523, NOAA-20 576, NOAA-21 588) and 4,670 GOES-18 pixels
from 5,757 scans (Aug 8-27; 3 scans missing in the bucket).

## Outputs

`processed/`
- `fire_points.csv|parquet`: `time_utc, lat, lon, frp_mw, source, confidence` + extras
  (VIIRS: brightness, scan/track, day/night; GOES: fire temp, fire area m², mask, DQF,
  `in_frp_total`). `source` ∈ VIIRS_SNPP, VIIRS_NOAA20, VIIRS_NOAA21, GOES18_FDC.
- `feds_perimeters_12h.csv|geojson`: one row per 12 h step: `step_utc, obs_time_utc,
  feds_t_utc, is_new, area_km2, area_acres, n_pixels, n_newpixels, meanfrp, flinelen, ...`
- `feds_fireline_12h.geojson`: active fire front per step.
- `frp_timeseries.csv`: total FRP per VIIRS overpass and per GOES scan (scans with no fire = 0).
- `wfigs_final_perimeter.geojson`, `wfigs_neighbour_perimeters.geojson`,
  `wfigs_merged_incidents.csv`, `feds_fire_ids_in_bbox.csv`, `summary.json`.

`figures/`
- `perimeters_12h_map.png`: FEDS perimeters colored by time, NISAR passes marked
- `feds_vs_wfigs_final.png`: FEDS final vs WFIGS final
- `burned_area_vs_time.png`
- `frp_vs_time.png`: VIIRS + GOES total FRP
- `viirs_frp_map.png`
- `bug_fire_spread.html`: interactive time-lapse (open in a browser; no server needed).
  Rebuild with `python -m firedata.web bug_fire`.

## Caveats

- VIIRS data are NRT (version 2.0NRT); re-run when the SP archive covers August 2026.
- GOES FRP counts only fire categories good, saturated, cloud-contaminated, high- and
  medium-probability (low-probability excluded); some fire pixels have no FRP
  (`n_without_frp`). GOES pixel locations are not terrain-corrected.
- Basemap tiles: Esri World Shaded Relief.
