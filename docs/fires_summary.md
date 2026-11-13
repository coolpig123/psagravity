# Fire ground truth: data and agreement summary

One table per fire, one row per data item. Every agreement value is measured against the **official final perimeter** (NIFC WFIGS). Definitions are below the tables.

## Bug Fire, CA/NV

Official area 93,733 acres (379.3 km²). FEDS ∪ VIIRS ∪ GOES cover 96.6% of it. Details: `data/fires/bug_fire/README.md`.

| Item | Source | Time range | Resolution | Update rate | Coverage | Agreement vs official |
|---|---|---|---|---|---|---|
| FEDS | NASA FEDS VIIRS modeled fire perimeters (VEDA OGC API, *_nrt) | 2026-08-09 09:30 to 2026-08-15 09:30 UTC | polygons from 375 m VIIRS pixels | 12 h | 9 perimeters with new data (22 steps); fire id(s) 69872 | IoU 32.7%<br>area ratio 37.8% |
| VIIRS | VIIRS 375 m active fire, S-NPP + NOAA-20 + NOAA-21 (NASA FIRMS, VIIRS_NOAA20_NRT, VIIRS_NOAA21_NRT, VIIRS_SNPP_NRT) | 2026-08-09 08:58 to 2026-08-19 22:14 UTC | 375 m (pixel grows to ~0.8 km at scan edge) | ~6 overpasses/day (3 satellites × day/night) | 1,687 detections | covers 38.7%<br>inside 89.3%<br>first detection +10h27m<br>3.03 km from origin |
| GOES | GOES-18 ABI L2 FDC, CONUS sector (NOAA, AWS) | 2026-08-08 22:31 to 2026-08-26 21:06 UTC | 2 km nominal; ~6.8 km² per pixel here | 5 min | 4,652 fire pixels in 5,757 scans | covers 95.3%<br>inside 51.8%<br>first detection +0 min<br>1.34 km from origin |
| Final perimeter | NIFC WFIGS interagency perimeter (2026-NVCCD-030683) | perimeter dated 2026-10-01 18:38 UTC | mapped polygon (Mixed Methods) | final only (no history) | 93,733 acres (379.3 km²) | reference (100%) |
| NISAR GCOV | NISAR L2 GCOV (ASF), standard production | A106: 07-27 → 08-20; A34: 07-22 → 08-15; D42: 07-23 → 08-16 | L-band HH/HV, geocoded grid | 12-day repeat per track; several tracks | 24 passes, 3 same-track pre/post pairs (shortest gap 24 days) | N/A: burn-scar agreement not computed yet (issue #7) |

## Colwash Fire, WA

Official area 39,103 acres (158.2 km²). FEDS ∪ VIIRS ∪ GOES cover 100.0% of it. Details: `data/fires/colwash/README.md`.

| Item | Source | Time range | Resolution | Update rate | Coverage | Agreement vs official |
|---|---|---|---|---|---|---|
| FEDS | NASA FEDS VIIRS modeled fire perimeters (VEDA OGC API, *_nrt) | 2026-08-23 09:30 to 2026-08-23 20:30 UTC | polygons from 375 m VIIRS pixels | 12 h | 2 perimeters with new data (11 steps); fire id(s) 73983 | IoU 2.9%<br>area ratio 3.8% |
| VIIRS | VIIRS 375 m active fire, S-NPP + NOAA-20 + NOAA-21 (NASA FIRMS, VIIRS_NOAA20_NRT, VIIRS_NOAA21_NRT, VIIRS_SNPP_NRT) | 2026-08-22 21:38 to 2026-08-24 11:13 UTC | 375 m (pixel grows to ~0.8 km at scan edge) | ~6 overpasses/day (3 satellites × day/night) | 118 detections | covers 6.2%<br>inside 79.1%<br>first detection +6h39m<br>7.11 km from origin |
| GOES | GOES-18 ABI L2 FDC, CONUS sector (NOAA, AWS) | 2026-08-22 15:46 to 2026-08-25 21:16 UTC | 2 km nominal; ~8.1 km² per pixel here | 5 min | 1,121 fire pixels in 2,302 scans | covers 100%<br>inside 36.1%<br>first detection +48 min<br>7.03 km from origin |
| Final perimeter | NIFC WFIGS interagency perimeter (2026-WAYAA-000090) | perimeter dated 2026-08-23 00:20 UTC | mapped polygon (Mixed Methods) | final only (no history) | 39,103 acres (158.2 km²) | reference (100%) |
| NISAR GCOV | NISAR L2 GCOV (ASF), standard production | A106: 08-20 → 09-01; A5: 08-13 → 08-25; D71: 08-18 → 08-30; D143: 08-11 → 09-04 | L-band HH/HV, geocoded grid | 12-day repeat per track; several tracks | 27 passes, 4 same-track pre/post pairs (shortest gap 12 days) | N/A: burn-scar agreement not computed yet (issue #7) |

## Biscar Fire, CA

Official area 69,352 acres (280.7 km²). FEDS ∪ VIIRS ∪ GOES cover 94.1% of it. Details: `data/fires/biscar/README.md`.

| Item | Source | Time range | Resolution | Update rate | Coverage | Agreement vs official |
|---|---|---|---|---|---|---|
| FEDS | NASA FEDS VIIRS modeled fire perimeters (VEDA OGC API, *_nrt) | 2026-07-20 20:30 to 2026-07-22 20:30 UTC | polygons from 375 m VIIRS pixels | 12 h | 3 perimeters with new data (14 steps); fire id(s) 64520, 64525, 64741 | IoU 10.2%<br>area ratio 11.2% |
| VIIRS | VIIRS 375 m active fire, S-NPP + NOAA-20 + NOAA-21 (NASA FIRMS, VIIRS_NOAA20_SP, VIIRS_NOAA21_SP, VIIRS_SNPP_SP) | 2026-07-18 19:57 to 2026-07-22 21:16 UTC | 375 m (pixel grows to ~0.8 km at scan edge) | ~6 overpasses/day (3 satellites × day/night) | 623 detections | covers 25.9%<br>inside 90.4%<br>first detection +1h51m<br>8.19 km from origin |
| GOES | GOES-18 ABI L2 FDC, CONUS sector (NOAA, AWS) | 2026-07-18 20:06 to 2026-08-11 19:51 UTC | 2 km nominal; ~7.0 km² per pixel here | 5 min | 1,489 fire pixels in 7,195 scans | covers 90.6%<br>inside 62.9%<br>first detection +2h00m<br>10.35 km from origin |
| Final perimeter | NIFC WFIGS interagency perimeter (2026-CANOD-004474) | perimeter dated 2026-07-24 04:36 UTC | mapped polygon (Mixed Methods) | final only (no history) | 69,352 acres (280.7 km²) | reference (100%) |
| NISAR GCOV | NISAR L2 GCOV (ASF), standard production | A106: 07-15 → 07-27; A34: 07-10 → 07-22; D42: 07-11 → 07-23; D143: 07-18 → 08-11 | L-band HH/HV, geocoded grid | 12-day repeat per track; several tracks | 34 passes, 4 same-track pre/post pairs (shortest gap 12 days) | N/A: burn-scar agreement not computed yet (issue #7) |

## Crosswhite Fire, OR

**In progress**: data download / processing not finished yet. This table will be filled in when the pipeline run completes.

## Metric definitions

W = official final perimeter (NIFC WFIGS). Pixel footprints: VIIRS = scan × track rectangle of each 375 m detection; GOES = fixed-grid pixel corners. Points flagged `ambiguous` (also near a neighbouring fire) are excluded.

- **feds_iou**: IoU of the final FEDS perimeter with the official perimeter W: area(FEDS ∩ W) / area(FEDS ∪ W).
- **feds_area_ratio**: Final FEDS area / official area.
- **viirs_cover**: Share of the official area inside at least one VIIRS 375 m pixel footprint.
- **viirs_inside**: Share of the VIIRS pixel-footprint area that lies inside the official perimeter.
- **goes_cover**: Share of the official area inside at least one GOES fire-pixel footprint (pixels are ~7 km²).
- **goes_inside**: Share of the GOES fire-pixel footprint area that lies inside the official perimeter.
- **all_cover**: Share of the official area covered by FEDS, VIIRS or GOES (union).
- **ignition_error_min**: First detection time minus official discovery time, in minutes (positive = detected later).
- **ignition_dist_km**: Distance from the first detection to the official point of origin, in km.
- **all_cover**: share of W covered by the union of the final FEDS perimeter, VIIRS footprints and GOES footprints.

**Reading the numbers.** *cover* says how much of the official area a source saw; *inside* says how much of what it saw is inside the official perimeter (low = coarse pixels or detections spilling outside). GOES pixels are ~7 km², so GOES cover is high and GOES inside is low by construction. The official perimeter is an outer boundary and can include unburned islands, so no source is expected to reach 100%.

**N/A rules.** NISAR burn-scar agreement is N/A until issue #7 is done. A metric is N/A when the source has no data for the fire (e.g. no FEDS id overlaps the official perimeter).
