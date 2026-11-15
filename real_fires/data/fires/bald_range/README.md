# Bald Range fire, BC (2026), Canadian test fire: RCM C-band vs NISAR L-band

| Fact | Value | Source |
|---|---|---|
| Incident | K51490 "Bald Range", Okanagan, BC; cause unknown | BC Wildfire Service (BCWS) `BCWS_ActiveFires_PublicView` |
| Ignition (BCWS "ignition date") | 2026-08-07 23:20 UTC at 49.6262, -119.8835 | BCWS |
| Final size | 25,163 ha (polygon area 25,163 ha), status Out, perimeter dated 2026-08-29 | BCWS `BCWS_FirePerimeters_PublicView` |
| Fire out | 2026-10-28 17:11 UTC | BCWS |

Official extent = BCWS final perimeter (`processed/bcws_final_perimeter.geojson`, from `firedata/bcws.py`;
WFIGS is US only). The full VIIRS / GOES / FEDS ground-truth pipeline has not been run for this fire yet.

## RCM (RADARSAT Constellation Mission), C-band

Data: public RCM CEOS-ARD on AWS (`s3://rcm-ceos-ard`, NRCan/CCRS, RCM public EULA, no account): 30 m
compact-pol (RR, RL, RRRL), terrain-flattened gamma0, 20 m COG pixels. Only a ~29 × 26 km window is read.

Scenes, same beam and geometry, SC30MCPA ascending ~01:34 UTC (18:34 PDT the evening before):
08-02 (5.4 d before ignition) → 08-10 (2.1 d after) → 08-22 (14.1 d after); 89 % of the perimeter imaged.

**Why burned = RL increase (not a drop).** In C-band (5.6 cm) the forest canopy (needles, twigs, small branches)
dominates backscatter. Fire removes that canopy layer, so the radar sees the ground: a rough, charred surface,
often wetter than before because transpiration and interception stop. A rough, wet surface backscatters more than
the canopy did. In boreal fire scars this brightening is well documented: ERS-1 C-band scars in Alaska were reported
3–6 dB (one abstract: 3–8 dB) brighter than adjacent forest (Kasischke et al.), and ERS + RADARSAT across Canadian
boreal regions showed the same, but **only when the scar was wet**, and depending on incidence angle
(Bourgeau-Chavez et al. 2002, IJRS). Abstract figures only; full papers not checked. L-band (NISAR, 24 cm) reaches
trunks and large branches, so canopy loss usually shows as an HV **drop** there. Bald Range is dry interior forest
(not boreal peat), so a much smaller C-band rise than +3–6 dB is expected.

Rule (fixed beforehand, `scripts/rcm_burned_area.py`): RL change, mean of 5 × 5 pixels (100 m), each later scene
quantile-matched to the reference on unburned pixels (outside perimeter + 2 km); burned where RL rose ≥ 1.0 dB.
Sweep 0.5–3 dB in `processed/rcm_burned_scores.csv`.

Rejected scenes:
- SC30MCPD 08-09 14:08 UTC (RCM-1): RL / RR ~11.9 dB higher than every other date, compressed range and a seam;
  metadata identical to good scenes. Radiometrically broken, dropped.
- SC30MCPC (08-06 / 08-18 / 08-30): the product footprint polygon contains the origin, but the data swath covers
  0–0.1 % of the window. Not usable here.

## NISAR, L-band

Same-track GCOV pairs (ASF): A5 07-20 → 08-13 / 08-25, A77 07-25 → 08-18 / 08-30, D71 07-25 → 08-18 / 08-30,
D172 07-20 → 08-13 / 08-25. No NISAR acquisition between ~07-27 and 08-12 on any track. Rule unchanged from the
Bug Fire (`nisar_provider`: HV drop ≥ 1.5 dB, 3 × 3 cells of 100 m). Only the fire window is read.

## Findings

All scores on one 100 m grid vs the BCWS perimeter (`processed/nisar_vs_rcm_scores.csv`); AUC = how well the
change separates inside from outside the perimeter (0.5 = none, 1 = perfect).

- **RCM C-band RL rise ≥ 1 dB:** 08-10 (2 d after ignition) cover 29 %, precision 48 %, IoU 22 %, AUC 0.62;
  08-22 (14 d) cover 43 %, precision 51 %, IoU 30 %, AUC 0.67. False alarms outside are high (16–22 %).
- **NISAR L-band, existing HV-drop rule (≥ 1.5 dB):** almost nothing detected (cover 0–1.6 %, precision 96–100 %
  where anything is found). AUC of the drop is 0.53–0.69 on the 6 usable pairs; the change is real but mostly
  below 1.5 dB. Track A5 returned no valid pixels over the perimeter (0 %).
- **Opposite response holds in direction, weakly:** inside the perimeter L-band HV tends to drop and C-band RL
  tends to rise (both AUC > 0.5 on every usable pair), visible as red vs blue patches in `figures/nisar_vs_rcm.png`.
  The magnitudes are small (C-band median rise +0.5 to +0.8 dB vs. +3–6 dB reported for wet boreal scars).
- **Combination does not beat RCM alone:** NISAR OR RCM = RCM's numbers +0.0–0.9 IoU points (NISAR adds < 2 % of
  area); NISAR AND RCM keeps precision 100 % but cover ≤ 0.8 %. With the NISAR rule unchanged, NISAR adds nothing.
- RCM gives a usable during-fire map 2 days after ignition, when NISAR had no pass (gap ~07-27 to 08-12).

Outputs: `processed/rcm_burned_scores.csv`, `rcm_change_diagnostics.csv`, `rcm_growth_persistent.csv`,
`nisar_vs_rcm_scores.csv`; figures `figures/rcm_burned_area_growth.png`, `figures/nisar_vs_rcm.png`.

## Open questions

1. **Beam choice.** You asked for SC30MCPC with 08-06 as "before", but its data swath misses the fire (see above).
   Safer option taken: SC30MCPA, same geometry on all dates, 08-02 → 08-10 → 08-22. Alternative: SC30MCPD 08-01 →
   08-25 / 09-02 (08-09 broken), which showed a weaker rise (median +0.7 dB inside on 09-02).
2. **RCM threshold.** 1.0 dB was fixed before scoring on this chain; at 0.5 dB cover rises but false alarms go
   above 30 %. Not re-tuned on this fire.
3. **BCWS ignition time** is taken as UTC from the service; not yet cross-checked against the first VIIRS / GOES
   detection (pipeline not run).
4. **Sources for +5–6 dB (RADARSAT-1).** The figure in the brief could not be confirmed; the README quotes the
   abstract ranges (+3–6 / 3–8 dB, ERS-1 / ERS + RADARSAT) instead.
