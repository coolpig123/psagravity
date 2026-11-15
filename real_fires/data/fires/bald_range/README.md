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
Bug Fire calibration (HV drop ≥ 0.25 dB, 3 × 3 cells of 100 m; the provider default is still 1.5 dB). Only the fire window is read.

## Findings

All scores on one 100 m grid vs the BCWS perimeter (`processed/nisar_vs_rcm_scores.csv`). AUC = how well the
change separates inside from outside the perimeter (0.5 = none, 1 = perfect); it does not depend on the threshold.
NISAR threshold = **0.25 dB HV drop, calibrated on the Bug Fire only** (best mean IoU over 3 same-track pairs,
`../bug_fire/processed/nisar_threshold_calibration.csv`, issue #7) and applied here unchanged.

| Sensor / rule | Pair | Days after ignition | Cover % | Precision % | IoU % | AUC |
|---|---|---|---|---|---|---|
| RCM RL rise ≥ 1 dB | SC30MCPA 08-02 → 08-10 | 2.1 | 29.2 | 47.9 | 22.2 | 0.62 |
| RCM RL rise ≥ 1 dB | SC30MCPA 08-02 → 08-22 | 14.1 | 43.3 | 50.8 | 30.5 | 0.67 |
| NISAR HV drop ≥ 0.25 dB | D172 07-20 → 08-13 | 5.2 | 18.1 | 93.8 | 17.9 | 0.55 |
| NISAR HV drop ≥ 0.25 dB | D71 07-25 → 08-18 | 10.2 | 22.1 | 98.6 | 22.1 | 0.69 |
| NISAR HV drop ≥ 0.25 dB | A77 07-25 → 08-18 | 10.6 | 27.3 | 89.4 | 26.4 | 0.63 |
| NISAR HV drop ≥ 0.25 dB | D172 07-20 → 08-25 | 17.2 | 30.5 | 92.1 | 29.7 | 0.68 |
| NISAR HV drop ≥ 0.25 dB | D71 07-25 → 08-30 | 22.2 | 0.7 | 98.9 | 0.7 | 0.53 |
| NISAR HV drop ≥ 0.25 dB | A77 07-25 → 08-30 | 22.6 | 3.9 | 88.8 | 3.9 | 0.53 |
| NISAR OR RCM | D172 08-25 + RCM 08-22 | 17.2 | 59.1 | 57.1 | **40.9** | – |
| NISAR OR RCM | D71 08-18 + RCM 08-22 | 10.2 | 55.3 | 56.7 | 38.9 | – |
| NISAR OR RCM | A77 08-18 + RCM 08-22 | 10.6 | 57.5 | 56.3 | 39.8 | – |
| NISAR OR RCM (during fire) | D172 08-13 + RCM 08-10 | 5.2 | 40.0 | 55.0 | 30.1 | – |
| NISAR AND RCM | D172 08-25 + RCM 08-22 | 17.2 | 15.2 | 97.5 | 15.2 | – |

- **NISAR alone** (Bug-calibrated 0.25 dB): high precision (89–99 %), cover 18–31 % on the 08-13 to 08-25 pairs.
  The 08-30 pairs nearly vanish (AUC 0.53) for reasons not yet understood (possibly a wet or windy acquisition).
  Track A5 returned no valid pixels over the perimeter. With the old 1.5 dB rule NISAR found ≤ 1.6 % (previous run).
- **RCM alone:** cover 29–43 %, but precision ~50 % and 16–22 % false alarms outside the perimeter.
- **Opposite response holds in direction, weakly:** L-band HV drops and C-band RL rises inside the perimeter
  (AUC > 0.5 on every usable pair; red vs blue in `figures/nisar_vs_rcm.png`); C-band median rise is only
  +0.5 to +0.8 dB, far below the +3–6 dB reported for wet boreal scars.
- **NISAR OR RCM beats each alone:** IoU 38.9–40.9 % vs 30.5 % (RCM 08-22) and 22–30 % (NISAR), i.e. +8 to +10
  IoU points over RCM alone, mostly from cover (55–59 %); precision stays ~57 %, limited by RCM's false alarms.
  During the fire (08-13 + 08-10): 30.1 % vs 22.2 % RCM alone. NISAR AND RCM is very precise (95–99 %) but finds
  only 8–15 % of the area.
- RCM gives a during-fire map 2 days after ignition, when NISAR had no pass (gap ~07-27 to 08-12).

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
5. **NISAR threshold at the edge of the grid.** 0.25 dB was the lowest value tested on the Bug Fire and still the
   best mean IoU, so an even lower threshold might score higher; not tested. `nisar_provider.SCAR_DROP_DB` (1.5 dB,
   used for the spread-model inputs) was left unchanged; changing it is a separate decision.

## Radar fire state as the spread model's start (`firedata/fire_state.py`, `scripts/fire_state_demo.py`)

`fire_state(fire, time)` → burned-area polygon from the latest RCM (RL rise ≥ 1 dB) and / or NISAR (HV drop
≥ 0.25 dB) scene before `time` (sensors within 4 days OR-ed). It drops specks < 5 ha, fills holes < 5 ha and keeps
the largest clump reaching within 2 km of the ignition. The window is ignition ± 15 km; the BCWS perimeter is
not used. Demo: start = RCM 08-10 01:34 (2 days after ignition, before any NISAR pass), **3,378 ha**, which is
less than the ~9,500 ha reported on 08-08 local time, so the state under-maps. Model = `firespread.predict` + `simulate_spread`,
Bug-calibrated hindcast parameters held fixed (2 h per step, burn_prob ≥ 0.9). New-area scores
(`processed/fire_state_demo_scores.csv`):

| Target | Model IoU / precision / recall | Same-area circle IoU |
|---|---|---|
| FEDS 08-11 00:00 (+22 h) | 0.61 / 0.94 / 0.63 | 0.54 |
| FEDS 08-12 00:00 (+46 h) | 0.70 / 0.85 / 0.80 | 0.61 |
| FEDS 08-14 00:00 (+94 h) | 0.61 / 0.65 / 0.91 | 0.57 |
| Radar state 08-22 (12 d) | 0.10 / 0.10 / 0.93 | 0.11 |
| BCWS final 08-29 (19 d) | 0.18 / 0.18 / 0.97 | 0.18 |

The model beats the circle by +4 to +9 IoU points up to ~4 days. After that it burns most of the 15 km window
(85,000–122,000 ha new vs 9,600–22,000 ha observed) and is no better than the circle. FEDS is a single NRT
fireid (11710) and the scores are from one fire, so treat them as a demo. Figure: `figures/fire_state_demo.png`.

**Suggestions for spread_model (not changed here):** the over-prediction at long horizons comes from fixed
start-hour weather and no burnout / suppression. Possible fixes: (1) hourly weather inside `simulate_spread`
(the inputs already hold 26 h); (2) a per-step extinction or a cap on CA steps per day for night-time and
suppression; (3) accept a radar fire state (polygon plus scene list) as a `predict` input next to FEDS.
