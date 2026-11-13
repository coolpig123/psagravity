# Progress: more fires (Crosswhite, Colwash, Biscar) + summary, website, README

```
Overall: [██████░░░░░░░░░░] 38%   elapsed 50m / est. total ~4h15m  → ~3h25m left
Last update: 2026-11-13 12:04 UTC
```

| # | Step | Status | Est. | Actual |
|---|---|---|---|---|
| 1 | Generalize pipeline (multi-id FEDS, NISAR lookup, growth end, metrics) | ✅ | 1h30m | 15m |
| 2 | Crosswhite (OR) | 🔄 | 1h30m | 50m |
| 3 | Colwash (WA) | ✅ | 45m | 25m |
| 4 | Biscar (CA) | ✅ | 45m | 25m |
| 5 | Summary file + accuracy % | 🔄 | 1h00m | 15m |
| 6 | Multi-fire website (tested in a browser) | 🔄 | 1h30m | 20m |
| 7 | Short README (Hebrew) | 🔄 | 1h00m | 10m |

## Done so far

- Step 0 (PR #6): CLAUDE.md fixed (ignition 22:30 UTC, growth ended ~Aug 13); live page moved to `firedata.web.build_page()`.
- Issue #7: posted the NISAR track-pairing correction (07-22→08-15 A34, 07-23→08-16 D42, 07-27→08-20 A106).
- Pipeline generalized: all FEDS ids overlapping the official perimeter are merged; NISAR passes and same-track pairs from ASF; growth-end rule; per-fire metrics (`firedata.metrics`). Bug Fire re-run: same numbers.
- Colwash (WA): done. VIIRS covers only 6% of the official area; GOES saw the fire 48 min after discovery and covers 100% (coarse pixels); FEDS IoU 3%. 4 NISAR same-track pairs (12-day gaps). README + interactive page.
- Biscar (CA): done. FEDS split over 3 ids (IoU 10%); VIIRS covers 26%, GOES 91%; growth over by ~Jul 21 (95% GOES energy Jul 20 22:56 UTC). 4 NISAR same-track pairs. README + interactive page.
- Summary (`docs/fires_summary.md` + `.csv`) generated for Bug, Colwash, Biscar with metric definitions; Crosswhite listed as in progress.
- Multi-fire site in `app/` (fire picker, on-demand data, works from file://). Browser test `scripts/test_app.py` passes for all 3 fires at desktop and phone width. Crosswhite to be added.
- Root README: short Hebrew overview added at the top (what, data and why, status, numbers per fire, how to run, where things live). Crosswhite numbers pending.

## Problems / data mismatches

- Branch: `more-fires` is based on `bug-fire-data` (PR #6, not merged) because master has no `src/` yet. The new PR is stacked on #6.
- Crosswhite: README said ~343,000 acres = incident size (342,923); the WFIGS perimeter is 355,065 acres. It is a child of the Rowe Creek Complex (373,929 acres). WFIGS has no containment date.
- Biscar: WFIGS point of origin is 6.6 km outside the final perimeter (first detections ~8 km away, at 40.618, -120.300); the official perimeter is dated Jul 24 but containment was Aug 9.
- Colwash: official perimeter dated 9.4 h after discovery and never updated.
- FEDS: Biscar is split over several FEDS ids (max 20 km² each) and Colwash has almost no FEDS area (6 km² of 158 km²).
- FEDS: fire ids are not unique across FEDS regions; Crosswhite is tracked twice (CONUS 63374 and BorealNA 9744). Pipeline now keys by (id, region) and keeps CONUS.
- FEDS API is flaky: HTTP 500 on heavy pages and occasional empty answers. Pipeline now shrinks pages and retries; empty answers are not cached.
- Colwash: WFIGS point of origin is ~1.4 km north of the final perimeter.
- Growth-end rule: when FEDS is too small, the pipeline now uses 95% of GOES fire energy (Colwash: 2026-08-23 04:26 UTC). For Bug the GOES rule would give Aug 11 23:26 vs the FEDS rule's Aug 13 09:30 (Bug keeps the FEDS rule).
- Crosswhite: first run failed on FEDS (server errors, duplicate region ids); fixed and re-running. GOES download is 36 days, the longest step.

## Open questions for Ori

- Biscar ignition point: the official WFIGS origin (40.5457, -120.3316) is 6.6 km outside the perimeter; the first VIIRS/GOES detections are at ~40.618, -120.300. Safer choice taken: data keep the official point and the README reports the detection-based point. For the spread model I suggest starting from the detection-based point. Your call.
