# Progress: more fires (Crosswhite, Colwash, Biscar) + summary, website, README

```
Overall: [██████░░░░░░░░░░] 38%   elapsed 14m / est. total ~6h29m  → ~6h15m left
Last update: 2026-11-13 11:28 UTC
```

| # | Step | Status | Est. | Actual |
|---|---|---|---|---|
| 1 | Generalize pipeline (multi-id FEDS, NISAR lookup, growth end, metrics) | ✅ | 1h30m | 15m |
| 2 | Crosswhite (OR) | 🔄 | 1h30m | 5m |
| 3 | Colwash (WA) | 🔄 | 45m | 5m |
| 4 | Biscar (CA) | 🔄 | 45m | 5m |
| 5 | Summary file + accuracy % | ⏳ | 1h00m |  |
| 6 | Multi-fire website (tested in a browser) | ⏳ | 1h30m |  |
| 7 | Short README (Hebrew) | ⏳ | 1h00m |  |

## Done so far

- Step 0 (PR #6): CLAUDE.md fixed (ignition 22:30 UTC, growth ended ~Aug 13); live page moved to `firedata.web.build_page()`.
- Issue #7: posted the NISAR track-pairing correction (07-22→08-15 A34, 07-23→08-16 D42, 07-27→08-20 A106).
- Pipeline generalized: all FEDS ids overlapping the official perimeter are merged; NISAR passes and same-track pairs from ASF; growth-end rule; per-fire metrics (`firedata.metrics`). Bug Fire re-run: same numbers.

## Problems / data mismatches

- Branch: `more-fires` is based on `bug-fire-data` (PR #6, not merged) because master has no `src/` yet. The new PR is stacked on #6.
- Crosswhite: README said ~343,000 acres = incident size (342,923); the WFIGS perimeter is 355,065 acres. It is a child of the Rowe Creek Complex (373,929 acres). WFIGS has no containment date.
- Biscar: WFIGS point of origin is ~2 km outside the final perimeter; the official perimeter is dated Jul 24 but containment was Aug 9.
- Colwash: official perimeter dated 9.4 h after discovery and never updated.
- FEDS: Biscar is split over several FEDS ids (max 20 km² each) and Colwash has almost no FEDS area (6 km² of 158 km²).

## Open questions for Ori

- (none yet)
