# CLAUDE.md

## Project

psagravity: NASA NISAR hackathon (Space Apps) project. See README.md for the full description.

We are building a **wildfire-spread prediction system**: given a fire's ignition point and size,
predict where it will spread. Inputs:

- NISAR L-band SAR layers (GCOV HH/HV, SME2): fuel / biomass, dryness, already-burned areas
- VIIRS 375 m active fire (location, FRP)
- GOES ABI fire detection (every 5 min)
- FEDS fire perimeters (every 12 h)
- DEM (NASADEM), wind / weather (MERRA-2, Google Weather)

Spread model: probabilistic cellular automaton (Alexandridis et al. 2008), parameters learned
from real fires with a differentiable implementation, validated by hindcast on fires observed
by NISAR since June 2026.

## Study fires

- **Bug Fire, CA/NV (primary)**: ignition 2026-08-08 22:30 UTC (WFIGS discovery) at
  39.727395, -120.0372941; growth ended ~Aug 13; ~93,733 acres; 100% contained 2026-08-27.
  NISAR GCOV same-track pre/post pairs (compare only within a track, see issue #7):
  A34 07-22 → 08-15, D42 07-23 → 08-16, A106 07-27 → 08-20 (13:09 / 02:55 / 13:01 UTC).
- Done (US, branch more-fires): Crosswhite (OR, merged lightning fires, 355,065 ac), Colwash (WA, 39,103 ac in
  < 1 day), Biscar (CA, 69,352 ac). Configs in `real_fires/src/firedata/config.py`; summary in `real_fires/docs/fires_summary.md`.
- Europe later (needs EUMETSAT / LSA SAF accounts; no GOES, FEDS or WFIGS there): Ávila (ES), Le Porge (FR).
- Pipeline (run from `real_fires/`): `scripts/build_ground_truth.py <fire>`, then `python -m firedata.report readme|summary`,
  `python -m firedata.web <fires> --app`, browser test `scripts/test_app.py`.

## Layout

```
README.md, CLAUDE.md, .gitignore, docs/architecture.png   # the only things in the repo root
real_fires/                                              # ground truth from real fires
  src/firedata/                                          # reusable functions (repeat per fire)
  scripts/                                               # build_ground_truth.py, test_app.py
  data/fires/<fire>/{raw,processed,figures}/ + README.md # raw/ is never committed
  docs/                                                  # fires_summary.md / .csv
  app/                                                   # the site
  .env (not committed), .env.example, requirements*.txt, pyproject.toml
gee_layers/                                              # model inputs from Google Earth Engine (package geelayers)
  src/geelayers/, tests/, data/ (outputs, not committed), .env (EE_PROJECT), README.md
spread_model/                                            # Alexandridis spread probabilities (package firespread)
  src/firespread/, tests/, data/ (outputs, not committed), README.md (formulas)
```

Never put new files in the repo root. Every output goes in a folder with a clear name (real_fires/app/,
real_fires/docs/, real_fires/data/fires/<fire>/, real_fires/scripts/, real_fires/src/). Temporary files go outside the repo.

## Conventions

- Python venv in `.venv`, dependencies pinned in `real_fires/requirements.txt`.
- All times in UTC.
- Secrets (Earthdata login, FIRMS MAP_KEY) live in `real_fires/.env`, never committed.
- `real_fires/data/**/raw` and `**/.env` are in `.gitignore`.
- If data is unavailable or differs from what was specified (dates, sizes, endpoints),
  report it to the user instead of guessing.

## Git workflow

- Never commit or push to `master`. Work on a feature branch.
- Commit in small steps with clear messages.
- When a task is done: push the branch and open a PR with `gh pr create`, with
  `Closes #<issue>` in the body (find the issue with `gh issue list`).
- Do NOT merge PRs; the user reviews them.
