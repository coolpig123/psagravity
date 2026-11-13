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
  NISAR GCOV pre: 2026-07-23 02:55 UTC, 2026-07-27 13:01 UTC;
  post: 2026-08-15 13:09 UTC, 2026-08-16 02:55 UTC, 2026-08-20 13:01 UTC.
- Backups: Ávila (Spain), Le Porge (France), Rowe Creek / Crosswhite (OR). See README.

## Layout

```
data/fires/<fire>/{raw,processed,figures}/ + README.md   # raw/ is never committed
src/                                                     # reusable functions (repeat per fire)
notebooks/, app/, docs/
```

## Conventions

- Python venv in `.venv`, dependencies pinned in `requirements.txt`.
- All times in UTC.
- Secrets (Earthdata login, FIRMS MAP_KEY) live in `.env`, never committed.
- `data/**/raw` and `.env` are in `.gitignore`.
- If data is unavailable or differs from what was specified (dates, sizes, endpoints),
  report it to the user instead of guessing.

## Git workflow

- Never commit or push to `master`. Work on a feature branch.
- Commit in small steps with clear messages.
- When a task is done: push the branch and open a PR with `gh pr create`, with
  `Closes #<issue>` in the body (find the issue with `gh issue list`).
- Do NOT merge PRs; the user reviews them.
