# hindcast: does the spread model predict where real fires went?

Start the spread model from a real FEDS perimeter at time *t* and run it forward. Then compare it with the FEDS
perimeters observed about 12 h and 24 h later. It must beat a dumb baseline: the start perimeter grown evenly in
every direction to the same area.

> **Scores are vs FEDS, not the official area.** FEDS (VIIRS-modeled perimeters) itself under-maps the official
> WFIGS perimeter: Bug FEDS = 37.8 % of the official area (IoU 32.7 %), Biscar 11.2 % (IoU 10.2 %). A predicted
> cell outside FEDS is not necessarily wrong, so precision here is a lower bound.

## Method

- **Model:** `firespread.predict(inputs=..., time=t)` gives the one-step spread probabilities, then
  `firespread.simulate.simulate_spread(steps=N, runs=200)` gives `burn_prob`. The model code is used as it is.
  `spread_model/` and `gee_layers/` are not modified (see [SUGGESTIONS.md](SUGGESTIONS.md)).
- **No answer in the inputs:** the Earth Engine layers use data from before *t*. The Sentinel-2 burn scars use
  the 30 days before *t*. NISAR is left out, because the post-fire passes show the final burned area.
- **Grid:** 100 m, start-perimeter bbox + 15 km. The inputs are cached in `hindcast/data/` (not committed), so
  calibration and re-runs cost no Earth Engine calls.
- **Windows:** every FEDS step with new data (`is_new`) is a start. Its targets are the steps at about +12 h and
  +24 h (really +11 h or +13 h, from the 09:30 / 20:30 UTC passes). A step without new data repeats the previous
  perimeter, so it is `stale` and not scored.
- **Metrics:** IoU, precision and recall of the **newly burned** area (mask minus the start perimeter). Totals
  including the start perimeter are in `*_total`.
- **Baselines:**
  - `base_*`: circle growth with the model's new area. This tests shape and direction.
  - `oracle_iou`: circle growth with the observed new area. This is an upper bound for a shape-free guess.
- **Calibration:** two knobs, hours per CA step and the `burn_prob` threshold. They are fitted together, by
  grid search, on Bug's first window (08-09 09:30 → +11 h / +24 h, 9.8 → 10.8 / 37.0 km²). Then they are held
  fixed for the later Bug windows, Biscar, and a few Crosswhite windows (validation only).

## Data notes (reported, not fixed)

- **Colwash: left out of the scores.** FEDS IoU vs WFIGS is 2.9 % and there are only 2 perimeters with new data.
- **Bug:** 08-10 20:30 and every step from 08-13 20:30 repeat the previous perimeter. That leaves 7 start
  windows and 11 scored targets (`python -m hindcast windows bug_fire`).
- **Biscar:** FEDS has only 3 perimeters with new data (25.9 → 30.6 → 31.4 km²) against 280.7 km² official.
  That gives 2 scored 24 h targets with little growth, so it's a weak test.

## Run

```bash
source .venv/bin/activate                       # repo venv
pip install -e gee_layers -e spread_model -e hindcast
python -m hindcast windows bug_fire             # offline
python -m hindcast calibrate bug_fire           # Earth Engine once -> results/calibration.json
python -m hindcast evaluate bug_fire --skip-first
python -m hindcast evaluate biscar
cd hindcast && pytest                           # offline tests
```

Outputs go in `hindcast/results/`: `<fire>_calibration.csv`, `calibration.json` and `<fire>_scores.csv`.
