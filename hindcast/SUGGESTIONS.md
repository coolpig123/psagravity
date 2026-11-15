# Suggestions for spread_model (for Ziv)

The hindcast uses `firespread` as it is. These changes would make multi-hour predictions more realistic. They are
not made here because `spread_model/` is Ziv's.

1. **Hourly weather in `simulate_spread`.** Today `p_spread` is computed once, from the weather at the start time,
   and held for every step. A 24 h run keeps the 09:30 UTC wind and humidity through the afternoon and the night.
   Proposal: accept a function `step -> p_spread`, or the inputs plus `hours_per_step`, and recompute
   `spread_probabilities(ds, time=t0 + step * hours_per_step)` whenever the weather hour changes. The inputs
   already carry hourly `wind_*`, `temp_c` and `rh_pct`.
2. **A time unit for a step.** A step is "one cell" with no duration. The hindcast calibrates `hours_per_step` on
   Bug (`hindcast/results/calibration.json`). Suggest adding `hours_per_step` as a parameter (default from that
   calibration) so the app can show "+12 h" instead of "+N steps".
3. **Return the per-run burned masks (optional).** `burn_prob` is an average, so a run can't be continued from
   where it stopped (e.g. chain 12 h windows, or restart when a new perimeter arrives, issue #5). An option
   `return_runs=True` returning the `(runs, y, x)` boolean array would allow both.
4. **Calibrate the paper constants on our fires.** P0 = 0.58, c1 and c2 come from a 1990 Greek fire. Once the
   hindcast scores exist, they're a ready objective for fitting `AlexandridisParams` (issue #4).
