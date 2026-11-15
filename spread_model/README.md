# spread_model: fire-spread probability (Alexandridis CA + fuel moisture)

The `firespread` package computes, for every cell next to the fire, the probability that the fire
spreads into it in the next step. It also reports the most likely spread direction. The model is the
probabilistic cellular automaton of Alexandridis et al. (2008), plus a fuel-moisture term.

Inputs come from two places:
- **Earth Engine** ([`gee_layers`](../gee_layers/README.md)): terrain, vegetation type, land cover, and hourly
  wind, temperature and humidity. They are fetched only around the fire.
- **NISAR** (`real_fires/src/firedata/nisar_provider.py`): vegetation density, soil moisture, vegetation water
  content and burn scars, read from NASA's GCOV and SME2 products over the area around the fire only.

Without NISAR data (no pass, or `nisar` not given), its terms are neutral (factor 1).

## Use it

```python
from firespread import predict, summary

result = predict(polygon=drawn_geojson, start="2026-08-09T21:00Z")  # or point=(lat, lon)
result = predict(polygon=drawn_geojson, start=..., nisar=get_nisar)  # + NISAR: Dataset, NetCDF path or function
print(summary(result))
result["ignition_prob"]     # (y, x): chance each cell catches fire next step  -> map layer for the app
result["p_spread"]          # (direction, y, x): P(burning cell -> neighbour in that direction)
result.attrs["main_direction"], result.attrs["main_direction_deg"], result.attrs["direction_strength"]
```

```bash
cd spread_model
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt -e .       # also installs ../gee_layers (Earth Engine setup: see its README)
pytest
python -m firespread --polygon fire.geojson --start 2026-08-09T21:00Z --out data/spread.nc
python -m firespread --inputs ../gee_layers/data/bug_bundle.nc --nisar nisar.nc   # reuse fetched inputs
```

`predict()` fetches only the Earth Engine layers the model uses (`MODEL_LAYERS`): terrain, landcover,
vegetation_cover and weather (wind, temp_c, rh_pct). It skips Sentinel-2 `vegetation` and gridMET `fuel_dryness`.
It also adds Sentinel-2 burn scars (below); turn them off with `burn_scars=False` / `--no-burn-scars`.

### Several steps: burn probability (Monte Carlo)

`spread_probabilities()` is one CA step: only the cells touching the fire get a probability.
[simulate.py](src/firespread/simulate.py) runs the automaton forward the way Alexandridis does:

- each step, every burning cell tries to ignite each neighbour once, with its `p_spread`;
- then the cell is burned out, and the newly lit cells burn in the next step.

It repeats this in 200 random runs:

```python
from firespread import predict
from firespread.simulate import simulate_spread
sim = simulate_spread(predict(polygon=..., start=..., weather_source="meteomatics", hours=12), steps=30)
sim["burn_prob"]      # (y, x): share of runs in which the cell burned within 30 steps
sim["arrival_step"]   # (y, x): mean step at which it caught fire
sim["arrival_hours"]  # (y, x): mean hours from start at which it caught fire
```

One step moves the fire at most one cell, so 30 steps at 100 m reach at most 3 km. When multi-hour weather forecasts are fetched (e.g. `weather_source="meteomatics"` or `"gfs"` with `hours=12`), `spread_probabilities()` computes `p_spread_time` `(time, direction, y, x)` across every forecast hour, and `simulate_spread()` interpolates those time-developing spread probabilities as the simulation advances in time (`time_developing=True`).

### When: arrival time per cell (survival model)

The CA's steps have no unit of time, and many cells near a fire never burn. [survival.py](src/firespread/survival.py)
learns hours from the real fires with a **cure-fraction mixture anchored on the physics**:

```
P(T <= h) = pi(x) * Phi((ln h - f(x)) / sigma),    x = fastest-path cost (CA steps)
logit pi(x) = c0 + c1 ln x  [+ XGBoost trees]       chance the fire ever reaches the cell
f(x)        = t0 + t1 ln x  [+ XGBoost AFT trees]   ln median arrival if it does (interval-censored, burned cells)
```

then calibrated (isotonic regression) on predictions for fires the model did not train on. The arrival times shown
(median, 25-75% range) are read off those same calibrated chances, so times and chances always agree.

```python
from firespread.survival import predict_arrival
res = predict_arrival(predict(polygon=..., start=...))   # loads src/firespread/models/arrival_aft.meta.json
res["p_ever"], res["p_arrive_6h"], ..., res["p_arrive_48h"], res["arrival_h"], res["arrival_h_q25"], res["arrival_h_q75"]
```

**The fastest path.** Dijkstra over the 8-neighbour grid from the burning cells; crossing from i to j costs
1 / rate(i -> j) CA steps, where the rate is the Alexandridis product P0 (1 + P_veg) P_emc P_w P_s D P_mask, **not
clipped to 1** (clipping would make strong and very strong wind equally fast and give 45-degree fronts a 1.41x
shortcut). So:
- a backing or flanking fire is slow, not extinguished (no die-off as in a one-step-burnout Monte Carlo run);
- water, rock, built-up and **burn scars** (NISAR, Sentinel-2) are walls the path goes around;
- the source is the fastest burning cell, not the nearest one.

The continuous NISAR terms (P_den, P_dry, P_vwc) are left out of the rate because training has no NISAR data, so
the website sees the same features as training. Along the path: cost, length, climb, descent, mean wind along it,
mean P_veg (all with physical sign constraints), for the trees; and `barrier_dist_m`, the shortest way round the
barriers in metres (the start-hour wind does not shape it), which the anchor line may use instead of or next to
the path cost.

**Labels**, one interval per cell, from `real_fires/data/fires/<fire>/processed/`:

| Cell | Lower bound | Upper bound |
|---|---|---|
| detected by VIIRS (within 300 m of a 375 m detection), inside the WFIGS final perimeter | the VIIRS pass before the upper bound | first detection, or the FEDS perimeter that first contains it if earlier |
| inside the final perimeter, never timed by VIIRS | the start | end of growth (99% of detections) |
| outside the final perimeter | - | never burns (the classifier's 0 class) |

FEDS gives only upper bounds: its area is a lower bound of the real one (38% for Bug). Cells no path reaches are
"never" on the website, so they are left out of fitting and calibration (they stay in the scores).

**Samples:** 2-6 start times per fire, at VIIRS passes while 3-85% of the final area is burned. Inputs from Earth
Engine (terrain, land cover, LANDFIRE, RTMA weather at the start hour), cached in `data/survival/`.

```bash
python -m firespread.survival train      # downloads what is missing (~5 min the first time), trains, saves
python -m firespread.survival evaluate   # prints the report saved with the model
```

**Choices, all leave-one-fire-out on the 3 training fires (Crosswhite plays no part):**

| Choice | Options | Chosen | Why |
|---|---|---|---|
| anchor terms | path cost, distance round barriers, both | **path cost** | mean held-out AUC 0.836 vs 0.830 and 0.829 (distance wins on Bug and Biscar, loses on Colwash) |
| tree rounds on top of the anchor lines | 0, 100, 300 | **0** (physics curve only) | mean held-out AUC 0.836 vs 0.832 and 0.825: trees learned fire-specific corrections |
| time likelihood | plain, right-truncated at the end of growth | **plain** | held-out timing error 1.21 vs 2.70 (ln units) |
| sigma | maximum likelihood with the anchor line | **1.03** | the calibration sets the probability scale |

The right-truncated likelihood is the textbook fix for the survivor bias of the labels (a cell only counts as
burned if it burned before its fire stopped growing). Here it made sigma wide (1.6) and the fast fires' times 3-40x
too slow; it is kept as an option (`TRUNCATE`) for when there are more fires.

**Results** (2026-11-15; each training fire scored by a model fitted and calibrated on the other two; Crosswhite
never used). AUC of P(T <= h), model / straight-line distance, and Brier score at 24 h:

| Fire | 6 h | 12 h | 24 h | 48 h | Brier 24 h (raw) | arrival error as shown (uncalibrated) |
|---|---|---|---|---|---|---|
| Bug | 0.90 / **0.98** | 0.88 / **0.97** | 0.82 / **0.91** | 0.79 / **0.87** | 0.027 (0.062) | 1.8x (4.6x) |
| Colwash | 0.84 / **0.90** | **0.76** / 0.68 | **0.75** / 0.68 | **0.76** / 0.70 | 0.007 (0.017) | 12.7x (4.7x) |
| Biscar | 0.96 / **0.98** | 0.92 / **0.94** | 0.86 / **0.88** | 0.79 / 0.79 | 0.027 (0.036) | 2.8x (1.7x) |
| Crosswhite | 0.98 / 0.98 | 0.98 / 0.98 | **0.97** / 0.96 | **0.95** / 0.93 | 0.009 (0.047) | 3.4x (12.5x, too early) |

The model's ranking is exactly the fastest-path cost's (it is a monotone function of it), and far better than the
Monte Carlo CA (0.61-0.86). Arrival error = typical factor between the median shown on the map (read off the
calibrated chances, see arrival_quantiles) and the observed interval; in brackets the uncalibrated time part. For cells
that burned.

What this says:
- The physics does the work: the path cost ranks cells better than any tree model on top of it, and better than
  straight-line distance on Colwash and Crosswhite, but not on Bug or Biscar.
- Calibration is what makes the chances usable (Brier 1.3-5x lower than raw). It is fitted on two fires at a time,
  so a fire unlike them is still off: on Bug, cells given 80-100% within 24 h burned 56% of the time; on Biscar,
  cells given 20-40% burned 53%.
- Fire speed differs ~10x between these fires (Colwash: 39k acres in under a day; Crosswhite: lightning fires that
  grew for weeks), and nothing in a drawn polygon says which kind of fire it is. The calibration, learned on the
  other two fires, moves the shown times toward them: better for Bug and Crosswhite, worse for Colwash and Biscar.
  Beyond 6 h and 48 h the arrival curve is extended with the steepest of its end segment, its overall 6-48 h slope
  and half the model's own slope, so a flat calibration step cannot give "never" or months. Fire speed is the main
  limit for times: the shown medians are off by 1.8x (Bug) to 12.7x (Colwash), so the website shows the 25-75%
  range with every median.

Not addressed yet (data limits, not code):
- **Fire speed.** The growth observed so far (two perimeters) would tell a fast fire from a slow one; the website
  takes one polygon.
- **Weather is the start hour only.** Live predictions would need GFS forecasts, training RTMA hours along the path.
- **No NISAR in training** (its continuous terms are kept out of the path rate instead).
- **Three training fires.** More fires (each new fire in `real_fires` is picked up by `TRAIN_FIRES`) and GOES 5-min
  timing for the labels are the most useful next steps.
- The expected areas sum chances over the whole grid; far cells keep a small chance of ever burning (about 0.15 on
  the Bug Fire test), so a larger `radius_km` adds a little.

### Web app

[web.py](src/firespread/web.py) turns a request from the map UI (`{polygon, start, radius_km, res_m, use_nisar,
burn_scars, steps = 30}`) into a prediction and a JSON answer. The map gets the cells with `burn_prob > 0.01`
(from the simulation above) as GeoJSON squares, plus a summary. [backend/server.py](../backend/server.py) serves it as `POST /api/predict`; how to start the website and
the server is in [frontend/README.md](../frontend/README.md).

### Burn scars from Sentinel-2 (dNBR)

[burn_scars.py](src/firespread/burn_scars.py) marks cells that already burned, so P_mask = 0 there:

```
NBR  = (B8A - B12) / (B8A + B12)                      near-infrared vs shortwave-infrared, 20 m
dNBR = NBR_reference - NBR_current                     burned if dNBR > 0.15
  reference = median of clear pixels, same season a year earlier (t - 365 d ± 30 d)
  current   = newest cloud- and shadow-free pixel in the 30 days before t
```

Using last year's same season keeps the vegetation's seasonal cycle out of the difference. It also means
we don't need to know when a fire happened. The result is `s2_burned` (1 scar, 0 not, -1 no clear pixel)
and `s2_dnbr`, computed at 20 m and averaged to the model cells.

Bug Fire check against the official perimeter (100 m cells):

| As of | Recall | Precision | False alarms outside |
|---|---|---|---|
| 08-16 00Z, before any clear post-fire image (smoke) | 34% | 92% | 1.2% |
| 08-17, first clear image after the fire | 78% | 97% | 1.1% |
| 08-31 | 70% | 96% | 1.1% |
| *NISAR HV change, 08-16 (for comparison)* | *41%* | *90%* | *2.0%* |

Optical can't see through smoke, so while a fire is burning the radar `nisar_burned` (which still applies) is
the only scar source. Recall is measured against the outer perimeter, which also contains unburned islands.
It is called only when `predict()` fetches the inputs, or with `burn_scars=True`.

## The model

For a burning cell *i* and each of its 8 neighbours *j*, at time *t*:

```
P_burn(i -> j, t) = clip( P0 · (1 + P_veg[j]) · (1 + P_den[j]) · P_w(i -> j, t) · P_s(i -> j) · P_m(j, t) · P_mask[j] · D(i -> j), 0, 1 )
```

An unburned cell *j* catches fire in the next step with probability `1 - Π_i (1 - P_burn(i -> j))` over its
burning neighbours, which gives `ignition_prob`.

| Factor | Formula | Values / inputs | Source |
|---|---|---|---|
| **P0** | constant | 0.58: spread on flat ground, no wind, reference fuel | paper |
| **P_veg** | lookup by vegetation type of *j* | forest **+0.4**, shrub **0**, grass / crops **−0.3** | GEE: LANDFIRE `fuel_lifeform` (WorldCover outside the US) |
| **P_den** | lookup by vegetation cover % of *j* | sparse (< 30 %) **−0.4**, normal **0**, dense (≥ 60 %) **+0.3**; no data → **0** | **NISAR only**: `nisar_cover_pct` |
| **P_w** | `exp(c1·V) · exp(c2·V·(cos θ − 1))` | V = wind speed (m/s) at *i*; θ = angle between the direction the wind blows **toward** and *i → j*; c1 = 0.045, c2 = 0.131 s/m | GEE: hourly wind |
| **P_s** | `exp(a·θs)`, `θs = atan((E_j − E_i) / l)` | E = elevation (m); l = cell size, × √2 for diagonals; θs in degrees (> 0 uphill); a = 0.078 /deg | GEE: NASADEM |
| **P_m** | `P_emc · (1 + P_dry) · (1 + P_vwc)` | fuel moisture, see below | GEE hourly + NISAR |
| **P_mask** | 1 or 0 | 0 if *j* cannot burn: burning now, no fuel, WorldCover built-up / bare / snow / water, NISAR or Sentinel-2 burn scar, or outside the grid | GEE + NISAR `nisar_burned` + Sentinel-2 `s2_burned` |
| **D** | 1 or 1/√2 | 1 for N/E/S/W, **1/√2 for diagonals** (they are √2 cells away; equal expected advance per step) | ours |

### P_m: fuel moisture (fast air moisture × slow ground and vegetation moisture)

```
P_m(j, t) = P_emc(j, t) · (1 + P_dry[j]) · (1 + P_vwc[j])
```

**P_emc** is how dry the fine dead fuel is right now. It comes from the hourly air temperature and humidity
at *j* (`temp_c`, `rh_pct`, from Earth Engine) through the Simard (1968) equilibrium moisture content, as used in
the US National Fire Danger Rating System. With `T_f = temp_c · 1.8 + 32` and `H = clip(rh_pct, 0, 100)`:

| Humidity | EMC (%) |
|---|---|
| H < 10 | `0.03229 + 0.281073·H − 0.000578·H·T_f` |
| 10 ≤ H < 50 | `2.22749 + 0.160107·H − 0.014784·T_f` |
| H ≥ 50 | `21.0606 + 0.005565·H² − 0.00035·H·T_f − 0.483199·H` |

```
P_emc = exp( −b_emc · (EMC − emc_ref) )        b_emc = 0.05 per %, emc_ref = 8 %
```

Hot, dry air (EMC < 8 %) gives P_emc > 1, so the fire spreads faster. Cool, humid air (EMC > 8 %) gives
P_emc < 1. Missing weather, or a cell with no weather, gives 1.

**P_dry** is how dry the ground is. It comes from NISAR soil moisture (`nisar_soil_moisture`, SME2): dry
(< 0.10 m³/m³) gives **+0.3**, normal **0**, wet (≥ 0.25) **−0.3**. Without NISAR it is **0**.

**P_vwc** is how much water the live vegetation holds. It comes from NISAR vegetation water content
(`nisar_vwc_kg_m2`, SME2): low (< 0.5 kg/m², e.g. cured grass or dry shrub) gives **+0.2**, normal **0**, high
(≥ 4 kg/m², well-watered live canopy) **−0.2**. Without NISAR it is **0**. VWC also grows with the amount of
vegetation, so it partly overlaps P_veg and P_den. That's why its effect is kept small until it is calibrated.

The output keeps every part: `emc_pct`, `p_emc`, `p_dry`, `p_vwc` and `p_m`. `attrs["source_p_m"]` and the
summary say which sources were used.

### Parameters

- **From the paper** (fitted on a 1990 fire in Greece): P0, c1, c2, a, P_veg and P_den.
- **Our additions, placeholders to fit later:** P_m (`b_emc`, `emc_ref_pct`, the P_dry and P_vwc thresholds and
  values) and D.

All of them are in `AlexandridisParams`, so they can be learned from our fires. `diagonal_factor=1.0` gives back
the paper's original model.

**Most likely direction:** for each of the 8 directions, the expected number of ignitions is summed over the
whole fire front (`expected_N`, ..., `expected_NW`). The resultant of those 8 vectors gives `main_direction_deg`.
`direction_strength` is 0 when the fire spreads evenly all round and 1 when everything goes one way. It uses the
clipped probabilities, because a cell can only catch fire once. The summary reports how many spreads were clipped.

**Weather time:** the model uses the last weather step at or before the requested time. If there is none, it uses
the first step. If the weather is more than 2 h older than the requested time, it still runs, but `weather_note`
and the summary warn about it.

## NISAR interface (for the NISAR module)

Everything the two modules share is in [src/firespread/nisar.py](src/firespread/nisar.py). The NISAR module can
plug in either way:

1. **A Dataset or NetCDF file:** `predict(..., nisar=ds)` or `--nisar file.nc`.
2. **A function** (recommended): `predict(..., nisar=get_nisar)`. It is called with a `NisarRequest` that holds
   exactly what to deliver:
   - `crs`, and `x`, `y`: the model grid (cell centres, metres);
   - `res_m`: the cell size;
   - `bbox_lonlat`: the area to search ASF for (only around the fire);
   - `time_utc`: the prediction start, so the NISAR module can pick the latest pass before it.

   The function returns a Dataset, or `None` when it has no data for that place or time.

```python
from firespread import NisarRequest, nisar_template

def get_nisar(request: NisarRequest):
    ds = nisar_template(request)                 # empty Dataset on the right grid, crs set
    ds["nisar_soil_moisture"] = (("y", "x"), soil_moisture_on_grid)   # shape (len(request.y), len(request.x))
    ds["nisar_cover_pct"] = (("y", "x"), cover_on_grid)
    ds["nisar_burned"] = (("y", "x"), burned_on_grid.astype("int16"))
    return ds
```

| Variable | Units | Meaning | Used in |
|---|---|---|---|
| `nisar_cover_pct` | % | vegetation / canopy cover from HV backscatter or biomass | P_den |
| `nisar_soil_moisture` | m³/m³ | surface soil moisture (SME2) | P_dry, inside P_m |
| `nisar_vwc_kg_m2` | kg/m² | vegetation water content (SME2, PMI retrieval) | P_vwc, inside P_m |
| `nisar_burned` | 0/1 (integer) | already burned, from L-band change detection | P_mask = 0 |

**The provider that exists today** is `firedata.nisar_provider.nisar_provider()` (in `real_fires/`):

```python
from firedata.nisar_provider import nisar_provider
result = predict(polygon=drawn_geojson, start="2026-08-09T21:00Z", nisar=nisar_provider())
```

It needs Earthdata credentials in `real_fires/.env`. What it does is described at the top of
[nisar_provider.py](../real_fires/src/firedata/nisar_provider.py).

Every variable is optional. Use NaN (float) or −1 (integer) for no data.

The data is aligned by `align_nisar` **by coordinates, never by array position**:
- On the request grid, values are used as they are.
- Same CRS but another resolution, extent or axis order gets regridded: continuous variables bilinearly, integer
  variables by nearest cell.
- A different or missing CRS, or an unknown variable name, raises an error.

For NISAR data much finer than the model grid (10-30 m vs 100 m), average it to about the model resolution first.

**Changing the NISAR contract** (adding a variable, say) means editing `NISAR_VARIABLES` in `nisar.py`, plus one
factor function in [alexandridis.py](src/firespread/alexandridis.py) and its test.

## Live check (Bug Fire, 2026-11-14)

This uses the published parameters, the diagonal correction, one CA step, 100 m cells and no NISAR yet. The
predicted direction is compared with where the next FEDS perimeter actually grew:

| Start (perimeter) | Predicted direction | Strength | Actual growth toward |
|---|---|---|---|
| ignition point, 08-08 22:30 | E (99°) | 0.57 | (no next perimeter yet) |
| 08-09 20:30 perimeter | SE (139°) | 0.27 | 74° (ENE) |
| 08-10 20:30 perimeter | NE (30°) | 0.22 | 354° (N) |
| 08-11 09:30 perimeter | SW (230°) | 0.05 (no clear direction) | 18° (NNE) |

The moisture term behaves as expected on the same fire. On 08-09 at 21:00 UTC (afternoon) EMC was 2.4-3.5 % and
P_m 1.25-1.32. Before dawn on 08-10 (09:00 UTC) EMC was 4.2-7.7 % and P_m 1.02-1.21. That is the daily cycle that
makes afternoons the main burning period.

The direction is 36-65° off, in roughly the right half-plane on 2 of 3 perimeters. That is expected before
calibration:
- The parameters come from Greece.
- One step covers 100 m, while the perimeters are 12 h apart.
- There is no NISAR density or soil moisture yet.

Next steps:
- learn the parameters from the four real fires;
- run many CA steps with hourly weather and compare whole perimeters;
- connect the NISAR provider.
