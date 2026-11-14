# spread_model: fire-spread probability (Alexandridis CA)

The `firespread` package computes, for every cell next to the fire, the probability that the fire
spreads into it in the next step. It also reports the most likely spread direction. The model is the
probabilistic cellular automaton of Alexandridis et al. (2008), plus a NISAR dryness term.

Inputs come from two places:
- **Earth Engine** ([`gee_layers`](../gee_layers/README.md)): terrain, vegetation type, land cover and wind.
  They are fetched only around the fire.
- **NISAR** (the NISAR module, in progress): vegetation density, dryness and burn scars.

Until the NISAR data arrives, its terms are neutral (factor 1).

## Use it

```python
from firespread import predict, summary

result = predict(polygon=drawn_geojson, start="2026-08-09T21:00Z")  # or point=(lat, lon)
result = predict(polygon=drawn_geojson, start=..., nisar=nisar_ds)  # once the NISAR module delivers
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
vegetation_cover and weather. It skips Sentinel-2 `vegetation` and gridMET `fuel_dryness`, because density
and dryness come from NISAR.

## The model

For a burning cell *i* and each of its 8 neighbours *j*:

```
P(i -> j) = P0 * (1 + P_veg[j]) * (1 + P_den[j]) * (1 + P_dry[j]) * P_wind(i -> j) * P_slope(i -> j) * P_mask[j] * D(i -> j)
```

The result is clipped to 1. An unburned cell *j* catches fire in the next step with probability
`1 - Π_i (1 - P(i -> j))` over its burning neighbours, which gives `ignition_prob`.

| Factor | Formula | Values / inputs | Source |
|---|---|---|---|
| **P0** | constant | 0.58: spread on flat ground, no wind, reference fuel | paper |
| **P_veg** | lookup by vegetation type of *j* | forest **+0.4**, shrub **0**, grass / crops **−0.3** | GEE: LANDFIRE `fuel_lifeform` (WorldCover outside the US) |
| **P_den** | lookup by vegetation cover % of *j* | sparse (< 30 %) **−0.4**, normal **0**, dense (≥ 60 %) **+0.3**; no data → **0** | **NISAR only**: `nisar_cover_pct` |
| **P_dry** | lookup by soil moisture of *j* | dry (< 0.10 m³/m³) **+0.3**, normal **0**, wet (≥ 0.25) **−0.3**; no data → **0** | **NISAR only**: `nisar_soil_moisture` (SME2) |
| **P_wind** | `exp(c1·V) · exp(c2·V·(cos θ − 1))` | V = wind speed (m/s) at *i*; θ = angle between the direction the wind blows **toward** and *i → j*; c1 = 0.045, c2 = 0.131 s/m | GEE: RTMA / ERA5-Land / GFS |
| **P_slope** | `exp(a·θs)`, `θs = atan((E_j − E_i) / l)` | E = elevation (m); l = cell size, × √2 for diagonals; θs in degrees (> 0 uphill); a = 0.078 /deg | GEE: NASADEM |
| **P_mask** | 1 or 0 | 0 if *j* cannot burn: burning now, no fuel, WorldCover built-up / bare / snow / water, NISAR burn scar, or outside the grid | GEE + NISAR `nisar_burned` |
| **D** | 1 or 1/√2 | 1 for N/E/S/W, **1/√2 for diagonals** | ours |

P0, c1, c2, a, P_veg and P_den are the published values (fitted on a 1990 fire in Greece). P_dry is not in the
paper: its thresholds and ±0.3 are placeholders. All of them sit in `AlexandridisParams` so they can be learned
from our fires.

**Why D:** a diagonal neighbour is √2 cells away. With the same probability as a side neighbour, a fire on flat,
calm ground advances √2 times faster along the diagonals. Dividing by √2 gives every direction the same expected
advance per step. `AlexandridisParams(diagonal_factor=1.0)` restores the paper's model.

**Most likely direction:** for each of the 8 directions, the expected number of ignitions is summed over the whole
fire front (`expected_N`, ..., `expected_NW`). The resultant of those 8 vectors gives `main_direction_deg`.
`direction_strength` is its length divided by the total: 0 means the fire spreads evenly all round, 1 means
everything goes one way. It uses the clipped probabilities, because that is what the automaton does: a cell can
only catch fire once. The summary reports how many spreads were clipped.

**Weather time:** the model uses the last weather step at or before the requested time. If there is none, it uses
the first step. If the weather is more than 2 h older than the requested time, it still runs, but `weather_note`
and the summary warn about it.

## NISAR interface (for the NISAR module)

Hand over an `xarray.Dataset` with dims `(y, x)`, `attrs["crs"]`, and `x` / `y` coordinates (cell centres, metres)
in that CRS. Easiest is to build it on the geelayers `Grid`: same `crs`, `x`, `y`. Use `grid.bbox_lonlat()` to
search ASF for that area only. Every variable is optional.

| Variable | Units | Meaning | Used in |
|---|---|---|---|
| `nisar_cover_pct` | % | vegetation / canopy cover from HV backscatter or biomass | P_den |
| `nisar_soil_moisture` | m³/m³ | surface soil moisture (SME2) | P_dry |
| `nisar_burned` | 0/1 (integer) | already burned, from L-band change detection | P_mask = 0 |

`predict(..., nisar=ds)` aligns it with `align_nisar`, by **coordinates, never by array position**:
- On the same grid, values are used as they are.
- A flipped y axis, another resolution or another extent in the same CRS gets regridded: continuous variables
  bilinearly, integer / 0-1 variables by nearest cell. Cells NISAR doesn't cover get no NISAR effect.
- A different or missing CRS, or an unknown variable name, raises an error instead of guessing.

For NISAR data much finer than the model grid (10-30 m vs 100 m), average it to about the model resolution first.

## Live check (Bug Fire, 2026-11-14)

This uses the published parameters, the diagonal correction, one CA step, 100 m cells and no NISAR yet. The
predicted direction is compared with where the next FEDS perimeter actually grew:

| Start (perimeter) | Predicted direction | Strength | Actual growth toward |
|---|---|---|---|
| ignition point, 08-08 22:30 | E (99°) | 0.57 | (no next perimeter yet) |
| 08-09 20:30 perimeter | SE (139°) | 0.27 | 74° (ENE) |
| 08-10 20:30 perimeter | NE (30°) | 0.22 | 354° (N) |
| 08-11 09:30 perimeter | SW (230°) | 0.05 (no clear direction) | 18° (NNE) |

It is 36-65° off, in roughly the right half-plane on 2 of 3 perimeters. That is expected before calibration:
- The parameters come from Greece.
- One step covers 100 m, while the perimeters are 12 h apart.
- Weather is a single hour.
- There is no density or dryness information yet.

Clipping is not the cause: only 4-5 % of spreads hit P = 1.

Next steps:
- learn the parameters from the four real fires;
- run many CA steps with hourly weather and compare whole perimeters;
- add the NISAR layers.
