# spread_model: fire-spread probability (Alexandridis CA)

The `firespread` package computes, for every cell next to the fire, the probability that the fire
spreads into it in the next step. It also reports the most likely spread direction. The model is the
probabilistic cellular automaton of Alexandridis et al. (2008).

Inputs come from [`gee_layers`](../gee_layers/README.md): terrain, vegetation, cover and weather on one grid,
fetched only around the fire. NISAR layers from the NISAR module plug in on the same grid when they are ready.

## Use it

```python
from firespread import predict, summary

result = predict(polygon=drawn_geojson, start="2026-08-09T21:00Z")  # or point=(lat, lon)
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
python -m firespread --inputs ../gee_layers/data/bug_bundle.nc         # reuse already-fetched inputs
```

## The model

For a burning cell *i* and each of its 8 neighbours *j*:

```
P(i -> j) = P0 * (1 + P_veg[j]) * (1 + P_den[j]) * P_wind(i -> j) * P_slope(i -> j) * P_mask[j]     (clipped to 1)
```

An unburned cell *j* catches fire in the next step with probability `1 - Π_i (1 - P(i -> j))` over its
burning neighbours. That gives `ignition_prob`.

| Factor | Formula | Values / inputs |
|---|---|---|
| **P0** | constant | 0.58: spread on flat ground, no wind, reference fuel (shrub, normal density) |
| **P_veg** | lookup by vegetation type of *j* | tree / forest **+0.4**, shrub **0**, grass / crops **-0.3**. From `fuel_lifeform` (LANDFIRE EVC); outside the US from `landcover` (WorldCover) |
| **P_den** | lookup by vegetation density of *j* | sparse (cover < 30 %) **-0.4**, normal (30-60 %) **0**, dense (≥ 60 %) **+0.3**. Cover from `nisar_cover_pct` → `cover_pct` (LANDFIRE) → NDVI (< 0.3 sparse, ≥ 0.6 dense) |
| **P_wind** | `exp(c1·V) · exp(c2·V·(cos θ − 1))` | V = wind speed (m/s) at *i*; θ = angle between the direction the wind blows **toward** and the direction *i → j*; c1 = 0.045, c2 = 0.131 s/m |
| **P_slope** | `exp(a·θs)`, `θs = atan((E_j − E_i) / l)` | E = elevation (m); l = cell size, × √2 for diagonals; θs in degrees (> 0 uphill); a = 0.078 /deg |
| **P_mask** | 1 or 0 | 0 if *j* cannot burn: burning now, `fuel_lifeform` = 0, WorldCover built-up / bare / snow / water, NISAR burn scar (`nisar_burned`), or outside the grid |

P0, c1, c2, a and the P_veg / P_den values are the published ones, calibrated on a 1990 fire in Greece.
They live in `AlexandridisParams` so they can be learned from our fires.

**Most likely direction:** for each of the 8 directions, the expected number of ignitions is summed over the
whole fire front (`expected_N`, ..., `expected_NW`). The resultant of those 8 vectors gives `main_direction_deg`.
`direction_strength` is its length divided by the total: 0 means the fire spreads evenly all round, 1 means
everything goes one way.

## NISAR (when ready)

The NISAR module should produce an `xarray.Dataset` on the **same grid** as the inputs (same `crs`, `transform`
and shape: use the geelayers `Grid` and `grid.bbox_lonlat()` to search only the area around the fire). Pass
it as `predict(..., nisar=ds)` or `--nisar file.nc`. Recognised variables:

| Variable | Meaning | Used in |
|---|---|---|
| `nisar_burned` | 1 = already burned (burn scar from L-band change detection) | P_mask = 0 |
| `nisar_cover_pct` | vegetation cover % (e.g. from HV backscatter / biomass) | P_den, before LANDFIRE |

Adding more (e.g. a NISAR fuel-moisture term) means adding one factor function in
[alexandridis.py](src/firespread/alexandridis.py) and a test.

## First live check (Bug Fire, 2026-11-14)

This uses the published parameters, one CA step, and 100 m cells. The predicted direction is compared with
where the next FEDS perimeter actually grew:

| Start (perimeter) | Predicted direction | Strength | Actual growth toward |
|---|---|---|---|
| ignition point, 08-08 22:30 | E (99°) | 0.59 | (no next perimeter yet) |
| 08-09 20:30 perimeter | SE (141°) | 0.28 | 74° (ENE) |
| 08-10 20:30 perimeter | NE (31°) | 0.22 | 354° (N) |
| 08-11 09:30 perimeter | SW (227°) | 0.05 (no clear direction) | 18° (NNE) |

So it points in roughly the right half-plane on 2 of 3 perimeters, but it is 37-67° off. That is expected
before calibration:
- The parameters come from Greece.
- One step covers 100 m, while the perimeters are 12 h apart.
- Weather is a single hour.
- P > 1 gets clipped often (22-78 spreads per run), which flattens the strongest directions.

Next steps:
- learn the parameters from the four real fires (differentiable version);
- run many CA steps with hourly weather and compare whole perimeters.
