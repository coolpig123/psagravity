# spread_model: fire-spread probability (Alexandridis CA + fuel moisture)

The `firespread` package computes, for every cell next to the fire, the probability that the fire
spreads into it in the next step. It also reports the most likely spread direction. The model is the
probabilistic cellular automaton of Alexandridis et al. (2008), plus a fuel-moisture term.

Inputs come from two places:
- **Earth Engine** ([`gee_layers`](../gee_layers/README.md)): terrain, vegetation type, land cover, and hourly
  wind, temperature and humidity. They are fetched only around the fire.
- **NISAR** (the NISAR module, in progress): vegetation density, soil moisture and burn scars.

Until the NISAR data arrives, its terms are neutral (factor 1).

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
| **P_m** | `P_emc · (1 + P_dry)` | fuel moisture, see below | GEE hourly + NISAR |
| **P_mask** | 1 or 0 | 0 if *j* cannot burn: burning now, no fuel, WorldCover built-up / bare / snow / water, NISAR burn scar, or outside the grid | GEE + NISAR `nisar_burned` |
| **D** | 1 or 1/√2 | 1 for N/E/S/W, **1/√2 for diagonals** (they are √2 cells away; equal expected advance per step) | ours |

### P_m: fuel moisture (fast air moisture × slow ground moisture)

```
P_m(j, t) = P_emc(j, t) · (1 + P_dry[j])
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

The output keeps every part: `emc_pct`, `p_emc`, `p_dry` and `p_m`. `attrs["source_p_m"]` and the summary say
which sources were used.

### Parameters

- **From the paper** (fitted on a 1990 fire in Greece): P0, c1, c2, a, P_veg and P_den.
- **Our additions, placeholders to fit later:** P_m (`b_emc`, `emc_ref_pct`, the P_dry thresholds and ±0.3) and D.

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
| `nisar_burned` | 0/1 (integer) | already burned, from L-band change detection | P_mask = 0 |

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
