"""From a fire polygon (or ignition point) to spread probabilities, fetching only the data around it."""
from __future__ import annotations

import numpy as np
import xarray as xr

from .alexandridis import NISAR_VARIABLES, AlexandridisParams, spread_probabilities

# Earth Engine layers the model uses: terrain (P_slope), vegetation_cover + landcover (P_veg, P_mask),
# weather (P_wind). Not fetched: "vegetation" (Sentinel-2) and "fuel_dryness" (gridMET), since density
# and dryness come from NISAR only. Saves one Sentinel-2 composite and one gridMET request per run.
MODEL_LAYERS = ("terrain", "landcover", "vegetation_cover", "weather")


def align_nisar(nisar: xr.Dataset, inputs: xr.Dataset) -> tuple[xr.Dataset, str]:
    """Put NISAR variables on the inputs grid, matching cells by coordinates (never by array index).

    Requires nisar.attrs["crs"] equal to the inputs CRS and x / y coords (metres, cell centres) in it.
    Handles a flipped y axis, a different resolution or extent: continuous variables are interpolated
    bilinearly, 0/1 and class variables (integer dtype) take the nearest NISAR cell; cells NISAR does
    not cover become NaN (float) / -1 (integer). Coarser-than-NISAR targets are point-sampled, so for
    much finer NISAR data (e.g. 10-30 m) average to roughly the inputs resolution first.
    """
    crs_n, crs_i = nisar.attrs.get("crs"), inputs.attrs.get("crs")
    if not crs_n:
        raise ValueError("NISAR Dataset needs attrs['crs'] (e.g. 'EPSG:32610') and x / y coords in that CRS; "
                         "build it on the geelayers Grid (grid.crs, grid.x, grid.y)")
    if crs_n != crs_i:
        raise ValueError(f"NISAR CRS {crs_n} differs from the inputs CRS {crs_i}: reproject onto the inputs grid "
                         "(same crs, transform, shape) first")
    if "x" not in nisar.coords or "y" not in nisar.coords:
        raise ValueError("NISAR Dataset needs x and y coordinates (cell centres, metres)")
    unknown = [v for v in nisar.data_vars if v not in NISAR_VARIABLES]
    if unknown:
        raise ValueError(f"unknown NISAR variables {unknown}; expected some of {list(NISAR_VARIABLES)}")

    tol = 1e-6 * float(inputs.attrs.get("res_m", 1.0))
    same = (nisar.sizes["x"] == inputs.sizes["x"] and nisar.sizes["y"] == inputs.sizes["y"]
            and np.allclose(nisar.x.values, inputs.x.values, atol=tol)
            and np.allclose(nisar.y.values, inputs.y.values, atol=tol))
    out = {}
    for name, da in nisar.data_vars.items():
        if set(da.dims) != {"y", "x"}:
            raise ValueError(f"NISAR variable {name} must have dims (y, x), got {da.dims}")
        da = da.transpose("y", "x")
        if same:
            out[name] = (("y", "x"), da.values)
        elif da.dtype.kind in "iub":       # classes / 0-1 flags: nearest cell, no blending
            v = da.astype(np.float32).interp(x=inputs.x, y=inputs.y, method="nearest").values
            out[name] = (("y", "x"), np.where(np.isnan(v), -1, v).astype(da.dtype if da.dtype.kind == "i" else np.int16))
        else:
            out[name] = (("y", "x"), da.interp(x=inputs.x, y=inputs.y, method="linear").values.astype(np.float32))
    how = "same grid" if same else (f"regridded by coordinates from {nisar.sizes['x']}x{nisar.sizes['y']} cells "
                                    "(linear for continuous, nearest for classes)")
    return xr.Dataset(out, coords={"y": inputs.y, "x": inputs.x}), how


def predict(polygon=None, point=None, start=None, *, inputs: xr.Dataset | None = None,
            nisar: xr.Dataset | None = None, time=None, hours: float = 6.0, radius_km: float = 5.0,
            res_m: float = 100.0, params: AlexandridisParams | None = None, **fetch_kwargs) -> xr.Dataset:
    """Spread probabilities around a fire.

    Either pass `inputs` (a geelayers Dataset you already have), or the fire (`polygon` drawn by the
    user, or `point=(lat, lon)`) and `start`, and the MODEL_LAYERS are fetched from Earth Engine for
    the fire + `radius_km` only.
    `nisar`: optional Dataset with NISAR_VARIABLES (nisar_cover_pct, nisar_soil_moisture, nisar_burned),
    aligned onto the inputs grid by coordinates (see align_nisar). Without it, P_den = P_dry = 0.
    """
    if inputs is None:
        from geelayers import get_fire_inputs   # only needed when fetching
        fetch_kwargs.setdefault("layers", MODEL_LAYERS)
        inputs = get_fire_inputs(polygon=polygon, point=point, start=start, hours=hours,
                                 radius_km=radius_km, res_m=res_m, **fetch_kwargs)
    if nisar is not None:
        aligned, how = align_nisar(nisar, inputs)
        inputs = inputs.assign({v: aligned[v] for v in aligned.data_vars})
        inputs.attrs["nisar_alignment"] = how
    return spread_probabilities(inputs, time=time or start, params=params)
