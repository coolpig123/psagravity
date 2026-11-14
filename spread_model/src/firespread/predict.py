"""From a fire polygon (or ignition point) to spread probabilities, fetching only the data around it."""
from __future__ import annotations

import xarray as xr

from .alexandridis import AlexandridisParams, spread_probabilities


def predict(polygon=None, point=None, start=None, *, inputs: xr.Dataset | None = None,
            nisar: xr.Dataset | None = None, time=None, hours: float = 6.0, radius_km: float = 5.0,
            res_m: float = 100.0, params: AlexandridisParams | None = None, **fetch_kwargs) -> xr.Dataset:
    """Spread probabilities around a fire.

    Either pass `inputs` (a geelayers Dataset you already have), or the fire (`polygon` drawn by the
    user, or `point=(lat, lon)`) and `start`, and the inputs are fetched from Earth Engine for the
    fire + `radius_km` only.
    `nisar`: optional Dataset on the same grid with NISAR_VARIABLES (e.g. nisar_burned, nisar_cover_pct);
    its variables are added to the inputs and take priority.
    """
    if inputs is None:
        from geelayers import get_fire_inputs   # only needed when fetching
        inputs = get_fire_inputs(polygon=polygon, point=point, start=start, hours=hours,
                                 radius_km=radius_km, res_m=res_m, **fetch_kwargs)
    if nisar is not None:
        if nisar.sizes.get("y") != inputs.sizes["y"] or nisar.sizes.get("x") != inputs.sizes["x"]:
            raise ValueError("nisar layers must be on the same grid as the inputs (resample onto geelayers Grid)")
        inputs = inputs.assign({v: (nisar[v].dims, nisar[v].values) for v in nisar.data_vars})
    return spread_probabilities(inputs, time=time or start, params=params)
