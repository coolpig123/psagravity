"""From a fire polygon (or ignition point) to spread probabilities, fetching only the data around it."""
from __future__ import annotations

import xarray as xr

from .alexandridis import AlexandridisParams, spread_probabilities
from .nisar import NisarInput, align_nisar, attach_nisar  # noqa: F401  (align_nisar re-exported)

# Earth Engine layers the model uses: terrain (P_s), vegetation_cover + landcover (P_veg, P_mask),
# weather (P_w from wind, P_m's EMC from temp_c + rh_pct). Not fetched: "vegetation" (Sentinel-2) and
# "fuel_dryness" (gridMET), since density and ground dryness come from NISAR.
MODEL_LAYERS = ("terrain", "landcover", "vegetation_cover", "weather")


def predict(polygon=None, point=None, start=None, *, inputs: xr.Dataset | None = None,
            nisar: NisarInput = None, time=None, hours: float = 6.0, radius_km: float = 5.0,
            res_m: float = 100.0, params: AlexandridisParams | None = None, **fetch_kwargs) -> xr.Dataset:
    """Spread probabilities around a fire.

    Either pass `inputs` (a geelayers Dataset you already have), or the fire (`polygon` drawn by the
    user, or `point=(lat, lon)`) and `start`, and the MODEL_LAYERS are fetched from Earth Engine for
    the fire + `radius_km` only.
    `nisar`: a Dataset, a NetCDF path, or a provider function (NisarRequest -> Dataset | None); see
    nisar.py. Without it, the NISAR terms (P_den, P_dry, burn scars) are neutral.
    """
    if inputs is None:
        from geelayers import get_fire_inputs   # only needed when fetching
        fetch_kwargs.setdefault("layers", MODEL_LAYERS)
        inputs = get_fire_inputs(polygon=polygon, point=point, start=start, hours=hours,
                                 radius_km=radius_km, res_m=res_m, **fetch_kwargs)
    inputs = attach_nisar(inputs, nisar)
    return spread_probabilities(inputs, time=time or start, params=params)
