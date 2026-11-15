"""From a fire polygon (or ignition point) to spread probabilities, fetching only the data around it."""
from __future__ import annotations

import xarray as xr

from .active_front import DEFAULT_LOOKBACK_H, attach_active_front
from .alexandridis import AlexandridisParams, spread_probabilities
from .nisar import NisarInput, align_nisar, attach_nisar  # noqa: F401  (align_nisar re-exported)

# Earth Engine layers the model uses: terrain (P_s), vegetation_cover + landcover (P_veg, P_mask),
# weather (P_w from wind, P_m's EMC from temp_c + rh_pct). Not fetched: "vegetation" (Sentinel-2) and
# "fuel_dryness" (gridMET), since density and ground dryness come from NISAR.
MODEL_LAYERS = ("terrain", "landcover", "vegetation_cover", "weather")


def predict(polygon=None, point=None, start=None, *, inputs: xr.Dataset | None = None,
            nisar: NisarInput = None, burn_scars: bool | None = None, time=None, hours: float = 6.0,
            radius_km: float = 5.0, res_m: float = 100.0, params: AlexandridisParams | None = None,
            active_front: bool | None = None, lookback_h: float = DEFAULT_LOOKBACK_H,
            notes: list[str] | None = None, **fetch_kwargs) -> xr.Dataset:
    """Spread probabilities around a fire.

    Either pass `inputs` (a geelayers Dataset you already have), or the fire (`polygon` drawn by the
    user, or `point=(lat, lon)`) and `start`, and the MODEL_LAYERS are fetched from Earth Engine for
    the fire + `radius_km` only.
    `nisar`: a Dataset, a NetCDF path, or a provider function (NisarRequest -> Dataset | None); see
    nisar.py. Without it, the NISAR terms (P_den, P_dry, burn scars) are neutral.
    `burn_scars`: add Sentinel-2 dNBR burn scars (s2_burned, burn_scars.py) from Earth Engine. Default:
    on when the inputs are fetched here, off when `inputs` is given (no Earth Engine calls then).
    `active_front`: spread only from the cells VIIRS saw burning in the `lookback_h` before `start` (NASA FIRMS,
    needs real_fires + FIRMS_MAP_KEY; active_front.py). Same default as burn_scars. Never fails: without hotspots
    the whole polygon spreads, and the reason is appended to `notes` and attrs["active_front_note"].
    """
    fetched = inputs is None
    if fetched:
        from geelayers import get_fire_inputs   # only needed when fetching
        fetch_kwargs.setdefault("layers", MODEL_LAYERS)
        inputs = get_fire_inputs(polygon=polygon, point=point, start=start, hours=hours,
                                 radius_km=radius_km, res_m=res_m, **fetch_kwargs)
    if burn_scars if burn_scars is not None else fetched:
        from .burn_scars import optical_burn_scars
        inputs = optical_burn_scars(inputs, time=start or inputs.attrs.get("time_utc"))
    inputs = attach_nisar(inputs, nisar)
    t = start or inputs.attrs.get("time_utc")
    if (active_front if active_front is not None else fetched) and t is not None and "active_front" not in inputs:
        inputs = attach_active_front(inputs, t, lookback_h=lookback_h, notes=notes)
    return spread_probabilities(inputs, time=time or start, params=params)
