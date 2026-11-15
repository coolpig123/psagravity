"""From a fire polygon (or ignition point) to spread probabilities, fetching only the data around it."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
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
    want_scars = burn_scars if burn_scars is not None else fetched
    scars = pool = None
    try:
        if fetched:
            from geelayers import get_fire_inputs   # only needed when fetching
            fetch_kwargs.setdefault("layers", MODEL_LAYERS)
            grid = _fire_grid(polygon, point, start, radius_km, res_m, fetch_kwargs) if want_scars else None
            if grid is not None:
                # The burn scars are the slowest fetch (~15 s of Earth Engine computing) and need only the grid:
                # start them now, so they run while the layers, the weather, NISAR and FIRMS are fetched.
                from . import burn_scars as bs
                pool = ThreadPoolExecutor(1)
                stub = xr.Dataset(coords={"y": grid.y, "x": grid.x}, attrs={"crs": grid.crs, "res_m": grid.res})
                scars = pool.submit(bs.optical_burn_scars, stub, time=start)
                fetch_kwargs["grid"] = grid
            inputs = get_fire_inputs(polygon=polygon, point=point, start=start, hours=hours,
                                     radius_km=radius_km, res_m=res_m, **fetch_kwargs)
        if want_scars and scars is None:
            from .burn_scars import optical_burn_scars
            inputs = optical_burn_scars(inputs, time=start or inputs.attrs.get("time_utc"))
        inputs = attach_nisar(inputs, nisar)
        t = start or inputs.attrs.get("time_utc")
        if (active_front if active_front is not None else fetched) and t is not None and "active_front" not in inputs:
            inputs = attach_active_front(inputs, t, lookback_h=lookback_h, notes=notes)
        if scars is not None:                       # same grid: copy the s2_ layers and attrs over
            done = scars.result()
            inputs = inputs.copy()
            for name in done.data_vars:
                inputs[name] = (done[name].dims, done[name].values, done[name].attrs)
            inputs.attrs.update({k: v for k, v in done.attrs.items() if k.startswith("s2_")})
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)   # an error elsewhere does not wait for the scars
    return spread_probabilities(inputs, time=time or start, params=params)


def _fire_grid(polygon, point, start, radius_km: float, res_m: float, fetch_kwargs: dict):
    """The grid get_fire_inputs would build for this fire (it is then passed to it, so both use the same one).
    None when the caller gave its own grid / bbox, or there is nothing to build it from."""
    if "grid" in fetch_kwargs or fetch_kwargs.get("bbox") is not None or start is None:
        return None
    from geelayers import Grid
    from geelayers.perimeter import resolve_perimeter
    if polygon is not None and point is None:
        t = pd.Timestamp(start)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        outline, _ = resolve_perimeter(polygon, t, fetch_kwargs.get("perimeter_time_key"))
        return Grid.from_polygon(outline, buffer_km=radius_km, res_m=res_m)
    if point is not None and polygon is None:
        return Grid.from_point(point[0], point[1], radius_km=radius_km, res_m=res_m)
    return None
