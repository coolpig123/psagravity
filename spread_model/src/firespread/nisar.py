"""The NISAR contract: what the NISAR module hands to the spread model, and how it is aligned.

This file is the only place the spread model and the NISAR module meet. The NISAR side can plug in
in either of two ways:

1. Pass a Dataset:   predict(..., nisar=ds)          or  python -m firespread ... --nisar file.nc
2. Pass a provider:  predict(..., nisar=get_nisar)   where get_nisar(request: NisarRequest) -> xr.Dataset | None

A provider is called after the Earth Engine inputs exist, with a NisarRequest describing exactly the
area, grid and time to deliver (so only the data around the fire is searched / downloaded).
`nisar_template(request)` gives an empty Dataset on the right grid to fill in.

Dataset rules (checked by align_nisar):
- dims (y, x); attrs["crs"] (e.g. "EPSG:32610"); x / y coords = cell centres in metres in that CRS
- only variables from NISAR_VARIABLES, all optional; NaN (float) / -1 (integer) = no data
- same grid as the request -> used as is; same CRS but other resolution / extent / axis order ->
  regridded by coordinates; other CRS -> error (reproject first)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import numpy as np
import xarray as xr

# name -> (units, meaning -> where it is used)
NISAR_VARIABLES: dict[str, tuple[str, str]] = {
    "nisar_cover_pct": ("%", "vegetation / canopy cover from L-band HV backscatter or biomass -> P_den (only source)"),
    "nisar_soil_moisture": ("m3/m3", "surface soil moisture (NISAR SME2) -> P_dry, the slow part of P_m (only source)"),
    "nisar_burned": ("0/1", "1 = already burned (L-band change detection burn scar) -> P_mask = 0 (integer dtype)"),
}


@dataclass(frozen=True)
class NisarRequest:
    """What a NISAR provider must deliver: data for this grid, valid at `time_utc`."""
    crs: str                                       # e.g. "EPSG:32610" (UTM, metres)
    x: np.ndarray                                  # cell-centre x, metres, west -> east
    y: np.ndarray                                  # cell-centre y, metres, north -> south
    res_m: float                                   # cell size
    bbox_lonlat: tuple[float, float, float, float]  # west, south, east, north (degrees) to search ASF with
    time_utc: str                                  # prediction start; use the latest NISAR pass before it

    @classmethod
    def from_inputs(cls, inputs: xr.Dataset) -> "NisarRequest":
        res = float(inputs.attrs.get("res_m", abs(float(inputs.x[1] - inputs.x[0]))))
        if "lat" in inputs.coords and "lon" in inputs.coords:
            lat, lon = inputs["lat"].values, inputs["lon"].values
            pad = res / 111_000 * 2            # half a cell is enough; 2 cells of degrees to be safe
            bbox = (float(lon.min() - pad), float(lat.min() - pad), float(lon.max() + pad), float(lat.max() + pad))
        else:
            bbox = (np.nan,) * 4
        return cls(crs=inputs.attrs.get("crs", ""), x=inputs.x.values, y=inputs.y.values, res_m=res,
                   bbox_lonlat=bbox, time_utc=str(inputs.attrs.get("time_utc", "")))


NisarProvider = Callable[[NisarRequest], Union[xr.Dataset, None]]
NisarInput = Union[xr.Dataset, str, Path, NisarProvider, None]


def nisar_template(request: NisarRequest) -> xr.Dataset:
    """Empty Dataset on the request grid, with the right coords and attrs: add NISAR variables to it."""
    return xr.Dataset(coords={"y": ("y", request.y, {"units": "m"}), "x": ("x", request.x, {"units": "m"})},
                      attrs={"crs": request.crs, "res_m": request.res_m, "time_utc": request.time_utc})


def align_nisar(nisar: xr.Dataset, inputs: xr.Dataset) -> tuple[xr.Dataset, str]:
    """Put NISAR variables on the inputs grid, matching cells by coordinates (never by array index).

    Continuous variables are interpolated bilinearly, 0/1 and class variables (integer dtype) take the
    nearest NISAR cell; cells NISAR does not cover become NaN (float) / -1 (integer). Targets coarser
    than NISAR are point-sampled, so average much finer NISAR data (10-30 m) to ~the inputs resolution first.
    """
    crs_n, crs_i = nisar.attrs.get("crs"), inputs.attrs.get("crs")
    if not crs_n:
        raise ValueError("NISAR Dataset needs attrs['crs'] (e.g. 'EPSG:32610') and x / y coords in that CRS; "
                         "start from nisar_template(NisarRequest.from_inputs(inputs))")
    if crs_n != crs_i:
        raise ValueError(f"NISAR CRS {crs_n} differs from the inputs CRS {crs_i}: reproject onto the inputs grid "
                         "(same crs, x, y) first")
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


def attach_nisar(inputs: xr.Dataset, nisar: NisarInput) -> xr.Dataset:
    """Resolve `nisar` (Dataset, NetCDF path, provider function or None) and add its variables to `inputs`."""
    if nisar is None:
        return inputs
    if callable(nisar):
        nisar = nisar(NisarRequest.from_inputs(inputs))
        if nisar is None:                       # provider has nothing for this place / time
            inputs = inputs.copy()
            inputs.attrs["nisar_alignment"] = "provider returned no NISAR data"
            return inputs
    elif isinstance(nisar, (str, Path)):
        nisar = xr.load_dataset(nisar, engine="h5netcdf")
    if not isinstance(nisar, xr.Dataset):
        raise TypeError(f"nisar must be an xarray Dataset, a NetCDF path or a provider function, got {type(nisar)}")
    aligned, how = align_nisar(nisar, inputs)
    inputs = inputs.assign({v: aligned[v] for v in aligned.data_vars})
    inputs.attrs["nisar_alignment"] = how
    return inputs
