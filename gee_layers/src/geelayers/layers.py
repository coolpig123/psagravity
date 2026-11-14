"""Catalog of Earth Engine datasets, each mapped to standard variable names and units.

Two kinds:
- LAYERS: one image per request (terrain, land cover, fuels, vegetation, daily fuel dryness).
- WEATHER: hourly sources returning an ImageCollection with the same variable names whatever the
  source, so the model can switch rtma <-> era5_land <-> gfs without code changes.

To add a layer: write a builder `(region, time) -> (ee.Image with bands named as `variables`, meta dict)`
and register it in LAYERS. Every variable name must also be described in VARIABLES.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import ee
import pandas as pd

# name -> (units, description). The one place that defines the output contract.
VARIABLES: dict[str, tuple[str, str]] = {
    # terrain
    "elevation_m": ("m", "Elevation above sea level (NASADEM)"),
    "slope_deg": ("degrees", "Terrain slope at grid resolution (derived from elevation_m)"),
    "aspect_deg": ("degrees", "Direction the slope faces, clockwise from north; NaN where flat"),
    # land cover / fuels
    "landcover": ("class", "ESA WorldCover 2021 class: 10 tree, 20 shrub, 30 grass, 40 crop, 50 built, "
                           "60 bare, 70 snow/ice, 80 water, 90 wetland, 95 mangrove, 100 moss; -1 = no data"),
    "evc": ("class", "LANDFIRE Existing Vegetation Cover code (LF 1.4.0, ~2014; US only): 101-109 tree, "
                     "111-119 shrub, 121-129 herb cover in 10% steps; other codes = water, developed, crops, "
                     "barren (legend in the dataset's EVC_class_names); -1 = no data"),
    "fuel_lifeform": ("class", "Dominant vegetation from evc: 0 = no wildland fuel, 1 = tree, 2 = shrub, 3 = herb/grass, "
                               "-1 = no data"),
    "cover_pct": ("%", "Vegetation cover of that lifeform from evc (class midpoint, 15-95); NaN when not wildland"),
    # vegetation state (Sentinel-2, cloud-masked median over the 30 days before `time`)
    "ndvi": ("1", "NDVI (B8, B4): greenness / amount of live vegetation"),
    "ndmi": ("1", "NDMI (B8, B11): vegetation water content; low = dry fuel"),
    # daily fire-danger (gridMET, CONUS, latest day starting before `time`)
    "fm100_pct": ("%", "100-hour dead fuel moisture"),
    "fm1000_pct": ("%", "1000-hour dead fuel moisture"),
    "erc": ("1", "Energy Release Component (NFDRS fuel model G)"),
    "burning_index": ("1", "Burning Index (NFDRS fuel model G)"),
    "vpd_kpa": ("kPa", "Daily mean vapour pressure deficit"),
    "rh_min_pct": ("%", "Daily minimum relative humidity"),
    "tmax_c": ("degC", "Daily maximum temperature"),
    "precip_daily_mm": ("mm", "Daily precipitation"),
    # hourly weather (any WEATHER source)
    "temp_c": ("degC", "2 m air temperature"),
    "rh_pct": ("%", "2 m relative humidity"),
    "wind_u_ms": ("m/s", "10 m wind, eastward component (toward east > 0)"),
    "wind_v_ms": ("m/s", "10 m wind, northward component (toward north > 0)"),
    "wind_speed_ms": ("m/s", "10 m wind speed (derived from u, v)"),
    "wind_dir_deg": ("degrees", "10 m wind direction the wind blows FROM, clockwise from north (derived)"),
    "gust_ms": ("m/s", "Wind gust (rtma, gfs)"),
    "precip_mm": ("mm", "Precipitation in the hour (gfs: instantaneous rate x 1 h)"),
    "soil_moisture": ("m3/m3", "Volumetric soil water, 0-7 cm (era5_land only)"),
    # initial fire state
    "initial_burned": ("class", "Initial fire mask at start time: 1 = burning/burned, 0 = unburned"),
}

Builder = Callable[[ee.Geometry, "pd.Timestamp | None"], "tuple[ee.Image, dict[str, str]]"]


@dataclass(frozen=True)
class Layer:
    name: str
    description: str
    dataset: str
    coverage: str            # "global" or "CONUS"
    native_res_m: float
    needs_time: bool
    categorical: bool
    variables: tuple[str, ...]
    build: Builder


def _ms(t: pd.Timestamp) -> int:
    return int(pd.Timestamp(t).timestamp() * 1000)


def _rh(t_c: ee.Image, td_c: ee.Image) -> ee.Image:
    """Relative humidity (%) from temperature and dew point (degC), Magnus formula."""
    e = td_c.multiply(17.625).divide(td_c.add(243.04)).exp()
    es = t_c.multiply(17.625).divide(t_c.add(243.04)).exp()
    return e.divide(es).multiply(100).min(100).rename("rh_pct")


def _latest_before(col: ee.ImageCollection, t: pd.Timestamp, lookback: pd.Timedelta,
                   what: str) -> tuple[ee.Image, pd.Timestamp]:
    """Most recent image starting in [t - lookback, t), and its start time. Fails loudly if there is none."""
    col = col.filterDate(_ms(t - lookback), _ms(t)).sort("system:time_start", False)
    info = col.limit(1).aggregate_array("system:time_start").getInfo()
    if not info:
        raise LookupError(f"{what}: no image in the {lookback} before {t}. The dataset may not cover this "
                          "date yet (check its latency) or this place.")
    return ee.Image(col.first()), pd.Timestamp(info[0], unit="ms", tz="UTC")


# ---- static / snapshot layers ---------------------------------------------------------------

def _terrain(region, t):
    nasadem = ee.Image("NASA/NASADEM_HGT/001").select("elevation").resample("bilinear")
    gmted = ee.Image("USGS/GMTED2010_FULL").select("be75").resample("bilinear")
    return nasadem.unmask(gmted).rename("elevation_m"), {}


def _landcover(region, t):
    return ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover"), {}


def _vegetation_cover(region, t):
    # The public catalog has no LANDFIRE fuel-model (FBFM40) layer; EVC gives lifeform + density,
    # which is what the Alexandridis p_veg / p_den factors need. fuel_lifeform / cover_pct are decoded locally.
    img = ee.ImageCollection("LANDFIRE/Vegetation/EVC/v1_4_0").filter(ee.Filter.eq("system:index", "CONUS")).first()
    return img.select("EVC").rename("evc"), {}


_S2_KEEP = [4, 5, 6, 7, 11]   # SCL: vegetation, bare, water, unclassified, snow (drop cloud, shadow, saturated)


def _vegetation(region, t):
    start = t - pd.Timedelta(days=30)

    def prep(img):
        scl = img.select("SCL")
        ok = scl.remap(_S2_KEEP, [1] * len(_S2_KEEP), 0)
        return ee.Image.cat([img.normalizedDifference(["B8", "B4"]).rename("ndvi"),
                             img.normalizedDifference(["B8", "B11"]).rename("ndmi")]).updateMask(ok)

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
           .filterDate(_ms(start), _ms(t)).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
    n = col.size().getInfo()
    return col.map(prep).median(), {"vegetation_window": f"{start.isoformat()} / {t.isoformat()}",
                                    "vegetation_n_scenes": str(n)}


def _fuel_dryness(region, t):
    img, used = _latest_before(ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(region), t,
                               pd.Timedelta(days=7), "gridMET")
    img = img.resample("bilinear")
    out = ee.Image.cat([
        img.select("fm100").rename("fm100_pct"), img.select("fm1000").rename("fm1000_pct"),
        img.select("erc").rename("erc"), img.select("bi").rename("burning_index"),
        img.select("vpd").rename("vpd_kpa"), img.select("rmin").rename("rh_min_pct"),
        img.select("tmmx").subtract(273.15).rename("tmax_c"), img.select("pr").rename("precip_daily_mm")])
    return out, {"fuel_dryness_day_utc": used.isoformat()}


LAYERS: dict[str, Layer] = {l.name: l for l in [
    Layer("terrain", "Elevation; slope and aspect derived at grid resolution", "NASA/NASADEM_HGT/001",
          "global (60N-56S)", 30, False, False, ("elevation_m",), _terrain),
    Layer("landcover", "Land cover class", "ESA/WorldCover/v200", "global", 10, False, True,
          ("landcover",), _landcover),
    Layer("vegetation_cover", "Vegetation lifeform and cover (fuel type / density); +fuel_lifeform, cover_pct",
          "LANDFIRE/Vegetation/EVC/v1_4_0", "CONUS", 30, False, True, ("evc",), _vegetation_cover),
    Layer("vegetation", "Pre-`time` greenness and vegetation moisture (30-day median)",
          "COPERNICUS/S2_SR_HARMONIZED", "global", 20, True, False, ("ndvi", "ndmi"), _vegetation),
    Layer("fuel_dryness", "Daily fuel moisture and fire danger indices", "IDAHO_EPSCOR/GRIDMET", "CONUS", 4000,
          True, False, ("fm100_pct", "fm1000_pct", "erc", "burning_index", "vpd_kpa", "rh_min_pct", "tmax_c",
                        "precip_daily_mm"), _fuel_dryness),
]}


# ---- hourly weather sources -----------------------------------------------------------------

@dataclass(frozen=True)
class WeatherSource:
    name: str
    description: str
    dataset: str
    coverage: str
    native_res_m: float
    variables: tuple[str, ...]    # server-side bands; wind speed/dir are added locally
    # (region, start, end, issued) -> (images with bands `variables` and property 'valid_time', meta dict)
    collection: Callable[..., "tuple[ee.ImageCollection, dict[str, str]]"]


def _rtma(region, start, end, issued=None):
    def std(img):
        img = img.resample("bilinear")
        t, td = img.select("TMP"), img.select("DPT")
        return ee.Image.cat([t.rename("temp_c"), _rh(t, td), img.select("UGRD").rename("wind_u_ms"),
                             img.select("VGRD").rename("wind_v_ms"), img.select("GUST").rename("gust_ms"),
                             img.select("ACPC01").rename("precip_mm")]
                            ).set("valid_time", img.get("system:time_start"))
    return (ee.ImageCollection("NOAA/NWS/RTMA").filterBounds(region)
            .filterDate(_ms(start), _ms(end)).map(std)), {}


def _era5_land(region, start, end, issued=None):
    def std(img):
        img = img.resample("bilinear")
        t = img.select("temperature_2m").subtract(273.15)
        td = img.select("dewpoint_temperature_2m").subtract(273.15)
        return ee.Image.cat([t.rename("temp_c"), _rh(t, td),
                             img.select("u_component_of_wind_10m").rename("wind_u_ms"),
                             img.select("v_component_of_wind_10m").rename("wind_v_ms"),
                             img.select("total_precipitation_hourly").multiply(1000).max(0).rename("precip_mm"),
                             img.select("volumetric_soil_water_layer_1").rename("soil_moisture")]
                            ).set("valid_time", img.get("system:time_start"))
    return (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterBounds(region)
            .filterDate(_ms(start), _ms(end)).map(std)), {}


def _gfs(region, start, end, issued=None):
    """GFS forecast from the latest run issued at or before `issued` (default: `start`)."""
    issued = pd.Timestamp(issued) if issued is not None else start
    runs = ee.ImageCollection("NOAA/GFS0P25").filterDate(_ms(issued - pd.Timedelta(days=2)), _ms(issued) + 1)
    run_ms = runs.aggregate_max("system:time_start").getInfo()
    if run_ms is None:
        raise LookupError(f"gfs: no forecast run issued in the 2 days before {issued}")

    def std(img):
        img = img.resample("bilinear")
        return ee.Image.cat([img.select("temperature_2m_above_ground").rename("temp_c"),
                             img.select("relative_humidity_2m_above_ground").rename("rh_pct"),
                             img.select("u_component_of_wind_10m_above_ground").rename("wind_u_ms"),
                             img.select("v_component_of_wind_10m_above_ground").rename("wind_v_ms"),
                             img.select("gust").rename("gust_ms"),
                             img.select("precipitation_rate").multiply(3600).rename("precip_mm")]   # kg/m2/s -> mm/h
                            ).set("valid_time", img.get("forecast_time"))
    return (runs.filter(ee.Filter.eq("system:time_start", run_ms))
            .filter(ee.Filter.gte("forecast_time", _ms(start))).filter(ee.Filter.lt("forecast_time", _ms(end)))
            .map(std)), {"gfs_issued_utc": pd.Timestamp(run_ms, unit="ms", tz="UTC").isoformat()}


WEATHER: dict[str, WeatherSource] = {w.name: w for w in [
    WeatherSource("rtma", "NOAA Real-Time Mesoscale Analysis: hourly observed-weather analysis", "NOAA/NWS/RTMA",
                  "CONUS", 2500, ("temp_c", "rh_pct", "wind_u_ms", "wind_v_ms", "gust_ms", "precip_mm"), _rtma),
    WeatherSource("era5_land", "ECMWF ERA5-Land hourly reanalysis (a few days latency)", "ECMWF/ERA5_LAND/HOURLY",
                  "global (land)", 11000,
                  ("temp_c", "rh_pct", "wind_u_ms", "wind_v_ms", "precip_mm", "soil_moisture"), _era5_land),
    WeatherSource("gfs", "NOAA GFS 0.25 deg forecast: future weather for live predictions", "NOAA/GFS0P25",
                  "global", 27800, ("temp_c", "rh_pct", "wind_u_ms", "wind_v_ms", "gust_ms", "precip_mm"), _gfs),
]}
