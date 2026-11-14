"""NISAR GCOV -> spread model, end to end on a small synthetic GCOV file (no download, no Earthdata login).

The synthetic file has the real GCOV layout (frequencyA HHHH / HVHV / xCoordinates / yCoordinates /
projection WKT) at 30 m in UTM 10N around the Bug Fire, with dense forest, a lake, and a strip with
no data (outside the swath).
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS

h5py = pytest.importorskip("h5py")
from firedata import nisar_provider as npv  # noqa: E402
from firedata.vegetation import FREQ_PATH  # noqa: E402

BUG = (39.727395, -120.0372941)


@pytest.fixture(scope="module")
def gcov(tmp_path_factory):
    from pyproj import Transformer
    cx, cy = Transformer.from_crs(4326, 32610, always_xy=True).transform(BUG[1], BUG[0])
    x = cx - 6000 + 30 * np.arange(400)          # 12 km, ascending
    y = cy + 6000 - 30 * np.arange(400)          # descending (north first), like GCOV
    xx, yy = np.meshgrid(x, y)
    hh = np.full(xx.shape, 10 ** (-9 / 10), np.float32)      # -9 dB
    hv = np.full(xx.shape, 10 ** (-15 / 10), np.float32)     # -15 dB: vegetated
    lake = (np.abs(xx - (cx + 2000)) < 600) & (np.abs(yy - cy) < 600)
    hh[lake], hv[lake] = 10 ** (-25 / 10), 10 ** (-30 / 10)  # specular water
    hh[:, -60:] = 0                                          # east strip: fill value (no data)
    hv[:, -60:] = 0
    path = tmp_path_factory.mktemp("nisar") / "NISAR_L2_PR_GCOV_TEST.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group(FREQ_PATH)
        g["HHHH"], g["HVHV"], g["xCoordinates"], g["yCoordinates"] = hh, hv, x, y
        g["projection"] = np.int32(32610)
        g["projection"].attrs["spatial_ref"] = CRS.from_epsg(32610).to_wkt()
    return {"path": path, "cx": cx, "cy": cy}


def model_inputs(epsg=32610, radius_km=4, res_m=100):
    """A geelayers-style inputs Dataset (no Earth Engine): shrub, flat, calm, fire at the Bug ignition."""
    from geelayers import Grid
    from geelayers.api import _empty_dataset
    g = Grid.from_point(*BUG, radius_km=radius_km, res_m=res_m, epsg=epsg)
    ds = _empty_dataset(g)
    shape = g.shape
    ds["elevation_m"] = (("y", "x"), np.zeros(shape, np.float32))
    ds["fuel_lifeform"] = (("y", "x"), np.full(shape, 2, np.int16))
    burning = np.zeros(shape, np.int16)
    burning[g.cell_of(*BUG)] = 1
    ds["initial_burned"] = (("y", "x"), burning)
    ds.attrs["time_utc"] = "2026-08-09T21:00:00+00:00"
    return ds, g


def test_layers_keep_no_data_as_no_data():
    hh = np.array([10 ** -0.9, 10 ** -2.5, 0.0, np.nan], np.float32)
    hv = np.array([10 ** -1.5, 10 ** -3.0, 0.0, 1.0], np.float32)
    out = npv.layers_from_backscatter(hh, hv)
    assert out["nisar_burned"].tolist() == [0, 1, -1, -1]          # forest, water, fill, NaN
    assert 0 < out["nisar_cover_pct"][0] <= 100 and np.isnan(out["nisar_cover_pct"][2:]).all()
    assert 0.02 <= out["nisar_soil_moisture"][0] <= 0.45 and np.isnan(out["nisar_soil_moisture"][2])


def test_provider_feeds_the_spread_model(gcov):
    from firespread import predict
    ds, g = model_inputs(radius_km=5)              # reaches into the no-data strip (east of +4.2 km)
    provider = npv.nisar_provider(h5_path=gcov["path"])
    res = predict(inputs=ds, nisar=provider)
    assert res.attrs["nisar_alignment"] == "same grid"                 # read straight onto the model grid
    assert "nisar_cover_pct" in res.attrs["source_p_den"] and "nisar_soil_moisture" in res.attrs["source_p_m"]
    # the lake is masked (water), the no-data strip is not (fire must not stop at a swath edge)
    r_lake, c_lake = g.cell_of(*_latlon(gcov["cx"] + 2000, gcov["cy"]))
    assert float(res["p_mask"][r_lake, c_lake]) == 0
    r_edge, c_edge = g.cell_of(*_latlon(gcov["cx"] + 4700, gcov["cy"]))
    assert float(res["p_mask"][r_edge, c_edge]) == 1
    assert float(res["p_den"][r_edge, c_edge]) == 0 and float(res["p_dry"][r_edge, c_edge]) == 0
    # vegetated cells get NISAR values
    r, c = g.cell_of(*BUG)
    assert float(res["ignition_prob"][r - 1, c]) > 0


def test_provider_on_another_utm_zone(gcov):
    """The model grid may be in UTM 11 (fire perimeter east of -120); the file is UTM 10."""
    from firespread import predict
    ds, _ = model_inputs(epsg=32611)
    res = predict(inputs=ds, nisar=npv.nisar_provider(h5_path=gcov["path"]))
    assert res.attrs["crs"] == "EPSG:32611" and res.attrs["nisar_alignment"] == "same grid"
    assert np.isfinite(res["p_den"].values).all()


def test_provider_outside_the_granule_returns_nothing(gcov):
    from firespread import predict
    from geelayers import Grid
    ds, _ = model_inputs()
    far = Grid.from_point(40.5, -121.5, radius_km=2, res_m=100, epsg=32610)
    from geelayers.api import _empty_dataset
    ds2 = _empty_dataset(far)
    ds2["initial_burned"] = (("y", "x"), np.pad(np.ones((1, 1), np.int16), ((20, 20), (20, 20))))
    ds2.attrs["time_utc"] = ds.attrs["time_utc"]
    res = predict(inputs=ds2, nisar=npv.nisar_provider(h5_path=gcov["path"]))
    assert res.attrs["nisar_alignment"] == "provider returned no NISAR data"


def test_latest_pass_before_start_time():
    class Req:
        time_utc = "2026-08-09T21:00:00+00:00"
        bbox_lonlat = (-120.1, 39.7, -120.0, 39.8)
    file_id, raw = npv._latest_pass_before(Req, "bug_fire", None, 30)
    passes = pd.read_csv(npv.FIRES["bug_fire"].root / "processed" / "nisar_passes.csv")
    t = pd.to_datetime(passes.time_utc, utc=True)
    expected = passes[t < pd.Timestamp(Req.time_utc)].assign(t=t).sort_values("t").iloc[-1]
    assert file_id == expected.file_id                                  # never a pass after the start
    file_id, _ = npv._latest_pass_before(Req, "bug_fire", "A34", 30)
    assert "_034_A_" in file_id
    Req.time_utc = "2026-01-01T00:00:00+00:00"
    assert npv._latest_pass_before(Req, "bug_fire", None, 30) is None


def test_to_spread_model_keeps_contract_variables():
    ds = xr.Dataset({v: (("y", "x"), np.zeros((2, 2))) for v in
                     ("nisar_cover_pct", "nisar_soil_moisture", "nisar_burned", "nisar_combustible")},
                    coords={"y": [1.0, 0.0], "x": [0.0, 1.0], "lat": (("y", "x"), np.zeros((2, 2))),
                            "lon": (("y", "x"), np.zeros((2, 2)))}, attrs={"crs": "EPSG:32610"})
    out = npv.to_spread_model(ds)
    assert set(out.data_vars) == set(npv.SPREAD_MODEL_VARIABLES) and out.attrs["crs"] == "EPSG:32610"


def _latlon(x, y):
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(32610, 4326, always_xy=True).transform(x, y)
    return lat, lon
