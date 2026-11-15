"""MERRA-2 access, offline: the native grid, the OPeNDAP constraint, derived wind / RH, and open_box with a
fake CMR search and fake OPeNDAP reader that write granules in the real layout (time at HH:30, lat, lon)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from firedata import merra2 as m2

BUG_BBOX = (-120.6, 39.2, -119.4, 40.3)


def test_grid_window_covers_box_with_one_cell_margin():
    lat_idx, lon_idx = m2.grid_window(BUG_BBOX)
    lat, lon = m2.cell_centres(lat_idx, lon_idx)
    assert lat[0] <= 39.2 and lat[1] > 39.2 and lat[-1] >= 40.3 and lat[-2] < 40.3
    assert lon[0] <= -120.6 and lon[1] > -120.6 and lon[-1] >= -119.4 and lon[-2] < -119.4
    assert np.allclose(np.diff(lat), 0.5) and np.allclose(np.diff(lon), 0.625)
    # margin widens the window, edges are clamped to the globe
    wide = m2.grid_window(BUG_BBOX, margin_deg=2.0)
    assert wide[0].stop - wide[0].start > lat_idx.stop - lat_idx.start
    assert m2.grid_window((-180, -90, 179.9, 90)) == (slice(0, 361), slice(0, 576))
    with pytest.raises(ValueError):
        m2.grid_window((10, 0, -10, 1))


def test_opendap_constraint():
    ce = m2.opendap_constraint(["U10M", "V10M"], slice(258, 262), slice(95, 98))
    assert ce == ("/U10M[0:1:23][258:1:261][95:1:97];/V10M[0:1:23][258:1:261][95:1:97];/lat[258:1:261];"
                  "/lon[95:1:97];/time[0:1:23]")
    # one hour, every 3rd cell: the last index is the last one actually read
    ce = m2.opendap_constraint(["U10M"], slice(0, 361, 3), slice(0, 576, 3), slice(21, 22))
    assert ce == "/U10M[21:1:21][0:3:360][0:3:573];/lat[0:3:360];/lon[0:3:573];/time[21:1:21]"


def test_wind_stride_divides_the_longitudes():
    assert m2.wind_stride(150) == 1
    assert m2.wind_stride(577) == 3 and m2.wind_stride(1000) == 6     # 576 % 6 == 0 (5 does not divide 576)


def test_wind_direction_convention():
    spd, d = m2.wind_speed_dir([5, 0, -5, 0], [0, 5, 0, -5])
    assert np.allclose(spd, 5)
    assert np.allclose(d, [270, 180, 90, 0])     # from W, from S, from E, from N


def test_rh_from_dewpoint():
    assert np.isclose(m2.rh_from_dewpoint(300.0, 300.0), 100)
    assert 20 < float(m2.rh_from_dewpoint(303.15, 283.15)) < 35   # 30 C, dew point 10 C: ~29%


def fake_day(day: pd.Timestamp, variables, lat_idx, lon_idx) -> xr.Dataset:
    lat, lon = m2.cell_centres(lat_idx, lon_idx)
    time = pd.date_range(day.tz_localize(None) + pd.Timedelta(minutes=30), periods=24, freq="h")
    shape = (24, len(lat), len(lon))
    hour = np.arange(24, dtype="float32")[:, None, None] * np.ones(shape, "float32")
    values = {"U10M": 5.0 + 0 * hour, "V10M": 0 * hour, "T2M": 290.0 + hour, "T2MDEW": 280.0 + 0 * hour}
    return xr.Dataset({v: (("time", "lat", "lon"), values.get(v, hour).astype("float32")) for v in variables},
                      coords={"time": time, "lat": lat, "lon": lon})


@pytest.fixture
def fake_remote(monkeypatch):
    calls = []

    def search(start, end):
        days = pd.date_range(m2._utc(start).normalize(), m2._utc(end) - pd.Timedelta(seconds=1), freq="D")
        days = [d for d in days if d != pd.Timestamp("2026-08-10", tz="UTC")]   # one unpublished day
        return pd.DataFrame({"date": days, "granule": [f"g{d:%Y%m%d}" for d in days],
                             "file": [f"MERRA2_400.tavg1_2d_slv_Nx.{d:%Y%m%d}.nc4" for d in days],
                             "data": None, "opendap": "https://example", "s3": None})

    def read(granule, variables, lat_idx, lon_idx, out, hours=m2.ALL_HOURS):
        calls.append(granule["file"])
        day = fake_day(granule["date"], variables, lat_idx, lon_idx).isel(time=hours)
        m2._tidy(day, variables).to_netcdf(out, engine="h5netcdf")

    monkeypatch.setattr(m2, "search_granules", search)
    monkeypatch.setattr(m2, "_read_opendap", read)
    return calls


def test_open_box_reads_window_skips_missing_day_and_caches(fake_remote, tmp_path):
    vars_ = ("U10M", "V10M", "T2M", "T2MDEW")
    ds = m2.open_box(BUG_BBOX, "2026-08-09T22:00Z", "2026-08-12T00:00Z", vars_, cache_dir=tmp_path)
    t = pd.DatetimeIndex(ds.time.values)
    assert t[0] == pd.Timestamp("2026-08-09 22:30") and t[-1] == pd.Timestamp("2026-08-11 23:30")
    assert len(t) == 2 + 24 and pd.Timestamp("2026-08-10 12:30") not in t     # 08-10 missing
    assert ds.attrs["missing_days"] == "2026-08-10"
    assert np.allclose(ds.wind_speed_10m_ms, 5) and np.allclose(ds.wind_dir_10m_deg, 270)
    assert np.allclose(ds.temp_c.sel(time="2026-08-11 03:30"), 290 + 3 - 273.15, atol=1e-4)
    assert {"rh_pct", "wind_speed_10m_ms"} <= set(ds.data_vars)
    assert len(fake_remote) == 2
    m2.open_box(BUG_BBOX, "2026-08-09T22:00Z", "2026-08-12T00:00Z", vars_, cache_dir=tmp_path)
    assert len(fake_remote) == 2          # second call: all from the cache

    df = m2.point_series(ds, 39.727, -120.037)
    assert str(df.index.tz) == "UTC" and len(df) == len(t)
    assert np.allclose(df.wind_dir_10m_deg, 270) and "lat" not in df.columns


def test_open_box_no_data(monkeypatch, tmp_path):
    monkeypatch.setattr(m2, "search_granules", lambda s, e: pd.DataFrame(columns=["date", "file"]))
    with pytest.raises(LookupError):
        m2.open_box(BUG_BBOX, "2030-01-01", "2030-01-02", cache_dir=tmp_path)


def test_unknown_variable(tmp_path):
    g = pd.Series({"file": "x.nc4", "date": pd.Timestamp("2026-08-08", tz="UTC")})
    with pytest.raises(ValueError):
        m2.fetch_day(g, slice(0, 2), slice(0, 2), ["WIND"], tmp_path)


def test_wind_grid_json(fake_remote, tmp_path):
    w = m2.wind_grid(BUG_BBOX, "2026-08-09T21:40Z", cache_dir=tmp_path)
    assert w["hour_start_utc"] == "2026-08-09T21:00:00+00:00" and w["time_utc"] == "2026-08-09T21:30:00+00:00"
    assert len(w["u"]) == len(w["lat"]) and len(w["u"][0]) == len(w["lon"])
    assert w["lat"] == sorted(w["lat"]) and w["lon"] == sorted(w["lon"])
    assert w["mean_speed_ms"] == 5.0 and w["mean_dir_deg"] == 270.0
    with pytest.raises(LookupError):
        m2.wind_grid(BUG_BBOX, "2026-08-10T12:00Z", cache_dir=tmp_path)   # the unpublished day


def test_wind_grid_whole_world_wraps(fake_remote, tmp_path):
    w = m2.wind_grid((-180, -90, 180, 90), "2026-08-09T21:00Z", cache_dir=tmp_path)
    assert w["global_lon"] and w["stride"] == 3
    assert len(w["lat"]) <= m2.WIND_MAX_CELLS and len(w["lon"]) <= m2.WIND_MAX_CELLS + 1
    assert w["lon"][0] == -180 and w["lon"][-1] == 180                        # first column repeated at +180
    assert np.allclose(np.diff(w["lon"]), 3 * 0.625)                           # evenly spaced across the date line
    assert w["u"][0][0] == w["u"][0][-1] and len(w["u"][0]) == len(w["lon"])
