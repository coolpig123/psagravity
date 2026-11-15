"""Offline tests of the replay weather swap (MERRA-2 in place of RTMA)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

merra2 = pytest.importorskip("firedata.merra2")
from hindcast import replay  # noqa: E402


def test_with_merra2_replaces_weather(monkeypatch, tmp_path):
    times = pd.date_range("2026-08-10T09:00", periods=3, freq="h")
    lat = np.full((2, 2), 39.7, np.float32)
    lon = np.full((2, 2), -120.0, np.float32)
    ds = xr.Dataset({"wind_u_ms": (("time", "y", "x"), np.zeros((3, 2, 2), np.float32)),
                     "gust_ms": (("time", "y", "x"), np.ones((3, 2, 2), np.float32))},
                    coords={"time": times, "y": [1, 0], "x": [0, 1], "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)})
    mt = pd.date_range("2026-08-10T07:30", periods=6, freq="h")
    shape = (6, 3, 3)
    m = xr.Dataset({"U10M": (("time", "lat", "lon"), np.zeros(shape)),            # wind from the north:
                    "V10M": (("time", "lat", "lon"), np.full(shape, -4.0)),        # blowing toward the south
                    "temp_c": (("time", "lat", "lon"), np.broadcast_to(np.arange(6.0)[:, None, None], shape).copy()),
                    "rh_pct": (("time", "lat", "lon"), np.full(shape, 20.0))},
                   coords={"time": mt, "lat": [39.5, 40.0, 40.5], "lon": [-120.625, -120.0, -119.375]})
    monkeypatch.setattr(merra2, "open_box", lambda *a, **k: m)
    monkeypatch.setattr(replay, "CACHE_DIR", tmp_path)
    out = replay.with_merra2(ds, "x")
    assert "gust_ms" not in out
    assert np.allclose(out["wind_dir_deg"], 0.0) and np.allclose(out["wind_speed_ms"], 4.0)
    assert np.allclose(out["temp_c"].isel(y=0, x=0), [1.5, 2.5, 3.5])       # 09:00 between 08:30 and 09:30
    assert out.attrs["weather_source"] == "merra2"
