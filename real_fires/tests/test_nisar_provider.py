"""NISAR access + provider, offline: synthetic GCOV / SME2 files with the real layout, and a fake ASF search.

GCOV: 10 m, UTM 11N (EPSG:32611), HHHH / HVHV / mask (0 invalid, 255 outside, 1.. valid), like the real
Bug Fire granules. SME2: 200 m, EASE-Grid 2.0 (EPSG:6933), soilMoisture + retrievalQualityFlag (bit 0 = not
recommended), PMI vegetationWaterContentRetrieved + PMI flag (0 = clear), waterbodyFraction.
"""
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

h5py = pytest.importorskip("h5py")
from firedata import nisar_access as na  # noqa: E402
from firedata import nisar_provider as npv  # noqa: E402

BUG = (39.727395, -120.0372941)
T_START = pd.Timestamp("2026-08-09T21:00Z")


# ---- synthetic granules ----------------------------------------------------------------------

def model_grid(radius_km=2.0, res_m=100.0, epsg=32611):
    from geelayers import Grid
    return Grid.from_point(*BUG, radius_km=radius_km, res_m=res_m, epsg=epsg)


class Request:
    """What the spread model hands a provider (firespread.NisarRequest)."""
    def __init__(self, g, time=T_START):
        self.crs, self.x, self.y, self.res_m = g.crs, g.x, g.y, g.res
        self.bbox_lonlat, self.time_utc = g.bbox_lonlat(), str(time)


def write_gcov(path, g, hh_db=-9.0, hv_db=-15.0, *, edit=None, margin_m=600, res=10.0, x_range=None):
    """GCOV over the model grid g (+ margin) at `res`, UTM 11N. edit(xx, yy, hh, hv, mask) changes fields in place."""
    xmin, ymin, xmax, ymax = (g.x0 - margin_m, g.y0 - g.height * g.res - margin_m,
                              g.x0 + g.width * g.res + margin_m, g.y0 + margin_m)
    if x_range:
        xmin, xmax = x_range
    x = np.arange(xmin + res / 2, xmax, res)
    y = np.arange(ymax - res / 2, ymin, -res)
    xx, yy = np.meshgrid(x, y)
    hh = np.full(xx.shape, 10 ** (hh_db / 10), np.float32)
    hv = np.full(xx.shape, 10 ** (hv_db / 10), np.float32)
    mask = np.ones(xx.shape, np.uint8)
    if edit:
        edit(xx, yy, hh, hv, mask)
    with h5py.File(path, "w") as f:
        grp = f.create_group(na.GCOV_GROUP)
        grp["HHHH"], grp["HVHV"], grp["mask"] = hh, hv, mask
        grp["mask"].attrs["_FillValue"] = np.uint8(255)
        grp["xCoordinates"], grp["yCoordinates"] = x, y
        grp["projection"] = np.uint32(32611)
        grp["projection"].attrs["epsg_code"] = np.uint32(32611)
    return path


def write_sme2(path, g, sm=0.08, flag=12, vwc=0.3, pmi_flag=0, water=0.0, *, edit=None):
    """SME2 over g (+ margin) at 200 m in EASE-Grid 2.0 (EPSG:6933)."""
    tr = Transformer.from_crs(g.crs, 6933, always_xy=True)
    ex, ey = tr.transform([g.x0 - 2000, g.x0 + g.width * g.res + 2000], [g.y0 + 2000, g.y0 - g.height * g.res - 2000])
    x = np.arange(min(ex), max(ex), 200.18)
    y = np.arange(max(ey), min(ey), -200.18)
    shape = (len(y), len(x))
    fields = {"soilMoisture": np.full(shape, sm, np.float32), "retrievalQualityFlag": np.full(shape, flag, np.int16),
              "vwc": np.full(shape, vwc, np.float32), "pmi": np.full(shape, pmi_flag, np.int16),
              "water": np.full(shape, water, np.float32)}
    if edit:
        edit(x, y, fields)
    with h5py.File(path, "w") as f:
        grp = f.create_group(na.SME2_GROUP)
        grp["soilMoisture"] = fields["soilMoisture"]
        grp["soilMoisture"].attrs["_FillValue"] = np.float32(-9999)
        grp["retrievalQualityFlag"] = fields["retrievalQualityFlag"]
        grp["retrievalQualityFlag"].attrs["_FillValue"] = np.int16(-9999)
        grp["algorithmCandidates/PMI/vegetationWaterContentRetrieved"] = fields["vwc"]
        grp["algorithmCandidates/PMI/retrievalQualityFlag"] = fields["pmi"]
        grp["ancillaryData/waterbodyFraction"] = fields["water"]
        grp["xCoordinates"], grp["yCoordinates"] = x, y
        grp["projection"] = np.uint32(6933)
        grp["projection"].attrs["epsg_code"] = np.uint32(6933)
    return path


def fake_search(catalog):
    """search_granules stand-in: catalog = list of (product, time, track, frame, local path)."""
    def search(product, bbox, start, end):
        rows = [{"time_utc": pd.Timestamp(t), "track_id": tr, "frame": fr, "file_id": f"{product}_{tr}_{fr}_{t}",
                 "url": str(p), "version": 1}
                for prod, t, tr, fr, p in catalog if prod == product and pd.Timestamp(start) <= pd.Timestamp(t) < pd.Timestamp(end)]
        return pd.DataFrame(rows, columns=["time_utc", "track_id", "frame", "file_id", "url", "version"])
    return search


def latlon_of(g, x, y):
    lon, lat = Transformer.from_crs(g.crs, 4326, always_xy=True).transform(x, y)
    return lat, lon


# ---- access: reading only the area, averaging, sampling, masks, frames, cache -----------------

def test_mean_aggregation_is_exact_cell_average(tmp_path):
    g = model_grid()
    tg = na.TargetGrid.from_request(Request(g))
    # HV = a linear ramp in x: the mean over each 100 m cell equals the ramp at the cell centre
    def ramp(xx, yy, hh, hv, mask):
        hv[:] = ((xx - g.x0) / 1e5 + 0.01).astype(np.float32)
    path = write_gcov(tmp_path / "g.h5", g, edit=ramp)
    out = na.read_layers(path, na.GCOV_GROUP, ["HVHV"], tg, cache_dir=None)
    expected = ((g.x - g.x0) / 1e5 + 0.01)[None, :] * np.ones((g.height, 1))
    assert out["HVHV"].shape == g.shape
    assert np.allclose(out["HVHV"], expected, rtol=1e-4)


def test_invalid_samples_are_excluded_and_outside_is_nan(tmp_path):
    g = model_grid()
    tg = na.TargetGrid.from_request(Request(g))
    def edit(xx, yy, hh, hv, mask):
        hv[:, : xx.shape[1] // 2] = 1.0           # west half: bright, but flagged invalid
        mask[:, : xx.shape[1] // 2] = 0
    path = write_gcov(tmp_path / "g.h5", g, edit=edit)
    out = na.read_layers(path, na.GCOV_GROUP, ["HVHV", "mask"], tg, prepare=npv._gcov_valid, cache_dir=None)
    hv = out["HVHV"]
    assert np.nanmax(hv) == pytest.approx(10 ** -1.5, rel=1e-4)      # invalid bright pixels never averaged in
    assert np.isnan(hv[:, 0]).all() and np.isfinite(hv[:, -1]).all()
    # a granule that does not reach the grid: all NaN, no error
    far = write_gcov(tmp_path / "far.h5", g, x_range=(g.x0 + 50_000, g.x0 + 51_000))
    out = na.read_layers(far, na.GCOV_GROUP, ["HVHV"], tg, cache_dir=None)
    assert np.isnan(out["HVHV"]).all()


def test_frames_are_mosaicked(tmp_path):
    g = model_grid()
    tg = na.TargetGrid.from_request(Request(g))
    mid = g.x0 + g.width * g.res / 2
    a = write_gcov(tmp_path / "a.h5", g, hv_db=-15, x_range=(g.x0 - 600, mid))
    b = write_gcov(tmp_path / "b.h5", g, hv_db=-20, x_range=(mid, g.x0 + g.width * g.res + 600))
    p = na.Pass("A106", pd.Timestamp("2026-07-27T13:01Z"),
                [{"file_id": "a", "url": str(a)}, {"file_id": "b", "url": str(b)}])
    out = na.read_pass(p, na.GCOV_GROUP, ["HVHV"], tg, cache_dir=None)["HVHV"]
    assert np.isfinite(out).all()
    assert out[0, 0] == pytest.approx(10 ** -1.5, rel=1e-4) and out[0, -1] == pytest.approx(10 ** -2.0, rel=1e-4)


def test_coarser_source_is_sampled_and_flags_applied(tmp_path):
    g = model_grid()
    tg = na.TargetGrid.from_request(Request(g))
    def edit(x, y, f):
        west = x[None, :] < np.median(x)
        f["retrievalQualityFlag"][:] = np.where(west, 15, 14)    # west: not recommended (bit 0 set)
        f["pmi"][:] = np.where(west, 0, 3)                        # east: PMI anomaly
    path = write_sme2(tmp_path / "s.h5", g, sm=0.08, vwc=0.3, edit=edit)
    out = na.read_layers(path, na.SME2_GROUP, npv.SME2_DATASETS, tg, prepare=npv._sme2_valid, cache_dir=None)
    sm, vwc = out["soil_moisture"], out["vwc"]
    assert sm.shape == g.shape
    assert np.isnan(sm[:, 0]).all() and np.allclose(sm[:, -1], 0.08)       # only recommended retrievals
    assert np.allclose(vwc[:, 0], 0.3) and np.isnan(vwc[:, -1]).all()      # only clear PMI retrievals


def test_cache_avoids_rereading(tmp_path, monkeypatch):
    g = model_grid()
    tg = na.TargetGrid.from_request(Request(g))
    path = write_gcov(tmp_path / "g.h5", g)
    first = na.read_layers(path, na.GCOV_GROUP, ["HVHV"], tg, cache_dir=tmp_path / "cache")
    monkeypatch.setattr(na, "open_granule", lambda *a: pytest.fail("should come from the cache"))
    second = na.read_layers(path, na.GCOV_GROUP, ["HVHV"], tg, cache_dir=tmp_path / "cache")
    assert np.array_equal(first["HVHV"], second["HVHV"], equal_nan=True)


class LocalAsRemote:
    """Stands in for the Earthdata HTTPS filesystem: serves local files, counts opens and range requests."""
    def __init__(self):
        import fsspec
        self.fs, self.opens, self.ranges = fsspec.filesystem("file"), 0, 0

    def open(self, url, **kw):
        self.opens += 1
        return self.fs.open(url.removeprefix("remote://"), "rb")

    def cat_ranges(self, paths, starts, ends, on_error="raise"):
        self.ranges += len(paths)
        return self.fs.cat_ranges([p.removeprefix("remote://") for p in paths], starts, ends)


def chunked_copy(src, dst, chunks=(64, 64)):
    """The same granule stored like real NISAR files: chunked, deflate + shuffle (HH/HV), deflate only (mask)."""
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        gi, go = fi[na.GCOV_GROUP], fo.create_group(na.GCOV_GROUP)
        for k in ("xCoordinates", "yCoordinates", "projection"):
            gi.copy(k, go)
        for k in ("HHHH", "HVHV", "mask"):
            d = go.create_dataset(k, data=gi[k][:], chunks=chunks, compression="gzip",
                                  compression_opts=1 if k != "mask" else 9, shuffle=k != "mask")
            d.attrs.update(gi[k].attrs)
    return dst


def test_remote_parallel_chunks_match_h5py_and_are_cached(tmp_path, monkeypatch):
    g = model_grid()
    def pattern(xx, yy, hh, hv, mask):                          # varied values and some invalid samples
        hv[:] = (0.02 + 0.01 * np.sin(xx / 300) * np.cos(yy / 200)).astype(np.float32)
        mask[(xx // 170 + yy // 130) % 7 == 0] = 0
        mask[:, :25] = 255
    local = chunked_copy(write_gcov(tmp_path / "plain.h5", g, edit=pattern), tmp_path / "chunked.h5")
    remote = LocalAsRemote()
    monkeypatch.setattr(na, "_remote", lambda url: (remote, url, None))
    url = "remote://" + str(local)
    names = ["HHHH", "HVHV", "mask"]

    tg = na.TargetGrid.from_request(Request(g))
    want = na.read_layers(local, na.GCOV_GROUP, names, tg, prepare=npv._gcov_valid, cache_dir=None)
    got = na.read_layers(url, na.GCOV_GROUP, names, tg, prepare=npv._gcov_valid, cache_dir=tmp_path / "c",
                         cache_key="G1")
    for k in want:
        assert np.allclose(got[k], want[k], equal_nan=True, rtol=1e-6)   # identical up to float summation order
    assert remote.opens == 1 and remote.ranges > 0

    # another grid over the same area (a new polygon): no remote access at all, from the chunk cache
    tg2 = na.TargetGrid.from_request(Request(model_grid(radius_km=1.5)))
    remote.opens, remote.ranges = 0, 0
    again = na.read_layers(url, na.GCOV_GROUP, names, tg2, prepare=npv._gcov_valid, cache_dir=tmp_path / "c",
                           cache_key="G1")
    assert remote.opens == 0 and remote.ranges == 0
    direct = na.read_layers(local, na.GCOV_GROUP, names, tg2, prepare=npv._gcov_valid, cache_dir=None)
    assert np.allclose(again["HVHV"], direct["HVHV"], equal_nan=True, rtol=1e-6)


def test_dataset_missing_from_older_granule_is_no_data(tmp_path, monkeypatch):
    """Older SME2 versions lack some datasets (seen over Haifa): that variable is NaN, the rest still reads."""
    g = model_grid()
    tg = na.TargetGrid.from_request(Request(g))
    path = write_sme2(tmp_path / "sme2.h5", g, sm=0.15, flag=0)
    with h5py.File(path, "a") as f:
        del f[na.SME2_GROUP + "/algorithmCandidates/PMI/vegetationWaterContentRetrieved"]
    remote = LocalAsRemote()
    monkeypatch.setattr(na, "_remote", lambda url: (remote, url, None))
    for i, source in enumerate((path, "remote://" + str(path))):    # local and remote readers
        out = na.read_layers(source, na.SME2_GROUP, npv.SME2_DATASETS, tg, prepare=npv._sme2_valid,
                             cache_dir=tmp_path / f"c{i}", cache_key="old")
        assert np.isnan(out["vwc"]).all()
        assert np.allclose(out["soil_moisture"], 0.15)
    assert remote.opens == 1                                          # the remote reader really ran


def test_stalled_remote_read_is_retried(tmp_path, monkeypatch):
    g = model_grid()
    tg = na.TargetGrid.from_request(Request(g))
    path = write_gcov(tmp_path / "g.h5", g)
    calls = {"n": 0}
    real = na._read_window

    def flaky(source, *args):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("no data for 120 s")
        return real(path, *args)
    monkeypatch.setattr(na, "_read_window", flaky)
    out = na.read_layers("https://example/remote.h5", na.GCOV_GROUP, ["HVHV"], tg, cache_dir=None)
    assert calls["n"] == 3 and np.isfinite(out["HVHV"]).all()
    calls["n"] = -10                                             # always fails: gives up after READ_ATTEMPTS
    monkeypatch.setattr(na, "_read_window", lambda *a: (_ for _ in ()).throw(TimeoutError("stalled")))
    with pytest.raises(TimeoutError):
        na.read_layers("https://example/remote.h5", na.GCOV_GROUP, ["HVHV"], tg, cache_dir=None)


def test_search_keeps_latest_version_and_groups_frames(monkeypatch):
    feats = []
    for fid, t, frame in [("NISAR_L3_PR_SME2_026_042_D_068_X_001", "2026-07-23T02:55:30Z", 68),
                          ("NISAR_L3_PR_SME2_026_042_D_068_X_002", "2026-07-23T02:55:30Z", 68),
                          ("NISAR_L3_PR_SME2_026_042_D_069_X_001", "2026-07-23T02:56:04Z", 69),
                          ("NISAR_L3_UR_SME2_026_042_D_069_X_001", "2026-07-23T02:56:04Z", 69),
                          ("NISAR_L3_PR_SME2_026_106_A_022_X_001", "2026-07-27T13:01:46Z", 22)]:
        feats.append({"properties": {"fileID": fid, "startTime": t, "flightDirection": "DESCENDING" if "_D_" in fid
                                     else "ASCENDING", "pathNumber": int(fid.split("_")[5]), "frameNumber": frame,
                                     "url": "u/" + fid}})

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"features": feats}
    monkeypatch.setattr(na.requests, "get", lambda *a, **k: Resp())
    df = na.search_granules("SME2", (-120.1, 39.6, -119.9, 39.8), "2026-07-01", "2026-08-01")
    assert len(df) == 3 and "_D_068_X_002" in " ".join(df.file_id) and "_068_X_001" not in " ".join(df.file_id)
    passes = na.group_passes(df)
    assert [(p.track_id, len(p.granules)) for p in passes] == [("D42", 2), ("A106", 1)]


# ---- provider: pass choice, cover, scars, SME2, end to end -----------------------------------

def test_cover_curve():
    hv = 10 ** (np.array([-30, -21, -17.6, -12, -5, np.nan]) / 10)
    c = npv.cover_from_hv(hv)
    assert c[:5].tolist() == pytest.approx([10, 10, 30, 55, 70]) and np.isnan(c[5])


def test_provider_end_to_end(tmp_path):
    from firespread import AlexandridisParams, NisarRequest, predict
    g = model_grid(radius_km=2.0)
    xs, ys = g.x, g.y
    scar_x = (g.x0 + 300, g.x0 + 900)                 # 600 m wide strip burned between the two passes
    lake = (xs[-8], ys[8])                           # SME2 water body near the north-east corner

    def scarred(xx, yy, hh, hv, mask):
        s = (xx > scar_x[0]) & (xx < scar_x[1])
        hv[s] *= 10 ** (-6 / 10)                       # -6 dB after the fire
    def sme2_edit(x, y, f):
        tr = Transformer.from_crs(g.crs, 6933, always_xy=True)
        lx, ly = tr.transform(*lake)
        near = (np.abs(x[None, :] - lx) < 300) & (np.abs(y[:, None] - ly) < 300)
        f["water"][near] = 1.0
    ref = write_gcov(tmp_path / "ref.h5", g, hv_db=-14)
    new = write_gcov(tmp_path / "new.h5", g, hv_db=-14, edit=scarred)
    other_track = write_gcov(tmp_path / "other.h5", g, hv_db=-25)
    after = write_gcov(tmp_path / "after.h5", g, hv_db=-30)
    sme2 = write_sme2(tmp_path / "s.h5", g, sm=0.05, vwc=6.0, edit=sme2_edit)
    search = fake_search([
        ("GCOV", "2026-06-15T13:01Z", "A106", 22, ref),
        ("GCOV", "2026-07-20T02:55Z", "D42", 68, other_track),   # other track: not a scar reference
        ("GCOV", "2026-07-27T13:01Z", "A106", 22, new),          # latest before the start -> used
        ("GCOV", "2026-08-15T13:09Z", "A34", 21, after),         # after the start: never used
        ("SME2", "2026-07-27T13:01Z", "A106", 22, sme2),
    ])
    provider = npv.nisar_provider(search=search, cache_dir=None)
    req = Request(g)
    ds = provider(req)
    assert ds.attrs["nisar_pass_utc"].startswith("2026-07-27") and ds.attrs["nisar_track"] == "A106"
    assert ds.attrs["nisar_scar_reference_utc"].startswith("2026-06-15")
    assert set(ds.data_vars) == set(npv.SPREAD_MODEL_VARIABLES)
    burned = ds["nisar_burned"].values
    in_scar = (g.x > scar_x[0] + 100) & (g.x < scar_x[1] - 100)
    assert (burned[:, in_scar] == 1).all() and (burned[:, -20:-15] == 0).all()
    r, c = g.cell_of(*latlon_of(g, *lake))
    assert burned[r, c] == 1                                     # SME2 water body
    assert np.allclose(ds["nisar_soil_moisture"], 0.05) and np.allclose(ds["nisar_vwc_kg_m2"], 6.0)
    assert np.allclose(ds["nisar_cover_pct"].values[:, -1], npv.cover_from_hv(np.array([10 ** -1.4]))[0])

    # through the spread model: same grid, the scar cannot burn, P_dry / P_vwc from SME2
    from geelayers.api import _empty_dataset
    inputs = _empty_dataset(g)
    inputs["elevation_m"] = (("y", "x"), np.zeros(g.shape, np.float32))
    inputs["fuel_lifeform"] = (("y", "x"), np.full(g.shape, 2, np.int16))
    fire = np.zeros(g.shape, np.int16)
    fire[g.height // 2, g.width // 2] = 1
    inputs["initial_burned"] = (("y", "x"), fire)
    inputs.attrs["time_utc"] = str(T_START)
    res = predict(inputs=inputs, nisar=provider)
    assert res.attrs["nisar_alignment"] == "same grid"
    assert float(res["p_mask"][:, in_scar].max()) == 0
    assert float(res["p_dry"].max()) == pytest.approx(AlexandridisParams().dry_dry)
    assert float(res["p_vwc"].min()) == pytest.approx(AlexandridisParams().vwc_high)


def test_provider_without_passes_or_sme2(tmp_path):
    g = model_grid()
    only_gcov = write_gcov(tmp_path / "g.h5", g)
    ds = npv.nisar_provider(search=fake_search([("GCOV", "2026-08-01T13:00Z", "A106", 22, only_gcov)]),
                            cache_dir=None)(Request(g))
    assert "scars not detected" in ds.attrs["nisar_scar_reference_utc"]
    assert "not provided" in ds.attrs["nisar_sme2_granules"] and np.isnan(ds["nisar_soil_moisture"]).all()
    assert npv.nisar_provider(search=fake_search([]), cache_dir=None)(Request(g)) is None
    old = npv.nisar_provider(search=fake_search([("GCOV", "2026-06-01T13:00Z", "A106", 22, only_gcov)]),
                             cache_dir=None, lookback_days=30)(Request(g))
    assert old is None                                           # too old for lookback_days


def test_scar_change_is_averaged_over_neighbours():
    hv_ref = np.full((7, 7), 0.02)
    hv = hv_ref.copy()
    hv[3, 3] *= 10 ** (-6 / 10)                  # one speckly cell dropping 6 dB: not a scar after averaging
    hv[:, 5:] *= 10 ** (-3 / 10)                 # a 2-cell wide band dropping 3 dB: a scar
    d = npv.hv_change_db(hv, hv_ref, 3)
    assert d[3, 3] > -1.5 and (d[:, 6] <= -1.5).all()
    assert npv.hv_change_db(hv, hv_ref, 1)[3, 3] == pytest.approx(-6)
    hh = np.full((7, 7), 0.1)
    burned = npv.layers_from_backscatter(hh, hv, hv_ref)["nisar_burned"]
    assert burned[3, 3] == 0 and (burned[:, 6] == 1).all()
    hv_ref[0, 0] = np.nan                         # missing reference: ignored in the average, not an error
    assert np.isfinite(npv.hv_change_db(hv, hv_ref, 3)).all()


def test_backfill_from_older_passes(tmp_path):
    """Newest pass sees only the east half; the west half comes from an older pass, all variables from it."""
    g = model_grid(radius_km=2.0)
    mid = g.x0 + g.width * g.res / 2
    newest = write_gcov(tmp_path / "a34.h5", g, hv_db=-14, x_range=(mid, g.x0 + g.width * g.res + 600))
    older = write_gcov(tmp_path / "d42.h5", g, hv_db=-18)
    oldest = write_gcov(tmp_path / "a106.h5", g, hv_db=-10)
    sme2_new = write_sme2(tmp_path / "s_a34.h5", g, sm=0.05, vwc=0.3)
    sme2_old = write_sme2(tmp_path / "s_d42.h5", g, sm=0.20, vwc=5.0)
    search = fake_search([
        ("GCOV", "2026-07-27T13:01Z", "A106", 22, oldest),
        ("GCOV", "2026-08-04T02:55Z", "D42", 68, older),
        ("GCOV", "2026-08-08T13:09Z", "A34", 21, newest),
        ("SME2", "2026-08-04T02:55Z", "D42", 68, sme2_old),
        ("SME2", "2026-08-08T13:09Z", "A34", 21, sme2_new),
    ])
    ds = npv.nisar_provider(search=search, cache_dir=None, detect_scars=False)(Request(g))
    assert ds.attrs["nisar_coverage_pct"] == 100.0
    assert ds.attrs["nisar_pass_utc"].startswith("2026-08-08") and ds.attrs["nisar_track"] == "A34"
    used = ds.attrs["nisar_passes_used"]
    assert "A34" in used and "D42" in used and "A106" not in used     # stopped once covered
    east, west = g.x > mid + 100, g.x < mid - 100
    cover = ds["nisar_cover_pct"].values
    assert np.allclose(cover[:, east], npv.cover_from_hv(np.array([10 ** -1.4]))[0])
    assert np.allclose(cover[:, west], npv.cover_from_hv(np.array([10 ** -1.8]))[0])
    # a cell takes radar AND SME2 from the same pass
    assert np.allclose(ds["nisar_soil_moisture"].values[:, east], 0.05)
    assert np.allclose(ds["nisar_soil_moisture"].values[:, west], 0.20)
    assert np.allclose(ds["nisar_vwc_kg_m2"].values[:, west], 5.0)
    assert (ds["nisar_burned"].values != -1).all()
    detail = __import__("json").loads(ds.attrs["nisar_passes_detail"])
    assert [d["track"] for d in detail] == ["A34", "D42"] and sum(d["cells_pct"] for d in detail) == pytest.approx(100)


def test_full_coverage_reads_only_the_newest_pass(tmp_path):
    g = model_grid()
    full = write_gcov(tmp_path / "full.h5", g)
    search = fake_search([("GCOV", "2026-08-01T13:00Z", "D42", 68, tmp_path / "missing.h5"),   # would fail if read
                          ("GCOV", "2026-08-08T13:00Z", "A34", 21, full)])
    ds = npv.nisar_provider(search=search, cache_dir=None, detect_scars=False)(Request(g))
    assert ds.attrs["nisar_passes_used"].count("%") == 1 and ds.attrs["nisar_coverage_pct"] == 100.0


def test_partial_coverage_kept_when_no_older_pass(tmp_path):
    g = model_grid()
    mid = g.x0 + g.width * g.res / 2
    half = write_gcov(tmp_path / "half.h5", g, x_range=(mid, g.x0 + g.width * g.res + 600))
    ds = npv.nisar_provider(search=fake_search([("GCOV", "2026-08-08T13:00Z", "A34", 21, half)]),
                            cache_dir=None, detect_scars=False)(Request(g))
    assert 40 < ds.attrs["nisar_coverage_pct"] < 60
    west = g.x < mid - 100
    assert (ds["nisar_burned"].values[:, west] == -1).all()      # unknown, never "burned"


def test_earthdata_login_once_per_process(monkeypatch):
    import types
    import earthaccess
    import fsspec
    logins = {"n": 0}
    monkeypatch.setattr(earthaccess, "login", lambda **k: logins.__setitem__("n", logins["n"] + 1))
    monkeypatch.setattr(earthaccess, "__store__",
                        types.SimpleNamespace(auth=types.SimpleNamespace(token={"access_token": "t"})), raising=False)
    monkeypatch.setattr(fsspec, "filesystem", lambda *a, **k: object())
    na._https_filesystem.cache_clear()
    fs1, fs2 = na._https_filesystem(), na._https_filesystem()
    assert fs1 is fs2 and logins["n"] == 1
    na._https_filesystem.cache_clear()                           # what a failed read does before retrying
    na._https_filesystem()
    assert logins["n"] == 2
    na._https_filesystem.cache_clear()


def test_compute_burned_mask_no_data_is_minus_one():
    from firedata.nisar_layers import compute_burned_mask
    hh = np.array([10 ** -0.9, 10 ** -2.5, np.nan, 10 ** -0.9], np.float32)
    hv = np.array([10 ** -1.5, 10 ** -3.0, 1.0, np.nan], np.float32)
    burned, combustible = compute_burned_mask(hh, hv)
    assert burned.tolist() == [0, 1, -1, -1] and combustible.tolist() == [1, 0, -1, -1]


def test_build_nisar_inputs_script(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_nisar_inputs.py"
    spec = importlib.util.spec_from_file_location("build_nisar_inputs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = {}

    def fake_provider(**kw):
        def get(request):
            seen["time"], seen["crs"] = request.time_utc, request.crs
            import xarray as xr
            return xr.Dataset({"nisar_cover_pct": (("y", "x"), np.full((len(request.y), len(request.x)), 20.0))},
                              coords={"y": request.y, "x": request.x},
                              attrs={"crs": request.crs, "nisar_passes_used": "x", "nisar_coverage_pct": 100.0})
        return get
    monkeypatch.setattr(mod, "nisar_provider", fake_provider)
    out = tmp_path / "bug.nc"
    monkeypatch.setattr(sys, "argv", ["build_nisar_inputs.py", "bug_fire", "--radius-km", "2", "--out", str(out)])
    mod.main()
    assert out.exists() and seen["time"].startswith("2026-08-08T22:30")      # pre = ignition time


def test_no_data_stays_no_data():
    hh = np.array([10 ** -0.9, 10 ** -2.5, np.nan], np.float32)
    hv = np.array([10 ** -1.5, 10 ** -3.0, 1.0], np.float32)
    out = npv.layers_from_backscatter(hh, hv)
    assert out["nisar_burned"].tolist() == [0, 1, -1] and np.isnan(out["nisar_cover_pct"][2])


def test_to_spread_model_keeps_contract_variables():
    import xarray as xr
    names = ("nisar_cover_pct", "nisar_soil_moisture", "nisar_burned", "nisar_combustible")
    ds = xr.Dataset({v: (("y", "x"), np.zeros((2, 2))) for v in names},
                    coords={"y": [1.0, 0.0], "x": [0.0, 1.0], "lat": (("y", "x"), np.zeros((2, 2))),
                            "lon": (("y", "x"), np.zeros((2, 2)))}, attrs={"crs": "EPSG:32610"})
    out = npv.to_spread_model(ds)
    assert set(out.data_vars) == {"nisar_cover_pct", "nisar_soil_moisture", "nisar_burned"}
