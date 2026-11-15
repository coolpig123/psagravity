"""Calibrate the NISAR burn-scar HV-drop threshold on the Bug Fire (issue #7), vs the WFIGS perimeter.

Same-track GCOV pairs A34 07-22 -> 08-15, D42 07-23 -> 08-16, A106 07-27 -> 08-20; HV change averaged over
3 x 3 cells of 100 m (nisar_provider.hv_change_db); burned where HV dropped >= threshold. Only the fire window
is read from ASF. The threshold with the best mean IoU over the three pairs is written to
data/fires/bug_fire/processed/nisar_threshold_calibration.json and then used unchanged on other fires.

Run from real_fires/:  python scripts/nisar_threshold_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcm_burned_area as R                                   # noqa: E402  (auc_rise)
from firedata import rcm                                      # noqa: E402  (scores)
from firedata.config import REPO_ROOT                         # noqa: E402
from firedata.nisar_access import GCOV_GROUP, TargetGrid, group_passes, read_pass, search_granules  # noqa: E402
from firedata.nisar_provider import SCAR_SMOOTH_CELLS, _gcov_valid, hv_change_db  # noqa: E402

THRESHOLDS_DB = [0.25, 0.5, 0.75, 1.0, 1.5]
PAIRS = {"A34": ("07-22", "08-15"), "D42": ("07-23", "08-16"), "A106": ("07-27", "08-20")}
CRS, RES, BUFFER_M, REF_BUFFER_M = 32610, 100.0, 3000, 2000
D = REPO_ROOT / "data" / "fires" / "bug_fire"


def window(perim_utm: gpd.GeoDataFrame):
    w, s, e, n = perim_utm.buffer(BUFFER_M).total_bounds
    w, n = np.floor(w / RES) * RES, np.ceil(n / RES) * RES
    nx, ny = int(np.ceil((e - w) / RES)), int(np.ceil((n - s) / RES))
    tr = from_origin(w, n, RES, RES)
    grid = TargetGrid(f"EPSG:{CRS}", w + (np.arange(nx) + 0.5) * RES, n - (np.arange(ny) + 0.5) * RES, RES)
    W = rasterize([perim_utm.geometry.union_all()], out_shape=(ny, nx), transform=tr).astype(bool)
    out = ~rasterize([perim_utm.buffer(REF_BUFFER_M).union_all()], out_shape=(ny, nx), transform=tr).astype(bool)
    return grid, tr, W, out


def find(passes, track, mmdd):
    hits = [q for q in passes if q.track_id == track and f"{q.time_utc:%m-%d}" == mmdd]
    if len(hits) != 1:
        raise RuntimeError(f"{track} {mmdd}: {len(hits)} passes")
    return hits[0]


def hv_pair_change(passes, track, ref, post, grid):
    hv0 = read_pass(find(passes, track, ref), GCOV_GROUP, ["HVHV", "mask"], grid, prepare=_gcov_valid)["HVHV"]
    hv = read_pass(find(passes, track, post), GCOV_GROUP, ["HVHV", "mask"], grid, prepare=_gcov_valid)["HVHV"]
    return hv_change_db(hv, hv0, SCAR_SMOOTH_CELLS)


def main():
    perim = gpd.read_file(D / "processed" / "wfigs_final_perimeter.geojson")
    grid, tr, W, out = window(perim.to_crs(CRS))
    passes = group_passes(search_granules("GCOV", tuple(perim.to_crs(4326).total_bounds), "2026-07-15", "2026-08-25"))
    rows = []
    for track, (ref, post) in PAIRS.items():
        d = hv_pair_change(passes, track, ref, post, grid)
        v = np.isfinite(d)
        auc = R.auc_rise(-d[W & v], -d[out & v])
        for thr in THRESHOLDS_DB:
            rows.append({"pair": f"{track} {ref} → {post}", "drop_db": thr, "auc_drop": round(auc, 3),
                         "median_dHV_in_db": round(float(np.nanmedian(d[W & v])), 2),
                         "median_dHV_out_db": round(float(np.nanmedian(d[out & v])), 2),
                         **rcm.scores(np.nan_to_num(d) <= -thr, W, v, RES * RES)})
    df = pd.DataFrame(rows)
    df.to_csv(D / "processed" / "nisar_threshold_calibration.csv", index=False)
    mean = df.groupby("drop_db")[["cover_pct", "precision_pct", "iou_pct", "false_alarm_pct"]].mean().round(1)
    best = float(mean.iou_pct.idxmax())
    (D / "processed" / "nisar_threshold_calibration.json").write_text(json.dumps(
        {"chosen_drop_db": best, "criterion": "best mean IoU over the 3 same-track pairs vs WFIGS perimeter",
         "smooth_cells": SCAR_SMOOTH_CELLS, "res_m": RES, "pairs": list(df.pair.unique()),
         "mean_by_threshold": mean.reset_index().to_dict("records")}, indent=1))
    pd.set_option("display.width", 220)
    print(df.to_string(index=False)); print(mean.to_string()); print("chosen drop_db:", best)


if __name__ == "__main__":
    main()
