"""NISAR L-band vs RCM C-band burned area on the Bald Range fire, both scored against the BCWS perimeter.

NISAR: same-track GCOV pairs, existing burn-scar rule unchanged (nisar_provider: HV drop >= SCAR_DROP_DB
after SCAR_SMOOTH_CELLS x SCAR_SMOOTH_CELLS averaging at 100 m). Only the fire window is read from ASF.
RCM: SC30MCPA RL rise >= RISE_DB (scripts/rcm_burned_area.py), sampled onto the same 100 m grid
(cell centre of each 5 x 5 block of the 20 m grid; the RCM change is already a 5 x 5 mean).
Combinations: NISAR drop OR / AND RCM rise, pairing each NISAR date with the nearest RCM date.

Run from real_fires/ after scripts/rcm_burned_area.py:  python scripts/nisar_vs_rcm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from rasterio.features import rasterize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcm_burned_area as R                                   # noqa: E402  (same folder)
from firedata import bcws, rcm                                # noqa: E402
from firedata.nisar_access import GCOV_GROUP, TargetGrid, group_passes, read_pass, search_granules  # noqa: E402
from firedata.nisar_provider import SCAR_DROP_DB, SCAR_SMOOTH_CELLS, _gcov_valid, hv_change_db  # noqa: E402

# Same-track pairs (ASF search, 2026): track -> (reference, [later passes]); ignition 08-07 23:20 UTC.
PAIRS = {"A5": ("07-20", ["08-13", "08-25"]), "A77": ("07-25", ["08-18", "08-30"]),
         "D71": ("07-25", ["08-18", "08-30"]), "D172": ("07-20", ["08-13", "08-25"])}
CELL = 5                                   # 5 x 20 m = 100 m


def main():
    proc, figs = R.D / "processed", R.D / "figures"
    fire = bcws.download_fire(R.FIRE_NUMBER, R.YEAR, R.D / "raw")
    ignition = pd.Timestamp(fire.attr_FireDiscoveryDateTime.iloc[0])
    f = fire.to_crs(R.CRS)
    tr20, sh20 = rcm.grid(tuple(f.buffer(R.BUFFER_M).total_bounds))
    ny, nx = sh20[0] // CELL, sh20[1] // CELL
    x = tr20.c + (np.arange(nx) + 0.5) * 100.0
    y = tr20.f - (np.arange(ny) + 0.5) * 100.0
    grid = TargetGrid(f"EPSG:{R.CRS}", x, y, 100.0)
    from rasterio.transform import from_origin
    tr100 = from_origin(tr20.c, tr20.f, 100.0, 100.0)
    W = rasterize([f.geometry.iloc[0]], out_shape=(ny, nx), transform=tr100).astype(bool)
    ref_out = ~rasterize([f.buffer(R.REF_BUFFER_M).iloc[0]], out_shape=(ny, nx), transform=tr100).astype(bool)

    # --- RCM change on the 100 m grid (centre pixel of each 5 x 5 block of the 5 x 5-smoothed 20 m change)
    def rcm_load(frames):
        a = {k: rcm.read_clipped(frames[0], k, R.CRS, tr20, sh20, R.D / "raw" / "rcm") for k in ("RL", "bitmask")}
        a["RL"][a["bitmask"] != 1] = np.nan
        return a["RL"]
    near20 = rasterize([f.buffer(R.REF_BUFFER_M).iloc[0]], out_shape=sh20, transform=tr20).astype(bool)
    rl0 = rcm_load(R.REF)
    rcm_d = {}
    for frames in R.POST:
        rl = rcm_load(frames)
        sel = np.isfinite(rl) & np.isfinite(rl0) & ~near20
        d = hv_change_db(R.quantile_map_db(rl, rl0, sel), rl0, R.SMOOTH)
        rcm_d[R.when(frames[0])] = d[CELL // 2::CELL, CELL // 2::CELL][:ny, :nx]

    # --- NISAR same-track pairs
    bbox = tuple(fire.buffer(0.05).total_bounds)
    passes = group_passes(search_granules("GCOV", bbox, "2026-07-15", "2026-09-05"))
    def find(track, mmdd):
        hits = [q for q in passes if q.track_id == track and f"{q.time_utc:%m-%d}" == mmdd]
        if len(hits) != 1:
            raise RuntimeError(f"{track} {mmdd}: {len(hits)} passes; track ids: "
                               f"{sorted({(q.track_id, f'{q.time_utc:%m-%d}') for q in passes})}")
        return hits[0]
    nisar_d = {}
    for track, (ref, posts) in PAIRS.items():
        hv0 = read_pass(find(track, ref), GCOV_GROUP, ["HVHV", "mask"], grid, prepare=_gcov_valid)["HVHV"]
        for mm in posts:
            p = find(track, mm)
            hv = read_pass(p, GCOV_GROUP, ["HVHV", "mask"], grid, prepare=_gcov_valid)["HVHV"]
            nisar_d[(track, ref, p.time_utc)] = hv_change_db(hv, hv0, SCAR_SMOOTH_CELLS)

    # --- Scores on the common 100 m grid
    rows = []
    def add(sensor, label, t, burned, valid, auc):
        rows.append({"sensor": sensor, "pair": label, "date": f"{t:%Y-%m-%d %H:%M}",
                     "days_after_ignition": round((t - ignition).total_seconds() / 86400, 1),
                     "auc": None if auc is None else round(auc, 3), **rcm.scores(burned, W, valid, 1e4)})
    for t, d in rcm_d.items():
        v = np.isfinite(d)
        add("RCM C-band RL rise", f"SC30MCPA 08-02 → {t:%m-%d}", t, np.nan_to_num(d) >= R.RISE_DB, v,
            R.auc_rise(d[W & v], d[ref_out & v]))
    for (track, ref, t), d in nisar_d.items():
        v = np.isfinite(d)
        nb = np.nan_to_num(d) <= -SCAR_DROP_DB
        add("NISAR L-band HV drop", f"{track} {ref} → {t:%m-%d}", t, nb, v, R.auc_rise(-d[W & v], -d[ref_out & v]))
        tr = min(rcm_d, key=lambda u: abs(u - t))
        r = rcm_d[tr]
        v2 = v & np.isfinite(r)
        rb = np.nan_to_num(r) >= R.RISE_DB
        add("NISAR drop OR RCM rise", f"{track} {t:%m-%d} + RCM {tr:%m-%d}", t, nb | rb, v2, None)
        add("NISAR drop AND RCM rise", f"{track} {t:%m-%d} + RCM {tr:%m-%d}", t, nb & rb, v2, None)
    df = pd.DataFrame(rows)
    df.to_csv(proc / "nisar_vs_rcm_scores.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["detected_inside_ha"]).to_string(index=False))

    # --- Figure: NISAR change | RCM change | BCWS perimeter (best-AUC NISAR pair after growth vs nearest RCM)
    nis = df[df.sensor.str.startswith("NISAR L")]
    best = nis.loc[nis.auc.idxmax()]
    key = next(k for k in nisar_d if f"{k[0]} {k[1]} → {k[2]:%m-%d}" == best.pair)
    tr_ = min(rcm_d, key=lambda u: abs(u - key[2]))
    rc = df[(df.sensor == "RCM C-band RL rise") & df.pair.str.endswith(f"{tr_:%m-%d}")].iloc[0]
    ext = (tr100.c, tr100.c + nx * 100, tr100.f - ny * 100, tr100.f)
    ign = gpd.GeoSeries(gpd.points_from_xy(fire.attr_InitialLongitude, fire.attr_InitialLatitude), crs=4326).to_crs(R.CRS)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.8), constrained_layout=True)
    im0 = ax[0].imshow(nisar_d[key], cmap="RdBu", vmin=-4, vmax=4, extent=ext)
    ax[0].set_title(f"NISAR L-band HV change, {best.pair} ({best.days_after_ignition:.0f} d after ignition)\n"
                    f"rule: drop ≥ {SCAR_DROP_DB} dB · cover {best.cover_pct:.0f}% · precision {best.precision_pct:.0f}% · "
                    f"IoU {best.iou_pct:.0f}% · AUC(drop) {best.auc:.2f}", fontsize=9)
    ax[1].imshow(rcm_d[tr_], cmap="RdBu", vmin=-4, vmax=4, extent=ext)
    ax[1].set_title(f"RCM C-band RL change, {rc.pair} ({rc.days_after_ignition:.0f} d after ignition)\n"
                    f"rule: rise ≥ {R.RISE_DB} dB · cover {rc.cover_pct:.0f}% · precision {rc.precision_pct:.0f}% · "
                    f"IoU {rc.iou_pct:.0f}% · AUC(rise) {rc.auc:.2f}", fontsize=9)
    ax[2].imshow(np.where(W, 1.0, np.nan), cmap="Greys", vmin=0, vmax=2, extent=ext)
    ax[2].set_title(f"BCWS official final perimeter (K51490)\n{fire.bcws_hectares.iloc[0]:,.0f} ha, "
                    f"dated {pd.Timestamp(fire.poly_PolygonDateTime.iloc[0]):%Y-%m-%d}", fontsize=9)
    fig.colorbar(im0, ax=ax[:2], shrink=0.8, label="change (dB; red = darker, blue = brighter)")
    for a in ax:
        f.boundary.plot(ax=a, color="k", lw=0.8)
        ign.plot(ax=a, marker="*", color="k", markersize=110)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Bald Range fire (BC, 2026): NISAR L-band vs RCM C-band change, 100 m grid, vs BCWS perimeter "
                 f"(★ ignition {ignition:%Y-%m-%d %H:%M} UTC)", fontsize=11)
    out = figs / "nisar_vs_rcm.png"
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    main()
