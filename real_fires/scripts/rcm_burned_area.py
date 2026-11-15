"""RCM C-band burned-area prototype on the Bald Range fire (BC, 2026), scored against the BCWS perimeter.

Same beam and geometry (SC30MCPA, ascending ~01:34 UTC = 18:34 PDT the previous evening):
reference 08-02 (5.4 days before ignition) vs 08-10 (2.1 days after) and 08-22 (14.1 days after).
Only a window around the fire is read from the public RCM CEOS-ARD bucket; clips are cached in
data/fires/bald_range/raw/rcm/ (not committed).

Rule: RL (cross-pol) change averaged over SMOOTH x SMOOTH pixels; burned where RL ROSE by >= RISE_DB.
Why an increase and not a drop: see data/fires/bald_range/README.md (C-band brightening of fire scars).
RISE_DB is fixed beforehand; the sweep is reported for transparency, not used to pick the result.

Radiometric check: each later scene is quantile-mapped (in dB) onto the reference using only unburned
reference pixels (outside the official perimeter + 2 km). On consistent scenes this is ~identity; the raw
offset is reported so a broken scene (like SC30MCPD 08-09, +11.9 dB) is visible.

Run from real_fires/:  python scripts/rcm_burned_area.py
"""
from __future__ import annotations

import json

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from firedata import bcws, rcm
from firedata.config import REPO_ROOT
from firedata.nisar_provider import hv_change_db

FIRE_NUMBER, YEAR = "K51490", 2026
CRS = 32611                      # UTM 11N, native grid of these RCM tiles
BUFFER_M = 3000
REF_BUFFER_M = 2000              # reference (unburned) pixels lie this far outside the perimeter
SMOOTH = 5                       # 5 x 5 pixels of 20 m = 100 m
RISE_DB = 1.0                    # fixed beforehand (see README); literature boreal scars: +3..6 dB when wet
SWEEP_DB = [0.5, 1.0, 1.5, 2.0, 3.0]
BEAM = "SC30MCPA"
# SC30MCPC (08-06 / 08-18 / 08-30) was the first choice but its data swath misses the fire (0-0.1 % valid
# pixels in the window, although the product footprint polygon contains the origin), so SC30MCPA is used.
REF = ["RCM1_OK4114697_PK4293771_1_SC30MCPA_20260802_013425_CH_CV_MLC"]
POST = [["RCM3_OK4114697_PK4308170_1_SC30MCPA_20260810_013442_CH_CV_MLC"],
        ["RCM3_OK4114697_PK4329615_1_SC30MCPA_20260822_013440_CH_CV_MLC"]]
D = REPO_ROOT / "data" / "fires" / "bald_range"


def when(scene: str) -> pd.Timestamp:
    p = scene.split("_")
    return pd.Timestamp(f"{p[5]}T{p[6]}", tz="UTC")


def quantile_map_db(x: np.ndarray, ref: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Map x (linear) onto ref's distribution, in dB, using the pixels in `sel` to fit the mapping."""
    q = np.linspace(0.5, 99.5, 199)
    xs, rs = np.nanpercentile(10 * np.log10(x[sel]), q), np.nanpercentile(10 * np.log10(ref[sel]), q)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10 ** (np.interp(10 * np.log10(x), xs, rs) / 10)


def auc_rise(inside: np.ndarray, outside: np.ndarray) -> float:
    """P(random inside pixel rose more than a random outside pixel); 0.5 = no separation, 1 = perfect."""
    a, b = inside[np.isfinite(inside)], outside[np.isfinite(outside)]
    r = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def main():
    raw, proc, figs = (D / k for k in ("raw", "processed", "figures"))
    for p in (proc, figs):
        p.mkdir(parents=True, exist_ok=True)
    fire = bcws.download_fire(FIRE_NUMBER, YEAR, raw)
    fire.to_file(proc / "bcws_final_perimeter.geojson", driver="GeoJSON")
    ignition = pd.Timestamp(fire.attr_FireDiscoveryDateTime.iloc[0])
    f = fire.to_crs(CRS)
    transform, shape = rcm.grid(tuple(f.buffer(BUFFER_M).total_bounds))
    px_m2 = transform.a ** 2
    W = rasterize([f.geometry.iloc[0]], out_shape=shape, transform=transform).astype(bool)
    near = rasterize([f.buffer(REF_BUFFER_M).iloc[0]], out_shape=shape, transform=transform).astype(bool)

    def load(frames):
        """Mosaic adjacent frames of one pass: first frame with a valid pixel (bitmask == 1) wins."""
        rl, rr = np.full(shape, np.nan, np.float32), np.full(shape, np.nan, np.float32)
        for s in frames:
            a = {k: rcm.read_clipped(s, k, CRS, transform, shape, raw / "rcm") for k in ("RL", "RR", "bitmask")}
            ok = (a["bitmask"] == 1) & np.isfinite(a["RL"]) & np.isfinite(a["RR"]) & np.isnan(rl)
            rl[ok], rr[ok] = a["RL"][ok], a["RR"][ok]
        return rl, rr, np.isfinite(rl)

    rl0, rr0, ok0 = load(REF)
    rows, diag, burned_by_date = [], [], {}
    for frames in POST:
        rl, rr, ok = load(frames)
        valid = ok0 & ok
        sel = valid & ~near
        offset = float(np.nanmedian(10 * np.log10(rl[sel] / rl0[sel])))
        d_rl = hv_change_db(quantile_map_db(rl, rl0, sel), rl0, SMOOTH)
        d_rr = hv_change_db(quantile_map_db(rr, rr0, sel), rr0, SMOOTH)
        d_ratio = d_rl - d_rr
        valid &= np.isfinite(d_rl)
        t = when(frames[0])
        h = (t - ignition).total_seconds() / 3600
        feats = (("RL", d_rl), ("RR", d_rr), ("RLRR", d_ratio))
        diag.append({"date": f"{t:%Y-%m-%d}", "raw_RL_offset_outside_db": round(offset, 2),
                     **{f"median_d{k}_in_db": round(float(np.nanmedian(v[W & valid])), 2) for k, v in feats},
                     **{f"median_d{k}_out_db": round(float(np.nanmedian(v[sel & valid])), 2) for k, v in feats},
                     **{f"auc_rise_d{k}": round(auc_rise(v[W & valid], v[sel & valid]), 3) for k, v in feats}})
        for thr in SWEEP_DB:
            b = np.nan_to_num(d_rl, nan=0.0) >= thr
            rows.append({"date": f"{t:%Y-%m-%d %H:%M}", "hours_after_ignition": round(h, 1), "rise_db": thr,
                         **rcm.scores(b, W, valid, px_m2)})
            if thr == RISE_DB:
                burned_by_date[t] = (b & valid, d_rl, valid)

    df = pd.DataFrame(rows)
    df.to_csv(proc / "rcm_burned_scores.csv", index=False)
    pd.DataFrame(diag).to_csv(proc / "rcm_change_diagnostics.csv", index=False)

    # Growth map: first date from which a pixel stays burned on every later date (persistence cuts speckle).
    dates = sorted(burned_by_date)
    first = np.zeros(shape, np.uint8)
    for i, t in enumerate(dates):
        stays = np.logical_and.reduce([burned_by_date[u][0] for u in dates[i:]])
        first[(first == 0) & stays] = i + 1
    with rasterio.open(proc / "rcm_first_burned.tif", "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=1, dtype="uint8", crs=CRS, transform=transform, compress="deflate", nodata=0) as d:
        d.write(first, 1)
        d.update_tags(values=", ".join(f"{i + 1}={t:%Y-%m-%d}" for i, t in enumerate(dates)),
                      rule=f"RL rise >= {RISE_DB} dB, {SMOOTH}x{SMOOTH} px, persistent to {dates[-1]:%Y-%m-%d}")
    valid_all = np.logical_and.reduce([burned_by_date[t][2] for t in dates])
    growth = [{"date": f"{t:%Y-%m-%d}", **rcm.scores((first > 0) & (first <= i + 1), W, valid_all, px_m2)}
              for i, t in enumerate(dates)]
    pd.DataFrame(growth).to_csv(proc / "rcm_growth_persistent.csv", index=False)
    (proc / "rcm_burned_summary.json").write_text(json.dumps(
        {"fire": "Bald Range (BCWS K51490)", "official_ha": float(fire.bcws_hectares.iloc[0]), "beam": BEAM,
         "reference_scenes": REF, "post_scenes": POST,
         "rule": f"RL rise >= {RISE_DB} dB, {SMOOTH}x{SMOOTH} px (100 m)",
         "per_date": df[df.rise_db == RISE_DB].to_dict("records"), "persistent_growth": growth,
         "diagnostics": diag}, indent=1, default=str))

    # Figure: RL change per date + growth map, BCWS perimeter in black.
    ext = (transform.c, transform.c + shape[1] * transform.a, transform.f + shape[0] * transform.e, transform.f)
    ign = gpd.GeoSeries(gpd.points_from_xy(fire.attr_InitialLongitude, fire.attr_InitialLatitude), crs=4326).to_crs(CRS)
    n = len(dates)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1) + 1, 5.6), constrained_layout=True)
    for ax, t in zip(axes[:n], dates):
        b, d_rl, valid = burned_by_date[t]
        im = ax.imshow(np.where(valid, d_rl, np.nan), cmap="RdBu", vmin=-4, vmax=4, extent=ext)
        r = df[(df.date.str.startswith(f"{t:%Y-%m-%d}")) & (df.rise_db == RISE_DB)].iloc[0]
        a = next(x for x in diag if x["date"] == f"{t:%Y-%m-%d}")
        ax.set_title(f"RL change {t:%m-%d %H:%M} UTC vs 08-02 ({r.hours_after_ignition / 24:.1f} d after ignition)\n"
                     f"rise ≥ {RISE_DB} dB: {r.detected_ha:,.0f} ha · cover {r.cover_pct:.0f}% · precision "
                     f"{r.precision_pct:.0f}% · IoU {r.iou_pct:.0f}% · AUC {a['auc_rise_dRL']:.2f}", fontsize=9)
    fig.colorbar(im, ax=axes[:n], shrink=0.8, label=f"RL change (dB, {SMOOTH}×{SMOOTH} px mean; blue = brighter)")
    cols = ["#f98e09", "#bc3754", "#57106e"][:n]
    axes[n].imshow(np.where(first > 0, first, np.nan), cmap=ListedColormap(cols), vmin=0.5, vmax=n + 0.5,
                   extent=ext, interpolation="nearest")
    for c, g in zip(cols, growth):
        axes[n].plot([], [], "s", color=c, label=f"by {g['date'][5:]}: {g['detected_inside_ha']:,.0f} ha inside "
                                                  f"({g['cover_pct']:.0f}% cover, {g['precision_pct']:.0f}% precision)")
    axes[n].legend(loc="lower left", fontsize=8)
    axes[n].set_title(f"Burned area growth (RL rise ≥ {RISE_DB} dB, persistent)\nBCWS final perimeter "
                      f"{fire.bcws_hectares.iloc[0]:,.0f} ha", fontsize=9)
    for ax in axes:
        f.boundary.plot(ax=ax, color="k", lw=0.8)
        ign.plot(ax=ax, marker="*", color="k", markersize=120)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Bald Range fire (BC, 2026): RCM C-band compact-pol ({BEAM}, 30 m ARD) burned area vs BCWS "
                 f"official perimeter. Ignition {ignition:%Y-%m-%d %H:%M} UTC (★)", fontsize=11)
    out = figs / "rcm_burned_area_growth.png"
    fig.savefig(out, dpi=100)
    print(df.to_string(index=False)); print(pd.DataFrame(diag).T.to_string()); print(pd.DataFrame(growth).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
