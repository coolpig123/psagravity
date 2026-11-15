"""RCM C-band burned-area prototype on the Bald Range fire (BC, 2026), scored against the BCWS perimeter.

Same beam and geometry (SC30MCPD, descending ~14:08 UTC): reference 08-01 (6 days before ignition)
vs 08-09 (~39 h after ignition), 08-25 and 09-02. Only a window around the fire is read from the public
RCM CEOS-ARD bucket; clips are cached in data/fires/bald_range/raw/rcm/ (not committed).

Rule (fixed beforehand, same form as the NISAR burn-scar rule in nisar_provider.py): RL (cross-pol) change
averaged over SMOOTH x SMOOTH pixels, burned where the drop is >= DROP_DB. A sweep of thresholds is reported
for transparency, not used to pick the result.

Radiometric normalisation: each later scene is quantile-mapped (in dB) onto the reference scene using only
unburned reference pixels (outside the official perimeter + 2 km). On consistent scenes this is ~identity;
it is needed for 08-09, whose RL/RR are ~11 dB higher everywhere with a compressed range (calibration issue).

Run from real_fires/:  python scripts/rcm_burned_area.py
"""
from __future__ import annotations

import json

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
DROP_DB = 1.5
SWEEP_DB = [1.0, 1.5, 2.0, 3.0]
REF = "RCM2_OK4114697_PK4292834_1_SC30MCPD_20260801_140821_CH_CV_MLC"
POST = ["RCM1_OK4114697_PK4307194_1_SC30MCPD_20260809_140811_CH_CV_MLC",
        "RCM2_OK4114697_PK4335941_1_SC30MCPD_20260825_140819_CH_CV_MLC",
        "RCM1_OK4114697_PK4349967_1_SC30MCPD_20260902_140809_CH_CV_MLC"]
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


def auc(inside: np.ndarray, outside: np.ndarray) -> float:
    """P(random inside pixel has a larger *drop* than a random outside pixel); 0.5 = no separation."""
    a, b = -inside[np.isfinite(inside)], -outside[np.isfinite(outside)]
    r = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def main():
    raw, proc, figs = (D / k for k in ("raw", "processed", "figures"))
    for p in (proc, figs):
        p.mkdir(parents=True, exist_ok=True)
    fire = bcws.download_fire(FIRE_NUMBER, YEAR, raw)
    fire.to_file(proc / "bcws_final_perimeter.geojson", driver="GeoJSON")
    f = fire.to_crs(CRS)
    transform, shape = rcm.grid(tuple(f.buffer(BUFFER_M).total_bounds))
    px_m2 = transform.a ** 2
    W = rasterize([f.geometry.iloc[0]], out_shape=shape, transform=transform).astype(bool)
    near = rasterize([f.buffer(REF_BUFFER_M).iloc[0]], out_shape=shape, transform=transform).astype(bool)

    def load(scene):
        a = {k: rcm.read_clipped(scene, k, CRS, transform, shape, raw / "rcm") for k in ("RL", "RR", "bitmask")}
        ok = (a["bitmask"] == 1) & np.isfinite(a["RL"]) & np.isfinite(a["RR"])
        return a["RL"], a["RR"], ok

    rl0, rr0, ok0 = load(REF)
    rows, diag, burned_by_date = [], [], {}
    for scene in POST:
        rl, rr, ok = load(scene)
        valid = ok0 & ok
        sel = valid & ~near
        offset = float(np.nanmedian(10 * np.log10(rl[sel] / rl0[sel])))
        rl_n, rr_n = quantile_map_db(rl, rl0, sel), quantile_map_db(rr, rr0, sel)
        d_rl = hv_change_db(rl_n, rl0, SMOOTH)
        d_rr = hv_change_db(rr_n, rr0, SMOOTH)
        d_ratio = d_rl - d_rr
        valid &= np.isfinite(d_rl)
        t = when(scene)
        h = (t - pd.Timestamp(fire.attr_FireDiscoveryDateTime.iloc[0])).total_seconds() / 3600
        diag.append({"date": f"{t:%Y-%m-%d}", "raw_RL_offset_outside_db": round(offset, 2),
                     **{f"median_d{k}_in_db": round(float(np.nanmedian(v[W & valid])), 2) for k, v in
                        (("RL", d_rl), ("RR", d_rr), ("RLRR", d_ratio))},
                     **{f"median_d{k}_out_db": round(float(np.nanmedian(v[sel & valid])), 2) for k, v in
                        (("RL", d_rl), ("RR", d_rr), ("RLRR", d_ratio))},
                     **{f"auc_d{k}": round(auc(v[W & valid], v[sel & valid]), 3) for k, v in
                        (("RL", d_rl), ("RR", d_rr), ("RLRR", d_ratio))}})
        for thr in SWEEP_DB:
            b = np.nan_to_num(d_rl, nan=0.0) <= -thr
            rows.append({"date": f"{t:%Y-%m-%d %H:%M}", "hours_after_ignition": round(h, 1), "drop_db": thr,
                         **rcm.scores(b, W, valid, px_m2)})
            if thr == DROP_DB:
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
                      rule=f"RL drop >= {DROP_DB} dB, {SMOOTH}x{SMOOTH} px, persistent to {dates[-1]:%Y-%m-%d}")
    valid_all = np.logical_and.reduce([burned_by_date[t][2] for t in dates])
    growth = [{"date": f"{t:%Y-%m-%d}", **rcm.scores((first > 0) & (first <= i + 1), W, valid_all, px_m2)}
              for i, t in enumerate(dates)]
    pd.DataFrame(growth).to_csv(proc / "rcm_growth_persistent.csv", index=False)
    (proc / "rcm_burned_summary.json").write_text(json.dumps(
        {"fire": "Bald Range (BCWS K51490)", "official_ha": float(fire.bcws_hectares.iloc[0]),
         "reference_scene": REF, "post_scenes": POST, "rule": f"RL drop >= {DROP_DB} dB, {SMOOTH}x{SMOOTH} px (100 m)",
         "per_date": df[df.drop_db == DROP_DB].to_dict("records"), "persistent_growth": growth,
         "diagnostics": diag}, indent=1, default=str))

    # Figure: RL change per date + growth map, BCWS perimeter in black.
    ext = (transform.c, transform.c + shape[1] * transform.a, transform.f + shape[0] * transform.e, transform.f)
    ign = fire.set_geometry(pd.Series(
        __import__("geopandas").points_from_xy(fire.attr_InitialLongitude, fire.attr_InitialLatitude)), crs=4326).to_crs(CRS)
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4), constrained_layout=True)
    for ax, t in zip(axes[:3], dates):
        b, d_rl, valid = burned_by_date[t]
        im = ax.imshow(np.where(valid, d_rl, np.nan), cmap="RdBu", vmin=-4, vmax=4, extent=ext)
        r = df[(df.date.str.startswith(f"{t:%Y-%m-%d}")) & (df.drop_db == DROP_DB)].iloc[0]
        ax.set_title(f"RL change {t:%m-%d %H:%M} UTC vs 08-01 ({r.hours_after_ignition:.0f} h after ignition)\n"
                     f"detected {r.detected_ha:,.0f} ha · cover {r.cover_pct:.0f}% · precision {r.precision_pct:.0f}% "
                     f"· IoU {r.iou_pct:.0f}%", fontsize=9)
    fig.colorbar(im, ax=axes[:3], shrink=0.8, label=f"RL change (dB, {SMOOTH}×{SMOOTH} px mean)")
    cols = ["#fde725", "#f98e09", "#bc3754"][:len(dates)]
    axes[3].imshow(np.where(first > 0, first, np.nan), cmap=ListedColormap(cols), vmin=0.5, vmax=len(dates) + 0.5,
                   extent=ext, interpolation="nearest")
    for c, g in zip(cols, growth):
        axes[3].plot([], [], "s", color=c, label=f"by {g['date'][5:]}: {g['detected_inside_ha']:,.0f} ha inside "
                                                  f"({g['cover_pct']:.0f}% cover)")
    axes[3].legend(loc="lower left", fontsize=8)
    axes[3].set_title(f"Burned area growth (RL drop ≥ {DROP_DB} dB, persistent)\nBCWS final perimeter "
                      f"{fire.bcws_hectares.iloc[0]:,.0f} ha", fontsize=9)
    for ax in axes:
        f.boundary.plot(ax=ax, color="k", lw=0.8)
        ign.plot(ax=ax, marker="*", color="k", markersize=120)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Bald Range fire (BC, 2026): RCM C-band compact-pol (SC30MCPD, 30 m ARD) burned area vs BCWS "
                 "official perimeter. Ignition 2026-08-07 23:20 UTC (★)", fontsize=11)
    out = figs / "rcm_burned_area_growth.png"
    fig.savefig(out, dpi=110)
    print(df.to_string(index=False)); print(pd.DataFrame(diag).T.to_string()); print(pd.DataFrame(growth).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
