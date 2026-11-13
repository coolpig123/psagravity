"""Standard ground-truth figures for one fire."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from .config import FireConfig

matplotlib.use("Agg")

# Reference palette (dataviz skill): categorical slots in fixed order, one-hue sequential ramps.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"GOES18_FDC": "#2a78d6", "VIIRS_SNPP": "#eb6834", "VIIRS_NOAA20": "#1baf7a",
          "VIIRS_NOAA21": "#eda100"}
TIME_CMAP = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281", "#0d366b"])
FRP_CMAP = LinearSegmentedColormap.from_list("orange_seq", ["#fbd9c8", "#f3a07a", "#eb6834", "#b8461b", "#6e2508"])
NISAR_COLOR = "#6250d6"
EQUAL_AREA = 5070

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def _basemap(ax, crs):
    try:
        import contextily as cx
        cx.add_basemap(ax, crs=crs, source=cx.providers.Esri.WorldShadedRelief, attribution_size=6)
    except Exception as e:  # offline: plot without tiles
        print(f"  (no basemap: {e.__class__.__name__})")


def _nisar_lines(ax, F: FireConfig, xlim):
    for kind, passes in F.nisar_passes_utc.items():
        for t in passes:
            if xlim[0] <= t <= xlim[1]:
                ax.axvline(t, color=NISAR_COLOR, lw=1, ls=(0, (4, 3)), zorder=1)
                ax.text(t, 1.0, f" NISAR {t:%m-%d %H:%M}", transform=ax.get_xaxis_transform(),
                        rotation=90, va="top", ha="right", fontsize=7, color=NISAR_COLOR)


def _event_lines(ax, events: dict[str, pd.Timestamp]):
    for label, t in events.items():
        ax.axvline(t, color=MUTED, lw=1, zorder=1)
        ax.text(t, 1.0, f" {label}", transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="right", fontsize=7, color=INK2)


def _time_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel("Time (UTC)")


def perimeter_map(F: FireConfig, perims: gpd.GeoDataFrame, final: gpd.GeoDataFrame, out: Path):
    p = perims[perims.is_new].to_crs(3857)
    f = final.to_crs(3857)
    t0 = p.obs_time_utc.min().floor("D")
    t1 = max(p.obs_time_utc.max(), max(F.nisar_passes_utc.get("post", [t0]))).ceil("D")
    norm = matplotlib.colors.Normalize(mdates.date2num(t0), mdates.date2num(t1))

    fig, ax = plt.subplots(figsize=(8, 7.2))
    ax.grid(False)
    for _, r in p.sort_values("obs_time_utc", ascending=False).iterrows():  # newest (largest) first
        gpd.GeoSeries([r.geometry], crs=3857).plot(
            ax=ax, color=TIME_CMAP(norm(mdates.date2num(r.obs_time_utc))), edgecolor=SURFACE, lw=0.6, zorder=2)
    f.boundary.plot(ax=ax, color=INK, lw=1.1, ls="--", zorder=3)
    ig = gpd.GeoSeries(gpd.points_from_xy([F.ignition_lon], [F.ignition_lat]), crs=4326).to_crs(3857)
    ax.scatter(ig.x, ig.y, marker="*", s=180, color="#e34948", edgecolor=SURFACE, lw=1, zorder=5)
    pad = 2500
    x0, y0, x1, y1 = f.total_bounds
    ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
    _basemap(ax, 3857)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=TIME_CMAP)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03)
    cb.ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    cb.ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    cb.set_label("FEDS perimeter observation time (UTC); purple ticks = NISAR post-fire passes", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    for t in F.nisar_passes_utc.get("post", []):
        cb.ax.axvline(mdates.date2num(t), color=NISAR_COLOR, lw=2)

    last = p.iloc[p.obs_time_utc.argmax()]
    pre = ", ".join(f"{t:%m-%d %H:%M}" for t in F.nisar_passes_utc.get("pre", []))
    post = ", ".join(f"{t:%m-%d %H:%M}" for t in F.nisar_passes_utc.get("post", []))
    ax.set_title(f"{F.name}: FEDS perimeters every 12 h")
    ax.text(0.06, 0.0,
            f"Dashed: NIFC WFIGS final perimeter ({final.poly_GISAcres.iloc[0]:,.0f} acres)\n"
            f"Star: ignition point\n"
            f"NISAR pre: {pre}  |  post: {post} UTC\n"
            f"Last FEDS growth observed {last.obs_time_utc:%m-%d %H:%M} UTC; all post passes see this perimeter",
            transform=fig.transFigure, fontsize=7.5, color=INK2, va="top")
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def burned_area(F: FireConfig, perims: gpd.GeoDataFrame, final: gpd.GeoDataFrame, out: Path):
    p = perims.sort_values("obs_time_utc")
    xlim = (F.start_utc, F.end_utc)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.step(p.obs_time_utc, p.area_acres, where="post", color=SERIES["GOES18_FDC"], lw=2, zorder=3)
    new = p[p.is_new]
    ax.scatter(new.obs_time_utc, new.area_acres, s=40, color=SERIES["GOES18_FDC"], edgecolor=SURFACE,
               lw=2, zorder=4, label="FEDS perimeter (new observation)")
    rep = p[~p.is_new]
    ax.scatter(rep.obs_time_utc, rep.area_acres, s=40, facecolor=SURFACE, edgecolor=SERIES["GOES18_FDC"],
               lw=1.5, zorder=4, label="FEDS perimeter (repeated, no growth)")
    final_acres = final.poly_GISAcres.iloc[0]
    ax.axhline(final_acres, color=INK, lw=1, ls="--")
    ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.94, final_acres, f"NIFC WFIGS final: {final_acres:,.0f} acres ",
            va="bottom", ha="right", fontsize=8, color=INK)
    ax.text(p.obs_time_utc.iloc[-1], new.area_acres.max(), f"  FEDS max {new.area_acres.max():,.0f} acres",
            va="center", ha="left", fontsize=8, color=INK2)
    events = {"discovery": final.attr_FireDiscoveryDateTime.iloc[0],
              "100% contained": final.attr_ContainmentDateTime.iloc[0]}
    _event_lines(ax, {k: v for k, v in events.items() if pd.notna(v)})
    _nisar_lines(ax, F, xlim)
    ax.set_xlim(*xlim); ax.set_ylim(0, final_acres * 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("Burned area (acres)")
    _time_axis(ax)
    ax.set_title(f"{F.name}: burned area vs time")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=8)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def frp_timeseries(F: FireConfig, frp: pd.DataFrame, final: gpd.GeoDataFrame, out: Path):
    xlim = (F.start_utc, F.end_utc)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for src in SERIES:
        s = frp[frp.source == src]
        if s.empty:
            continue
        if src.startswith("GOES"):
            ax.plot(s.time_utc, s.frp_total_mw, color=SERIES[src], lw=1, zorder=2,
                    label=f"GOES-18 FDC (every 5 min, 2 km)")
        else:
            ax.scatter(s.time_utc, s.frp_total_mw, s=48, color=SERIES[src], edgecolor=SURFACE, lw=1.5, zorder=4,
                       label=f"VIIRS {src.split('_')[1]} overpass (375 m)")
    _event_lines(ax, {"discovery": final.attr_FireDiscoveryDateTime.iloc[0]})
    _nisar_lines(ax, F, xlim)
    ax.set_xlim(*xlim); ax.set_ylim(bottom=0)
    ax.set_ylabel("Total fire radiative power (MW)")
    _time_axis(ax)
    ax.set_title(f"{F.name}: total FRP per scan / overpass")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def viirs_frp_map(F: FireConfig, points: pd.DataFrame, final: gpd.GeoDataFrame, out: Path):
    v = points[points.source.str.startswith("VIIRS") & (points.frp_mw > 0)].sort_values("frp_mw")
    g = gpd.GeoDataFrame(v, geometry=gpd.points_from_xy(v.lon, v.lat), crs=4326).to_crs(3857)
    f = final.to_crs(3857)
    fig, ax = plt.subplots(figsize=(8, 7.2))
    ax.grid(False)
    f.boundary.plot(ax=ax, color=INK, lw=1, ls="--", zorder=2)
    norm = LogNorm(vmin=max(v.frp_mw.min(), 0.5), vmax=v.frp_mw.max())
    sc = ax.scatter(g.geometry.x, g.geometry.y, c=v.frp_mw, cmap=FRP_CMAP, norm=norm, s=9,
                    edgecolor="none", zorder=3)
    pad = 2500
    x0, y0, x1, y1 = f.total_bounds
    ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
    _basemap(ax, 3857)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03)
    cb.set_label("VIIRS fire radiative power per 375 m pixel (MW, log scale)", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    ax.set_title(f"{F.name}: VIIRS 375 m detections ({len(v):,} points, S-NPP + NOAA-20 + NOAA-21)")
    ax.text(0.06, 0.0, "Dashed: NIFC WFIGS final perimeter. Darker = higher FRP; high-FRP points drawn on top.",
            transform=fig.transFigure, fontsize=7.5, color=INK2, va="top")
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def make_all(F: FireConfig, points, perims, final, frp, fig_dir: Path):
    fig_dir = Path(fig_dir)
    perimeter_map(F, perims, final, fig_dir / "perimeters_12h_map.png")
    burned_area(F, perims, final, fig_dir / "burned_area_vs_time.png")
    frp_timeseries(F, frp, final, fig_dir / "frp_vs_time.png")
    viirs_frp_map(F, points, final, fig_dir / "viirs_frp_map.png")
    print(f"  figures written to {fig_dir}")
