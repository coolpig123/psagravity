"""Self-contained interactive time-progression page for one fire.

    python -m firedata.web bug_fire            # -> data/fires/bug_fire/figures/<slug>_spread.html

Reads the processed outputs of scripts/build_ground_truth.py and embeds everything (data and a
shaded-relief basemap) in one HTML file, so the page works offline and needs no server.
Layers a fire lacks (GOES, FEDS) are hidden automatically.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import FIRES, REPO_ROOT, FireConfig

SAT_INDEX = {"VIIRS_SNPP": 0, "VIIRS_NOAA20": 1, "VIIRS_NOAA21": 2}


def _basemap(x0, y0, x1, y1, zoom=12, max_px=1400) -> str:
    """Esri World Shaded Relief for the extent (EPSG:3857), grayscale JPEG data URI ('' offline)."""
    try:
        import contextily as cx
        from PIL import Image
        img, (ex0, ex1, ey0, ey1) = cx.bounds2img(x0, y0, x1, y1, zoom=zoom,
                                                  source=cx.providers.Esri.WorldShadedRelief)
    except Exception as e:  # offline: page still works, just without relief
        print(f"  (no basemap: {e.__class__.__name__})")
        return ""
    H, W = img.shape[:2]
    box = (int((x0 - ex0) / (ex1 - ex0) * W), int((ey1 - y1) / (ey1 - ey0) * H),
           int((x1 - ex0) / (ex1 - ex0) * W), int((ey1 - y0) / (ey1 - ey0) * H))
    im = Image.fromarray(img[..., :3]).crop(box).convert("L")
    if im.width > max_px:
        im = im.resize((max_px, round(im.height * max_px / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _read(path: Path, kind: str):
    if not path.exists():
        return None
    return gpd.read_file(path) if kind == "geo" else pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def build_page_data(F: FireConfig, zoom: int = 12) -> dict:
    """Everything the page needs, in normalized map coordinates (0..1, north up) and
    minutes since the start of the fire's data window."""
    P = F.root / "processed"
    T0 = F.start_utc.normalize()
    mins = lambda t: int((pd.Timestamp(t) - T0).total_seconds() // 60)

    final = gpd.read_file(P / "wfigs_final_perimeter.geojson").to_crs(3857)
    x0, y0, x1, y1 = final.total_bounds
    pad = 0.08 * max(x1 - x0, y1 - y0)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    nx = lambda x: round(float((x - x0) / (x1 - x0)), 5)
    ny = lambda y: round(float((y1 - y) / (y1 - y0)), 5)

    def rings(geom, tol=25):
        g = geom.simplify(tol)
        return [[[nx(a), ny(b)] for a, b in p.exterior.coords] for p in getattr(g, "geoms", [g])]

    def xy(lon, lat):
        p = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(3857).iloc[0]
        return [nx(p.x), ny(p.y)]

    summary = json.loads((P / "summary.json").read_text())
    final_acres = round(summary["wfigs"]["final_acres"])

    # FEDS perimeters (only steps with new observations)
    perims, growth_end = [], None
    feds = _read(P / "feds_perimeters_12h.geojson", "geo")
    if feds is not None and len(feds):
        feds = feds.to_crs(3857)
        feds["obs_time_utc"] = pd.to_datetime(feds.obs_time_utc, utc=True)
        feds = feds[feds.is_new.astype(bool)].sort_values("obs_time_utc")
        perims = [{"t": mins(r.obs_time_utc), "acres": round(r.area_acres), "km2": round(r.area_km2, 1),
                   "rings": rings(r.geometry)} for _, r in feds.iterrows()]
        for prev, cur in zip(perims, perims[1:]):
            if cur["acres"] - prev["acres"] > 250:
                growth_end = cur["t"]

    pts = pd.read_parquet(P / "fire_points.parquet")
    if "ambiguous" in pts:
        pts = pts[~pts.ambiguous]
    v = pts[pts.source.str.startswith("VIIRS")]
    vg = gpd.GeoSeries(gpd.points_from_xy(v.lon, v.lat), crs=4326).to_crs(3857)
    viirs = [[mins(t), nx(p.x), ny(p.y), round(float(f), 1), SAT_INDEX.get(s, 0)]
             for t, p, f, s in zip(v.time_utc, vg, v.frp_mw.fillna(0), v.source)]

    # GOES pixels as footprint quads
    g = pts[pts.source.str.startswith("GOES")]
    quads, goes_pts, goes_series, goes_label = [], [], [], ""
    if len(g):
        from . import goes as goes_mod
        sat = "goes" + g.source.iloc[0][4:6]
        goes_label = f"GOES-{g.source.iloc[0][4:6]}"
        pix = g.groupby(["lat", "lon"]).size().reset_index()[["lat", "lon"]]
        pix["id"] = range(len(pix))
        fp = goes_mod.pixel_footprints(pix, satellite=sat, when=g.time_utc.min()).to_crs(3857)
        quads = [[[nx(a), ny(b)] for a, b in list(poly.exterior.coords)[:4]] for poly in fp]
        g = g.merge(pix, on=["lat", "lon"])
        goes_pts = [[mins(t), int(i), round(float(f), 1) if pd.notna(f) else 0]
                    for t, i, f in zip(g.time_utc, g.id, g.frp_mw)]

    frp = pd.read_csv(P / "frp_timeseries.csv", parse_dates=["time_utc"])
    gs = frp[frp.source.str.startswith("GOES")].sort_values("time_utc")
    goes_series = [[mins(t), round(float(f))] for t, f in zip(gs.time_utc, gs.frp_total_mw)]
    vs = frp[frp.source.str.startswith("VIIRS")]
    passes = [[mins(t), SAT_INDEX.get(s, 0), round(float(f)), int(n)]
              for t, s, f, n in zip(vs.time_utc, vs.source, vs.frp_total_mw, vs.n_points)]

    merged = []
    mpath = P / "wfigs_merged_incidents.csv"
    if mpath.exists() and mpath.stat().st_size > 5:
        m = pd.read_csv(mpath)
        merged = [{"t": mins(pd.Timestamp(r.FireDiscoveryDateTime)), "xy": xy(r.InitialLongitude, r.InitialLatitude),
                   "acres": int(r.IncidentSize or 0), "name": str(r.IncidentName)} for r in m.itertuples()]

    if summary.get("growth_end_utc") not in (None, "None", "NaT"):  # pipeline's rule (FEDS / GOES / VIIRS)
        growth_end = mins(pd.Timestamp(summary["growth_end_utc"]))
    ign_t = mins(pd.Timestamp(summary["wfigs"]["discovery_utc"]))
    run_end = (growth_end if growth_end is not None else max([p[0] for p in viirs] + [ign_t])) + 24 * 60
    start = (ign_t // 60 - 3) * 60
    pairs_csv = P / "nisar_pairs.csv"
    if pairs_csv.exists() and pairs_csv.stat().st_size > 40:
        nisar_post = sorted(mins(pd.Timestamp(t)) for t in pd.read_csv(pairs_csv).post_utc)
    else:
        nisar_post = [mins(t) for t in F.nisar_passes_utc.get("post", [])]
    full_end = max([run_end] + [t + 12 * 60 for t in nisar_post])
    peak_t = goes_series[int(np.argmax([s[1] for s in goes_series]))][0] if goes_series else \
        (passes[int(np.argmax([p[2] for p in passes]))][0] if passes else ign_t + 60)
    cont = summary["wfigs"].get("containment_utc")

    return {
        "name": F.name, "t0": T0.isoformat(), "basemap": _basemap(x0, y0, x1, y1, zoom),
        "aspect": round(float((x1 - x0) / (y1 - y0)), 4),
        "kmPerUnitX": round(float((x1 - x0) * np.cos(np.radians(F.ignition_lat)) / 1000), 3),
        "final": {"rings": rings(final.geometry.iloc[0]), "acres": final_acres},
        "perims": perims, "growthEnd": growth_end, "viirs": viirs, "goesPix": quads, "goes": goes_pts,
        "goesSeries": goes_series, "goesLabel": goes_label, "passes": passes, "merged": merged,
        "ignition": {"t": ign_t, "xy": xy(F.ignition_lon, F.ignition_lat)},
        "nisar": nisar_post,
        "containment": mins(pd.Timestamp(cont)) if cont and cont != "NaT" else None,
        "ranges": {"run": [start, run_end], "nisar": [start, full_end]},
        "initialT": int(peak_t),
        "tz": {"name": F.tz_name, "offsetH": F.tz_offset_h},
    }


def _day_label(T0: pd.Timestamp, a: int, b: int) -> str:
    da, db = T0 + pd.Timedelta(minutes=a), T0 + pd.Timedelta(minutes=b)
    return f"{da:%b} {da.day}–{db.day}" if da.month == db.month else f"{da:%b} {da.day} – {db:%b} {db.day}"


def page_text(F: FireConfig, data: dict) -> dict:
    """Fire-specific strings shown on the page (footer as escaped HTML)."""
    T0 = pd.Timestamp(data["t0"])
    e = html.escape
    fe = len(data["perims"]) > 0
    final_pct = round(max(p["acres"] for p in data["perims"]) / data["final"]["acres"] * 100) if fe else None
    sources = ["VIIRS 375 m active fire from S-NPP, NOAA-20 and NOAA-21 (NASA FIRMS)."]
    if data["goes"]:
        sources.append(f"{data['goesLabel']} ABI fire detection (NOAA, every 5 min).")
    if fe:
        sources.append("FEDS modeled perimeters every 12 h (NASA).")
    sources.append("Official perimeter and dates from NIFC WFIGS. Basemap: Esri World Shaded Relief.")
    notes = []
    if fe:
        notes.append(f"FEDS perimeters are drawn from VIIRS pixels and reach {final_pct}% of the official area, "
                     "so treat them as a lower bound.")
    if data["goes"]:
        notes.append("GOES pixels are large and not terrain-corrected; they give timing, not exact location.")
    notes.append("Detections that could belong to a neighbouring fire are left out.")
    return {
        "name": F.name.split(",")[0],
        "subtitle": F.subtitle or F.name,
        "runLabel": f"Main run ({_day_label(T0, *data['ranges']['run'])})",
        "nisarLabel": f"Through NISAR passes ({_day_label(T0, *data['ranges']['nisar'])})",
        "finalLabel": f"NIFC WFIGS, {data['final']['acres']:,} acres",
        "footerHtml": (f'<p><strong>Sources.</strong> {e(" ".join(sources))}</p>'
                       f'<p><strong>Reading it.</strong> {e(" ".join(notes))}</p>'),
    }


def _template() -> str:
    return (files("firedata") / "templates" / "fire_page.html").read_text()


def _json(obj) -> str:
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def render_page(F: FireConfig, data: dict) -> str:
    """Single-fire page with the data embedded."""
    data = {**data, "text": page_text(F, data)}
    block = f'<script type="application/json" id="fire-data">{_json(data)}</script>'
    return _template().replace("__TITLE__", html.escape(f"{F.name.split(',')[0]} Spread")).replace("__DATA_BLOCK__", block)


def build_page(fire: str, out: Path | None = None) -> Path:
    """Build the interactive page for a fire (by slug, e.g. 'bug_fire') and return its path."""
    F = FIRES[fire]
    out = Path(out) if out else F.dir("figures") / f"{F.slug}_spread.html"
    out.write_text(render_page(F, build_page_data(F)))
    print(f"  page written to {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def build_app(fires: list[str], out_dir: Path | None = None) -> Path:
    """Multi-fire site: app/index.html + app/data/<slug>.js (loaded when a fire is selected).
    Static files only; open index.html directly or serve the folder."""
    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "app"
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    index = []
    for slug in fires:
        F = FIRES[slug]
        data = build_page_data(F)
        data["text"] = page_text(F, data)
        (out_dir / "data" / f"{slug}.js").write_text(
            f"window.FIRE_DATA = window.FIRE_DATA || {{}};\nwindow.FIRE_DATA[{json.dumps(slug)}] = {_json(data)};\n")
        index.append({"slug": slug, "label": f"{F.name} · {data['final']['acres']:,} acres"})
        print(f"  app data: {slug} ({(out_dir / 'data' / f'{slug}.js').stat().st_size / 1024:.0f} KB)")
    block = f"<script>window.FIRE_INDEX = {_json(index)};</script>"
    page = _template().replace("__TITLE__", "Fire Spread Explorer").replace("__DATA_BLOCK__", block)
    (out_dir / "index.html").write_text(page)
    print(f"  app written to {out_dir / 'index.html'}")
    return out_dir / "index.html"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fires", nargs="+", choices=FIRES)
    ap.add_argument("--app", action="store_true", help="build the multi-fire site in app/ instead of one page per fire")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.app:
        build_app(a.fires, a.out)
    else:
        for f in a.fires:
            build_page(f, a.out)
