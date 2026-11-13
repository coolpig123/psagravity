"""NISAR GCOV passes over a fire, from the ASF search API (no login needed to search).

Before/after comparisons are only valid on the same track (same orbit direction and look
geometry), so passes are grouped by track and same-track pre/post pairs are listed.
"""
from __future__ import annotations

import pandas as pd
import requests

SEARCH = "https://api.daac.asf.alaska.edu/services/search/param"


def search_passes(lon: float, lat: float, start_utc, end_utc, product: str = "GCOV") -> pd.DataFrame:
    """Standard-production (PR) NISAR passes covering a point in [start, end)."""
    r = requests.get(SEARCH, params={
        "dataset": "NISAR", "processingLevel": product, "intersectsWith": f"POINT({lon} {lat})",
        "start": pd.Timestamp(start_utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": pd.Timestamp(end_utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "output": "geojson"}, timeout=180)
    r.raise_for_status()
    rows = []
    for f in r.json().get("features", []):
        p = f["properties"]
        if "_PR_" not in p.get("fileID", ""):   # skip urgent-response duplicates
            continue
        rows.append({"time_utc": pd.Timestamp(p["startTime"]).tz_convert("UTC") if pd.Timestamp(p["startTime"]).tzinfo
                     else pd.Timestamp(p["startTime"], tz="UTC"),
                     "direction": p.get("flightDirection"), "track": int(p.get("pathNumber")),
                     "frame": int(p.get("frameNumber")), "file_id": p.get("fileID")})
    df = pd.DataFrame(rows, columns=["time_utc", "direction", "track", "frame", "file_id"])
    if df.empty:
        return df
    df["track_id"] = df.direction.str[0] + df.track.astype(str)
    # one row per acquisition: adjacent frames of the same pass start ~30 s apart
    df = df.sort_values(["track_id", "time_utc"])
    new_acq = (df.track_id != df.track_id.shift()) | (df.time_utc.diff() > pd.Timedelta(minutes=5))
    return df[new_acq.to_numpy()].sort_values("time_utc").reset_index(drop=True)


def classify(passes: pd.DataFrame, fire_start_utc, growth_end_utc) -> pd.DataFrame:
    """Label each pass pre / during / post and mark same-track pre/post pairs."""
    df = passes.copy()
    if df.empty:
        return df.assign(phase=[], pair_track=[])
    df["phase"] = "during"
    df.loc[df.time_utc < fire_start_utc, "phase"] = "pre"
    df.loc[df.time_utc > growth_end_utc, "phase"] = "post"
    pre_tracks = set(df.loc[df.phase == "pre", "track_id"])
    post_tracks = set(df.loc[df.phase == "post", "track_id"])
    df["pair_track"] = df.track_id.isin(pre_tracks & post_tracks)
    return df


def best_pairs(passes: pd.DataFrame) -> pd.DataFrame:
    """For each track with both, the last pre pass and the first post pass."""
    out = []
    for tid, g in passes[passes.pair_track].groupby("track_id"):
        pre, post = g[g.phase == "pre"].time_utc.max(), g[g.phase == "post"].time_utc.min()
        out.append({"track_id": tid, "pre_utc": pre, "post_utc": post, "gap_days": round((post - pre).total_seconds() / 86400, 1)})
    return pd.DataFrame(out, columns=["track_id", "pre_utc", "post_utc", "gap_days"]).sort_values("gap_days").reset_index(drop=True)
