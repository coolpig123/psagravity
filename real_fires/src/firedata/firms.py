"""VIIRS 375 m active fire points from the NASA FIRMS area API.

FIRMS serves science-quality (SP) data only up to ~2-3 months before present; newer
dates are only in the NRT sources. `pick_sources` chooses per date using the
data_availability endpoint.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

from .config import get_secret

API = "https://firms.modaps.eosdis.nasa.gov/api"
SATELLITES = {"SNPP": "VIIRS_SNPP", "NOAA20": "VIIRS_NOAA20", "NOAA21": "VIIRS_NOAA21"}
SAT_CODE = {"N": "SNPP", "N20": "NOAA20", "N21": "NOAA21"}
MAX_DAYS = 5  # FIRMS area API limit per request


def data_availability(map_key: str | None = None) -> pd.DataFrame:
    key = map_key or get_secret("FIRMS_MAP_KEY")
    r = requests.get(f"{API}/data_availability/csv/{key}/ALL", timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["min_date"] = pd.to_datetime(df["min_date"])
    df["max_date"] = pd.to_datetime(df["max_date"])
    return df.set_index("data_id")


def pick_source(sat: str, day: pd.Timestamp, avail: pd.DataFrame) -> str | None:
    """Prefer SP (standard processing) when it covers `day`, else NRT."""
    day = pd.Timestamp(day).tz_localize(None).normalize()
    for suffix in ("SP", "NRT"):
        sid = f"{SATELLITES[sat]}_{suffix}"
        if sid in avail.index and avail.loc[sid, "min_date"] <= day <= avail.loc[sid, "max_date"]:
            return sid
    return None


def fetch_area(source: str, bbox, start: pd.Timestamp, days: int, map_key: str) -> pd.DataFrame:
    area = ",".join(f"{v:g}" for v in bbox)
    url = f"{API}/area/csv/{map_key}/{source}/{area}/{days}/{start:%Y-%m-%d}"
    for attempt in range(3):
        r = requests.get(url, timeout=120)
        if r.ok and not r.text.startswith(("Invalid", "Error")):
            return pd.read_csv(io.StringIO(r.text))
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"FIRMS request failed for {source} {start:%F}: {r.status_code} {r.text[:200]}")


def download_viirs(bbox, start_utc, end_utc, raw_dir: Path, map_key: str | None = None) -> pd.DataFrame:
    """Download VIIRS points for all three satellites; one raw CSV per (source, chunk)."""
    key = map_key or get_secret("FIRMS_MAP_KEY")
    avail = data_availability(key)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    days = pd.date_range(start_utc.normalize(), end_utc.normalize(), freq="D", inclusive="left")
    for sat in SATELLITES:
        # group consecutive days with the same source into <=5-day chunks
        by_source: dict[str, list[pd.Timestamp]] = {}
        for d in days:
            src = pick_source(sat, d, avail)
            if src is None:
                print(f"  ! no FIRMS source covers {sat} on {d:%F}")
                continue
            by_source.setdefault(src, []).append(d)
        for src, ds in by_source.items():
            for i in range(0, len(ds), MAX_DAYS):
                chunk = ds[i:i + MAX_DAYS]
                out = raw_dir / f"{src}_{chunk[0]:%Y%m%d}_{len(chunk)}d.csv"
                if out.exists():
                    df = pd.read_csv(out)
                else:
                    df = fetch_area(src, bbox, chunk[0], len(chunk), key)
                    df.to_csv(out, index=False)
                df["firms_source"] = src
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tidy_viirs(df: pd.DataFrame) -> pd.DataFrame:
    """Raw FIRMS CSV -> standard columns (time_utc, lat, lon, frp_mw, source, ...)."""
    if df.empty:
        return df
    t = pd.to_datetime(df["acq_date"] + " " + df["acq_time"].astype(int).astype(str).str.zfill(4),
                       format="%Y-%m-%d %H%M", utc=True)
    conf = df["confidence"].astype(str).map({"l": "low", "n": "nominal", "h": "high"}).fillna(df["confidence"].astype(str))
    out = pd.DataFrame({
        "time_utc": t,
        "lat": df["latitude"],
        "lon": df["longitude"],
        "frp_mw": df["frp"],
        "source": "VIIRS_" + df["satellite"].map(SAT_CODE),
        "confidence": conf,
        "daynight": df["daynight"],
        "bright_ti4_k": df["bright_ti4"],
        "scan_km": df["scan"],
        "track_km": df["track"],
        "product": df["firms_source"],
    })
    return out.drop_duplicates(["time_utc", "lat", "lon", "source"]).sort_values("time_utc").reset_index(drop=True)
