"""Hindcast windows from the real_fires FEDS 12 h perimeters.

A window = a start perimeter at t (observed, new FEDS data) + the perimeters observed about 12 h and 24 h later.
FEDS steps are 09:30 / 20:30 UTC, so "+12 h" is really +11 h or +13 h; the actual observation times are used.

A FEDS step with is_new = False has no new satellite data: FEDS repeats the previous perimeter. Such a step is
not an observation, so it can be neither a start nor a target (flagged as `stale`, left out of the scores).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FIRES_DIR = REPO / "real_fires" / "data" / "fires"

# Fires left out of the scores (reported, not hidden). Value = reason.
EXCLUDED_FIRES = {
    "colwash": "FEDS does not track it: IoU vs the official WFIGS perimeter 2.9 % (area 3.8 %), "
               "only 2 perimeters with new data",
}
# Targets ~12 h and ~24 h after the start. A target must fall within +- TOLERANCE of the horizon.
HORIZONS_H = (12, 24)
TOLERANCE_H = 2.5


@dataclass
class Target:
    horizon_h: int           # nominal: 12 or 24
    time: pd.Timestamp       # actual FEDS observation time
    feature: dict | None     # GeoJSON feature, None if missing
    area_km2: float
    stale: bool              # FEDS had no new data at this step -> not scored


@dataclass
class Window:
    fire: str
    start: pd.Timestamp
    start_feature: dict
    start_area_km2: float
    targets: list[Target] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.fire}_{self.start:%m%d_%H%M}"

    def scored_targets(self) -> list[Target]:
        return [t for t in self.targets if t.feature is not None and not t.stale]


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_perimeters(fire: str, fires_dir: Path = FIRES_DIR) -> list[dict]:
    """FEDS 12 h perimeters of a fire, sorted by observation time."""
    path = fires_dir / fire / "processed" / "feds_perimeters_12h.geojson"
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    return sorted(features, key=lambda f: _utc(f["properties"]["obs_time_utc"]))


def build_windows(fire: str, features: list[dict] | None = None) -> list[Window]:
    """One window per FEDS perimeter with new data that has a later perimeter at +12 h or +24 h."""
    features = load_perimeters(fire) if features is None else features
    times = [_utc(f["properties"]["obs_time_utc"]) for f in features]
    windows = []
    for i, f in enumerate(features):
        p = f["properties"]
        if not p.get("is_new", True):
            continue
        w = Window(fire, times[i], f, float(p["area_km2"]))
        for h in HORIZONS_H:
            goal = times[i] + pd.Timedelta(hours=h)
            near = [(abs(t - goal), j) for j, t in enumerate(times) if j > i]
            near = [(d, j) for d, j in near if d <= pd.Timedelta(hours=TOLERANCE_H)]
            if not near:
                w.targets.append(Target(h, goal, None, float("nan"), stale=False))
                continue
            j = min(near)[1]
            q = features[j]["properties"]
            w.targets.append(Target(h, times[j], features[j], float(q["area_km2"]),
                                    stale=not q.get("is_new", True)))
        if w.scored_targets():
            windows.append(w)
    return windows


def describe(windows: list[Window]) -> pd.DataFrame:
    """Table of windows and targets (which are scored, which are stale)."""
    rows = []
    for w in windows:
        for t in w.targets:
            rows.append({"window": w.id, "start_utc": w.start, "start_km2": round(w.start_area_km2, 1),
                         "horizon_h": t.horizon_h, "target_utc": t.time,
                         "hours": round((t.time - w.start).total_seconds() / 3600, 1),
                         "target_km2": round(t.area_km2, 1),
                         "status": "missing" if t.feature is None else ("stale" if t.stale else "scored")})
    return pd.DataFrame(rows)
