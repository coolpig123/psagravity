"""Choose the fire polygon to start from.

Two kinds of input look alike but mean different things:
- shapes a user drew (one or several features, no times): all of them are burning -> union.
- a perimeter time series (e.g. real_fires FEDS 12 h perimeters, one feature per time step):
  the fire at `start` is the last perimeter observed at or before `start`, NOT the union
  (the union is the final burned area, which would hand the model the answer).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Property names that hold the time a perimeter was observed, most specific first.
# FEDS (real_fires): obs_time_utc = satellite pass the perimeter is based on; step_utc = its 12 h step.
TIME_KEYS = ("obs_time_utc", "time_utc", "perimeter_time_utc", "datetime", "time", "date",
             "poly_PolygonDateTime", "step_utc")


def _load(obj: Any) -> Any:
    if hasattr(obj, "__geo_interface__"):
        return obj.__geo_interface__
    if isinstance(obj, (str, Path)):
        text = str(obj).strip()
        return json.loads(text) if text.startswith(("{", "[")) else json.loads(Path(obj).read_text(encoding="utf-8"))
    return obj


def _time_key(features: list[dict], key: str | None) -> str | None:
    """The property that dates every feature, or None if the features are not a time series."""
    props = [f.get("properties") or {} for f in features]
    candidates = [key] if key else list(TIME_KEYS)
    for k in candidates:
        if all(p.get(k) not in (None, "") for p in props):
            return k
    if key:
        raise ValueError(f"perimeter time key {key!r} is missing on some features")
    return None


def resolve_perimeter(polygon: Any, start, time_key: str | None = None) -> tuple[Any, dict[str, str]]:
    """Return (geometry to rasterize, metadata).

    A FeatureCollection whose features all carry a time property is treated as a time series:
    the last feature observed at or before `start` is returned. Anything else is returned unchanged
    (several untimed features = several burning areas, merged).
    """
    data = _load(polygon)
    if not (isinstance(data, dict) and data.get("type") == "FeatureCollection"):
        return data, {}
    features = data.get("features", [])
    if len(features) <= 1:
        return data, {}
    key = _time_key(features, time_key)
    if key is None:
        return data, {"initial_perimeter": f"union of {len(features)} untimed features"}

    start = pd.Timestamp(start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    times = []
    for f in features:
        t = pd.Timestamp(f["properties"][key])
        times.append(t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC"))
    if len(set(times)) == 1:   # several areas observed at the same time: all burning
        return data, {"initial_perimeter": f"union of {len(features)} features with the same {key}"}
    before = [(t, i) for i, t in enumerate(times) if t <= start]
    if not before:
        raise ValueError(f"perimeter time series ({key}): the first perimeter is at {min(times)}, after start "
                         f"{start}. Use a later start or give the ignition point instead.")
    t, i = max(before)
    return features[i], {"initial_perimeter": f"feature {i} of {len(features)}, {key} = {t.isoformat()}",
                         "initial_perimeter_time_utc": t.isoformat()}
