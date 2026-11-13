"""Per-fire configuration. Add a new FireConfig for each study fire."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def get_secret(name: str) -> str:
    """Read a key from the environment / .env, failing loudly if it is missing."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to {REPO_ROOT / '.env'} (see .env.example).")
    return value


@dataclass
class FireConfig:
    slug: str                      # folder name under data/fires/
    name: str                      # human-readable name
    wfigs_name: str                # poly_IncidentName / IncidentName in NIFC WFIGS
    wfigs_min_acres: float         # disambiguates fires that share a name
    ignition_lat: float
    ignition_lon: float
    ignition_utc: pd.Timestamp
    start_utc: pd.Timestamp        # data window start
    end_utc: pd.Timestamp          # data window end
    bbox: tuple[float, float, float, float]  # west, south, east, north (search box)
    feds_fireid: float | None = None
    goes_end_utc: pd.Timestamp | None = None  # GOES window end (defaults to end_utc)
    nisar_passes_utc: dict[str, list[pd.Timestamp]] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return REPO_ROOT / "data" / "fires" / self.slug

    def dir(self, kind: str) -> Path:
        d = self.root / kind
        d.mkdir(parents=True, exist_ok=True)
        return d


T = pd.Timestamp

BUG_FIRE = FireConfig(
    slug="bug_fire",
    name="Bug Fire, CA/NV",
    wfigs_name="Bug",
    wfigs_min_acres=1000,
    # User-provided ignition point and time. WFIGS lists discovery 2026-08-08 22:30:49 UTC
    # at 39.726833, -120.035167 (see data/fires/bug_fire/README.md).
    ignition_lat=39.727395,
    ignition_lon=-120.0372941,
    ignition_utc=T("2026-08-08 23:08", tz="UTC"),
    start_utc=T("2026-08-08 00:00", tz="UTC"),
    end_utc=T("2026-08-28 00:00", tz="UTC"),
    # Final WFIGS perimeter is (-120.042, 39.690, -119.709, 39.969); search box adds a margin
    # wide enough to catch neighbouring fires (Stallion, Hawk) so they can be excluded.
    bbox=(-120.6, 39.2, -119.4, 40.3),
    feds_fireid=69872.0,
    nisar_passes_utc={
        "pre": [T("2026-07-23 02:55", tz="UTC"), T("2026-07-27 13:01", tz="UTC")],
        "post": [T("2026-08-15 13:09", tz="UTC"), T("2026-08-16 02:55", tz="UTC"),
                 T("2026-08-20 13:01", tz="UTC")],
    },
)

FIRES = {f.slug: f for f in [BUG_FIRE]}
