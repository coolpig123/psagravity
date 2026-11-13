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
    feds_fireid: float | None = None   # force one FEDS id; None = all ids overlapping the official perimeter
    goes_end_utc: pd.Timestamp | None = None  # GOES window end (defaults to end_utc)
    nisar_passes_utc: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
    tz_name: str = "PDT"           # local time shown on the web page
    tz_offset_h: float = -7        # local = UTC + offset (summer time)
    subtitle: str = ""             # one line for the web page header
    notes: list[str] = field(default_factory=list)  # fire-specific findings for the README

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
    # Ignition time = WFIGS discovery (the brief said 23:08 UTC). The point is the brief's;
    # WFIGS origin 39.726833, -120.035167 is ~200 m away (see data/fires/bug_fire/README.md).
    ignition_lat=39.727395,
    ignition_lon=-120.0372941,
    ignition_utc=T("2026-08-08 22:30:49", tz="UTC"),
    start_utc=T("2026-08-08 00:00", tz="UTC"),
    end_utc=T("2026-08-28 00:00", tz="UTC"),
    # Final WFIGS perimeter is (-120.042, 39.690, -119.709, 39.969); search box adds a margin
    # wide enough to catch neighbouring fires (Stallion, Hawk) so they can be excluded.
    bbox=(-120.6, 39.2, -119.4, 40.3),
    subtitle="California / Nevada, August 2026. Where the fire burned, hour by hour.",
    nisar_passes_utc={
        "pre": [T("2026-07-23 02:55", tz="UTC"), T("2026-07-27 13:01", tz="UTC")],
        "post": [T("2026-08-15 13:09", tz="UTC"), T("2026-08-16 02:55", tz="UTC"),
                 T("2026-08-20 13:01", tz="UTC")],
    },
)

# Backup / additional US fires (WFIGS facts checked 2026-11-13). Ignition = WFIGS discovery,
# point = WFIGS point of origin. NISAR passes are looked up from ASF by the pipeline.
CROSSWHITE = FireConfig(
    slug="crosswhite",
    name="Crosswhite Fire, OR",
    wfigs_name="Crosswhite",          # child of the Rowe Creek Complex (2026-ORPRD-000445)
    wfigs_min_acres=100000,
    ignition_lat=44.803333,
    ignition_lon=-120.199167,
    ignition_utc=T("2026-07-16 00:51:54", tz="UTC"),
    start_utc=T("2026-07-15 00:00", tz="UTC"),
    end_utc=T("2026-08-25 00:00", tz="UTC"),
    goes_end_utc=T("2026-08-20 00:00", tz="UTC"),
    # final perimeter (-120.914, 44.338, -120.132, 44.952)
    bbox=(-121.3, 44.0, -119.75, 45.3),
    subtitle="Oregon, July-August 2026. Largest fire of the Rowe Creek Complex.",
    notes=[
        "Child incident of the ROWE CREEK COMPLEX (2026-ORPRD-000491, 373,929 acres). The README "
        "listed ~343,000 acres: that is the incident size (342,923); the WFIGS perimeter is 355,065 acres.",
        "WFIGS has no containment or fire-out date for this incident, and the perimeter is dated 2026-08-16.",
        "Several other complex fires burned next to it (0449 Porcupine Ridge, 0456, 0476 Hoag, 0567 Little Buck); "
        "detections closer to them are excluded.",
    ],
)

COLWASH = FireConfig(
    slug="colwash",
    name="Colwash Fire, WA",
    wfigs_name="Colwash",
    wfigs_min_acres=10000,
    ignition_lat=46.213617,
    ignition_lon=-120.120392,
    ignition_utc=T("2026-08-22 14:58:43", tz="UTC"),
    start_utc=T("2026-08-22 00:00", tz="UTC"),
    end_utc=T("2026-08-30 00:00", tz="UTC"),
    # final perimeter (-120.200, 46.086, -119.756, 46.201)
    bbox=(-120.55, 45.85, -119.4, 46.45),
    subtitle="Washington, August 2026. 39,000 acres in under a day.",
    notes=[
        "The official perimeter is dated 2026-08-23 00:20 UTC, 9.4 h after discovery, and was not updated later. "
        "Containment was 2026-08-23 12:53 UTC.",
        "The WFIGS point of origin (46.213617, -120.120392) lies ~1.4 km north of the final perimeter.",
        "The run happened mostly between VIIRS overpasses: VIIRS footprints cover only ~6% of the official area and the "
        "first VIIRS detection came 6.6 h after discovery. GOES saw it 48 min after discovery; GOES fire power peaked "
        "near 15 GW around 2026-08-23 01:00 UTC and 95% of the GOES fire energy was released by 2026-08-23 04:26 UTC.",
        "FEDS is almost empty for this fire (2 perimeters, 6 km², IoU 3%): its 12 h steps are too slow for a run of a few hours.",
    ],
)

BISCAR = FireConfig(
    slug="biscar",
    name="Biscar Fire, CA",
    wfigs_name="Biscar",
    wfigs_min_acres=10000,
    ignition_lat=40.545702,
    ignition_lon=-120.331628,
    ignition_utc=T("2026-07-18 18:06:09", tz="UTC"),
    start_utc=T("2026-07-18 00:00", tz="UTC"),
    end_utc=T("2026-08-12 00:00", tz="UTC"),
    # final perimeter (-120.307, 40.549, -119.958, 40.738); WFIGS origin lies ~2 km outside it
    bbox=(-120.65, 40.25, -119.6, 41.05),
    subtitle="Northeastern California, July-August 2026.",
    notes=[
        "The WFIGS point of origin (40.545702, -120.331628) lies ~2 km outside the final perimeter.",
        "The official perimeter is dated 2026-07-24 but containment was 2026-08-09: growth after Jul 24, "
        "if any, is not in the official perimeter.",
    ],
)

FIRES = {f.slug: f for f in [BUG_FIRE, CROSSWHITE, COLWASH, BISCAR]}
