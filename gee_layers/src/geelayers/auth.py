"""Earth Engine login.

Two ways, both configured in gee_layers/.env (see .env.example) or the environment:
- personal: run `earthengine authenticate` once, then set EE_PROJECT to a Cloud project
  registered for Earth Engine (https://code.earthengine.google.com/register).
- service account (servers / CI): also set EE_SERVICE_ACCOUNT_KEY to the JSON key file path.
"""
from __future__ import annotations

import os
from pathlib import Path

import ee
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]   # gee_layers/
load_dotenv(PROJECT_DIR / ".env")

# The high-volume endpoint is meant for many parallel computePixels calls like ours.
HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"

_initialized = False


def init(project: str | None = None, force: bool = False) -> None:
    """Initialize Earth Engine once per process. Safe to call repeatedly."""
    global _initialized
    if _initialized and not force:
        return
    project = project or os.environ.get("EE_PROJECT")
    if not project:
        raise RuntimeError(
            "EE_PROJECT is not set. Put your Earth Engine Cloud project id in "
            f"{PROJECT_DIR / '.env'} (see .env.example) or pass init(project=...).")
    key = os.environ.get("EE_SERVICE_ACCOUNT_KEY")
    credentials = ee.ServiceAccountCredentials(None, key_file=key) if key else None
    try:
        ee.Initialize(credentials, project=project, opt_url=HIGH_VOLUME_URL)
    except Exception as e:  # most common: never ran `earthengine authenticate`
        raise RuntimeError(
            f"Earth Engine init failed for project {project!r}: {e}\n"
            "Run `earthengine authenticate` once, and check that the project is registered "
            "for Earth Engine.") from e
    _initialized = True
