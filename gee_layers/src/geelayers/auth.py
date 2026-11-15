"""Earth Engine login.

Works out of the box for the team: the project defaults to the team's Earth Engine Cloud project (DEFAULT_PROJECT,
not a secret: access is granted per Google account in the project's IAM page), and the first run opens the Google
login in the browser if this computer has never logged in. Override in gee_layers/.env (see .env.example):
- EE_PROJECT: another Cloud project registered for Earth Engine (https://code.earthengine.google.com/register).
- EE_SERVICE_ACCOUNT_KEY: path to a service-account JSON key (servers / CI; never commit the key).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv

from ._ee_compat import ee

PROJECT_DIR = Path(__file__).resolve().parents[2]   # gee_layers/
load_dotenv(PROJECT_DIR / ".env")

# The team's project. Teammates need a role on it (IAM: Earth Engine Resource Writer + Service Usage Consumer).
DEFAULT_PROJECT = "wildfire-hackathon-514520"

# The high-volume endpoint is meant for many parallel computePixels calls like ours.
HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"

_initialized = False
_init_lock = threading.Lock()     # init() may be called from several threads at once (parallel fetches)


def _has_saved_login() -> bool:
    return Path(ee.oauth.get_credentials_path()).exists()


def init(project: str | None = None, force: bool = False) -> None:
    """Initialize Earth Engine once per process. Safe to call repeatedly."""
    global _initialized
    if _initialized and not force:
        return
    with _init_lock:
        if _initialized and not force:      # another thread finished while this one waited
            return
        _initialize(project)
        _initialized = True


def _initialize(project: str | None) -> None:
    project = project or os.environ.get("EE_PROJECT") or DEFAULT_PROJECT
    key = os.environ.get("EE_SERVICE_ACCOUNT_KEY")
    if key:
        credentials = ee.ServiceAccountCredentials(None, key_file=key)
    else:
        if not _has_saved_login():   # first run on this computer: Google login in the browser, saved for next time
            print("[earthengine] no saved Google login on this computer: opening the login page in the browser...")
            ee.Authenticate()
        credentials = "persistent"   # the login saved by ee.Authenticate / `earthengine authenticate`
    try:
        ee.Initialize(credentials, project=project, opt_url=HIGH_VOLUME_URL)
    except Exception as e:
        hint = ("Your Google account has no access to this project: ask the project owner to add your Google "
                "email in Google Cloud Console -> IAM -> Grant access (roles: Earth Engine Resource Writer and "
                "Service Usage Consumer), or set EE_PROJECT in gee_layers/.env to your own Earth Engine project."
                if "permission" in str(e).lower() or "403" in str(e) else
                "Run `earthengine authenticate` again, and check that the project is registered for Earth Engine.")
        raise RuntimeError(f"Earth Engine init failed for project {project!r}: {e}\n{hint}") from e
