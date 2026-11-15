# israel_static: the website without a server (Haifa)

The published site (https://coolpig123.github.io/psagravity/) has no Python server, no Earth Engine and no
logins. Everything it needs around **Haifa** was downloaded once into `frontend/public/haifa/` (committed, 26 MB),
and the browser runs the repo's own spread model on it. All the buttons work, inside that area and period:

| On the site | Where it comes from |
|---|---|
| **Predict** (CA chance, arrival times, all views, click a cell) | `firespread` (unchanged) running in the browser with [Pyodide](https://pyodide.org); inputs cut from the files below |
| **Wind** animation and label | MERRA-2 10 m wind, hourly, read by the page itself |
| **Extract Data** | the table of model inputs around the polygon, shown under the map |
| Place search, my location, map | OpenStreetMap / Nominatim, as before (no key) |

## What was downloaded

Area: longitude 34.85 to 35.25, latitude 32.55 to 32.95 (Haifa, the Carmel, the bay; the dashed green box on the
map). 100 m cells, UTM 36N, 384 x 451 cells. Period: from 2026-06-01 (the 2026 fire season).

| Data | Source | Period | Size |
|---|---|---|---|
| Elevation, land cover | NASADEM, ESA WorldCover (Earth Engine) | fixed | 0.5 MB |
| Hourly weather | ERA5-Land (Earth Engine), on a 4 km grid | 2026-06-01 to 2026-11-09 | 4.8 MB |
| Hourly weather | GFS (Earth Engine): hours 0-5 of every run, then the forecast of the last run | 2026-06-01 to 2026-11-20 | 8.3 MB |
| Wind for the animation | MERRA-2 (GES DISC), 16 x 13 cells around Israel | 2026-06-01 to 2026-10-01 (all that is published) | 2.5 MB |
| NISAR cover, soil moisture, vegetation water, burned | `firedata.nisar_provider`, one snapshot after each of 32 passes | 2026-06-21 to 2026-11-03 | 10 MB |
| Sentinel-2 burn scars (dNBR) | `firespread.burn_scars`, one snapshot every 5 days | 2026-06-01 to 2026-11-13 | 0.2 MB |
| VIIRS detections (active front) | NASA FIRMS | **not downloaded: no `FIRMS_MAP_KEY` in `real_fires/.env`** | - |

Differences from the live app (`npm run dev` with the Python server), all reported under the map when they apply:

- **Weather "Auto"** = ERA5-Land where it exists, GFS after it. Meteomatics (needs an account) and RTMA (US only)
  fall back to these, with a note. ERA5-Land has no sea cells: coast cells take the nearest land cell (the live
  app leaves them without weather).
- **Burn scars and NISAR** come from the latest snapshot before the Start time, not computed for that exact time.
- **Active front**: without the VIIRS detections the whole polygon is the front, as in the live app without a key.
- A polygon outside the box, or a Start outside the period, gives a message saying what is available.

Checked against the live pipeline (same polygon on the Carmel, 2026-08-02 12:00 UTC, ERA5-Land): same direction
(127.4 vs 127.3 degrees), expected area 20.97 vs 21.07 km², chance per cell correlated 0.999. Static: 1 s of model
time (about 10 s in the browser); live: 24 s with everything cached, about 1 minute the first time.

## Files

```
israel_static/
  scripts/build_haifa_data.py   # downloads everything (needs the logins), writes frontend/public/haifa/
  browser/staticfire.py         # runs in the browser: inputs from the files + the two endpoints
  browser/xgboost.py            # numpy stand-in for xgboost (Pyodide's xgboost cannot read our model files)
  tests/test_staticfire.py      # offline tests on the committed data (pytest)
  tests/test_site.py            # browser test of the built site (Playwright)
  data/                         # download caches, not committed
frontend/public/haifa/          # the data the site reads: index.json, static.npz, weather/, wind/, nisar/, s2/
frontend/src/staticSite.ts      # answers /api/wind, /api/predict, /api/polygon on the static site
frontend/src/staticWorker.ts    # Web Worker: Pyodide + the Python above
```

The Python the browser runs is not copied: `frontend/vite.config.ts` packs `spread_model/src/firespread`, three
modules of `gee_layers` and `israel_static/browser` into `py-bundle.json` at build time.

## Run it locally

```powershell
cd frontend
$env:VITE_STATIC_DATA = "haifa"; npm run dev:web        # http://localhost:5173, no Python server needed
```

The first Predict waits for Python to load in the browser (about half a minute on the first visit, 33 MB from the
jsDelivr CDN, then cached by the browser); it starts loading when the page opens.

## Refresh or extend the data

Needs the logins of [frontend/README.md](../frontend/README.md) (Earth Engine, Earthdata). From the repo root:

```powershell
backend\.venv\Scripts\python israel_static\scripts\build_haifa_data.py all        # about 40 minutes
backend\.venv\Scripts\python israel_static\scripts\build_haifa_data.py firms manifest   # after adding FIRMS_MAP_KEY
```

Steps: `static era5 gfs wind firms s2 nisar manifest` (any subset; downloads are cached in `israel_static/data/`).
Another area or period: change `BBOX`, `START` and `NAME` at the top of the script. Then commit
`frontend/public/haifa/`.

## Tests

```powershell
backend\.venv\Scripts\python -m pytest israel_static/tests/test_staticfire.py
cd frontend; $env:BASE_PATH = "/psagravity/"; $env:VITE_STATIC_DATA = "haifa"; npm run build; npx vite preview --port 4173
real_fires\.venv\Scripts\python israel_static\tests\test_site.py http://localhost:4173/psagravity/   # other terminal
```

`test_site.py --channel msedge` uses the installed Edge instead of Playwright's Chromium.
