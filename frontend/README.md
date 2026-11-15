# Running the app (website + Python server)

Commands are for PowerShell, one per line, starting in the repo root.

**Once:**

```powershell
cd frontend
npm install
cd ..
python -m venv backend\.venv
backend\.venv\Scripts\python -m pip install -r gee_layers\requirements.txt -e gee_layers -e spread_model
backend\.venv\Scripts\python -m pip install -r real_fires\requirements.txt -e real_fires   # NISAR reader (optional)
Copy-Item real_fires\.env.example real_fires\.env
notepad real_fires\.env                                 # your own Earthdata login, for NISAR (optional)
```

Without `backend\.venv`, the server uses `real_fires\.venv` if that exists.

**Earth Engine needs no file:** the project defaults to the team project (`DEFAULT_PROJECT` in
[gee_layers/src/geelayers/auth.py](../gee_layers/src/geelayers/auth.py)), and the first Predict opens the Google
login in the browser (saved for next time). Your Google account must be added to that project once by its owner:
Google Cloud Console -> IAM -> Grant access, roles **Earth Engine Resource Writer** and **Service Usage Consumer**.
Without NISAR login the prediction still runs, with a note saying NISAR was skipped. Passwords and keys are never
committed: everyone uses their own login.

**Every time (one command starts the website and the Python server):**

```powershell
cd frontend
npm run dev                                             # then open http://localhost:5173
```

Stop with `Ctrl+C`. `npm run dev:web` runs the website alone; `python backend\server.py` runs the server alone (port 8000).

## The online version (no server)

https://coolpig123.github.io/psagravity/ is this website built with `VITE_STATIC_DATA=haifa`: no Python server,
no Earth Engine, no login. The data around Haifa (from 2026-06-01) was downloaded in advance into
`public/haifa/`, and Predict, the arrival times, the wind and Extract Data run in the browser on it (the same
Python model, with Pyodide). Details, limits and how to refresh the data:
[israel_static/README.md](../israel_static/README.md). To try it locally without the Python server:

```powershell
cd frontend
$env:VITE_STATIC_DATA = "haifa"; npm run dev:web
```

## Finding a place

When the website opens it asks the browser for your location; if you allow it, the map flies there (blue dot).
Either way, the search bar at the top-left of the map finds any place (type it, press Enter): the map flies there,
framing the place a little zoomed out, and lists the other matches below the bar. The target button next to it goes
back to your location. Places come from [OpenStreetMap Nominatim](https://nominatim.org/) (free, no key; one
search per Enter, as its usage policy asks), so this also works on the GitHub Pages version.

## Wind

The wind is on from the moment the website opens, with or without a polygon: the whole map is covered with the wind
at the **Start (UTC)** time, at any zoom up to the whole world, in the style of
[earth.nullschool.net](https://earth.nullschool.net): colours = wind speed (0-20 m/s scale next to the toggle),
moving white lines = particles carried by the wind. Pan or zoom and the wind follows the view (new data only when the
view leaves the area already loaded, which is widened to 5° tiles, or when zooming in deserves a finer grid: zoomed
far out the server sends every 2nd-3rd MERRA-2 cell). The **wind** checkbox (right of the regions checkbox) turns it
off and on; changing the start time reloads it. Once a polygon is drawn, the label also gives the wind at the
polygon; hover it for the hour and source.

The data is NASA **MERRA-2** 10 m wind (hourly means, ~55 km grid), read by the server (`POST /api/wind`,
[firedata.merra2](../real_fires/src/firedata/merra2.py)) from GES DISC: only the area on screen, a few
seconds the first time and instant after (cached in `real_fires/data/merra2_cache/raw`). It needs `earthaccess` in the
server's venv (the `real_fires\requirements.txt` install above) and your Earthdata login in `real_fires\.env`, with the
application "NASA GESDISC DATA ARCHIVE" approved once in your Earthdata profile. MERRA-2 is published 2-6 weeks
after each month, so recent start times say "no MERRA-2 data for this start time yet".

## Predict where the fire spreads

1. Zoom to the fire, press **Draw**, click the outline of the burning area, press **Finish**.
2. Set **Start (UTC)**, the time the polygon is burning (default `2026-08-09T21:00Z`, the Bug Fire at
   39.727, -120.037: draw there to test).
3. Press **Predict** (enabled for polygons up to 400 km²).

The server (`POST /api/predict`) fetches only the area of the polygon + 5 km from Earth Engine (terrain, land
cover, weather, Sentinel-2 burn scars) and NISAR (ASF), runs the spread model forward 30 steps (one step = at most one
100 m cell, so up to ~3 km), and the map zooms to the result and colours each cell by its chance of burning
within those steps (yellow -> orange -> red). Under the
map: the most likely spread direction, the expected new burned area, and the weather time, NISAR passes and
Sentinel-2 coverage that were used. **Clear** or **Draw** removes the prediction.

**Arrival times:** the server also runs the survival model (`spread_model/src/firespread/survival.py`, trained on
the real fires). Under the map, **Show** switches the colours between the CA chance, the **chance the fire ever
gets there**, the **median hours until it arrives if it does** (dark red = soon, yellow = days; cells with at least a
10% chance), and the **chance it arrives within 6 / 12 / 24 / 48 h**. Click a cell for all its numbers, including the
25-75% range of the arrival time (fire speed varies ~10x between fires, so the range is wide). The panel gives the expected burned area within each horizon. Needs `xgboost` and `scipy`
in the server's venv (`pip install -r spread_model/requirements.txt`); without them, or without the model file, the
map shows the CA chance only and a note says why. How good it is: see "When: arrival time per cell" in
[spread_model/README.md](../spread_model/README.md).

Expect **about 1 minute** from pressing Predict to the map (measured near the Bug Fire: 75 s for a new area,
68 s for a polygon 1 km from an earlier one, 52 s for the same polygon again). Most of it is the Sentinel-2 burn
scars and the Earth Engine inputs; NISAR takes ~50 s the first time for an area and a few seconds after that
(downloaded pieces are cached in `real_fires/data/nisar_cache/raw/chunks`).

After pulling new code, restart `npm run dev`: the Python server keeps the code it started with. A second server
on the same port now refuses to start (before, on Windows, both ran and the old one kept answering). If the NISAR reader is not installed or the Earthdata login
fails, the prediction still runs without NISAR and a note says so.

Request and response format: [spread_model/src/firespread/web.py](../spread_model/src/firespread/web.py).
Offline tests: `python -m pytest spread_model/tests/test_web.py backend/tests`.

## Extract Data

**Extract Data** is a debugging tool. It sends the finished polygon to `POST /api/polygon`, and the server prints
the points and a table of the Earth Engine inputs (`get_fire_inputs`: min / mean / max / % missing per variable)
in the `[api]` lines of the terminal. It does not run the model or change the map. It ignores the Start box: it
always uses 2 days before now (UTC) with 3 hours of weather. Some layers cover only the contiguous US.
