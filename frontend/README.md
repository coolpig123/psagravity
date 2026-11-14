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
backend\.venv\Scripts\earthengine authenticate          # Google login in the browser
Copy-Item gee_layers\.env.example gee_layers\.env
notepad gee_layers\.env                                 # set EE_PROJECT=<your Earth Engine Cloud project id>
Copy-Item real_fires\.env.example real_fires\.env
notepad real_fires\.env                                 # Earthdata login, for NISAR (optional)
```

Without `backend\.venv`, the server uses `real_fires\.venv` if that exists.

**Every time (one command starts the website and the Python server):**

```powershell
cd frontend
npm run dev                                             # then open http://localhost:5173
```

Stop with `Ctrl+C`. `npm run dev:web` runs the website alone; `python backend\server.py` runs the server alone (port 8000).

## Predict where the fire spreads

1. Zoom to the fire, press **Draw**, click the outline of the burning area, press **Finish**.
2. Set **Start (UTC)**, the time the polygon is burning (default `2026-08-09T21:00Z`, the Bug Fire at
   39.727, -120.037: draw there to test).
3. Press **Predict** (enabled for polygons up to 400 km²).

The server (`POST /api/predict`) fetches only the area of the polygon + 5 km from Earth Engine (terrain, land
cover, weather, Sentinel-2 burn scars) and NISAR (ASF), runs the spread model, and the map shows the cells next
to the fire coloured by their chance of catching fire in the next model step (yellow -> orange -> red). Under the
map: the most likely spread direction, the expected new burned area, and the weather time, NISAR passes and
Sentinel-2 coverage that were used. **Clear** or **Draw** removes the prediction.

Expect **4-7 minutes** from pressing Predict to the map (measured on the Bug Fire: 240-401 s; a repeat of the same
polygon was not faster). The time goes into the Earth Engine inputs (terrain, land cover, 6 hours of weather), the
Sentinel-2 burn scars and reading NISAR from ASF. If the NISAR reader is not installed or the Earthdata login
fails, the prediction still runs without NISAR and a note says so.

Request and response format: [spread_model/src/firespread/web.py](../spread_model/src/firespread/web.py).
Offline tests: `python -m pytest spread_model/tests/test_web.py backend/tests`.

## Extract Data

**Extract Data** is a debugging tool. It sends the finished polygon to `POST /api/polygon`, and the server prints
the points and a table of the Earth Engine inputs (`get_fire_inputs`: min / mean / max / % missing per variable)
in the `[api]` lines of the terminal. It does not run the model or change the map. It ignores the Start box: it
always uses 2 days before now (UTC) with 3 hours of weather. Some layers cover only the contiguous US.
