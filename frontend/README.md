# Running the app (website + Python server)

Commands are for PowerShell, one per line, starting in the repo root.

**Once:**

```powershell
cd frontend
npm install
cd ..
python -m venv backend\.venv
backend\.venv\Scripts\python -m pip install -r gee_layers\requirements.txt -e gee_layers
backend\.venv\Scripts\earthengine authenticate          # Google login in the browser
Copy-Item gee_layers\.env.example gee_layers\.env
notepad gee_layers\.env                                 # set EE_PROJECT=<your Earth Engine Cloud project id>
```

**Every time:**

```powershell
cd frontend
npm run dev                                             # then open http://localhost:5173
```

Draw a polygon in the contiguous US (some layers cover only there), press **Finish**, then **Extract Data**: the points and the
`get_fire_inputs` summary print in the `[api]` lines of that terminal. Stop with `Ctrl+C`.

Notes: the start time is 2 days ago (UTC, so all data already exists) with 3 hours of weather, until the UI has a time picker.
`npm run dev:web` runs the website alone, without the server; `python backend\server.py` runs the server alone (port 8000).
