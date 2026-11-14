# psagravity: Wildfire Spread Prediction with NISAR

<div dir="rtl">

## בקצרה (קריאה של 3 דקות)

### מה הפרויקט עושה
שריפה כבר בוערת: איפה היא תהיה בעוד כמה שעות? נותנים למערכת את נקודת ההצתה והגודל, והיא חוזה את ההתפשטות.
המודל הוא cellular automaton הסתברותי, שהפרמטרים שלו נלמדים משריפות אמיתיות. השכבות האיטיות (דלק, יובש, אזורים שכבר נשרפו) מגיעות מ-**NISAR** (מכ״ם L-band), והשכבות המהירות מלוויינים שרואים אש כל כמה דקות.

### איזה דאטה ולמה
| מקור | מה נותן | למה |
|---|---|---|
| **NISAR GCOV** (ASF) | backscatter HH/HV, כל 12 יום לכל track | רואה דרך עשן ועננים; דלק, יובש, burn scar |
| **VIIRS** 375 m (FIRMS) | נקודות אש + FRP, ~6 מעברים ביום | מיקום מדויק של האש |
| **GOES-18 FDC** (NOAA) | פיקסלי אש + FRP כל 5 דקות, ~7 קמ״ר לפיקסל | תזמון: מתי האש הגיעה לכל מקום |
| **FEDS** (NASA) | perimeter כל 12 שעות | כיוון וצורת ההתפשטות (השטח הוא חסם תחתון) |
| **NIFC WFIGS** | ה-perimeter הרשמי הסופי, תאריכים | ground truth לשטח הסופי |

**החלטת ground truth:** שטח = WFIGS הסופי; תזמון = VIIRS + GOES; FEDS = כיוון וצורה בלבד.

### מה כבר נעשה
- Pipeline אחד לכל שריפה (`real_fires/src/firedata/`): הורדה, שיוך detections לשריפה (ומיון שריפות שכנות), טבלאות, גרפים, README לכל שריפה.
- 4 שריפות בארה״ב: Bug (ראשית), Colwash, Biscar, Crosswhite.
- טבלת סיכום עם אחוזי התאמה מול ה-perimeter הרשמי: [`real_fires/docs/fires_summary.md`](real_fires/docs/fires_summary.md).
- אתר אחד לכל השריפות עם ציר זמן: [`real_fires/app/index.html`](real_fires/app/index.html).
- **האפליקציה עובדת מקצה לקצה:** מציירים פוליגון של האש על המפה, בוחרים שעת התחלה (UTC) ולוחצים **Predict**. אחרי כדקה מופיעה מפת חום של הסיכוי שכל תא יישרף, עם כיוון ההתפשטות הסביר ונתוני הקלט ששימשו. עובד בכל העולם, לא רק בארה״ב. הוראות: [`frontend/README.md`](frontend/README.md).
- מודל Alexandridis מלא (`spread_model/`) עם 200 הרצות Monte Carlo, ושכבות קלט מ-Google Earth Engine (`gee_layers/`): גובה, כיסוי קרקע, מזג אוויר, וצלקות שריפה מ-Sentinel-2.
- NISAR מחובר למודל: כיסוי צמחייה, לחות קרקע ואזורים שנשרפו, קריאה רק של אזור האש (מהיר פי ~5 עם cache).
- פירוט מלא של מה נעשה: [Status](#status-2026-11-14) למטה.
- לא נעשה עדיין: זמן אמיתי (שעות) לכל צעד, כיול הפרמטרים על שריפות אמיתיות ו-hindcast, השוואת burn scar של NISAR (issue #7), שריפות באירופה.

### המספרים העיקריים
| שריפה | שטח רשמי | FEDS IoU | VIIRS covers | GOES: איחור בזיהוי ראשון | NISAR pairs |
|---|---|---|---|---|---|
| Bug, CA/NV | 93,733 acres | 33% | 39% | 0 דק׳ | 3 |
| Colwash, WA | 39,103 acres | 3% | 6% | 48 דק׳ | 4 |
| Biscar, CA | 69,352 acres | 10% | 26% | 120 דק׳ | 4 |
| Crosswhite, OR | 355,065 acres | 64% | 83% | −11 דק׳ (לפני הדיווח) | 4 |

המסקנה: בשריפות מהירות FEDS ו-VIIRS מפספסים חלק גדול מהשטח (Colwash נשרפה בפחות מיום; VIIRS ראה רק 6%). בשריפה ארוכה כמו Crosswhite הם מכסים את רוב השטח. שימו לב: Crosswhite היא איחוד של כמה שריפות ברק (כמה נקודות הצתה). GOES רואה כמעט הכל ומהר, אבל בפיקסלים גסים. לכן משלבים: GOES לזמן, VIIRS למיקום, WFIGS לשטח הסופי. ההגדרות המדויקות של כל אחוז: [`real_fires/docs/fires_summary.md`](real_fires/docs/fires_summary.md).

### איך מריצים

</div>

```bash
cd real_fires
uv venv .venv && uv pip install -r requirements.txt -e .   # or: python -m venv .venv
cp .env.example .env                                        # FIRMS_MAP_KEY (+ Earthdata for NISAR)
.venv/bin/python scripts/build_ground_truth.py colwash      # one fire: data, tables, figures
.venv/bin/python -m firedata.report readme colwash          # fire README
.venv/bin/python -m firedata.report summary bug_fire colwash biscar crosswhite
.venv/bin/python -m firedata.web bug_fire colwash biscar crosswhite --app   # the site in real_fires/app/
```

<div dir="rtl">

הורדה ראשונה של שריפה לוקחת 5–40 דקות (בעיקר GOES). אחר כך הכל מה-cache ב-`real_fires/data/fires/<fire>/raw/` (לא נשמר ב-git).

### איפה מה נמצא
| נתיב | מה |
|---|---|
| `real_fires/src/firedata/` | קוד: `firms`, `goes`, `feds`, `wfigs`, `nisar`, `process`, `metrics`, `report`, `web`, `config` (שריפה חדשה = `FireConfig` חדש) |
| `real_fires/scripts/build_ground_truth.py` | מריץ את כל ה-pipeline לשריפה אחת |
| `real_fires/data/fires/<fire>/` | `processed/` (טבלאות), `figures/` (גרפים + דף אינטראקטיבי), `README.md` |
| `real_fires/docs/fires_summary.md` | טבלת הסיכום + הגדרות המדדים |
| `real_fires/app/` | האתר |
| `gee_layers/` | שכבות קלט למודל מ-Google Earth Engine: גובה, שיפוע, כיסוי קרקע, דלק, יובש, מזג אוויר (ראו `gee_layers/README.md`) |
| `spread_model/` | מודל Alexandridis: הסתברות התפשטות לכל תא ליד האש, והכיוון הסביר ביותר (נוסחאות ב-`spread_model/README.md`) |
| `frontend/` | האתר (React + MapLibre): ציור פוליגון, Predict, מפת חום. הרצה: `cd frontend` ואז `npm run dev` |
| `backend/` | שרת Python ‏(`POST /api/predict`) שמריץ את כל ה-pipeline על הפוליגון |

</div>

---

*Full project description (English) below.*


**When a wildfire is already burning, where will it go next?**

psagravity takes the location and size of a fire that has already been detected, and predicts how it will spread over the next hours. It combines the state of the ground from **NASA-ISRO NISAR** L-band radar (fuel, dryness, burned areas) with active-fire detections, terrain and wind.

Bonus: an **ignition-risk map** that flags dry, fuel-rich areas close to people before a fire starts.

Built for the NASA Space Apps challenge on tracking Earth's surface changes with NISAR.

---

## The problem

Once a fire is detected, incident managers need to know where it will advance: where to send crews, which communities to evacuate first, and where natural firebreaks are. The answer depends on fuel, fuel moisture, terrain and wind, and most of these change across the landscape.

Optical satellites lose the ground under smoke and clouds, exactly when it matters most. Radar does not.

## Why NISAR

| What NISAR gives us | What it does not |
|---|---|
| Sees through smoke and clouds, day and night | Revisits the same place only every 6–12 days |
| L-band (24 cm) is sensitive to vegetation amount and water content | Standard products arrive 36–72 h after acquisition |
| 10–20 m resolution, dual polarization (HH/HV) | |

So NISAR is used for the layers that change **slowly**: fuel structure, dryness and already-burned areas. Everything that changes hour by hour comes from faster sources.

## How it works

![Architecture](docs/architecture.png)

| Layer | Updates | Content |
|---|---|---|
| Static | once | Slope and aspect (NASADEM), roads and buildings (OSM), vegetation type (HLS) |
| Slow: NISAR | every 6–12 days | Fuel / biomass (HV), dryness and soil moisture, burned areas as firebreaks |
| Fast | hourly | Wind and humidity (Google Weather / MERRA-2), active fire (VIIRS), user-given ignition point |

### Spread model

A probabilistic cellular automaton on a grid, following Alexandridis et al. (2008). The probability that a burning cell ignites its neighbor is:

$$p_{burn} = p_h\,(1+p_{fuel})\,(1+p_{dry})\;p_{wind}\;p_{slope}$$

- `p_wind = exp(c1·V) · exp(c2·V·(cosθ − 1))`, where θ is the angle between the wind and the spread direction
- `p_slope = exp(a·θ_s)`: fire runs faster uphill
- `p_fuel`, `p_dry` come from the NISAR layers and dead fuel moisture

The parameters (`p_h`, `c1`, `c2`, `a`, …) are **learned from real fires** with a differentiable implementation (as in PyTorchFire), and re-tuned during a fire whenever a new perimeter is observed.

### Validation

Hindcast on real fires observed by NISAR since June 2026: feed the model only what was known at the time, then compare the predicted spread with the observed perimeters (FEDS every 12 h, VIIRS, official perimeters). Metrics: IoU, F1, arrival-time error, **with and without the NISAR layers**.

## Data sources

| Data | Source | Provided by the challenge |
|---|---|---|
| NISAR GCOV (L-band backscatter, HH/HV) | [ASF Vertex](https://search.asf.alaska.edu/) | ✅ |
| NISAR SME2 (soil moisture) | ASF | ✅ |
| VIIRS 375 m active fire (location, FRP) | [Earthdata Search](https://search.earthdata.nasa.gov/) / FIRMS | ✅ |
| FEDS fire perimeters (every 12 h) | [NASA FEDS](https://earth-information-system.github.io/fireatlas/docs/) | ✅ (NASA) |
| NASADEM (30 m) | Earthdata Search | ✅ |
| HLS (Landsat + Sentinel-2) | Earthdata Search | ✅ |
| MERRA-2 (historical weather) | Earthdata Search | ✅ |
| SAOCOM L-band (pre-NISAR fires) | [CONAE](https://catalogos.conae.gov.ar/catalogo/catalogoSatSaocomInfo.html) | ✅ (partner) |
| GOES ABI fire detection (every 5 min) | NOAA | external |
| Live wind and weather | Google Weather API | external |
| Roads and buildings | OpenStreetMap | external |

## Study fires

| Fire | Start | Size | NISAR before / after |
|---|---|---|---|
| **Bug Fire, CA/NV** (primary) | 2026-08-08 | 93,733 acres | 07-23, 07-27 / 08-15, 08-16, 08-20 |
| **Ávila, Spain** (backup) | 2026-07-22 | ~50,000 ha | 07-14, 07-21 / 07-26, 08-19 |
| Le Porge, France | 2026-07-22 | 42,000 ha | 07-14, 07-17 / 07-24, 07-26, 08-10 |
| Rowe Creek / Crosswhite, OR | 2026-07-15 | ~343,000 acres | 07-13, 07-15 / 07-18 → 07-27 |

## Status (2026-11-14)

**Working end to end, anywhere in the world:** draw the burning area on the map, set the start time (UTC), press
**Predict**. About a minute later the map shows the chance that each 100 m cell burns within the next 30 model
steps, with the most likely direction, the expected burned area and the inputs that were used. Setup and run:
[`frontend/README.md`](frontend/README.md).

### What is built

**Ground truth from real fires** (`real_fires/`): VIIRS, GOES, FEDS and WFIGS for the Bug, Colwash, Biscar and
Crosswhite fires, with arrival times, perimeters, metrics and a multi-fire site.

**Model inputs from Google Earth Engine** (`gee_layers/`), fetched only for the polygon + 5 km:
- Terrain: NASADEM elevation, slope and aspect, with GMTED filling the gaps beyond NASADEM's coverage (north of
  60°N, for example).
- Land cover: ESA WorldCover worldwide, plus LANDFIRE fuel (cover and height) in the US.
- Weather at the start time: RTMA in the US; ERA5-Land elsewhere; GFS forecasts for the last few days, when
  reanalysis is not available yet.
- One call, `get_fire_inputs(polygon, start)`, returns everything on one grid.
- Works outside the US too.

**Burn scars from Sentinel-2:**
- The model compares the burn index (dNBR) with the same season a year earlier, at 20 m.
- Already-burned land then blocks the fire.
- On the Bug Fire it finds 97% precision and 78% recall against the official perimeter.

**NISAR** (`real_fires/src/firedata/nisar_*`):
- GCOV HH/HV gives vegetation cover and burned areas; SME2 gives soil moisture and vegetation water content.
- Only the chunks over the fire area are read, in parallel, and cached per granule (~5x faster). Gaps in the swath
  are filled from older passes.
- If a granule lacks a layer, or the login fails, the model runs without NISAR and says so.

**Spread model** (`spread_model/`):
- Alexandridis et al. (2008) cellular automaton:
  `P = P0·(1+P_veg)·(1+P_den)·P_wind·P_slope·P_moisture·P_mask`.
  - `P_moisture` combines the equilibrium moisture of the air with NISAR soil moisture.
  - `P_mask` stops the fire at no fuel, water, built-up land, NISAR or Sentinel-2 burn scars, and the open sea.
- A Monte Carlo simulation (200 runs) gives the burn probability and the arrival step for each cell.
- If nothing can spread, the app explains why (for example, "all neighbours are water").

**Web app** (`frontend/` + `backend/`):
- Website (React + MapLibre): draw the polygon, set the UTC start time, press Predict; the heatmap and a summary
  panel appear.
- Python API: `POST /api/predict`.
- `npm run dev` starts both.
- Earth Engine uses the team project by default and opens the Google login on first use.

**Checks:**
- 89 offline tests (`gee_layers`, `spread_model`, `real_fires`, `backend`).
- A live robustness sweep on 13 places, all of which return a valid map:
  - Bug Fire, a recent date with forecast weather, and a tiny polygon smaller than one cell;
  - Nigeria in the wet and dry seasons, and Lagos city;
  - Sahara, Amazon, Alaska, Blue Mountains (Australia) and Greece;
  - open ocean, and the Haifa coast (half sea).

### Ziv's work

- **Model inputs:** `gee_layers`, the Earth Engine inputs (terrain, land cover and fuel, weather, one call per
  polygon), with global support.
- **Spread model:** `spread_model`, the Alexandridis model.
  - Moisture factor from air humidity plus NISAR soil moisture.
  - Masks for no fuel, water, built-up land, burn scars and the open sea.
  - Monte Carlo simulation, and the reasons shown when the map is empty.
- **Sentinel-2 burn scars** (dNBR), included in the model.
- **NISAR in the model:**
  - connected through `nisar_provider`;
  - reads only the fire area, gets SME2 soil moisture and vegetation water content, and fills swath gaps from
    older passes;
  - the ~5x faster parallel, cached reader.
- **Website to model:** `POST /api/predict`, the Predict button, the start-time input, the heatmap and the summary
  panel.
- **Robustness:** the 13-place sweep, which led to these fixes:
  - elevation gaps;
  - missing NISAR layers;
  - fire spreading over the sea;
  - a stale server that kept answering on port 8000.
- **Tests and setup:** offline tests for every part, and Earth Engine that needs no setup.

### Next

1. **Time:** give every step a duration from wind, slope and fuel (Rothermel rate of spread), with random
   variation and hourly weather, so the map shows hours instead of steps.
2. **Calibrate and validate:** fit the parameters to the VIIRS / GOES arrival times of the 4 US fires, then
   hindcast with and without NISAR.
3. NISAR burn-scar comparison (issue #7), and the European fires.

## Roadmap

- [x] Select fires and build ground truth (location, intensity, size vs. time)
- [x] Spread model: spec + input grid
- [ ] Spread model: simulator + learning the parameters (simulator done; learning to do)
- [ ] Spread model: live update + evaluation
- [ ] Ignition-risk map
- [ ] Web app: terrain layers over time, spread animation, risk map (draw + predict + heatmap done)
- [ ] Final report and video

Progress is tracked in the GitHub project **psagravity tasks**.

## Repository structure

```
psagravity/
├── README.md, CLAUDE.md
├── docs/                  # project figures (architecture.png)
├── gee_layers/            # model inputs from Google Earth Engine: terrain, fuels, dryness, weather (see its README)
├── spread_model/          # Alexandridis CA: spread probability per cell + most likely direction (see its README)
├── frontend/              # website: draw the fire, Predict, heatmap (see its README to run the app)
├── backend/               # Python API server (POST /api/predict) used by the website
└── real_fires/            # ground truth from real fires, NISAR reader
    ├── src/firedata/      # pipeline code (package)
    ├── scripts/           # build_ground_truth.py, test_app.py
    ├── data/fires/<fire>/ # processed data, figures, README per fire (raw data not committed)
    ├── docs/              # fires_summary.md / .csv
    └── app/               # web map (all fires)
```

## Getting started

```bash
git clone https://github.com/coolpig123/psagravity.git
cd psagravity/real_fires
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .
```

Accounts needed: [NASA Earthdata](https://urs.earthdata.nasa.gov/) (free), FIRMS MAP_KEY (free).

**To run the prediction app** (website + model): follow [`frontend/README.md`](frontend/README.md). You need a
Google account with access to the team's Earth Engine project (ask the project owner to add you); the Earthdata
login is optional (without it the model runs without NISAR).

## Team

- [name]
- [name]
- [name]

## References

- Rosen et al. NISAR mission. [NISAR Data User Guide (ASF)](https://nisar-docs.asf.alaska.edu/)
- Alexandridis et al. (2008). A cellular automata model for forest fire spread prediction: the case of the wildfire that swept through Spetses Island in 1990. *Applied Mathematics and Computation.*
- Andrews (2018). [The Rothermel surface fire spread model and associated developments](https://research.fs.usda.gov/treesearch/55928). RMRS-GTR-371.
- Xia & Cheng (2025). [PyTorchFire: a GPU-accelerated wildfire simulator with differentiable cellular automata](https://arxiv.org/abs/2502.18738).
- Rochoux et al. (2014, 2015). Towards predictive data-driven simulations of wildfire spread, [Part I](https://nhess.copernicus.org/articles/14/2951/2014/), [Part II](https://nhess.copernicus.org/articles/15/1721/2015/). *NHESS.*
- Rao et al. (2020). [SAR-enhanced mapping of live fuel moisture content](https://doi.org/10.1016/j.rse.2020.111797). *Remote Sensing of Environment.*
- Schroeder et al. (2014). [The New VIIRS 375 m active fire detection data product](https://doi.org/10.1016/j.rse.2013.12.008). *Remote Sensing of Environment.*
- Chen et al. (2022). [California wildfire spread derived using VIIRS satellite observations and an object-based tracking system](https://doi.org/10.1038/s41597-022-01343-0). *Scientific Data.* (FEDS)
- Huot et al. (2022). [Next Day Wildfire Spread](https://arxiv.org/abs/2112.02447). *IEEE TGRS.*
- Gelabert et al. (2025). [Assessing human-caused wildfire ignition likelihood across Europe](https://nhess.copernicus.org/articles/25/4713/2025/). *NHESS.*

## License

[MIT / Apache-2.0: choose one]
