# GeoFlare

**AI-powered wildfire detection, classification & geospatial intelligence system**

GeoFlare is an end-to-end fire-intelligence platform that fuses **NASA FIRMS satellite thermal data**, **live weather & air quality**, **OSM land-use**, and **satellite imagery (CNN)** to detect fires and classify them as **agricultural**, **forest**, or **industrial** — all surfaced on a live mission-control map with a free-tier cloud deployment.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-3B7DD8?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/ml-random_forest_fusion-F7B955" alt="ML">
  <img src="https://img.shields.io/badge/db-PostGIS-316192?logo=postgresql&logoColor=white" alt="PostGIS">
  <img src="https://img.shields.io/badge/map-Leaflet-199900?logo=mapbox" alt="Leaflet">
  <img src="https://img.shields.io/badge/api-Flask-000000?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/deploy-Render%20(Free)-46E3B7?logo=render" alt="Render">
  <img src="https://img.shields.io/badge/docker-ready-2496ED?logo=docker" alt="Docker">
</p>

---

## Highlights

- Live thermal-anomaly map of India with real **NASA FIRMS** (VIIRS) hotspots
- **Three-source fused detection**: FIRMS + environment (tabular RF), satellite-image CNN, and OSM land-use
- **Fire-type classification** into agricultural / forest / industrial with confidence
- Live **Open-Meteo weather + air quality** per hotspot — no API key required
- Image upload: fire/no-fire CNN and fused environmental + image verdicts
- **PostGIS** persistence with a zero-dependency GeoJSON fallback store
- Free-tier **Render** deployment (CPU-only torch, Docker image) with health checks
- Clean-up on cold start, lazy model loading, graceful fallbacks for every external service

## Table of Contents

- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Run the web app](#run-the-web-app)
- [API reference](#api-reference)
- [Retraining the models](#retraining-the-models)
- [Cloud deployment](#cloud-deployment)
- [Testing & validation](#testing--validation)
- [Dataset notes](#dataset-notes)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## Architecture

```
 ┌─────────────────────────────── VISUALISATION ───────────────────────────────┐
 │  Frontend (Leaflet map + FIRMS overlay + AI drawer)  ───  frontend/          │
 └───────────────▲──────────────────────────────────────────────┬──────────────┘
                 │  GET /api/fires · POST /api/predict           │  POST /api/predict/fused
                 │                                               │  (image upload)
 ┌───────────────┴──────────────────────────────────────────────▼──────────────┐
 │  Flask backend  ────────────────────────────────────────────────  src/server.py│
 │                                                                             │
 │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────┐ │
 │   │   FIRMS       │   │   Open-Meteo  │   │   OSM /       │   │   Esri    │ │
 │   │  hotspots     │   │  weather+AQI  │   │   Overpass    │   │ satellite │ │
 │   │  src/firms.py │   │  src/weather  │   │ src/geo_split │   │   tile    │ │
 │   └──────┬────────┘   └──────┬────────┘   └──────┬────────┘   └─────┬─────┘ │
 │          │                   │                   │                 │       │
 │      FIRMS cluster     temperature ·       land-use shares     image CNN     │
 │      statistics        humidity · PM2.5   (agri/forest/ind)   fire type      │
 │          │                   │                   │                 │       │
 │          └───────────────────┴───────────────────┴─────────────────┘       │
 │                                  │                                          │
 │                      Weighted fusion (0.5/0.35/0.15)                        │
 │                                  │                                          │
 │                    Fire? no ──► reply (no fire)                             │
 │                    Fire? yes ──► fire-type classification ──► persist       │
 │                                                                             │
 │   ┌────────────────────────────────────────────────────────────┐             │
 │   │  Persistence: PostGIS  (supabase/self-host)               │             │
 │   │  · geom point · class · confidence · temp/hum/co2/pm      │             │
 │   │  Fallback: local GeoJSON store  (data/gis)                │             │
 │   └────────────────────────────────────────────────────────────┘             │
 └───────────────────────────────────────────────────────────────────────────────┘
```

## How it works

### 1. Detection (fire or no-fire) — three sources fused

For every click on a hotspot, GeoFlare assembles three independent signals and
blends them with fixed weights (missing sources are renormalized automatically):

| Vote | Source | Weight | Signal |
|------|--------|--------|--------|
| Tabular | NASA FIRMS cluster statistics + weather/environment | **0.50** | Random Forest trained on 14 sensor + FIRMS features |
| Image | Esri satellite tile through a fire CNN | **0.35** | P(fire) from `image_det.pth` |
| OSM | OpenStreetMap land-use within ~3 km | **0.15** | Flammable land share (forest + agri + industrial) |

```
p = 0.50 · tabular_fire + 0.35 · image_fire + 0.15 · osm_fire     fire if p ≥ 0.5
```

### 2. Classification (agricultural / forest / industrial)

Only meaningful for confirmed fires. The classifier fuses trained tabular models
with OSM and a `fire_type.pth` CNN:

- `agricultural_fire_random_forest_v2` → P(agricultural) from FIRMS cluster stats within 5 km
- `shared_fire_model` (temp, humidity) → P(forest) vs P(industrial)
- OSM land-use proximity → fallback forest/industrial/agricultural shares, also used as detection vote
- `fire_type.pth` CNN → forest vs industrial when the image fires

The winning class (renormalized) is stored with its confidence.

### 3. Persistence

Classified events are saved to a PostGIS database (latitude/longitude as a
geometry point, plus class, confidence, temperature, humidity, CO₂, PM2.5 and
raw evidence). If no database is reachable the app transparently falls back to a
local GeoJSON store so the map overlay always works.

---

## Repository layout

```
SIH_2K26/
├── frontend/                 # Leaflet map app (index.html, styles.css, app.js)
├── src/
│   ├── server.py             # Flask API: fires, weather, predict, fused, detections
│   ├── firms.py              # NASA FIRMS real-time hotspot fetch/parse
│   ├── weather.py            # Open-Meteo current weather + air quality
│   ├── geo_split.py          # OSM land-use shares via Overpass (3 endpoints, throttled)
│   ├── fire_classifier.py    # Fire-type classification (agri/forest/industrial)
│   ├── database.py           # PostGIS storage with GeoJSON fallback
│   ├── predict.py            # JSON in → JSON out with confidence + fusion
│   ├── features.py           # Sanitize / feature-order / engineering
│   ├── preprocess.py         # Cleaning + feature engineering + timestamps
│   ├── split_data.py         # Train/val/test splits + scaling
│   ├── train.py              # Train multiple tabular models, pick best
│   ├── evaluate.py           # Metrics + plots
│   ├── main.py               # Full pipeline runner
│   └── image_det/            # Image CNN: train / predict / model
├── scripts/                  # Dataset building, training, evaluation
├── models/                   # best_model.pkl, scaler.pkl, medians.json, *.pth
├── data/                     # raw, preprocessed, split, imagery, gis store
├── outputs/                  # evaluation reports, confusion matrices, ROC curves
├── notebooks/                # Jupyter experiments
├── logs/                     # pipeline run logs
├── Dockerfile                # CPU-only torch production image
├── render.yaml               # Render Blueprint (free-tier deploy)
├── requirements.txt          # local dev dependencies
└── requirements-deploy.txt   # minimal runtime dependencies for the container
```

## Quick start

### Local development

```bash
# 1. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env           # add your FIRMS_MAP_KEY (optional)

# 4. Prepare, train and evaluate the tabular pipeline
python scripts/prepare_data.py
python scripts/train_forest.py
python scripts/train_industrial.py
python scripts/evaluate_models.py
python -m src.inference
```

Or run the end-to-end tabular pipeline directly:

```bash
python src/main.py                # random split
python src/main.py --time-based   # temporal split (more realistic)
```

## Run the web app

Start the Flask backend with the project venv (it uses torch for the image CNN):

```bash
.venv\Scripts\python.exe src\server.py --port 5000
```

Then open **http://localhost:5000**

- Get a free NASA FIRMS key: <https://firms.modaps.eosdis.nasa.gov/api/map_key/>
- Put it in `.env` as `FIRMS_MAP_KEY=...` (auto-loaded), or pass `--firms-key YOUR_KEY`.
- No key? Start with `--demo` for synthetic markers — the UI also has a demo toggle.
- Point the backend at PostGIS with `GIS_BACKEND=postgis` and a `DATABASE_URL`
  (or leave it unset to use the built-in GeoJSON store).

### Frontend flow

1. Map loads Indian state boundaries + FIRMS thermal anomalies (VIIRS, last 2 days), colored by FRP.
2. Click a hotspot → `POST /api/predict {lat, lon, utc}`.
3. Backend fetches live weather + air quality (Open-Meteo) for that exact location.
4. Features are assembled (live values + training medians for sensor-only fields) and the fused model runs.
5. The panel shows fire/no-fire verdict, confidence, probabilities, fire type, full weather, AQI and the exact features used.
6. The panel's image box posts a photo to `/api/predict/fused`, which runs the environmental model **and** the image CNN and returns both plus the fused verdict.

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Liveness + `model_loaded` / `firms_configured` status |
| GET | `/api/config` | Client config (firms status, bbox, demo flags) |
| GET | `/api/fires?days=2` | FIRMS hotspots (uses `.env` key, `&key=` also accepted) |
| GET | `/api/weather?lat=&lon=` | Live weather + air quality (Open-Meteo) |
| POST | `/api/predict` | `{"lat","lon","utc"}` → fused detection + classification JSON |
| POST | `/api/image/predict` | multipart `image` → fire/no-fire via CNN |
| POST | `/api/predict/fused` | multipart `lat,lon,utc,features,image` → combined environmental + image verdict |
| GET | `/api/detections?limit=200` | Persisted classified events as GeoJSON FeatureCollection |

### Example — `/api/predict`

```json
{
  "lat": 21.18, "lon": 72.83, "utc": 1600000000
}
```

```json
{
  "fire_detected": true,
  "prediction": 1,
  "confidence": 0.669,
  "probability": {"no_fire": 0.331, "fire": 0.669},
  "message": "FIRE DETECTED",
  "classification": {
    "classification": "industrial",
    "confidence": 0.551,
    "probabilities": {"agricultural": 0.394, "forest": 0.055, "industrial": 0.551},
    "source": "trained models + OSM land-use",
    "details": ["agri_rf=0.51 ...", "shared_rf=0.00 forest ...", "osm_landuse=..."],
    "evidence": {"firms": {...}, "osm": {...}, "shared": {...}, "image": {...}}
  },
  "inputs": { "firms": {...}, "osm": {...}, "satellite": {...}, "detection": {...} }
}
```

### Example — `/api/predict/fused` (image + environment)

```json
{
  "tabular": { "fire_detected": true, "probability": {"fire": 0.87, "no_fire": 0.13}, "...": "..." },
  "image":   { "fire_detected": true, "probability": {"fire": 0.87, "no_fire": 0.13} },
  "fused":   {
    "fire_detected": true,
    "probability": {"fire": 0.74, "no_fire": 0.26},
    "confidence": 0.74,
    "weights": {"tabular": 0.6, "image": 0.4}
  }
}
```

---

## Retraining the models

### Image fire detection (binary)

The image CNN detects fire vs no-fire from satellite or RGB photos, independent
of the tabular model:

```bash
.venv\Scripts\python.exe src\image_det\make_smoke_data.py                # synthetic smoke-test images
.venv\Scripts\python.exe src\image_det\train_image.py --data data\images --epochs 15
.venv\Scripts\python.exe src\image_det\predict_image.py test.jpg
.venv\Scripts\python.exe src\image_det\predict_image.py data\images\test --json
```

Any `fire/` + `nofire/` folder works as data. Recommended free datasets:

- The Wildfire Dataset — <https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset>
- DFS Fire & Smoke (9,462 boxes) — <https://github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset>
- Satellite benchmark: TS-SatFire — <https://www.nature.com/articles/s41597-025-06271-3>

### Tabular fire type models

```bash
python scripts/train_agriculture_model_v2.py        # agricultural RF (FIRMS cluster stats)
python scripts/train_shared_model.py                # forest vs industrial (temp, humidity)
python scripts/train_image_model.py                 # fire-type CNN (fire_type.pth)
```

Trained artifacts land in `models/` and are baked into the Docker image.

## Cloud deployment

Free-tier deployment is handled by **Render** through `render.yaml`
(a Blueprint that builds the Docker image and sets the service env vars).

### Deploy

1. Push `render.yaml` to your repo on a branch you deploy from (or **New → Blueprint** from the dashboard).
2. Render builds the CPU-only torch image and exposes `gunicorn` on `$PORT`.
3. Set the required environment variables in the service:
   - `FIRMS_MAP_KEY` — NASA FIRMS map key (optional, disables demo fallback)
   - `DATABASE_URL` — `postgres://...` Supabase/PostGIS connection (optional)
   - `GIS_BACKEND=postgis` — enables the PostGIS store (defaults to GeoJSON failover)
4. The health check at `/api/health` marks the service live.

```yaml
# render.yaml (excerpt)
services:
  - type: web
    name: geoflare-firepredict
    runtime: docker
    plan: free
    dockerfilePath: ./Dockerfile
    autoDeploy: true
    healthCheckPath: /api/health
    envVars:
      - key: FIRMS_MAP_KEY        # sync: false → set in dashboard
      - key: DATABASE_URL         # sync: false → set in dashboard
      - key: GIS_BACKEND
        value: postgis
```

### Notes for the free tier

- **Cold starts**: the free instance sleeps after inactivity; the first request
  can take ~20–30 s while models lazy-load.
- **IPv6**: some Supabase hosts resolve to IPv6 only. Render free tier lacks IPv6
  routing — use the provider's IPv4 **session-pooler** endpoint when that happens.
- Models and dependencies are pre-baked into the image, so the app runs without
  any runtime downloads.

## Testing & validation

- `python -m pytest` — unit tests (add your own as the suite grows)
- `python scripts/evaluate_models.py` — tabular model metrics (precision, recall, F1, ROC/PR-AUC)
- `python scripts/test_inference.py` — smoke test the JSON→JSON inference path
- The web UI itself doubles as an integration harness: click any hotspot, review
  the fused votes, fire type and weather panel.

## Dataset notes

- FIRMS **VIIRS S-NPP NRT** products (last 2 days) are fetched live at request time.
- The sensor dataset used for the tabular model is the 14-input fire-environment
  CSV (`Temperature[C]`, `Humidity[%]`, TVOC, eCO₂, PM1.0, PM2.5, NC0.5/1.0/2.5,
  pressure, H₂, ethanol, CNT) with UTC expanded into `hour`/`day_of_week`/`month`
  plus engineered ratios (`temp_humidity_ratio`, `pm_ratio`, `tvoc_eco2_ratio`) → 19 features.
- Omitted features during inference are filled with training medians.
- A naive random split routinely scores ~100% on this dataset because rows are
  temporally duplicated — prefer `--time-based` splits for honest numbers.

## Roadmap

- Live forecast integration (wind, drought indices) into the fusion
- Spatial clustering + change detection across FIRMS acquisitions
- Annotated evaluation on freshly labelled Indian wildfire ground truth

## Acknowledgements

- **NASA FIRMS** — realtime VIIRS thermal anomaly data
- **Open-Meteo** — free weather and air-quality APIs
- **OpenStreetMap / Overpass** — land-use intelligence
- **Esri World Imagery** — satellite basemaps served to the CNN
- **Leaflet** — the interactive map frontend

---

<p align="center">
  Built for the Smart India Hackathon 2026 · <b>GeoFlare — see the fire before it spreads</b>
</p>