# GeoFlare: AI powered fire detection, classification & Geospatial Analysis Intelligence System

Predict whether there is a fire based on environmental sensor data.

## Folder Structure

```
SIH_2K26/
├── dataset/               # your raw dataset (firDetection.csv)
├── data/
│   ├── raw/                # Original dataset (place fire_data.csv here)
│   ├── preprocessed/       # Cleaned & feature-engineered data (+ timestamp for time splits)
│   ├── split/              # Train/val/test splits + scaler + medians
│   └── images/             # fire/nofire images for image-based detection
├── frontend/               # Web map app (Leaflet + FIRMS + AI panel)
├── models/                 # best_model.pkl, image_det.pth, scaler.pkl, medians.json
├── outputs/
│   └── evaluation/         # Metrics, confusion matrices, ROC curves, feature importance
├── logs/                   # Pipeline run logs
├── notebooks/              # For Jupyter experiments
├── .venv/                  # Project python env (torch + all deps, long-path aware)
├── src/
│   ├── config.py           # Paths & constants
│   ├── features.py         # Shared sanitize/feature-order/engineering logic
│   ├── preprocess.py       # Clean + feature engineering + timestamp
│   ├── split_data.py       # Train/val/test split + scaling (+ time-based option)
│   ├── train.py            # Train multiple tabular models, pick best
│   ├── evaluate.py         # Metrics + plots
│   ├── predict.py          # JSON in -> JSON out with confidence
│   ├── test_samples.py     # Check model on known rows from your dataset
│   ├── server.py           # Flask backend (map data + weather + AI)
│   ├── firms.py            # NASA FIRMS real-time fire fetch
│   ├── weather.py          # Open-Meteo weather + air quality
│   ├── image_det/          # Image fire detection CNN (train/predict/model)
│   └── main.py             # Full pipeline runner
├── requirements.txt
└── README.md
```

## Features Used (14 sensor inputs)

UTC, Temperature[C], Humidity[%], TVOC[ppb], eCO2[ppm], Raw H2, Raw Ethanol, Pressure[hPa], PM1.0, PM2.5, NC0.5, NC1.0, NC2.5, CNT

**Target:** Fire Alarm

During preprocessing these become 19 model features: the 13 numeric readings (UTC is expanded into `hour`/`day_of_week`/`month`) plus 3 ratio features (`temp_humidity_ratio`, `pm_ratio`, `tvoc_eco2_ratio`).

## Quick Start (train)

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Place your dataset as `data/raw/fire_data.csv`

3. Run the full pipeline (random split):
   ```
   python src/main.py
   ```

   Or with a **time-based split** (train on past, validate/test on future):
   ```
   python src/main.py --time-based
   ```

Or run each step manually:
```
python src/preprocess.py --raw data/raw/fire_data.csv --output data/preprocessed/fire_data_processed.csv
python src/split_data.py --input data/preprocessed/fire_data_processed.csv --output data/split --mode time
python src/train.py --split-dir data/split --output models
python src/evaluate.py --split-dir data/split --models-dir models --output outputs/evaluation
```

`--mode random` is the default; `--mode time` disables shuffling and splits by timestamp.

## Prediction (JSON -> JSON)

Feed environmental conditions and get a fire prediction with confidence as JSON.

Inline JSON:
```
python src/predict.py --input "{\"Temperature[C]\": 30.5, \"Humidity[%]\": 20.1, \"TVOC[ppb]\": 900, \"eCO2[ppm]\": 1500, \"PM2.5\": 8.4}"
```

JSON file:
```
python src/predict.py --input-file conditions.json
```

Stdin (pipe):
```
echo '{"Temperature[C]": 30.5, "Humidity[%]": 20.1, ...}' | python src/predict.py
```

Input keys accept either the raw names (`Temperature[C]`) or sanitized names (`Temperature_C`). `UTC` can be a Unix epoch (seconds) or a datetime string. Omitted features are filled with training medians, so you only need to supply the conditions you have.

Output format:
```json
{
  "fire_detected": true,
  "prediction": 1,
  "confidence": 0.847708,
  "probability": {"no_fire": 0.152292, "fire": 0.847708},
  "message": "FIRE DETECTED",
  "input_features": {"Temperature_C": 9.381, "hour": 1.0, ...}
}
```

## Web App (Map + Real-time FIRMS + AI)

A "mission control" style map of India with live NASA FIRMS thermal anomalies.
Clicking an anomaly fetches real weather for that coordinate and runs the ML model
in the background, returning the fire prediction + confidence + weather.

```
frontend/                # index.html, styles.css, app.js, india.geojson
src/server.py            # Flask API: /api/fires, /api/weather, /api/predict, static frontend
src/firms.py             # NASA FIRMS CSV fetch/parse (VIIRS S-NPP NRT, India)
src/weather.py           # Open-Meteo current weather + air quality (no API key needed)
```

### Run

Start the server with the project venv so the image endpoint can use torch too:

```
.venv\Scripts\python.exe src\server.py --port 5000
```

Then open http://localhost:5000

- Get a free NASA FIRMS map key at https://firms.modaps.eosdis.nasa.gov/api/map_key/
- Put it in `.env` as `FIRMS_MAP_KEY=...` (loaded automatically), or `--firms-key YOUR_KEY`.
- No key? Start with `--demo` to see synthetic markers; the UI also has a Demo toggle.

### Flow

1. Map loads with Indian state boundaries + NASA FIRMS hotspots (VIIRS S-NPP NRT, last 2 days), colored by FRP intensity.
2. Click a thermal anomaly → frontend calls `POST /api/predict {lat, lon, utc}`.
3. Backend fetches live weather (temp, humidity, pressure) + air quality (PM2.5, PM10) from Open-Meteo for that exact location.
4. Features are assembled (real weather values + training medians for sensor-only fields) and the ML model runs.
5. Panel shows: 🔥 verdict, confidence gauge, fire/no-fire probability, full weather, AQI, and the exact feature values used.
6. The panel also has an **Image detection** upload box — drop a fire/satellite photo and the CNN (`models/image_det.pth`) detects fire independently.

Backend endpoints:
- `GET /api/fires?days=2` — FIRMS hotspots (uses `.env` key; `&key=` also accepted)
- `GET /api/weather?lat=&lon=` — live weather + AQI
- `POST /api/predict` `{"lat":…,"lon":…,"utc":…}` — AI prediction JSON
- `POST /api/image/predict` (multipart `image` field) — fire/no-fire via CNN

## Image based Fire Detection (binary)

A separate CNN detects fire vs no-fire from images (works with satellite or RGB photos).
It is independent of the tabular model above.

```
data/images/                # images for image-based detection
  train/{fire,nofire}/*.jpg
  val/{fire,nofire}/*.jpg    (optional — created from train if missing)
  test/{fire,nofire}/*.jpg
src/image_det/
  train_image.py             # train the detection CNN
  predict_image.py           # run detection on an image / folder
  make_smoke_data.py         # generates synthetic images for a quick smoke test
models/image_det.pth         # trained weights + class_names
```

Uses a local Python environment that supports long paths (the Windows-Store Python can't
install torch). All commands below use the project venv:

```
.venv\Scripts\python.exe src\image_det\make_smoke_data.py        # synthetic smoke-test images
.venv\Scripts\python.exe src\image_det\train_image.py --data data\images --epochs 15
.venv\Scripts\python.exe src\image_det\predict_image.py test.jpg
.venv\Scripts\python.exe src\image_det\predict_image.py data\images\test --json
```

### Real dataset

Any `fire/` + `nofire/` image folders work. Recommended free datasets:

- The Wildfire Dataset (2,700 aerial/ground images): https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset
- DFS Fire & Smoke (9,462 boxes): https://github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset
- Satellite benchmarks: TS-SatFire — https://www.nature.com/articles/s41597-025-06271-3

Download, extract, then drop images into `data/images/train/{fire,nofire}` etc. and retrain.

## Models Trained

Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM. The best model is saved as `models/best_model.pkl`.

- **Random split**: val metrics pick the best model (F1 on validation).
- **Time-based split**: if validation has a single class (common with chronological splits), selection falls back to test F1 with a warning.

## Notes

- `Unnamed: 0` index columns are dropped automatically.
- Column names with `[`/`]` (e.g. `Temperature[C]`) are sanitized because XGBoost/LightGBM reject them.
- UTC is parsed as Unix seconds (or ISO string) into `hour`/`day_of_week`/`month`; a raw `timestamp` column is kept only for time-based splitting and is never used as a model feature.
- The real 100% score on a random split is expected on this dataset but misleading (temporally duplicated rows). Always prefer `--time-based` for realistic performance.