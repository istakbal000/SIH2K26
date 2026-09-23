import os
import sys
import json
import time
import threading
import random
import logging
from pathlib import Path
from functools import lru_cache

from flask import Flask, jsonify, request, send_from_directory

from firms import fetch_fires, FirmsError
from firms import DEFAULT_BBOX
from weather import get_weather, get_air_quality, weather_to_features
from predict import predict_dict, sanitize, fuse_predictions
import predict as predict_module
from fire_classifier import get_classifier, class_meta
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


def load_dotenv():
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


load_dotenv()

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                info_path = BASE_DIR / "models/best_model_info.json"
                try:
                    name = json.loads(info_path.read_text()).get("model_name", "best_model")
                except Exception:
                    name = "best_model"
                _model = {
                    "name": name,
                    "model": predict_module.joblib.load(BASE_DIR / "models/best_model.pkl"),
                    "scaler": predict_module.joblib.load(BASE_DIR / "models/scaler.pkl"),
                    "medians": json.loads((BASE_DIR / "models/medians.json").read_text()),
                }
    return _model


def firms_key_from_request():
    key = request.args.get("key")
    if key:
        return key
    key = os.environ.get("FIRMS_MAP_KEY")
    if key:
        return key
    return None


_fires_cache = {"ts": 0, "data": None}
_db = {"ready": None, "module": None}


def db() -> "module or None":
    """Lazily import the merged repo's PostGIS module. Returns None if unavailable."""
    if _db["module"] is not None:
        return _db["module"]
    if _db["ready"] is False:
        return None
    try:
        import database
        _db["module"] = database
        try:
            database.init_db()
        except Exception as e:
            logger.warning("PostGIS init failed (%s) — detections will not persist", e)
        return database
    except Exception as e:
        logger.warning("PostGIS driver unavailable (%s) — install psycopg[binary] + run PostgreSQL to enable storage", e)
        _db["ready"] = False
        return None


def classify_and_persist(lat: float, lon: float, utc, weather_result: dict, fire: bool, source: str = "map_analyze"):
    """Run fire-type classification and store the event in PostGIS when a fire is detected.

    Returns a dict with 'classification' (and optionally 'saved'), or None when no fire
    was detected (classification is only meaningful for confirmed fires).
    """
    if not fire:
        return None

    hotspots = _fires_cache.get("data") or []
    classifier = get_classifier()
    cls = classifier.classify(lat, lon, utc, hotspots, weather=(weather_result or {}).get("weather"))
    event = {
        "classification": cls["classification"],
        "confidence": cls["confidence"],
        "probabilities": cls["probabilities"],
        "class_meta": class_meta(cls["classification"]),
        "source": cls["source"],
        "details": cls["details"],
    }

    dbmod = db()
    if dbmod is not None:
        w = (weather_result or {}).get("weather") or {}
        aq = (weather_result or {}).get("air_quality") or {}
        saved = dbmod.save_user_detection(
            lat=lat, lon=lon,
            scenario=None,
            classification=cls["classification"],
            confidence=cls["confidence"],
            temp=w.get("temperature_c"),
            hum=w.get("humidity_pct"),
            co2=w.get("co2_ppm"),
            pm=aq.get("pm2_5"),
            source=source,
            raw_data={
                "fire_detected": fire,
                "probabilities": cls["probabilities"],
                "details": cls["details"],
            },
        )
        event["saved"] = saved
    return event


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


@app.route("/api/health")
def health():
    loaded = _model is not None
    return jsonify({"status": "ok", "model_loaded": loaded, "firms_configured": bool(firms_key_from_request())})


@app.route("/api/config")
def api_config():
    configured = app.config.get("DEMO", False) is not True and bool(firms_key_from_request())
    return jsonify({
        "firms_configured": configured,
        "default_source": "firms" if configured else "demo",
        "firms_bbox": os.environ.get("FIRMS_BBOX") or DEFAULT_BBOX,
        "firms_days": 2,
    })


@app.route("/api/fires")
def fires():
    key = firms_key_from_request()
    days = int(request.args.get("days", 2))
    bbox = request.args.get("bbox") or os.environ.get("FIRMS_BBOX") or DEFAULT_BBOX

    if request.args.get("demo") == "1" or not key:
        logger.info("Serving DEMO fires (no FIRMS key)")
        return jsonify({"source": "demo", "count": 30, "fires": _demo_fires()})

    now = time.time()
    if _fires_cache["ts"] > now - 60 and _fires_cache["data"] and request.args.get("refresh") != "1":
        return jsonify({"source": "firms", "count": len(_fires_cache["data"]), "fires": _fires_cache["data"]})

    try:
        data = fetch_fires(key, bbox=bbox, days=days)
        _fires_cache.update(ts=now, data=data)
        return jsonify({"source": "firms", "count": len(data), "fires": data})
    except FirmsError as e:
        return jsonify({"source": "error", "error": str(e)}), 502


@app.route("/api/weather")
def weather():
    lat, lon = _parse_coords()
    if lat is None:
        return jsonify({"error": "lat and lon required"}), 400
    try:
        w = get_weather(lat, lon)
        aq = get_air_quality(lat, lon)
        return jsonify({"weather": w, "air_quality": aq})
    except Exception as e:
        return jsonify({"error": f"Weather service error: {e}"}), 502


def assemble_features(lat: float, lon: float, utc, supplied: dict) -> tuple:
    """Fetch live weather + air quality and assemble the input feature dict for the tabular model."""
    weather_result = {"weather": None, "air_quality": None}
    try:
        w = get_weather(lat, lon)
        aq = get_air_quality(lat, lon)
        weather_result = {"weather": w, "air_quality": aq}
        logger.info("  [weather] live Open-Meteo -> temp=%s C  humidity=%s%%  pressure=%s hPa",
                    w.get("temperature_c"), w.get("humidity_pct"), w.get("pressure_hpa"))
        logger.info("  [air]    PM2.5=%s  PM10=%s  US-AQI=%s", aq.get("pm2_5"), aq.get("pm10"), aq.get("aqi"))
    except Exception as e:
        logger.warning("  [weather] unavailable (%s) — will use training medians", e)

    features = {}
    if weather_result["weather"] and weather_result["air_quality"]:
        features = weather_to_features(weather_result["weather"], weather_result["air_quality"])

    for k, v in (supplied or {}).items():
        try:
            features[k] = float(v)
        except (TypeError, ValueError):
            if isinstance(v, (int, float)):
                features[k] = v

    if utc is not None:
        features["UTC"] = int(float(utc))
    elif weather_result["weather"] and weather_result["weather"].get("observation_time"):
        try:
            from datetime import datetime, timezone
            obs = datetime.fromisoformat(weather_result["weather"]["observation_time"].replace("Z", "+00:00"))
            features["UTC"] = int(obs.timestamp())
        except Exception:
            features["UTC"] = int(time.time())
    return features, weather_result


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    lat, lon = _parse_coords(body)
    if lat is None:
        return jsonify({"error": "lat and lon are required"}), 400

    utc = body.get("utc")
    logger.info("=" * 62)
    logger.info("  ML RUN START  |  thermal anomaly at (%.4f, %.4f) utc=%s", lat, lon, utc)
    logger.info("=" * 62)

    features, weather_result = assemble_features(lat, lon, utc, body.get("features"))

    art = get_model()
    logger.info("  [model]  loaded -> %s", art["name"])

    result = predict_dict(features, art["model"], art["scaler"], art["medians"])

    live = {sanitize(k) for k in features} | {"hour", "day_of_week", "month"}
    for k, v in result["input_features"].items():
        src = "LIVE " if k in live else "median"
        logger.info("  [feat]   %-20s = %-10s (%s)", k, v, src)

    logger.info("  [infer]  class proba -> NO_FIRE=%.4f  FIRE=%.4f",
                result["probability"]["no_fire"], result["probability"]["fire"])
    logger.info("  [infer]  verdict -> %s  (confidence %.1f%%)",
                result["message"], result["confidence"] * 100)
    logger.info("  [done]   reply sent to map UI")
    logger.info("-" * 62)

    classification = classify_and_persist(lat, lon, features.get("UTC"), weather_result,
                                          fire=result["fire_detected"], source="map_analyze")
    if classification:
        logger.info("  [classify] detected fire type -> %s (%.1f%%)  %s",
                    classification["classification"], classification["confidence"] * 100,
                    classification.get("saved", {}).get("saved") and "saved to GIS" or "")

    return jsonify({
        "fire_detected": result["fire_detected"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probability": result["probability"],
        "message": result["message"],
        "coordinate": {"latitude": lat, "longitude": lon, "utc": features.get("UTC")},
        "weather": weather_result["weather"],
        "air_quality": weather_result["air_quality"],
        "features_used": result["input_features"],
        "classification": classification,
    })


_IMAGE_MODEL = None
_image_model_lock = threading.Lock()


def get_image_model():
    """Load src/image_det model once. Returns (model, class_names) or None if unavailable."""
    global _IMAGE_MODEL
    if _IMAGE_MODEL is None:
        with _image_model_lock:
            if _IMAGE_MODEL is None:
                try:
                    sys.path.insert(0, str(BASE_DIR / "src"))
                    from image_det.model import load_checkpoint
                    model, class_names, _ = load_checkpoint(BASE_DIR / "models" / "image_det.pth")
                    _IMAGE_MODEL = (model, class_names)
                except Exception as e:
                    logger.warning("Image model unavailable (%s)", e)
                    _IMAGE_MODEL = False
    return _IMAGE_MODEL or None


def run_image_predict(file):
    """Run the image CNN on an uploaded file. Returns a fire/no-fire result dict."""
    import io
    import torch
    import torchvision
    from PIL import Image

    art = get_image_model()
    if art is None:
        raise RuntimeError("Image model not available; run src/image_det/train_image.py first")
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    model, class_names = art
    with torch.no_grad():
        tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logits = model(tf(img).unsqueeze(0))
        proba = torch.softmax(logits, 1)[0]
    p_fire = float(proba[class_names.index("fire")])
    fire = bool(p_fire >= 0.5)
    return {
        "fire_detected": fire,
        "prediction": int(fire),
        "confidence": round(p_fire if fire else 1 - p_fire, 6),
        "probability": {
            "no_fire": round(1 - p_fire, 6),
            "fire": round(p_fire, 6),
        },
        "message": "FIRE DETECTED" if fire else "NO FIRE",
    }


@app.route("/api/image/predict", methods=["POST"])
def image_predict():
    """Detect fire in an uploaded image. Returns fire/no-fire + confidence."""
    if get_image_model() is None:
        return jsonify({"error": "Image model not available. Run image training first, then restart server with .venv python."}), 501

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "field 'image' (file) is required"}), 400

    try:
        return jsonify(run_image_predict(file))
    except Exception as e:
        return jsonify({"error": f"Image prediction failed: {e}"}), 500


@app.route("/api/predict/fused", methods=["POST"])
def predict_fused():
    """Combined detection: environmental parameters (tabular model, from form fields) + satellite image (CNN)."""
    body = {k: v for k, v in request.form.items()}
    if request.files.get("image") is None:
        return jsonify({"error": "field 'image' (file) is required"}), 400

    lat, lon = _parse_coords(body)
    if lat is None:
        return jsonify({"error": "lat and lon are required"}), 400

    logger.info("=" * 62)
    logger.info("  FUSED RUN START  |  env-params + image at (%.4f, %.4f)", lat, lon)
    logger.info("=" * 62)

    supplied = body.get("features")
    if isinstance(supplied, str):
        try:
            supplied = json.loads(supplied)
        except Exception:
            supplied = {}

    features, weather_result = assemble_features(lat, lon, body.get("utc"), supplied)

    art = get_model()
    logger.info("  [model]  tabular -> %s", art["name"])
    tabular = predict_dict(features, art["model"], art["scaler"], art["medians"])
    tabular["coordinate"] = {"latitude": lat, "longitude": lon, "utc": features.get("UTC")}
    tabular["weather"] = weather_result["weather"]
    tabular["air_quality"] = weather_result["air_quality"]
    tabular["features_used"] = tabular.pop("input_features")

    logger.info("  [infer]  tabular -> %s (%.2f%%)", tabular["message"], tabular["probability"]["fire"] * 100)
    try:
        image = run_image_predict(request.files["image"])
        logger.info("  [infer]  image CNN -> %s (%.2f%%)", image["message"], image["probability"]["fire"] * 100)
    except Exception as e:
        logger.warning("  [infer]  image CNN failed (%s)", e)
        return jsonify({"error": f"Image prediction failed: {e}"}), 500

    fused = fuse_predictions(tabular, image)
    logger.info("  [fuse]   p=%.3f*%.2f + %.3f*%.2f => %.3f  (>0.5 fire)",
                fused["weights"]["tabular"], tabular["probability"]["fire"],
                fused["weights"]["image"], image["probability"]["fire"],
                fused["probability"]["fire"])
    logger.info("  [done]   fused verdict -> %s (%.1f%%)", fused["message"], fused["confidence"] * 100)

    classification = classify_and_persist(lat, lon, features.get("UTC"), weather_result,
                                          fire=fused["fire_detected"], source="map_analyze_fused")
    if classification:
        logger.info("  [classify] detected fire type -> %s (%.1f%%)  %s",
                    classification["classification"], classification["confidence"] * 100,
                    classification.get("saved", {}).get("saved") and "saved to GIS" or "")
    logger.info("-" * 62)

    return jsonify({"tabular": tabular, "image": image, "fused": fused, "classification": classification})


@app.route("/api/detections")
def detections():
    """Recent classified fire events persisted in PostGIS, as GeoJSON FeatureCollection."""
    dbmod = db()
    if dbmod is None:
        return jsonify({"type": "FeatureCollection", "features": [], "error": "GIS storage unavailable"}), 200
    limit = int(request.args.get("limit", 200))
    try:
        return jsonify(dbmod.get_recent_detections(limit=limit))
    except Exception as e:
        logger.warning("Fetch detections failed (%s)", e)
        return jsonify({"type": "FeatureCollection", "features": [], "error": str(e)}), 200


def _parse_coords(body=None):
    src = body if isinstance(body, dict) else None
    lat = (src or request.args).get("lat") or (src or request.args).get("latitude")
    lon = (src or request.args).get("lon") or (src or request.args).get("longitude")
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None


def _demo_fires():
    centers = [(23.15, 79.95), (28.05, 78.0), (12.97, 77.59), (21.18, 72.83),
               (19.08, 72.88), (30.73, 76.78), (18.52, 73.86), (26.91, 75.79),
               (13.08, 80.27), (22.57, 88.36), (26.44, 80.85), (16.51, 80.64)]
    fires = []
    now = int(time.time())
    for i in range(30):
        lat, lon = centers[i % len(centers)]
        fires.append({
            "latitude": round(lat + random.uniform(-2, 2), 5),
            "longitude": round(lon + random.uniform(-2, 2), 5),
            "frp": round(random.uniform(5, 150), 1),
            "brightness": round(random.uniform(310, 360), 1),
            "confidence": random.choice(["high", "nominal", "low", "high", "high"]),
            "satellite": "demo",
            "instrument": "demo",
            "daynight": random.choice(["D", "N"]),
            "acq_date": "2026-09-11",
            "acq_time": str(random.randint(1, 2359)),
            "utc": now - random.randint(0, 48 * 3600),
        })
    return fires


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fire detection web server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--firms-key", type=str, default=None, help="NASA FIRMS map key (or set FIRMS_MAP_KEY)")
    parser.add_argument("--demo", action="store_true", help="Serve demo fire markers without FIRMS key")
    args = parser.parse_args()

    if args.firms_key:
        os.environ["FIRMS_MAP_KEY"] = args.firms_key
    if args.demo:
        os.environ.setdefault("FIRMS_MAP_KEY", "")
        logger.info("Demo mode enabled")

    logger.info(f"Serving frontend from {FRONTEND_DIR}")
    app.run(host=args.host, port=args.port, debug=False)