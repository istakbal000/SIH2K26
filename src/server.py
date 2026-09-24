import os
import sys
import json
import time
import math
import threading
import random
import logging
import tempfile
from pathlib import Path
from functools import lru_cache
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

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


def _load_local_gis():
    """Load the local GeoJSON GIS store (no database server required)."""
    try:
        import gis_store
        gis_store.init_db()
        logger.info("GIS backend: local GeoJSON store (data/gis/detections.geojson)")
        return gis_store
    except Exception as e:
        logger.warning("Local GIS store unavailable (%s)", e)
        _db["ready"] = False
        return None


def db() -> "module or None":
    """Lazily load a GIS storage backend: PostgreSQL/PostGIS when available,
    otherwise a local GeoJSON store so the map overlay always works."""
    if _db["module"] is not None:
        return _db["module"]
    if _db["ready"] is False:
        return None

    want = os.environ.get("GIS_BACKEND", "").strip().lower()
    if want in ("geojson", "local"):
        _db["module"] = _load_local_gis()
        return _db["module"]

    try:
        import database
        database.init_db()
        _db["module"] = database
        logger.info("GIS backend: PostgreSQL/PostGIS")
        return database
    except Exception as e:
        logger.warning("PostGIS unavailable (%s) — falling back to local GeoJSON GIS store", e)

    _db["module"] = _load_local_gis()
    return _db["module"]


def classify_and_persist(lat: float, lon: float, utc, weather_result: dict, fire: bool, source: str = "map_analyze", image_path=None, cls=None):
    """Run fire-type classification and store the event in PostGIS when a fire is detected.

    Returns a dict with 'classification' (and optionally 'saved'), or None when no fire
    was detected (classification is only meaningful for confirmed fires).
    """
    if not fire:
        return None

    hotspots = _fires_cache.get("data") or []
    if cls is None:
        classifier = get_classifier()
        cls = classifier.classify(lat, lon, utc, hotspots, weather=(weather_result or {}).get("weather"), image=image_path)
    event = {
        "classification": cls["classification"],
        "confidence": cls["confidence"],
        "probabilities": cls["probabilities"],
        "class_meta": class_meta(cls["classification"]),
        "source": cls["source"],
        "details": cls["details"],
        "evidence": cls.get("evidence"),
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


def _hav_km(alat, alon, blat, blon):
    import math
    R = 6371.0
    p1, p2 = math.radians(alat), math.radians(blat)
    dp = math.radians(blat - alat)
    dl = math.radians(blon - alon)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _r3(x):
    try:
        if x is None or x == "":
            return None
        return round(float(x), 3)
    except (TypeError, ValueError):
        return None


def _satellite_image_url(lat, lon, pad=0.02, size="460,320"):
    xmin, xmax = round(lon - pad, 5), round(lon + pad, 5)
    ymin, ymax = round(lat - pad, 5), round(lat + pad, 5)
    return (f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
            f"?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR=4326&imageSR=4326&size={size}&format=jpg&f=image")


def _fetch_satellite_tile(lat, lon, pad=0.02, size="320,240", timeout=10):
    """Download Esri World Imagery tile to a temp file. Returns (path, url) or (None, url)."""
    xmin, xmax = round(lon - pad, 5), round(lon + pad, 5)
    ymin, ymax = round(lat - pad, 5), round(lat + pad, 5)
    url = (f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
           f"?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR=4326&imageSR=4326&size={size}&format=jpg&f=image")
    try:
        req = Request(url, headers={"User-Agent": "SIH2K26-FirePredict/1.0"})
        with urlopen(req, timeout=timeout) as r:
            data = r.read()
        fd, p = tempfile.mkstemp(prefix="sat_", suffix=".jpg")
        os.close(fd)
        with open(p, "wb") as f:
            f.write(data)
        return p, url
    except Exception as e:
        logger.warning("Auto satellite fetch failed for (%.3f,%.3f): %s", lat, lon, e)
        return None, url


def _build_inputs(lat: float, lon: float, classification, detection=None):
    """Assemble the three input sources (NASA FIRMS, OSM land-use, satellite imagery) used by the result."""
    hotspots = _fires_cache.get("data") or []
    ev = (classification or {}).get("evidence") or {}
    firms_stats = ev.get("firms")
    if not firms_stats:
        try:
            firms_stats = get_classifier().agriculture_features(lat, lon, hotspots)
        except Exception:
            firms_stats = {}

    nearest = None
    best = float("inf")
    for h in hotspots or []:
        try:
            d = _hav_km(lat, lon, float(h["latitude"]), float(h["longitude"]))
        except (KeyError, TypeError, ValueError):
            continue
        if d < best:
            best, nearest = d, h

    firms = {
        "source": "NASA FIRMS (VIIRS)",
        "satellite": (nearest or {}).get("satellite"),
        "instrument": (nearest or {}).get("instrument"),
        "acq_date": (nearest or {}).get("acq_date"),
        "acq_time": (nearest or {}).get("acq_time"),
        "daynight": (nearest or {}).get("daynight"),
        "confidence": (nearest or {}).get("confidence"),
        "num_hotspots_5km": firms_stats.get("num_hotspots"),
        "mean_brightness": _r3(firms_stats.get("mean_brightness")),
        "max_brightness": _r3(firms_stats.get("max_brightness")),
        "mean_bright_t31": _r3(firms_stats.get("mean_bright_t31")),
        "max_bright_t31": _r3(firms_stats.get("max_bright_t31")),
        "mean_frp": _r3(firms_stats.get("mean_frp")),
        "max_frp": _r3(firms_stats.get("max_frp")),
        "min_distance_km": _r3(firms_stats.get("min_distance_km")),
    }

    geo = ev.get("osm")
    if geo:
        otp = {"source": "OpenStreetMap land-use", "applied": True,
               "agricultural": geo.get("agricultural"), "forest": geo.get("forest"),
               "industrial": geo.get("industrial"), "radius_km": geo.get("radius_km", 3)}
    elif classification:
        otp = {"source": "OpenStreetMap land-use", "applied": False, "note": "not used for this fire type"}
    else:
        otp = {"source": "OpenStreetMap land-use", "applied": False, "note": "not probed (no fire detected)"}

    img = ev.get("image")
    if img and img.get("applied"):
        cnn = {"applied": True, "fire_prob": img.get("fire_prob"),
               "probabilities": img.get("probabilities")}
    elif img and img.get("note"):
        cnn = {"applied": False, "note": img.get("note")}
    else:
        cnn = {"applied": False, "note": "no satellite image analysed"}
    sat = {"source": "Satellite imagery", "image_url": _satellite_image_url(lat, lon), "cnn": cnn}

    return {"firms": firms, "osm": otp, "satellite": sat, "detection": detection}


def _fuse_detection(tab_fire, img_fire, osm_fire, w_tab=0.5, w_img=0.35, w_osm=0.15):
    """Blend the three input sources into one fire/no-fire verdict.

    tab_fire  : FIRMS hotspots + environment (tabular RF)
    img_fire  : satellite image CNN fire probability (None when unavailable)
    osm_fire  : flammable land-use share from OSM (None when unavailable)
    """
    votes = {}
    acc, wsum, used = 0.0, 0.0, {"tabular": True, "image": img_fire is not None, "osm": osm_fire is not None}
    votes["tabular"] = {"p": round(tab_fire, 4), "source": "NASA FIRMS hotspots + environment (RF)"}
    acc += w_tab * tab_fire
    wsum += w_tab
    if used["image"]:
        votes["image"] = {"p": round(img_fire, 4), "source": "satellite image CNN"}
        acc += w_img * img_fire
        wsum += w_img
    else:
        votes["image"] = {"p": None, "source": "satellite image CNN"}
    if used["osm"]:
        votes["osm"] = {"p": round(osm_fire, 4), "source": "OSM land-use signal"}
        acc += w_osm * osm_fire
        wsum += w_osm
    else:
        votes["osm"] = {"p": None, "source": "OSM land-use signal"}
    p = (acc / wsum) if wsum > 0 else tab_fire
    fire = bool(p >= 0.5)
    return {
        "p": round(p, 4),
        "fire_detected": fire,
        "confidence": round(p if fire else 1 - p, 4),
        "probability": {"no_fire": round(1 - p, 4), "fire": round(p, 4)},
        "message": "FIRE DETECTED" if fire else "NO FIRE",
        "prediction": 1 if fire else 0,
        "votes": votes,
        "weights": {k: (w_tab if k == "tabular" else w_img if k == "image" else w_osm) if used[k] else 0.0 for k in used},
    }


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

    sat_path, _sat_url = _fetch_satellite_tile(lat, lon)
    if sat_path:
        logger.info("  [sat]   auto-fetched satellite tile -> running image CNN")

    cls = None
    try:
        classifier = get_classifier()
        cls = classifier.classify(lat, lon, features.get("UTC") or utc, _fires_cache.get("data") or [],
                                  weather=(weather_result or {}).get("weather"), image=sat_path)
    except Exception as e:
        logger.warning("  [classify] classification failed (%s)", e)

    if sat_path:
        try:
            os.remove(sat_path)
        except OSError:
            pass

    fire_ev = (cls or {}).get("evidence") or {}
    img_fire = None
    _img_ev = fire_ev.get("image")
    if _img_ev and _img_ev.get("applied"):
        img_fire = float(_img_ev.get("fire_prob"))
    osm_fire = None
    _geo_ev = fire_ev.get("osm")
    if _geo_ev:
        osm_fire = min(1.0, float(_geo_ev.get("forest") or 0) + float(_geo_ev.get("agricultural") or 0)
                       + float(_geo_ev.get("industrial") or 0))

    det = _fuse_detection(float(result["probability"]["fire"]), img_fire, osm_fire)
    logger.info("  [fuse]   detection votes -> %s = %.2f",
                {k: (v["p"] if v.get("p") is not None else None) for k, v in det["votes"].items()}, det["p"])
    logger.info("  [fuse]   verdict -> %s (confidence %.1f%%)", det["message"], det["confidence"] * 100)
    logger.info("  [done]   reply sent to map UI")
    logger.info("-" * 62)

    classification = classify_and_persist(lat, lon, features.get("UTC"), weather_result,
                                          fire=det["fire_detected"], source="map_analyze",
                                          cls=cls)
    if classification:
        logger.info("  [classify] detected fire type -> %s (%.1f%%)  %s",
                    classification["classification"], classification["confidence"] * 100,
                    classification.get("saved", {}).get("saved") and "saved to GIS" or "")

    return jsonify({
        "fire_detected": det["fire_detected"],
        "prediction": det["prediction"],
        "confidence": det["confidence"],
        "probability": det["probability"],
        "message": det["message"],
        "coordinate": {"latitude": lat, "longitude": lon, "utc": features.get("UTC")},
        "weather": weather_result["weather"],
        "air_quality": weather_result["air_quality"],
        "features_used": result["input_features"],
        "classification": classification if det["fire_detected"] else None,
        "inputs": _build_inputs(lat, lon,
                                classification if det["fire_detected"] else {"evidence": (cls or {}).get("evidence")},
                                detection=det),
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


def _save_temp_upload(file, suffix=".jpg"):
    """Persist an uploaded image to a temp file (rewinds the stream first). Returns path or None."""
    import tempfile
    try:
        stream = getattr(file, "stream", None)
        if stream is not None:
            try:
                stream.seek(0)
            except Exception:
                pass
        data = file.read() if hasattr(file, "read") else stream.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            return tmp.name
    except Exception as e:
        logger.warning("Could not store uploaded image for classification (%s)", e)
        return None


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

    img_tmp = _save_temp_upload(request.files["image"])
    try:
        classification = classify_and_persist(lat, lon, features.get("UTC"), weather_result,
                                              fire=fused["fire_detected"], source="map_analyze_fused",
                                              image_path=img_tmp)
        if classification:
            logger.info("  [classify] detected fire type -> %s (%.1f%%)  %s",
                        classification["classification"], classification["confidence"] * 100,
                        classification.get("saved", {}).get("saved") and "saved to GIS" or "")
    finally:
        if img_tmp:
            try:
                os.remove(img_tmp)
            except OSError:
                pass
    logger.info("-" * 62)

    return jsonify({"tabular": tabular, "image": image, "fused": fused, "classification": classification,
                    "inputs": _build_inputs(lat, lon, classification)})


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
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"))
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