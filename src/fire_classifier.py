"""Fire-type classification for an already-detected fire.

Turns a detected hotspot into a fire class label by fusing the classifiers that
already exist in this repo:

* agricultural_fire_random_forest_v2  -> P(agricultural), from FIRMS cluster stats
* shared_fire_model (temp, humidity)  -> P(forest) vs industrial   (optional)

If no classifier can produce a score the result is "Unclassified" so detection
never blocks on classification.
"""
import math
import json
import logging
import threading
from pathlib import Path

import joblib
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import geo_split
from image_det.model import load_checkpoint as load_image_checkpoint

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
AGRICULTURE_MODEL = BASE_DIR / "data/models/agricultural_fire_random_forest_v2.pkl"
SHARED_MODEL = BASE_DIR / "models/shared_fire_model.pkl"
TYPE_MODEL = BASE_DIR / "models/fire_type.pth"

CLASS_ORDER = ("agricultural", "forest", "industrial")

_CONFIDENCE_TEXT = {"high": 100.0, "nominal": 67.0, "low": 33.0, "0": 0.0, "50": 50.0, "100": 100.0}


def _haversine_km(alat, alon, blat, blon):
    R = 6371.0
    p1, p2 = math.radians(alat), math.radians(blat)
    dp = math.radians(blat - alat)
    dl = math.radians(blon - alon)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _num(v, default=0.0):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _conf_score(conf):
    if isinstance(conf, (int, float)):
        return float(conf)
    return _CONFIDENCE_TEXT.get(str(conf).strip().lower(), 67.0)


def _acq_hour(f):
    t = str(f.get("acq_time", "") or "").zfill(4)
    try:
        return int(t[:2]) + int(t[2:]) / 60
    except ValueError:
        try:
            return int(float(f.get("acq_time", 0))) / 100
        except (TypeError, ValueError):
            return 0.0


class FireClassifier:
    """Classifies a detected fire into a type using the repo's trained models."""

    def __init__(self):
        self._lock = threading.Lock()
        self._agri = None       # (rfc, feature_names)
        self._shared = None     # RandomForest (temp, humidity) -> is_forest
        self._type = None       # (torch module, class_names, extra) fire-type CNN
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                if AGRICULTURE_MODEL.exists():
                    bag = joblib.load(AGRICULTURE_MODEL)
                    rfc = bag["model"] if isinstance(bag, dict) and "model" in bag else bag
                    feats = list(getattr(rfc, "feature_names_in_", bag.get("features", [])))
                    self._agri = (rfc, feats)
                    logger.info("FireClassifier: agriculture RF ready (%d features)", len(feats))
                else:
                    logger.warning("FireClassifier: agriculture model missing at %s", AGRICULTURE_MODEL)
            except Exception as e:
                logger.warning("FireClassifier: agriculture model failed to load (%s)", e)

            try:
                if SHARED_MODEL.exists():
                    self._shared = joblib.load(SHARED_MODEL)
                    logger.info("FireClassifier: shared forest/industrial tabular model ready")
                else:
                    logger.info("FireClassifier: shared forest/industrial model not present (optional)")
            except Exception as e:
                logger.warning("FireClassifier: shared model failed to load (%s)", e)

            try:
                if TYPE_MODEL.exists():
                    model_, names_, extra_ = load_image_checkpoint(TYPE_MODEL)
                    self._type = (model_, names_, extra_)
                    logger.info("FireClassifier: fire-type image CNN ready (%s)", names_)
                else:
                    logger.info("FireClassifier: fire-type image CNN not present (optional)")
            except Exception as e:
                logger.warning("FireClassifier: fire-type image CNN failed to load (%s)", e)
            self._loaded = True

    def agriculture_features(self, lat, lon, hotspots, radius_km=5.0):
        """FIRMS cluster statistics around (lat, lon) matching the agriculture model."""
        feats = {
            "num_hotspots": 0,
            "mean_brightness": 0.0, "max_brightness": 0.0,
            "mean_bright_t31": 0.0, "max_bright_t31": 0.0,
            "mean_frp": 0.0, "max_frp": 0.0,
            "min_distance_km": 0.0,
            "mean_acq_hour": 0.0,
            "num_daytime_hotspots": 0, "num_nighttime_hotspots": 0,
            "mean_scan": 0.0, "mean_track": 0.0, "max_scan": 0.0, "max_track": 0.0,
        }
        if not hotspots:
            return feats

        near = []
        for f in hotspots:
            try:
                d = _haversine_km(lat, lon, float(f["latitude"]), float(f["longitude"]))
            except (TypeError, KeyError, ValueError):
                continue
            if d <= radius_km:
                near.append((d, f))

        if not near:
            return feats

        n = float(len(near))
        feats["num_hotspots"] = len(near)
        feats["mean_brightness"] = sum(_num(f.get("brightness")) for _, f in near) / n
        feats["max_brightness"] = max(_num(f.get("brightness")) for _, f in near)
        feats["mean_bright_t31"] = sum(_num(f.get("bright_t31")) for _, f in near) / n
        feats["max_bright_t31"] = max(_num(f.get("bright_t31")) for _, f in near)
        feats["mean_frp"] = sum(_num(f.get("frp")) for _, f in near) / n
        feats["max_frp"] = max(_num(f.get("frp")) for _, f in near)
        feats["min_distance_km"] = min(d for d, _ in near)
        feats["mean_acq_hour"] = sum(_acq_hour(f) for _, f in near) / n
        feats["num_daytime_hotspots"] = sum(1 for _, f in near if f.get("daynight") == "D")
        feats["num_nighttime_hotspots"] = sum(1 for _, f in near if f.get("daynight") == "N")
        feats["mean_scan"] = sum(_num(f.get("scan")) for _, f in near) / n
        feats["mean_track"] = sum(_num(f.get("track")) for _, f in near) / n
        feats["max_scan"] = max(_num(f.get("scan")) for _, f in near)
        feats["max_track"] = max(_num(f.get("track")) for _, f in near)
        return feats

    def _run_type_cnn(self, path):
        """Run the forest/industrial/nofire CNN on an image file. Returns a prob dict or None."""
        if self._type is None:
            return None
        model, class_names, extra = self._type
        img_size = int(extra.get("img_size", 224)) if isinstance(extra, dict) and extra else 224
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            logits = model(tf(Image.open(path).convert("RGB")).unsqueeze(0))
            p = torch.softmax(logits, 1)[0]
        return {cn: float(p[i]) for i, cn in enumerate(class_names)}

    def classify(self, lat, lon, utc, hotspots, weather=None, image=None):
        """Returns {classification, confidence, probabilities, source, details, evidence}."""
        self._ensure_loaded()
        probs = {}
        notes = []
        geo_used = False
        geo_evidence = None
        firms_evidence = None
        image_evidence = None
        shared_evidence = None

        # Agricultural: trained RF on FIRMS cluster statistics (always available).
        if self._agri is not None:
            try:
                rfc, feats = self._agri
                row = self.agriculture_features(lat, lon, hotspots or [])
                firms_evidence = {k: row[k] for k in row}
                df = pd.DataFrame([{k: row.get(k, 0.0) for k in feats}])[feats]
                p = float(rfc.predict_proba(df)[0][1])
                probs["agricultural"] = p
                notes.append(f"agri_rf={p:.2f} (cluster: {row['num_hotspots']} hotspots in 5km)")
            except Exception as e:
                logger.warning("FireClassifier: agriculture inference failed (%s)", e)
                notes.append("agri_rf=error")

        # Forest vs industrial: trained shared model first (temp, humidity).
        shared_used = False
        if self._shared is not None:
            try:
                w = weather or {}
                temp = _num(w.get("temperature_c"), float("nan"))
                hum = _num(w.get("humidity_pct"), float("nan"))
                if math.isnan(temp) or math.isnan(hum):
                    notes.append("shared_rf=skipped (no live temperature/humidity)")
                else:
                    p = float(self._shared.predict_proba([[temp, hum]])[0][1])
                    probs["forest"] = p
                    probs["industrial"] = 1 - p
                    notes.append(f"shared_rf={p:.2f} forest (t={temp:.1f}C, rh={hum:.0f}%)")
                    shared_used = True
                    shared_evidence = {"temp_c": temp, "humidity_pct": hum, "forest": float(p)}
            except Exception as e:
                logger.warning("FireClassifier: shared model inference failed (%s)", e)
                notes.append("shared_rf=error")

        # Fallback: real OSM land-use proximity around the point (forest/industrial/agriculture).
        if not shared_used:
            try:
                geo = geo_split.class_scores(lat, lon)
            except Exception as e:
                geo = None
                logger.warning("FireClassifier: OSM land-use failed (%s)", e)
            if geo:
                base = geo["agricultural"] + geo["forest"] + geo["industrial"]
                if base > 0:
                    probs["forest"] = geo["forest"] / base
                    probs["industrial"] = geo["industrial"] / base
                    probs.setdefault("agricultural", geo["agricultural"] / base)
                    notes.append("osm_landuse=" + ",".join(
                        f"{k}={v:.2f}" for k, v in geo.items() if k in ("agricultural", "forest", "industrial")))
                    geo_used = True
                    geo_evidence = {
                        "agricultural": round(geo["agricultural"] / base, 4),
                        "forest": round(geo["forest"] / base, 4),
                        "industrial": round(geo["industrial"] / base, 4),
                        "radius_km": int(geo.get("radius_km", 3) or 3),
                    }
                else:
                    notes.append("osm_landuse=no developed land within 3km")
            else:
                notes.append("osm_landuse=unavailable")

        if isinstance(image, (str, Path)):
            try:
                t = self._run_type_cnn(image)
            except Exception as e:
                t = None
                logger.warning("FireClassifier: type CNN inference failed (%s)", e)
            if t:
                p_photo_fire = 1.0 - float(t.get("nofire", 0.0))
                image_evidence = {
                    "applied": True,
                    "fire_prob": round(p_photo_fire, 4),
                    "probabilities": {k: round(float(v), 4) for k, v in t.items()},
                }
                if p_photo_fire >= 0.5:
                    f = float(t.get("forest", 0.0))
                    ind = float(t.get("industrial", 0.0))
                    fi = f + ind
                    if fi > 0:
                        probs.pop("forest", None)
                        probs.pop("industrial", None)
                        probs["forest"] = f / fi
                        probs["industrial"] = ind / fi
                        notes.append(f"image_type_cnn=fire({p_photo_fire:.2f}) forest={f/fi:.2f} industrial={ind/fi:.2f}")
                    else:
                        notes.append(f"image_type_cnn=fire({p_photo_fire:.2f}) no type signal")
                else:
                    notes.append(f"image_type_cnn=nofire({p_photo_fire:.2f})")
        elif image is not None and isinstance(image, dict) and image.get("fire_detected"):
            p = image.get("probability", {}).get("fire", 0.5)
            notes.append(f"image_cnn_fire={p:.2f}")

        evidence = {
            "firms": firms_evidence,
            "osm": geo_evidence,
            "shared": shared_evidence,
            "image": image_evidence,
        }

        if not probs:
            return {
                "classification": "Unclassified",
                "confidence": 0.0,
                "probabilities": {k: 0.0 for k in CLASS_ORDER},
                "source": "none",
                "details": notes,
                "evidence": evidence,
            }

        total = sum(probs.values())
        label = max(probs, key=probs.get)
        normalized = {k: round((v / total) if total > 0 else 0.0, 6) for k, v in probs.items()}
        # Confidence: normalized share of the winning class when several pipelines
        # produced scores; the raw model probability when only one did.
        confidence = normalized[label] if len(probs) > 1 else round(probs[label], 6)
        return {
            "classification": label,
            "confidence": confidence,
            "probabilities": normalized,
            "source": "trained models + OSM land-use" if geo_used else "trained models",
            "details": notes,
            "evidence": evidence,
        }


_classifier = None
_classifier_lock = threading.Lock()


def get_classifier():
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = FireClassifier()
    return _classifier


CLASS_META = {
    "agricultural": {"color": "#f59e0b", "emoji": "🌾", "label": "Agricultural"},
    "forest":      {"color": "#16a34a", "emoji": "🌲", "label": "Forest"},
    "industrial":  {"color": "#dc2626", "emoji": "🏭", "label": "Industrial"},
    "Unclassified": {"color": "#9ca3af", "emoji": "❔", "label": "Unclassified"},
}


def class_meta(label):
    return CLASS_META.get(label, CLASS_META["Unclassified"])