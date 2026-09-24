"""Local GeoJSON-backed GIS store — drop-in replacement for src/database.py
when PostgreSQL/PostGIS is not running.

Stores classified fire events as a GeoJSON FeatureCollection under
data/gis/detections.geojson so the map overlay works without a database
server. The object interface and property schema match database.py exactly,
making this swappable with the PostGIS backend once one is available
(override with the GIS_BACKEND env var, e.g. GIS_BACKEND=postgis).
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
GIS_DIR = BASE_DIR / "data" / "gis"
GIS_FILE = GIS_DIR / "detections.geojson"
MAX_RECORDS = 1000

_lock = threading.Lock()


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_all():
    if not GIS_FILE.exists():
        return []
    try:
        with open(GIS_FILE, "r", encoding="utf-8") as f:
            fc = json.load(f)
        return fc.get("features", [])
    except Exception as e:
        logger.warning("gis_store: could not read %s (%s)", GIS_FILE, e)
        return []


def _write_all(features):
    GIS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = GIS_FILE.with_suffix(".geojson.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=1)
    os.replace(tmp, GIS_FILE)


def init_db():
    """Ensure the data directory exists. No-op beyond that for the file store."""
    GIS_DIR.mkdir(parents=True, exist_ok=True)
    return True


def has_postgis():
    return False


def save_user_detection(lat, lon, scenario, classification, confidence, temp, hum, co2, pm, source="user_web", raw_data=None):
    """Append a classified event as a GeoJSON Point feature. Returns database.py-shape dict."""
    try:
        with _lock:
            feats = _read_all()
            rid = (feats[0]["properties"].get("id", 0) if feats else 0) + 1
            created = _now_iso()
            props = {
                "id": rid,
                "latitude": lat,
                "longitude": lon,
                "scenario": scenario,
                "classification": classification,
                "confidence": float(confidence) if confidence is not None else None,
                "temperature": temp,
                "humidity": hum,
                "co2": co2,
                "pm": pm,
                "source": source,
                "created_at": created,
            }
            if isinstance(raw_data, dict):
                props["raw"] = raw_data
            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": props,
            }
            feats.insert(0, feature)
            if len(feats) > MAX_RECORDS:
                del feats[MAX_RECORDS:]
            _write_all(feats)
        return {"saved": True, "record_id": rid, "created_at": created, "geojson": feature["geometry"]}
    except Exception as e:
        logger.error("gis_store: failed to save detection (%s)", e)
        return {"saved": False, "error": str(e)}


def get_recent_detections(limit=200):
    """Return the stored events as a GeoJSON FeatureCollection (newest first)."""
    try:
        feats = _read_all()
        feats = feats[: max(0, int(limit))]
        return {"type": "FeatureCollection", "features": feats}
    except Exception as e:
        logger.error("gis_store: failed to retrieve detections (%s)", e)
        return {"type": "FeatureCollection", "features": [], "error": str(e)}