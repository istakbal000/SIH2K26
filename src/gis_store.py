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


# ---------------------------------------------------------------------------
# Persistent thermal source monitoring (GeoJSON mirror)
# ---------------------------------------------------------------------------

SOURCES_FILE = GIS_DIR / "sources.json"
OBS_FILE = GIS_DIR / "observations.json"
_MAX_SOURCES = 200


def _read_json(path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("gis_store: could not read %s (%s)", path, e)
        return default


def _write_json(path, data):
    GIS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def register_source(name, lat, lon, radius_m=1500, kind="persistent"):
    with _lock:
        sources = _read_json(SOURCES_FILE, {})
        if len(sources) >= _MAX_SOURCES:
            return {"error": "source limit reached"}
        rid = (max((int(k) for k in sources), default=0)) + 1
        rec = {
            "id": rid,
            "name": name,
            "kind": kind,
            "latitude": float(lat),
            "longitude": float(lon),
            "radius_m": float(radius_m or 1500),
            "active": True,
            "created_at": _now_iso(),
        }
        sources[str(rid)] = rec
        _write_json(SOURCES_FILE, sources)
    return rec


def list_sources():
    with _lock:
        sources = _read_json(SOURCES_FILE, {})
        obs = _read_json(OBS_FILE, {})
        out = {}
        for rid, s in sources.items():
            rec = dict(s)
            o = obs.get(str(rid), [])
            rec["observation_count"] = len(o)
            rec["last_observed_at"] = o[0].get("observed_at") if o else None
            rec["max_frp"] = max((x.get("frp") for x in o if x.get("frp") is not None), default=None)
            rec["max_brightness"] = max((x.get("brightness") for x in o if x.get("brightness") is not None), default=None)
            out[rid] = rec
        return out


def delete_source(source_id):
    with _lock:
        sources = _read_json(SOURCES_FILE, {})
        obs = _read_json(OBS_FILE, {})
        removed = sources.pop(str(source_id), None)
        if removed is not None:
            obs.pop(str(source_id), None)
            _write_json(SOURCES_FILE, sources)
            _write_json(OBS_FILE, obs)
        return {"deleted": removed is not None}


def log_source_observation(source_id, observed_at, frp=None, brightness=None,
                           confidence=None, satellite=None, latitude=None, longitude=None):
    with _lock:
        obs = _read_json(OBS_FILE, {})
        arr = obs.setdefault(str(source_id), [])
        seen = {(x.get("observed_at"), x.get("latitude"), x.get("longitude")) for x in arr}
        if (observed_at, latitude, longitude) in seen:
            return {"saved": False, "duplicate": True}
        arr.insert(0, {
            "source_id": source_id,
            "observed_at": observed_at,
            "frp": frp,
            "brightness": brightness,
            "confidence": confidence,
            "satellite": satellite,
            "latitude": latitude,
            "longitude": longitude,
            "logged_at": _now_iso(),
        })
        del arr[500:]
        _write_json(OBS_FILE, obs)
    return {"saved": True, "observation_id": arr[0]["observed_at"]}


def get_source_activity(source_id, limit=50):
    with _lock:
        obs = _read_json(OBS_FILE, {})
        return obs.get(str(source_id), [])[: max(0, int(limit))]