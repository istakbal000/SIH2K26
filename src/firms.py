import csv
import io
import calendar
import urllib.request
from datetime import datetime

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DEFAULT_BBOX = "66,5,99,38"
DEFAULT_SENSOR = "VIIRS_SNPP_NRT"
DEFAULT_DAYS = 2

class FirmsError(Exception):
    pass

def _epoch_from_acq(acq_date: str, acq_time) -> int:
    try:
        dt = datetime.strptime(acq_date, "%Y-%m-%d")
        t = str(int(acq_time)).zfill(4)
        dt = dt.replace(hour=int(t[:2]), minute=int(t[2:]))
        return calendar.timegm(dt.timetuple())
    except Exception:
        return None

def fetch_fires(key: str, bbox: str = DEFAULT_BBOX,
                sensor: str = DEFAULT_SENSOR, days: int = DEFAULT_DAYS) -> list:
    url = f"{FIRMS_BASE}/{key}/{sensor}/{bbox}/{days}"
    req = urllib.request.Request(url, headers={"User-Agent": "fire-detection-app"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise FirmsError(f"FIRMS API error {e.code}: {e.reason}") from e
    except Exception as e:
        raise FirmsError(f"Could not reach FIRMS API: {e}") from e

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise FirmsError("FIRMS returned no data (no fires or wrong key)")

    fires = []
    for r in rows:
        try:
            lat = float(r["latitude"])
            lon = float(r["longitude"])
        except (KeyError, ValueError):
            continue
        acq_date = r.get("acq_date", "")
        acq_time = r.get("acq_time", "")
        frp = _to_float(r.get("frp"))
        confidence = r.get("confidence", "")
        fires.append({
            "latitude": lat,
            "longitude": lon,
            "frp": frp,
            "brightness": _to_float(r.get("bright_ti4")),
            "bright_t31": _to_float(r.get("bright_ti5") or r.get("bright_t31")),
            "confidence": confidence,
            "satellite": r.get("satellite", ""),
            "instrument": r.get("instrument", ""),
            "daynight": r.get("daynight", ""),
            "acq_date": acq_date,
            "acq_time": acq_time,
            "scan": _to_float(r.get("scan")),
            "track": _to_float(r.get("track")),
            "utc": _epoch_from_acq(acq_date, acq_time) or 0,
        })
    return fires

def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None