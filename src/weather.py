import urllib.request
import urllib.parse
import urllib.error
import json
import time
import threading

CURRENT_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FALLBACK_URL = "https://wttr.in/{lat},{lon}?format=j1"
FALLBACK_USER_AGENT = "curl"
METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
METNO_USER_AGENT = "fire-detection-app/1.0 (geoflare-sih2026)"

_CACHE_TTL = 600  # seconds
_cache = {}
_cache_lock = threading.Lock()
_cache_ts = {}

_LAST_REQUEST_TS = 0.0
_REQUEST_MIN_INTERVAL = 1.1  # seconds between calls to stay clear of rate limits
_CALLS_WINDOW = []
_CALLS_WINDOW_MAX = 20
_CALLS_WINDOW_SECONDS = 60
_REQUEST_LOCK = threading.Lock()

WEATHER_CODE_NAMES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Severe hail storm",
}

def _throttle():
    """Rate-limit outbound calls: at most N per window and one per ~1.1s."""
    global _LAST_REQUEST_TS
    with _REQUEST_LOCK:
        now = time.time()
        wait = (_LAST_REQUEST_TS + _REQUEST_MIN_INTERVAL) - now
        if wait > 0:
            time.sleep(wait)
        cutoff = now - _CALLS_WINDOW_SECONDS
        _CALLS_WINDOW[:] = [t for t in _CALLS_WINDOW if t > cutoff]
        if len(_CALLS_WINDOW) >= _CALLS_WINDOW_MAX:
            time.sleep(min(_CALLS_WINDOW[0] + _CALLS_WINDOW_SECONDS - now, 30))
            _CALLS_WINDOW[:] = [t for t in _CALLS_WINDOW if t > time.time() - _CALLS_WINDOW_SECONDS]
        _CALLS_WINDOW.append(now)
        _LAST_REQUEST_TS = time.time()


def _get_json(url, timeout=12, retries=2, headers=None):
    _throttle()
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "fire-detection-app"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise last_err


def _fallback_metno(lat, lon) -> dict:
    data = _get_json(METNO_URL.format(lat=round(lat, 4), lon=round(lon, 4)),
                     timeout=15, retries=0, headers={"User-Agent": METNO_USER_AGENT})
    series = ((data.get("properties") or {}).get("timeseries")) or []
    if not series:
        raise RuntimeError("met.no returned no timeseries")
    det = (series[0].get("data") or {}).get("instant", {}).get("details", {})
    sym = (((series[0].get("data") or {}).get("next_1_hours") or {}).get("summary") or {}).get("symbol_code")
    text = str(sym).split("_")[0].replace("clearsky", "Clear").replace("fair", "Fair").replace("lightcloud", "Fair") if sym else None
    precipitation_mm = det.get("precipitation_amount") or ((series[1].get("data") or {}).get("instant", {}).get("details", {}) or {}).get("precipitation_amount")
    return {
        "temperature_c": _num(det.get("air_temperature")),
        "humidity_pct": _num(det.get("relative_humidity")),
        "pressure_hpa": _num(det.get("air_pressure_at_sea_level") or det.get("air_pressure")),
        "wind_speed_kmh": _num(det.get("wind_speed")),
        "cloud_cover_pct": _num(det.get("cloud_area_fraction")),
        "precipitation_mm": _num(precipitation_mm),
        "weather_code": None,
        "weather_text": (text or "Unknown"),
        "observation_time": series[0].get("time"),
        "source": "met.no",
    }


def _fallback_weather(lat, lon) -> dict:
    data = _get_json(FALLBACK_URL.format(lat=round(lat, 4), lon=round(lon, 4)),
                     timeout=15, retries=0, headers={"User-Agent": FALLBACK_USER_AGENT})
    cond = (data.get("current_condition") or [{}])[0]
    desc = (cond.get("weatherDesc") or [{}])
    text = desc[0].get("value") if desc else None
    return {
        "temperature_c": _num(cond.get("temp_C")),
        "humidity_pct": _num(cond.get("humidity")),
        "pressure_hpa": _num(cond.get("pressure")),
        "wind_speed_kmh": _num(cond.get("windspeedKmph")),
        "cloud_cover_pct": _num(cond.get("cloudcover")),
        "precipitation_mm": _num(cond.get("precipMM")),
        "weather_code": None,
        "weather_text": text,
        "observation_time": cond.get("observation_time"),
        "source": "wttr.in",
    }


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cache_get(key):
    with _cache_lock:
        ts = _cache_ts.get(key)
        if ts and (time.time() - ts) < _CACHE_TTL:
            return _cache.get(key)
        return None


def _cache_put(key, value):
    with _cache_lock:
        _cache[key] = value
        _cache_ts[key] = time.time()


def _coord_key(lat, lon):
    return f"{round(lat, 2):.2f},{round(lon, 2):.2f}"

def get_weather(lat, lon) -> dict:
    key = _coord_key(lat, lon)
    cached = _cache_get(f"w:{key}")
    if cached:
        return cached
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,"
                   "wind_speed_10m,cloud_cover,precipitation,weather_code",
        "timezone": "auto",
    }
    url = CURRENT_URL + "?" + urllib.parse.urlencode(params)
    try:
        data = _get_json(url).get("current", {})
    except Exception:
        data = None
    if not data:
        for fb in (_fallback_metno, _fallback_weather):
            try:
                return fb(lat, lon)
            except Exception:
                continue
        raise
    code = data.get("weather_code")
    result = {
        "temperature_c": data.get("temperature_2m"),
        "humidity_pct": data.get("relative_humidity_2m"),
        "pressure_hpa": data.get("surface_pressure"),
        "wind_speed_kmh": data.get("wind_speed_10m"),
        "cloud_cover_pct": data.get("cloud_cover"),
        "precipitation_mm": data.get("precipitation"),
        "weather_code": code,
        "weather_text": WEATHER_CODE_NAMES.get(code, "Unknown"),
        "observation_time": data.get("time"),
        "source": "open-meteo",
    }
    _cache_put(f"w:{key}", result)
    return result

def get_air_quality(lat, lon) -> dict:
    key = _coord_key(lat, lon)
    cached = _cache_get(f"aq:{key}")
    if cached:
        return cached
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": "pm2_5,pm10,us_aqi",
    }
    url = AQ_URL + "?" + urllib.parse.urlencode(params)
    try:
        data = _get_json(url).get("current", {})
    except Exception:
        return {}
    result = {
        "pm2_5": data.get("pm2_5"),
        "pm10": data.get("pm10"),
        "aqi": data.get("us_aqi"),
    }
    _cache_put(f"aq:{key}", result)
    return result

def weather_to_features(weather: dict, aq: dict) -> dict:
    features = {}
    if weather.get("temperature_c") is not None:
        features["Temperature[C]"] = weather["temperature_c"]
    if weather.get("humidity_pct") is not None:
        features["Humidity[%]"] = weather["humidity_pct"]
    if weather.get("pressure_hpa") is not None:
        features["Pressure[hPa]"] = weather["pressure_hpa"]
    if aq.get("pm2_5") is not None:
        features["PM2.5"] = aq["pm2_5"]
    if aq.get("pm10") is not None:
        features["PM1.0"] = aq["pm10"]
    return features