import urllib.request
import urllib.parse
import json

CURRENT_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_CODE_NAMES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Severe hail storm",
}

def _get_json(url, timeout=12):
    req = urllib.request.Request(url, headers={"User-Agent": "fire-detection-app"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_weather(lat, lon) -> dict:
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,"
                   "wind_speed_10m,cloud_cover,precipitation,weather_code",
        "timezone": "auto",
    }
    url = CURRENT_URL + "?" + urllib.parse.urlencode(params)
    data = _get_json(url).get("current", {})
    code = data.get("weather_code")
    return {
        "temperature_c": data.get("temperature_2m"),
        "humidity_pct": data.get("relative_humidity_2m"),
        "pressure_hpa": data.get("surface_pressure"),
        "wind_speed_kmh": data.get("wind_speed_10m"),
        "cloud_cover_pct": data.get("cloud_cover"),
        "precipitation_mm": data.get("precipitation"),
        "weather_code": code,
        "weather_text": WEATHER_CODE_NAMES.get(code, "Unknown"),
        "observation_time": data.get("time"),
    }

def get_air_quality(lat, lon) -> dict:
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": "pm2_5,pm10,us_aqi",
    }
    url = AQ_URL + "?" + urllib.parse.urlencode(params)
    data = _get_json(url).get("current", {})
    return {
        "pm2_5": data.get("pm2_5"),
        "pm10": data.get("pm10"),
        "aqi": data.get("us_aqi"),
    }

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