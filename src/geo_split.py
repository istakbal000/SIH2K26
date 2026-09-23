"""Geospatial land-use context for fire-type classification.

Queries OpenStreetMap (Overpass API) for the dominant land-use around a fire point
and turns it into per-class proximity scores: forest, industrial, agricultural.
This is the same "OSM buffers" idea already configured in config.yaml
(osm_buffers), implemented without any extra dependency (stdlib urllib only).

Beatings: results are cached per coordinate, requests are throttled, HTTP 429
(rate limit) is honored with Retry-After, and the search radius escalates when a
small radius finds nothing.
"""
import json
import ssl
import time
import urllib.parse
import urllib.request
import logging

logger = logging.getLogger(__name__)

ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
BASE_RADIUS = 3000                 # meters
ESCALATIONS = (6000, 10000)        # fallback radii when nothing found
CACHE_TTL = 3600                   # seconds (1 h)
MIN_REQUEST_INTERVAL = 1.5         # seconds between Overpass calls
MAX_RETRIES = 2
HEADERS = {"User-Agent": "geoflare-firewatch geo-split/1.0 (contact: local demo)"}

_LANDUSE_RE = r"^(forest|forestry|industrial|commercial|quarry|landfill|railway|port|harbour|depot|works|farmland|farmyard|orchard|vineyard|allotments|plant_nursery|agriculture|greenhouse_horticulture)$"
_NATURAL_RE = r"^(wood|scrub|heath|maquis|forest|grassland|meadow|water|wetland|basin|reservoir)$"
_BOUNDARY_RE = r"^(national_park|protected_area)$"
_MANMADE_RE = r"^(works|factory|industrial|depot|storage_tank|petroleum|oil|plant|bridge)$"
_LEISURE_RE = r"^(nature_reserve)$"

_CACHE = {}
_last_request = [0.0]


def _throttle():
    wait = MIN_REQUEST_INTERVAL - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)


def _classify_element(tags):
    lu = str(tags.get("landuse", "")).lower().strip()
    nat = str(tags.get("natural", "")).lower().strip()
    boundary = str(tags.get("boundary", "")).lower().strip()
    man = str(tags.get("man_made", "")).lower().strip()
    leisure = str(tags.get("leisure", "")).lower().strip()

    if boundary in ("national_park", "protected_area") or leisure == "nature_reserve" \
            or nat in ("wood", "scrub", "heath", "maquis", "forest", "grassland", "meadow"):
        return "forest"
    if nat in ("water", "wetland", "basin", "reservoir"):
        return "water"
    if lu in ("industrial", "commercial", "quarry", "landfill", "railway", "port",
              "harbour", "depot", "works") or man in ("works", "factory", "industrial",
              "depot", "storage_tank", "petroleum", "oil", "plant"):
        return "industrial"
    if lu in ("farmland", "farmyard", "orchard", "vineyard", "allotments",
              "plant_nursery", "agriculture", "greenhouse_horticulture"):
        return "agricultural"
    return "other"


def _query(lat, lon, radius):
    return f"""
[out:json][timeout:15];
(
  way["landuse"~"{_LANDUSE_RE}"](around:{radius},{lat},{lon});
  way["natural"~"{_NATURAL_RE}"](around:{radius},{lat},{lon});
  way["boundary"~"{_BOUNDARY_RE}"](around:{radius},{lat},{lon});
  way["man_made"~"{_MANMADE_RE}"](around:{radius},{lat},{lon});
  way["leisure"~"{_LEISURE_RE}"](around:{radius},{lat},{lon});
  relation["landuse"~"{_LANDUSE_RE}"](around:{radius},{lat},{lon});
  relation["boundary"~"{_BOUNDARY_RE}"](around:{radius},{lat},{lon});
);
out tags 40;
"""


def _open(req):
    try:
        return urllib.request.urlopen(req, timeout=25)
    except ssl.SSLCertVerificationError:
        ctx = ssl._create_unverified_context()
        return urllib.request.urlopen(req, timeout=25, context=ctx)


def _fetch(lat, lon, radius):
    global _last_request
    for attempt in range(MAX_RETRIES + 1):
        for endpoint in ENDPOINTS:
            _throttle()
            data = urllib.parse.urlencode({"data": _query(lat, lon, radius)}).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=HEADERS)
            try:
                with _open(req) as resp:
                    _last_request[0] = time.time()
                    return json.loads(resp.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as e:
                _last_request[0] = time.time()
                if e.code == 429 and attempt < MAX_RETRIES:
                    retry_after = int(e.headers.get("Retry-After", "5") or 5)
                    logger.warning("geo_split: %s rate-limited (429) — retry after %ss", endpoint, retry_after)
                    time.sleep(min(retry_after, 20))
                    break
                if e.code in (429, 500, 502, 503, 504) and endpoint != ENDPOINTS[-1]:
                    logger.warning("geo_split: %s returned %s — trying next endpoint", endpoint, e.code)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                _last_request[0] = time.time()
                if endpoint != ENDPOINTS[-1]:
                    logger.warning("geo_split: %s failed (%s) — trying next endpoint", endpoint, e)
                    continue
                raise


def _aggregate(payload):
    counts = {c: 0 for c in ("forest", "industrial", "agricultural", "water", "other")}
    for el in payload.get("elements") or []:
        w = 1.0
        if el.get("type") == "way" and len(el.get("nodes", [])) >= 4:
            w = 2.0
        elif el.get("type") == "relation":
            w = 3.0
        cls = _classify_element(el.get("tags") or {})
        counts[cls] = counts.get(cls, 0) + w
    return counts


def land_use(lat, lon, radius=BASE_RADIUS):
    """Return counts + fractions of land-use classes around (lat, lon).

    Escalates the search radius when the base radius finds nothing.
    Returns None when OSM is unreachable.
    Shape: {'total', 'counts', 'fractions', 'primary'}.
    """
    key = (round(lat, 2), round(lon, 2))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    counts = None
    used_radius = radius
    try:
        for r in (radius,) + ESCALATIONS:
            payload = _fetch(lat, lon, r)
            counts = _aggregate(payload)
            if sum(counts.values()) > 0:
                used_radius = r
                break
        if counts is None:
            counts = {c: 0 for c in ("forest", "industrial", "agricultural", "water", "other")}
    except Exception as e:
        logger.warning("geo_split: OSM unavailable for (%.3f, %.3f): %s", lat, lon, e)
        return None

    total = sum(counts.values())
    fractions = {c: (round(v / total, 4) if total else 0.0) for c, v in counts.items()}
    result = {
        "total": int(total),
        "radius_m": int(used_radius),
        "counts": counts,
        "fractions": fractions,
        "primary": max(fractions, key=fractions.get) if total > 0 else "none",
    }
    _CACHE[key] = (time.time(), result)
    return result


def class_scores(lat, lon, radius=BASE_RADIUS):
    """Factions of {agricultural, forest, industrial} in [0,1]. None when OSM unreachable."""
    lu = land_use(lat, lon, radius)
    if not lu or not lu.get("total"):
        return None
    return lu["fractions"]