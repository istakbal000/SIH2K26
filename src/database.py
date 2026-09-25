import os
import json
import logging
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

load_dotenv()

logger = logging.getLogger(__name__)

def _clean(s):
    """Strip whitespace so stray spaces/newlines in env vars can't break conninfo."""
    return "" if s is None else str(s).strip()


def get_db_connection():
    """
    Creates a new database connection using environment variables.
    Defaults to localhost postgres if variables are not set.
    """
    host = _clean(os.getenv("POSTGRES_HOST", "localhost"))
    port = _clean(os.getenv("POSTGRES_PORT", "5432"))
    dbname = _clean(os.getenv("POSTGRES_DB", "postgres"))
    user = _clean(os.getenv("POSTGRES_USER", "postgres"))
    password = _clean(os.getenv("POSTGRES_PASSWORD", ""))

    conninfo = f"host={host} port={port} dbname={dbname} user={user} password={password}"

    # If a direct DATABASE_URL is provided it wins -- but only if it looks like a valid
    # libpq URI (avoids garbage values breaking DNS resolution).
    db_url = _clean(os.getenv("DATABASE_URL"))
    if db_url.startswith("postgres://") or db_url.startswith("postgresql://"):
        conninfo = db_url
    elif db_url:
        logger.warning("DATABASE_URL ignored (not a postgres:// or postgresql:// URI)")

    return psycopg.connect(conninfo, row_factory=dict_row)

def check_postgis_support(conn):
    """
    Checks if PostGIS extension is installed or can be installed.
    Returns True if available and enabled, False otherwise.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'postgis';")
            if cur.fetchone():
                return True
            
            cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'postgis';")
            if cur.fetchone():
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                    conn.commit()
                    return True
                except Exception as ext_err:
                    conn.rollback()
                    logger.warning(f"Could not enable PostGIS extension: {ext_err}")
    except Exception as e:
        conn.rollback()
        logger.warning(f"PostGIS capability check encountered an issue: {e}")
    return False

def has_geometry_column(conn):
    """
    Checks if user_fire_detections table has the geom column.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'user_fire_detections' AND column_name = 'geom';
            """)
            return cur.fetchone() is not None
    except Exception:
        conn.rollback()
        return False

def init_db():
    """
    Initializes the database schema and extensions.
    Gracefully falls back to standard coordinates if PostGIS is not installed.
    """
    try:
        with get_db_connection() as conn:
            has_postgis = check_postgis_support(conn)
            
            with conn.cursor() as cur:
                if has_postgis:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_fire_detections (
                            id SERIAL PRIMARY KEY,
                            latitude DOUBLE PRECISION NOT NULL,
                            longitude DOUBLE PRECISION NOT NULL,
                            geom GEOMETRY(Point, 4326),
                            scenario VARCHAR(50),
                            classification VARCHAR(50) NOT NULL,
                            confidence DOUBLE PRECISION NOT NULL,
                            temperature DOUBLE PRECISION,
                            humidity DOUBLE PRECISION,
                            co2 DOUBLE PRECISION,
                            pm DOUBLE PRECISION,
                            source VARCHAR(50) DEFAULT 'user_web',
                            raw_data JSONB,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_user_fire_detections_geom 
                        ON user_fire_detections USING GIST(geom);
                    """)
                else:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_fire_detections (
                            id SERIAL PRIMARY KEY,
                            latitude DOUBLE PRECISION NOT NULL,
                            longitude DOUBLE PRECISION NOT NULL,
                            scenario VARCHAR(50),
                            classification VARCHAR(50) NOT NULL,
                            confidence DOUBLE PRECISION NOT NULL,
                            temperature DOUBLE PRECISION,
                            humidity DOUBLE PRECISION,
                            co2 DOUBLE PRECISION,
                            pm DOUBLE PRECISION,
                            source VARCHAR(50) DEFAULT 'user_web',
                            raw_data JSONB,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_fire_detections_created_at 
                    ON user_fire_detections (created_at DESC);
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS monitored_sources (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(120) NOT NULL,
                        kind VARCHAR(50) DEFAULT 'persistent',
                        latitude DOUBLE PRECISION NOT NULL,
                        longitude DOUBLE PRECISION NOT NULL,
                        radius_m DOUBLE PRECISION DEFAULT 1500,
                        active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS source_observations (
                        id SERIAL PRIMARY KEY,
                        source_id INTEGER NOT NULL REFERENCES monitored_sources(id) ON DELETE CASCADE,
                        observed_at BIGINT,
                        frp DOUBLE PRECISION,
                        brightness DOUBLE PRECISION,
                        confidence VARCHAR(20),
                        satellite VARCHAR(30),
                        latitude DOUBLE PRECISION,
                        longitude DOUBLE PRECISION,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_source_observations_source
                    ON source_observations (source_id, observed_at DESC);
                """)
                if has_postgis:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_monitored_sources_geom
                        ON monitored_sources USING GIST(
                            ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
                        );
                    """)
            conn.commit()

            # If PostGIS is now available but the table was created earlier without geom column, upgrade it
            if has_postgis and not has_geometry_column(conn):
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            ALTER TABLE user_fire_detections ADD COLUMN IF NOT EXISTS geom GEOMETRY(Point, 4326);
                            UPDATE user_fire_detections 
                            SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) 
                            WHERE geom IS NULL;
                            CREATE INDEX IF NOT EXISTS idx_user_fire_detections_geom 
                            ON user_fire_detections USING GIST(geom);
                        """)
                    conn.commit()
                    logger.info("Upgraded user_fire_detections with PostGIS geometry column.")
                except Exception as upgrade_err:
                    conn.rollback()
                    logger.warning(f"Could not add PostGIS geom column during upgrade: {upgrade_err}")

            logger.info(f"Database initialized successfully (PostGIS active: {has_postgis}).")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def save_user_detection(lat, lon, scenario, classification, confidence, temp, hum, co2, pm, source="user_web", raw_data=None):
    """
    Saves a user detection into the database (with PostGIS geometry if available).
    """
    try:
        with get_db_connection() as conn:
            use_geom = has_geometry_column(conn)
            with conn.cursor() as cur:
                if use_geom:
                    query = """
                        INSERT INTO user_fire_detections (
                            latitude, longitude, geom, scenario, classification, confidence,
                            temperature, humidity, co2, pm, source, raw_data
                        ) VALUES (
                            %(lat)s, %(lon)s, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326),
                            %(scenario)s, %(classification)s, %(confidence)s,
                            %(temp)s, %(hum)s, %(co2)s, %(pm)s, %(source)s, %(raw_data)s
                        ) RETURNING id, created_at, ST_AsGeoJSON(geom) AS geojson;
                    """
                else:
                    query = """
                        INSERT INTO user_fire_detections (
                            latitude, longitude, scenario, classification, confidence,
                            temperature, humidity, co2, pm, source, raw_data
                        ) VALUES (
                            %(lat)s, %(lon)s,
                            %(scenario)s, %(classification)s, %(confidence)s,
                            %(temp)s, %(hum)s, %(co2)s, %(pm)s, %(source)s, %(raw_data)s
                        ) RETURNING id, created_at;
                    """
                
                params = {
                    "lat": lat,
                    "lon": lon,
                    "scenario": scenario,
                    "classification": classification,
                    "confidence": confidence,
                    "temp": temp,
                    "hum": hum,
                    "co2": co2,
                    "pm": pm,
                    "source": source,
                    "raw_data": json.dumps(raw_data) if raw_data else None
                }
                cur.execute(query, params)
                record = cur.fetchone()
            conn.commit()
            
            geojson = json.loads(record["geojson"]) if (use_geom and record.get("geojson")) else {
                "type": "Point",
                "coordinates": [lon, lat]
            }

            return {
                "saved": True, 
                "record_id": record["id"], 
                "created_at": str(record["created_at"]),
                "geojson": geojson
            }
    except Exception as e:
        logger.error(f"Failed to save detection to database: {e}")
        return {"saved": False, "error": str(e)}

def get_recent_detections(limit=50):
    """
    Retrieves the most recent detections from the database.
    """
    try:
        with get_db_connection() as conn:
            use_geom = has_geometry_column(conn)
            with conn.cursor() as cur:
                if use_geom:
                    cur.execute("""
                        SELECT id, latitude, longitude, scenario, classification, 
                               confidence, temperature, humidity, co2, pm, created_at,
                               ST_AsGeoJSON(geom) AS geojson
                        FROM user_fire_detections
                        ORDER BY created_at DESC
                        LIMIT %s;
                    """, (limit,))
                else:
                    cur.execute("""
                        SELECT id, latitude, longitude, scenario, classification, 
                               confidence, temperature, humidity, co2, pm, created_at
                        FROM user_fire_detections
                        ORDER BY created_at DESC
                        LIMIT %s;
                    """, (limit,))
                rows = cur.fetchall()
                
                features = []
                for row in rows:
                    if use_geom and row.get("geojson"):
                        geometry = json.loads(row["geojson"])
                    else:
                        geometry = {
                            "type": "Point",
                            "coordinates": [row["longitude"], row["latitude"]]
                        }
                    
                    features.append({
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": {
                            "id": row["id"],
                            "latitude": row["latitude"],
                            "longitude": row["longitude"],
                            "scenario": row["scenario"],
                            "classification": row["classification"],
                            "confidence": row["confidence"],
                            "temperature": row["temperature"],
                            "humidity": row["humidity"],
                            "co2": row["co2"],
                            "pm": row["pm"],
                            "created_at": str(row["created_at"])
                        }
                    })
                
                return {
                    "type": "FeatureCollection",
                    "features": features
                }
    except Exception as e:
        logger.error(f"Failed to retrieve detections: {e}")
        return {"type": "FeatureCollection", "features": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Persistent thermal source monitoring
# ---------------------------------------------------------------------------

def register_source(name, lat, lon, radius_m=1500, kind="persistent"):
    """Register a location to be monitored for recurring thermal anomalies."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO monitored_sources (name, latitude, longitude, radius_m, kind)
                    VALUES (%(name)s, %(lat)s, %(lon)s, %(radius_m)s, %(kind)s)
                    RETURNING id, name, kind, latitude, longitude, radius_m, active, created_at;
                """, {"name": name, "lat": lat, "lon": lon, "radius_m": radius_m, "kind": kind})
                row = cur.fetchone()
            conn.commit()
        return row or {"error": "no row returned"}
    except Exception as e:
        logger.error(f"Failed to register source: {e}")
        return {"error": str(e)}


def list_sources():
    """List monitored sources with observation counts and last-seen info."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT s.id, s.name, s.kind, s.latitude, s.longitude,
                           s.radius_m, s.active, s.created_at,
                           COUNT(o.id) AS observation_count,
                           MAX(o.observed_at) AS last_observed_at,
                           MAX(o.frp) AS max_frp,
                           MAX(o.brightness) AS max_brightness
                    FROM monitored_sources s
                    LEFT JOIN source_observations o ON o.source_id = s.id
                    GROUP BY s.id
                    ORDER BY s.created_at ASC;
                """)
                rows = cur.fetchall()
        return {str(r["id"]): dict(r) for r in rows}
    except Exception as e:
        logger.error(f"Failed to list sources: {e}")
        return {}


def delete_source(source_id):
    """Delete a monitored source (and cascade its observations)."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM monitored_sources WHERE id = %s RETURNING id;", (source_id,))
                deleted = cur.fetchone()
            conn.commit()
        return {"deleted": bool(deleted)}
    except Exception as e:
        logger.error(f"Failed to delete source: {e}")
        return {"deleted": False, "error": str(e)}


def log_source_observation(source_id, observed_at, frp=None, brightness=None,
                           confidence=None, satellite=None, latitude=None, longitude=None):
    """Log a thermal anomaly observed near a monitored source (deduped per source+time+point)."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM source_observations
                    WHERE source_id = %(source_id)s
                      AND observed_at = %(observed_at)s
                      AND latitude = %(latitude)s
                      AND longitude = %(longitude)s
                    LIMIT 1;
                """, {"source_id": source_id, "observed_at": observed_at,
                      "latitude": latitude, "longitude": longitude})
                if cur.fetchone():
                    return {"saved": False, "duplicate": True}
                cur.execute("""
                    INSERT INTO source_observations (
                        source_id, observed_at, frp, brightness, confidence,
                        satellite, latitude, longitude
                    ) VALUES (
                        %(source_id)s, %(observed_at)s, %(frp)s, %(brightness)s,
                        %(confidence)s, %(satellite)s, %(latitude)s, %(longitude)s
                    )
                    RETURNING id;
                """, {
                    "source_id": source_id,
                    "observed_at": observed_at,
                    "frp": frp,
                    "brightness": brightness,
                    "confidence": confidence,
                    "satellite": satellite,
                    "latitude": latitude,
                    "longitude": longitude,
                })
                rid = cur.fetchone()["id"]
            conn.commit()
        return {"saved": True, "observation_id": rid}
    except Exception as e:
        logger.error(f"Failed to log observation: {e}")
        return {"saved": False, "error": str(e)}


def get_source_activity(source_id, limit=50):
    """Return the observation history for a monitored source (newest first)."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_id, observed_at, frp, brightness, confidence,
                           satellite, latitude, longitude, created_at
                    FROM source_observations
                    WHERE source_id = %s
                    ORDER BY observed_at DESC
                    LIMIT %s;
                """, (source_id, max(0, int(limit))))
                rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to fetch activity: {e}")
        return []
