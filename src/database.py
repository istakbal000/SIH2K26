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
