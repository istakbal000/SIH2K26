import os
import json
import logging
from contextlib import contextmanager
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Creates a new database connection using environment variables.
    Defaults to localhost postgres if variables are not set.
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    dbname = os.getenv("POSTGRES_DB", "postgres")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    
    conninfo = f"host={host} port={port} dbname={dbname} user={user} password={password}"
    
    # Check if a direct DATABASE_URL is provided instead
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        conninfo = db_url

    return psycopg.connect(conninfo, row_factory=dict_row)

def init_db():
    """
    Initializes the database schema and extensions.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # 1. Ensure PostGIS is installed
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                except Exception as e:
                    logger.warning(f"Could not create postgis extension (ensure you have superuser privileges or PostGIS is installed): {e}")

                # 2. Create the table
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
                
                # 3. Create indices
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_fire_detections_geom 
                    ON user_fire_detections USING GIST(geom);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_fire_detections_created_at 
                    ON user_fire_detections (created_at DESC);
                """)
            conn.commit()
            logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def save_user_detection(lat, lon, scenario, classification, confidence, temp, hum, co2, pm, source="user_web", raw_data=None):
    """
    Saves a user detection into the PostGIS database.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
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
            return {"saved": True, "record_id": record["id"], "created_at": str(record["created_at"])}
    except Exception as e:
        logger.error(f"Failed to save detection to database: {e}")
        return {"saved": False, "error": str(e)}

def get_recent_detections(limit=50):
    """
    Retrieves the most recent detections from the database.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, latitude, longitude, scenario, classification, 
                           confidence, temperature, humidity, co2, pm, created_at,
                           ST_AsGeoJSON(geom) AS geojson
                    FROM user_fire_detections
                    ORDER BY created_at DESC
                    LIMIT %s;
                """, (limit,))
                rows = cur.fetchall()
                
                features = []
                for row in rows:
                    features.append({
                        "type": "Feature",
                        "geometry": json.loads(row["geojson"]) if row.get("geojson") else None,
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
