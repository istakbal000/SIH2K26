import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import get_db_connection, init_db, save_user_detection

def check_postgis():
    """Checks if PostGIS is installed on the PostgreSQL server."""
    logger.info("Checking for PostGIS availability on PostgreSQL server...")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT default_version, installed_version FROM pg_available_extensions WHERE name = 'postgis';")
                row = cur.fetchone()
                if not row:
                    logger.error("PostGIS extension is NOT available on this server.")
                    logger.info("Please install PostGIS using the EnterpriseDB Stack Builder or PostGIS installer for Windows.")
                    return False
                
                logger.info(f"PostGIS available (Default Version: {row['default_version']}, Installed Version: {row['installed_version']})")
                if not row['installed_version']:
                    logger.info("Attempting to install PostGIS extension in the current database...")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                    conn.commit()
                    logger.info("PostGIS extension installed successfully.")
                else:
                    logger.info("PostGIS is already installed in this database.")
        return True
    except Exception as e:
        logger.error(f"Failed to check PostGIS: {e}")
        return False

def test_database():
    load_dotenv()
    
    logger.info("Connecting to database...")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()
                logger.info(f"Connected to PostgreSQL: {version['version']}")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.info("Check your .env settings (POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD).")
        return

    postgis_ok = check_postgis()
    if not postgis_ok:
        logger.info("ℹ️ Running in standard PostgreSQL coordinates mode (PostGIS extension optional).")

    logger.info("Initializing database schema...")
    init_db()
    
    logger.info("Inserting a test detection...")
    result = save_user_detection(
        lat=28.6139, 
        lon=77.2090, 
        scenario="Forest", 
        classification="FOREST FIRE", 
        confidence=95.5, 
        temp=35.0, 
        hum=20.0, 
        co2=410.0, 
        pm=25.0,
        source="test_script"
    )
    
    if result.get("saved"):
        mode = "PostGIS spatial index" if postgis_ok else "Standard lat/lon coordinates"
        logger.info(f"✅ Detection successfully saved to PostgreSQL ({mode})! Record ID: {result['record_id']}")
    else:
        logger.error(f"❌ Failed to save detection: {result.get('error')}")

if __name__ == "__main__":
    test_database()
