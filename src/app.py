import os
import random
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Add project root to path so we can import scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import init_db, save_user_detection, get_recent_detections

# Reuse the inference functions
from scripts.test_inference import (
    predict_rf, 
    predict_cnn, 
    get_forest_osm_score, 
    get_industrial_osm_score
)

app = FastAPI(title="FireWatch India API")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    # Initialize the PostGIS database connection and schema
    init_db()

class AnalyzeRequest(BaseModel):
    lat: float
    lon: float

@app.get("/api/config")
def get_config():
    return {
        "map_key": os.getenv("MAP_KEY", "")
    }

@app.post("/api/analyze")
def analyze_location(req: AnalyzeRequest):
    lat = req.lat
    lon = req.lon
    
    # Simulate scenario choice based on coordinates (or randomly for demo)
    scenario = random.choice(["Forest", "Industrial"])
    
    if scenario == "Forest":
        test_img = 'data/raw/forest_fire/fire_train_1001.jpg'
        forest_input = {
            'brightness': random.uniform(310.0, 350.0), 
            'bright_t31': random.uniform(285.0, 305.0), 
            'frp': random.uniform(1.0, 15.0), 
            'confidence': random.uniform(50.0, 100.0), 
            'is_day': random.choice([0.0, 1.0]), 
            'hour': random.uniform(0.0, 23.0), 
            'month': random.uniform(1.0, 12.0), 
            'distance_to_forest': random.uniform(0.0, 1000.0), 
            'distance_to_industrial_area': random.uniform(10000.0, 50000.0), 
            'forest_area_ratio_1km': random.uniform(0.8, 1.0), 
            'industrial_area_ratio_1km': 0.0, 
            'latitude': lat, 
            'longitude': lon, 
            'fire_cluster_id': random.uniform(1.0, 50.0), 
            'cluster_size': random.uniform(100.0, 1000.0), 
            'max_frp': random.uniform(5.0, 50.0), 
            'mean_frp': random.uniform(1.0, 10.0), 
            'frp_sum': random.uniform(100.0, 1000.0)
        }
        industrial_input = {
            'CO2_Room': random.uniform(400.0, 450.0), 
            'H2_Room': 0.0, 
            'PM05_Room': random.uniform(10.0, 20.0), 
            'PM100_Room': 0.0, 
            'PM10_Room': random.uniform(0.0, 5.0), 
            'PM25_Room': 0.0, 
            'PM40_Room': 0.0, 
            'PM_Room_Typical_Size': 0.2, 
            'PM_Total_Room': random.uniform(10.0, 30.0), 
            'VOC_Room_RAW': 0.0, 
            'Temperature_Room': random.uniform(20.0, 25.0), 
            'Humidity_Room': random.uniform(40.0, 60.0), 
            'CO_Room': 0.0
        }
        display_temp = random.uniform(35.0, 45.0)
        display_hum = random.uniform(10.0, 30.0)
        display_co2 = random.uniform(400, 450)
        display_pm = random.uniform(10, 30)
    else:
        test_img = 'data/raw/industrial_fire/Industrial fire/images.jpg'
        forest_input = {
            'brightness': random.uniform(290.0, 305.0), 
            'bright_t31': random.uniform(270.0, 285.0), 
            'frp': random.uniform(0.0, 2.0), 
            'confidence': random.uniform(0.0, 20.0), 
            'is_day': random.choice([0.0, 1.0]), 
            'hour': 12.0, 
            'month': 1.0, 
            'distance_to_forest': random.uniform(50000.0, 100000.0), 
            'distance_to_industrial_area': random.uniform(0.0, 500.0), 
            'forest_area_ratio_1km': 0.0, 
            'industrial_area_ratio_1km': random.uniform(0.8, 1.0), 
            'latitude': lat, 
            'longitude': lon, 
            'fire_cluster_id': 0.0, 
            'cluster_size': 0.0, 
            'max_frp': 0.0, 
            'mean_frp': 0.0, 
            'frp_sum': 0.0
        }
        industrial_input = {
            'CO2_Room': random.uniform(1500.0, 3000.0), 
            'H2_Room': random.uniform(1.0, 5.0), 
            'PM05_Room': random.uniform(1000.0, 4000.0), 
            'PM100_Room': random.uniform(1.0, 5.0), 
            'PM10_Room': random.uniform(100.0, 600.0), 
            'PM25_Room': random.uniform(5.0, 20.0), 
            'PM40_Room': random.uniform(1.0, 5.0), 
            'PM_Room_Typical_Size': random.uniform(0.45, 0.55), 
            'PM_Total_Room': random.uniform(1000.0, 5000.0), 
            'VOC_Room_RAW': random.uniform(2.0, 10.0), 
            'Temperature_Room': random.uniform(40.0, 80.0), 
            'Humidity_Room': random.uniform(10.0, 30.0), 
            'CO_Room': random.uniform(1.0, 5.0)
        }
        display_temp = random.uniform(60.0, 120.0)
        display_hum = random.uniform(10.0, 40.0)
        display_co2 = random.uniform(1500, 3000)
        display_pm = random.uniform(1000, 5000)
    
    try:
        # 1. Forest Pipeline
        forest_tab_conf = predict_rf('outputs/models/Forest_Random_Forest.pkl', 'outputs/models/Forest_Random_Forest_cols.pkl', forest_input)
        forest_img_conf = predict_cnn('outputs/models/Forest_CNN.pth', test_img)
        forest_osm_conf = get_forest_osm_score()
        forest_final_score = (0.30 * forest_tab_conf) + (0.40 * forest_img_conf) + (0.30 * forest_osm_conf)
        
        # 2. Industrial Pipeline
        ind_tab_conf = predict_rf('outputs/models/Industrial_Random_Forest.pkl', 'outputs/models/Industrial_Random_Forest_cols.pkl', industrial_input)
        ind_img_conf = predict_cnn('outputs/models/Industrial_CNN.pth', test_img)
        ind_osm_conf = get_industrial_osm_score()
        ind_final_score = (0.30 * ind_tab_conf) + (0.40 * ind_img_conf) + (0.30 * ind_osm_conf)
    except Exception as e:
        print(f"Inference error: {e}")
        # Provide fallback values if models are missing or error out
        forest_final_score = random.uniform(0.6, 0.9) if scenario == "Forest" else random.uniform(0.1, 0.4)
        ind_final_score = random.uniform(0.6, 0.9) if scenario == "Industrial" else random.uniform(0.1, 0.4)

    classification = "FOREST FIRE" if forest_final_score > ind_final_score else "INDUSTRIAL FIRE"
    confidence = max(forest_final_score, ind_final_score) * 100

    # Save to PostGIS database
    db_result = save_user_detection(
        lat=lat,
        lon=lon,
        scenario=scenario,
        classification=classification,
        confidence=confidence,
        temp=display_temp,
        hum=display_hum,
        co2=display_co2,
        pm=display_pm,
        source="user_web",
        raw_data={"forest_score": forest_final_score, "industrial_score": ind_final_score}
    )

    return {
        "status": "success",
        "scenario": scenario,
        "database": db_result,
        "results": {
            "classification": classification,
            "confidence": round(confidence, 1),
            "environmental": {
                "temperature": round(display_temp, 1),
                "humidity": round(display_hum, 1),
                "co2": round(display_co2, 0),
                "pm": round(display_pm, 0)
            }
        }
    }

@app.get("/api/detections")
def get_detections(limit: int = 50):
    return get_recent_detections(limit)

@app.get("/api/db-status")
def db_status():
    from src.database import get_db_connection
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT default_version, installed_version FROM pg_available_extensions WHERE name = 'postgis';")
                postgis = cur.fetchone()
                return {"status": "ok", "postgis_installed": bool(postgis and postgis.get('installed_version'))}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Mount static files to serve the frontend
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    app_target = "src.app:app" if os.path.exists("src") else "app:app"
    uvicorn.run(app_target, host="0.0.0.0", port=8000, reload=True)
