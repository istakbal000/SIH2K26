import pandas as pd
import os
from pathlib import Path
from .config import config, get_project_root

def get_raw_data_path():
    return get_project_root() / config['data']['raw_dir']

def get_processed_data_path():
    return get_project_root() / config['data']['processed_dir']

def find_dataset(prefix, extension='.csv'):
    """Finds a dataset in the raw directory or subdirectories matching a given prefix."""
    raw_dir = get_raw_data_path()
    if not raw_dir.exists():
        return None
        
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().startswith(prefix.lower()) and file.endswith(extension):
                return Path(root) / file
    return None

def load_firms_data():
    file_path = find_dataset('fire_nrt') or find_dataset('firms')
    if file_path:
        print(f"Found FIRMS dataset at {file_path}")
        return pd.read_csv(file_path)
    
    print("\n" + "="*50)
    print("REAL DATA REQUIRED: NASA FIRMS Satellite Detections")
    print("SOURCE: https://firms.modaps.eosdis.nasa.gov/")
    print("EXPECTED FORMAT: CSV with columns like latitude, longitude, acq_date, acq_time, etc.")
    print("="*50 + "\n")
    return None

def load_weather_data():
    file_path = find_dataset('weather')
    if file_path:
        print(f"Found Weather dataset at {file_path}")
        return pd.read_csv(file_path)
    
    print("\n" + "="*50)
    print("REAL DATA REQUIRED: Historical Weather Data (e.g., ERA5)")
    print("SOURCE: Copernicus/ECMWF or similar")
    print("EXPECTED FORMAT: CSV containing historical temperature, humidity, wind, rainfall.")
    print("="*50 + "\n")
    return None

def load_osm_pois():
    file_path = find_dataset('central', extension='.pbf') or find_dataset('osm', extension='.pbf')
    if file_path:
        print(f"Found OSM dataset at {file_path}")
        return str(file_path) # Return path instead of reading immediately due to PBF format
    
    print("\n" + "="*50)
    print("REAL DATA REQUIRED: OpenStreetMap POIs and Land Cover")
    print("SOURCE: OpenStreetMap (Overpass API / Geofabrik)")
    print("EXPECTED FORMAT: .osm.pbf with coordinates and tags for forest/industrial areas.")
    print("="*50 + "\n")
    return None

def load_uci_forest_data():
    file_path = find_dataset('forestfires')
    if file_path:
        print(f"Found UCI Forest Fires dataset at {file_path}")
        return pd.read_csv(file_path)
    
    print("\n" + "="*50)
    print("REAL DATA REQUIRED: UCI Forest Fires Dataset")
    print("SOURCE: archive (5)/forestfires.csv")
    print("EXPECTED FORMAT: CSV with FFMC, DMC, temp, RH, etc.")
    print("="*50 + "\n")
    return None

def load_industrial_sensor_data():
    # The file has a very long name starting with "Indoor Fire Dataset"
    file_path = find_dataset('Indoor Fire Dataset', extension='.csv')
    if file_path:
        print(f"Found Industrial Sensor dataset at {file_path}")
        return pd.read_csv(file_path)
    
    print("\n" + "="*50)
    print("REAL DATA REQUIRED: Industrial Sensor Dataset")
    print("EXPECTED FORMAT: CSV with sensor readings and 'fire' column.")
    print("="*50 + "\n")
    return None

def inspect_raw_data():
    print("Inspecting existing raw datasets...")
    firms_df = load_firms_data()
    weather_df = load_weather_data()
    osm_df = load_osm_pois()
    
    data_dict = {
        'firms': firms_df,
        'weather': weather_df,
        'osm': osm_df
    }
    
    available = {k: v for k, v in data_dict.items() if v is not None}
    print(f"Detected {len(available)} raw data sources.")
    return available
