import os
import sys
from pathlib import Path
import pandas as pd

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_industrial_sensor_data, get_processed_data_path

def main():
    print("=== Preparing Industrial Sensor Data ===")
    
    df_raw = load_industrial_sensor_data()
    if df_raw is None:
        print("Failed to load raw industrial sensor data.")
        return
        
    print(f"Loaded {len(df_raw)} records from industrial sensor dataset.")
    
    # Select features
    features = [
        'CO2_Room', 'H2_Room', 'PM05_Room', 'PM100_Room', 'PM10_Room',
        'PM25_Room', 'PM40_Room', 'PM_Room_Typical_Size', 'PM_Total_Room',
        'VOC_Room_RAW', 'Temperature_Room', 'Humidity_Room', 'CO_Room'
    ]
    target = 'fire'
    
    # Ensure all required columns are present
    available_cols = set(df_raw.columns)
    missing_features = [f for f in features if f not in available_cols]
    if missing_features:
        print(f"Warning: missing feature columns: {missing_features}")
        features = [f for f in features if f in available_cols]
        
    if target not in available_cols:
        print(f"Fatal: target column '{target}' not found!")
        return
        
    # Extract dataset
    columns_to_keep = features + [target]
    df_processed = df_raw[columns_to_keep].copy()
    
    # Rename target column to match what training expects
    df_processed.rename(columns={target: 'is_industrial_fire'}, inplace=True)
    
    # Clean data (drop rows with missing values in these critical sensor columns)
    initial_len = len(df_processed)
    df_processed.dropna(inplace=True)
    print(f"Dropped {initial_len - len(df_processed)} rows with missing values.")
    
    # Save processed dataset
    processed_dir = get_processed_data_path()
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / 'industrial_fire_training.csv'
    
    df_processed.to_csv(out_path, index=False)
    print(f"Saved {len(df_processed)} processed industrial records to {out_path}")

if __name__ == "__main__":
    main()
