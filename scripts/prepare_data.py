import os
import sys
from pathlib import Path
import pandas as pd

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import inspect_raw_data, load_firms_data, get_processed_data_path
from src.data_cleaning import clean_firms_data
from src.feature_engineering import generate_samples

def main():
    print("=== Starting Data Preparation Pipeline ===")
    
    # 1. Inspect data and print required messages if missing
    data_dict = inspect_raw_data()
    
    # 2. Process FIRMS data
    firms_raw = data_dict.get('firms')
    if firms_raw is None:
        print("\nFATAL: FIRMS data is mandatory for this pipeline.")
        print("Please place a FIRMS CSV in data/raw/ and try again.")
        print("Exiting pipeline gracefully.")
        
        # Save a dummy data dictionary so the structure exists
        _save_data_dictionary()
        return
        
    print(f"\nProcessing {len(firms_raw)} raw FIRMS records...")
    firms_clean = clean_firms_data(firms_raw)
    
    # 3. Generate Features and Samples
    forest_df, industrial_df = generate_samples(firms_clean, data_dict)
    
    # 4. Save Processed datasets
    processed_dir = get_processed_data_path()
    
    if not forest_df.empty:
        forest_path = processed_dir / 'forest_fire_training.csv'
        forest_df.to_csv(forest_path, index=False)
        print(f"Saved {len(forest_df)} forest training samples to {forest_path}")
        
    if not industrial_df.empty:
        ind_path = processed_dir / 'industrial_fire_training.csv'
        industrial_df.to_csv(ind_path, index=False)
        print(f"Saved {len(industrial_df)} industrial training samples to {ind_path}")
        
    # 5. Create Data Dictionary
    _save_data_dictionary()
    
    print("\n=== Data Preparation Complete ===")

def _save_data_dictionary():
    dict_path = get_processed_data_path() / 'data_dictionary.csv'
    
    data = [
        ['latitude', 'Latitude coordinate', 'NASA FIRMS', 'degrees', 'float', 'point', 'satellite pass', 'detection-time'],
        ['longitude', 'Longitude coordinate', 'NASA FIRMS', 'degrees', 'float', 'point', 'satellite pass', 'detection-time'],
        ['brightness', 'Brightness temperature 21 (Kelvin)', 'NASA FIRMS', 'K', 'float', 'pixel', 'satellite pass', 'detection-time'],
        ['frp', 'Fire Radiative Power (MW)', 'NASA FIRMS', 'MW', 'float', 'pixel', 'satellite pass', 'detection-time'],
        ['is_day', 'Day/Night Flag (1=Day)', 'NASA FIRMS', 'binary', 'int', 'pixel', 'satellite pass', 'detection-time'],
        ['distance_to_forest', 'Distance to nearest forest (m)', 'OSM', 'm', 'float', 'vector', 'static', 'pre-fire'],
        ['temperature', 'Air temperature', 'Weather API', 'C', 'float', 'grid', 'hourly', 'detection-time'],
        ['is_forest_fire', 'Target label for forest fire model', 'Derived', 'binary', 'int', 'point', 'static', 'post-fire']
    ]
    
    df = pd.DataFrame(data, columns=[
        'feature', 'description', 'source', 'unit', 'data_type', 
        'spatial_resolution', 'temporal_resolution', 'pre_or_post_fire'
    ])
    
    df.to_csv(dict_path, index=False)
    print(f"Saved data dictionary to {dict_path}")

if __name__ == "__main__":
    main()
