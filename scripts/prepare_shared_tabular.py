import os
import sys
from pathlib import Path
import pandas as pd

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import get_processed_data_path

def main():
    print("=== Preparing Shared Tabular Data (Forest vs Industrial) ===")
    
    processed_dir = get_processed_data_path()
    forest_path = processed_dir / 'forest_fire_training.csv'
    
    if not forest_path.exists():
        print(f"ERROR: {forest_path} not found. Run prepare_data.py first.")
        return
        
    df_forest_raw = pd.read_csv(forest_path)
    
    forest_cols = {}
    if 'temperature' in df_forest_raw.columns:
        forest_cols['temperature'] = 'temperature'
    elif 'bright_t31' in df_forest_raw.columns:
        # fallback to brightness temp if weather temp missing
        forest_cols['bright_t31'] = 'temperature' 
        
    if 'humidity' in df_forest_raw.columns:
        forest_cols['humidity'] = 'humidity'
    else:
        # Dummy fallback
        df_forest_raw['humidity'] = 40.0
        forest_cols['humidity'] = 'humidity'
        
    df_forest = df_forest_raw[list(forest_cols.keys())].copy()
    df_forest.rename(columns=forest_cols, inplace=True)
    df_forest['is_forest_fire'] = 1.0
    
    # 2. Load Industrial Fire data
    industrial_path = processed_dir / 'industrial_fire_training.csv'
    if not industrial_path.exists():
        print(f"ERROR: {industrial_path} not found. Run prepare_industrial_data.py first.")
        return
        
    df_ind_raw = pd.read_csv(industrial_path)
    
    # Only keep rows where the industrial fire is actually happening (fire == 1)
    # Because we are classifying "Given there is a fire, is it forest or industrial?"
    # Oh wait, df_ind_raw has 'is_industrial_fire'. We only want positive cases!
    df_ind_raw = df_ind_raw[df_ind_raw['is_industrial_fire'] == 1.0]
    
    ind_cols = {}
    if 'Temperature_Room' in df_ind_raw.columns:
        ind_cols['Temperature_Room'] = 'temperature'
    if 'Humidity_Room' in df_ind_raw.columns:
        ind_cols['Humidity_Room'] = 'humidity'
        
    df_ind = df_ind_raw[list(ind_cols.keys())].copy()
    df_ind.rename(columns=ind_cols, inplace=True)
    df_ind['is_forest_fire'] = 0.0
    
    # Also we should only keep positive cases for forest fires!
    # Let's reload and filter forest fires
    df_forest_raw = pd.read_csv(forest_path)
    if 'is_forest_fire' in df_forest_raw.columns:
        df_forest_raw = df_forest_raw[df_forest_raw['is_forest_fire'] == 1.0]
    
    if 'humidity' not in df_forest_raw.columns:
        df_forest_raw['humidity'] = 40.0
        
    df_forest = df_forest_raw[list(forest_cols.keys())].copy()
    df_forest.rename(columns=forest_cols, inplace=True)
    df_forest['is_forest_fire'] = 1.0
    
    # 3. Combine
    df_shared = pd.concat([df_forest, df_ind], ignore_index=True)
    df_shared.dropna(inplace=True)
    
    # 4. Save
    out_path = processed_dir / 'shared_fire_classification.csv'
    df_shared.to_csv(out_path, index=False)
    
    print(f"Successfully combined {len(df_forest)} forest fire records and {len(df_ind)} industrial fire records.")
    print(f"Saved {len(df_shared)} total records to {out_path}")

if __name__ == "__main__":
    main()
