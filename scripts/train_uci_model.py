import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_uci_forest_data
from src.uci_model import get_uci_model_candidates
from src.config import get_project_root, config

def train_uci_model():
    print("=== Training UCI Meteorological Fire Model ===")
    
    df = load_uci_forest_data()
    if df is None:
        print("ERROR: Could not load UCI Forest Fires dataset.")
        return
        
    print(f"Loaded {len(df)} records from UCI dataset.")
    
    # Target variable
    target_col = 'area'
    
    # The area variable is heavily skewed towards 0.0. 
    # Standard practice is to apply a log transform: log(area + 1)
    y = np.log1p(df[target_col])
    X = df.drop(columns=[target_col])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    candidates = get_uci_model_candidates()
    best_model = None
    best_name = ""
    best_rmse = float('inf')
    best_r2 = -float('inf')
    
    for name, pipeline in candidates.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_test)
        
        # Calculate metrics on the log-transformed scale
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        print(f"{name} Validation - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_model = pipeline
            best_name = name
            
    print(f"\nBest Model selected: {best_name} with RMSE: {best_rmse:.4f} and R2: {best_r2:.4f}")
    
    # Save the model
    models_dir = get_project_root() / config['output']['models']
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'uci_meteorological_model.pkl'
    
    joblib.dump(best_model, model_path)
    print(f"Saved model to {model_path}")
    print("=== Training Complete ===")

if __name__ == "__main__":
    train_uci_model()
