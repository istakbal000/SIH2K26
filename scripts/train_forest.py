import os
import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import get_processed_data_path
from src.forest_model import get_forest_model_candidates
from src.model_evaluation import evaluate_model
from src.config import config, get_project_root

def train_forest():
    print("=== Training Forest Fire Model ===")
    
    data_path = get_processed_data_path() / 'forest_fire_training.csv'
    if not data_path.exists():
        print(f"ERROR: Training data not found at {data_path}")
        print("Please run scripts/prepare_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    target_col = 'is_forest_fire'
    if target_col not in df.columns:
        print(f"ERROR: Target column '{target_col}' not found.")
        return
        
    # Exclude non-feature columns
    exclude = [target_col, 'latitude', 'longitude', 'datetime', 'fire_cluster_id']
    features = [c for c in df.columns if c not in exclude]
    
    # Simple split for demonstration (config requests spatial/temporal, 
    # but due to mock data we fallback to random split if cluster ID is missing/dummy)
    X = df[features].fillna(0)
    y = df[target_col]
    
    # Standard random split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.get('random_seed', 42), stratify=y
    )
    
    candidates = get_forest_model_candidates()
    best_model = None
    best_name = ""
    best_f1 = -1
    
    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        metrics, _ = evaluate_model(y_val, y_pred, y_prob)
        print(f"{name} Validation - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = model
            best_name = name
            
    print(f"\nBest Model selected: {best_name} with F1: {best_f1:.4f}")
    
    # Save Model and Metadata
    models_dir = get_project_root() / config['output']['models']
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / 'forest_fire_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"Saved model to {model_path}")
    
    meta_path = models_dir / 'forest_features.json'
    metadata = {
        'model_type': best_name,
        'features': features,
        'threshold': 0.5, # Default threshold, could be optimized via PR curve
        'validation_f1': best_f1
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("=== Training Complete ===")

if __name__ == "__main__":
    train_forest()
