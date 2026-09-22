import os
import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import get_processed_data_path
from src.model_evaluation import evaluate_model
from src.config import config, get_project_root

def train_shared_tabular_model():
    print("=== Training Shared Tabular Model (Forest vs Industrial) ===")
    
    data_path = get_processed_data_path() / 'shared_fire_classification.csv'
    if not data_path.exists():
        print(f"ERROR: Training data not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    target_col = 'is_forest_fire'
    features = ['temperature', 'humidity']
    
    X = df[features].fillna(0)
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # We use a lightweight Random Forest to avoid memory limits
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=10,
        n_jobs=-1,
        random_state=42, 
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    metrics, _ = evaluate_model(y_val, y_pred, y_prob)
    print(f"Validation - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    models_dir = get_project_root() / config['output']['models']
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / 'shared_fire_model.pkl'
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    
    meta_path = models_dir / 'shared_features.json'
    metadata = {
        'model_type': 'RandomForest',
        'features': features,
        'threshold': 0.5,
        'validation_f1': metrics['f1']
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("=== Training Complete ===")

if __name__ == "__main__":
    train_shared_tabular_model()
