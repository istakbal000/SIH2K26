import os
import sys
import json
import joblib
import pandas as pd
from pathlib import Path

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import get_processed_data_path
from src.model_evaluation import evaluate_model, plot_confusion_matrix, plot_feature_importance
from src.config import config, get_project_root

def evaluate_single_model(model_type, data_path, model_path, features_path):
    print(f"=== Evaluating {model_type} Fire Model ===")
    
    if not (data_path.exists() and model_path.exists() and features_path.exists()):
        print(f"Missing required files for {model_type} evaluation.")
        return
        
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    
    with open(features_path, 'r') as f:
        metadata = json.load(f)
        
    features = metadata['features']
    target_col = f'is_{model_type.lower()}_fire'
    
    # Check if target is present
    if target_col not in df.columns:
        print(f"Target {target_col} missing in {model_type} data.")
        return
        
    X = df[features].fillna(0)
    y_true = df[target_col]
    
    # We should ideally evaluate on a held-out test set.
    # For now, we evaluate on the full dataset or validation portion if we kept track.
    # Since we didn't save a test set explicitly, we'll evaluate on the loaded data.
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    metrics, cm = evaluate_model(y_true, y_pred, y_prob)
    
    outputs_dir = get_project_root() / 'outputs'
    metrics_dir = outputs_dir / 'metrics'
    plots_dir = outputs_dir / 'plots'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Metrics
    metrics_file = metrics_dir / f'{model_type.lower()}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save Plots
    cm_path = plots_dir / f'{model_type.lower()}_confusion_matrix.png'
    plot_confusion_matrix(cm, f'{model_type} Confusion Matrix', cm_path)
    
    fi_path = plots_dir / f'{model_type.lower()}_feature_importance.png'
    plot_feature_importance(model, features, f'{model_type} Feature Importance', fi_path)
    
    print(f"Metrics: F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Outputs saved to {outputs_dir}\n")

def main():
    processed_dir = get_processed_data_path()
    models_dir = get_project_root() / config['output']['models']
    
    evaluate_single_model(
        'Forest',
        processed_dir / 'forest_fire_training.csv',
        models_dir / 'forest_fire_model.pkl',
        models_dir / 'forest_features.json'
    )
    
    evaluate_single_model(
        'Industrial',
        processed_dir / 'industrial_fire_training.csv',
        models_dir / 'industrial_fire_model.pkl',
        models_dir / 'industrial_features.json'
    )

if __name__ == "__main__":
    main()
