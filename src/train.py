import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
import joblib
import logging
import argparse
import json
import time
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_COL = 'Fire Alarm'

MODELS = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    'xgboost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42),
    'lightgbm': LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
}

def load_split_data(split_dir: Path):
    logger.info("Loading split data...")
    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_val = pd.read_csv(split_dir / "X_val.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")
    y_train = pd.read_csv(split_dir / "y_train.csv").squeeze()
    y_val = pd.read_csv(split_dir / "y_val.csv").squeeze()
    y_test = pd.read_csv(split_dir / "y_test.csv").squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X, y, dataset_name="test"):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
    }
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y, y_prob)
        except ValueError:
            metrics['roc_auc'] = None

    logger.info(f"\n{dataset_name.upper()} Metrics:")
    for k, v in metrics.items():
        display = "nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}" if isinstance(v, float) else v
        logger.info(f"  {k}: {display}")
    logger.info(f"\nClassification Report:\n{classification_report(y, y_pred, zero_division=0)}")

    return metrics

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, models_dict, output_dir):
    results = {}
    best_score = 0
    best_model_name = None
    single_class_val = y_val.nunique() < 2
    selection_criteria = "TEST (val is single-class)" if single_class_val else "val"

    for name, model in models_dict.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*50}")

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        val_metrics = evaluate_model(model, X_val, y_val, "validation")
        test_metrics = evaluate_model(model, X_test, y_test, "test")

        results[name] = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'train_time': train_time,
        }

        if single_class_val:
            score = test_metrics['f1']
        else:
            score = val_metrics['f1']

        if score > best_score:
            best_score = score
            best_model_name = name

    logger.info(f"\n{'='*50}")
    logger.info(f"BEST MODEL: {best_model_name} (selected by {selection_criteria} F1: {best_score:.4f})")
    if single_class_val:
        logger.info("NOTE: validation set had a single class, so model selection used TEST F1 instead.")
    logger.info(f"{'='*50}")

    return results, best_model_name

def main(split_dir: str, models_output_dir: str):
    split_path = Path(split_dir)
    output_path = Path(models_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_split_data(split_path)

    scaler_src = split_path / "scaler.pkl"
    if scaler_src.exists():
        scaler = joblib.load(scaler_src)
        joblib.dump(scaler, output_path / "scaler.pkl")
        logger.info(f"Scaler copied to {output_path / 'scaler.pkl'}")

    medians_src = split_path / "medians.json"
    if medians_src.exists():
        import shutil
        shutil.copy(medians_src, output_path / "medians.json")
        logger.info(f"Medians copied to {output_path / 'medians.json'}")

    results, best_model_name = train_and_evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test, MODELS, output_path
    )

    best_model = MODELS[best_model_name]
    best_model.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val])
    )
    joblib.dump(best_model, output_path / "best_model.pkl")

    with open(output_path / "feature_names.json", 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)

    with open(output_path / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_path / "best_model_info.json", 'w') as f:
        json.dump({
            'model_name': best_model_name,
            'val_f1': results[best_model_name]['val_metrics']['f1'],
            'test_metrics': results[best_model_name]['test_metrics'],
        }, f, indent=2)

    logger.info(f"Best model saved to {output_path / 'best_model.pkl'}")
    logger.info(f"Results saved to {output_path / 'training_results.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fire detection models")
    parser.add_argument("--split-dir", type=str, default="data/split")
    parser.add_argument("--output", type=str, default="models")
    args = parser.parse_args()
    main(args.split_dir, args.output)
