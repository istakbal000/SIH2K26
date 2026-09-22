import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib
import logging
import argparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from train import TARGET_COL, load_split_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(split_dir: str, models_dir: str, output_dir: str):
    split_path = Path(split_dir)
    models_path = Path(models_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_split_data(split_path)

    if (models_path / "best_model.pkl").exists():
        model_path = models_path / "best_model.pkl"
        model_name = "Best Model"
    else:
        model_files = list(models_path.glob("*.pkl"))
        if not model_files:
            logger.error("No model files found in models directory")
            return
        model_path = model_files[0]
        model_name = model_path.stem

    logger.info(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    for X, y, name in [(X_train, y_train, "train"), (X_val, y_val, "validation"), (X_test, y_test, "test")]:
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

        logger.info(f"\n{name.upper()} Metrics:")
        for k, v in metrics.items():
            display = "nan" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}" if isinstance(v, float) else v
            logger.info(f"  {k}: {display}")

        logger.info(f"\nClassification Report:\n{classification_report(y, y_pred, zero_division=0)}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred, labels=[0, 1])}")

        with open(output_path / f"{name}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path / f"confusion_matrix_{name}.png", dpi=150)
        plt.close()

        if y_prob is not None and metrics['roc_auc'] is not None and y.nunique() > 1:
            fpr, tpr, _ = roc_curve(y, y_prob)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path / f"roc_curve_{name}.png", dpi=150)
            plt.close()

    if hasattr(model, 'feature_importances_'):
        feature_names = X_train.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(output_path / "feature_importance.png", dpi=150)
        plt.close()

    logger.info(f"Evaluation complete! Artifacts saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the best fire detection model")
    parser.add_argument("--split-dir", type=str, default="data/split")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--output", type=str, default="outputs/evaluation")
    args = parser.parse_args()
    main(args.split_dir, args.models_dir, args.output)