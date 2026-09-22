import pandas as pd
import numpy as np
import json
import sys
import argparse
import logging
from pathlib import Path
import joblib

from config import TARGET_COL
from features import sanitize, FINAL_FEATURES, add_ratio_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_input(args):
    if args.input:
        return json.loads(args.input)
    if args.input_file:
        return json.loads(Path(args.input_file).read_text())
    raw = sys.stdin.read().strip()
    return json.loads(raw)

def normalize_keys(data: dict) -> dict:
    return {sanitize(k): v for k, v in data.items()}

def build_feature_row(data: dict, medians: dict) -> pd.DataFrame:
    row = {}
    for feat in FINAL_FEATURES:
        if feat in data and data[feat] is not None:
            row[feat] = data[feat]
        elif feat in medians:
            row[feat] = medians[feat]
        else:
            row[feat] = 0.0

    df = pd.DataFrame([row])

    if 'UTC' in data and data['UTC'] is not None:
        utc = data['UTC']
        if isinstance(utc, (int, float)):
            ts = pd.to_datetime(utc, unit='s', errors='coerce')
        else:
            ts = pd.to_datetime(utc, errors='coerce')
        df['hour'] = medians.get('hour', 0) if pd.isna(ts) else ts.hour
        df['day_of_week'] = medians.get('day_of_week', 0) if pd.isna(ts) else ts.dayofweek
        df['month'] = medians.get('month', 1) if pd.isna(ts) else ts.month
    else:
        df['hour'] = medians.get('hour', 0)
        df['day_of_week'] = medians.get('day_of_week', 0)
        df['month'] = medians.get('month', 1)

    df = add_ratio_features(df)

    return df[FINAL_FEATURES]

def predict_dict(data: dict, model, scaler, medians: dict) -> dict:
    data = normalize_keys(data)
    X = build_feature_row(data, medians)
    engineered = {c: (None if pd.isna(X.iloc[0][c]) else float(X.iloc[0][c])) for c in FINAL_FEATURES}

    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    proba = model.predict_proba(X_scaled)[0]
    pred = int(model.predict(X_scaled)[0])
    confidence = float(proba[list(model.classes_).index(pred)])

    return {
        "fire_detected": bool(pred == 1),
        "prediction": pred,
        "confidence": round(confidence, 6),
        "probability": {
            "no_fire": round(float(proba[0]), 6),
            "fire": round(float(proba[1]), 6),
        },
        "message": "FIRE DETECTED" if pred == 1 else "NO FIRE",
        "input_features": engineered,
    }

def fuse_predictions(tabular: dict, image: dict, w_tabular: float = 0.6) -> dict:
    """Weighted fusion of the tabular (environmental) fire probability and the image CNN probability."""
    w_tabular = max(0.0, min(1.0, w_tabular))
    w_image = round(1.0 - w_tabular, 4)
    p_fire = w_tabular * tabular["probability"]["fire"] + w_image * image["probability"]["fire"]
    fire = p_fire >= 0.5
    confidence = p_fire if fire else 1 - p_fire
    return {
        "fire_detected": bool(fire),
        "prediction": int(fire),
        "confidence": round(confidence, 6),
        "probability": {"no_fire": round(1 - p_fire, 6), "fire": round(p_fire, 6)},
        "message": "FIRE DETECTED" if fire else "NO FIRE",
        "weights": {"tabular": round(w_tabular, 4), "image": w_image},
    }

def main():
    parser = argparse.ArgumentParser(description="Predict fire from environmental conditions and output JSON")
    parser.add_argument("--input", type=str, default=None, help="JSON string with environmental readings")
    parser.add_argument("--input-file", type=str, default=None, help="Path to a JSON file with environmental readings")
    parser.add_argument("--model", type=str, default="models/best_model.pkl", help="Path to trained model")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl", help="Path to fitted scaler")
    parser.add_argument("--medians", type=str, default="models/medians.json", help="Path to training medians for missing features")
    parser.add_argument("--feature-names", type=str, default="models/feature_names.json", help="Path to training feature names")
    parser.add_argument("--output", type=str, default=None, help="Optional file path to write JSON result to")
    args = parser.parse_args()

    data = load_input(args)

    model = joblib.load(Path(args.model))
    scaler = joblib.load(Path(args.scaler))

    medians = json.loads(Path(args.medians).read_text())
    feature_names = json.loads(Path(args.feature_names).read_text())

    result = predict_dict(data, model, scaler, medians)

    if list(result["input_features"]) != list(feature_names):
        logger.warning(f"Feature mismatch:\n  model expects {len(feature_names)} features\n  got {len(result['input_features'])}")

    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        Path(args.output).write_text(output)
        logger.info(f"Result written to {args.output}")

if __name__ == "__main__":
    main()