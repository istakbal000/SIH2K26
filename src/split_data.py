import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import argparse

from config import TARGET_COL
from features import FINAL_FEATURES, TIMESTAMP_COL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading preprocessed data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Data shape: {df.shape}")
    return df

def make_features(df: pd.DataFrame):
    X = df[FINAL_FEATURES]
    y = df[TARGET_COL]
    return X, y

def create_indices(df, mode, test_size, val_size, random_state):
    n = len(df)
    if mode == 'time':
        logger.info("Using TIME-BASED split (no shuffle)")
        if TIMESTAMP_COL in df.columns:
            df = df.sort_values(TIMESTAMP_COL, kind='mergesort').reset_index(drop=True)
        else:
            logger.warning(f"Column '{TIMESTAMP_COL}' not found, splitting by row order")
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        train_idx = np.arange(train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
    else:
        logger.info("Using RANDOM split (shuffled)")
        idx = np.arange(n)
        trainval_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=df[TARGET_COL]
        )
        val_ratio = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=val_ratio, random_state=random_state,
            stratify=df[TARGET_COL].iloc[trainval_idx]
        )
    return df, train_idx, val_idx, test_idx

def split_and_scale(df: pd.DataFrame, output_dir: Path, mode='random', test_size=0.2, val_size=0.1, random_state=42):
    logger.info("Splitting data into train/val/test...")

    df, train_idx, val_idx, test_idx = create_indices(df, mode, test_size, val_size, random_state)

    X, y = make_features(df)
    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    output_dir.mkdir(parents=True, exist_ok=True)

    medians = X_train.median().round(6).to_dict()
    with open(output_dir / "medians.json", 'w') as f:
        import json
        json.dump(medians, f, indent=2)

    X_train_scaled.to_csv(output_dir / "X_train.csv", index=False)
    X_val_scaled.to_csv(output_dir / "X_val.csv", index=False)
    X_test_scaled.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    joblib.dump(scaler, output_dir / "scaler.pkl")

    with open(output_dir / "feature_names.json", 'w') as f:
        import json
        json.dump(list(X_train.columns), f, indent=2)

    logger.info(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
    logger.info(f"Target distribution:\nTrain: {y_train.value_counts().to_dict()}\nVal: {y_val.value_counts().to_dict()}\nTest: {y_test.value_counts().to_dict()}")

    for name, y_part in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        if y_part.nunique() < 2:
            logger.warning(f"WARNING: {name} set contains only ONE class ({y_part.value_counts().to_dict()}). "
                           f"Metrics on this split may be misleading.")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def main(processed_path: str, output_dir: str, mode: str):
    df = load_processed_data(Path(processed_path))
    split_and_scale(df, Path(output_dir), mode=mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and scale fire detection dataset")
    parser.add_argument("--input", type=str, default="data/preprocessed/fire_data_processed.csv")
    parser.add_argument("--output", type=str, default="data/split")
    parser.add_argument("--mode", type=str, choices=['random', 'time'], default='random',
                        help="random = shuffled split, time = chronological split (train on past, test on future)")
    args = parser.parse_args()
    main(args.input, args.output, args.mode)