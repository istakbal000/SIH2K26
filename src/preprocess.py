import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

from config import TARGET_COL, FEATURES
from features import sanitize, engineer_features, TIMESTAMP_COL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = FEATURES

def load_raw_data(raw_path: Path) -> pd.DataFrame:
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw data shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    df = df.copy()

    df.columns = df.columns.str.strip()

    index_cols = [c for c in df.columns if c.startswith('Unnamed')]
    if index_cols:
        logger.info(f"Dropping index columns: {index_cols}")
        df = df.drop(columns=index_cols)

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in data: {missing_cols}")

    rename_map = {c: sanitize(c) for c in df.columns}
    df = df.rename(columns=rename_map)

    numeric_cols = [sanitize(c) for c in FEATURE_COLS]
    numeric_cols = [c for c in numeric_cols if c in df.columns and c != 'UTC']

    df = df.dropna(subset=[TARGET_COL])

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    logger.info(f"Cleaned data shape: {df.shape}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Performing feature engineering...")
    df = df.copy()

    if 'UTC' in df.columns:
        if pd.api.types.is_numeric_dtype(df['UTC']):
            df[TIMESTAMP_COL] = pd.to_datetime(df['UTC'], unit='s', errors='coerce')
        else:
            df[TIMESTAMP_COL] = pd.to_datetime(df['UTC'], errors='coerce')

    df = engineer_features(df)

    df = df.dropna()

    logger.info(f"Feature engineered data shape: {df.shape}")
    return df

def save_preprocessed(df: pd.DataFrame, output_path: Path):
    logger.info(f"Saving preprocessed data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Preprocessing complete!")

def main(raw_path: str, output_path: str):
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = feature_engineering(df)
    save_preprocessed(df, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess fire detection dataset")
    parser.add_argument("--raw", type=str, default="data/raw/fire_data.csv", help="Path to raw CSV file")
    parser.add_argument("--output", type=str, default="data/preprocessed/fire_data_processed.csv", help="Output path for preprocessed data")
    args = parser.parse_args()
    main(args.raw, args.output)