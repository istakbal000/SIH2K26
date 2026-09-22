import pandas as pd
from detection_config import FEATURES, TARGET_COL

TIMESTAMP_COL = 'timestamp'

def sanitize(name: str) -> str:
    return name.replace('[', '_').replace(']', '')

BASE_FEATURES = [sanitize(f) for f in FEATURES if f != 'UTC']
TIME_FEATURES = ['hour', 'day_of_week', 'month']
RATIO_FEATURES = ['temp_humidity_ratio', 'pm_ratio', 'tvoc_eco2_ratio']

FINAL_FEATURES = BASE_FEATURES + TIME_FEATURES + RATIO_FEATURES

def add_datetime_features(df):
    if 'UTC' not in df.columns:
        return df
    utc = df['UTC']
    if pd.api.types.is_numeric_dtype(utc):
        parsed = pd.to_datetime(utc, unit='s', errors='coerce')
    else:
        parsed = pd.to_datetime(utc, errors='coerce')
    df['hour'] = parsed.dt.hour
    df['day_of_week'] = parsed.dt.dayofweek
    df['month'] = parsed.dt.month
    return df

def add_ratio_features(df):
    t = sanitize('Temperature[C]')
    h = sanitize('Humidity[%]')
    if t in df.columns and h in df.columns:
        df['temp_humidity_ratio'] = df[t] / (df[h] + 1)
    if 'PM2.5' in df.columns and 'PM1.0' in df.columns:
        df['pm_ratio'] = df['PM2.5'] / (df['PM1.0'] + 1)
    tv = sanitize('TVOC[ppb]')
    ec = sanitize('eCO2[ppm]')
    if tv in df.columns and ec in df.columns:
        df['tvoc_eco2_ratio'] = df[tv] / (df[ec] + 1)
    return df

def engineer_features(df):
    df = add_datetime_features(df)
    df = add_ratio_features(df)
    if 'UTC' in df.columns:
        df = df.drop(columns=['UTC'])
    return df