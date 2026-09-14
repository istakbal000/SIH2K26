import pandas as pd

def extract_firms_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts core features directly from the FIRMS dataset.
    """
    if df is None or df.empty:
        return pd.DataFrame()
        
    features = pd.DataFrame(index=df.index)
    
    # Core variables
    if 'brightness' in df.columns:
        features['brightness'] = df['brightness']
    if 'bright_t31' in df.columns:
        features['bright_t31'] = df['bright_t31']
    if 'frp' in df.columns:
        features['frp'] = df['frp']
    if 'confidence' in df.columns:
        # Depending on instrument (MODIS vs VIIRS), confidence can be categorical (l,n,h) or numerical (0-100)
        # We will attempt to map categorical to numerical if necessary, or just leave it.
        features['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(50)
        
    if 'daynight' in df.columns:
        features['is_day'] = (df['daynight'].astype(str).str.upper() == 'D').astype(int)
        
    # Time-based features
    if 'datetime' in df.columns:
        features['hour'] = df['datetime'].dt.hour
        features['month'] = df['datetime'].dt.month
        
    return features
