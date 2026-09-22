import pandas as pd

def clean_firms_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the FIRMS dataset.
    - Handles missing values
    - Parses dates and times
    - Ensures numerical types for relevant columns
    """
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    # Required columns for FIRMS processing
    required_cols = ['latitude', 'longitude', 'acq_date', 'acq_time']
    for col in required_cols:
        if col not in df.columns:
            print(f"WARNING: Missing required column {col} in FIRMS data.")
            return df
            
    # Parse date and time
    try:
        # acq_time might be a string like '0435' or int like 435
        df['acq_time_str'] = df['acq_time'].astype(str).str.zfill(4)
        df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time_str'].str[:2] + ':' + df['acq_time_str'].str[2:4])
    except Exception as e:
        print(f"Error parsing date/time: {e}")
        
    # Handle optional numerical columns if they exist
    numeric_cols = ['brightness', 'bright_t31', 'frp', 'confidence']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Drop rows without valid coordinates
    df = df.dropna(subset=['latitude', 'longitude', 'acq_date'])
    
    return df
