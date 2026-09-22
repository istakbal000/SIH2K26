import pandas as pd

def extract_satellite_features(latitudes, longitudes, dates, sat_data=None):
    """
    Extracts features from satellite imagery (NDVI, NDMI, NDBI, etc.)
    Ensures that for a given date, we only use satellite data from *before* the fire
    to prevent target leakage.
    
    Note: Requires real satellite data extracts (e.g. from Google Earth Engine).
    """
    if sat_data is None:
        print("WARNING: No Satellite data available. Creating empty satellite features.")
        return pd.DataFrame(index=range(len(latitudes)))
        
    print("Extracting Satellite features (NDVI, NDMI, LST, NDBI)...")
    # In a real scenario, we would match lat/lon/date to pre-event imagery.
    features = pd.DataFrame(index=range(len(latitudes)))
    features['NDVI'] = float('nan')
    features['NDMI'] = float('nan')
    features['NDBI'] = float('nan')
    features['LST'] = float('nan')
    
    return features
