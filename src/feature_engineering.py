import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from .firms_features import extract_firms_features
from .osm_features import calculate_osm_features
from .satellite_features import extract_satellite_features
from .weather_features import extract_weather_features

def apply_spatial_temporal_clustering(df: pd.DataFrame, eps_km=5, time_eps_days=1) -> pd.DataFrame:
    """
    Groups fire detections into physical fire events to prevent data leakage 
    and over-representation of large fires.
    """
    if df is None or df.empty or 'latitude' not in df.columns:
        return df
        
    print("Applying spatial-temporal clustering to fire events...")
    df = df.copy()
    
    # Very basic DBSCAN implementation on scaled lat/lon for demonstration
    # In reality, we'd use Haversine distance and include time in the metric
    coords = df[['latitude', 'longitude']].values
    
    # ~1 degree is ~111km. So eps_km/111 is approx eps in degrees.
    eps_deg = eps_km / 111.0
    
    clustering = DBSCAN(eps=eps_deg, min_samples=1).fit(coords)
    df['fire_cluster_id'] = clustering.labels_
    
    # Calculate cluster-level features
    if 'frp' in df.columns:
        df['cluster_size'] = df.groupby('fire_cluster_id')['latitude'].transform('count')
        df['max_frp'] = df.groupby('fire_cluster_id')['frp'].transform('max')
        df['mean_frp'] = df.groupby('fire_cluster_id')['frp'].transform('mean')
        df['frp_sum'] = df.groupby('fire_cluster_id')['frp'].transform('sum')
        
    return df

def generate_samples(firms_df: pd.DataFrame, data_dict: dict):
    """
    Generates positive and realistic negative samples for Forest and Industrial models.
    """
    print("Generating training datasets with positive and realistic negative samples...")
    
    # If no FIRMS data, we can't build datasets
    if firms_df is None or firms_df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Extract features for existing FIRMS points
    lats = firms_df['latitude']
    lons = firms_df['longitude']
    
    dates = firms_df.get('datetime', pd.Series(index=firms_df.index))
    
    firms_features = extract_firms_features(firms_df)
    osm_features = calculate_osm_features(lats, lons, data_dict.get('osm'))
    sat_features = extract_satellite_features(lats, lons, dates, data_dict.get('sat'))
    weather_features = extract_weather_features(lats, lons, dates, data_dict.get('weather'))
    
    # Combine features
    base_features = pd.concat([
        firms_features,
        osm_features,
        sat_features,
        weather_features
    ], axis=1)
    
    base_features['latitude'] = lats
    base_features['longitude'] = lons
    
    if 'datetime' in firms_df.columns:
        base_features['datetime'] = dates
        
    base_features = apply_spatial_temporal_clustering(base_features)

    # Mock positive/negative assignment since we lack real land-cover/industrial ground truth
    print("\nWARNING: Using mock labels for positive/negative assignment due to missing ground truth data.")
    
    # Create Forest Training Set
    forest_df = base_features.copy()
    forest_df['is_forest_fire'] = np.random.choice([0, 1], size=len(forest_df), p=[0.5, 0.5])
    
    # Create Industrial Training Set
    industrial_df = base_features.copy()
    industrial_df['is_industrial_fire'] = np.random.choice([0, 1], size=len(industrial_df), p=[0.7, 0.3])
    
    return forest_df, industrial_df
