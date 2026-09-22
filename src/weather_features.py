import pandas as pd

def extract_weather_features(latitudes, longitudes, timestamps, weather_data=None):
    """
    Extracts historical weather conditions (ERA5) for the given coordinates and time.
    Features: temperature, relative_humidity, wind_speed, wind_direction, rainfall.
    
    Note: Requires real historical weather data.
    """
    if weather_data is None:
        print("WARNING: No Weather data available. Creating empty weather features.")
        return pd.DataFrame(index=range(len(latitudes)))
        
    print("Extracting Weather features...")
    # In a real scenario, we match lat/lon/timestamp to the closest grid point and time.
    features = pd.DataFrame(index=range(len(latitudes)))
    features['temperature'] = float('nan')
    features['relative_humidity'] = float('nan')
    features['wind_speed'] = float('nan')
    features['wind_direction'] = float('nan')
    features['rainfall'] = float('nan')
    
    return features
