import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
# pyrefly: ignore [missing-import]
import xgboost as xgb

def get_uci_preprocessor():
    """
    Creates a preprocessor for the UCI Forest Fires dataset.
    Numerical features: X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, rain
    Categorical features: month, day
    """
    numeric_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    categorical_features = ['month', 'day']

    # Standard scaling for numeric features, One-hot encoding for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    return preprocessor

def get_uci_model_candidates():
    """
    Returns a dictionary of candidate regression pipelines.
    """
    preprocessor = get_uci_preprocessor()
    
    candidates = {
        'RandomForest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
    }
    
    return candidates
