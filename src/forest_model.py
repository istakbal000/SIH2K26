from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# pyrefly: ignore [missing-import]
import xgboost as xgb
from .config import config

def get_forest_model_candidates():
    """
    Returns a dictionary of uninitialized models to evaluate, 
    based on the configuration in config.yaml.
    """
    candidates = {}
    seed = config.get('random_seed', 42)
    
    if config['models'].get('random_forest', True):
        candidates['RandomForest'] = RandomForestClassifier(
            n_estimators=50, 
            max_depth=10,
            n_jobs=-1,
            random_state=seed, 
            class_weight='balanced'
        )
        
    return candidates
