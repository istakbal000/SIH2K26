import yaml
from pathlib import Path

# Base project path relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config():
    config_path = PROJECT_ROOT / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

def get_project_root():
    return PROJECT_ROOT

config = load_config()
