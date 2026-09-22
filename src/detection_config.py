from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"
SPLIT_DATA_DIR = DATA_DIR / "split"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

TARGET_COL = 'Fire Alarm'

FEATURES = [
    'UTC', 'Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]',
    'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5',
    'NC0.5', 'NC1.0', 'NC2.5', 'CNT'
]

DEFAULT_RAW_DATA = RAW_DATA_DIR / "fire_data.csv"
DEFAULT_PREPROCESSED_DATA = PREPROCESSED_DATA_DIR / "fire_data_processed.csv"