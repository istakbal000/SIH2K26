import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "data" / "models" / "agricultural_fire_random_forest_2020.pkl"

TEST_DATA_PATH = PROJECT_DIR / "data" / "Agriculture" / "cleaned" / "agriculture_testing_2021.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "agriculture_predictions_with_confidence_2021.csv"


# ==========================================
# FEATURES USED BY THE RANDOM FOREST
# ==========================================

FEATURE_COLUMNS = [
    "num_hotspots",
    "mean_brightness",
    "max_brightness",
    "mean_bright_t31",
    "max_bright_t31",
    "mean_frp",
    "max_frp",
    "min_distance_km"
]


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading Random Forest model...")

loaded_data = joblib.load(MODEL_PATH)

print("Model file loaded successfully!")

if isinstance(loaded_data, dict):
    model = loaded_data["model"]
else:
    model = loaded_data

print("Random Forest model loaded successfully!")


# ==========================================
# LOAD TEST DATA
# ==========================================

print("\nLoading 2021 test data...")

df = pd.read_csv(TEST_DATA_PATH)

print(f"Test samples: {len(df)}")


# ==========================================
# PREPARE FEATURES
# ==========================================

X = df[FEATURE_COLUMNS].copy()

X = X.apply(pd.to_numeric, errors="coerce")

X = X.fillna(0)


# ==========================================
# PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(X)

probabilities = model.predict_proba(X)


# ==========================================
# GET CONFIDENCE SCORE
# ==========================================

class_to_index = {
    class_value: index
    for index, class_value in enumerate(model.classes_)
}

agricultural_index = class_to_index[1]
non_agricultural_index = class_to_index[0]


agricultural_probability = probabilities[:, agricultural_index]
non_agricultural_probability = probabilities[:, non_agricultural_index]


# ==========================================
# CREATE OUTPUT
# ==========================================

df["predicted_class"] = predictions

df["predicted_fire_type"] = df["predicted_class"].map({
    0: "Non-Agricultural Fire",
    1: "Agricultural Fire"
})


df["agricultural_probability"] = agricultural_probability

df["non_agricultural_probability"] = non_agricultural_probability


# Confidence = probability of predicted class

df["confidence_score"] = probabilities.max(axis=1)

df["confidence_percentage"] = (
    df["confidence_score"] * 100
).round(2)


# ==========================================
# SAVE RESULT
# ==========================================

df.to_csv(OUTPUT_PATH, index=False)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("PREDICTION COMPLETED")
print("==========================================")

print(f"\nOutput saved to:")
print(OUTPUT_PATH)

print("\nPrediction distribution:")
print(df["predicted_fire_type"].value_counts())


print("\nSample predictions:")

print(
    df[
        [
            "latitude",
            "longitude",
            "predicted_fire_type",
            "confidence_percentage"
        ]
    ].head(10).to_string(index=False)
)


print("\n==========================================")
print("Done!")
print("==========================================")