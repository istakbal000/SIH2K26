import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_DIR
    / "data"
    / "Agriculture"
    / "cleaned"
    / "agriculture_training_2020.csv"
)

TEST_FILE = (
    PROJECT_DIR
    / "data"
    / "Agriculture"
    / "cleaned"
    / "agriculture_testing_2021.csv"
)

MODEL_DIR = (
    PROJECT_DIR
    / "data"
    / "models"
)

RESULT_DIR = (
    PROJECT_DIR
    / "data"
    / "results"
)

MODEL_FILE = (
    MODEL_DIR
    / "agricultural_fire_random_forest_2020.pkl"
)

PREDICTION_FILE = (
    RESULT_DIR
    / "agriculture_test_predictions_2021.csv"
)

IMPORTANCE_FILE = (
    RESULT_DIR
    / "agriculture_feature_importance.csv"
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "num_hotspots",
    "mean_brightness",
    "max_brightness",
    "mean_bright_t31",
    "max_bright_t31",
    "mean_frp",
    "max_frp",
    "mean_confidence",
    "min_distance_km"
]

TARGET_COLUMN = "agricultural_fire"


# ============================================================
# START
# ============================================================

print("\n================================================")
print("AGRICULTURAL FIRE CLASSIFIER")
print("RANDOM FOREST")
print("================================================")


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading 2020 training dataset...")

train_df = pd.read_csv(TRAIN_FILE)

print("Training samples:", len(train_df))

print("\n2020 label distribution:")
print(
    train_df[TARGET_COLUMN].value_counts()
)


# ============================================================
# LOAD TESTING DATA
# ============================================================

print("\nLoading 2021 testing dataset...")

test_df = pd.read_csv(TEST_FILE)

print("Testing samples:", len(test_df))

print("\n2021 label distribution:")
print(
    test_df[TARGET_COLUMN].value_counts()
)


# ============================================================
# CHECK FEATURES
# ============================================================

print("\n================================================")
print("CHECKING FEATURES")
print("================================================")

missing_train = [
    column
    for column in FEATURE_COLUMNS
    if column not in train_df.columns
]

missing_test = [
    column
    for column in FEATURE_COLUMNS
    if column not in test_df.columns
]

if missing_train:
    print("\nMissing features in training data:")
    print(missing_train)
    raise ValueError("Training features are missing.")

if missing_test:
    print("\nMissing features in testing data:")
    print(missing_test)
    raise ValueError("Testing features are missing.")

print("\nAll required features are present.")


# ============================================================
# PREPARE X AND Y
# ============================================================

X_train = train_df[FEATURE_COLUMNS].copy()
y_train = train_df[TARGET_COLUMN].astype(int)

X_test = test_df[FEATURE_COLUMNS].copy()
y_test = test_df[TARGET_COLUMN].astype(int)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_features = []

for column in FEATURE_COLUMNS:

    if X_train[column].nunique() <= 1:
        constant_features.append(column)

if constant_features:

    print("\nConstant features detected:")
    print(constant_features)

    X_train = X_train.drop(
        columns=constant_features
    )

    X_test = X_test.drop(
        columns=constant_features
    )

    active_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in constant_features
    ]

else:

    active_features = FEATURE_COLUMNS.copy()

print("\nFeatures used for training:")

for feature in active_features:
    print(" -", feature)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\n================================================")
print("TRAINING RANDOM FOREST")
print("================================================")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("\nModel training completed.")


# ============================================================
# PREDICTION ON 2021
# ============================================================

print("\n================================================")
print("TESTING ON UNSEEN 2021 DATA")
print("================================================")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)


# Probability of Agricultural Fire

agricultural_probability = y_probability[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n================================================")
print("MODEL PERFORMANCE")
print("================================================")

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Agricultural",
            "Agricultural"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n================================================")
print("CONFUSION MATRIX")
print("================================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n                 Predicted")
print("              Non-Agri  Agri")
print(
    f"Actual Non-Agri   {cm[0][0]:4d}   {cm[0][1]:4d}"
)

print(
    f"Actual Agri       {cm[1][0]:4d}   {cm[1][1]:4d}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n================================================")
print("FEATURE IMPORTANCE")
print("================================================")

importance_df = pd.DataFrame({
    "feature": active_features,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print(
    importance_df.to_string(index=False)
)


# ============================================================
# CREATE PREDICTION FILE
# ============================================================

print("\n================================================")
print("SAVING PREDICTIONS")
print("================================================")

prediction_df = test_df[
    [
        "field_id",
        "field_name",
        "field_category",
        "latitude",
        "longitude",
        TARGET_COLUMN
    ]
].copy()

prediction_df["predicted_agricultural_fire"] = y_pred

prediction_df["agricultural_fire_probability"] = (
    agricultural_probability
)

prediction_df["prediction"] = np.where(
    y_pred == 1,
    "Agricultural",
    "Non-Agricultural"
)

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)

print(
    "\nPrediction file saved:"
)

print(PREDICTION_FILE)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)

print(
    "\nFeature importance saved:"
)

print(IMPORTANCE_FILE)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "features": active_features
    },
    MODEL_FILE
)

print(
    "\nModel saved:"
)

print(MODEL_FILE)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n================================================")
print("TRAINING COMPLETE")
print("================================================")

print(
    "\nTraining year : 2020"
)

print(
    "Testing year  : 2021"
)

print(
    "Training rows :", len(X_train)
)

print(
    "Testing rows  :", len(X_test)
)

print(
    f"\nAccuracy      : {accuracy:.2%}"
)

print(
    f"Precision     : {precision:.2%}"
)

print(
    f"Recall        : {recall:.2%}"
)

print(
    f"F1 Score      : {f1:.2%}"
)

print("\n================================================")
print("DONE")
print("================================================")