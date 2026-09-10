import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

AGRI_FILE = (
    PROJECT_DIR
    / "data"
    / "Agriculture"
    / "cleaned"
    / "agriculture_2020.csv"
)

FIRMS_FILE = (
    PROJECT_DIR
    / "data"
    / "firm_data"
    / "DL_FIRE_J1V-C2_802955"
    / "fire_archive_J1V-C2_802955.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "Agriculture"
    / "cleaned"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "agriculture_training_2020_v2.csv"
)

RADIUS_KM = 2.0
EARTH_RADIUS_KM = 6371.0


# ============================================================
# START
# ============================================================

print("\n================================================")
print("BUILDING 2020 AGRICULTURE TRAINING DATASET V2")
print("================================================")


# ============================================================
# LOAD AGRICULTURAL REFERENCE DATA
# ============================================================

print("\nLoading agricultural reference data...")

agri = pd.read_csv(AGRI_FILE)

print(
    "Agricultural fields:",
    len(agri)
)


# ============================================================
# LOAD FIRMS
# ============================================================

print("\nLoading FIRMS data...")

firms = pd.read_csv(FIRMS_FILE)

print(
    "Total FIRMS rows:",
    len(firms)
)


# ============================================================
# DATE AND TIME PROCESSING
# ============================================================

print("\nProcessing FIRMS dates and times...")

firms["acq_date"] = pd.to_datetime(
    firms["acq_date"],
    errors="coerce"
)

firms["acq_time"] = pd.to_numeric(
    firms["acq_time"],
    errors="coerce"
)

# Convert HHMM format into decimal hour.
# Example:
# 630  -> 6.50
# 1230 -> 12.50
# 1830 -> 18.50

firms["acq_hour"] = (
    (firms["acq_time"] // 100)
    +
    ((firms["acq_time"] % 100) / 60)
)


# ============================================================
# FILTER 2020
# ============================================================

firms_2020 = firms[
    firms["acq_date"].dt.year == 2020
].copy()

print(
    "\nFIRMS observations in 2020:",
    len(firms_2020)
)


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "brightness",
    "bright_t31",
    "frp",
    "confidence",
    "scan",
    "track"
]

for column in numeric_columns:

    firms_2020[column] = pd.to_numeric(
        firms_2020[column],
        errors="coerce"
    )


# ============================================================
# DAY/NIGHT CLEANING
# ============================================================

firms_2020["daynight"] = (
    firms_2020["daynight"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ============================================================
# HAVERSINE FUNCTION
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return EARTH_RADIUS_KM * c


# ============================================================
# PROCESS FIELDS
# ============================================================

results = []

print("\nProcessing agricultural fields...")


for index, field in agri.iterrows():

    field_lat = field["latitude"]
    field_lon = field["longitude"]

    distances = haversine_distance(
        field_lat,
        field_lon,
        firms_2020["latitude"].values,
        firms_2020["longitude"].values
    )

    nearby_mask = (
        distances <= RADIUS_KM
    )

    nearby = firms_2020[
        nearby_mask
    ].copy()

    nearby["distance_km"] = (
        distances[nearby_mask]
    )


    # ========================================================
    # NO HOTSPOTS
    # ========================================================

    if len(nearby) == 0:

        result = {

            "field_id":
                field["field_id"],

            "field_name":
                field["field_name"],

            "field_category":
                field["field_category"],

            "latitude":
                field_lat,

            "longitude":
                field_lon,

            "num_hotspots":
                0,

            "mean_brightness":
                0,

            "max_brightness":
                0,

            "mean_bright_t31":
                0,

            "max_bright_t31":
                0,

            "mean_frp":
                0,

            "max_frp":
                0,

            "mean_confidence":
                0,

            "min_distance_km":
                RADIUS_KM,

            "mean_acq_hour":
                0,

            "num_daytime_hotspots":
                0,

            "num_nighttime_hotspots":
                0,

            "mean_scan":
                0,

            "mean_track":
                0,

            "max_scan":
                0,

            "max_track":
                0,

            "agricultural_fire":
                int(
                    field["agricultural_fire"]
                )
        }


    # ========================================================
    # HOTSPOTS FOUND
    # ========================================================

    else:

        daytime_count = (
            nearby["daynight"]
            .eq("D")
            .sum()
        )

        nighttime_count = (
            nearby["daynight"]
            .eq("N")
            .sum()
        )

        result = {

            "field_id":
                field["field_id"],

            "field_name":
                field["field_name"],

            "field_category":
                field["field_category"],

            "latitude":
                field_lat,

            "longitude":
                field_lon,

            "num_hotspots":
                len(nearby),

            "mean_brightness":
                nearby["brightness"].mean(),

            "max_brightness":
                nearby["brightness"].max(),

            "mean_bright_t31":
                nearby["bright_t31"].mean(),

            "max_bright_t31":
                nearby["bright_t31"].max(),

            "mean_frp":
                nearby["frp"].mean(),

            "max_frp":
                nearby["frp"].max(),

            "mean_confidence":
                nearby["confidence"].mean(),

            "min_distance_km":
                nearby["distance_km"].min(),

            "mean_acq_hour":
                nearby["acq_hour"].mean(),

            "num_daytime_hotspots":
                int(daytime_count),

            "num_nighttime_hotspots":
                int(nighttime_count),

            "mean_scan":
                nearby["scan"].mean(),

            "mean_track":
                nearby["track"].mean(),

            "max_scan":
                nearby["scan"].max(),

            "max_track":
                nearby["track"].max(),

            "agricultural_fire":
                int(
                    field["agricultural_fire"]
                )
        }


    results.append(result)


# ============================================================
# CREATE DATAFRAME
# ============================================================

training_data = pd.DataFrame(
    results
)


# ============================================================
# CLEAN NUMERIC FEATURES
# ============================================================

numeric_output_columns = [

    "mean_brightness",
    "max_brightness",

    "mean_bright_t31",
    "max_bright_t31",

    "mean_frp",
    "max_frp",

    "mean_confidence",
    "min_distance_km",

    "mean_acq_hour",

    "num_daytime_hotspots",
    "num_nighttime_hotspots",

    "mean_scan",
    "mean_track",

    "max_scan",
    "max_track"
]


for column in numeric_output_columns:

    training_data[column] = pd.to_numeric(
        training_data[column],
        errors="coerce"
    )

    training_data[column] = (
        training_data[column]
        .fillna(0)
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

training_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n================================================")
print("2020 TRAINING DATASET V2 CREATED")
print("================================================")

print(
    "\nNumber of samples:",
    len(training_data)
)

print("\nColumns:")

for column in training_data.columns:
    print(" -", column)


print("\n================================================")
print("LABEL DISTRIBUTION")
print("================================================")

print(
    training_data[
        "agricultural_fire"
    ].value_counts()
)


print("\n================================================")
print("TEMPORAL FEATURE SUMMARY")
print("================================================")

print(
    training_data[
        [
            "mean_acq_hour",
            "num_daytime_hotspots",
            "num_nighttime_hotspots"
        ]
    ].describe()
)


print("\n================================================")
print("OUTPUT FILE")
print("================================================")

print(
    OUTPUT_FILE
)


print("\n================================================")
print("DONE")
print("================================================")