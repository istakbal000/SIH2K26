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
    / "agriculture_training_2020.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Search radius around each agricultural reference field
RADIUS_KM = 2.0

# Earth's radius
EARTH_RADIUS_KM = 6371.0


# ============================================================
# START
# ============================================================

print("\n==============================================")
print("BUILDING 2020 AGRICULTURE TRAINING DATASET")
print("==============================================")


# ============================================================
# LOAD AGRICULTURAL REFERENCE DATA
# ============================================================

print("\nLoading agricultural reference data...")

agri = pd.read_csv(AGRI_FILE)

print("Agricultural fields:", len(agri))


# ============================================================
# LOAD FIRMS DATA
# ============================================================

print("\nLoading FIRMS data...")

firms = pd.read_csv(FIRMS_FILE)

print("FIRMS rows:", len(firms))


# ============================================================
# CONVERT DATE
# ============================================================

firms["acq_date"] = pd.to_datetime(
    firms["acq_date"],
    errors="coerce"
)


# ============================================================
# KEEP ONLY 2020 FIRMS DATA
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

# FIRMS confidence sometimes contains values such as "nn".
# Convert invalid values to NaN.

firms_2020["confidence"] = pd.to_numeric(
    firms_2020["confidence"],
    errors="coerce"
)

firms_2020["brightness"] = pd.to_numeric(
    firms_2020["brightness"],
    errors="coerce"
)

firms_2020["bright_t31"] = pd.to_numeric(
    firms_2020["bright_t31"],
    errors="coerce"
)

firms_2020["frp"] = pd.to_numeric(
    firms_2020["frp"],
    errors="coerce"
)

firms_2020["scan"] = pd.to_numeric(
    firms_2020["scan"],
    errors="coerce"
)

firms_2020["track"] = pd.to_numeric(
    firms_2020["track"],
    errors="coerce"
)


# ============================================================
# HAVERSINE DISTANCE FUNCTION
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate distance between geographic coordinates
    using the Haversine formula.

    Returns distance in kilometers.
    """

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
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return EARTH_RADIUS_KM * c


# ============================================================
# BUILD TRAINING FEATURES
# ============================================================

results = []

print("\nProcessing agricultural fields...")


for index, field in agri.iterrows():

    field_lat = field["latitude"]
    field_lon = field["longitude"]

    # --------------------------------------------------------
    # Calculate distance from this field to every FIRMS point
    # --------------------------------------------------------

    distances = haversine_distance(
        field_lat,
        field_lon,
        firms_2020["latitude"].values,
        firms_2020["longitude"].values
    )

    # --------------------------------------------------------
    # Select FIRMS hotspots within 2 km
    # --------------------------------------------------------

    nearby_mask = distances <= RADIUS_KM

    nearby = firms_2020[
        nearby_mask
    ].copy()

    nearby["distance_km"] = distances[
        nearby_mask
    ]


    # ========================================================
    # CASE 1: NO FIRMS HOTSPOT WITHIN 2 KM
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

            "agricultural_fire":
                int(field["agricultural_fire"])
        }


    # ========================================================
    # CASE 2: FIRMS HOTSPOTS FOUND
    # ========================================================

    else:

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

            # Number of FIRMS hotspots around field
            "num_hotspots":
                len(nearby),

            # Brightness features
            "mean_brightness":
                nearby["brightness"].mean(),

            "max_brightness":
                nearby["brightness"].max(),

            # Thermal band features
            "mean_bright_t31":
                nearby["bright_t31"].mean(),

            "max_bright_t31":
                nearby["bright_t31"].max(),

            # Fire Radiative Power
            "mean_frp":
                nearby["frp"].mean(),

            "max_frp":
                nearby["frp"].max(),

            # Confidence
            "mean_confidence":
                nearby["confidence"].mean(),

            # Distance to closest FIRMS hotspot
            "min_distance_km":
                nearby["distance_km"].min(),

            # TARGET
            "agricultural_fire":
                int(field["agricultural_fire"])
        }


    results.append(result)


# ============================================================
# CREATE DATAFRAME
# ============================================================

training_data = pd.DataFrame(
    results
)


# ============================================================
# HANDLE MISSING NUMERIC VALUES
# ============================================================

numeric_columns = [
    "mean_brightness",
    "max_brightness",
    "mean_bright_t31",
    "max_bright_t31",
    "mean_frp",
    "max_frp",
    "mean_confidence",
    "min_distance_km"
]

for column in numeric_columns:

    training_data[column] = pd.to_numeric(
        training_data[column],
        errors="coerce"
    )

    training_data[column] = (
        training_data[column]
        .fillna(0)
    )


# ============================================================
# SAVE DATASET
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
# DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("2020 TRAINING DATASET CREATED")
print("==============================================")

print(
    "\nNumber of training samples:",
    len(training_data)
)

print("\nColumns:")

for column in training_data.columns:
    print(" -", column)


print("\n----------------------------------------------")
print("LABEL DISTRIBUTION")
print("----------------------------------------------")

print(
    training_data[
        "agricultural_fire"
    ].value_counts()
)


print("\n----------------------------------------------")
print("HOTSPOT STATISTICS")
print("----------------------------------------------")

print(
    training_data[
        "num_hotspots"
    ].describe()
)


print("\n----------------------------------------------")
print("FEATURE PREVIEW")
print("----------------------------------------------")

print(
    training_data.head()
)


print("\n----------------------------------------------")
print("OUTPUT FILE")
print("----------------------------------------------")

print(
    OUTPUT_FILE
)


print("\n==============================================")
print("DONE")
print("==============================================")