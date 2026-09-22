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
    / "agriculture_2021.csv"
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
    / "agriculture_testing_2021.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Same radius used for the 2020 training dataset.
# Keeping this identical is important.
RADIUS_KM = 2.0

EARTH_RADIUS_KM = 6371.0


# ============================================================
# START
# ============================================================

print("\n==============================================")
print("BUILDING 2021 AGRICULTURE TESTING DATASET")
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
# CONVERT FIRMS DATE
# ============================================================

firms["acq_date"] = pd.to_datetime(
    firms["acq_date"],
    errors="coerce"
)


# ============================================================
# KEEP ONLY 2021 FIRMS DATA
# ============================================================

firms_2021 = firms[
    firms["acq_date"].dt.year == 2021
].copy()

print(
    "\nFIRMS observations in 2021:",
    len(firms_2021)
)


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

# Some FIRMS confidence values can contain strings such as "nn".
# Invalid values are converted to NaN.

firms_2021["confidence"] = pd.to_numeric(
    firms_2021["confidence"],
    errors="coerce"
)

firms_2021["brightness"] = pd.to_numeric(
    firms_2021["brightness"],
    errors="coerce"
)

firms_2021["bright_t31"] = pd.to_numeric(
    firms_2021["bright_t31"],
    errors="coerce"
)

firms_2021["frp"] = pd.to_numeric(
    firms_2021["frp"],
    errors="coerce"
)

firms_2021["scan"] = pd.to_numeric(
    firms_2021["scan"],
    errors="coerce"
)

firms_2021["track"] = pd.to_numeric(
    firms_2021["track"],
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
    Calculate geographic distance using
    the Haversine formula.

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
# BUILD TESTING FEATURES
# ============================================================

results = []

print("\nProcessing agricultural fields...")


for index, field in agri.iterrows():

    field_lat = field["latitude"]
    field_lon = field["longitude"]

    # --------------------------------------------------------
    # Calculate distance from this field
    # to every 2021 FIRMS hotspot
    # --------------------------------------------------------

    distances = haversine_distance(
        field_lat,
        field_lon,
        firms_2021["latitude"].values,
        firms_2021["longitude"].values
    )

    # --------------------------------------------------------
    # Select FIRMS hotspots within 2 km
    # --------------------------------------------------------

    nearby_mask = distances <= RADIUS_KM

    nearby = firms_2021[
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

            # TRUE TEST LABEL
            # This is kept aside until model evaluation.
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

            # Number of nearby FIRMS hotspots
            "num_hotspots":
                len(nearby),

            # Brightness
            "mean_brightness":
                nearby["brightness"].mean(),

            "max_brightness":
                nearby["brightness"].max(),

            # Thermal band
            "mean_bright_t31":
                nearby["bright_t31"].mean(),

            "max_bright_t31":
                nearby["bright_t31"].max(),

            # Fire Radiative Power
            "mean_frp":
                nearby["frp"].mean(),

            "max_frp":
                nearby["frp"].max(),

            # FIRMS confidence
            "mean_confidence":
                nearby["confidence"].mean(),

            # Distance to nearest FIRMS hotspot
            "min_distance_km":
                nearby["distance_km"].min(),

            # TRUE TEST LABEL
            "agricultural_fire":
                int(field["agricultural_fire"])
        }


    results.append(result)


# ============================================================
# CREATE DATAFRAME
# ============================================================

testing_data = pd.DataFrame(
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

    testing_data[column] = pd.to_numeric(
        testing_data[column],
        errors="coerce"
    )

    testing_data[column] = (
        testing_data[column]
        .fillna(0)
    )


# ============================================================
# SAVE TESTING DATASET
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

testing_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("2021 TESTING DATASET CREATED")
print("==============================================")

print(
    "\nNumber of testing samples:",
    len(testing_data)
)


print("\nColumns:")

for column in testing_data.columns:
    print(" -", column)


print("\n----------------------------------------------")
print("TEST LABEL DISTRIBUTION")
print("----------------------------------------------")

print(
    testing_data[
        "agricultural_fire"
    ].value_counts()
)


print("\n----------------------------------------------")
print("HOTSPOT STATISTICS")
print("----------------------------------------------")

print(
    testing_data[
        "num_hotspots"
    ].describe()
)


print("\n----------------------------------------------")
print("FEATURE PREVIEW")
print("----------------------------------------------")

print(
    testing_data.head()
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