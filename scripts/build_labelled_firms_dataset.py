import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

AGRI_DIR = (
    PROJECT_DIR
    / "data"
    / "Agriculture"
    / "cleaned"
)

FIRMS_FILE = (
    PROJECT_DIR
    / "data"
    / "firm_data"
    / "DL_FIRE_J1V-C2_802955"
    / "fire_archive_J1V-C2_802955.csv"
)

OUTPUT_DIR = AGRI_DIR


# ============================================================
# LABELING SETTINGS
# ============================================================

MAX_DISTANCE_METERS = 500
MAX_DATE_DIFFERENCE_DAYS = 3

EARTH_RADIUS_METERS = 6371000


# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print("LOADING DATA")
print("==============================================")

firms = pd.read_csv(FIRMS_FILE)

agri_2020 = pd.read_csv(
    AGRI_DIR / "agriculture_2020.csv"
)

agri_2021 = pd.read_csv(
    AGRI_DIR / "agriculture_2021.csv"
)

print("FIRMS rows:", len(firms))
print("2020 reference fields:", len(agri_2020))
print("2021 reference fields:", len(agri_2021))


# ============================================================
# DATE CONVERSION
# ============================================================

firms["acq_date"] = pd.to_datetime(
    firms["acq_date"],
    errors="coerce"
)

agri_2020["date"] = pd.to_datetime(
    agri_2020["date"],
    errors="coerce"
)

agri_2021["date"] = pd.to_datetime(
    agri_2021["date"],
    errors="coerce"
)


# ============================================================
# FUNCTION
# ============================================================

def build_dataset(agri, year):

    print("\n==============================================")
    print(f"BUILDING LABELLED DATASET FOR {year}")
    print("==============================================")

    # --------------------------------------------------------
    # Geographic study area
    # --------------------------------------------------------

    min_lat = agri["latitude"].min()
    max_lat = agri["latitude"].max()

    min_lon = agri["longitude"].min()
    max_lon = agri["longitude"].max()

    # Small geographic buffer
    buffer = 0.1

    firms_local = firms[
        (firms["latitude"] >= min_lat - buffer) &
        (firms["latitude"] <= max_lat + buffer) &
        (firms["longitude"] >= min_lon - buffer) &
        (firms["longitude"] <= max_lon + buffer)
    ].copy()

    print(
        "FIRMS records in study area:",
        len(firms_local)
    )

    # --------------------------------------------------------
    # Create BallTree
    # --------------------------------------------------------

    firms_coords = np.radians(
        firms_local[
            ["latitude", "longitude"]
        ].values
    )

    tree = BallTree(
        firms_coords,
        metric="haversine"
    )

    results = []

    # --------------------------------------------------------
    # Process each agricultural reference field
    # --------------------------------------------------------

    for _, field in agri.iterrows():

        field_lat = field["latitude"]
        field_lon = field["longitude"]
        field_date = field["date"]

        field_coord = np.radians(
            [[field_lat, field_lon]]
        )

        # ----------------------------------------------------
        # Find FIRMS points within 500 m
        # ----------------------------------------------------

        radius_radians = (
            MAX_DISTANCE_METERS /
            EARTH_RADIUS_METERS
        )

        indices = tree.query_radius(
            field_coord,
            r=radius_radians
        )[0]

        # ----------------------------------------------------
        # Process candidate FIRMS points
        # ----------------------------------------------------

        for idx in indices:

            fire = firms_local.iloc[idx]

            fire_lat = fire["latitude"]
            fire_lon = fire["longitude"]

            # ------------------------------------------------
            # Calculate EXACT distance for this FIRMS point
            # ------------------------------------------------

            fire_coord = np.radians(
                [[fire_lat, fire_lon]]
            )

            distance_radians = (
                tree
                .query(
                    fire_coord,
                    k=1
                )[0][0][0]
            )

            # NOTE:
            # The BallTree above contains all FIRMS points,
            # so this gives the distance to the nearest FIRMS
            # point, which may not be the current candidate.
            #
            # Therefore calculate the Haversine distance
            # directly below.

            lat1 = np.radians(field_lat)
            lon1 = np.radians(field_lon)

            lat2 = np.radians(fire_lat)
            lon2 = np.radians(fire_lon)

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

            distance_m = (
                EARTH_RADIUS_METERS * c
            )

            # ------------------------------------------------
            # Date difference
            # ------------------------------------------------

            date_difference = abs(
                (
                    fire["acq_date"]
                    - field_date
                ).days
            )

            # ------------------------------------------------
            # Strict filtering
            # ------------------------------------------------

            if distance_m > MAX_DISTANCE_METERS:
                continue

            if date_difference > MAX_DATE_DIFFERENCE_DAYS:
                continue

            # ------------------------------------------------
            # Only burnt/unburnt reference categories
            # ------------------------------------------------

            field_label = field[
                "agricultural_fire"
            ]

            if pd.isna(field_label):
                continue

            # ------------------------------------------------
            # Store labelled FIRMS record
            # ------------------------------------------------

            results.append({

                # FIRMS location
                "latitude": fire_lat,
                "longitude": fire_lon,

                # FIRMS thermal features
                "brightness": fire["brightness"],
                "bright_t31": fire["bright_t31"],
                "frp": fire["frp"],
                "confidence": fire["confidence"],
                "scan": fire["scan"],
                "track": fire["track"],

                # FIRMS metadata
                "satellite": fire["satellite"],
                "instrument": fire["instrument"],
                "daynight": fire["daynight"],

                # Date
                "acq_date": fire["acq_date"],
                "acq_time": fire["acq_time"],

                # Reference information
                "reference_field_id":
                    field["field_id"],

                "reference_field_category":
                    field["field_category"],

                "distance_to_reference_m":
                    distance_m,

                "date_difference_days":
                    date_difference,

                # TARGET
                "agricultural_fire":
                    int(field_label)
            })

    result = pd.DataFrame(results)

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n----------------------------------------------")
    print(f"RESULTS FOR {year}")
    print("----------------------------------------------")

    print(
        "Labelled FIRMS rows:",
        len(result)
    )

    if len(result) > 0:

        print("\nLabel distribution:")
        print(
            result[
                "agricultural_fire"
            ].value_counts()
        )

        print("\nReference category distribution:")
        print(
            result[
                "reference_field_category"
            ].value_counts()
        )

        print("\nDistance statistics:")
        print(
            result[
                "distance_to_reference_m"
            ].describe()
        )

        print("\nDate difference distribution:")
        print(
            result[
                "date_difference_days"
            ].value_counts()
            .sort_index()
        )

        output_file = (
            OUTPUT_DIR
            / f"firms_agricultural_labelled_{year}.csv"
        )

        result.to_csv(
            output_file,
            index=False
        )

        print(
            "\nSaved:",
            output_file
        )

    else:

        print(
            "\nWARNING:"
            " No labelled FIRMS records found."
        )


# ============================================================
# BUILD 2020 DATASET
# ============================================================

build_dataset(
    agri_2020,
    2020
)


# ============================================================
# BUILD 2021 DATASET
# ============================================================

build_dataset(
    agri_2021,
    2021
)


print("\n==============================================")
print("LABELLED FIRMS DATASET CREATION COMPLETED")
print("==============================================")