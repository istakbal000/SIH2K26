import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

AGRI_DIR = PROJECT_DIR / "data" / "Agriculture" / "cleaned"

FIRMS_FILE = (
    PROJECT_DIR
    / "data"
    / "firm_data"
    / "DL_FIRE_J1V-C2_802955"
    / "fire_archive_J1V-C2_802955.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Distances we want to investigate
RADII_METERS = [250, 500, 1000, 2000]

# Date windows we want to investigate
DATE_WINDOWS = [0, 1, 3]


# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print("LOADING AGRICULTURAL REFERENCE DATA")
print("==============================================")

agri_2020 = pd.read_csv(
    AGRI_DIR / "agriculture_2020.csv"
)

agri_2021 = pd.read_csv(
    AGRI_DIR / "agriculture_2021.csv"
)

print("2020 agricultural rows:", len(agri_2020))
print("2021 agricultural rows:", len(agri_2021))


print("\n==============================================")
print("LOADING FIRMS DATA")
print("==============================================")

firms = pd.read_csv(FIRMS_FILE)

print("FIRMS rows:", len(firms))


# ============================================================
# PREPARE DATES
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
# FUNCTION TO ANALYZE MATCHING
# ============================================================

def analyze_matching(agri, year):

    print("\n")
    print("==============================================")
    print(f"ANALYZING YEAR: {year}")
    print("==============================================")

    # --------------------------------------------------------
    # Bounding box around agricultural reference locations
    # --------------------------------------------------------

    min_lat = agri["latitude"].min()
    max_lat = agri["latitude"].max()

    min_lon = agri["longitude"].min()
    max_lon = agri["longitude"].max()

    print("Agricultural reference bounds:")
    print("Latitude :", min_lat, "to", max_lat)
    print("Longitude:", min_lon, "to", max_lon)

    # --------------------------------------------------------
    # Add a small geographic buffer
    # --------------------------------------------------------

    lat_buffer = 0.1
    lon_buffer = 0.1

    firms_local = firms[
        (firms["latitude"] >= min_lat - lat_buffer) &
        (firms["latitude"] <= max_lat + lat_buffer) &
        (firms["longitude"] >= min_lon - lon_buffer) &
        (firms["longitude"] <= max_lon + lon_buffer)
    ].copy()

    print("\nFIRMS points inside local study area:")
    print(len(firms_local))

    if len(firms_local) == 0:
        print("NO FIRMS DATA FOUND IN STUDY AREA.")
        return

    # --------------------------------------------------------
    # Create BallTree
    # --------------------------------------------------------

    earth_radius_m = 6371000

    firms_coords = np.radians(
        firms_local[
            ["latitude", "longitude"]
        ].values
    )

    tree = BallTree(
        firms_coords,
        metric="haversine"
    )

    # --------------------------------------------------------
    # Analyze each date window
    # --------------------------------------------------------

    for date_window in DATE_WINDOWS:

        print("\n----------------------------------------------")
        print(f"DATE WINDOW: +/- {date_window} day(s)")
        print("----------------------------------------------")

        total_fields = len(agri)

        for radius in RADII_METERS:

            fields_with_firms = 0
            total_matches = 0

            for _, field in agri.iterrows():

                field_date = field["date"]

                start_date = (
                    field_date -
                    pd.Timedelta(days=date_window)
                )

                end_date = (
                    field_date +
                    pd.Timedelta(days=date_window)
                )

                # FIRMS records in date window
                firms_date = firms_local[
                    (firms_local["acq_date"] >= start_date) &
                    (firms_local["acq_date"] <= end_date)
                ]

                if len(firms_date) == 0:
                    continue

                # Coordinates of FIRMS records
                firms_date_coords = np.radians(
                    firms_date[
                        ["latitude", "longitude"]
                    ].values
                )

                date_tree = BallTree(
                    firms_date_coords,
                    metric="haversine"
                )

                field_coord = np.radians(
                    [[
                        field["latitude"],
                        field["longitude"]
                    ]]
                )

                radius_radians = radius / earth_radius_m

                indices = date_tree.query_radius(
                    field_coord,
                    r=radius_radians
                )[0]

                match_count = len(indices)

                if match_count > 0:
                    fields_with_firms += 1
                    total_matches += match_count

            percentage = (
                fields_with_firms / total_fields
            ) * 100

            print(
                f"Radius {radius:4d} m | "
                f"Fields matched: "
                f"{fields_with_firms}/{total_fields} "
                f"({percentage:.2f}%) | "
                f"Total FIRMS matches: {total_matches}"
            )


# ============================================================
# RUN ANALYSIS
# ============================================================

analyze_matching(
    agri_2020,
    2020
)

analyze_matching(
    agri_2021,
    2021
)


print("\n==============================================")
print("MATCHING DIAGNOSTIC COMPLETED")
print("==============================================")