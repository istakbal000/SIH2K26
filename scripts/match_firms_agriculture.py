import pandas as pd
import geopandas as gpd
from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

FIRMS_FILE = (
    PROJECT_DIR
    / "data"
    / "firm_data"
    / "DL_FIRE_J1V-C2_802955"
    / "fire_archive_J1V-C2_802955.csv"
)

AGRICULTURE_DIR = PROJECT_DIR / "data" / "Agriculture"

OUTPUT_DIR = AGRICULTURE_DIR / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# ZENODO FILES
# ==========================================

FILE_2020 = (
    AGRICULTURE_DIR
    / "partially_completely_burnt_2020_11_10.geojson"
)

FILE_2021 = (
    AGRICULTURE_DIR
    / "partially_completely_burnt_2021_10_29.geojson"
)


# ==========================================
# READ FIRMS
# ==========================================

print("\nReading FIRMS data...")

firms = pd.read_csv(FIRMS_FILE)

print(f"Total FIRMS records: {len(firms)}")
print(f"FIRMS columns: {list(firms.columns)}")


# ==========================================
# MATCHING FUNCTION
# ==========================================

def analyze_spatial_distance(
    firms_data,
    agriculture_file,
    target_date,
    output_file
):

    print("\n========================================")
    print(f"Processing date: {target_date}")
    print("========================================")

    # --------------------------------------
    # Read Zenodo agricultural polygons
    # --------------------------------------

    agriculture = gpd.read_file(
        agriculture_file
    )

    print(
        f"Agricultural field polygons: "
        f"{len(agriculture)}"
    )

    print(
        f"Agriculture CRS: "
        f"{agriculture.crs}"
    )

    # --------------------------------------
    # Select FIRMS date
    # --------------------------------------

    firms_date = firms_data[
        firms_data["acq_date"] == target_date
    ].copy()

    print(
        f"FIRMS records on {target_date}: "
        f"{len(firms_date)}"
    )

    # --------------------------------------
    # Create FIRMS points
    # --------------------------------------

    firms_gdf = gpd.GeoDataFrame(
        firms_date,
        geometry=gpd.points_from_xy(
            firms_date["longitude"],
            firms_date["latitude"]
        ),
        crs="EPSG:4326"
    )

    # --------------------------------------
    # Convert both datasets to UTM Zone 43N
    # --------------------------------------

    firms_projected = firms_gdf.to_crs(
        "EPSG:32643"
    )

    agriculture_projected = agriculture.to_crs(
        "EPSG:32643"
    )

    # --------------------------------------
    # Find nearest agricultural field
    # WITHOUT a distance limit
    # --------------------------------------

    print(
        "\nFinding nearest agricultural field "
        "for every FIRMS hotspot..."
    )

    matched = gpd.sjoin_nearest(
        firms_projected,
        agriculture_projected[
            [
                "field_id",
                "field_name",
                "field_category",
                "geometry"
            ]
        ],
        how="left",
        distance_col="distance_to_field_m"
    )

    # --------------------------------------
    # Distance statistics
    # --------------------------------------

    distances = matched[
        "distance_to_field_m"
    ].dropna()

    print("\nDistance statistics:")
    print(distances.describe())

    print("\n========================================")
    print("DISTANCE THRESHOLD ANALYSIS")
    print("========================================")

    thresholds = [
        100,
        250,
        500,
        1000,
        2000,
        5000
    ]

    for threshold in thresholds:

        count = (
            distances <= threshold
        ).sum()

        print(
            f"Within {threshold:>4} metres: "
            f"{count}"
        )

    # --------------------------------------
    # Show closest 20 FIRMS points
    # --------------------------------------

    print("\n========================================")
    print("20 CLOSEST FIRMS POINTS")
    print("========================================")

    closest = matched.sort_values(
        "distance_to_field_m"
    ).head(20)

    print(
        closest[
            [
                "latitude",
                "longitude",
                "distance_to_field_m",
                "field_id",
                "field_category"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------
    # For now use 2 km as an analysis range
    # ONLY so we can inspect possible matches.
    # We are NOT assigning final labels yet.
    # --------------------------------------

    analysis_range = matched[
        matched["distance_to_field_m"] <= 2000
    ].copy()

    print(
        f"\nFIRMS points within 2 km: "
        f"{len(analysis_range)}"
    )

    print("\nField categories within 2 km:")

    if len(analysis_range) > 0:

        print(
            analysis_range[
                "field_category"
            ].value_counts()
        )

    else:

        print("No points within 2 km.")

    # --------------------------------------
    # Create label ONLY for the 2 km
    # analysis subset.
    # This is temporary.
    # --------------------------------------

    analysis_range[
        "agricultural_fire"
    ] = analysis_range[
        "field_category"
    ].map({
        "completely_burnt": 1,
        "partially_burnt": 1,
        "golden_yellow_unburnt": 0,
        "green_unburnt": 0,
        "unburnt": 0
    })

    # --------------------------------------
    # Save analysis dataset
    # --------------------------------------

    output_columns = [
        "latitude",
        "longitude",
        "brightness",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "confidence",
        "version",
        "bright_t31",
        "frp",
        "daynight",
        "type",
        "field_id",
        "field_name",
        "field_category",
        "distance_to_field_m",
        "agricultural_fire"
    ]

    result = analysis_range[
        output_columns
    ].copy()

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved analysis file: "
        f"{output_file}"
    )

    print(
        f"Rows saved: {len(result)}"
    )


# ==========================================
# PROCESS 2020
# ==========================================

analyze_spatial_distance(
    firms,
    FILE_2020,
    "2020-11-10",
    OUTPUT_DIR / "firms_agriculture_2020.csv"
)


# ==========================================
# PROCESS 2021
# ==========================================

analyze_spatial_distance(
    firms,
    FILE_2021,
    "2021-10-29",
    OUTPUT_DIR / "firms_agriculture_2021.csv"
)


# ==========================================
# DONE
# ==========================================

print("\n========================================")
print("Spatial distance analysis completed!")
print("========================================")