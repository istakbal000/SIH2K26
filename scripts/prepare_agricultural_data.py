import geopandas as gpd
import pandas as pd
from pathlib import Path


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent

AGRICULTURE_DIR = PROJECT_DIR / "data" / "Agriculture"
OUTPUT_DIR = AGRICULTURE_DIR / "cleaned"

# Create output folder if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Input files
file_2020 = AGRICULTURE_DIR / "partially_completely_burnt_2020_11_10.geojson"
file_2021 = AGRICULTURE_DIR / "partially_completely_burnt_2021_10_29.geojson"


def prepare_file(input_file, output_file, year, date):
    print(f"\nReading: {input_file.name}")

    # Read GeoJSON
    gdf = gpd.read_file(input_file)

    print(f"Number of fields: {len(gdf)}")
    print(f"Columns: {list(gdf.columns)}")

    # Create a representative point inside each field polygon
    points = gdf.geometry.representative_point()

    # Extract longitude and latitude
    gdf["longitude"] = points.x
    gdf["latitude"] = points.y

    # Add year and date
    gdf["year"] = year
    gdf["date"] = date

    # Convert field category into binary agricultural-fire label
    gdf["agricultural_fire"] = gdf["field_category"].map({
    "completely_burnt": 1,
    "partially_burnt": 1,
    "golden_yellow_unburnt": 0,
    "green_unburnt": 0,
    "unburnt": 0
})

    # Keep only the useful columns
    result = gdf[
        [
            "field_id",
            "field_name",
            "field_category",
            "latitude",
            "longitude",
            "year",
            "date",
            "agricultural_fire"
        ]
    ].copy()

    # Check for categories that were not recognized
    if result["agricultural_fire"].isna().any():
        print("\nWARNING: Unknown field categories found:")
        print(result.loc[
            result["agricultural_fire"].isna(),
            "field_category"
        ].unique())

    # Save CSV
    result.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Rows saved: {len(result)}")

    print("\nCategory distribution:")
    print(result["field_category"].value_counts())

    print("\nAgricultural-fire label distribution:")
    print(result["agricultural_fire"].value_counts())


# Prepare 2020 data
prepare_file(
    file_2020,
    OUTPUT_DIR / "agriculture_2020.csv",
    2020,
    "2020-11-10"
)


# Prepare 2021 data
prepare_file(
    file_2021,
    OUTPUT_DIR / "agriculture_2021.csv",
    2021,
    "2021-10-29"
)


print("\n===================================")
print("Agricultural data preparation done!")
print("===================================")