import rasterio
from pathlib import Path

# Project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Satellite images folder
SATELLITE_DIR = PROJECT_DIR / "data" / "sattelite_img" / "raw"

print("===================================")
print("Checking satellite files")
print("===================================")

print(f"\nProject directory:")
print(PROJECT_DIR)

print(f"\nSatellite directory:")
print(SATELLITE_DIR)

# Check whether the folder exists
if not SATELLITE_DIR.exists():
    print("\nERROR: Satellite folder does not exist!")
    print(f"Expected folder: {SATELLITE_DIR}")
    exit()

# Find all TIFF files
files = list(SATELLITE_DIR.glob("*.tif"))

print(f"\nFound {len(files)} TIFF files.\n")

if len(files) == 0:
    print("ERROR: No .tif files found!")
    exit()

# Inspect every TIFF file
for file in files:
    print("-----------------------------------")
    print(f"File: {file.name}")

    try:
        with rasterio.open(file) as src:
            print(f"Width: {src.width}")
            print(f"Height: {src.height}")
            print(f"CRS: {src.crs}")
            print(f"Resolution: {src.res}")
            print(f"Data type: {src.dtypes[0]}")
            print(f"Number of bands: {src.count}")
            print(f"Bounds: {src.bounds}")

    except Exception as e:
        print(f"ERROR reading file: {e}")

print("\n===================================")
print("Satellite file check complete")
print("===================================")