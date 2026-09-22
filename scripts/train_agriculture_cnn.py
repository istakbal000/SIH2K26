import os
from pathlib import Path

import numpy as np
import rasterio
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_DIR / "data" / "sattelite_img" / "raw"
MODEL_DIR = PROJECT_DIR / "data" / "models"
RESULT_DIR = PROJECT_DIR / "data" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SATELLITE BAND FILES
# ============================================================

BAND_FILES = {
    "B01": IMAGE_DIR / "HLS.S30.T43SCS.2021282T053731.v2.0.B01.tif",
    "B03": IMAGE_DIR / "HLS.S30.T43SCS.2021282T053731.v2.0.B03.tif",
    "B04": IMAGE_DIR / "HLS.S30.T43SCS.2021282T053731.v2.0.B04.tif",
    "B05": IMAGE_DIR / "HLS.S30.T43SCS.2021282T053731.v2.0.B05.tif",
    "B10": IMAGE_DIR / "HLS.S30.T43SCS.2021282T053731.v2.0.B10.tif",
    "B11": IMAGE_DIR / "HLS.S30.T43SCS.2021282T053731.v2.0.B11.tif",
    "B12": IMAGE_DIR / "HLS.S30.T43SCS.2021282T053731.v2.0.B12.tif",
}


# ============================================================
# 3. CNN SETTINGS
# ============================================================

PATCH_SIZE = 64
STRIDE = 64

EPOCHS = 15
BATCH_SIZE = 16

RANDOM_STATE = 42


# ============================================================
# 4. LOAD AND STACK SATELLITE BANDS
# ============================================================

def load_satellite_image():
    print("\nLoading satellite bands...")

    image_bands = []

    for band_name, file_path in BAND_FILES.items():

        print(f"Reading {band_name}: {file_path.name}")

        if not file_path.exists():
            raise FileNotFoundError(
                f"Band file not found:\n{file_path}"
            )

        with rasterio.open(file_path) as src:
            band = src.read(1).astype(np.float32)

        image_bands.append(band)

        print(
            f"{band_name}: "
            f"shape={band.shape}, "
            f"min={np.nanmin(band):.3f}, "
            f"max={np.nanmax(band):.3f}"
        )

    stacked_image = np.stack(image_bands, axis=-1)

    print("\nStacked image shape:")
    print(stacked_image.shape)

    return stacked_image


# ============================================================
# 5. CLEAN AND NORMALIZE IMAGE
# ============================================================

def normalize_image(image):

    print("\nNormalizing satellite image...")

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    normalized = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[-1]):

        band = image[:, :, channel]

        low = np.percentile(band, 2)
        high = np.percentile(band, 98)

        if high > low:
            band = np.clip(band, low, high)
            band = (band - low) / (high - low)

        normalized[:, :, channel] = band

    return normalized


# ============================================================
# 6. CREATE IMAGE PATCHES
# ============================================================

def create_patches(image):

    print("\nCreating image patches...")

    patches = []

    height, width, channels = image.shape

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):

        for x in range(0, width - PATCH_SIZE + 1, STRIDE):

            patch = image[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
                :
            ]

            patches.append(patch)

    patches = np.array(patches, dtype=np.float32)

    print("Total patches:", len(patches))
    print("Patch shape:", patches.shape)

    return patches


# ============================================================
# 7. CREATE PROTOTYPE LABELS
# ============================================================

def create_labels(number_of_patches):
    """
    Temporary prototype labels.

    NOTE:
    These labels are placeholders because the available satellite
    scene does not have direct image-level ground-truth labels.
    They are NOT used for the final scientific evaluation.
    """

    labels = np.zeros(number_of_patches, dtype=np.int32)

    # Create a balanced prototype dataset so that the CNN
    # training pipeline can be executed with the available scene.
    half = number_of_patches // 2
    labels[half:] = 1

    print("\nPrototype label distribution:")
    print("Class 0 (Non-Agricultural):", np.sum(labels == 0))
    print("Class 1 (Agricultural):", np.sum(labels == 1))

    return labels


# ============================================================
# 8. BUILD CNN
# ============================================================

def build_cnn(input_shape):

    model = models.Sequential([

        layers.Input(shape=input_shape),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.4),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" AGRICULTURAL FIRE CNN")
    print("==========================================")

    # Load satellite bands
    image = load_satellite_image()

    # Normalize
    image = normalize_image(image)

    # Create patches
    X = create_patches(image)

    # Create labels
    y = create_labels(len(X))

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_cnn(X_train.shape[1:])

    print("\nCNN architecture:")
    model.summary()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n==========================================")
    print(" STARTING CNN TRAINING")
    print("==========================================")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n==========================================")
    print(" TESTING CNN")
    print("==========================================")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=0
    ).flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Non-Agricultural",
                "Agricultural"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR /
        "agricultural_fire_cnn.h5"
    )

    model.save(model_path)

    print("\nCNN model saved to:")
    print(model_path)

    print("\n==========================================")
    print(" CNN TRAINING AND TESTING COMPLETE")
    print("==========================================")


if __name__ == "__main__":
    main()