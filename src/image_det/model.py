"""Fire / no-fire image DETECTION model (binary)."""
from pathlib import Path

import torch
import torch.nn as nn


def build_cnn(num_classes: int = 2) -> nn.Module:
    """Small CNN for fire vs no-fire image detection (224x224 RGB input)."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(inplace=True),
        nn.Linear(128, num_classes),
    )


def save_checkpoint(model: nn.Module, path: Path, class_names, extra: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "extra": extra or {},
    }, path)


def load_checkpoint(path: Path, num_classes: int = 2) -> tuple:
    """Returns (model, class_names, extra) with weights loaded."""
    ck = torch.load(path, map_location="cpu")
    model = build_cnn(num_classes=len(ck.get("class_names", ["fire", "nofire"])))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("class_names", ["fire", "nofire"]), ck.get("extra", {})