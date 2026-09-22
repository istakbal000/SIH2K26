"""Generate synthetic fire/nofire images to smoke-test the image pipeline.

Creates data/images/{train,val,test}/{fire,nofire}/*.jpg placeholders.
Replace with your real dataset before training for real.
"""
import argparse
import sys
import random
from pathlib import Path

import numpy as np
from PIL import Image

random.seed(7)


def fire_image(size=(96, 96)) -> np.ndarray:
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            base = np.clip(int(80 + 60 * np.sin(x / 8) + 30 * np.exp(-(((x - w * 0.65) ** 2) + (y - h * 0.75) ** 2) / 600)), 0, 255)
            r = int(np.clip(200 + base + random.randint(-30, 30), 0, 255))
            g = int(np.clip(40 + random.randint(-20, 40), 0, 255))
            b = int(np.clip(10 + random.randint(0, 20), 0, 100))
            img[y, x] = (r, g, b)
    return img


def nofire_image(size=(96, 96)) -> np.ndarray:
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            r = int(np.clip(30 + 20 * random.random() + 10 * np.sin(y / 12), 0, 255))
            g = int(np.clip(90 + 50 * random.random(), 0, 255))
            b = int(np.clip(40 + 30 * random.random(), 0, 255))
            img[y, x] = (r, g, b)
    return img


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic smoke-test fire detection images")
    ap.add_argument("--root", type=str, default="data/images")
    ap.add_argument("--per-split", type=int, default=40, help="images per class per split")
    args = ap.parse_args()

    root = Path(args.root)
    for split, kind in [("train", fire_image), ("train", nofire_image), ("val", fire_image), ("val", nofire_image),
                        ("test", fire_image), ("test", nofire_image)]:
        name = "fire" if kind is fire_image else "nofire"
        d = root / split / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_split):
            Image.fromarray(kind()).save(d / f"syn_{i:03d}.jpg")
    print(f"Synthetic images written to {root}")


if __name__ == "__main__":
    main()