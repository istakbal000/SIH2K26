"""Build real, stratified image datasets from the raw labels.

Sources (from data/raw, pushed by the team):
    forest_fire    -> class  forest  (and detection class  fire)
    industrial_fire-> class  industrial (and detection class fire)
    nofire         -> class  nofire

Outputs:
    data/images/train|val|test/{fire,nofire}          (binary detection)
    data/images_type/train|val|test/{forest,industrial,nofire}  (3-class fire type)

A deterministic stratified 80/10/10 split is used; images are copied so the raw
files stay untouched for the tabular pipelines.
"""
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCES = {
    "forest": Path("data/raw/forest_fire"),
    "industrial": Path("data/raw/industrial_fire"),
    "nofire": Path("data/raw/nofire"),
}

DET_ROOT = Path("data/images")
TYPE_ROOT = Path("data/images_type")


def collect(root: Path):
    """All readable images under root (recursive, sorted)."""
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


def split_stratified(items, ratios=(0.8, 0.1, 0.1), seed=42):
    rng = random.Random(seed)
    pool = items[:]
    rng.shuffle(pool)
    n = len(pool)
    n1 = int(n * ratios[0])
    n2 = int(n * ratios[1])
    return pool[:n1], pool[n1:n1 + n2], pool[n1 + n2:]


def write_split(dest_root, class_name, train, val, test):
    for sub, items in (("train", train), ("val", val), ("test", test)):
        d = dest_root / sub / class_name
        d.mkdir(parents=True, exist_ok=True)
        for src in items:
            dst = d / src.name
            if not dst.exists():
                shutil.copy2(src, dst)


def main():
    det = {c: collect(p) for c, p in SOURCES.items()}
    for c, items in det.items():
        print(f"  {c:10s} {len(items):4d} images")
    if any(not items for items in det.values()):
        print("ERROR: one or more raw class folders are empty/missing.")
        sys.exit(1)

    # ---- Binary detection: fire = forest + industrial ----
    fire_items = det["forest"] + det["industrial"]
    nofire_items = det["nofire"]
    ftr, fva, fte = split_stratified(fire_items)
    ntr, nva, nte = split_stratified(nofire_items)
    shutil.rmtree(DET_ROOT, ignore_errors=True)
    write_split(DET_ROOT, "fire", ftr, fva, fte)
    write_split(DET_ROOT, "nofire", ntr, nva, nte)
    print(f"\nDetection  -> {DET_ROOT}")
    print(f"  train fire={len(ftr)} nofire={len(ntr)} | val fire={len(fva)} nofire={len(nva)} | test fire={len(fte)} nofire={len(nte)}")

    # ---- 3-class fire type ----
    splits = {c: split_stratified(v) for c, v in det.items()}
    shutil.rmtree(TYPE_ROOT, ignore_errors=True)
    for c, (tr, va, te) in splits.items():
        write_split(TYPE_ROOT, c, tr, va, te)
        print(f"Type {c:10s}: train={len(tr)} val={len(va)} test={len(te)}")
    print(f"Type model root -> {TYPE_ROOT}")


if __name__ == "__main__":
    main()