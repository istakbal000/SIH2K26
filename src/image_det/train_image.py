"""Train the fire / no-fire image detection CNN.

Dataset layout (any fire/no-fire image folder is fine):
    data/images/train/{fire,nofire}/*.jpg
    data/images/val/{fire,nofire}/*.jpg    (optional)
    data/images/test/{fire,nofire}/*.jpg   (optional)

If val/ is missing it is created from the last 15% of train/ (stratified per class).

Usage:
    .venv\\Scripts\\python src\\image_det\\train_image.py --data data/images --epochs 20
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from image_det.model import build_cnn, save_checkpoint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("train_image")

IMG_SIZE = 224

CLASS_FOLDERS = ["fire", "nofire"]


def collect_samples(root: Path):
    """Return {name: [(path, label_idx)...]} using the two class folders present."""
    if not root.exists():
        raise FileNotFoundError(f"Missing image root: {root}")
    classes = [d for d in CLASS_FOLDERS if (root / d).is_dir()]
    if not classes:
        raise FileNotFoundError(f"No fire/nofire folders under {root}")
    samples = {}
    for label, name in enumerate(classes):
        files = sorted(p for p in (root / name).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if not files:
            logger.warning("  class '%s' has 0 images", name)
        samples[name] = [(str(p), label) for p in files]
    return samples, classes


def split_train(train_root: Path, samples: dict, classes, val_ratio: float = 0.15):
    """Move a stratified slice of train/ -> val/ if val is empty."""
    val_root = train_root.parent / "val"
    if val_root.exists() and any((val_root / c).exists() for c in classes):
        return
    import random
    random.seed(42)
    val_root.mkdir(parents=True, exist_ok=True)
    for name, items in samples.items():
        keep = items[:]
        random.shuffle(keep)
        n_val = max(1, int(len(keep) * val_ratio))
        moved = keep[:n_val]
        (val_root / name).mkdir(parents=True, exist_ok=True)
        import shutil
        for src, _ in moved:
            try:
                shutil.move(src, val_root / name / Path(src).name)
            except Exception:
                pass
    logger.info("  split train -> val created (%.0f%% per class)", val_ratio * 100)


class ImageFolder(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def main():
    ap = argparse.ArgumentParser(description="Train fire vs no-fire image detection model (binary)")
    ap.add_argument("--data", type=str, default="data/images", help="Root with train[/val/test]/{fire,nofire}")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    ap.add_argument("--out", type=str, default="models/image_det.pth")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--from-checkpoint", type=str, default=None, help="Resume weights (e.g. to fine-tune)")
    args = ap.parse_args()

    base = Path(args.data)
    train_root = base / "train"
    train_samples, classes = collect_samples(train_root)
    split_train(train_root, train_samples, classes)

    train_items = [it for v in train_samples.values() for it in v]
    val_items, test_items = [], []
    val_root, test_root = base / "val", base / "test"
    if val_root.exists():
        vs, _ = collect_samples(val_root)
        val_items = [it for v in vs.values() for it in v]
    if test_root.exists():
        ts, _ = collect_samples(test_root)
        test_items = [it for v in ts.values() for it in v]

    if not train_items:
        logger.error("No training images found. Put images in data/images/train/{fire,nofire}.")
        sys.exit(1)

    def count(items, name=""):
        n = sum(1 for _, l in items if l == 0), sum(1 for _, l in items if l == 1)
        return n

    logger.info("=" * 60)
    logger.info(" FIRE DETECTION (IMAGE) TRAINING")
    logger.info("=" * 60)
    logger.info(" classes: %s", classes)
    logger.info(" train:  %d images  %s", len(train_items), count(train_items))
    if val_items:
        logger.info(" val:    %d images  %s", len(val_items), count(val_items))
    if test_items:
        logger.info(" test:   %d images  %s", len(test_items), count(test_items))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(" device: %s", device)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((args.size, args.size), InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.size, args.size), InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        norm,
    ])

    train_loader = DataLoader(ImageFolder(train_items, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ImageFolder(val_items, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0) if val_items else None

    model = build_cnn(num_classes=len(classes))
    if args.from_checkpoint:
        ck = torch.load(args.from_checkpoint, map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        logger.info(" resumed from %s", args.from_checkpoint)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        acc = correct / max(total, 1)
        msg = f"[{epoch:02d}/{args.epochs}] loss={running/max(total,1):.4f}  train_acc={acc*100:.2f}%"

        if val_loader:
            model.eval()
            correct2 = total2 = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    correct2 += (out.argmax(1) == y).sum().item()
                    total2 += y.size(0)
            vacc = correct2 / max(total2, 1)
            msg += f"  val_acc={vacc*100:.2f}%"
            this = vacc
        else:
            this = acc

        logger.info(msg + "  (%.1fs)" % (time.time() - (getattr(main, "_t0", time.time()))))
        main._t0 = time.time()

        if this > best_acc:
            best_acc = this
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "class_names": classes,
        "extra": {"best_val_acc": round(float(best_acc), 4), "img_size": args.size, "classes": classes},
    }, out_path)
    logger.info("Saved model -> %s (best acc %.2f%%)", out_path, best_acc * 100)

    if test_items and (test_loader := DataLoader(ImageFolder(test_items, eval_tf), batch_size=args.batch_size)):
        model.load_state_dict(best_state)
        model.eval()
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        ys, preds = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds.extend(out.argmax(1).cpu().tolist())
                ys.extend(y.cpu().tolist())
        logger.info("\nTest set (%d images):", len(ys))
        logger.info("\n%s", classification_report(ys, preds, target_names=classes, zero_division=0))
        cm = confusion_matrix(ys, preds)
        logger.info("Confusion matrix (rows=true, cols=pred):\n%s", np.array2string(np.array(cm), precision=0))


if __name__ == "__main__":
    main()