"""Run the trained fire image-detection model on an image or a folder of images."""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from image_det.model import load_checkpoint  # noqa: E402


def preprocess(img, size=224):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(img.convert("RGB")).unsqueeze(0)


def predict_image(model, class_names, img_path):
    img = Image.open(img_path)
    with torch.no_grad():
        logits = model(preprocess(img))
        proba = torch.softmax(logits, 1)[0]
    fire_idx = class_names.index("fire")
    p_fire = float(proba[fire_idx])
    pred = 1 if p_fire >= 0.5 else 0
    label = class_names[int(proba.argmax())]
    return {
        "fire_detected": bool(pred == 1),
        "prediction": pred,
        "confidence": round(p_fire if pred == 1 else 1 - p_fire, 6),
        "probability": {
            "no_fire": round(float(proba[1 - fire_idx]), 6),
            "fire": round(p_fire, 6),
        },
        "result": label,
        "image": str(img_path),
    }


def main():
    ap = argparse.ArgumentParser(description="Fire/no-fire prediction on image(s)")
    ap.add_argument("path", type=str, help="Image file or folder")
    ap.add_argument("--model", type=str, default="models/image_det.pth")
    ap.add_argument("--json", action="store_true", help="Output JSON only")
    args = ap.parse_args()

    model, class_names, _ = load_checkpoint(Path(args.model))
    p = Path(args.path)
    files = sorted(p.rglob("*")) if p.is_dir() else [p]
    files = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    results = [predict_image(model, class_names, f) for f in files]
    n_fire = sum(1 for r in results if r["fire_detected"])

    if args.json:
        print(json.dumps({"count": len(results), "fire": n_fire, "no_fire": len(results) - n_fire, "predictions": results}, indent=2))
        return

    for r in results:
        bar = "🔥 FIRE" if r["fire_detected"] else "🌿 no fire"
        print(f"{bar}  {r['confidence']*100:5.1f}%  {r['image']}")
    print(f"\n{n_fire}/{len(results)} images detected as fire")


if __name__ == "__main__":
    main()