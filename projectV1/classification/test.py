#!/usr/bin/env python3
"""Klassifikations-Modell testen - Gegenstueck zu
``projectV1/backend/object_detection/test.py``.

Zwei Modi:

    # Split auswerten (Top-1/Top-5)
    python projectV1/classification/test.py --split test

    # Einzelbild oder Ordner klassifizieren + beschriftete Kopien speichern
    python projectV1/classification/test.py --source multi_lego_yolo/20261111_202954.jpg
    python projectV1/classification/test.py --source Legosteine_comp/3002
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def font(size: int = 18):
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description="Klassifikator testen.")
    ap.add_argument("--weights", type=Path,
                    default=HERE / "runs/yolo26m_cls_70_20_10/weights/best.pt")
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--source", type=Path, default=None,
                    help="Bild oder Ordner; ohne Angabe wird der Split ausgewertet")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, default=HERE / "runs/predict")
    args = ap.parse_args()

    model = YOLO(str(args.weights))

    if args.source is None:
        metrics = model.val(data=str(args.data.resolve()), split=args.split,
                            imgsz=args.imgsz, device=args.device)
        print(f"Split {args.split}: Top-1 {metrics.top1:.3f}  Top-5 {metrics.top5:.3f}")
        return 0

    source = args.source.resolve()
    images = [source] if source.is_file() else sorted(
        p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if not images:
        print(f"Keine Bilder in {source}")
        return 1

    results = model.predict([str(p) for p in images], imgsz=args.imgsz,
                            device=args.device, verbose=False)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    label_font = font(18)
    for image_path, result in zip(images, results):
        data = result.probs.data.cpu().numpy()
        order = np.argsort(data)[::-1][:5]
        names = [model.names[int(i)] for i in order]
        rows.append({
            "image": str(image_path),
            "prediction": names[0],
            "confidence": f"{float(data[order[0]]):.4f}",
            "top5": " ".join(names),
        })

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            text = f"{names[0]} {float(data[order[0]]):.2f}"
            box = draw.textbbox((0, 0), text, font=label_font)
            draw.rectangle((0, 0, box[2] + 8, box[3] + 6), fill=(0, 0, 0))
            draw.text((4, 2), text, fill=(255, 255, 255), font=label_font)
            img.save(args.out / image_path.name, quality=92)

    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'Bild':<40} {'Vorhersage':>10}  Konf.")
    print("-" * 62)
    for row in rows:
        print(f"{Path(row['image']).name:<40} {row['prediction']:>10}  {row['confidence']}")
    print(f"\nBilder:       {len(rows)}")
    print(f"CSV:          {args.out / 'predictions.csv'}")
    print(f"Beschriftet:  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
