#!/usr/bin/env python3
"""Klassifiziert zufaellige Stichproben aus einem Ordner-als-Klasse-Datensatz.

Pro Klassenordner werden ``--per-class`` Bilder zufaellig gewaehlt (fester
Seed, reproduzierbar) und mit einem YOLO-Klassifikator vorhergesagt. Ausgabe:
Tabelle auf der Konsole, ``samples.csv`` und optional ein Kontaktabzug.

Beispiel:

    python projectV1/backend/eval/test_random_samples.py \
        --images Legosteine_comp \
        --weights projectV1/runs/cls_yolo26m_clean/weights/best.pt \
        --per-class 1 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CELL = 180
HEADER = 34


def load_font(size: int = 14):
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(path: Path, box: int) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((box, box), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (box, box), "white")
        canvas.paste(img, ((box - img.width) // 2, (box - img.height) // 2))
        return canvas


def crop_from_txt(image: Path, margin: float) -> tuple[Image.Image, bool]:
    """Schneidet die erste YOLO-Box aus dem Bild (wie beim Trainingsdatensatz)."""
    txt = image.with_suffix(".txt")
    if txt.is_file():
        with Image.open(image) as img:
            img = img.convert("RGB")
            for line in txt.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = (float(v) for v in parts[:5])
                x1, y1 = (cx - bw / 2) * img.width, (cy - bh / 2) * img.height
                x2, y2 = (cx + bw / 2) * img.width, (cy + bh / 2) * img.height
                mx, my = (x2 - x1) * margin, (y2 - y1) * margin
                left, top = max(0, int(x1 - mx)), max(0, int(y1 - my))
                right, bottom = min(img.width, int(x2 + mx)), min(img.height, int(y2 + my))
                if right - left >= 8 and bottom - top >= 8:
                    return img.crop((left, top, right, bottom)), True
    with Image.open(image) as img:
        return img.convert("RGB"), False


def make_sheet(rows: list[dict], out: Path) -> None:
    cols = 5
    lines = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * CELL, lines * (CELL + HEADER)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(14)
    for index, row in enumerate(rows):
        x = (index % cols) * CELL
        y = (index // cols) * (CELL + HEADER)
        mark = "OK" if row["correct"] else ("TOP5" if row["in_top5"] else "FALSCH")
        color = "#1a7f37" if row["correct"] else ("#9a6700" if row["in_top5"] else "#b42318")
        draw.text((x + 3, y + 3), f"{row['class']} -> {row['pred']}", fill="black", font=font)
        draw.text((x + 3, y + 17), f"{mark}  {row['confidence']:.2f}", fill=color, font=font)
        sheet.paste(fit(Path(row.get("input", row["image"])), CELL), (x, y + HEADER))
    sheet.save(out, quality=90)


def main() -> int:
    ap = argparse.ArgumentParser(description="Zufallsstichprobe pro Klasse klassifizieren.")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, default=None,
                    help="Zielordner fuer samples.csv/sheet.jpg (Default: neben den Gewichten)")
    ap.add_argument("--crop-from-txt", action="store_true",
                    help="Teil mit der YOLO-Box aus der .txt schneiden (wie im Training)")
    ap.add_argument("--crop-margin", type=float, default=0.06,
                    help="Rand um die Box beim Crop (Default wie build_fast_cls.py)")
    args = ap.parse_args()

    out = args.out or (args.weights.parent.parent / "samples_legosteine_comp")
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    rng = random.Random(args.seed)
    samples: list[tuple[str, Path]] = []
    for class_dir in sorted(p for p in args.images.iterdir() if p.is_dir()):
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not images:
            continue
        for image in rng.sample(images, min(args.per_class, len(images))):
            samples.append((class_dir.name, image))

    inputs: list[Path] = []
    cropped_flags: list[bool] = []
    for true_class, image in samples:
        if args.crop_from_txt:
            crop, used_box = crop_from_txt(image, args.crop_margin)
            target = out / "inputs" / f"{true_class}_{image.stem}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target, quality=95)
            inputs.append(target)
            cropped_flags.append(used_box)
        else:
            inputs.append(image)
            cropped_flags.append(False)

    predictions = model.predict(
        [str(path) for path in inputs], imgsz=args.imgsz, device=args.device,
        verbose=False,
    )

    rows: list[dict] = []
    for (true_class, image), input_path, is_crop, result in zip(
            samples, inputs, cropped_flags, predictions):
        data = result.probs.data.cpu().numpy()
        order = np.argsort(data)[::-1][:5]
        pred_class = model.names[int(order[0])]
        top5 = [model.names[int(i)] for i in order]
        rows.append({
            "class": true_class,
            "image": str(image),
            "input": str(input_path),
            "cropped": is_crop,
            "pred": pred_class,
            "confidence": float(data[order[0]]),
            "top5": ",".join(top5),
            "correct": pred_class == true_class,
            "in_top5": true_class in top5,
        })

    with (out / "samples.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_sheet(rows, out / "sheet.jpg")

    correct = sum(row["correct"] for row in rows)
    in_top5 = sum(row["in_top5"] for row in rows)
    cropped = sum(row["cropped"] for row in rows)
    print(f"{'Klasse':>8} | {'Bild':<24} | {'Vorhersage':>10} | Konf. | Top-1 | Box")
    print("-" * 76)
    for row in rows:
        mark = "OK" if row["correct"] else ("top5" if row["in_top5"] else "FALSCH")
        box = "crop" if row["cropped"] else "full"
        print(f"{row['class']:>8} | {Path(row['image']).name:<24} | "
              f"{row['pred']:>10} | {row['confidence']:>5.2f} | {mark:<6} | {box}")
    print("-" * 76)
    print(f"Top-1: {correct}/{len(rows)} = {100 * correct / len(rows):.1f} %")
    print(f"Top-5: {in_top5}/{len(rows)} = {100 * in_top5 / len(rows):.1f} %")
    print(f"Gecroppt: {cropped}/{len(rows)} (Rest ohne Box -> Ganzbild)")
    print(f"CSV:    {out / 'samples.csv'}")
    print(f"Sheet:  {out / 'sheet.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
