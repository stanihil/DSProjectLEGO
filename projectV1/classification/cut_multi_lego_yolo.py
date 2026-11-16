#!/usr/bin/env python3
"""Schneidet die Handyfotos aus ``multi_lego_yolo`` an den Boxen aus.

Die YOLO-``.txt`` dort hat nur Klasse 0 (``0 cx cy w h``), also keine
Teilenummer. Die steht in ``projectV1/dataset/generic/<name>.json``
(X-AnyLabeling, handgezeichnet). Beide Boxen sind praktisch identisch
(Median-IoU 1,00), geschnitten wird an der JSON-Box.

Ergebnis:

    cuttet_multi_lego_yolo/
      3001/20261111_202954_03.jpg
      3004/20261111_202954_07.jpg
      ...
      manifest.csv     # Crop -> Teilenummer, Quelle, Box, IoU zur YOLO-Box
      classes.txt      # Klassen = Ordnernamen

Bilder ohne brauchbare Teilenummer (``lego_brick``, ``30062``) werden
standardmäßig übersprungen, Boxen ohne JSON-Partner ebenfalls.

    python projectV1/classification/cut_multi_lego_yolo.py
    python projectV1/classification/cut_multi_lego_yolo.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Labels ohne Teilenummer. Sie sind als Fehlklassen dokumentiert (README,
# alter Handyfoto-Test) und taugen nicht als Klassenordner.
JUNK_LABELS = {"lego_brick", "30062"}


def rect_xyxy(shape: dict) -> tuple[float, float, float, float]:
    """Bounding-Box über alle Punkte; bei 4-Punkt-Rechtecken sind die
    gespeicherten Ecken nicht in Lese-Reihenfolge."""
    xs = [p[0] for p in shape["points"]]
    ys = [p[1] for p in shape["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop_with_margin(img: Image.Image, box: tuple, margin: float) -> Image.Image:
    x1, y1, x2, y2 = box
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    left = max(0, int(x1 - mx))
    top = max(0, int(y1 - my))
    right = min(img.width, int(x2 + mx))
    bottom = min(img.height, int(y2 + my))
    if right - left < 8 or bottom - top < 8:
        return img
    return img.crop((left, top, right, bottom))


def read_yolo_boxes(txt_path: Path, width: int, height: int) -> list[tuple]:
    boxes = []
    if not txt_path.is_file():
        return boxes
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in parts[:5])
        boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height))
    return boxes


def match(boxes_a: list[tuple], boxes_b: list[tuple], min_iou: float) -> list[tuple]:
    """Greedy-Matching nach IoU: [(idx_a, idx_b, iou), ...]."""
    pairs = sorted(((iou(a, b), ia, ib)
                    for ia, a in enumerate(boxes_a)
                    for ib, b in enumerate(boxes_b)), reverse=True)
    used_a, used_b, out = set(), set(), []
    for value, ia, ib in pairs:
        if value < min_iou:
            break
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        out.append((ia, ib, value))
    return out


def class_key(name: str):
    return (0, int(name)) if name.isdigit() else (1, name)


def main() -> int:
    ap = argparse.ArgumentParser(description="multi_lego_yolo an den Boxen ausschneiden.")
    ap.add_argument("--images", type=Path, default=REPO / "multi_lego_yolo")
    ap.add_argument("--labels", type=Path, default=REPO / "projectV1/dataset/generic")
    ap.add_argument("--out", type=Path, default=REPO / "cuttet_multi_lego_yolo")
    ap.add_argument("--margin", type=float, default=0.06, help="Rand um die Box")
    ap.add_argument("--min-iou", type=float, default=0.3,
                    help="Ab diesem IoU gilt eine YOLO-Box als zur JSON-Box gehoerig")
    ap.add_argument("--keep-junk", action="store_true",
                    help=f"auch {sorted(JUNK_LABELS)} als Ordner anlegen")
    ap.add_argument("--dry-run", action="store_true", help="nur zaehlen, nichts schreiben")
    ap.add_argument("--force", action="store_true", help="vorhandenen Ordner ersetzen")
    args = ap.parse_args()

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not images:
        print(f"Keine Bilder in {args.images}")
        return 1
    if args.out.exists() and not args.dry_run:
        if not args.force:
            print(f"Ziel existiert bereits: {args.out}\nMit --force neu schneiden.")
            return 1
        shutil.rmtree(args.out)

    counts: Counter = Counter()
    manifest: list[dict] = []
    missing_json: list[str] = []
    junk: list[tuple[str, str]] = []
    unmatched_yolo: list[tuple[str, int]] = []

    for image_path in images:
        label_path = args.labels / f"{image_path.stem}.json"
        if not label_path.is_file():
            missing_json.append(image_path.name)
            continue

        shapes = json.loads(label_path.read_text())["shapes"]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            json_boxes = [rect_xyxy(s) for s in shapes]
            yolo_boxes = read_yolo_boxes(image_path.with_suffix(".txt"), img.width, img.height)
            pairs = {ia: (ib, value) for ia, ib, value in match(json_boxes, yolo_boxes, args.min_iou)}
            used_yolo = {ib for ib, _ in pairs.values()}
            unmatched_yolo += [(image_path.name, i) for i in range(len(yolo_boxes)) if i not in used_yolo]

            for index, shape in enumerate(shapes):
                label = shape["label"]
                if label in JUNK_LABELS and not args.keep_junk:
                    junk.append((image_path.name, label))
                    continue

                crop = crop_with_margin(img, json_boxes[index], args.margin)
                yolo_index, value = pairs.get(index, ("", ""))
                name = f"{image_path.stem}_{index:02d}.jpg"
                counts[label] += 1
                manifest.append({
                    "file": f"{label}/{name}",
                    "class": label,
                    "source_image": image_path.name,
                    "json_box": index,
                    "yolo_box": yolo_index,
                    "iou": f"{value:.3f}" if value != "" else "",
                    "box": " ".join(f"{v:.1f}" for v in json_boxes[index]),
                })
                if not args.dry_run:
                    target = args.out / label / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(target, quality=95)
                crop.close()

    classes = sorted(counts, key=class_key)
    total = sum(counts.values())

    print(f"Bilder gelesen:   {len(images)}")
    print(f"Crops:            {total} in {len(classes)} Klassen")
    for name in classes:
        print(f"  {name:>8}  {counts[name]:>4}")
    if junk:
        print(f"Übersprungen (kein Teilenummer-Label): {len(junk)}  "
              f"{dict(Counter(l for _, l in junk))}")
    if unmatched_yolo:
        print(f"YOLO-Boxen ohne JSON-Partner: {len(unmatched_yolo)} (übersprungen)")
    if missing_json:
        print(f"[WARN] ohne JSON: {missing_json}")

    if args.dry_run:
        print("\n[dry-run] nichts geschrieben")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.txt").write_text("\n".join(classes) + "\n")
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nZiel:     {args.out}")
    print(f"Manifest: {args.out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
