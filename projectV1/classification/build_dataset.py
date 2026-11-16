#!/usr/bin/env python3
"""Baut den Klassifikations-Datensatz aus ``Legosteine_comp``.

19 Klassen, eine pro Unterordner. Jedes Bild wird - wenn eine YOLO-``.txt``
daneben liegt - mit der Box zugeschnitten (6 % Rand). Bilder ohne Box werden
als Ganzes verwendet und im Manifest als ``full`` markiert.

Aufteilung 70/20/10 pro Klasse. Ergebnis ist ein Ultralytics-Ordner:

    data/
      train/<teilenummer>/*.jpg
      val/<teilenummer>/*.jpg
      test/<teilenummer>/*.jpg
      classes.txt
      manifest.csv
      dataset_stats.json
      data.yaml

Beispiel:

    python projectV1/classification/build_dataset.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crop_box(img: Image.Image, box: tuple[float, float, float, float], margin: float) -> Image.Image:
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


def collect_samples(class_dir: Path, margin: float) -> list[dict]:
    """Ein Sample pro YOLO-Box; ohne Box das ganze Bild."""
    samples: list[dict] = []
    for image_path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXT):
        txt_path = image_path.with_suffix(".txt")
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        boxes: list[tuple[float, float, float, float]] = []
        if txt_path.is_file():
            for line in txt_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = (float(value) for value in parts[:5])
                boxes.append((
                    (cx - bw / 2) * img.width,
                    (cy - bh / 2) * img.height,
                    (cx + bw / 2) * img.width,
                    (cy + bh / 2) * img.height,
                ))

        if not boxes:
            samples.append({
                "name": f"full_{image_path.stem}",
                "image": img.copy(),
                "source": str(image_path),
                "kind": "full",
            })
        for index, box in enumerate(boxes):
            samples.append({
                "name": f"crop_{image_path.stem}_{index}",
                "image": crop_box(img, box, margin).copy(),
                "source": str(image_path),
                "kind": "crop",
            })
        img.close()
    return samples


def split_samples(
    samples: list[dict], val_frac: float, test_frac: float, rng: random.Random
) -> tuple[list[dict], list[dict], list[dict]]:
    """70/20/10 pro Klasse, mindestens ein Bild in val und test."""
    n = len(samples)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    if n == 1:
        return shuffled, [], []
    if n == 2:
        return shuffled[:1], shuffled[1:], []

    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    while n - n_val - n_test < 1 and (n_val or n_test):
        if n_test >= n_val and n_test:
            n_test -= 1
        else:
            n_val -= 1
    return (
        shuffled[n_test + n_val:],
        shuffled[n_test:n_test + n_val],
        shuffled[:n_test],
    )


def write_data_yaml(out: Path, classes: list[str]) -> None:
    """Doku/Klassenliste. Ultralytics erwartet bei Classification den Ordner."""
    lines = [
        f"path: {out}",
        "train: train",
        "val: val",
        "test: test",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines += [f"  {index}: '{name}'" for index, name in enumerate(classes)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="19-Klassen-Datensatz aus Legosteine_comp bauen.")
    ap.add_argument("--source", type=Path, default=REPO / "Legosteine_comp")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--crop-margin", type=float, default=0.06)
    ap.add_argument("--repeats", type=int, default=3, help="Oversampling der Trainingsbilder")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.source.is_dir():
        print(f"Quelle nicht gefunden: {args.source}")
        return 1
    if args.out.exists():
        if not args.force:
            print(f"Ziel existiert bereits: {args.out}\nMit --force neu bauen.")
            return 1
        shutil.rmtree(args.out)

    class_dirs = sorted(
        p for p in args.source.iterdir()
        if p.is_dir() and any(f.suffix.lower() in IMAGE_EXT for f in p.iterdir())
    )
    classes = [p.name for p in class_dirs]
    if not classes:
        print(f"Keine Klassenordner in {args.source}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.txt").write_text("\n".join(classes) + "\n")

    split_counts: Counter = Counter()
    kind_counts: Counter = Counter()
    manifest: list[dict] = []
    per_class: dict[str, dict[str, int]] = {}

    for class_index, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        samples = collect_samples(class_dir, args.crop_margin)
        if not samples:
            print(f"[WARN] {class_name}: keine Bilder")
            continue
        rng = random.Random(args.seed + class_index)
        train, val, test = split_samples(samples, args.val_frac, args.test_frac, rng)

        for split_name, split_samples_list in (("train", train), ("val", val), ("test", test)):
            count = args.repeats if split_name == "train" else 1
            for sample in split_samples_list:
                for repeat in range(count):
                    dst = args.out / split_name / class_name / f"{sample['name']}_{repeat}.jpg"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    sample["image"].save(dst, quality=95)
                    split_counts[split_name] += 1
                    kind_counts[sample["kind"]] += 1
                    manifest.append({
                        "split": split_name,
                        "class": class_name,
                        "kind": sample["kind"],
                        "source": sample["source"],
                    })
                sample["image"].close()

        per_class[class_name] = {"train": len(train) * args.repeats,
                                 "val": len(val), "test": len(test)}

    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "class", "kind", "source"])
        writer.writeheader()
        writer.writerows(manifest)

    missing_val = [c for c in classes if not (args.out / "val" / c).is_dir()]
    missing_test = [c for c in classes if not (args.out / "test" / c).is_dir()]
    stats = {
        "source": str(args.source),
        "classes": len(classes),
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "crop_margin": args.crop_margin,
        "repeats": args.repeats,
        "split_counts": dict(split_counts),
        "kind_counts": dict(kind_counts),
        "classes_without_val": missing_val,
        "classes_without_test": missing_test,
        "per_class": per_class,
    }
    (args.out / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )
    write_data_yaml(args.out, classes)

    print(f"Ziel:    {args.out}")
    print(f"Klassen: {len(classes)}: {', '.join(classes)}")
    print(f"Bilder:  {dict(split_counts)}  (crop/full: {dict(kind_counts)})")
    print(f"Train pro Klasse (mit {args.repeats}x): "
          f"min {min(v['train'] for v in per_class.values())}, "
          f"max {max(v['train'] for v in per_class.values())}")
    if missing_val or missing_test:
        print(f"[WARN] ohne val: {len(missing_val)}  ohne test: {len(missing_test)}")
    print(f"data.yaml: {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
