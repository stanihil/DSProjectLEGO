#!/usr/bin/env python3
"""Findet Ausreisser innerhalb jeder Klasse eines Ordner-als-Klasse-Datensatzes.

Idee: Fuer jede Klasse wird ein Medoid (das "typischste" Bild) per dHash
bestimmt. Bilder mit grossem Abstand zum Medoid sind Kandidaten fuer falsche
Labels (oder starke Ausreisser bei Farbe/Aufnahme). Ausgegeben werden:

* ``outliers.csv``       – Bild, Klasse, Abstand zum Medoid
* ``class_stats.csv``    – pro Klasse: Anzahl, Median-/Max-Abstand
* Kontaktabzuege der Klassen mit den meisten Ausreissern.

Aufruf:

    python projectV1/backend/dataset/find_class_outliers.py \
        --src ".datasets/05 - dataset/photos" --out projectV1/runs/audit_05/outliers
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CELL = 160


def dhash(path: str, size: int = 8) -> int | None:
    try:
        with Image.open(path) as img:
            small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            pixels = np.asarray(small, dtype=np.int16)
    except Exception:
        return None
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def load_font(size: int = 14):
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(path: str, box: int = CELL) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((box, box), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (box, box), "white")
        canvas.paste(img, ((box - img.width) // 2, (box - img.height) // 2))
        return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="Ab dieser Perzentile (pro Klasse) gilt ein Bild als Ausreisser")
    ap.add_argument("--min-distance", type=int, default=14,
                    help="Mindestabstand zum Medoid, damit ein Bild als Ausreisser zaehlt")
    ap.add_argument("--max-sheets", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in args.src.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    print(f"Bilder: {len(files)}")

    hashes: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path, value in zip(files, pool.map(dhash, files, chunksize=128)):
            if value is not None:
                hashes[path] = value
    print(f"gehasht: {len(hashes)}")

    by_class: dict[str, list[str]] = defaultdict(list)
    for path in hashes:
        by_class[path.split("/")[-2]].append(path)

    outlier_rows: list[dict] = []
    stats_rows: list[dict] = []
    outliers_by_class: dict[str, list[tuple[int, str, str]]] = {}
    for class_name, paths in sorted(by_class.items()):
        values = np.array([hashes[p] for p in paths], dtype=np.uint64)
        distances = np.bitwise_count(values[:, None] ^ values[None, :]).sum(axis=1)
        medoid_index = int(distances.argmin())
        medoid = paths[medoid_index]
        to_medoid = np.bitwise_count(values ^ values[medoid_index])
        limit = max(args.min_distance, float(np.percentile(to_medoid, args.percentile)))
        flagged = [
            (int(to_medoid[i]), paths[i], medoid)
            for i in range(len(paths)) if to_medoid[i] > limit
        ]
        flagged.sort(reverse=True)
        for distance, path, medoid_path in flagged:
            outlier_rows.append({
                "path": path, "class": class_name, "distance": distance, "medoid": medoid_path,
            })
        outliers_by_class[class_name] = flagged
        stats_rows.append({
            "class": class_name,
            "images": len(paths),
            "median_distance": float(np.median(to_medoid)),
            "max_distance": int(to_medoid.max()),
            "outliers": len(flagged),
        })

    with (args.out / "outliers.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "class", "distance", "medoid"])
        writer.writeheader()
        writer.writerows(sorted(outlier_rows, key=lambda r: -r["distance"]))
    with (args.out / "class_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class", "images", "median_distance",
                                                "max_distance", "outliers"])
        writer.writeheader()
        writer.writerows(sorted(stats_rows, key=lambda r: -r["outliers"]))

    total = len(hashes)
    print(f"Ausreisser: {len(outlier_rows)} ({100 * len(outlier_rows) / max(1, total):.2f} %)")
    print("Klassen mit den meisten Ausreissern:")
    for row in sorted(stats_rows, key=lambda r: -r["outliers"])[:10]:
        print(f"  {row['class']:>8}  n={row['images']:<4} median={row['median_distance']:.0f} "
              f"max={row['max_distance']}  outliers={row['outliers']}")

    # Kontaktabzuege: pro Klasse Ausreisser + Medoid
    font = load_font(14)
    worst = [r["class"] for r in sorted(stats_rows, key=lambda r: -r["outliers"])
             if r["outliers"] > 0][:args.max_sheets]
    for sheet_index, class_name in enumerate(worst):
        flagged = outliers_by_class[class_name]
        medoid_path = flagged[0][2] if flagged else None
        entries = [(distance, f"{class_name}  d={distance}", path)
                   for distance, path, _ in flagged[:11]]
        if medoid_path:
            entries.insert(0, (0, f"{class_name}  MEDOID", medoid_path))
        cols = 4
        lines = math.ceil(len(entries) / cols)
        sheet = Image.new("RGB", (cols * CELL, lines * (CELL + 20)), "white")
        draw = ImageDraw.Draw(sheet)
        for position, (_, title, path) in enumerate(entries):
            x = (position % cols) * CELL
            y = (position // cols) * (CELL + 20)
            draw.text((x + 2, y + 2), title, fill="black", font=font)
            try:
                sheet.paste(fit(path), (x, y + 20))
            except Exception:
                pass
            draw.text((x + 2, y + 20 + CELL - 14), Path(path).name[:24], fill="#666666", font=font)
        path = args.out / f"outliers_{sheet_index:02d}_{class_name}.jpg"
        sheet.save(path, quality=88)
    print(f"Reports: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
