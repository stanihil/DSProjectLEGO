#!/usr/bin/env python3
"""Prueft einen Klassifikations-Datensatz (Ordner = Klasse) auf Qualitaetsprobleme.

Geprueft wird:

* unlesbare / kaputte Bilder,
* Aufloesung (Mindestkante),
* Near-Duplicates per dHash (64 Bit),
* **Label-Konflikte**: dasselbe/nahzu identische Bild in zwei verschiedenen Klassen.

Ein Label-Konflikt ist das wichtigste Signal: wenn dieselbe Aufnahme in den
Ordnern ``3001`` und ``3002`` liegt, ist mindestens eine der beiden falsch.
Solche Bilder werden in ``conflicts.csv`` ausgegeben und koennen mit
``--exclude`` aus dem Training genommen werden.

Aufruf:

    python projectV1/backend/dataset/audit_dataset.py \
        --src ".datasets/05 - dataset/photos" --out projectV1/runs/audit_05
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(path: str, size: int = 8) -> tuple[int, int, int] | None:
    """64-Bit-Difference-Hash + Bildbreite/-hoehe."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            pixels = np.asarray(small, dtype=np.int16)
    except Exception:
        return None
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value, width, height


def main() -> int:
    ap = argparse.ArgumentParser(description="Qualitaets-Audit fuer Ordner-als-Klasse-Datensaetze.")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-side", type=int, default=160)
    ap.add_argument("--max-distance", type=int, default=4, help="dHash-Abstand fuer 'near duplicate'")
    ap.add_argument("--conflict-distance", type=int, default=4,
                    help="dHash-Abstand, ab dem ein klassenuebergreifendes Paar als Konflikt gilt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exclude", type=Path, default=None,
                    help="CSV mit Spalte 'path', deren Bilder nicht mehr geprueft werden")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"Quelle nicht gefunden: {args.src}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    excluded: set[str] = set()
    if args.exclude and args.exclude.is_file():
        with args.exclude.open(newline="", encoding="utf-8") as fh:
            excluded = {row["path"] for row in csv.DictReader(fh)}

    files = sorted(
        str(p) for p in args.src.rglob("*")
        if p.suffix.lower() in IMAGE_EXT and str(p) not in excluded
    )
    print(f"Bilder: {len(files)}  Klassen: {len([d for d in args.src.iterdir() if d.is_dir()])}")

    hashes: dict[str, tuple[int, int, int]] = {}
    bad: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path, result in zip(files, pool.map(dhash, files, chunksize=64)):
            if result is None:
                bad.append(path)
            else:
                hashes[path] = result
    print(f"gelesen: {len(hashes)}  kaputt: {len(bad)}")

    sizes = np.array([[h[1], h[2]] for h in hashes.values()])
    min_sides = sizes.min(axis=1)
    small = int((min_sides < args.min_side).sum())

    def class_of(path: str) -> str:
        return Path(path).relative_to(args.src).parts[0]

    # Alle Paare per numpy-popcount vergleichen (52k x 52k, blockweise).
    paths = list(hashes)
    values = np.array([hashes[path][0] for path in paths], dtype=np.uint64)
    classes = np.array([class_of(path) for path in paths])
    duplicate_pairs: list[tuple[str, str, int]] = []
    conflicts: list[tuple[str, str, int]] = []
    block_size = 1024
    for start in range(0, len(paths), block_size):
        block = values[start:start + block_size]
        distance = np.bitwise_count(block[:, None] ^ values[None, :])
        rows, cols = np.where(distance <= args.max_distance)
        for row, col in zip(rows.tolist(), cols.tolist()):
            first = start + row
            if col <= first:
                continue
            value = int(distance[row, col])
            duplicate_pairs.append((paths[first], paths[col], value))
            if value <= args.conflict_distance and classes[first] != classes[col]:
                conflicts.append((paths[first], paths[col], value))

    duplicate_pairs.sort(key=lambda item: item[2])
    conflicts.sort(key=lambda item: item[2])

    with (args.out / "duplicates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path_a", "path_b", "distance", "class_a", "class_b"])
        for a, b, d in duplicate_pairs:
            writer.writerow([a, b, d, class_of(a), class_of(b)])
    with (args.out / "conflicts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "other_path", "distance", "class", "other_class"])
        for a, b, d in conflicts:
            writer.writerow([a, b, d, class_of(a), class_of(b)])
    with (args.out / "broken.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(bad))

    per_class = Counter(class_of(path) for path in hashes)
    conflict_classes = Counter()
    for a, _, _ in conflicts:
        conflict_classes[class_of(a)] += 1

    report = {
        "source": str(args.src),
        "images": len(hashes),
        "broken": len(bad),
        "min_side": args.min_side,
        "images_below_min_side": small,
        "classes": len(per_class),
        "images_per_class": dict(sorted(per_class.items())),
        "near_duplicate_pairs": len(duplicate_pairs),
        "conflict_pairs": len(conflicts),
        "classes_with_conflicts": dict(conflict_classes.most_common()),
    }
    (args.out / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"unter {args.min_side} px: {small} ({100 * small / max(1, len(hashes)):.1f} %)")
    print(f"Near-Duplicate-Paare:  {len(duplicate_pairs)}")
    print(f"Label-Konflikte:       {len(conflicts)} Paare in {len(conflict_classes)} Klassen")
    if conflicts:
        print("  Beispiele:")
        for a, b, d in conflicts[:5]:
            print(f"    {class_of(a):>8} vs {class_of(b):<8} d={d}  {Path(a).name}")
    print(f"Report: {args.out / 'audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
