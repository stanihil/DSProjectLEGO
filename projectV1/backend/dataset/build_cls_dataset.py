#!/usr/bin/env python3
"""Baut aus einem Ordner-Datensatz (Klasse = Ordnername) einen
Ultralytics-Klassifikations-Datensatz mit train/val/test.

    Quelle:  .datasets/05 - dataset/photos        (447 Klassen)
    Ziel:    projectV1/dataset/classification/
               train/<klasse>/<bild>  -> Symlink auf das Original
               val/<klasse>/<bild>
               test/<klasse>/<bild>

Fuer ``yolo26s-cls`` sind keine Bounding-Boxen noetig: der Ordnername ist das
Label.  Standardmaessig werden symbolische Links erzeugt (kein Kopieren).

Beispiele:
    # kompletter 447-Klassen-Datensatz
    python projectV1/backend/dataset/build_cls_dataset.py

    # kleiner Smoke-Test-Datensatz (max. 10 Bilder pro Klasse)
    python projectV1/backend/dataset/build_cls_dataset.py \
        --dst projectV1/dataset/classification_smoke --max-per-class 10 --force

    # nur die 20 eigenen Klassen
    python projectV1/backend/dataset/build_cls_dataset.py \
        --classes 2654,3001,3002,3003,3004,3005,3009,3010,3020,3021,3022,3023,3062,3622,3623,3666,3710,4073,54200,6215
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GROUP_RE = re.compile(r"^(c\d+_\d+)_")


def list_classes(src: Path) -> list[str]:
    """Klassenordner alphabetisch (identisch zur Reihenfolge von Ultralytics)."""
    return sorted(p.name for p in src.iterdir() if p.is_dir())


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def group_key(path: Path) -> str:
    """Naeherungsweise Aufnahmegruppe aus dem Dateinamen (Farbe + Ansicht)."""
    match = GROUP_RE.match(path.name)
    return match.group(1) if match else path.stem


def split_class(
    images: list[Path],
    val_frac: float,
    test_frac: float,
    rng: random.Random,
    use_groups: bool,
) -> tuple[list[Path], list[Path], list[Path], str]:
    """Teilt die Bilder *einer* Klasse in train/val/test."""
    n = len(images)
    if n == 0:
        return [], [], [], "empty"

    if use_groups:
        groups: dict[str, list[Path]] = defaultdict(list)
        for image in images:
            groups[group_key(image)].append(image)
        if len(groups) >= 5:
            keys = sorted(groups)
            rng.shuffle(keys)
            n_test = max(1, round(n * test_frac))
            n_val = max(1, round(n * val_frac))
            test, val, train = [], [], []
            for key in keys:
                if len(test) < n_test:
                    test.extend(groups[key])
                elif len(val) < n_val:
                    val.extend(groups[key])
                else:
                    train.extend(groups[key])
            if train and val and test:
                return train, val, test, "group"
        # zu wenige (oder ungeeignet verteilte) Gruppen -> zufaelliger Bild-Split

    shuffled = list(images)
    rng.shuffle(shuffled)
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    n_val = max(1, round(n * val_frac)) if n - n_test >= 2 else 0
    while n - n_test - n_val < 1 and (n_test or n_val):
        if n_test >= n_val and n_test:
            n_test -= 1
        else:
            n_val -= 1
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test, "random"


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(os.path.relpath(src, start=dst.parent), dst)
    else:
        shutil.copy2(src, dst)


def parse_classes(arg: str | None, available: list[str]) -> list[str]:
    if not arg:
        return available
    path = Path(arg)
    if path.is_file():
        requested = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        requested = [item.strip() for item in arg.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        print(f"[WARN] Unbekannte Klassen werden ignoriert: {unknown}")
    return [name for name in available if name in set(requested)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Klassifikations-Datensatz aus Klassenordnern bauen.")
    ap.add_argument("--src", type=Path, default=Path(".datasets/05 - dataset/photos"))
    ap.add_argument("--dst", type=Path, default=Path("projectV1/dataset/classification"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--classes", default=None, help="Komma-Liste oder Datei mit Klassennamen")
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = alle Bilder")
    ap.add_argument("--no-group-split", action="store_true",
                    help="rein zufaellig splitten statt nach Aufnahmegruppe")
    ap.add_argument("--force", action="store_true", help="vorhandenes Ziel loeschen")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"Quelle nicht gefunden: {args.src}")
        return 1

    available = list_classes(args.src)
    classes = parse_classes(args.classes, available)
    if not classes:
        print("Keine Klassen ausgewaehlt.")
        return 1

    if args.dst.exists():
        if not args.force:
            print(f"Ziel existiert bereits: {args.dst}\nMit --force loeschen und neu bauen.")
            return 1
        shutil.rmtree(args.dst)

    manifest_path = args.dst / "manifest.csv"
    args.dst.mkdir(parents=True, exist_ok=True)
    (args.dst / "classes.txt").write_text("\n".join(classes) + "\n")

    stats = {
        "source": str(args.src),
        "classes": len(classes),
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "mode": args.mode,
        "classes_without_val": [],
        "classes_without_test": [],
        "split_counts": Counter(),
        "split_mode": Counter(),
        "per_class": {},
    }

    manifest_rows = []
    total = 0
    for class_index, class_name in enumerate(classes):
        images = list_images(args.src / class_name)
        rng = random.Random(args.seed + class_index)
        if args.max_per_class > 0 and len(images) > args.max_per_class:
            images = rng.sample(images, args.max_per_class)

        train, val, test, mode = split_class(
            images, args.val_frac, args.test_frac, rng, not args.no_group_split
        )
        stats["split_mode"][mode] += 1
        stats["per_class"][class_name] = {"train": len(train), "val": len(val), "test": len(test)}
        if not val:
            stats["classes_without_val"].append(class_name)
        if not test:
            stats["classes_without_test"].append(class_name)

        for split_name, split_images in (("train", train), ("val", val), ("test", test)):
            for image in split_images:
                dst = args.dst / split_name / class_name / image.name
                link_or_copy(image, dst, args.mode)
                manifest_rows.append({
                    "split": split_name,
                    "class": class_name,
                    "group": group_key(image),
                    "file": image.name,
                    "source": str(image.resolve()),
                })
                total += 1
                stats["split_counts"][split_name] += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "class", "group", "file", "source"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    stats["total_images"] = total
    stats["split_counts"] = dict(stats["split_counts"])
    stats["split_mode"] = dict(stats["split_mode"])
    (args.dst / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"Quelle:  {args.src}  ({len(available)} Klassen verfuegbar)")
    print(f"Ziel:    {args.dst}  ({len(classes)} Klassen, Modus {args.mode})")
    print(f"Bilder:  {total}  ->  train {stats['split_counts'].get('train', 0)}, "
          f"val {stats['split_counts'].get('val', 0)}, test {stats['split_counts'].get('test', 0)}")
    print(f"Split:   {stats['split_mode']}")
    if stats["classes_without_val"]:
        print(f"[WARN] {len(stats['classes_without_val'])} Klassen ohne val: "
              f"{stats['classes_without_val'][:10]} ...")
    if stats["classes_without_test"]:
        print(f"[WARN] {len(stats['classes_without_test'])} Klassen ohne test: "
              f"{stats['classes_without_test'][:10]} ...")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
