#!/usr/bin/env python3
"""Baut einen Klassifikations-Datensatz im Ultralytics-Ordnerformat.

Quellen sind Ordner mit einem Unterordner pro Teilenummer
(``Legosteine_comp``, ``cuttet_multi_lego_yolo``, ...). Jedes Bild wird -
wenn eine YOLO-``.txt`` daneben liegt - mit der Box zugeschnitten (6 % Rand).
Bilder ohne Box werden als Ganzes verwendet (``kind=full``).

Eine Quelle:

    python projectV1/classification/build_dataset.py --force

Mehrere Quellen, Split nach Quellfoto (für Datensätze, in denen ein Foto
mehrere Crops liefert):

    python projectV1/classification/build_dataset.py \
        --source Legosteine_comp --source cuttet_multi_lego_yolo \
        --group-by-source --repeats 1 \
        --out projectV1/classification/data_merged --force

Ergebnis:

    data/
      train/<teilenummer>/*.jpg
      val/<teilenummer>/*.jpg
      test/<teilenummer>/*.jpg
      classes.txt
      manifest.csv
      dataset_stats.json
      data.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
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


def split_by_source(
    samples_by_class: dict[str, list[dict]], val_frac: float, test_frac: float,
    rng: random.Random,
) -> dict[str, dict[str, list[dict]]]:
    """Ganze Quellfotos in denselben Split.

    Nötig, wenn ein Foto mehrere Crops liefert (Handyfotos): sonst liegt
    dasselbe Foto mit anderem Teil gleichzeitig in Training und Test.
    """
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for class_name, samples in samples_by_class.items():
        for sample in samples:
            groups[sample["source"]].append((class_name, sample))

    keys = sorted(groups)
    rng.shuffle(keys)
    total = sum(len(group) for group in groups.values())
    targets = {"val": round(total * val_frac), "test": round(total * test_frac)}
    filled = {"train": 0, "val": 0, "test": 0}
    assigned: dict[str, str] = {}

    for key in keys:
        size = len(groups[key])
        if filled["test"] < targets["test"]:
            split = "test"
        elif filled["val"] < targets["val"]:
            split = "val"
        else:
            split = "train"
        assigned[key] = split
        filled[split] += size

    def counts_by_class() -> dict[str, Counter]:
        counts = {s: Counter() for s in ("train", "val", "test")}
        for key, split in assigned.items():
            for class_name, _ in groups[key]:
                counts[split][class_name] += 1
        return counts

    # Zuerst: jede Klasse soll in val und test vorkommen
    # (kleinste passende Gruppe aus dem Training nachziehen).
    counts = counts_by_class()
    for split in ("val", "test"):
        present = {name for key, s in assigned.items() if s == split for name, _ in groups[key]}
        for missing in sorted(set(samples_by_class) - present):
            candidates = [key for key in keys
                          if assigned[key] == "train"
                          and any(name == missing for name, _ in groups[key])]
            if not candidates:
                continue
            key = min(candidates, key=lambda k: len(groups[k]))
            assigned[key] = split
            for name, _ in groups[key]:
                counts["train"][name] -= 1
                counts[split][name] += 1

    # Danach die Mindestquote reparieren. Die darf die Coverage nicht wieder
    # aufheben, also bleiben in val/test je Klasse mindestens ein Bild.
    for class_name, samples in sorted(samples_by_class.items()):
        need = math.ceil(len(samples) * 0.5) - counts["train"][class_name]
        while need > 0:
            candidates = []
            for key in keys:
                split = assigned[key]
                if split == "train":
                    continue
                names = {name for name, _ in groups[key]}
                if class_name not in names:
                    continue
                if any(counts[split][name] - sum(1 for n, _ in groups[key] if n == name) < 1
                       for name in names):
                    continue
                candidates.append(key)
            if not candidates:
                break
            key = min(candidates, key=lambda k: len(groups[k]))
            source = assigned[key]
            assigned[key] = "train"
            for name, _ in groups[key]:
                counts[source][name] -= 1
                counts["train"][name] += 1
            need -= sum(1 for name, _ in groups[key] if name == class_name)

    out: dict[str, dict[str, list[dict]]] = {"train": defaultdict(list),
                                            "val": defaultdict(list),
                                            "test": defaultdict(list)}
    for key, split in assigned.items():
        for class_name, sample in groups[key]:
            out[split][class_name].append(sample)
    return {split: dict(classes) for split, classes in out.items()}


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
    ap = argparse.ArgumentParser(
        description="Klassifikations-Datensatz aus Teilenummer-Ordnern bauen.")
    ap.add_argument("--source", type=Path, action="append", default=None,
                    help="Quellordner (mehrfach angebbar); Default Legosteine_comp")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--crop-margin", type=float, default=0.06)
    ap.add_argument("--repeats", type=int, default=3, help="Oversampling der Trainingsbilder")
    ap.add_argument("--group-by-source", action="store_true",
                    help="Split nach Quellfoto statt nach Einzelbild")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sources = args.source or [REPO / "Legosteine_comp"]
    for source in sources:
        if not source.is_dir():
            print(f"Quelle nicht gefunden: {source}")
            return 1
    if args.out.exists():
        if not args.force:
            print(f"Ziel existiert bereits: {args.out}\nMit --force neu bauen.")
            return 1
        shutil.rmtree(args.out)

    samples_by_class: dict[str, list[dict]] = {}
    for source in sources:
        for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXT]
            if not images:
                continue
            samples_by_class.setdefault(class_dir.name, []).extend(
                collect_samples(class_dir, args.crop_margin))

    classes = sorted(samples_by_class)
    if not classes:
        print(f"Keine Klassen in {', '.join(str(s) for s in sources)}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.txt").write_text("\n".join(classes) + "\n")

    if args.group_by_source:
        splits = split_by_source(samples_by_class, args.val_frac, args.test_frac,
                                 random.Random(args.seed))
    else:
        splits = {"train": {}, "val": {}, "test": {}}
        for index, (class_name, samples) in enumerate(sorted(samples_by_class.items())):
            train, val, test = split_samples(samples, args.val_frac, args.test_frac,
                                             random.Random(args.seed + index))
            splits["train"][class_name] = train
            splits["val"][class_name] = val
            splits["test"][class_name] = test

    split_counts: Counter = Counter()
    kind_counts: Counter = Counter()
    manifest: list[dict] = []
    per_class: dict[str, dict[str, int]] = {name: {"train": 0, "val": 0, "test": 0}
                                            for name in classes}

    for split_name in ("train", "val", "test"):
        for class_name, samples in sorted(splits[split_name].items()):
            count = args.repeats if split_name == "train" else 1
            for sample in samples:
                for repeat in range(count):
                    dst = (args.out / split_name / class_name
                           / f"{sample['name']}_{repeat}.jpg")
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
            per_class[class_name][split_name] = len(samples) * count

    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "class", "kind", "source"])
        writer.writeheader()
        writer.writerows(manifest)

    missing_val = [c for c in classes if not (args.out / "val" / c).is_dir()]
    missing_test = [c for c in classes if not (args.out / "test" / c).is_dir()]
    stats = {
        "sources": [str(s) for s in sources],
        "classes": len(classes),
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "crop_margin": args.crop_margin,
        "repeats": args.repeats,
        "group_by_source": args.group_by_source,
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

    print(f"Quellen: {', '.join(str(s) for s in sources)}")
    print(f"Ziel:    {args.out}")
    print(f"Klassen: {len(classes)}")
    print(f"Bilder:  {dict(split_counts)}  (crop/full: {dict(kind_counts)})")
    print(f"Train pro Klasse (mit {args.repeats}x): "
          f"min {min(v['train'] for v in per_class.values())}, "
          f"max {max(v['train'] for v in per_class.values())}")
    if missing_val or missing_test:
        print(f"[WARN] ohne val: {missing_val}  ohne test: {missing_test}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
