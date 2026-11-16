#!/usr/bin/env python3
"""Baut den Klassifikations-Datensatz fuer ``projectV1/classification``.

Das ist das Klassifikations-Gegenstueck zum Detection-Training:
dort liefert ``split_datasaet.py`` per ``autosplit`` die Aufteilung
70/20/10, hier uebernimmt das dieses Skript. Ergebnis ist ein
Ultralytics-kompatibler Ordner:

    data/
      train/<teilenummer>/*.jpg
      val/<teilenummer>/*.jpg
      test/<teilenummer>/*.jpg
      classes.txt
      manifest.csv
      dataset_stats.json

Warum gruppenweise geteilt wird: im 05-Datensatz gibt es pro Teil mehrere
Aufnahmen derselben Ansicht (``c<farbe>_<ansicht>_...``). Landet eine
Aufnahme in ``train`` und eine near-identische in ``val``, wird die
Validierung geschoent. Deshalb wandert eine ganze Aufnahmegruppe immer in
denselben Split.

Beispiel:

    python projectV1/classification/build_dataset.py --force
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

from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parent

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Aufnahmegruppe fuer die Auswahl: Farbe + Ansicht (max. N Bilder pro Gruppe).
VIEW_RE = re.compile(r"^(c\d+_\d+)_")
# Split-Gruppe: zusaetzlich die Quelle (z. B. c0_1_Con31). Ganze Gruppe -> ein Split.
SPLIT_RE = re.compile(r"^(c\d+_\d+_[A-Za-z0-9]+)_")
# Klasse fehlt im 05-Datensatz und kommt nur aus echten Crops.
EXTRA_CLASSES = {"4073"}


def list_classes(src: Path) -> list[str]:
    return sorted(p.name for p in src.iterdir() if p.is_dir())


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return (0, 0)


def view_key(path: Path) -> str:
    match = VIEW_RE.match(path.name)
    return match.group(1) if match else path.stem


def group_key(path: Path) -> str:
    match = SPLIT_RE.match(path.name)
    return match.group(1) if match else path.stem


def select_best(images: list[Path], per_class: int, max_per_view: int, min_side: int) -> list[Path]:
    """Beste Bilder: grosse zuerst, hoechstens N pro Aufnahmegruppe."""
    sized = []
    for path in images:
        width, height = image_size(path)
        if min(width, height) >= min_side:
            sized.append((width * height, path))
    if len(sized) < 3:  # zu kleine Bilder: lieber unscharf als gar kein Bild
        sized = []
        for path in images:
            width, height = image_size(path)
            sized.append((width * height, path))
    sized.sort(key=lambda item: item[0], reverse=True)

    n_groups = max(1, len({view_key(path) for _, path in sized}))
    cap = max(max_per_view, -(-per_class // n_groups))  # ceil

    per_group: Counter = Counter()
    chosen: list[Path] = []
    for _, path in sized:
        key = view_key(path)
        if per_group[key] >= cap:
            continue
        per_group[key] += 1
        chosen.append(path)
        if len(chosen) >= per_class:
            break
    if len(chosen) < per_class:  # zweite Runde ohne Gruppendeckel
        for _, path in sized:
            if path in chosen:
                continue
            chosen.append(path)
            if len(chosen) >= per_class:
                break
    return chosen


def split_selected(
    images: list[Path], val_frac: float, test_frac: float, rng: random.Random
) -> tuple[list[Path], list[Path], list[Path]]:
    """Gruppenbewusster 70/20/10-Split.

    Eine ganze Gruppe (Farbe + Ansicht + Quelle) wandert immer in denselben
    Split. Reicht die Anzahl der Gruppen nicht fuer drei Splits, wird
    aufgefuellt, damit jede Klasse in train/val/test vorkommt.
    """
    n = len(images)
    if n == 0:
        return [], [], []
    if n == 1:
        return images, [], []
    if n == 2:
        return images[:1], images[1:], []

    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[group_key(image)].append(image)

    if len(groups) >= 3:
        target = {
            "train": max(1, round(n * (1 - val_frac - test_frac))),
            "val": max(1, round(n * val_frac)),
            "test": max(1, round(n * test_frac)),
        }
        # Grosse Gruppen zuerst; die ersten drei gehen garantiert in
        # train/val/test, der Rest wird nach groesstem Rueckstand verteilt.
        ordered = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
        assigned: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
        for index, (_, group_images) in enumerate(ordered):
            if index < 3:
                split_name = ("train", "val", "test")[index]
            else:
                split_name = max(
                    assigned,
                    key=lambda name: target[name] - len(assigned[name]),
                )
            assigned[split_name].extend(group_images)
        if all(assigned.values()):
            return assigned["train"], assigned["val"], assigned["test"]

    # Zu wenige Gruppen: zufaellig pro Bild, aber jeder Split bleibt belegt.
    shuffled = list(images)
    rng.shuffle(shuffled)
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    while n - n_val - n_test < 1 and (n_val or n_test):
        if n_test >= n_val and n_test:
            n_test -= 1
        else:
            n_val -= 1
    return shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val], shuffled[:n_test]


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(os.path.relpath(src.resolve(), start=dst.parent.resolve()), dst)


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


def add_crops(
    out_dir: Path,
    rows_by_class: dict[str, list[dict]],
    stats: Counter,
    split_counts: Counter,
    manifest: list[dict],
    repeats: int,
) -> None:
    """Schreibt echte Crops nach train/ (mit Oversampling).

    Klassen ohne Studio-Bilder (z. B. 4073) bekommen je ein Bild in val und
    test, damit alle Splits dieselben Klassenordner haben.
    """
    for class_name, rows in rows_by_class.items():
        if stats.get(class_name, 0) == 0 and len(rows) >= 3:
            splits = ["val", "test"] + ["train"] * (len(rows) - 2)
        else:
            splits = ["train"] * len(rows)

        for row, split_name in zip(rows, splits):
            count = repeats if split_name == "train" else 1
            for repeat in range(count):
                dst = out_dir / split_name / class_name / f"{row['name']}_{repeat}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                row["image"].save(dst, quality=92)
                split_counts[split_name] += 1
                manifest.append({
                    "split": split_name, "class": class_name,
                    "source": f"crop:{row['source']}",
                })
            stats[class_name] += count if split_name == "train" else 0
            row["image"].close()


def collect_generic(generic_dir: Path, classes: set[str], exclude: set[str],
                    margin: float, rows: list[dict]) -> None:
    for json_path in sorted(generic_dir.glob("*.json")):
        image_path = json_path.with_suffix(".jpg")
        if not image_path.is_file() or image_path.name in exclude:
            continue
        data = json.loads(json_path.read_text())
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        for index, shape in enumerate(data.get("shapes", [])):
            if shape.get("shape_type") != "rectangle":
                continue
            label = str(shape.get("label", "")).strip()
            if label not in classes:
                continue
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            crop = crop_box(img, (min(xs), min(ys), max(xs), max(ys)), margin)
            rows.append({
                "class": label,
                "name": f"gen_{image_path.stem}_{index}",
                "image": crop.copy(),
                "source": str(image_path),
            })
        img.close()


def collect_legosteine(comp_dir: Path, classes: set[str], margin: float, rows: list[dict]) -> None:
    for class_dir in sorted(p for p in comp_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name
        if class_name not in classes:
            continue
        for txt_path in sorted(class_dir.glob("*.txt")):
            image_path = txt_path.with_suffix(".jpg")
            if not image_path.is_file():
                continue
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            for index, line in enumerate(txt_path.read_text().splitlines()):
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = (float(value) for value in parts[:5])
                x1 = (cx - bw / 2) * img.width
                y1 = (cy - bh / 2) * img.height
                x2 = (cx + bw / 2) * img.width
                y2 = (cy + bh / 2) * img.height
                crop = crop_box(img, (x1, y1, x2, y2), margin)
                rows.append({
                    "class": class_name,
                    "name": f"comp_{image_path.stem}_{index}",
                    "image": crop.copy(),
                    "source": str(image_path),
                })
            img.close()


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
    ap = argparse.ArgumentParser(description="Klassifikations-Datensatz (70/20/10) bauen.")
    ap.add_argument("--src05", type=Path, default=REPO / ".datasets/05 - dataset/photos",
                    help="Studio-Bilder, ein Ordner pro Teilenummer")
    ap.add_argument("--generic", type=Path, default=PROJECT / "dataset/generic",
                    help="Handyfotos mit X-AnyLabeling-JSON")
    ap.add_argument("--legosteine", type=Path, default=REPO / "Legosteine_comp",
                    help="Handyfotos mit YOLO-txt")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--per-class", type=int, default=40, help="max. Studio-Bilder pro Klasse")
    ap.add_argument("--max-per-view", type=int, default=3, help="max. Bilder pro c<farbe>_<ansicht>")
    ap.add_argument("--min-side", type=int, default=160, help="min. Bildkante in px")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--crop-margin", type=float, default=0.06, help="Rand um echte Boxen")
    ap.add_argument("--repeats", type=int, default=2, help="Oversampling der echten Crops")
    ap.add_argument("--exclude-list", type=Path, default=REPO / "multi_lego_yolo",
                    help="Ordner mit Testbildern (*.jpg), die nicht ins Training duerfen")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.src05.is_dir():
        print(f"Quelle nicht gefunden: {args.src05}")
        return 1
    if args.out.exists():
        if not args.force:
            print(f"Ziel existiert bereits: {args.out}\nMit --force neu bauen.")
            return 1
        shutil.rmtree(args.out)

    classes = sorted(set(list_classes(args.src05)) | EXTRA_CLASSES)
    class_set = set(classes)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.txt").write_text("\n".join(classes) + "\n")

    exclude: set[str] = set()
    if args.exclude_list and args.exclude_list.is_dir():
        exclude = {p.name for p in args.exclude_list.glob("*.jpg")}
        print(f"Ausgeschlossene Testbilder: {len(exclude)}")

    stats: Counter = Counter()
    split_counts: Counter = Counter()
    manifest: list[dict] = []

    for class_index, class_name in enumerate(classes):
        if class_name in EXTRA_CLASSES:
            continue  # kommt nur aus Legosteine_comp/generic
        if class_index % 25 == 0:
            print(f"  Klasse {class_index + 1}/{len(classes)}: {class_name}", flush=True)
        images = select_best(
            list_images(args.src05 / class_name), args.per_class, args.max_per_view, args.min_side
        )
        rng = random.Random(args.seed + class_index)
        train, val, test = split_selected(images, args.val_frac, args.test_frac, rng)
        for split_name, split_images in (("train", train), ("val", val), ("test", test)):
            for image in split_images:
                link(image, args.out / split_name / class_name / image.name)
                split_counts[split_name] += 1
                manifest.append({"split": split_name, "class": class_name, "source": str(image)})
        stats[class_name] += len(train)

    crop_rows: list[dict] = []
    if args.generic.is_dir():
        collect_generic(args.generic, class_set, exclude, args.crop_margin, crop_rows)
    if args.legosteine.is_dir():
        collect_legosteine(args.legosteine, class_set, args.crop_margin, crop_rows)

    crop_classes = Counter(row["class"] for row in crop_rows)
    print(f"Echte Crops: {len(crop_rows)} aus {len(crop_classes)} Klassen")
    rows_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in crop_rows:
        rows_by_class[row["class"]].append(row)
    add_crops(args.out, rows_by_class, stats, split_counts, manifest, args.repeats)

    for split_name in ("val", "test"):
        split_dir = args.out / split_name
        if split_dir.is_dir():
            (split_dir / ".gitkeep").touch(exist_ok=True)

    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "class", "source"])
        writer.writeheader()
        writer.writerows(manifest)

    missing_val = [c for c in classes if not (args.out / "val" / c).is_dir()]
    missing_test = [c for c in classes if not (args.out / "test" / c).is_dir()]
    per_class_train = {c: stats.get(c, 0) for c in classes}
    dataset_stats = {
        "source_studio": str(args.src05),
        "classes": len(classes),
        "per_class": args.per_class,
        "max_per_view": args.max_per_view,
        "min_side": args.min_side,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "excluded_test_images": sorted(exclude),
        "real_crops": len(crop_rows),
        "repeats": args.repeats,
        "split_counts": dict(split_counts),
        "classes_without_val": missing_val,
        "classes_without_test": missing_test,
        "train_per_class": per_class_train,
    }
    (args.out / "dataset_stats.json").write_text(
        json.dumps(dataset_stats, indent=2, ensure_ascii=False)
    )
    write_data_yaml(args.out, classes)

    print(f"Ziel:    {args.out}")
    print(f"Klassen: {len(classes)}")
    print(f"Bilder:  {dict(split_counts)}")
    print(f"Train pro Klasse: min {min(per_class_train.values())}, "
          f"median {sorted(per_class_train.values())[len(per_class_train) // 2]}, "
          f"max {max(per_class_train.values())}")
    if missing_val or missing_test:
        print(f"[WARN] ohne val: {len(missing_val)}  ohne test: {len(missing_test)}")
    print(f"data.yaml: {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
