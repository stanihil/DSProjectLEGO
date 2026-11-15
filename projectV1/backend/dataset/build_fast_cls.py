#!/usr/bin/env python3
"""Baut einen kleinen, balancierten Klassifikations-Datensatz fuer ein schnelles
Training ("beste Bilder") und mischt echte Handyfotos bei.

Warum:
  * 52.597 Studio-Bilder x 447 Klassen sind auf der M1 zu langsam fuer einen
    schnellen Durchlauf (~19 min/Epoche).
  * Klassen sind extrem unbalanciert (1-692 Bilder). Ein Deckel pro Klasse
    gibt seltenen Klassen genauso viel Gewicht wie haeufigen.
  * Das Modell soll spaeter Boxen aus Handyfotos klassifizieren. Ein paar
    echte Crops im Training schliessen die Domain-Luecke.

Auswahl ("beste Bilder"):
  * nur Bilder mit Mindestkante (Standard 160 px, Details bleiben erhalten),
  * nach Aufloesung sortiert (grosse Bilder zuerst),
  * hoechstens N Bilder pro Aufnahmegruppe ``c<farbe>_<ansicht>``
    (Diversitaet statt near-duplicates),
  * maximal ``--per-class`` Bilder pro Klasse.

Zusaetzlich (nur train):
  * Crops aus ``projectV1/dataset/generic`` (X-AnyLabeling-JSON, echte Fotos),
  * Crops aus ``Legosteine_comp`` (YOLO-txt, echte Fotos).

Beispiel:

    python projectV1/backend/dataset/build_fast_cls.py \
        --out projectV1/dataset/cls_fast --per-class 20 \
        --exclude-list multi_lego_yolo --force
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

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GROUP_RE = re.compile(r"^(c\d+_\d+)_")
# Klassen, die im 05-Datensatz fehlen, aber in den Zielklassen vorkommen.
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


def group_key(path: Path) -> str:
    match = GROUP_RE.match(path.name)
    return match.group(1) if match else path.stem


def select_best(images: list[Path], per_class: int, max_per_view: int, min_side: int) -> list[Path]:
    """Sortiert nach Aufloesung, deckelt pro Aufnahmegruppe.

    Der Deckel pro Gruppe ist adaptiv: Klassen mit nur einer Ansicht (z. B. alle
    Bilder ``c0_1``) wuerden sonst auf zwei Bilder zusammenfallen. Er ist
    mindestens ``max_per_view`` und so gross, dass ``per_class`` erreichbar ist.
    """
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

    n_groups = max(1, len({group_key(path) for _, path in sized}))
    cap = max(max_per_view, -(-per_class // n_groups))  # ceil

    per_group: Counter = Counter()
    chosen: list[Path] = []
    for _, path in sized:
        key = group_key(path)
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
    """Gruppenbewusster Split; bei zu wenigen Gruppen zufaellig."""
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

    if len(groups) >= 4:
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = max(1, round(n * val_frac))
        n_test = max(1, round(n * test_frac))
        val: list[Path] = []
        test: list[Path] = []
        train: list[Path] = []
        for key in keys:
            if len(val) < n_val:
                val.extend(groups[key])
            elif len(test) < n_test:
                test.extend(groups[key])
            else:
                train.extend(groups[key])
        if train and val and test:
            return train, val, test

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
    """Schreibt Crops nach train/ (mit Oversampling).

    Klassen ohne Studio-Bilder (z. B. 4073) bekommen zusaetzlich je ein Bild
    in val und test, damit alle Splits dieselben Klassenordner haben.
    """
    for class_name, rows in rows_by_class.items():
        if stats.get(class_name, 0) == 0 and len(rows) >= 3:
            splits = ["val", "test"] + ["train"] * (len(rows) - 2)
        else:
            splits = ["train"] * len(rows)

        for index, (row, split_name) in enumerate(zip(rows, splits)):
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Schneller, balancierter Klassifikations-Datensatz.")
    ap.add_argument("--src05", type=Path, default=Path(".datasets/05 - dataset/photos"))
    ap.add_argument("--generic", type=Path, default=Path("projectV1/dataset/generic"))
    ap.add_argument("--legosteine", type=Path, default=Path("Legosteine_comp"))
    ap.add_argument("--out", type=Path, default=Path("projectV1/dataset/cls_fast"))
    ap.add_argument("--per-class", type=int, default=20, help="max. Studio-Bilder pro Klasse")
    ap.add_argument("--max-per-view", type=int, default=2, help="max. Bilder pro c<farbe>_<ansicht>")
    ap.add_argument("--min-side", type=int, default=160, help="min. Bildkante in px")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--crop-margin", type=float, default=0.08, help="Rand um echte Boxen")
    ap.add_argument("--repeats", type=int, default=2, help="Oversampling der echten Crops")
    ap.add_argument("--exclude-list", type=Path, default=None,
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
        "excluded_test_images": sorted(exclude),
        "real_crops": len(crop_rows),
        "repeats": args.repeats,
        "split_counts": dict(split_counts),
        "classes_without_val": missing_val,
        "classes_without_test": missing_test,
        "train_per_class": per_class_train,
    }
    (args.out / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2, ensure_ascii=False))

    print(f"Ziel:    {args.out}")
    print(f"Klassen: {len(classes)}  (inkl. {sorted(EXTRA_CLASSES - set(list_classes(args.src05)))} nur aus echten Crops)")
    print(f"Bilder:  {dict(split_counts)}")
    print(f"Train pro Klasse: min {min(per_class_train.values())}, "
          f"median {sorted(per_class_train.values())[len(per_class_train) // 2]}, "
          f"max {max(per_class_train.values())}")
    if missing_val or missing_test:
        print(f"[WARN] ohne val: {len(missing_val)} {missing_val[:8]}  ohne test: {len(missing_test)}")
    print(f"Manifest: {args.out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
