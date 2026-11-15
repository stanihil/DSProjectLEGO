#!/usr/bin/env python3
"""Auto-Labeling fuer Ordner-Datensaetze (ein Objekt pro Bild).

Gleiche Grundidee wie ``helper/auto_label_yolo.py``:
YOLO-Detektion -> groesste Box -> YOLO-txt.  Erweitert um

* Klassenliste aus den Ordnernamen (statt fest verdrahteter 19 Klassen),
* Resume (vorhandene Labels werden uebersprungen),
* Limit pro Klasse fuer schnelle Pilotlaeufe,
* Box-Validierung (zu klein / zu gross wird verworfen),
* Statistik pro Klasse als CSV + optionale Preview-Bilder.

Beispiel (Pilot auf dem 05-Datensatz):

    python helper/label_dataset.py \
        --src ".datasets/05 - dataset/photos" \
        --out runs/labels_05 \
        --limit-per-class 3 \
        --preview 1 \
        --device mps

Ausgabe:

    runs/labels_05/classes.txt              Klassen-ID -> Teilenummer
    runs/labels_05/labels/<klasse>/*.txt    YOLO-Labels (Klassen-ID, cx, cy, w, h)
    runs/labels_05/labeling_stats.csv       Statistik pro Klasse
    runs/labels_05/preview/<klasse>/*.jpg   optional, eingezeichnete Boxen
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ClassStats:
    name: str
    images: int = 0
    detections: int = 0
    accepted: int = 0
    rejected_area: int = 0
    empty: int = 0
    fallback: int = 0
    conf_sum: float = 0.0
    area_sum: float = 0.0

    def row(self, class_id: int) -> dict:
        return {
            "class_id": class_id,
            "class": self.name,
            "images": self.images,
            "detections": self.detections,
            "accepted": self.accepted,
            "empty": self.empty,
            "rejected_area": self.rejected_area,
            "fallback_full_image": self.fallback,
            "mean_conf": round(self.conf_sum / self.accepted, 3) if self.accepted else 0.0,
            "mean_box_area": round(self.area_sum / self.accepted, 4) if self.accepted else 0.0,
        }


def list_classes(src: Path) -> list[str]:
    """Alle Unterordner als Klassen, alphabetisch (wie Ultralytics)."""
    return sorted(p.name for p in src.iterdir() if p.is_dir())


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def largest_valid_box(result, min_area: float, max_area: float):
    """Groesste Box, deren Flaechenanteil im erlaubten Bereich liegt.

    Rueckgabe: (xyxy, conf, area_frac) oder None.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    h, w = result.orig_shape
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    areas = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])) / float(w * h)

    for i in np.argsort(areas)[::-1]:
        if min_area <= areas[i] <= max_area:
            return xyxy[i], float(confs[i]), float(areas[i])
    return None


def write_yolo_label(path: Path, class_id: int, box, image_w: int, image_h: int) -> None:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / image_w
    cy = ((y1 + y2) / 2) / image_h
    bw = (x2 - x1) / image_w
    bh = (y2 - y1) / image_h
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def draw_preview(image_path: Path, out_path: Path, label: str, box, conf: float) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        img, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def process_class(
    model: YOLO,
    src_class_dir: Path,
    class_name: str,
    class_id: int,
    args: argparse.Namespace,
    stats: ClassStats,
) -> None:
    images = list_images(src_class_dir)
    if args.limit_per_class > 0:
        rng = random.Random(args.seed + class_id)
        images = rng.sample(images, min(args.limit_per_class, len(images)))
    if args.resume:
        images = [p for p in images if not (args.out / "labels" / class_name / f"{p.stem}.txt").exists()]
    if not images:
        return

    stats.images = len(images)
    label_dir = args.out / "labels" / class_name
    label_dir.mkdir(parents=True, exist_ok=True)

    preview_done = 0
    for start in range(0, len(images), args.batch):
        chunk = images[start:start + args.batch]
        try:
            results = model.predict(
                [str(p) for p in chunk],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device or None,
                verbose=False,
            )
        except Exception as exc:  # einzelnes kaputtes Bild nicht den Lauf killen
            print(f"  [WARN] Batch in {class_name} fehlgeschlagen: {exc}")
            results = []
            for p in chunk:
                try:
                    results.append(model.predict(
                        str(p), conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                        device=args.device or None, verbose=False,
                    )[0])
                except Exception:
                    results.append(None)

        for image_path, result in zip(chunk, results):
            label_path = label_dir / f"{image_path.stem}.txt"
            if result is None:
                stats.empty += 1
                continue
            stats.detections += len(result.boxes) if result.boxes is not None else 0
            found = largest_valid_box(result, args.min_area, args.max_area)

            if found is not None:
                box, conf, area = found
                h, w = result.orig_shape
                write_yolo_label(label_path, class_id, box, w, h)
                stats.accepted += 1
                stats.conf_sum += conf
                stats.area_sum += area
                if preview_done < args.preview:
                    draw_preview(
                        image_path,
                        args.out / "preview" / class_name / f"preview_{image_path.stem}.jpg",
                        class_name, box, conf,
                    )
                    preview_done += 1
            elif args.fallback == "full":
                write_yolo_label(label_path, class_id, (0, 0, result.orig_shape[1], result.orig_shape[0]),
                                 result.orig_shape[1], result.orig_shape[0])
                stats.fallback += 1
            elif not result.boxes:
                stats.empty += 1
            else:
                stats.rejected_area += 1

    print(
        f"  {class_name:<8} bilder={stats.images:<4} erkannt={stats.accepted:<4} "
        f"leer={stats.empty:<4} verworfen={stats.rejected_area:<4} fallback={stats.fallback}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Auto-Labeling fuer Ordner-Datensaetze mit einem Objekt pro Bild.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--src", type=Path, required=True, help="Wurzel mit einem Unterordner pro Klasse")
    ap.add_argument("--out", type=Path, default=Path("runs/labels"), help="Ausgabeordner")
    ap.add_argument("--model", type=Path, default=Path("helper/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="", help="z.B. mps, cpu, 0 (leer = automatisch)")
    ap.add_argument("--min-area", type=float, default=0.01, help="min. Boxflaeche relativ zum Bild")
    ap.add_argument("--max-area", type=float, default=0.95, help="max. Boxflaeche relativ zum Bild")
    ap.add_argument("--limit-per-class", type=int, default=0, help="0 = alle Bilder")
    ap.add_argument("--preview", type=int, default=0, help="Preview-Bilder pro Klasse")
    ap.add_argument("--fallback", choices=["none", "full"], default="none",
                    help="full = ganze Bildflaeche als Box, wenn nichts erkannt wurde")
    ap.add_argument("--resume", action="store_true", help="vorhandene Labels ueberspringen")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.src.is_dir():
        print(f"Quelle nicht gefunden: {args.src}")
        return 1
    if not args.model.exists():
        print(f"Modell nicht gefunden: {args.model}")
        return 1

    classes = list_classes(args.src)
    if not classes:
        print(f"Keine Klassenordner in {args.src}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.txt").write_text("\n".join(classes) + "\n")

    print(f"Modell: {args.model}")
    print(f"Quelle: {args.src} ({len(classes)} Klassen)")
    print(f"Ausgabe: {args.out}")
    model = YOLO(str(args.model))

    rows = []
    total = defaultdict(int)
    for class_id, class_name in enumerate(classes):
        stats = ClassStats(class_name)
        process_class(model, args.src / class_name, class_name, class_id, args, stats)
        rows.append(stats.row(class_id))
        for key in ("images", "accepted", "empty", "rejected_area", "fallback"):
            total[key] += getattr(stats, key)

    stats_path = args.out / "labeling_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = max(1, total["images"])
    print("\nGesamt")
    print(f"  Bilder:            {total['images']}")
    print(f"  akzeptierte Boxen: {total['accepted']} ({100 * total['accepted'] / n:.1f}%)")
    print(f"  leer (keine Det.): {total['empty']} ({100 * total['empty'] / n:.1f}%)")
    print(f"  Flaeche verworfen: {total['rejected_area']} ({100 * total['rejected_area'] / n:.1f}%)")
    print(f"  Fallback full:     {total['fallback']}")
    print(f"\nStatistik: {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
