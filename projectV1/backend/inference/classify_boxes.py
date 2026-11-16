#!/usr/bin/env python3
"""Klassifiziert vorhandene Boxen in Bildern mit einem YOLO-Klassifikator.

Typischer Fall: ``multi_lego_yolo/`` enthaelt Handyfotos mit mehreren LEGO-
Teilen und YOLO-txt-Dateien, in denen alle Boxen die Klasse 0 haben (also
Boxen ohne echte Labels). Dieses Skript schneidet jede Box aus, laesst den
Klassifikator die Teilenummer vorhersagen und schreibt

    <out>/predictions.csv        eine Zeile pro Box (Top-1 + Top-5)
    <out>/annotated/<bild>.jpg   Bild mit Teilenummer und Konfidenz
    <out>/summary.csv            gezaehlte Teile pro Bild
    <out>/sheet.jpg              Kontaktabzug der annotierten Bilder

Beispiele:

    python projectV1/backend/inference/classify_boxes.py \
        --images multi_lego_yolo \
        --weights projectV1/runs/cls_yolo26m_clean/weights/best.pt \
        --out projectV1/runs/cls_yolo26m_clean/infer_multi

    # ohne vorhandene txt-Boxen: Detektor als Box-Vorschlag nutzen
    python projectV1/backend/inference/classify_boxes.py \
        --images multi_lego_yolo --detector helper/best.pt --conf 0.3 ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Boxen mit einem YOLO-Klassifikator labeln.")
    ap.add_argument("--images", type=Path, required=True, help="Ordner oder einzelne Dateien")
    ap.add_argument("--labels", type=Path, default=None,
                    help="Ordner mit YOLO-txt (Standard: neben den Bildern)")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--detector", type=Path, default=None,
                    help="optional: Modell fuer Box-Vorschlaege, wenn keine txt existiert")
    ap.add_argument("--conf", type=float, default=0.25, help="Detektor-Konfidenz (nur mit --detector)")
    ap.add_argument("--margin", type=float, default=0.10, help="Rand um die Box beim Crop")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, default=Path("projectV1/runs/infer_boxes"))
    ap.add_argument("--min-conf", type=float, default=0.0, help="nur Treffer ab dieser Konfidenz zeigen")
    ap.add_argument("--truth-json", type=Path, default=None,
                    help="Ordner mit X-AnyLabeling-JSON (Ground Truth) fuer eine Trefferquote")
    return ap.parse_args()


def list_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)
    return [path]


def read_boxes(image: np.ndarray, label_path: Path | None, detector: YOLO | None,
               conf: float, device: str) -> list[tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    if label_path is not None and label_path.is_file():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (float(value) for value in parts[1:5])
            x1 = int((cx - bw / 2) * width)
            y1 = int((cy - bh / 2) * height)
            x2 = int((cx + bw / 2) * width)
            y2 = int((cy + bh / 2) * height)
            boxes.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
    elif detector is not None:
        result = detector.predict(image, conf=conf, verbose=False, device=device)[0]
        for x1, y1, x2, y2 in result.boxes.xyxy.cpu().numpy().astype(int):
            boxes.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
    return boxes


def crop_with_margin(image: np.ndarray, box: tuple[int, int, int, int], margin: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    mx = int((x2 - x1) * margin)
    my = int((y2 - y1) * margin)
    height, width = image.shape[:2]
    return image[max(0, y1 - my):min(height, y2 + my), max(0, x1 - mx):min(width, x2 + mx)]


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_truth(truth_dir: Path, images: list[Path], boxes_per_image, predictions, labeled,
                   out_dir: Path, iou_threshold: float = 0.35) -> None:
    """Vergleicht Vorhersagen mit X-AnyLabeling-JSON (Ground Truth)."""
    import json

    total = correct = unmatched = 0
    top5_hits = 0
    per_class: Counter = Counter()
    per_class_ok: Counter = Counter()
    errors: Counter = Counter()

    for image_index, path in enumerate(images):
        json_path = truth_dir / f"{path.stem}.json"
        if not json_path.is_file():
            continue
        data = json.loads(json_path.read_text())
        truth = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            truth.append({"label": str(shape["label"]), "box": (min(xs), min(ys), max(xs), max(ys)),
                          "used": False})

        for box_index, box in enumerate(boxes_per_image[image_index]):
            row = labeled.get((image_index, box_index))
            if row is None:
                continue
            best = None
            best_iou = 0.0
            for candidate in truth:
                if candidate["used"]:
                    continue
                value = box_iou(box, candidate["box"])
                if value > best_iou:
                    best, best_iou = candidate, value
            if best is None or best_iou < iou_threshold:
                unmatched += 1
                continue
            best["used"] = True
            total += 1
            per_class[best["label"]] += 1
            if row["part"] == best["label"]:
                correct += 1
                top5_hits += 1
                per_class_ok[best["label"]] += 1
            else:
                errors[(best["label"], row["part"])] += 1
                top5_parts = {item.rsplit(":", 1)[0] for item in row["top5"].split()}
                if best["label"] in top5_parts:
                    top5_hits += 1

    if total == 0:
        print("Keine Zuordnung zu Ground-Truth-Boxen moeglich.")
        return

    per_class_rows = []
    for label, count in per_class.most_common():
        hits = per_class_ok.get(label, 0)
        per_class_rows.append({"class": label, "boxes": count, "top1": hits,
                               "acc": round(hits / count, 4)})

    result = {
        "boxes_matched": total,
        "boxes_unmatched": unmatched,
        "top1_acc": round(correct / total, 4),
        "top5_acc": round(top5_hits / total, 4),
        "per_class": per_class_rows,
        "top_errors": [
            {"true": true, "pred": pred, "count": count}
            for (true, pred), count in errors.most_common(15)
        ],
    }
    (out_dir / "truth_eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with (out_dir / "truth_eval.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class", "boxes", "top1", "acc"])
        writer.writeheader()
        writer.writerows(sorted(per_class_rows, key=lambda row: row["acc"]))

    print(f"\nGround Truth ({total} zugeordnete Boxen, {unmatched} ohne IoU-Match):")
    print(f"  Top-1: {100 * correct / total:.1f} %   Top-5: {100 * top5_hits / total:.1f} %")
    worst = sorted(per_class_rows, key=lambda row: row["acc"])[:5]
    print("  schwaechste Klassen: " + ", ".join(f"{r['class']} {r['acc']:.0%}" for r in worst))
    print(f"  Details: {out_dir / 'truth_eval.json'}")


def main() -> int:
    args = parse_args()
    images = list_images(args.images)
    if not images:
        print(f"Keine Bilder in {args.images}")
        return 1
    if not args.weights.is_file():
        print(f"Gewichte nicht gefunden: {args.weights}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "annotated").mkdir(exist_ok=True)

    detector = YOLO(str(args.detector)) if args.detector else None
    model = YOLO(str(args.weights))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    crops: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []  # (Bildindex, Boxindex)
    boxes_per_image: list[list[tuple[int, int, int, int]]] = []
    images_bgr: list[np.ndarray] = []

    for image_index, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            print(f"[WARN] unlesbar: {path}")
            images_bgr.append(np.zeros((1, 1, 3), np.uint8))
            boxes_per_image.append([])
            continue
        label_path = (args.labels / f"{path.stem}.txt") if args.labels else path.with_suffix(".txt")
        boxes = read_boxes(image, label_path, detector, args.conf, args.device)
        images_bgr.append(image)
        boxes_per_image.append(boxes)
        for box_index, box in enumerate(boxes):
            crops.append(crop_with_margin(image, box, args.margin))
            meta.append((image_index, box_index))
        print(f"{path.name}: {len(boxes)} Boxen")

    if not crops:
        print("Keine Boxen gefunden.")
        return 1

    results = model.predict(
        crops, imgsz=args.imgsz, batch=args.batch, device=args.device, verbose=False
    )

    predictions: list[dict] = []
    for (image_index, box_index), result in zip(meta, results):
        probs = result.probs
        top5 = [int(i) for i in probs.top5]
        top5_conf = [float(c) for c in probs.top5conf]
        predictions.append({
            "image_index": image_index,
            "box_index": box_index,
            "class_id": top5[0],
            "part": names.get(top5[0], str(top5[0])),
            "confidence": top5_conf[0],
            "top5": " ".join(f"{names.get(i, i)}:{c:.3f}" for i, c in zip(top5, top5_conf)),
        })

    labeled: dict[tuple[int, int], dict] = {
        (row["image_index"], row["box_index"]): row for row in predictions
    }

    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "box", "x1", "y1", "x2", "y2", "part", "confidence", "top5"])
        for row in predictions:
            x1, y1, x2, y2 = boxes_per_image[row["image_index"]][row["box_index"]]
            writer.writerow([images[row["image_index"]].name, row["box_index"], x1, y1, x2, y2,
                             row["part"], f"{row['confidence']:.4f}", row["top5"]])

    summary_rows = []
    annotated_images = []
    for image_index, path in enumerate(images):
        image = images_bgr[image_index]
        if image.size == 1:
            continue
        height, width = image.shape[:2]
        # Schrift/Groesse an die Bildaufloesung anpassen (Handyfotos sind 3000-4000 px breit)
        fs = max(1.0, min(height, width) / 900)
        font = 0.55 * fs
        thick = max(2, int(round(2 * fs)))
        counts: Counter = Counter()
        for box_index, (x1, y1, x2, y2) in enumerate(boxes_per_image[image_index]):
            row = labeled.get((image_index, box_index))
            if row is None:
                continue
            counts[row["part"]] += 1
            if row["confidence"] < args.min_conf:
                continue
            colour = PALETTE[row["class_id"] % len(PALETTE)]
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, thick)
            text = f"{row['part']} {row['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
            cv2.rectangle(image, (x1, max(0, y1 - th - int(8 * fs))), (x1 + tw + int(6 * fs), y1), colour, -1)
            cv2.putText(image, text, (x1 + int(3 * fs), max(th + int(4 * fs), y1 - int(6 * fs))),
                        cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick, cv2.LINE_AA)

        # Inventar oben links einblenden
        lines = [f"{part} x{count}" for part, count in counts.most_common(8)]
        for index, line in enumerate(lines):
            origin = (int(10 * fs), int(30 * fs) + index * int(26 * fs))
            cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * fs, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * fs, (60, 255, 60), thick, cv2.LINE_AA)

        out_path = args.out / "annotated" / path.name
        cv2.imwrite(str(out_path), image)
        annotated_images.append((path.name, image))
        for part, count in counts.items():
            summary_rows.append({"image": path.name, "part": part, "count": count})

    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "part", "count"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Kontaktabzug: 4 Bilder mit den meisten Boxen, gross genug zum Lesen
    counts_by_name = {path.name: len(boxes) for path, boxes in zip(images, boxes_per_image)}
    order = sorted(range(len(annotated_images)), key=lambda i: -counts_by_name[annotated_images[i][0]])
    tile_size, header = 900, 40
    tiles = []
    for index in order[:4]:
        image = annotated_images[index][1]
        height, width = image.shape[:2]
        scale = tile_size / max(height, width)
        tile = cv2.resize(image, (int(width * scale), int(height * scale)))
        canvas = np.full((tile_size + header, tile_size, 3), 255, np.uint8)
        canvas[header:header + tile.shape[0], :tile.shape[1]] = tile
        cv2.putText(canvas, annotated_images[index][0], (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(canvas)
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.full((tile_size + header, tile_size, 3), 255, np.uint8))
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        cv2.imwrite(str(args.out / "sheet.jpg"), np.vstack(rows))

    total = len(predictions)
    mean_conf = float(np.mean([row["confidence"] for row in predictions]))
    low = sum(1 for row in predictions if row["confidence"] < 0.25)
    print(f"\nBoxen klassifiziert: {total}  mittlere Top-1-Konfidenz: {mean_conf:.3f}")
    print(f"Boxen unter 0,25 Konfidenz: {low} ({100 * low / max(1, total):.0f} %)")

    if args.truth_json and args.truth_json.is_dir():
        evaluate_truth(args.truth_json, images, boxes_per_image, predictions, labeled, args.out)
    print(f"Vorhersagen: {args.out / 'predictions.csv'}")
    print(f"Bilder:      {args.out / 'annotated'}")
    print(f"Kontaktabzug:{args.out / 'sheet.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
