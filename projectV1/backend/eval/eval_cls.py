#!/usr/bin/env python3
"""Evaluiert ein trainiertes YOLO-Klassifikationsmodell.

Aufruf:

    python projectV1/backend/eval/eval_cls.py \
        --weights projectV1/runs/cls_yolo26m_clean/weights/best.pt \
        --data projectV1/dataset/cls_clean --split test

Ausgabe (neben dem Modellordner):

    metrics_<split>.json    top1, top5, pro-Klasse-Accuracy, haeufigste Verwechslungen
    per_class_<split>.csv    Klasse, Bilder, korrekt, Accuracy, Top-5-Rate

Die Auswertung laeuft ueber einen eigenen Predict-Loop. Ultralytics 8.4
entfernt ``model.validator``, deshalb wird ``model.val()`` nicht mehr benutzt.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Klassifikationsmodell auf einem Split evaluieren.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("projectV1/dataset/classification"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, default=None, help="Zielordner (Standard: neben den Gewichten)")
    return ap.parse_args()


def collect_samples(root: Path, class_ids: dict[str, int]) -> list[tuple[Path, int]]:
    """Alle Bilder eines Klassifikations-Splits mit Klassen-ID."""
    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_id = class_ids.get(class_dir.name)
        if class_id is None:
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXT:
                samples.append((path, class_id))
    return samples


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        print(f"Gewichte nicht gefunden: {args.weights}")
        return 1
    split_dir = args.data / args.split
    if not split_dir.is_dir():
        print(f"Split nicht gefunden: {split_dir}")
        return 1

    model = YOLO(str(args.weights))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    names = {int(key): str(value) for key, value in names.items()}
    class_ids = {name: index for index, name in names.items()}

    samples = collect_samples(split_dir, class_ids)
    if not samples:
        print(f"Keine Bilder in {split_dir}")
        return 1

    predictions: list[list[int]] = []
    targets: list[int] = []
    for start in range(0, len(samples), args.batch):
        chunk = samples[start:start + args.batch]
        results = model.predict(
            [str(path) for path, _ in chunk],
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            verbose=False,
        )
        for (_, target), result in zip(chunk, results):
            predictions.append([int(i) for i in result.probs.top5])
            targets.append(target)
        print(f"\r{len(predictions)}/{len(samples)}", end="", flush=True)
    print()

    counts_total = Counter(targets)
    counts_top1 = Counter()
    counts_top5 = Counter()
    confusions = Counter()
    for top5, target in zip(predictions, targets):
        if top5 and top5[0] == target:
            counts_top1[target] += 1
        if target in top5:
            counts_top5[target] += 1
        if top5 and top5[0] != target:
            confusions[(target, top5[0])] += 1

    per_class = []
    for class_id in range(len(names)):
        total = counts_total.get(class_id, 0)
        if total == 0:
            continue
        per_class.append({
            "class_id": class_id,
            "class": names[class_id],
            "images": total,
            "top1": counts_top1.get(class_id, 0),
            "top5": counts_top5.get(class_id, 0),
            "top1_acc": round(counts_top1.get(class_id, 0) / total, 4),
            "top5_acc": round(counts_top5.get(class_id, 0) / total, 4),
        })

    n = len(targets)
    top1 = sum(counts_top1.values()) / n
    top5 = sum(counts_top5.values()) / n
    metrics = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "classes": len(names),
        "images": n,
        "top1": round(top1, 4),
        "top5": round(top5, 4),
        "mean_class_top1": round(sum(row["top1_acc"] for row in per_class) / len(per_class), 4),
        "worst_classes": sorted(per_class, key=lambda row: row["top1_acc"])[:25],
        "top_confusions": [
            {"true": names[t], "pred": names[p], "count": c}
            for (t, p), c in confusions.most_common(25)
        ],
    }

    out_dir = args.out or (args.weights.parent.parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (out_dir / f"per_class_{args.split}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_class[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(per_class, key=lambda row: row["top1_acc"]))

    print(f"Split: {args.split}  Bilder: {n}  Klassen: {len(names)}")
    print(f"Top-1: {top1:.3f}   Top-5: {top5:.3f}   "
          f"mittlere Klassen-Top-1: {metrics['mean_class_top1']:.3f}")
    worst = metrics["worst_classes"]
    print(f"Schlechteste Klassen: " + ", ".join(f"{r['class']}={r['top1_acc']:.2f}" for r in worst[:5]))
    print(f"Ergebnisse: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
