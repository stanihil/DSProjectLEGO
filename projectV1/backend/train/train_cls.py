#!/usr/bin/env python3
"""Trainiert ein YOLO-Klassifikationsmodell (z.B. ``yolo26s-cls``) auf einem
Ultralytics-Klassifikations-Datensatz.

Der Datensatz wird mit ``build_cls_dataset.py`` erzeugt und hat die Struktur::

    <data>/
      train/<klasse>/*.jpg
      val/<klasse>/*.jpg
      test/<klasse>/*.jpg     (optional)

Beispiele:
    # Smoke-Test: 3 Epochen, kleine Aufloesung
    python projectV1/backend/train/train_cls.py \
        --data projectV1/dataset/classification_smoke --epochs 3

    # volles Training
    python projectV1/backend/train/train_cls.py \
        --data projectV1/dataset/classification --epochs 40 --batch 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser(
        description="YOLO-Klassifikation trainieren.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", type=Path, default=Path("projectV1/dataset/classification"))
    ap.add_argument("--model", default="yolo26s-cls.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="mps", help="mps, cpu, 0 ...")
    ap.add_argument("--workers", type=int, default=0,
                    help="Ultralytics setzt auf MPS/CPU intern 0 Worker; mehr Worker sind auf M1 langsamer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--fraction", type=float, default=1.0, help="Anteil der Trainingsbilder (Smoke-Test)")
    ap.add_argument("--project", type=Path, default=Path("projectV1/runs"))
    ap.add_argument("--name", default="cls_yolo26s")
    ap.add_argument("--cache", default=False, help="False | True | ram | disk")
    ap.add_argument("--optimizer", default="AdamW", help="AdamW, SGD, auto ...")
    ap.add_argument("--lr0", type=float, default=1e-3, help="Start-Lernrate (auto waehlt sonst sehr kleine Werte)")
    ap.add_argument("--lrf", type=float, default=0.01, help="End-Lernrate als Faktor von lr0")
    ap.add_argument("--cos-lr", action="store_true", help="Cosine-LR-Schedule")
    ap.add_argument("--erasing", type=float, default=0.2)
    ap.add_argument("--hsv-s", type=float, default=0.4)
    ap.add_argument("--no-pretrained", action="store_true", help="ohne ImageNet-Gewichte starten")
    ap.add_argument("--exist-ok", action="store_true")
    args = ap.parse_args()

    for split in ("train", "val"):
        if not (args.data / split).is_dir():
            print(f"Fehlender Split: {args.data / split}")
            return 1

    stats_path = args.data / "dataset_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        print(f"Datensatz: {stats.get('classes')} Klassen, {stats.get('total_images')} Bilder "
              f"({stats.get('split_counts')})")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        fraction=args.fraction,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        cache=args.cache,
        pretrained=not args.no_pretrained,
        plots=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        # LEGO: keine Spiegelung, sonst werden gespiegelte Teile zu anderen Teilenummern.
        fliplr=0.0,
        flipud=0.0,
        erasing=args.erasing,
        hsv_s=args.hsv_s,
        auto_augment="randaugment",
    )

    save_dir = Path(getattr(results, "save_dir", args.project / args.name))
    metrics = getattr(results, "results_dict", None)
    if metrics:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "final_metrics.json").write_text(
            json.dumps({k: float(v) for k, v in metrics.items()}, indent=2)
        )
        print(f"\nErgebnisse: {save_dir}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
