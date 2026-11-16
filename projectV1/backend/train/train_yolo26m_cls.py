#!/usr/bin/env python3
"""Trainiert ``yolo26m-cls.pt`` fuer die LEGO-Teilklassifikation.

Der Lauf ist bewusst als *einmaliges*, sauberes Training angelegt:

* Start von den offiziellen ``yolo26m-cls.pt``-Gewichten (kein Fine-Tuning
  des alten ``best.pt``).
* Keine Spiegel-Augmentierung (``fliplr=0``, ``flipud=0``), weil gespiegelte
  LEGO-Teilepaare die Klassen ununterscheidbar machen.
* AdamW + Cosine-LR, leichte Farbe/Scale-Augmentierung.
* Getrennte ``val``-/``test``-Ordner; nach dem Training wird der beste
  Checkpoint zusaetzlich auf ``test`` ausgewertet.
* Epochenzeiten und Endmetriken landen in ``train_summary.json``.

Beispiel:

    source .venv/bin/activate
    python projectV1/backend/train/train_yolo26m_cls.py \
        --data projectV1/dataset/cls_clean --epochs 12 --batch 32

Das Skript bricht frueh ab, wenn Train/Val nicht dieselben Klassen haben oder
Klassen ohne Bilder existieren.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(root: Path) -> dict[str, int]:
    return {
        folder.name: sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
        for folder in sorted(p for p in root.iterdir() if p.is_dir())
    }


def check_dataset(data: Path) -> None:
    train_dir, val_dir = data / "train", data / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Datensatz unvollstaendig: {data} (train/val fehlen)")
    train, val = count_images(train_dir), count_images(val_dir)
    if not train:
        raise SystemExit("Keine Trainingsbilder gefunden")
    missing_val = sorted(set(train) - set(val))
    empty_train = sorted(name for name, count in train.items() if count == 0)
    if missing_val:
        raise SystemExit(f"Klassen ohne val-Ordner: {missing_val[:10]} (+{max(0, len(missing_val) - 10)})")
    if empty_train:
        raise SystemExit(f"Leere Trainingsklassen: {empty_train[:10]}")
    print(f"Datensatz: {data}")
    print(f"  Klassen: {len(train)}   train: {sum(train.values())}   val: {sum(val.values())}")
    print(f"  pro Klasse train: min {min(train.values())}, "
          f"median {sorted(train.values())[len(train) // 2]}, max {max(train.values())}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sauberes Training von yolo26m-cls.")
    ap.add_argument("--data", type=Path, default=Path("projectV1/dataset/cls_clean"))
    ap.add_argument("--model", default="yolo26m-cls.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32, help="an 8-GB-M1 angepasst")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", type=Path, default=Path("projectV1/runs"))
    ap.add_argument("--name", default="cls_yolo26m_clean")
    ap.add_argument("--exist-ok", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="1 Epoche, wenige Bilder – nur zum Pruefen der Pipeline")
    args = ap.parse_args()

    check_dataset(args.data)

    train_kwargs = dict(
        data=str(args.data),
        epochs=1 if args.quick else args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        dropout=args.dropout,
        seed=args.seed,
        deterministic=True,
        # LEGO-Teile sind chiral / haben eine feste Oberseite:
        fliplr=0.0,
        flipud=0.0,
        # milde, sinnvolle Augmentierung fuer Studio- und Handybilder
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        scale=0.3,
        translate=0.05,
        erasing=0.15,
        auto_augment=None,
        val=True,
        plots=True,
        save=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )
    if args.quick:
        train_kwargs["fraction"] = 0.1

    model = YOLO(args.model)
    started = time.time()
    print(f"Start: {time.strftime('%H:%M:%S')}  Modell={args.model}  "
          f"Epochen={train_kwargs['epochs']}  imgsz={args.imgsz}  batch={args.batch}")
    model.train(**train_kwargs)
    duration = time.time() - started

    run_dir = Path(model.trainer.save_dir)
    summary: dict = {
        "model": args.model,
        "data": str(args.data),
        "epochs": train_kwargs["epochs"],
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "duration_seconds": round(duration, 1),
        "duration_minutes": round(duration / 60, 2),
        "run_dir": str(run_dir),
    }

    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        with results_csv.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        times = [float(row["time"]) for row in rows if row.get("time")]
        epochs_times = [round(b - a, 1) for a, b in zip([0.0] + times, times)]
        summary["epoch_seconds"] = epochs_times
        if times:
            summary["seconds_per_epoch_mean"] = round(sum(epochs_times) / len(epochs_times), 1)
        summary["last_epoch"] = rows[-1] if rows else {}

    best = run_dir / "weights" / "best.pt"
    if best.is_file():
        summary["best_weights"] = str(best)
        trained = YOLO(str(best))
        val_metrics = trained.val(data=str(args.data), split="val",
                                  imgsz=args.imgsz, batch=args.batch,
                                  device=args.device, workers=args.workers)
        summary["val"] = {
            "top1": float(val_metrics.top1),
            "top5": float(val_metrics.top5),
        }
        if not args.skip_test and (args.data / "test").is_dir():
            test_metrics = trained.val(data=str(args.data), split="test",
                                       imgsz=args.imgsz, batch=args.batch,
                                       device=args.device, workers=args.workers)
            summary["test"] = {
                "top1": float(test_metrics.top1),
                "top5": float(test_metrics.top5),
            }

    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("\nZusammenfassung")
    print(f"  Dauer: {summary['duration_minutes']} min "
          f"({summary.get('seconds_per_epoch_mean', '?')} s/Epoche)")
    for split in ("val", "test"):
        if split in summary:
            print(f"  {split}: Top-1 {summary[split]['top1']:.4f}   "
                  f"Top-5 {summary[split]['top5']:.4f}")
    print(f"  Summary: {run_dir / 'train_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
