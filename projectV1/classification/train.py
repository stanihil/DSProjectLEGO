#!/usr/bin/env python3
"""Training des LEGO-Teilklassifikators - bewusst genauso einfach wie
``projectV1/backend/object_detection/train.py``.

Ein Modell laden, ein ``model.train(...)``, fertig. Datensatz, Klassenliste
und Laeufe liegen alle in ``projectV1/classification/``.

Beispiel:

    python projectV1/classification/train.py
    python projectV1/classification/train.py --epochs 60 --model yolo26m-cls.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="LEGO-Klassifikator trainieren.")
    ap.add_argument("--model", default="yolo26m-cls.pt",
                    help="Basismodell (Pfad oder Ultralytics-Name)")
    ap.add_argument("--data", type=Path, default=HERE / "data",
                    help="Ordner mit train/ val/ test/ (keine YAML)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--patience", type=int, default=20,
                    help="Early Stopping (0 = aus)")
    ap.add_argument("--name", default="yolo26m_cls_legosteine")
    ap.add_argument("--project", type=Path, default=HERE / "runs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="legosteine-19",
                    help="Kurzname fuer den Dateinamen des Modells")
    ap.add_argument("--no-publish", dest="publish", action="store_false",
                    help="nicht nach projectV1/models/ kopieren")
    args = ap.parse_args()

    data = args.data.resolve()
    if not (data / "train").is_dir():
        print(f"Kein train/ in {data} - zuerst build_dataset.py ausfuehren.")
        return 1

    model_path = Path(args.model)
    if not model_path.exists() and (REPO / model_path).exists():
        model_path = REPO / model_path

    model = YOLO(str(model_path))
    model.train(
        data=str(data),          # Classification: Ordner, keine data.yaml
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=args.name,
        workers=args.workers,
        lr0=args.lr0,
        cos_lr=True,
        optimizer="AdamW",
        # LEGO hat chirale Teilepaare: gespiegelte Steine sind andere Teile.
        fliplr=0.0,
        flipud=0.0,
        seed=args.seed,
    )
    print(f"\nLauf: {args.project.resolve() / args.name}")
    print(f"Best: {args.project.resolve() / args.name / 'weights' / 'best.pt'}")

    if args.publish:
        sys.path.insert(0, str(HERE.parent / "models"))
        from publish import publish

        publish(args.project.resolve() / args.name, dataset=args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
