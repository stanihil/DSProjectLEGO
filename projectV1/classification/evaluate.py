#!/usr/bin/env python3
"""Klassifikations-Split auswerten: gesamt, pro Klasse, pro Herkunft.

``test.py --split`` liefert nur Top-1/Top-5 über den ganzen Split. Hier kommt
dazu, welche Klasse wie oft daneben liegt und ob die Bilder aus dem Studio
(``Legosteine_comp``) oder von Handyfotos (``cuttet_multi_lego_yolo``) stammen
- die beiden Quellen sind unterschiedlich schwer.

    python projectV1/classification/evaluate.py \
        --weights latest --data projectV1/classification/data_merged --split test

    # Vorheriges Modell auf demselben Split, zum Vergleich
    python projectV1/classification/evaluate.py \
        --weights projectV1/models/cls_legosteine-19_yolo26s-cls_224_2026-11-16.pt \
        --data projectV1/classification/data_merged --split test

    # Ergebnis als JSON ablegen
    ... --json runs/eval_merged_test.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_weights(weights: Path) -> Path:
    if str(weights).lower() != "latest":
        return weights
    manifest = HERE.parent / "models" / "manifest.json"
    if not manifest.is_file():
        raise SystemExit("Kein models/manifest.json – noch nichts veröffentlicht.")
    artifacts = json.loads(manifest.read_text())["artifacts"]
    return HERE.parent / "models" / artifacts[-1]["file"]


def origin_of(stem: str, studio_stems: set[str]) -> str:
    """Grobe Herkunft: Studio-Ganzbilder heissen ``IMG_*_comp``, Handyfotos
    tragen den Aufnahmezeitstempel. Massgeblich ist das Manifest."""
    return "studio" if stem in studio_stems else "phone"


def main() -> int:
    ap = argparse.ArgumentParser(description="Split auswerten (gesamt/pro Klasse/Herkunft).")
    ap.add_argument("--weights", type=Path, default=Path("latest"))
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--json", type=Path, default=None, help="Ergebnis als JSON speichern")
    args = ap.parse_args()

    weights = resolve_weights(args.weights)
    data = args.data.resolve()
    split_dir = data / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"Kein {split_dir}")

    # Herkunft je Quellfoto aus dem Manifest. Dateinamen sind
    # "<crop|full>_<quellstamm>[_<index>]_<repeat>"; die Ziffernsuffixe werden
    # abgeschnitten, bis ein bekannter Quellstamm übrig bleibt.
    studio_stems: set[str] = set()
    known_stems: set[str] = set()
    manifest = data / "manifest.csv"
    if manifest.is_file():
        for row in csv.DictReader(manifest.open()):
            stem = Path(row["source"]).stem
            known_stems.add(stem)
            if "Legosteine_comp" in Path(row["source"]).as_posix():
                studio_stems.add(stem)

    files = sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if not files:
        raise SystemExit(f"Keine Bilder in {split_dir}")

    model = YOLO(str(weights))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print(f"Modell:  {weights}  ({len(names)} Klassen)")
    print(f"Daten:   {split_dir}  ({len(files)} Bilder)")

    results = model.predict([str(f) for f in files], imgsz=args.imgsz,
                            device=args.device, verbose=False)

    per_class = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0})
    per_origin = defaultdict(lambda: {"n": 0, "top1": 0})
    hits = 0
    top5_hits = 0
    misses = []

    for path, result in zip(files, results):
        truth = path.parent.name
        probs = result.probs.data.cpu().numpy()
        order = np.argsort(probs)[::-1]
        predicted = names[int(order[0])]
        top5 = [names[int(i)] for i in order[:5]]
        ok = predicted == truth
        ok5 = truth in top5
        hits += ok
        top5_hits += ok5

        stem = path.stem
        for prefix in ("crop_", "full_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        while stem and stem not in known_stems:
            head, sep, tail = stem.rpartition("_")
            if not sep or not tail.isdigit():
                break
            stem = head
        origin = origin_of(stem, studio_stems)

        for key in (truth, "*"):
            per_class[key]["n"] += 1
            per_class[key]["top1"] += ok
            per_class[key]["top5"] += ok5
        per_origin[origin]["n"] += 1
        per_origin[origin]["top1"] += ok

        if not ok:
            misses.append({
                "file": path.relative_to(split_dir).as_posix(),
                "truth": truth,
                "prediction": predicted,
                "confidence": round(float(probs[order[0]]), 4),
            })

    n = len(files)
    print(f"\nTop-1 {hits / n:.3f}   Top-5 {top5_hits / n:.3f}   ({hits}/{n})")

    print(f"\n{'Herkunft':<10} {'Bilder':>6} {'Top-1':>7}")
    for origin, s in sorted(per_origin.items()):
        print(f"{origin:<10} {s['n']:>6} {s['top1'] / s['n']:>7.3f}")

    print(f"\n{'Klasse':>7} {'Bilder':>6} {'Top-1':>7}")
    for key in sorted(k for k in per_class if k != "*"):
        s = per_class[key]
        print(f"{key:>7} {s['n']:>6} {s['top1'] / s['n']:>7.3f}")

    if misses:
        print(f"\nFehler ({len(misses)}):")
        for m in misses:
            print(f"  {m['file']:<44} {m['truth']:>6} -> {m['prediction']:<6} {m['confidence']:.2f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "weights": str(weights),
            "data": str(data),
            "split": args.split,
            "images": n,
            "top1": round(hits / n, 4),
            "top5": round(top5_hits / n, 4),
            "per_origin": {k: {"n": v["n"], "top1": round(v["top1"] / v["n"], 4)}
                           for k, v in per_origin.items()},
            "per_class": {k: {"n": v["n"], "top1": round(v["top1"] / v["n"], 4)}
                          for k, v in per_class.items() if k != "*"},
            "misses": misses,
        }, indent=2, ensure_ascii=False) + "\n")
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
