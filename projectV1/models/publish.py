#!/usr/bin/env python3
"""Veroeffentlicht einen Trainingslauf als klar benanntes Modell.

In jedem Lauf heisst die beste Datei ``weights/best.pt`` - man sieht ihr
nicht an, aus welchem Training sie stammt. Dieses Skript kopiert sie nach
``projectV1/models/`` und benennt sie nach Task, Datensatz, Architektur,
Bildgroesse und Datum:

    cls_legosteine-19_yolo26m-cls_224_2026-11-16.pt

Dazu kommt eine JSON-Datei mit denselben Angaben plus Metriken,
Basisgewichten, Git-Commit und SHA-256. Alle Modelle stehen zusaetzlich in
``models/manifest.json``.

    # Lauf veroeffentlichen
    python projectV1/models/publish.py --run projectV1/classification/runs/yolo26m_cls_legosteine

    # Nur ansehen, nichts schreiben
    python projectV1/models/publish.py --run ... --dry-run

    # Vorhandene Modelle auflisten
    python projectV1/models/publish.py --list
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MANIFEST = HERE / "manifest.json"

TASK_SLUG = {"classify": "cls", "detect": "det", "segment": "seg", "obb": "obb"}

# Welche Spalte entscheidet ueber das beste Modell?
BEST_METRIC = {
    "cls": "metrics/accuracy_top1",
    "det": "metrics/mAP50-95(B)",
    "seg": "metrics/mAP50-95(M)",
    "obb": "metrics/mAP50-95(B)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_args(run: Path) -> dict:
    """args.yaml ohne PyYAML-Zwang: flache ``key: value``-Zeilen reichen."""
    args_path = run / "args.yaml"
    if not args_path.is_file():
        return {}
    args: dict = {}
    for line in args_path.read_text().splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        args[key.strip()] = value.strip().strip("'\"")
    return args


def read_metrics(run: Path, task: str) -> dict:
    """Bestes Ergebnis aus results.csv (nicht die letzte Epoche)."""
    csv_path = run / "results.csv"
    if not csv_path.is_file():
        return {}
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return {}
    column = BEST_METRIC.get(task)
    if column not in rows[0]:
        return {}
    best = max(rows, key=lambda r: float(r[column]))

    def num(row: dict, key: str) -> float | None:
        try:
            return round(float(row[key]), 4)
        except (KeyError, ValueError):
            return None

    metrics = {
        "best_epoch": int(float(best["epoch"])),
        "epochs_run": int(float(rows[-1]["epoch"])),
        # results.csv fuehrt "time" kumulativ in Sekunden
        "train_minutes": round(float(rows[-1]["time"]) / 60, 1) if "time" in best else None,
    }
    if task == "cls":
        metrics["top1"] = num(best, "metrics/accuracy_top1")
        metrics["top5"] = num(best, "metrics/accuracy_top5")
    else:
        metrics["mAP50-95"] = num(best, column)
        metrics["mAP50"] = num(best, column.replace("50-95", "50"))
    return {k: v for k, v in metrics.items() if v is not None}


def read_classes(run: Path, args: dict) -> list[str]:
    """Klassenliste aus dem Datensatz, sonst aus den Gewichten."""
    data = Path(args.get("data", ""))
    if data.is_dir():
        classes_file = data / "classes.txt"
        if classes_file.is_file():
            return [c for c in classes_file.read_text().split() if c]
        names = sorted(p.name for p in data.iterdir() if p.is_dir() and p.name != "train")
        if names:
            return names
    try:
        from ultralytics import YOLO

        model = YOLO(str(run / "weights" / "best.pt"))
        if isinstance(model.names, dict):
            return [model.names[i] for i in range(len(model.names))]
        return list(model.names)
    except Exception:
        return []


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_manifest() -> dict:
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text())
    return {"updated_at": None, "artifacts": []}


def save_manifest(manifest: dict) -> None:
    manifest["artifacts"].sort(key=lambda e: e["published_at"])
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def artifact_name(task: str, dataset: str, arch: str, imgsz: int, day: str,
                  taken: set[str]) -> str:
    base = f"{TASK_SLUG.get(task, task)}_{dataset}_{arch}_{imgsz}_{day}"
    name, counter = base, 1
    while f"{name}.pt" in taken:
        counter += 1
        name = f"{base}-{counter}"
    return name


def publish(run: Path, dataset: str | None = None, dry_run: bool = False) -> dict:
    """Kopiert ``best.pt`` als sprechend benanntes Artefakt nach models/."""
    run = run.resolve()
    weights = run / "weights" / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"Kein weights/best.pt in {run}")

    args = read_args(run)
    task = args.get("task") or run.parent.parent.name
    if task == "classify":
        task = "cls"
    task = TASK_SLUG.get(task, task)

    arch = Path(args.get("model", "unknown")).stem or "unknown"
    imgsz = int(float(args.get("imgsz", 0) or 0))
    dataset = dataset or run.name
    day = datetime.fromtimestamp(weights.stat().st_mtime).strftime("%Y-%m-%d")

    digest = sha256(weights)
    manifest = load_manifest()
    existing = next((e for e in manifest["artifacts"] if e["sha256"] == digest), None)

    if existing:
        name = Path(existing["file"]).stem
        print(f"Schon veroeffentlicht: {existing['file']} (gleicher Inhalt)")
        target = HERE / existing["file"]
    else:
        taken = {e["file"] for e in manifest["artifacts"]}
        name = artifact_name(task, dataset, arch, imgsz, day, taken)
        target = HERE / f"{name}.pt"

    classes = read_classes(run, args)
    base = Path(args.get("model", ""))
    base_path = base if base.is_file() else REPO / base

    entry = {
        "file": target.name,
        "sha256": digest,
        "task": task,
        "dataset": dataset,
        "class_count": len(classes),
        "classes": classes,
        "arch": arch,
        "base_weights": base.name or None,
        "base_weights_sha256": sha256(base_path) if base_path.is_file() else None,
        "imgsz": imgsz,
        "epochs": int(float(args.get("epochs", 0) or 0)),
        "metrics": read_metrics(run, task),
        "run_dir": str(run.relative_to(REPO)) if run.is_relative_to(REPO) else str(run),
        "git_commit": git_commit(),
        "published_at": (existing or {}).get(
            "published_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
    }

    if dry_run:
        print(json.dumps(entry, indent=2, ensure_ascii=False))
        print(f"\n[dry-run] wuerde schreiben: {target}")
        return entry

    shutil.copy2(weights, target)
    target.with_suffix(".json").write_text(
        json.dumps(entry, indent=2, ensure_ascii=False) + "\n")

    others = [e for e in manifest["artifacts"] if e["sha256"] != digest]
    manifest["artifacts"] = others + [entry]
    save_manifest(manifest)

    print(f"Modell:  {target}")
    print(f"Metriken: {entry['metrics']}")
    print(f"Klassen: {entry['class_count']}  Lauf: {entry['run_dir']}")
    return entry


def list_artifacts() -> int:
    manifest = load_manifest()
    artifacts = manifest["artifacts"]
    if not artifacts:
        print("Noch nichts veroeffentlicht.")
        return 0
    print(f"{'Datei':<52} {'Kl.':>4}  {'Metrik':<22} Lauf")
    print("-" * 110)
    for entry in artifacts:
        metrics = entry.get("metrics") or {}
        if "top1" in metrics:
            metric = f"top1 {metrics['top1']:.3f} / top5 {metrics.get('top5', 0):.3f}"
        elif "mAP50-95" in metrics:
            metric = f"mAP50-95 {metrics['mAP50-95']:.3f}"
        else:
            metric = "-"
        print(f"{entry['file']:<52} {entry['class_count']:>4}  {metric:<22} {entry['run_dir']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Trainingslauf als benanntes Modell ablegen.")
    ap.add_argument("--run", type=Path, help="Laufordner mit weights/best.pt")
    ap.add_argument("--dataset", default=None,
                    help="Kurzname des Datensatzes, z. B. legosteine-19 (Default: Laufname)")
    ap.add_argument("--list", action="store_true", help="veroeffentlichte Modelle anzeigen")
    ap.add_argument("--dry-run", action="store_true", help="nur zeigen, nichts kopieren")
    args = ap.parse_args()

    if args.list:
        return list_artifacts()
    if not args.run:
        ap.error("--run oder --list angeben")
    publish(args.run, dataset=args.dataset, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
