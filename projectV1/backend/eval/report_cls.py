#!/usr/bin/env python3
"""Erzeugt aus einem Trainingslauf einen kompakten Report als Markdown und PNG.

Quellen:

    <run>/results.csv            Epochen-Verlauf (Ultralytics)
    <run>/args.yaml              Trainingskonfiguration
    <run>/eval/metrics_*.json    Ausgabe von eval_cls.py
    <run>/eval/per_class_*.csv   Ausgabe von eval_cls.py

Aufruf:

    python projectV1/backend/eval/report_cls.py \
        --run projectV1/runs/cls_yolo26m_clean --split test

Ausgabe:

    <run>/report.md   Tabellen zum Einfügen in README/Issues
    <run>/report.png  dieselben Tabellen als Bild
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
        return rows


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_duration(seconds: float | None) -> str:
    if not seconds:
        return "–"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min {sec:02d} s"


def history_rows(results: list[dict]) -> list[list[str]]:
    rows = []
    for row in results:
        epoch = as_float(row.get("epoch", ""))
        loss = as_float(row.get("train/loss", ""))
        top1 = as_float(row.get("metrics/accuracy_top1", ""))
        top5 = as_float(row.get("metrics/accuracy_top5", ""))
        lr = as_float(row.get("lr/pg0", ""))
        time_s = as_float(row.get("time", ""))
        rows.append([
            "–" if epoch is None else f"{int(epoch)}",
            "–" if loss is None else f"{loss:.3f}",
            "–" if top1 is None else f"{top1 * 100:.1f} %",
            "–" if top5 is None else f"{top5 * 100:.1f} %",
            "–" if lr is None else f"{lr:.2e}",
            fmt_duration(time_s),
        ])
    return rows


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def trim_white(img: Image.Image, padding: int = 10) -> Image.Image:
    """Schneidet weisse Raender weg (matplotlib laesst sonst grosse Luecken)."""
    gray = img.convert("L")
    mask = gray.point(lambda value: 0 if value > 250 else 255)
    bbox = mask.getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    return img.crop((
        max(0, left - padding), max(0, top - padding),
        min(img.width, right + padding), min(img.height, bottom + padding),
    ))


def table_image(title: str, header: list[str], rows: list[list[str]], font_size: int = 11):
    """Rendert eine Tabelle als eigenstaendiges PNG (automatisch zugeschnitten)."""
    buffer = io.BytesIO()
    if not rows:
        rows = [["–"] * len(header)]
    fig = plt.figure(figsize=(7.5, 0.38 * len(rows) + 0.9), dpi=150)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=8)
    table = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.5)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if row == 0:
            cell.set_facecolor("#2f3b52")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f6fa")
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return trim_white(Image.open(buffer).convert("RGB"))


def text_image(text: str, size: int = 19, weight: str = "bold"):
    """Rendert eine Titelzeile als PNG."""
    buffer = io.BytesIO()
    fig = plt.figure(figsize=(9, 0.6), dpi=150)
    fig.text(0.0, 0.4, text, fontsize=size, fontweight=weight, color="#1c2434")
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return trim_white(Image.open(buffer).convert("RGB"), padding=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Trainings-Report als Markdown und PNG.")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run
    if not run.is_dir():
        print(f"Lauf nicht gefunden: {run}")
        return 1

    args_yaml = {}
    if (run / "args.yaml").is_file():
        args_yaml = yaml.safe_load((run / "args.yaml").read_text()) or {}

    results = read_csv(run / "results.csv")
    metrics = {}
    per_class = []
    metrics_path = run / "eval" / f"metrics_{args.split}.json"
    per_class_path = run / "eval" / f"per_class_{args.split}.csv"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
    if per_class_path.is_file():
        per_class = read_csv(per_class_path)
        for row in per_class:
            row["top1_acc"] = float(row["top1_acc"])
            row["top5_acc"] = float(row["top5_acc"])
            row["images"] = int(row["images"])

    hist = history_rows(results)
    last = results[-1] if results else {}
    total_time = as_float(last.get("time", "")) if last else None

    overview = [
        ["Modell", str(args_yaml.get("model", "–"))],
        ["Klassen", str(metrics.get("classes", args_yaml.get("nc", "–")))],
        ["Epochen geplant / gelaufen", f"{args_yaml.get('epochs', '–')} / {len(results) or '–'}"],
        ["Batch / Auflösung", f"{args_yaml.get('batch', '–')} / {args_yaml.get('imgsz', '–')} px"],
        ["Trainingszeit", fmt_duration(total_time)],
    ]

    data_stats = {}
    data_dir = args_yaml.get("data")
    if data_dir and (Path(data_dir) / "dataset_stats.json").is_file():
        data_stats = json.loads((Path(data_dir) / "dataset_stats.json").read_text())
        split_counts = data_stats.get("split_counts", {})
        overview.insert(2, ["Bilder train / val / test", (
            f"{split_counts.get('train', '–')} / {split_counts.get('val', '–')} / "
            f"{split_counts.get('test', '–')}")])

    summary_rows = []
    if metrics:
        summary_rows = [
            ["Top-1", f"{metrics['top1'] * 100:.1f} %"],
            ["Top-5", f"{metrics['top5'] * 100:.1f} %"],
            ["Mittlere Klassen-Top-1", f"{metrics['mean_class_top1'] * 100:.1f} %"],
            ["Bilder im Test", str(metrics["images"])],
        ]

    worst = sorted(per_class, key=lambda r: (r["top1_acc"], -r["images"]))[:10]
    best = sorted(per_class, key=lambda r: (-r["top1_acc"], -r["images"]))[:10]

    def class_rows(rows):
        return [[r["class"], str(r["images"]), f"{r['top1_acc'] * 100:.0f} %",
                 f"{r['top5_acc'] * 100:.0f} %"] for r in rows]

    confusions = metrics.get("top_confusions", [])[:10]
    conf_rows = [[c["true"], c["pred"], str(c["count"])] for c in confusions]

    header_hist = ["Epoche", "Loss", "Top-1", "Top-5", "LR", "Zeit"]
    header_cls = ["Klasse", "Bilder", "Top-1", "Top-5"]
    header_conf = ["Echt", "Vorhergesagt", "Anzahl"]

    md = [
        f"# Trainingsreport `{run.name}`",
        "",
        f"Split: **{args.split}**",
        "",
        "## Konfiguration",
        "",
        markdown_table(["Feld", "Wert"], overview),
        "",
        "## Testergebnis",
        "",
        markdown_table(["Kennzahl", "Wert"], summary_rows) if summary_rows else "_Noch keine Evaluation vorhanden._",
        "",
        "## Trainingsverlauf",
        "",
        markdown_table(header_hist, hist) if hist else "_Noch keine `results.csv`._",
        "",
        "## Schlechteste Klassen",
        "",
        markdown_table(header_cls, class_rows(worst)) if worst else "_keine Daten_",
        "",
        "## Beste Klassen",
        "",
        markdown_table(header_cls, class_rows(best)) if best else "_keine Daten_",
        "",
        "## Häufigste Verwechslungen",
        "",
        markdown_table(header_conf, conf_rows) if conf_rows else "_keine Daten_",
        "",
    ]

    out = args.out or run
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")

    # PNG: jede Tabelle als eigenes Bild rendern und vertikal zusammensetzen.
    # (Ein einzelnes GridSpec-Figure mit mehreren Tabellen ueberlappt, weil
    # matplotlib-Tabellen ihre Hoehe selbst bestimmen.)
    images = [text_image(f"Klassifikation {run.name} – {args.split}", size=19, weight="bold")]
    images.append(table_image("Konfiguration", ["Feld", "Wert"], overview))
    images.append(table_image(f"Ergebnis ({args.split})", ["Kennzahl", "Wert"], summary_rows))
    if hist:
        images.append(table_image("Trainingsverlauf", header_hist, hist[-12:]))
    if worst:
        images.append(table_image("Schlechteste 10 Klassen", header_cls, class_rows(worst)))
        images.append(table_image("Beste 10 Klassen", header_cls, class_rows(best)))
    if conf_rows:
        images.append(table_image("Häufigste Verwechslungen", header_conf, conf_rows))

    gap = 22
    width = max(img.width for img in images) + 40
    height = sum(img.height for img in images) + gap * (len(images) + 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = gap
    for img in images:
        canvas.paste(img, ((width - img.width) // 2, y))
        y += img.height + gap
    canvas.save(out / "report.png")

    print(f"Report: {out / 'report.md'}")
    print(f"Bild:   {out / 'report.png'}")
    print()
    print(markdown_table(["Kennzahl", "Wert"], summary_rows) if summary_rows
          else "_Noch keine Evaluation vorhanden (eval_cls.py zuerst laufen lassen)._")
    if per_class:
        print()
        print(markdown_table(header_cls, class_rows(worst)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
