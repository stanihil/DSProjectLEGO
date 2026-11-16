#!/usr/bin/env python3
"""Erzeugt Kontaktabzuege fuer Label-Konflikt-Paare aus ``conflicts.csv``.

Jedes Paar wird als zwei Bilder nebeneinander gezeigt, darueber Klasse und
dHash-Abstand. So laesst sich visuell entscheiden, ob es ein echter
Doppel-Label-Fall (dasselbe Teil in zwei Ordnern) oder nur eine grobe
dHash-Kollision zweier aehnlicher Teile ist.

Aufruf:

    python projectV1/backend/dataset/make_conflict_sheets.py \
        --conflicts projectV1/runs/audit_05/conflicts.csv \
        --out projectV1/runs/audit_05/sheets \
        --per-sheet 12 --max-distance 2
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL = 220
LABEL_H = 34


def load_font(size: int = 16) -> ImageFont.ImageFont:
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(path: str, box: int = CELL) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((box, box), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (box, box), "white")
        canvas.paste(img, ((box - img.width) // 2, (box - img.height) // 2))
        return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conflicts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--max-distance", type=int, default=99)
    ap.add_argument("--only-classes", default="", help="Komma-Liste: nur Paare mit einer dieser Klassen")
    ap.add_argument("--sort", choices=("distance", "class"), default="distance")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.conflicts.open(encoding="utf-8"))
            if int(r["distance"]) <= args.max_distance]
    wanted = {c.strip() for c in args.only_classes.split(",") if c.strip()}
    if wanted:
        rows = [r for r in rows if r["class"] in wanted or r["other_class"] in wanted]
    if args.sort == "class":
        rows.sort(key=lambda r: (r["class"], r["other_class"], int(r["distance"])))
    else:
        rows.sort(key=lambda r: (int(r["distance"]), r["class"], r["other_class"]))

    args.out.mkdir(parents=True, exist_ok=True)
    font = load_font(15)
    small = load_font(13)
    sheets = math.ceil(len(rows) / args.per_sheet)
    for index in range(sheets):
        chunk = rows[index * args.per_sheet:(index + 1) * args.per_sheet]
        cols = 2
        rows_count = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (cols * 2 * CELL, rows_count * (CELL + LABEL_H)), "white")
        draw = ImageDraw.Draw(sheet)
        for position, row in enumerate(chunk):
            col = position % cols
            line = position // cols
            x = col * 2 * CELL
            y = line * (CELL + LABEL_H)
            try:
                sheet.paste(fit(row["path"]), (x, y + LABEL_H))
                sheet.paste(fit(row["other_path"]), (x + CELL, y + LABEL_H))
            except Exception as exc:  # pragma: no cover - defekte Datei
                draw.text((x + 6, y + LABEL_H + 6), f"ERR {exc}", fill="red", font=small)
            title = f"{row['class']}  vs  {row['other_class']}   d={row['distance']}"
            draw.text((x + 4, y + 4), title, fill="black", font=font)
            draw.text((x + 4, y + LABEL_H + 2), Path(row["path"]).name[:26], fill="#555555", font=small)
            draw.text((x + CELL + 4, y + LABEL_H + 2), Path(row["other_path"]).name[:26],
                      fill="#555555", font=small)
            draw.line((x + CELL, y, x + CELL, y + CELL + LABEL_H), fill="#cccccc")
        path = args.out / f"conflicts_{index:03d}.jpg"
        sheet.save(path, quality=88)
        print(path)
    print(f"{len(rows)} Paare -> {sheets} Sheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
