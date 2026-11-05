"""Auto-Labeling: ein Modell zeichnet die Box, das Label ist der Ordnername.

Erwartete Struktur:
  <src>/2654/*.jpg
  <src>/3002/*.jpg
  ...                      (Ordnername = Teilenummer = Klasse)

Ausgabe (--out, Standard dataset_yolo/), direkt fuer Training mit yolo26m:
  images/train, images/val
  labels/train, labels/val   (YOLO-txt: <klasse> xc yc w h, normalisiert)
  classes.txt                (eine Teilenummer pro Zeile, Reihenfolge = Klassen-ID)
  data.yaml
  no_detection.txt           (Bilder ohne Box, nicht im Datensatz, ausser --fallback-full)
  suspect.txt                (Box vorhanden, aber unplausibel: zu klein/gross/unsicher)
  preview/ok, preview/suspect (Kontrollbilder mit Box + Label)

Pro Bild wird genau EINE Box gesetzt (die mit der hoechsten Confidence). Eine Box
gilt nur als gueltig, wenn Confidence und Flaechenanteil plausibel sind.

Beispiele:
  python auto_label_yolo.py --src Legosteine_comp
  python auto_label_yolo.py --src Legosteine_comp --model yolov8l-worldv2.pt --prompts "lego brick"
  python auto_label_yolo.py --src Legosteine_comp --accept-conf 0.4 --min-area 0.05 --preview 5
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

EXT = {".jpg", ".jpeg", ".png"}


def load_model(name: str, prompts: list[str]):
    """COCO-, YOLO-World- und YOLOE-Modelle laden."""
    low = name.lower()
    if "yoloe" in low:
        from ultralytics import YOLOE
        model = YOLOE(name)
        if "-pf" not in low:
            model.set_classes(prompts, model.get_text_pe(prompts))
        return model
    model = YOLO(name)
    if "world" in low:
        model.set_classes(prompts)
    return model


def save_preview(path: Path, boxes_xyxy, label: str, out_file: Path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    d = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in boxes_xyxy:
        d.rectangle([x1, y1, x2, y2], outline="red", width=max(2, im.width // 200))
        d.text((x1 + 4, y1 + 4), label, fill="red")
    im.thumbnail((800, 800))
    im.save(out_file)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Ordner mit Teilenummer-Unterordnern")
    ap.add_argument("--out", type=Path, default=Path("dataset_yolo"))
    ap.add_argument("--model", default="yolov8m-worldv2.pt", help="Modell fuer die Boxen")
    ap.add_argument("--prompts", nargs="+", default=["lego brick"], help="nur World/YOLOE-Modelle")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--accept-conf", type=float, default=0.25,
                    help="Box gilt nur ab dieser Confidence als sicher (darunter: verdaechtig)")
    ap.add_argument("--min-area", type=float, default=0.02,
                    help="Mindestflaeche der Box als Anteil des Bildes (kleiner: verdaechtig)")
    ap.add_argument("--max-area", type=float, default=0.95,
                    help="Maximalflaeche der Box als Anteil des Bildes (groesser: verdaechtig)")
    ap.add_argument("--keep-suspect", action="store_true",
                    help="verdaechtige Boxen trotzdem in den Datensatz uebernehmen")
    ap.add_argument("--fallback-full", action="store_true",
                    help="Bild ohne Erkennung: Box ueber das ganze Bild statt ueberspringen")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview", type=int, default=3, help="Kontrollbilder pro Klasse (0 = aus)")
    ap.add_argument("--overwrite", action="store_true", help="vorhandenen --out-Ordner loeschen")
    args = ap.parse_args()

    # Klassen und Bilder einsammeln
    items = []  # (klasse, pfad)
    for folder in sorted(p for p in args.src.iterdir() if p.is_dir()):
        imgs = sorted(f for f in folder.rglob("*") if f.suffix.lower() in EXT)
        items += [(folder.name, f) for f in imgs]
    if not items:
        raise SystemExit(f"Keine Bilder in {args.src} gefunden.")
    classes = sorted({c for c, _ in items})
    cid = {c: i for i, c in enumerate(classes)}
    print(f"{len(classes)} Klassen, {len(items)} Bilder, Modell: {args.model}")

    if args.out.exists() and any(args.out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.out} existiert schon. --overwrite zum Ersetzen oder anderen --out waehlen.")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    ok_dir, sus_dir = args.out / "preview" / "ok", args.out / "preview" / "suspect"
    if args.preview:
        ok_dir.mkdir(parents=True)
        sus_dir.mkdir(parents=True)

    # Boxen vorhersagen (Klasse des Modells wird ignoriert)
    model = load_model(args.model, args.prompts)
    labeled = defaultdict(list)   # klasse -> [(pfad, [(xc, yc, w, h), ...])]
    missing = []   # keine Box gefunden
    suspect = []   # (pfad, grund): Box gefunden, aber unplausibel
    stats = defaultdict(lambda: {"n": 0, "sus": 0, "conf": []})
    shown = defaultdict(int)

    # max_det=1: pro Bild genau EINE Box (die mit der hoechsten Confidence)
    stream = model.predict(source=[str(p) for _, p in items], stream=True, conf=args.conf,
                           imgsz=args.imgsz, device=args.device, max_det=1, verbose=False)
    for k, ((cls, path), r) in enumerate(zip(items, stream), 1):
        s = stats[cls]
        s["n"] += 1
        if k % 50 == 0:
            print(f"  {k}/{len(items)} Bilder")

        if len(r.boxes) == 0:
            if args.fallback_full:
                labeled[cls].append((path, [(0.5, 0.5, 1.0, 1.0)]))
            else:
                missing.append(path)
            continue

        i = max(range(len(r.boxes)), key=lambda j: float(r.boxes.conf[j]))
        conf = float(r.boxes.conf[i])
        xc, yc, w, h = r.boxes.xywhn[i].tolist()
        xyxy = [tuple(r.boxes.xyxy[i].tolist())]
        area = w * h
        s["conf"].append(conf)

        # Plausibilitaet: sitzt wirklich eine sinnvolle Box um das Teil?
        reasons = []
        if conf < args.accept_conf:
            reasons.append(f"niedrige Confidence {conf:.2f}")
        if area < args.min_area:
            reasons.append(f"Box zu klein ({area:.1%} des Bildes)")
        if area > args.max_area:
            reasons.append(f"Box fast das ganze Bild ({area:.1%})")

        if reasons:
            s["sus"] += 1
            suspect.append((path, "; ".join(reasons)))
            if args.preview:
                save_preview(path, xyxy, cls, sus_dir / f"{cls}_{path.stem}.jpg")
            if not args.keep_suspect:
                continue
        elif args.preview and shown[cls] < args.preview:
            shown[cls] += 1
            save_preview(path, xyxy, cls, ok_dir / f"{cls}_{path.stem}.jpg")
        labeled[cls].append((path, [(xc, yc, w, h)]))

    # Split pro Klasse und Export
    rng = random.Random(args.seed)
    used = set()
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
    for cls, entries in labeled.items():
        rng.shuffle(entries)
        n_val = min(max(1, round(len(entries) * args.val_ratio)), len(entries) - 1) if len(entries) > 1 else 0
        for idx, (path, boxes) in enumerate(entries):
            split = "val" if idx < n_val else "train"
            stem = f"{cls}__{path.stem}"
            n = 1
            while (split, stem) in used:
                stem, n = f"{cls}__{path.stem}_{n}", n + 1
            used.add((split, stem))
            shutil.copy2(path, args.out / "images" / split / f"{stem}{path.suffix.lower()}")
            lines = [f"{cid[cls]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for xc, yc, w, h in boxes]
            (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    (args.out / "classes.txt").write_text("\n".join(classes) + "\n")
    cfg = {"path": str(args.out.resolve()), "train": "images/train", "val": "images/val",
           "names": {i: c for c, i in cid.items()}}
    (args.out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if missing:
        (args.out / "no_detection.txt").write_text("\n".join(str(p) for p in missing) + "\n")
    if suspect:
        (args.out / "suspect.txt").write_text("\n".join(f"{p}\t{why}" for p, why in suspect) + "\n")

    # Zusammenfassung
    print(f"\n{'Klasse':<10}{'Bilder':>7}{'im Set':>8}{'verdaechtig':>13}{'mean conf':>11}")
    for cls in classes:
        s = stats[cls]
        mc = sum(s["conf"]) / len(s["conf"]) if s["conf"] else 0.0
        print(f"{cls:<10}{s['n']:>7}{len(labeled[cls]):>8}{s['sus']:>13}{mc:>11.2f}")
    total_in = sum(len(v) for v in labeled.values())
    print(f"\nIm Datensatz:   {total_in}/{len(items)} Bilder (je genau 1 Box)")
    print(f"Ohne Erkennung: {len(missing)}" + (f"  -> {args.out / 'no_detection.txt'}" if missing else ""))
    print(f"Verdaechtig:    {len(suspect)}" + (f"  -> {args.out / 'suspect.txt'}, Bilder in preview/suspect/" if suspect else ""))
    print(f"Fertig: {args.out}/  (data.yaml, classes.txt, images/, labels/)")


if __name__ == "__main__":
    main()