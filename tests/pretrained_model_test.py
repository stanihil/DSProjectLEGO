"""Testet vortrainierte YOLO-Modelle OHNE Training auf wenigen Lego-Bildern.

Frage: Erkennt ein vortrainiertes (COCO-)Modell auf den Bildern ueberhaupt
etwas bzw. setzt es eine Box um den Stein? Klassennamen sind egal (COCO kennt
kein LEGO), wichtig ist: Wird der Stein lokalisiert?

Ausgabe (in --out, Standard: pretrained_test/):
  comparison.png   Zeilen = Bilder, Spalten = Modelle (zum Anschauen)
  annotated/       Einzelbilder mit Boxen pro Modell
  summary.csv      Kennzahlen pro Modell

Beispiele:
  python test_pretrained_yolo.py
  python test_pretrained_yolo.py --n 3 --conf 0.1
  python test_pretrained_yolo.py --images Legosteine_comp/3001/a.jpg Legosteine_comp/3002/b.jpg
  python test_pretrained_yolo.py --images bilder                      # alle Bilder im Ordner
  python test_pretrained_yolo.py --images "bilder/*.jpg"              # Glob-Muster
  python test_pretrained_yolo.py --models yolov8n.pt yolov8s-worldv2.pt --prompts "lego brick"
  python test_pretrained_yolo.py --models yolo26n.pt yoloe-26s-seg.pt --prompts "lego brick"
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

EXT = {".jpg", ".jpeg", ".png"}
DEFAULT_MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt",
]


def pick_images(data_dir: Path, n: int) -> list[Path]:
    """Je ein Bild aus n gleichmaessig verteilten Klassen (deterministisch)."""
    folders = sorted(p for p in data_dir.iterdir() if p.is_dir())
    step = max(1, len(folders) // n)
    picked = []
    for folder in folders[::step][:n]:
        imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXT)
        if imgs:
            picked.append(imgs[0])
    return picked


def expand_images(items: list[str]) -> list[Path]:
    """Dateien, Ordner (alle Bilder darin) und Glob-Muster -> Liste von Bildpfaden."""
    found = []
    for item in items:
        p = Path(item).expanduser()
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXT)
        elif p.is_file():
            found.append(p)
        else:
            matches = sorted(Path().glob(item))
            found += [m for m in matches if m.is_file() and m.suffix.lower() in EXT]
            if not matches:
                print(f"[WARNUNG] Nicht gefunden: {item}")
    return found


def load_model(name: str, prompts: list[str]):
    """Laedt COCO-, YOLO-World- und YOLOE-Modelle."""
    low = name.lower()
    if "yoloe" in low:
        from ultralytics import YOLOE  # braucht aktuelles ultralytics
        model = YOLOE(name)
        if "-pf" not in low:  # '-pf' = prompt-free, braucht keine Prompts
            model.set_classes(prompts, model.get_text_pe(prompts))
        return model
    model = YOLO(name)
    if "world" in low:
        model.set_classes(prompts)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", type=Path, help="eigene Bildpfade (sonst automatisch)")
    ap.add_argument("--dir", type=Path, default=Path("Legosteine_comp"))
    ap.add_argument("--n", type=int, default=5, help="Anzahl Bilder (1-5), wenn --images fehlt")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--prompts", nargs="+", default=["lego brick"], help="nur fuer *-world*-Modelle")
    ap.add_argument("--conf", type=float, default=0.05, help="niedrige Schwelle, damit man auch schwache Treffer sieht")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--out", type=Path, default=Path("pretrained_test"))
    args = ap.parse_args()

    if args.images:
        images = expand_images([str(p) for p in args.images])
        if not images:
            raise SystemExit(
                f"Keine Bilder gefunden in: {[str(p) for p in args.images]}\n"
                f"Aktuelles Verzeichnis: {Path.cwd()}\n"
                "Tipp: Ordner angeben (--images bilder) oder echte Dateinamen verwenden."
            )
    else:
        images = pick_images(args.dir, max(1, min(args.n, 5)))
        if not images:
            raise SystemExit(f"Keine Bilder in {args.dir} gefunden.")
    print("Bilder:", *[str(p) for p in images], sep="\n  ")

    ann_dir = args.out / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    tiles = {}   # (bild_idx, modell) -> PIL
    rows = []
    for name in args.models:
        print(f"\n=== {name} ===")
        try:
            model = load_model(name, args.prompts)
            model.predict(str(images[0]), conf=args.conf, imgsz=args.imgsz,
                          device=args.device, verbose=False)  # Warm-up
        except Exception as e:
            print(f"[FEHLER] {name}: {e}")
            continue

        stem = Path(name).stem
        (ann_dir / stem).mkdir(exist_ok=True)
        n_hit = n_dets = 0
        top_confs, times = [], []
        for i, img in enumerate(images):
            r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz,
                              device=args.device, verbose=False)[0]
            confs = r.boxes.conf.tolist()
            labels = [r.names[int(c)] for c in r.boxes.cls.tolist()]
            times.append(r.speed["inference"])
            n_dets += len(confs)
            if confs:
                n_hit += 1
                top_confs.append(max(confs))
            best = sorted(zip(confs, labels), reverse=True)[:3]
            txt = ", ".join(f"{l} {c:.2f}" for c, l in best) or "nichts"
            print(f"  {img.parent.name}/{img.name}: {len(confs)} Det. -> {txt}")

            plotted = Image.fromarray(r.plot()[..., ::-1])  # BGR -> RGB
            plotted.save(ann_dir / stem / f"{img.parent.name}_{img.stem}.jpg")
            plotted.thumbnail((args.tile, args.tile))
            tiles[(i, stem)] = plotted

        rows.append({
            "model": name,
            "images_with_detection": f"{n_hit}/{len(images)}",
            "total_detections": n_dets,
            "mean_top_conf": round(sum(top_confs) / len(top_confs), 3) if top_confs else 0.0,
            "inference_ms": round(sum(times) / len(times), 1),
        })

    if not rows:
        raise SystemExit("Kein Modell erfolgreich.")

    # Vergleichsbild: Zeilen = Bilder, Spalten = Modelle
    stems = [Path(r["model"]).stem for r in rows]
    T, H = args.tile, 24
    sheet = Image.new("RGB", (T * len(stems), H + T * len(images)), "white")
    d = ImageDraw.Draw(sheet)
    for c, s in enumerate(stems):
        d.text((c * T + 6, 6), s, fill="black")
        for i in range(len(images)):
            sheet.paste(tiles[(i, s)], (c * T, H + i * T))
    sheet.save(args.out / "comparison.png")

    cols = list(rows[0].keys())
    w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("\n" + " | ".join(c.ljust(w[c]) for c in cols))
    print("-+-".join("-" * w[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w[c]) for c in cols))

    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nFertig: {args.out / 'comparison.png'}  |  {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()