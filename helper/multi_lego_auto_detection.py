import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

class LegoMultiDetector:
    def __init__(self, model_path='best.pt', default_class_id=0):
        """
        Detektor für mehrere LEGO-Steine pro Bild.
        """
        print(f"Lade Modell: {model_path}")
        self.model = YOLO(model_path)
        self.default_class_id = default_class_id
        print(f"✓ Modell geladen. Alle erkannten Boxen erhalten vorerst die Class-ID: {default_class_id}")

    def process_folder(self, folder_path, create_preview=False):
        target_dir = Path(folder_path)
        
        if not target_dir.exists() or not target_dir.is_dir():
            print(f"❌ Ungültiger Ordner: {folder_path}")
            return

        # Preview-Ordner erstellen, falls gewünscht
        preview_dir = None
        if create_preview:
            preview_dir = target_dir / "preview_multi"
            preview_dir.mkdir(exist_ok=True)
            print(f"📁 Preview-Bilder werden gespeichert in: {preview_dir}")

        # Alle Bilddateien finden
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [p for p in target_dir.iterdir() if p.suffix.lower() in image_extensions and p.name != "preview_multi"]
        
        print(f"\n🚀 Starte Erkennung in '{target_dir.name}' ({len(images)} Bilder)...")
        print("-" * 50)

        success_count = 0
        empty_count = 0

        for img_path in sorted(images):
            label_path = img_path.with_suffix('.txt')
            
            # 1. Detection durchführen
            results = self.model(img_path, conf=0.25, iou=0.5)
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                print(f"   ⚪ {img_path.name}: Keine Steine erkannt")
                empty_count += 1
                continue

            # 2. Alle Boxen extrahieren
            boxes = results[0].boxes.xyxy.cpu().numpy()
            h, w = results[0].orig_shape
            
            # 3. Label-Datei mit ALLEN Boxen schreiben
            with open(label_path, 'w') as f:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    
                    # YOLO-Normalisierung
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # Class-ID ist hier der Platzhalter (z.B. 0)
                    f.write(f"{self.default_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            num_boxes = len(boxes)
            print(f"   ✅ {img_path.name}: {num_boxes} Box(en) gespeichert")
            success_count += 1

            # 4. Optional: Preview-Bild mit allen Boxen zeichnen
            if create_preview:
                img = cv2.imread(str(img_path))
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    # Grüne Box, Dicke 2
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Text mit Class-ID
                    cv2.putText(img, str(self.default_class_id), (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                preview_path = preview_dir / f"preview_{img_path.name}"
                cv2.imwrite(str(preview_path), img)

        print("-" * 50)
        print(f"✅ Fertig! {success_count} Bilder mit Labels, {empty_count} Bilder ohne Erkennung.")
        if create_preview:
            print(f"👀 Prüfe die Ergebnisse im Ordner: {preview_dir}")


if __name__ == '__main__':
    # Argumente für die Befehlszeile einrichten
    parser = argparse.ArgumentParser(description="Erkennt ALLE LEGO-Steine in einem Ordner.")
    parser.add_argument("--folder", type=str, default=None, 
                        help="Pfad zum Ordner mit den Bildern (z.B. ./meine_bilder)")
    parser.add_argument("--model", type=str, default="best.pt", 
                        help="Pfad zum YOLO-Modell (Standard: best.pt)")
    parser.add_argument("--class-id", type=int, default=0, 
                        help="Temporäre Class-ID für die Labels (Standard: 0)")
    parser.add_argument("--preview", action="store_true", 
                        help="Erstellt einen 'preview_multi' Ordner mit Bildern und eingezeichneten Boxen")

    args = parser.parse_args()

    # Wenn kein Ordner angegeben wurde, nachfragen oder aktuellen Ordner nutzen
    if args.folder is None:
        print("⚠ Kein Ordner angegeben. Nutze aktuellen Ordner '.'")
        target_folder = "."
    else:
        target_folder = args.folder

    # Detektor starten
    detector = LegoMultiDetector(model_path=args.model, default_class_id=args.class_id)
    detector.process_folder(folder_path=target_folder, create_preview=args.preview)
