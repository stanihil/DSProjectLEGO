import cv2
from pathlib import Path
import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"

def visualize_labels(base_folder='Legosteine_comp'):
    """
    Zeigt alle Bilder mit ihren YOLO-Labels und Bounding Boxes an.
    """
    base_path = Path(base_folder)
    
    if not base_path.exists():
        print(f"❌ Ordner nicht gefunden: {base_folder}")
        return
    
    # classes.txt laden (für die Anzeige der Teilenummern)
    classes_file = Path('classes.txt')
    class_names = []
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"✓ {len(class_names)} Klassen geladen aus classes.txt")
    else:
        print("⚠ classes.txt nicht gefunden, zeige nur Class-IDs")
    
    # Alle Bilder finden
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    all_images = []
    
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir():
            for ext in image_extensions:
                all_images.extend(sorted(folder.glob(f'*{ext}')))
    
    print(f"\n📊 {len(all_images)} Bilder gefunden")
    print("=" * 50)
    print("Steuerung:")
    print("  [n] oder [Leertaste] → Nächstes Bild")
    print("  [p] → Vorheriges Bild")
    print("  [d] → Label-Datei löschen (falsche Detection)")
    print("  [q] → Beenden")
    print("=" * 50)
    
    current_idx = 0
    
    while current_idx < len(all_images):
        img_path = all_images[current_idx]
        label_path = img_path.with_suffix('.txt')
        
        # Bild laden
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠ Konnte nicht laden: {img_path}")
            current_idx += 1
            continue
        
        h, w = img.shape[:2]
        num_boxes = 0
        
        # Labels laden und Boxen zeichnen
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Zurück zu Pixel-Koordinaten
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Farbe basierend auf Class-ID (unterschiedliche Farben)
                colors = [
                    (0, 255, 0),    # Grün
                    (255, 0, 0),    # Blau
                    (0, 0, 255),    # Rot
                    (255, 255, 0),  # Cyan
                    (255, 0, 255),  # Magenta
                    (0, 255, 255),  # Gelb
                    (128, 0, 255),  # Orange
                    (255, 128, 0),  # Pink
                ]
                color = colors[class_id % len(colors)]
                
                # Box zeichnen (dicker für bessere Sichtbarkeit)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # Label-Text
                if class_id < len(class_names):
                    label_text = f"{class_id}: {class_names[class_id]}"
                else:
                    label_text = f"{class_id}"
                
                # Text-Hintergrund
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    img, 
                    (x1, y1 - text_h - 10), 
                    (x1 + text_w + 5, y1), 
                    color, 
                    -1
                )
                
                # Text (schwarz auf farbigem Hintergrund)
                cv2.putText(
                    img, label_text, 
                    (x1 + 2, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                )
                
                num_boxes += 1
        
        # Info-Text oben im Bild
        info_text = f"{current_idx + 1}/{len(all_images)} | {img_path.parent.name} | {num_boxes} Boxen"
        cv2.putText(
            img, info_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
        cv2.putText(
            img, info_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
        )
        
        # Bild anzeigen
        window_name = f"LEGO Labels - {img_path.name}"
        cv2.imshow(window_name, img)
        
        # Auf Tastendruck warten
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' oder ESC
            print("\n👋 Beendet")
            break
        elif key == ord('n') or key == ord(' '):  # 'n' oder Leertaste
            current_idx += 1
        elif key == ord('p'):  # 'p' für zurück
            current_idx = max(0, current_idx - 1)
        elif key == ord('d'):  # 'd' zum Löschen der Label-Datei
            if label_path.exists():
                label_path.unlink()
                print(f"  🗑 Gelöscht: {label_path.name}")
            current_idx += 1
        else:
            current_idx += 1
    
    cv2.destroyAllWindows()
    print("\n✅ Visualisierung beendet")

if __name__ == '__main__':
    visualize_labels('Legosteine_comp')
