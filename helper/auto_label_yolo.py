import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
from pathlib import Path
from ultralytics import YOLO
import yaml
import numpy as np

class LegoAutoLabeler:
    def __init__(self, base_folder='Legosteine_comp', model_path='best.pt'):
        """
        Auto-Labeler mit einem vortrainierten YOLO-Modell.
        Speichert pro Bild nur die GRÖSSTE erkannte Bounding Box.
        """
        self.base_folder = Path(base_folder)
        print(f"Lade trainiertes YOLO-Modell: {model_path}")
        self.model = YOLO(model_path)
        print(f"✓ Modell geladen. Das Modell kennt diese Klassen: {self.model.names}")
    
    def process_folder(self, part_number, class_id):
        """Verarbeitet alle Bilder in einem Teilenummern-Ordner"""
        part_folder = self.base_folder / part_number
        
        if not part_folder.exists():
            print(f"⚠ Ordner nicht gefunden: {part_folder}")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        images = []
        for ext in image_extensions:
            images.extend(part_folder.glob(f'*{ext}'))
        
        print(f"\n📁 Ordner {part_number} (Class-ID: {class_id}): {len(images)} Bilder")
        
        for img_path in images:
            self.process_image(img_path, class_id)
    
    def process_image(self, img_path, class_id):
        """
        Detektiert Objekte, findet die GRÖSSTE Box und speichert nur diese.
        """
        # Detection durchführen
        results = self.model(img_path, conf=0.25, iou=0.5)
        
        # Prüfen, ob überhaupt etwas erkannt wurde
        if results[0].boxes is None or len(results[0].boxes) == 0:
            print(f"   {img_path.name}: Keine Objekte erkannt → Keine Label-Datei erstellt")
            return
        
        # Bild-Größe holen
        h, w = results[0].orig_shape
        
        # Alle Boxen als NumPy-Array holen: Shape ist (Anzahl_Boxen, 4) -> [x1, y1, x2, y2]
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Fläche jeder Box berechnen: (x2 - x1) * (y2 - y1)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Index der Box mit der größten Fläche finden
        best_box_idx = np.argmax(areas)
        best_box = boxes[best_box_idx]
        
        x1, y1, x2, y2 = best_box
        
        # Normalisierte YOLO-Koordinaten für die größte Box berechnen
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        
        # NUR diese eine Box in die Datei schreiben
        label_path = img_path.with_suffix('.txt')
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        total_detected = len(boxes)
        if total_detected > 1:
            print(f"  ✓ {img_path.name}: {total_detected} Boxen erkannt, GRÖSSTE ausgewählt → Label {class_id}")
        else:
            print(f"  ✓ {img_path.name}: 1 Box erkannt → Label {class_id}")
    
    def create_class_mapping(self):
        """Erstellt Mapping und die data.yaml für das YOLO-Training"""
        folders = sorted([f.name for f in self.base_folder.iterdir() if f.is_dir()])
        mapping = {folder: idx for idx, folder in enumerate(folders)}
        
        # 1. classes.txt
        with open('classes.txt', 'w') as f:
            for part_num in mapping.keys():
                f.write(f"{part_num}\n")
        
        # 2. class_mapping.yaml
        with open('class_mapping.yaml', 'w') as f:
            yaml.dump({'names': mapping}, f, default_flow_style=False)
        
        # 3. data.yaml (Standard-Format für Ultralytics YOLO Training)
        names_dict = {idx: name for name, idx in mapping.items()}
        
        data_yaml = {
            'path': str(self.base_folder.resolve()), 
            'train': '.',                            
            'val': '.',                              
            'nc': len(mapping),                      
            'names': names_dict                      
        }
        
        with open('data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
            
        print(f"\n📋 Mapping & data.yaml erstellt ({len(mapping)} Klassen).")
        return mapping
    
    def process_all(self):
        if not self.base_folder.exists():
            print(f"❌ Basis-Ordner nicht gefunden: {self.base_folder}")
            return
        
        mapping = self.create_class_mapping()
        print(f"\n🚀 Starte Auto-Labeling für {len(mapping)} Ordner...")
        
        for part_number, class_id in mapping.items():
            self.process_folder(part_number, class_id)
        
        print("\n✅ Automatisches Labeling abgeschlossen!")
        print("💡 Tipp: Öffne 'data.yaml', um dein Modell damit zu trainieren.")

if __name__ == '__main__':
    BASE_FOLDER = 'Legosteine_comp'
    MODEL_PATH = 'best.pt'  # Dein Kaggle-Modell
    
    labeler = LegoAutoLabeler(base_folder=BASE_FOLDER, model_path=MODEL_PATH)
    labeler.process_all()