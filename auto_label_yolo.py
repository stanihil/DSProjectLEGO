import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
from pathlib import Path
from ultralytics import YOLO
import yaml

class LegoAutoLabeler:
    def __init__(self, base_folder='Legosteine_comp', model_path='yolov8s-worldv2.pt'):
        """
        Auto-Labeler mit YOLO-World (kann Objekte per Text erkennen)
        """
        self.base_folder = Path(base_folder)
        print(f"Lade YOLO-World Modell: {model_path}")
        self.model = YOLO(model_path)
        
        # Wir sagen dem Modell, wonach es suchen soll!
        self.model.set_classes(["lego brick", "lego piece", "toy", "plastic block"])
        print("✓ Modell geladen und auf LEGO spezialisiert!")
    
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
        """Detektiert Objekte und speichert sie mit der Class-ID des Ordners"""
        # Detection durchführen
        results = self.model(img_path, conf=0.15, iou=0.5)
        
        # Prüfen ob überhaupt Boxen gefunden wurden
        num_boxes = len(results[0].boxes) if results[0].boxes else 0
        
        if num_boxes == 0:
            # Keine Objekte gefunden → Keine .txt-Datei erstellen
            print(f"   {img_path.name}: Keine Objekte erkannt → Keine Label-Datei erstellt")
            return
        
        # Nur wenn Boxen gefunden wurden → Label-Datei erstellen
        label_path = img_path.with_suffix('.txt')
        
        with open(label_path, 'w') as f:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        h, w = result.orig_shape
                        
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"  ✓ {img_path.name}: {num_boxes} Objekte erkannt → Label {class_id} gespeichert")
    
    def create_class_mapping(self):
        """Erstellt Mapping von Teilenummern zu Class-IDs"""
        folders = sorted([f.name for f in self.base_folder.iterdir() if f.is_dir()])
        mapping = {folder: idx for idx, folder in enumerate(folders)}
        
        with open('class_mapping.yaml', 'w') as f:
            yaml.dump({'names': mapping}, f, default_flow_style=False)
        
        with open('classes.txt', 'w') as f:
            for part_num in mapping.keys():
                f.write(f"{part_num}\n")
        
        print(f"\n📋 Class Mapping erstellt ({len(mapping)} Klassen). Siehe classes.txt")
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

if __name__ == '__main__':
    BASE_FOLDER = 'Legosteine_comp'
    # Wir nutzen das 'world' Modell, das auf Text-Prompts hört
    MODEL_PATH = 'yolov8s-worldv2.pt'  
    
    labeler = LegoAutoLabeler(base_folder=BASE_FOLDER, model_path=MODEL_PATH)
    labeler.process_all()