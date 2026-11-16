# Klassifikation – eigenständiger Trainingsordner

Gegenstück zu `projectV1/backend/object_detection/`: ein Ordner, in dem
Datensatz, Klassenliste, Trainingsskript, Testskript und Läufe zusammen
liegen. Das Training selbst ist genauso aufgebaut wie beim Detektor:

```python
model = YOLO("yolo26m-cls.pt")
model.train(data=".../classification/data", epochs=30, device="mps", ...)
```

Der bestehende Lauf `projectV1/runs/cls_yolo26m_clean` bleibt unangetastet.

## Aufbau

```
projectV1/classification/
├── build_dataset.py     # baut data/ (70/20/10, gruppenbewusster Split)
├── train.py             # ein model.train(...)
├── test.py              # Split auswerten oder Bilder klassifizieren
├── data/
│   ├── train/<teilenummer>/*.jpg
│   ├── val/<teilenummer>/*.jpg
│   ├── test/<teilenummer>/*.jpg
│   ├── classes.txt      # 448 Klassen, alphabetisch
│   ├── manifest.csv     # welche Quelle in welchem Split
│   ├── dataset_stats.json
│   └── data.yaml        # Doku/Klassenliste (Ultralytics will den Ordner)
└── runs/
    └── yolo26m_cls_70_20_10/
        ├── weights/best.pt
        ├── results.csv / results.png
        └── confusion_matrix*.png
```

## 1. Datensatz bauen

```bash
python projectV1/classification/build_dataset.py --force
```

Quellen:

| Quelle | Verwendung |
|---|---|
| `.datasets/05 - dataset/photos` | 447 Studio-Klassen, max. 40 Bilder/Klasse, max. 3 pro Ansicht `c<farbe>_<ansicht>`, Mindestkante 160 px |
| `projectV1/dataset/generic` | echte Handyfoto-Crops (X-AnyLabeling-JSON), nur `train` |
| `Legosteine_comp` | echte Handyfoto-Crops (YOLO-txt), nur `train`, 2× oversampelt |
| `multi_lego_yolo` | **ausgeschlossen** (Testset) |

Aufteilung **70/20/10** wie beim Detection-`autosplit`, aber
**gruppenbewusst**: alle Aufnahmen derselben Gruppe
(`c<farbe>_<ansicht>_<quelle>`, z. B. `c0_1_Con31`) landen im selben Split.
Sonst wäre die Validierung durch Near-Duplicates geschönt. Reichen die
Gruppen nicht für drei Splits (kleine Klassen), wird aufgefüllt, damit die
Klasse trotzdem in `train`, `val` und `test` vorkommt.
Klasse `4073` fehlt im Studio-Datensatz und kommt nur aus echten Crops.

Erwartetes Ergebnis: 448 Klassen, ca. **11.4k / 3.1k / 1.7k** Bilder
(70,4 % / 19,1 % / 10,5 %).

## 2. Training

```bash
python projectV1/classification/train.py
```

| Einstellung | Wert |
|---|---|
| Basismodell | `yolo26m-cls.pt` (offizielle Gewichte) |
| Epochen | 30, Early Stopping `patience=10` |
| Bildgröße / Batch | 224 px / 32 |
| Optimizer | AdamW, `lr0=1e-3`, Cosine |
| Device | `mps` (Apple M1) |
| Spiegelung | `fliplr=0`, `flipud=0` |

**Warum keine Spiegelung:** LEGO hat chirale Teilepaare – ein gespiegelter
Stein ist ein anderes Teil. Das war schon beim sauberen Lauf so und bleibt
hier bewusst erhalten.

Der Lauf schreibt nach `runs/yolo26m_cls_70_20_10/`. Auf dem M1 ist mit
mehreren Stunden zu rechnen; die Epochen 1–5 liefen im Vergleichslauf mit
7–15 min, danach kann das Gerät thermisch drosseln.

## 3. Test

```bash
# Top-1/Top-5 auf einem Split
python projectV1/classification/test.py --split test

# Einzelbild oder ganzer Ordner, beschriftete Kopien + CSV
python projectV1/classification/test.py --source Legosteine_comp/3002
python projectV1/classification/test.py --source multi_lego_yolo
```

Ausgabe der Bild-Variante: `runs/predict/` mit beschrifteten JPGs und
`predictions.csv` (Top-1, Konfidenz, Top-5).

## Ergebnis

Wird nach dem Lauf hier ergänzt (Top-1/Top-5 auf `val` und `test` sowie
Dauer). Vergleichbar ist der bestehende Lauf:

| Lauf | Split | Modell | Top-1 | Top-5 |
|---|---|---|---|---|
| `cls_yolo26m_clean` | 80/10/10 | yolo26m-cls, 10 Ep. | 84,6 % | 95,7 % |
| `yolo26m_cls_70_20_10` | 70/20/10 | yolo26m-cls, 30 Ep. | – | – |

## Unterschiede zum bestehenden Lauf

| | `cls_yolo26m_clean` | dieser Ordner |
|---|---|---|
| Split | 80/10/10 | **70/20/10** |
| Code | viele Skripte in `backend/` | **3 Skripte, alles in einem Ordner** |
| Epochen | 10 | 30 + Early Stop |
| Ausgabe | `projectV1/runs/` | `projectV1/classification/runs/` |

Der alte Lauf bleibt als Referenz erhalten; dieser Ordner ist der neue,
in sich geschlossene Trainingsweg.
