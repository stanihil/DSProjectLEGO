# Klassifikation – 19 Klassen aus `Legosteine_comp`

Eigenständiger Trainingsordner, aufgebaut wie
`projectV1/backend/object_detection/`: Datensatz, Skripte und Läufe liegen
zusammen. Trainiert wird ausschließlich auf den Klassen, die in
`Legosteine_comp/` liegen.

```python
model = YOLO("yolo26m-cls.pt")
model.train(data=".../classification/data", epochs=100, device="mps", ...)
```

## Aufbau

```
projectV1/classification/
├── build_dataset.py     # baut data/ aus Legosteine_comp (70/20/10)
├── train.py             # ein model.train(...)
├── test.py              # Split auswerten oder Bilder klassifizieren
├── data/
│   ├── train/<teilenummer>/*.jpg
│   ├── val/<teilenummer>/*.jpg
│   ├── test/<teilenummer>/*.jpg
│   ├── classes.txt      # 19 Klassen, alphabetisch
│   ├── manifest.csv     # Quelle + crop/full je Bild
│   ├── dataset_stats.json
│   └── data.yaml        # Doku/Klassenliste (Ultralytics will den Ordner)
└── runs/
    ├── yolo26m_cls_legosteine/
    │   ├── weights/best.pt
    │   ├── results.csv / results.png
    │   └── confusion_matrix*.png
    ├── eval_val/           # test.py --split val
    └── eval_test/          # test.py --split test
```

## 1. Datensatz bauen

```bash
python projectV1/classification/build_dataset.py --force
```

- Quelle: `Legosteine_comp/<teilenummer>/`
- **19 Klassen:** 2654, 3002, 3003, 3004, 3005, 3009, 3010, 3020, 3021,
  3022, 3023, 3062, 3622, 3623, 3666, 3710, 4073, 54200, 6215
- Jedes Bild mit `.txt` wird an der YOLO-Box zugeschnitten (6 % Rand), also
  so, wie der Detektor später die Box liefert.
- Bilder **ohne** Box (10 Stück, darunter alle 5 von `4073`) werden als
  Ganzbild verwendet und im `manifest.csv` als `kind=full` markiert.
- Aufteilung **70/20/10** pro Klasse; `train` wird 3× oversampelt.

Ergebnis: 104 eindeutige Bilder → 65 train (×3 = 195), 20 val, 19 test.

> Die Validierung ist mit 19–20 Bildern sehr klein; jede Klasse ist dort
> mit genau einem Bild vertreten. Top-1 springt dadurch in 5-%-Schritten.
> Das ist bei 104 Bildern nicht anders möglich und wird hier nicht schön
> gerechnet.

## 2. Training

```bash
python projectV1/classification/train.py
```

| Einstellung | Wert |
|---|---|
| Basismodell | `yolo26m-cls.pt` (offizielle Gewichte) |
| Epochen | 100, Early Stopping `patience=20` |
| Bildgröße / Batch | 224 px / 32 |
| Optimizer | AdamW, `lr0=1e-3`, Cosine |
| Device | `mps` (Apple M1) |
| Spiegelung | `fliplr=0`, `flipud=0` |

**Warum keine Spiegelung:** LEGO hat chirale Teilepaare – ein gespiegelter
Stein ist ein anderes Teil.

## 3. Test

```bash
# Top-1/Top-5 auf einem Split (schreibt nach runs/eval_<split>/)
python projectV1/classification/test.py --split test

# Einzelbild oder Ordner, beschriftete Kopien + CSV
python projectV1/classification/test.py --source Legosteine_comp/3002
python projectV1/classification/test.py --source multi_lego_yolo
```

Bild-Variante: `runs/predict/` mit beschrifteten JPGs und
`predictions.csv` (Top-1, Konfidenz, Top-5).

## Ergebnis

Lauf: `runs/yolo26m_cls_legosteine/weights/best.pt`

| Split | Bilder | Top-1 | Top-5 |
|---|---|---|---|
| `val` | 20 | **0,750** | 0,850 |
| `test` | 19 | **0,737** (14/19) | 0,947 (18/19) |

- Training: 24 Epochen, dann Early Stop (`patience=20`), **4,3 min** auf MPS.
- `best.pt` ist der Stand aus Epoche 9 (`top1=0,75`, `top5=0,85`).
- Die 5 Fehler im Test stammen alle aus verwechselbaren Formen:

| echt | vorhergesagt | Konfidenz |
|---|---|---|
| 2654 | 3020 | 0,66 |
| 3010 | 3002 | 0,71 |
| 3021 | 3623 | 0,46 |
| 3622 | 3666 | 0,59 |
| 6215 | 3022 | 0,97 |

Richtig erkannt wurde meist mit hoher Konfidenz (Median 1,00, Minimum 0,36
bei `3020`). Details: `runs/eval_test/confusion_matrix*.png`,
`runs/eval_val/`.

### Was die Zahlen nicht sagen

- **Ein Bild pro Klasse** in `val` (20) und `test` (19). Ein Fehler sind
  sofort 5,3 Prozentpunkte; die Zahlen sind eine Stichprobe, keine
  belastbare Genauigkeit.
- `4073` hat in `Legosteine_comp` **keine Boxen** – alle 5 Bilder sind
  Ganzbilder (`kind=full`), das Testbild ist also genauso aufgenommen wie
  das Training. Die 1,00 dort ist erwartbar und kein Beleg für Robustheit.
- Getestet ist nur die Domäne `Legosteine_comp` (Freisteller, oft
  zugeschnitten). Auf den Handyfotos in `multi_lego_yolo` ist das Modell
  **nicht** geprüft.
- Kein Foto liegt in zwei Splits: die 104 Quellbilder sind sauber auf
  65/20/19 verteilt, es gibt keine Überlappung zwischen Training und
  Validierung/Test.

## Hinweis zum alten 448-Klassen-Lauf

Der frühere Datensatz `projectV1/dataset/cls_clean`, der Run
`projectV1/runs/cls_yolo26m_clean` und die 70/20/10-Zwischenversion wurden
auf Wunsch gelöscht. Es gibt jetzt genau einen Klassifikations-Datensatz
und einen Lauf: diesen hier.
