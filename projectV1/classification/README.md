# Klassifikation – 19 Klassen aus `Legosteine_comp`

Eigenständiger Trainingsordner, aufgebaut wie
`projectV1/backend/object_detection/`: Datensatz, Skripte und Läufe liegen
zusammen. Trainiert wird ausschließlich auf den Klassen, die in
`Legosteine_comp/` liegen.

```python
model = YOLO("yolo26s-cls.pt")
model.train(data=".../classification/data", epochs=100, device="mps", ...)
```

## Aufbau

```
projectV1/classification/
├── build_dataset.py     # baut data/ aus Legosteine_comp (70/20/10)
├── train.py             # ein model.train(...) + publish nach models/
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
    ├── cls_s_seed2/        # veröffentlichter Lauf
    │   ├── weights/best.pt
    │   ├── results.csv / results.png
    │   └── confusion_matrix*.png
    ├── eval_val/           # test.py --split val
    └── eval_test/          # test.py --split test
```

Das ausgelieferte Modell liegt nicht in `runs/`, sondern mit sprechendem
Namen in `projectV1/models/`:

```
cls_legosteine-19_yolo26s-cls_224_2026-11-16.pt
cls_legosteine-19_yolo26s-cls_224_2026-11-16.json
```

Schema, Manifest und `publish.py` stehen in `projectV1/models/README.md`.
`train.py` veröffentlicht den Lauf automatisch (`--no-publish` schaltet das
ab), `test.py --weights latest` nimmt das zuletzt veröffentlichte Modell.

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
| Basismodell | `yolo26s-cls.pt` (offizielle Gewichte, frisch geladen) |
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

# zuletzt veröffentlichtes Modell statt Lauf-Gewichte
python projectV1/classification/test.py --split test --weights latest

# Einzelbild oder Ordner, beschriftete Kopien + CSV
python projectV1/classification/test.py --source Legosteine_comp/3002
python projectV1/classification/test.py --source multi_lego_yolo
```

Bild-Variante: `runs/predict/` mit beschrifteten JPGs und
`predictions.csv` (Top-1, Konfidenz, Top-5).

## Ergebnis

Modell: `projectV1/models/cls_legosteine-19_yolo26s-cls_224_2026-11-16.pt`
(Kopie von `runs/cls_s_seed2/weights/best.pt`, SHA-256 und Seed im Sidecar)

| Split | Bilder | Top-1 | Top-5 |
|---|---|---|---|
| `val` | 20 | **0,700** | **1,000** |
| `test` | 19 | **0,895** (17/19) | **1,000** |

- Training: 28 Epochen, Early Stop (`patience=20`), 1,8 min auf MPS.
- Veröffentlicht ist Seed 2. Auswahl über `val`: Top-1 war bei allen
  `26s`-Seeds 0,700, Tiebreak über Top-5 (Seed 2: 1,000).

### `26s` oder `26m`?

Beide mit identischen Einstellungen, je 3 Seeds:

| Modell | val Top-1 | test Top-1 | test Top-5 | Params | Größe | GFLOPs |
|---|---|---|---|---|---|---|
| **26s** | 0,700 / 0,700 / 0,700 | 0,737 / 0,737 / 0,895 | 0,947 / 1,000 / 1,000 | 5,5 M | 11 MB | 12 |
| 26m | 0,750 / 0,700 / 0,700 | 0,737 / 0,684 / 0,789 | 0,947 / 0,895 / 0,895 | 10,4 M | 21 MB | 39 |

`26m` gewinnt nur ein einziges val-Bild (0,750 vs 0,700) und liegt im Test
im Schnitt darunter. Bei 19–20 Prüfbildern ist das Rauschen, kein Vorsprung –
deshalb `26s`: halbe Größe, ein Drittel der Rechenzeit. Die anderen fünf
Läufe sind gelöscht; reproduzierbar mit `--model`/`--seed`.

### Was die Zahlen nicht sagen

- **Ein Bild pro Klasse** in `val` (20) und `test` (19). Ein Fehler sind
  sofort 5,3 Prozentpunkte. Das ist eine Stichprobe, keine belastbare
  Genauigkeit.
- `4073` hat in `Legosteine_comp` **keine Boxen** – alle 5 Bilder sind
  Ganzbilder (`kind=full`). Die 1,00 dort sagt nichts über Crops oder
  Handyfotos.
- Getestet ist nur `Legosteine_comp` (Freisteller, oft zugeschnitten). Auf
  den Handyfotos in `multi_lego_yolo` ist das Modell **nicht** geprüft.
- Kein Foto liegt in zwei Splits (104 Quellbilder → 65/20/19).
- Die Testzahlen sind mit 19 Bildern so klein, dass ich das Modell über
  `val` ausgewählt habe; der Test dient nur der Kontrolle.

## Hinweis zum alten 448-Klassen-Lauf

Der frühere Datensatz `projectV1/dataset/cls_clean`, der Run
`projectV1/runs/cls_yolo26m_clean` und die 70/20/10-Zwischenversion wurden
auf Wunsch gelöscht. Es gibt jetzt genau einen Klassifikations-Datensatz
und einen Lauf: diesen hier.
