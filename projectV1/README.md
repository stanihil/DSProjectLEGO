# projectV1 – LEGO-Teilklassifikation

Ziel: ein Klassifikator, der **einzelne LEGO-Teile** auf Teilenummern abbildet
(z. B. `3001`, `3069b`, `4073`). Der Detektor findet die Boxen auf einem
Handyfoto, der Klassifikator sagt pro Box, welches Teil es ist.

Aktuell gibt es **einen** Klassifikations-Lauf:
`projectV1/classification/` mit den **19 Klassen aus `Legosteine_comp`**
(`yolo26s-cls.pt`), ausgeliefert als
`projectV1/models/cls_legosteine-19_yolo26s-cls_224_2026-11-16.pt`.
Aktuelle Doku: `projectV1/classification/README.md` und
`projectV1/models/README.md`.

Der frühere 448-Klassen-Lauf (`cls_yolo26m_clean`) und sein Datensatz
`cls_clean` wurden auf Wunsch gelöscht; die Ergebnisse in diesem README sind
historisch und beschreiben die damalige Pipeline.

## Ordnerstruktur

```
DSProjectLEGO/
├── .datasets/05 - dataset/photos/   Rohdaten: 52.597 Studiofotos, 447 Klassenordner
├── Legosteine_comp/                 Rohdaten: 104 handgelabelte echte Crops, 19 Klassen
├── multi_lego_yolo/                 Testset: 73 Handyfotos + Boxen (Klasse 0)
├── cuttet_multi_lego_yolo/          GENERIERT: 623 Crops aus den Handyfotos, 19 Klassen
├── yolo26m-cls.pt                   Basisgewichte (einmal geladen, wiederverwendet)
│
├── projectV1/
│   ├── classification/              ★ aktueller Klassifikations-Trainingsordner
│   │   ├── build_dataset.py         baut data/ aus Legosteine_comp (70/20/10)
│   │   ├── train.py                 Training, veröffentlicht nach models/
│   │   ├── test.py                  Split-Metrik oder Bild-/Ordner-Inferenz
│   │   ├── data/                    GENERIERT: 195/20/19 Bilder, 19 Klassen
│   │   └── runs/                    GENERIERT: Lauf + eval_val/eval_test
│   ├── models/                      ★ benannte Modelle mit Herkunft
│   │   ├── publish.py               Lauf -> cls_legosteine-19_yolo26s-cls_224_<datum>.pt
│   │   ├── manifest.json            alle Modelle, Metriken, Hashes
│   │   └── cls_*.pt + cls_*.json    Artefakt + Sidecar (siehe models/README.md)
│   ├── backend/
│   │   ├── object_detection/        Detektion: split_datasaet.py, train.py, test.py
│   │   ├── dataset/                 alte 448-Klassen-Aufbereitung (historisch)
│   │   ├── train/                   alter 448-Klassen-Lauf (historisch, Datensatz gelöscht)
│   │   ├── eval/                    eval_cls.py, report_cls.py, test_random_samples.py
│   │   └── inference/               classify_boxes.py (Boxen ausschneiden + klassifizieren)
│   ├── dataset/generic/             Handyfotos + JSON-Boxen (Detektion)
│   ├── UI/streamlit_UI.py           Demo-Oberfläche
│   └── README.md
│
└── helper/                          Detektor `best.pt` (Herkunft undokumentiert), Auto-Labeling
```

**Rohdaten ≠ generiert:** `.datasets/`, `Legosteine_comp/`, `generic/` und
`multi_lego_yolo/` sind Quellen und bleiben. `classification/data/` und
`classification/runs/` werden jederzeit aus den Quellen neu erzeugt und sind
gitignored. `models/*.pt` sind ebenfalls gitignored; versioniert werden nur
`publish.py`, `manifest.json` und die `.json`-Sidecars.

## Datenquellen

| Quelle | Inhalt | Verwendung |
|---|---|---|
| `.datasets/05 - dataset/photos` | 447 Klassen, 52.597 Studiofotos, ein Teil pro Bild | Haupt-Training |
| `Legosteine_comp/` | 19 Klassen, 104 echte Crops (168×224), X-AnyLabeling-/YOLO-Labels | echte Crops im Training |
| `projectV1/dataset/generic/` | 200 Handyfotos, 718 Boxen, handgezeichnete JSON-Labels | 127 Fotos → Training, 73 → Test |
| `multi_lego_yolo/` | 73 Handyfotos, 630 Detektor-Boxen (Klasse 0) | **nur Test** |

Der Klassenordner ist das Label. `4073` fehlt im 05-Datensatz und kommt
ausschließlich aus echten Crops; dadurch sind es **448 Klassen**.

## Datenqualität

Vor dem Training wurde der 05-Datensatz geprüft (`audit_dataset.py`,
`find_class_outliers.py`):

| Prüfung | Ergebnis | Konsequenz |
|---|---|---|
| Kaputte Bilder | 0 von 52.597 | – |
| Bilder unter 160 px Mindestkante | 1.259 (2,4 %) | ausgeschlossen (3 Klassen fallen auf ihre kleinen Bilder zurück) |
| Byte-identische Duplikate | 0 | – |
| Near-Duplicate-Paare (dHash ≤ 4) | 18.852 | kein Ausschluss; ohnehin durch gruppenbewussten Split getrennt |
| Klassenübergreifende dHash-Konflikte | 6.764 Paare in 300 Klassen | **visuell geprüft: Fehlalarme** (ähnliche Form + weißer Hintergrund, z. B. Zahnrad vs. Reifen). Keine Fehllabels. |
| Ausreißer pro Klasse (Abstand zum Medoid) | 623 Bilder (1,18 %) | visuell geprüft: nur Farbe/Unschärfe, keine falschen Labels |

Fazit: Die Ordnerlabels sind verlässlich. Ausgeschlossen werden nur zu kleine
Bilder, echte Testbilder und offensichtlich unbrauchbare Quellen.

## 1. Datensatz bauen

```bash
python projectV1/backend/dataset/build_fast_cls.py \
    --out projectV1/dataset/cls_clean \
    --per-class 40 --max-per-view 3 --min-side 160 \
    --exclude-list multi_lego_yolo --force
```

Was der Builder macht:

* **Beste Studio-Bilder:** nur Bilder mit Mindestkante 160 px, nach Auflösung
  sortiert, höchstens 3 pro Aufnahmegruppe `c<farbe>_<ansicht>`.
* **Balance:** maximal 40 Studio-Bilder pro Klasse (statt 1–899).
* **Echte Fotos:** Crops aus `generic/` (JSON) und `Legosteine_comp/`
  (YOLO-txt), 2× oversampelt, nur im Train-Split.
* **Leakage-Schutz:** Die 73 Testbilder aus `multi_lego_yolo/` sind auch in
  `generic/` enthalten und werden per Dateiname komplett aus dem Training
  gehalten.
* **Gruppenbewusster Split:** Alle Bilder einer Aufnahmegruppe landen im selben
  Split (train/val/test), damit Near-Duplicates nicht zwischen Training und
  Validierung lecken.

Ergebnis: **11.775 train / 2.263 val / 2.169 test**, 448 Klassen,
jede Klasse in jedem Split. `dataset_stats.json` und `manifest.csv` liegen im
Datensatzordner.

## 2. Training

**Der Trainingscode ist `projectV1/backend/train/train_yolo26m_cls.py`.**

```bash
source .venv/bin/activate
python projectV1/backend/train/train_yolo26m_cls.py \
    --data projectV1/dataset/cls_clean \
    --model yolo26m-cls.pt \
    --epochs 10 --imgsz 224 --batch 32 \
    --name cls_yolo26m_clean
```

Das Skript macht alles in einem Durchlauf:

1. prüft, dass train/val dieselben Klassen haben und keine Klasse leer ist,
2. startet von `yolo26m-cls.pt` (ImageNet-vortrainiert, **nicht** vom alten
   `best.pt`),
3. trainiert mit AdamW (`lr0=1e-3`, Cosine, Weight-Decay `5e-4`),
4. augmentiert nur Farbe/Scale/Translation/Erasing – **keine Spiegelung**
   (`fliplr=0`, `flipud=0`), weil gespiegelte LEGO-Teilepaare die Klassen
   ununterscheidbar machen,
5. wertet nach dem Training `best.pt` auf `val` und `test` aus,
6. schreibt `train_summary.json` mit Dauer, Sekunden/Epoche und Top-1/Top-5.

Wichtige Argumente:

| Argument | Default | Warum |
|---|---|---|
| `--model` | `yolo26m-cls.pt` | mittelgroßes Modell, passt zu 8 GB M1 |
| `--imgsz` | 224 | gleiche Auflösung wie die Inferenz-Crops |
| `--batch` | 32 | ~3,3 GB MPS-Speicher; 64 war in früheren Läufen zu viel |
| `--epochs` | 10 | mit Cosine-LR und Pretraining ausreichend |
| `--patience` | 4 | Early Stopping, falls Val nicht mehr steigt |
| `--lr0` | 1e-3 | `optimizer=auto` wählte früher 7e-6 → Modell lernte nichts |
| `--dropout` | 0.0 | weniger Regularisierung nötig, Daten sind sauber |

Gemessene Geschwindigkeit auf Apple M1 (8 GB, MPS):
**ca. 1,1 s/Batch → ca. 7 min/Epoche** (11.775 Bilder, Batch 32).
Ultralytics setzt auf MPS die DataLoader-Worker intern auf 0; das Laden ist
nicht der Flaschenhals.

Ausgaben im Laufordner `projectV1/runs/cls_yolo26m_clean/`:

```
weights/best.pt      bester Checkpoint
weights/last.pt      letzter Checkpoint
results.csv          Loss/Accuracy pro Epoche
args.yaml            vollständige Konfiguration
train_summary.json   Dauer + Val/Test-Metriken
```

## 3. Evaluation

```bash
python projectV1/backend/eval/eval_cls.py \
    --weights projectV1/runs/cls_yolo26m_clean/weights/best.pt \
    --data projectV1/dataset/cls_clean --split test

python projectV1/backend/eval/report_cls.py \
    --run projectV1/runs/cls_yolo26m_clean --split test
```

`eval_cls.py` nutzt einen eigenen Predict-Loop (Ultralytics 8.4 hat
`model.validator` entfernt) und schreibt `eval/metrics_test.json` +
`eval/per_class_test.csv`. `report_cls.py` erzeugt daraus `report.md` und
`report.png`.

## 4. Handyfotos testen

`multi_lego_yolo/` enthält 73 Handyfotos mit 630 Boxen (alle Klasse 0).
`classify_boxes.py` schneidet jede Box aus, klassifiziert sie und vergleicht
mit den JSON-Labels aus `generic/` (Zuordnung über IoU ≥ 0,35):

```bash
python projectV1/backend/inference/classify_boxes.py \
    --images multi_lego_yolo \
    --weights projectV1/runs/cls_yolo26m_clean/weights/best.pt \
    --truth-json projectV1/dataset/generic \
    --out projectV1/runs/cls_yolo26m_clean/infer_multi
```

Ausgabe: `predictions.csv`, `summary.csv`, `annotated/*.jpg`, `sheet.jpg`,
`truth_eval.json/csv`.

## 5. Zufallsstichprobe pro Klassenordner

`test_random_samples.py` zieht pro Unterordner eine feste Zufallsstichprobe
(Seed) und klassifiziert sie:

```bash
python projectV1/backend/eval/test_random_samples.py \
    --images Legosteine_comp \
    --weights projectV1/runs/cls_yolo26m_clean/weights/best.pt \
    --per-class 1 --seed 42 --crop-from-txt \
    --out projectV1/runs/cls_yolo26m_clean/samples_legosteine_comp_crop
```

`--crop-from-txt` schneidet das Teil mit der vorhandenen YOLO-Box
(6 % Rand, wie `build_fast_cls.py`). **Das ist Pflicht:** der Klassifikator
wurde auf engen Crops trainiert; ein 168×224-Ganzbild mit kleinem Teil
liefert nur 3/19 Treffer, mit Box 16/19.

## Aktueller Stand: `cls_yolo26m_clean`

Lauf: `yolo26m-cls.pt`, 448 Klassen, 10 Epochen, 224 px, Batch 32,
AdamW `lr0=1e-3`, Cosine, `fliplr=0`/`flipud=0`, Datensatz `cls_clean`.

### Ergebnisse

| Testset | Umfang | Top-1 | Top-5 |
|---|---|---|---|
| `cls_clean` Test-Split (Studio, nie gesehen) | 2.169 Bilder, 448 Klassen | **84,6 %** | **95,7 %** |
| `multi_lego_yolo` (73 Handyfotos, 626 Boxen) | echte Detektor-Boxen | **44,7 %** | **79,7 %** |

Vergleich zum alten Lauf `cls_fast_448_10e` (yolo26s, gleicher Test):

| Testset | alt Top-1 / Top-5 | neu Top-1 / Top-5 |
|---|---|---|
| Studio-Test-Split | 75,1 % / 92,6 % | **84,6 % / 95,7 %** |
| Handyfotos (`multi_lego_yolo`) | 36,1 % / 74,0 % | **44,7 % / 79,7 %** |

Die mittlere Top-1-Konfidenz auf den Handyfotos stieg von 0,60 auf **0,74**;
nur 3 von 630 Boxen liegen unter 0,25 (vorher 24).

### Trefferquote pro Testklasse (Handyfotos)

| Klasse | Boxen | Top-1 alt | Top-1 neu |
|---|---|---|---|
| 4073 | 23 | 65 % | **96 %** |
| 3020 | 25 | 52 % | **80 %** |
| 3022 | 23 | 61 % | **78 %** |
| 3005 | 62 | 44 % | **73 %** |
| 54200 | 14 | 64 % | **71 %** |
| 3023 | 35 | 54 % | **63 %** |
| 3004 | 62 | 42 % | **56 %** |
| 3001 | 44 | 23 % | **45 %** |
| 3666 | 11 | – | 45 % |
| 3003 | 54 | 56 % | 43 % |
| 3710 | 25 | 60 % | 40 % |
| 3002 | 48 | 23 % | **31 %** |
| 3021 | 28 | 7 % | **29 %** |
| 2654 | 31 | 52 % | 26 % |
| 6215 | 28 | 7 % | **25 %** |
| 3062 | 37 | 19 % | 19 % |
| 3010 | 23 | 17 % | 13 % |
| 3622 | 21 | 10 % | 5 % |
| 3623 | 27 | 0 % | 4 % |

`lego_brick` (4 Boxen) ist ein Papierkorb-Label ohne Teilenummer und `30062`
hat 1 Box – beide zählen als Fehlklassen.

### Häufigste Verwechslungen (neu)

| Echt | Vorhergesagt | Anzahl |
|---|---|---|
| 3004 (1×2) | 3005 (1×1) | 17 |
| 3062 (1×1 rund) | 4073 (Rundplatte) | 17 |
| 3003 (2×2) | 3022 (2×2 Platte) | 15 |
| 3010 (1×4) | 3004 (1×2) | 13 |
| 3002 (2×3) | 3001 (2×4) | 10 |

Typisches Muster: Verwechslung **innerhalb derselben Baureihe** (Stein ↔ Platte,
Länge 2 vs. 4). Top-5 deckt die meisten davon ab.

### Zeiten

| Schritt | Dauer |
|---|---|
| Datensatz bauen (`cls_clean`) | 26 s |
| Datenqualitäts-Audit (52.597 Bilder) | ~4 min |
| Training 10 Epochen | 7 h 25 min (M1-Drosselung ab Epoche 6; ohne Drosselung ~1,5 h) |
| Eval Test-Split (eigener Loop, 2.169 Bilder) | ~3 min |
| Handyfoto-Test (630 Boxen) | ~2 min |

Die Trainingsdauer ist auf dem 8-GB-M1 stark thermisch limitiert; die Epochen
1–5 liefen mit 7–15 min, die Epochen 6–7 mit 1,5–2,5 h.

### Schwächste Klassen (Studio-Test)

| Klasse | Bilder | Top-1 | Top-5 |
|---|---|---|---|
| 30165 | 9 | 0 % | 33 % |
| 20482 | 6 | 0 % | 83 % |
| 3640 | 6 | 0 % | 0 % |
| 6157 | 5 | 0 % | 100 % |
| 64683 | 5 | 0 % | 40 % |
| 27940 | 4 | 0 % | 100 % |
| 3676 | 4 | 0 % | 0 % |
| 58176 | 4 | 0 % | 0 % |

Alle haben ≤ 9 Trainingsbilder – die globale Top-1 von 84,6 % leidet an diesen
zwanzig Mini-Klassen, die mittlere Klassen-Top-1 liegt bei 85,8 %.

### Zufallsstichprobe `Legosteine_comp` (19 Klassen, Seed 42)

Ein zufälliges Bild pro Unterordner, einmal als Ganzbild und einmal als Crop
mit der vorhandenen `.txt`-Box:

| Eingabe | Treffer | Top-1 | Top-5 |
|---|---|---|---|
| 168×224-Ganzbild (ohne Crop) | 3/19 | 15,8 % | 36,8 % |
| Crop aus der `.txt`-Box | 16/19 | **84,2 %** | 84,2 % |
| Crop, nur Bilder **mit** Box | 16/16 | **100 %** | 100 % |

Die drei Fehlschläge sind **exakt** die drei Bilder ohne `.txt`
(`3009/IMG_3148`, `3622/IMG_3169`, `4073/*` – `4073` hat gar keine Boxen).
Sie werden als Ganzbild gezeigt und scheitern an der fehlenden Box,
nicht an der Klasse.

Über alle 94 zufällig gezogenen Bilder (`--per-class 5 --seed 7`):

| Eingabe | n | Top-1 | Top-5 |
|---|---|---|---|
| mit Crop | 84 | **98,8 %** | 100 % |
| ohne Box (Ganzbild) | 10 | 0 % | 20 % |

**Wichtig:** Diese Bilder stecken im Trainingsdatensatz (`comp_*_0_0.jpg`),
messen also kein Generalisieren – sie bestätigen nur, dass Labels, Klassen-
Reihenfolge und Vorverarbeitung stimmen. Die echten Generalisierungszahlen
sind der Test-Split (84,6 %) und die Handyfotos (44,7 %).

**Lehre für die Pipeline:** Der Klassifikator braucht immer einen engen Crop
(Detektor-Box). Ein ganzes Foto mit kleinem Teil im Bild funktioniert nicht.

## Design-Entscheidungen

1. **Saubere Daten statt maximaler Menge.** Der 05-Datensatz wird auf 40
   Bilder/Klasse begrenzt, zu kleine Bilder fliegen raus, echte Testbilder
   sind tabu.
2. **Kein Weiter-Training des alten Modells.** Start immer von den offiziellen
   `yolo26m-cls.pt`-Gewichten.
3. **Keine Spiegel-Augmentierung.** LEGO hat chirale Teilepaare.
4. **Gruppenbewusster Split.** Near-Duplicates aus derselben Aufnahme landen
   nie in train und val gleichzeitig.
5. **Echte Crops im Training.** `4073`, `54200` & Co. profitieren sichtbar von
   Handyfotos; die 73 Testfotos bleiben trotzdem komplett draußen.
