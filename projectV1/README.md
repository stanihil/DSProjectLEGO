# projectV1 – Klassifikations-Pipeline

Dieser Ordner enthält die aufgeräumte Neuimplementierung. Der erste Schritt ist
das Klassifikationsmodell `yolo26s-cls` auf den Einzelstein-Bildern aus
`.datasets/05 - dataset/photos`.

## Daten

| Quelle | Inhalt | Verwendung |
|---|---|---|
| `.datasets/05 - dataset/photos` | 447 Klassen, **52.597 Bilder**, ein Teil pro Bild | Klassifikations-Training (Pretraining) |
| `.datasets/05 - dataset/renders` | 447 Klassen, **567.481 Renderings** (median nur ~186 px) | optionales Pretraining, noch nicht eingebunden |
| `Legosteine_comp/` | 19 Klassen, 104 echte Fotos | späteres Real-Domain-Fine-Tuning |
| `projectV1/dataset/generic/` | 200 echte Handyfotos, 1.346 Boxen | Detektion + Real-Domain-Fine-Tuning |

Der Klassenordner ist das Label. Für `yolo26s-cls` sind **keine Bounding-Boxen
nötig**; die Ordnerstruktur `train/<klasse>/bild.jpg` genügt.

`4073` fehlt im 05-Datensatz komplett und muss später aus `Legosteine_comp` /
`generic` ergänzt werden.

## 1. Datensatz bauen

```bash
python projectV1/backend/dataset/build_cls_dataset.py
```

Ergebnis (Symlinks, kein Kopieren):

```
projectV1/dataset/classification/
  train/<klasse>/*.jpg   35.313 Bilder
  val/<klasse>/*.jpg      8.401 Bilder
  test/<klasse>/*.jpg     8.883 Bilder
  classes.txt            447 Klassennamen (alphabetisch = Klassen-ID)
  dataset_stats.json
  manifest.csv
```

Der Split ist gruppenbewusst: Dateien mit gleichem `c<farbe>_<ansicht>`-Präfix
landen im selben Split, damit near-duplicates nicht zwischen train und val
lecken. Klassen mit weniger als fünf Gruppen werden zufällig gesplittet.
`--no-group-split` erzwingt einen rein zufälligen Split, `--max-per-class`
erzeugt kleine Smoke-Datensätze.

Bekannte Einschränkungen des Datensatzes:

* 284 Klassen sind gruppenbewusst, 163 Klassen (zu wenige Gruppen) zufällig
  gesplittet – dort ist die Validierung optimistischer.
* 33 Klassen haben 6 Bilder oder weniger, teils nur 1 Trainingsbild
  (z. B. `15397`, `22888`, `24122`, `25893`). Für diese Klassen ist Top-1
  kaum aussagekräftig; die mittlere Klassen-Accuracy im Eval-Report ist
  dadurch pessimistischer als die globale Top-1.
* Jede Klasse hat mindestens ein val- und ein test-Bild.
* Alle Bilder sind Studio-/Freisteller-Aufnahmen. Das Modell lernt die
  Domain „ein Teil, weißer/neutraler Hintergrund“, nicht „Teil im Regal“.

## 2. Training

```bash
python projectV1/backend/train/train_cls.py \
    --data projectV1/dataset/classification \
    --epochs 15 --batch 64 --imgsz 224 --device mps
```

Wichtig: `fliplr=0.0` und `flipud=0.0`. LEGO hat gespiegelte Teilepaare
(siehe `.datasets/archive/readme.txt`); horizontales/vertikales Spiegeln macht
aus einem Teil ein anderes.

Ergebnisse landen unter `projectV1/runs/` (gitignored).

Gemessene Geschwindigkeit auf Apple M1 (8 GB, MPS):

| Einstellung | pro Batch (64) | Bilder/s | 1 Epoche (35.313 Bilder) |
|---|---|---|---|
| `imgsz=224` | ~1,2 s Modell + ~0,25 s Daten | ~56 | ~11 min |
| `imgsz=160` | ~0,7 s | ~92 | ~7 min |
| `imgsz=224`, `batch=128` | ~36 s | ~3 | nicht nutzbar (Speicher) |

Ultralytics setzt auf MPS/CPU die DataLoader-Worker intern auf 0
(`engine/trainer.py`); eigene Worker bringen auf M1 nichts (gemessen:
4 Worker = 112 Bilder/s, 8 Worker = 64 Bilder/s, 0 Worker = 260 Bilder/s
nur fürs Laden). Das Laden ist mit ~0,25 s/Batch ohnehin nicht der Flaschenhals,
die Rechenzeit des Modells schon.

## 3. Schneller Lauf: `cls_fast` (beste Bilder + echte Fotos)

Fuer einen schnellen, brauchbaren Zwischenstand statt des vollen 52k-Datensatzes:

```bash
python projectV1/backend/dataset/build_fast_cls.py \
    --out projectV1/dataset/cls_fast --per-class 24 \
    --exclude-list multi_lego_yolo --force

python projectV1/backend/train/train_cls.py \
    --data projectV1/dataset/cls_fast --epochs 10 --batch 64 --imgsz 224 \
    --optimizer AdamW --lr0 0.001 --cos-lr --erasing 0.2 --hsv-s 0.4 \
    --name cls_fast_448_10e
```

Was der Builder macht:

* **Beste Studio-Bilder:** nur Bilder mit Mindestkante 160 px, nach Aufloesung
  sortiert, höchstens `n` pro Aufnahmegruppe `c<farbe>_<ansicht>` (Deckel passt
  sich an, damit Klassen mit nur einer Ansicht nicht auf 2 Bilder fallen).
* **Balance:** maximal 24 Bilder pro Klasse statt 1–692.
* **Echte Fotos:** Crops aus `projectV1/dataset/generic` (JSON) und
  `Legosteine_comp` (YOLO-txt), 2× oversampelt, nur im Train-Split.
* **Leakage-Schutz:** `--exclude-list multi_lego_yolo` haelt die 73 Testbilder
  aus dem Training (sie sind auch in `generic` enthalten).
* **4073** fehlt in 05 und kommt ausschliesslich aus echten Crops →
  **448 Klassen**.

Ergebnis: 8.679 train / 1.332 val / 1.188 test, jede Klasse in jedem Split.

Wichtig: `optimizer=AdamW, lr0=0.001`. Ultralytics' `optimizer=auto` hatte
`lr0=7e-6` gewaehlt – damit war das Modell nach einer Epoche bei Top-1 1,6 %.

## 4. Boxen klassifizieren (Multi-Stein-Bild)

`multi_lego_yolo/` enthaelt 73 Handyfotos mit 630 Boxen, alle Klasse 0
(Boxen ohne Labels). `classify_boxes.py` schneidet jede Box aus, klassifiziert
sie und schreibt Bilder + CSV:

```bash
python projectV1/backend/inference/classify_boxes.py \
    --images multi_lego_yolo \
    --weights projectV1/runs/cls_fast_448_10e/weights/best.pt \
    --truth-json projectV1/dataset/generic \
    --out projectV1/runs/infer_multi
```

Ausgabe:

```
predictions.csv     Bild, Box, xyxy, Teilenummer, Konfidenz, Top-5
summary.csv         gezaehlte Teile pro Bild
annotated/*.jpg     Boxen mit Teilenummer + Konfidenz
sheet.jpg           4 Bilder als Kontaktabzug
truth_eval.json/csv Trefferquote gegen die JSON-Labels (falls vorhanden)
```

`--detector helper/best.pt` erzeugt Boxen, wenn keine `.txt` daneben liegt.

### Ergebnis mit `cls_fast_448_10e` (Epoche 6)

| Testset | Umfang | Top-1 | Top-5 |
|---|---|---|---|
| `cls_fast` Test-Split (Studiofotos) | 1.188 Bilder, 448 Klassen | **75,1 %** | **92,6 %** |
| `multi_lego_yolo` (Handyfotos, Pile) | 73 Bilder, 626 Boxen | **36,1 %** | **74,0 %** |

Zuordnung zu den JSON-Labels ueber IoU ≥ 0,35 (626 von 630 Boxen zugeordnet).
Die Boxen im Handyfoto-Test kommen vom Detektor und sind teilweise locker; sie
enthalten Nachbarteile oder Hintergrund.

**Groesste Verwechslungen** (gleiche Bauform, andere Groesse):

| Echt | Vorhergesagt | Anzahl |
|---|---|---|
| 3005 (1×1) | 54200 (Schraege) | 20 |
| 3021 (2×3 Platte) | 3020 (2×4 Platte) | 18 |
| 3623 | 3710 | 15 |
| 3004 (1×2) | 3005 (1×1) | 14 |
| 3062 | 6141 | 12 |

**Trefferquote der haeufigsten Testklassen:**

| Klasse | Boxen | Top-1 |
|---|---|---|
| 4073 | 23 | 65 % |
| 54200 | 14 | 64 % |
| 3022 | 23 | 61 % |
| 3710 | 25 | 60 % |
| 3003 | 54 | 56 % |
| 3023 | 35 | 54 % |
| 2654 | 31 | 52 % |
| 3020 | 25 | 52 % |
| 3005 | 62 | 44 % |
| 3004 | 62 | 42 % |
| 3002 | 48 | 23 % |
| 3001 | 44 | 23 % |
| 3062 | 37 | 19 % |
| 3010 | 23 | 17 % |
| 3622 | 21 | 10 % |
| 3021 | 28 | 7 % |
| 6215 | 28 | 7 % |
| 3623 | 27 | 0 % |

`4073` (nur aus echten Crops gelernt) und `54200` treffen am besten – die
Real-Domain-Crops wirken. Die Konfidenz ist schlecht kalibriert: bei Schwelle
0,5 bleiben 63 % der Boxen mit 41 % Top-1; Top-5 ist die nutzbarere Ausgabe.

**Zeiten:**

| Schritt | Dauer |
|---|---|
| Datensatz bauen (`cls_fast`) | 25 s |
| Training 6 Epochen (10 geplant) | 47 min 48 s |
| Inferenz 630 Boxen | 45 s |
| Evaluation Test-Split | 23 s |

Das Training wurde bei Epoche 7 durch macOS wegen Speicherdrucks beendet
(SIGKILL, 8-GB-M1); `best.pt`/`last.pt` von Epoche 6 sind nutzbar. Die
Val-Kurve war noch steigend (0,699 → 0,739).

## 5. Evaluation

```bash
python projectV1/backend/eval/eval_cls.py \
    --weights projectV1/runs/cls_fast_448_10e/weights/best.pt \
    --data projectV1/dataset/cls_fast --split test
```

Schreibt `eval/metrics_test.json` (Top-1/Top-5, schlechteste 25 Klassen,
häufigste Verwechslungen) und `eval/per_class_test.csv` neben den Gewichten.
Die Auswertung nutzt einen eigenen Predict-Loop, weil Ultralytics 8.4
`model.validator` nicht mehr bereitstellt.

Daraus einen fertigen Report (Markdown + PNG-Tabellen) bauen:

```bash
python projectV1/backend/eval/report_cls.py \
    --run projectV1/runs/cls_yolo26s_447_5e --split test
```

Ergebnis: `report.md` und `report.png` im Lauf-Ordner – eine Tabelle pro
Abschnitt (Konfiguration, Ergebnis, Trainingsverlauf, schlechteste/beste
Klassen, häufigste Verwechslungen).

## 6. Auto-Labeling (optional)

`helper/label_dataset.py` ist die verallgemeinerte Version von
`helper/auto_label_yolo.py`: Detektion → größte Box → YOLO-txt, mit Resume,
Limit pro Klasse und Statistik.

```bash
python helper/label_dataset.py \
    --src "Legosteine_comp" --out runs/labels_comp --preview 2
```

**Nicht** für den 05-Datensatz geeignet: `helper/best.pt` ist ein Detektor für
die 19 eigenen Stein-Klassen. Auf einem Pilotlauf über 894 Bilder aus 05 wurden
nur **52,8 %** der Bilder mit einer Box versehen, 46,4 % blieben leer, und die
gefundenen Boxen lagen häufig nur auf Nupsis, Löchern oder Ecken statt auf dem
ganzen Teil. Deshalb wird 05 für die Klassifikation ohne Boxen verwendet.

## Nächste Schritte

1. Epochen 7–10 nachziehen (`--batch 48` gegen den Speicherdruck) oder von
   `best.pt` mit kleiner LR weiterfein-tunen; die Val-Kurve war noch steigend.
2. Mehr echte Crops sammeln: die 20 Klassen mit Real-Crops (`4073`, `54200`)
   treffen deutlich besser als reine Studio-Klassen.
3. Groessen-Confusion entschaerfen: Boxen vor dem Klassifizieren auf ein
   Quadrat padden (statt Center-Crop) und damit auch trainieren.
4. Detektor (klassenagnostisch) + Klassifikator zur Inventar-Pipeline verbinden.
