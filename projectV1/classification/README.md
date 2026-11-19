# Klassifikation – 19 Klassen aus `Legosteine_comp`, 20 mit Handyfotos

Eigenständiger Trainingsordner, aufgebaut wie
`projectV1/backend/object_detection/`: Datensatz, Skripte und Läufe liegen
zusammen. Erstes Modell: 19 Klassen aus `Legosteine_comp/`. Zweites Modell
(Abschnitt 5): dieselben Bilder plus die 623 Handyfoto-Crops aus
`cuttet_multi_lego_yolo/` → 20 Klassen.

```python
model = YOLO("yolo26s-cls.pt")
model.train(data=".../classification/data", epochs=100, device="mps", ...)
```

## Aufbau

```
projectV1/classification/
├── build_dataset.py     # baut data/ aus Legosteine_comp (70/20/10)
├── cut_multi_lego_yolo.py  # schneidet die Handyfotos in Teilenummer-Ordner
├── train.py             # ein model.train(...) + publish nach models/
├── test.py              # Split auswerten oder Bilder klassifizieren
├── evaluate.py          # Split auswerten: pro Klasse + Studio/Handy
├── data/                # 19 Klassen, nur Legosteine_comp
├── data_merged/         # 20 Klassen, beide Quellen (Abschnitt 5)
│   ├── train|val|test/<teilenummer>/*.jpg
│   ├── classes.txt
│   ├── manifest.csv     # Quelle + crop/full je Bild
│   ├── dataset_stats.json
│   └── data.yaml
└── runs/
    ├── cls_s_seed2/        # veröffentlichter Lauf (19 Klassen)
    ├── cls_legosteine_s_merged/  # Weiter-Training (20 Klassen)
    ├── eval_val/           # test.py --split val
    └── eval_test/          # test.py --split test
```

Das ausgelieferte Modell liegt nicht in `runs/`, sondern mit sprechendem
Namen in `projectV1/models/`:

```
cls_legosteine-19_yolo26s-cls_224_2026-11-16.pt   # nur Legosteine_comp
cls_legosteine-20_yolo26s-cls_224_2026-11-16.pt   # + Handyfoto-Crops
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

## 4. Handyfoto-Crops aus `multi_lego_yolo`

Zweite Datenbasis: dieselben Handyfotos, an den Boxen zugeschnitten und nach
Teilenummer sortiert.

```bash
python projectV1/classification/cut_multi_lego_yolo.py --force
```

- Die Teilenummern stehen in `projectV1/dataset/generic/<name>.json`
  (handgezeichnet); die `.txt` in `multi_lego_yolo` hat nur Klasse 0.
- Geschnitten wird an der JSON-Box mit 6 % Rand. Die passende Detektor-Box
  und ihr IoU stehen im `manifest.csv` (Median 1,00, Minimum 0,72).
- Ergebnis: **623 Crops in 19 Klassen** in `cuttet_multi_lego_yolo/`, dazu
  `classes.txt` und `manifest.csv`.
- Übersprungen: 5 Boxen mit `lego_brick`/`30062` (keine Teilenummer) und
  4 Detektor-Boxen ohne JSON-Partner.
- `3001` ist enthalten, `3009` nicht (in den 73 Fotos nicht vorhanden).

Als Datensatz nutzbar mit
`build_dataset.py --source cuttet_multi_lego_yolo` (Bilder ohne `.txt` gelten
dann als Ganzbild).

> Achtung: Das sind genau die 73 Fotos, die bisher als unabhängiger
> Handyfoto-Test dienten. Wer damit trainiert, hat danach keinen
> unabhängigen Handyfoto-Test mehr.

## 5. Zweites Training: beide Quellen gemischt (20 Klassen)

```bash
python projectV1/classification/build_dataset.py \
    --source Legosteine_comp --source cuttet_multi_lego_yolo \
    --group-by-source --repeats 1 \
    --out projectV1/classification/data_merged --force

python projectV1/classification/train.py \
    --model projectV1/models/cls_legosteine-19_yolo26s-cls_224_2026-11-16.pt \
    --data projectV1/classification/data_merged \
    --name cls_legosteine_s_merged --dataset legosteine-20 --arch yolo26s-cls \
    --lr0 5e-4 --epochs 60 --patience 15
```

- **20 Klassen** = Vereinigung beider Quellen: alle 19 aus `Legosteine_comp`
  plus `3001` (nur in den Handyfotos), `3009` bleibt (nur im Studio).
- **Split nach Quellfoto** (`--group-by-source`): ein Foto liefert bis zu 51
  Crops, die dürfen nicht auf beide Seiten verteilt werden. Ergebnis:
  508 / 144 / 75, **kein Foto in zwei Splits**.
- Zusätzlich **Mindestquote 50 % pro Klasse im Training** (aufgerundet) und
  je ein Bild in `val`/`test`. `3009` hat nur 5 Bilder → 3/1/1.
- **Warm Start** vom 19-Klassen-Modell, `lr0=5e-4` (halbe Lernrate, weil der
  Rumpf schon trainiert ist). Der Klassifikationskopf wird wegen 19 → 20
  Klassen neu initialisiert, alles davor kommt aus den alten Gewichten.
- Keine Spiegelung wie oben.

Modell: `projectV1/models/cls_legosteine-20_yolo26s-cls_224_2026-11-16.pt`

| Split | Bilder | Top-1 | Top-5 |
|---|---|---|---|
| `val` | 144 | **0,715** | **0,938** |
| `test` | 75 | **0,720** | **0,973** |

27 Epochen (bestes bei 12), Early Stop, 4,5 min auf MPS.

Nach Herkunft auf demselben Testsplit:

| Modell | gesamt | Handyfotos (62) | Studio (13) |
|---|---|---|---|
| neu, 20 Klassen | **0,720** | **0,677** | 0,923 |
| alt, 19 Klassen | 0,253 | 0,097 | 1,000 |

Der Vergleich ist nur bei den Handyfotos sauber: die hat das alte Modell nie
gesehen. Dort springt es von 0,097 auf 0,677. Die Studio-Spalte ist beim
alten Modell geschönt – 11 der 13 Fotos waren in seinem Training. Sauber
ist für das neue Modell nur die eigene Spalte, und da bleiben die
Studio-Bilder bei 0,923: das Handyfoto-Training hat sie nicht kaputtgemacht.

### Was hier nicht stimmt(e)

- **Der alte `data/test` ist verbraucht:** 15 seiner 19 Fotos liegen jetzt im
  Training. Die früheren 0,895 lassen sich mit diesem Modell nicht mehr
  nachprüfen, und ein Vergleich „neues Modell auf `data/test`“ wäre wertlos.
- Handyfoto-Zahlen beruhen auf **73 Fotos**; alle Crops eines Fotos hängen
  zusammen (gleiches Licht, gleicher Hintergrund). 0,677 heißt „auf diesen
  73 Fotos“, nicht „auf Handyfotos allgemein“.
- `3001` hat 4 Testbilder, `3009` genau 1. Ein Fehler ist dort 25 bzw. 100
  Prozentpunkte.
- Die häufigsten Verwechslungen sind Nachbargrößen: `3001`↔`3002`/`3004`,
  `3020`↔`3021`, `3622`↔`3710`. Plausibel, aber mit dieser Datenmenge nicht
  behebbar.

## Hinweis zum alten 448-Klassen-Lauf

Der frühere Datensatz `projectV1/dataset/cls_clean`, der Run
`projectV1/runs/cls_yolo26m_clean` und die 70/20/10-Zwischenversion wurden
auf Wunsch gelöscht. Es gibt jetzt genau einen Klassifikations-Datensatz
und einen Lauf: diesen hier.
