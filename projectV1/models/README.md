# `models/` – benannte Trainingsergebnisse

Problem vorher: In jedem Lauf heißt das beste Modell `weights/best.pt`. Im
Dateisystem sieht man nicht, welches Training dahintersteckt, und Kopien
heißen `best (1).pt`, `best_neu.pt` usw.

Hier liegt deshalb **jedes** veröffentlichte Modell einmal mit sprechendem
Namen, dazu eine JSON-Datei mit der Herkunft und ein gemeinsames Manifest.

```
projectV1/models/
├── publish.py                                        # Lauf -> benanntes Modell
├── manifest.json                                     # alle Modelle + Metriken
├── cls_legosteine-19_yolo26s-cls_224_2026-11-16.pt
└── cls_legosteine-19_yolo26s-cls_224_2026-11-16.json
```

## Namensschema

```
<task>_<datensatz>_<architektur>_<imgsz>_<datum>.pt
```

| Teil | Beispiel | Bedeutung |
|---|---|---|
| task | `cls` | `cls`, `det`, `seg`, `obb` |
| datensatz | `legosteine-19` | Kurzname + Klassenzahl |
| architektur | `yolo26s-cls` | Basisgewichte ohne `.pt` |
| imgsz | `224` | Trainingsauflösung |
| datum | `2026-11-16` | Tag, an dem `best.pt` geschrieben wurde |

Kollidiert ein Name (zweiter Lauf am selben Tag mit gleicher Konfiguration),
wird `-2`, `-3`, … angehängt. Wird derselbe Lauf erneut veröffentlicht,
erkennt das Skript den identischen SHA-256 und legt **keine** zweite Kopie an.

## Benutzen

```bash
# Lauf veröffentlichen
python projectV1/models/publish.py \
  --run projectV1/classification/runs/yolo26m_cls_legosteine \
  --dataset legosteine-19

# nur zeigen, was passieren würde
python projectV1/models/publish.py --run ... --dry-run

# alle Modelle mit Metriken
python projectV1/models/publish.py --list
```

`projectV1/classification/train.py` ruft das nach dem Training automatisch
auf (`--dataset legosteine-19`, abschaltbar mit `--no-publish`). Der
Test-Code kann statt eines Pfads `--weights latest` nehmen und bekommt dann
das zuletzt veröffentlichte Modell.

## Was in der JSON-Datei steht

`file`, `sha256`, `task`, `dataset`, `classes` (alle Namen), `arch`,
`base_weights` + `base_weights_sha256`, `imgsz`, `epochs`, `metrics`
(Top-1/Top-5 bzw. mAP, bestes Epoche, Dauer), `run_dir`, `git_commit`,
`published_at`.

Wichtig für die Frage „welches Modell aus welchem Training“: `base_weights_sha256`
belegt, von welcher Datei das Training gestartet ist, `run_dir` zeigt auf den
vollständigen Lauf mit `args.yaml` und `results.csv`, `sha256` auf die
ausgelieferte Datei selbst.

## Basisgewichte

Die offiziellen Ultralytics-Gewichte werden **einmal** ins Repo-Wurzelverzeichnis
geladen und dann wiederverwendet – nicht pro Lauf neu:

| Datei | SHA-256 | Rolle |
|---|---|---|
| `yolo26s-cls.pt` | `81679002…` | ImageNet-Basis des ausgelieferten Modells |
| `yolo26m-cls.pt` | `9f6546f3…` | größere Alternative, im Vergleich unterlegen |

Ultralytics lädt nur nach, wenn die Datei fehlt. Jeder Lauf ersetzt lediglich
den Klassifikationskopf (`nc=1000` → `nc=19`), die Rumpfgewichte kommen aus
derselben Datei. Deshalb steht der Hash in jedem Modell-JSON.

## Was nicht hierher gehört

- `yolo26m.pt`, `yolo26l.pt`, `yolo26s.pt`, `yolo26x.pt`, `yolo26m-obb.pt`,
  `yolov8s-worldv2.pt`: unveränderte Basisgewichte, keine Trainingsergebnisse.
- `helper/best.pt`: altes Detektionsmodell (1 Klasse, `class_0`) ohne
  dokumentierten Lauf – Herkunft unbekannt, daher bewusst nicht im Manifest.
  Wer es braucht, veröffentlicht es mit einem sprechenden `--dataset`-Namen.

`*.pt` ist in `.gitignore`, die Modelldateien bleiben also lokal. Versioniert
werden `publish.py`, `manifest.json` und die `.json`-Sidecars – die beschreiben,
was es gibt und wie es entstanden ist, ohne 20-MB-Binaries im Git.
