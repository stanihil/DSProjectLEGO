"""Small YOLO prototype using the supplemental LEGO image collection.

The collection has image-level labels but no bounding-box annotations. This
prototype therefore writes one full-image YOLO box per image. Replace the
generated labels with real annotations before using results as a pile detector.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def discover_samples(
    root: Path, validation_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[str]]:
    class_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not class_dirs:
        raise ValueError(f"No numeric part-number folders found in {root}")

    class_names = [path.name for path in class_dirs]
    train_samples: list[tuple[Path, int]] = []
    validation_samples: list[tuple[Path, int]] = []
    rng = random.Random(seed)
    for class_index, class_dir in enumerate(class_dirs):
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < 2:
            raise ValueError(f"Need at least two images in {class_dir}")
        rng.shuffle(images)
        validation_count = max(1, round(len(images) * validation_fraction))
        validation_samples.extend(
            (path, class_index) for path in images[:validation_count]
        )
        train_samples.extend(
            (path, class_index) for path in images[validation_count:]
        )
    return train_samples, validation_samples, class_names


def write_yolo_dataset(
    samples: list[tuple[Path, int]],
    class_names: list[str],
    split: str,
    output_dir: Path,
) -> None:
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for index, (source, class_index) in enumerate(samples):
        image_name = f"{index:04d}_{source.name}"
        destination = image_dir / image_name
        shutil.copy2(source, destination)
        label_path = label_dir / f"{Path(image_name).stem}.txt"
        label_path.write_text(
            f"{class_index} 0.5 0.5 1.0 1.0\n", encoding="utf-8"
        )

    if split == "train":
        yaml = (
            f"path: {output_dir.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            f"nc: {len(class_names)}\n"
            f"names: {class_names!r}\n"
        )
        (output_dir / "dataset.yaml").write_text(yaml, encoding="utf-8")


def benchmark(
    model: YOLO,
    image_path: Path,
    confidence: float,
    output_path: Path,
    device: str,
) -> None:
    for _ in range(2):
        model.predict(source=str(image_path), conf=confidence, device=device, verbose=False)
    start = time.perf_counter()
    results = model.predict(
        source=str(image_path), conf=confidence, device=device, verbose=False, save=False
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    results[0].save(filename=str(output_path))
    detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"Inference time: {elapsed_ms:.1f} ms/image")
    print(f"Detections above {confidence:.2f}: {detections}")
    print(f"Annotated result: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Legosteine_comp"))
    parser.add_argument("--work-dir", type=Path, default=Path(".yolo_dataset"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("yolo_result.jpg"))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    train, validation, class_names = discover_samples(
        args.data_dir, args.validation_fraction, args.seed
    )
    if args.max_images:
        train = train[: args.max_images]
        validation = validation[: max(1, args.max_images // 4)]
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    write_yolo_dataset(train, class_names, "train", args.work_dir)
    write_yolo_dataset(validation, class_names, "val", args.work_dir)

    model = YOLO("yolov8n.yaml" if args.no_pretrained else "yolov8n.pt")
    print(
        f"Device: {args.device}; classes: {len(class_names)}; "
        f"train: {len(train)}; validation: {len(validation)}"
    )
    model.train(
        data=str(args.work_dir / "dataset.yaml"),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.image_size,
        device=args.device,
        project="runs",
        name="yolo_prototype",
        exist_ok=True,
        verbose=False,
    )
    benchmark(model, validation[0][0], args.confidence, args.output, args.device)


if __name__ == "__main__":
    main()
