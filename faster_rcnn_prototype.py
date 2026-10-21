"""Small Faster R-CNN prototype for the supplemental LEGO image collection.

The current collection has one labeled brick image per file but no bounding-box
annotations. Therefore, this prototype uses the complete image as one box.
Replace the full-image target construction in ``__getitem__`` with real
annotations when they are available.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class LegoFolderDataset(Dataset[tuple[Tensor, dict[str, Any]]]):
    """Dataset with the folder name as the class label and a full-image box."""

    def __init__(
        self,
        root: Path,
        samples: list[tuple[Path, int]],
        class_names: list[str],
    ) -> None:
        self.root = root
        self.samples = samples
        self.class_names = class_names

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Any]]:
        image_path, class_index = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = pil_to_tensor(image).float() / 255.0
        width, height = image.size
        target = {
            "boxes": torch.tensor(
                [[0.0, 0.0, float(max(1, width - 1)), float(max(1, height - 1))]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([class_index + 1], dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor([float(width * height)], dtype=torch.float32),
            "iscrowd": torch.zeros((1,), dtype=torch.int64),
        }
        return image_tensor, target


def discover_samples(
    root: Path, validation_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[str]]:
    """Find images and split each class independently to preserve class coverage."""
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


def collate_fn(
    batch: list[tuple[Tensor, dict[str, Any]]],
) -> tuple[list[Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_model(num_classes: int, pretrained: bool) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[tuple[list[Tensor], list[dict[str, Any]]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(
            f"  batch {batch_index}/{len(loader)} loss={loss.item():.4f}",
            flush=True,
        )
    return total_loss / max(1, len(loader))


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader[tuple[list[Tensor], list[dict[str, Any]]]],
    device: torch.device,
) -> float:
    """Compute validation loss by temporarily enabling training-mode loss output."""
    model.train()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss_dict = model(images, targets)
            total_loss += sum(loss_dict.values()).item()
    return total_loss / max(1, len(loader))


def benchmark(
    model: torch.nn.Module,
    dataset: LegoFolderDataset,
    device: torch.device,
    confidence: float,
    output_path: Path,
) -> None:
    model.eval()
    image, _ = dataset[0]
    image_batch = [image.to(device)]
    with torch.no_grad():
        for _ in range(2):
            model(image_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predictions = model(image_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

    prediction = predictions[0]
    kept = prediction["scores"] >= confidence
    image_pil = Image.open(dataset.samples[0][0]).convert("RGB")
    draw = ImageDraw.Draw(image_pil)
    for box, label, score in zip(
        prediction["boxes"][kept].cpu(),
        prediction["labels"][kept].cpu(),
        prediction["scores"][kept].cpu(),
    ):
        coordinates = tuple(int(value) for value in box.tolist())
        draw.rectangle(coordinates, outline="red", width=3)
        class_name = dataset.class_names[int(label) - 1] if int(label) <= len(dataset.class_names) else "unknown"
        draw.text((coordinates[0], coordinates[1]), f"{class_name}: {score:.2f}", fill="red")
    image_pil.save(output_path)
    print(f"Inference time: {elapsed_ms:.1f} ms/image")
    print(f"Detections above {confidence:.2f}: {int(kept.sum())}")
    print(f"Annotated result: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Legosteine_comp"))
    parser.add_argument("--epochs", type=int, default=2, help="Set to 0 with --benchmark-only.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("faster_rcnn_result.jpg"))
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for a quick smoke test.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Skip training and measure one inference pass with an initialized model.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 0 or args.batch_size < 1:
        raise ValueError("--epochs must be non-negative and --batch-size must be positive")
    if args.epochs == 0 and not args.benchmark_only:
        raise ValueError("--epochs=0 requires --benchmark-only")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available() else "cpu"
    )
    train_samples, validation_samples, class_names = discover_samples(
        args.data_dir, args.validation_fraction, args.seed
    )
    if args.max_images:
        train_samples = train_samples[: args.max_images]
        validation_samples = validation_samples[: max(1, args.max_images // 4)]
    train_dataset = LegoFolderDataset(args.data_dir, train_samples, class_names)
    validation_dataset = LegoFolderDataset(args.data_dir, validation_samples, class_names)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    print(
        f"Device: {device}; classes: {len(class_names)}; "
        f"train: {len(train_dataset)}; validation: {len(validation_dataset)}",
        flush=True,
    )
    print("Building Faster R-CNN model...", flush=True)
    model = build_model(len(class_names) + 1, not args.no_pretrained).to(device)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    if args.benchmark_only:
        print("Benchmark-only mode: skipping training.", flush=True)
    else:
        for epoch in range(args.epochs):
            print(f"Starting epoch {epoch + 1}/{args.epochs}...", flush=True)
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            validation_loss = evaluate_loss(model, validation_loader, device)
            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, validation_loss={validation_loss:.4f}",
                flush=True,
            )
    print("Running timed inference...", flush=True)
    benchmark(model, validation_dataset, device, args.confidence, args.output)


if __name__ == "__main__":
    main()
