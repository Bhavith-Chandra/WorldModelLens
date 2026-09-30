"""Small, dependency-light ImageNet loading helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def load_imagenet_subset(
    imagenet_root: str | Path,
    num_samples: int = 1000,
    num_classes: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return a balanced, deterministic ImageNet subset manifest."""
    if num_samples <= 0 or num_classes <= 1:
        raise ValueError("num_samples must be positive and num_classes must exceed one")
    if num_samples % num_classes != 0:
        raise ValueError("num_samples must be divisible by num_classes")

    root = Path(imagenet_root).expanduser().resolve()
    split = None
    for candidate in (root / "val", root / "train", root):
        if candidate.is_dir() and any(path.is_dir() for path in candidate.iterdir()):
            split = candidate
            break
    if split is None:
        raise FileNotFoundError(f"No ImageNet class directories found below {root}")

    per_class = num_samples // num_classes
    available: list[tuple[str, list[Path]]] = []
    for class_dir in sorted(path for path in split.iterdir() if path.is_dir()):
        paths = sorted(
            path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(paths) >= per_class:
            available.append((class_dir.name, paths))
    if len(available) < num_classes:
        raise ValueError(
            f"Found {len(available)} eligible classes; need {num_classes} "
            f"with at least {per_class} images each"
        )

    rng = random.Random(seed)
    chosen = sorted(rng.sample(available, num_classes))
    samples: list[dict[str, Any]] = []
    for label, (class_name, paths) in enumerate(chosen):
        for path in rng.sample(paths, per_class):
            samples.append({"path": str(path), "label": label, "class_name": class_name})
    rng.shuffle(samples)
    return samples


def load_imagenet_image(path: str | Path, image_size: int = 224) -> torch.Tensor:
    """Load, resize, normalize, and batch one ImageNet image for I-JEPA."""
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return ((tensor - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)
