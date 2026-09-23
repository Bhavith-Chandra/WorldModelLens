"""Regression tests for the shared Task 1/3 ImageNet migration."""

from pathlib import Path

import numpy as np
from PIL import Image

from examples.ijepa.evaluate_task1_deletion_insertion import sample_context_and_target
from examples.ijepa.evaluate_task3_category_heterogeneity import compute_top_k_jaccard
from world_model_lens.data import load_imagenet_image, load_imagenet_subset


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((12, 20, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_balanced_subset_is_deterministic(tmp_path):
    for class_index in range(3):
        for image_index in range(3):
            _write_image(
                tmp_path / "val" / f"class_{class_index}" / f"{image_index}.png",
                value=class_index * 30 + image_index,
            )

    first = load_imagenet_subset(tmp_path, num_samples=6, num_classes=3, seed=7)
    second = load_imagenet_subset(tmp_path, num_samples=6, num_classes=3, seed=7)

    assert first == second
    assert len(first) == 6
    assert {sample["class_name"] for sample in first} == {
        "class_0", "class_1", "class_2"
    }
    assert all(
        sum(sample["class_name"] == name for sample in first) == 2
        for name in ("class_0", "class_1", "class_2")
    )


def test_balanced_subset_distributes_non_divisible_remainder(tmp_path):
    for class_index in range(3):
        for image_index in range(4):
            _write_image(
                tmp_path / "val" / f"class_{class_index}" / f"{image_index}.png",
                value=class_index * 30 + image_index,
            )

    samples = load_imagenet_subset(tmp_path, num_samples=8, num_classes=3, seed=7)
    counts = sorted(
        sum(sample["class_name"] == name for sample in samples)
        for name in {sample["class_name"] for sample in samples}
    )

    assert len(samples) == 8
    assert counts == [2, 3, 3]


def test_shared_preprocessing_resizes_to_square(tmp_path):
    path = tmp_path / "rectangular.png"
    _write_image(path, value=128)

    tensor = load_imagenet_image(path, image_size=224)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype.is_floating_point


def test_vith14_mask_split_uses_256_patches():
    context, target, hidden_count = sample_context_and_target(256, seed=42)

    assert len(context) == 51
    assert hidden_count == 205
    assert target not in context
    assert all(0 <= patch_id < 256 for patch_id in context)


def test_task3_uses_true_top_k_jaccard():
    attention = np.asarray([9.0, 8.0, 1.0, 0.0])
    attribution = np.asarray([9.0, 1.0, 8.0, 0.0])

    assert compute_top_k_jaccard(attention, attribution, k=2) == 1.0 / 3.0
