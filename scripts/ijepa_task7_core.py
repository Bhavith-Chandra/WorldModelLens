"""Shared execution and metric primitives for the I-JEPA Task 7 scripts.

The Task 7 entry points deliberately keep plotting and question-specific
analysis in ``scripts/``.  This module owns the expensive/repeated parts:
deterministic data selection, model loading, fixed masks, prediction-error
collection, calibration PCA, and the three raw evaluation metrics.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import rankdata

from world_model_lens.data import load_imagenet_image, load_imagenet_subset
from world_model_lens.hub.model_hub import ModelHub

RAW_METRIC_KEYS = (
    "predictor_mse",
    "predictor_error_mahalanobis_zero",
    "predictor_error_mahalanobis_centered",
)


def base_config(output_root: str) -> dict[str, Any]:
    """Return the settings shared by every Task 7 entry point."""
    return {
        "MODEL_NAME": "ijepa-vit-h-in1k",
        "CHECKPOINT_PATH": None,
        "CACHE_DIR": None,
        "FORCE_DOWNLOAD": False,
        "IMAGENET_ROOT": "/content/imagenet/val",
        "OUTPUT_ROOT": output_root,
        "NUM_SAMPLES": 1000,
        "NUM_CLASSES": 50,
        "CALIBRATION_FRACTION": 0.5,
        "SEED": 42,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "PRECISION": "fp16" if torch.cuda.is_available() else "fp32",
        "BATCH_SIZE": 16,
        "TARGET_PATCH_SIDE": 4,
        "RIDGE_FRACTION": 1e-4,
        "PLOT_DPI": 180,
        "SHOW_PLOTS": False,
    }


def validate_base_config(config: dict[str, Any]) -> None:
    """Validate constraints shared by the Task 7 experiments."""
    num_samples = int(config["NUM_SAMPLES"])
    num_classes = int(config["NUM_CLASSES"])
    if num_samples <= 0 or num_classes <= 1:
        raise ValueError("NUM_SAMPLES must be positive and NUM_CLASSES must exceed one")
    if num_samples % num_classes:
        raise ValueError("NUM_SAMPLES must be divisible by NUM_CLASSES")
    if num_samples // num_classes < 4:
        raise ValueError("Task 7 needs at least four images per class")
    fraction = float(config["CALIBRATION_FRACTION"])
    if not 0.0 < fraction < 1.0:
        raise ValueError("CALIBRATION_FRACTION must lie strictly between zero and one")
    if int(config["BATCH_SIZE"]) <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    if float(config["RIDGE_FRACTION"]) < 0.0:
        raise ValueError("RIDGE_FRACTION cannot be negative")
    if config["PRECISION"] == "fp16" and not str(config["DEVICE"]).startswith("cuda"):
        raise ValueError("fp16 requires a CUDA device; use fp32 on CPU")


def seed_everything(config: dict[str, Any]) -> None:
    """Seed subset selection, splitting, and randomized PCA."""
    seed = int(config["SEED"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_directory(config: dict[str, Any]) -> Path:
    """Create a timestamped output directory and record its exact settings."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["OUTPUT_ROOT"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logged = dict(config)
    logged["RUN_ID"] = run_id
    logged["CREATED_AT_UTC"] = datetime.now(timezone.utc).isoformat()
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(logged, handle, sort_keys=False)
    return run_dir


def save_json(path: Path, value: Any) -> None:
    """Write a readable JSON artifact."""
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the balanced deterministic ImageNet subset in ``config``."""
    return load_imagenet_subset(
        config["IMAGENET_ROOT"],
        num_samples=int(config["NUM_SAMPLES"]),
        num_classes=int(config["NUM_CLASSES"]),
        seed=int(config["SEED"]),
    )


def stratified_split(
    samples: list[dict[str, Any]], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split every class independently into calibration and evaluation images."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        by_label.setdefault(int(sample["label"]), []).append(index)

    calibration: list[int] = []
    evaluation: list[int] = []
    for indices in by_label.values():
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        calibration.extend(indices[:count])
        evaluation.extend(indices[count:])
    rng.shuffle(calibration)
    rng.shuffle(evaluation)
    return calibration, evaluation


def split_from_config(
    samples: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[int], list[int]]:
    return stratified_split(
        samples,
        fraction=float(config["CALIBRATION_FRACTION"]),
        seed=int(config["SEED"]),
    )


def save_dataset_manifest(
    run_dir: Path,
    samples: list[dict[str, Any]],
    calibration_indices: Iterable[int],
) -> None:
    """Persist sample identity and split membership for corpus oversight."""
    calibration = set(calibration_indices)
    manifest = [
        {
            **sample,
            "sample_index": index,
            "split": "calibration" if index in calibration else "evaluation",
        }
        for index, sample in enumerate(samples)
    ]
    save_json(run_dir / "dataset_manifest.json", manifest)


def load_world_model(config: dict[str, Any]) -> Any:
    """Load the official context encoder, target encoder, and predictor."""
    checkpoint_path = config.get("CHECKPOINT_PATH")
    if checkpoint_path:
        adapter = ModelHub.load_checkpoint(
            checkpoint_path, backend="ijepa", device=config["DEVICE"]
        )
    else:
        adapter = ModelHub.load(
            config["MODEL_NAME"],
            cache_dir=config.get("CACHE_DIR"),
            device=config["DEVICE"],
            force_download=bool(config["FORCE_DOWNLOAD"]),
        )
    if config["PRECISION"] == "fp16":
        adapter = adapter.half()
    adapter.eval()
    return adapter


def model_device_dtype(adapter: Any) -> tuple[torch.device, torch.dtype]:
    parameter = next(adapter.context_encoder.parameters())
    return parameter.device, parameter.dtype


def build_fixed_masks(adapter: Any, target_patch_side: int) -> tuple[list[int], list[int]]:
    """Construct complementary context indices and a central square target."""
    num_patches = int(adapter.context_encoder.patch_embed.n_patches)
    grid = int(math.sqrt(num_patches))
    side = int(target_patch_side)
    if grid * grid != num_patches or side <= 0 or side >= grid:
        raise ValueError("TARGET_PATCH_SIDE is invalid for the checkpoint patch grid")
    start = (grid - side) // 2
    target = {
        row * grid + column
        for row in range(start, start + side)
        for column in range(start, start + side)
    }
    return [index for index in range(num_patches) if index not in target], sorted(target)


def load_image_batch(samples: list[dict[str, Any]], adapter: Any) -> torch.Tensor:
    """Load and normalize one ImageNet batch."""
    device, dtype = model_device_dtype(adapter)
    images = [load_imagenet_image(sample["path"], image_size=224) for sample in samples]
    return torch.cat(images, dim=0).to(device=device, dtype=dtype)


@torch.no_grad()
def collect_prediction_errors(
    adapter: Any,
    samples: list[dict[str, Any]],
    context_ids: list[int],
    target_ids: list[int],
    batch_size: int,
) -> torch.Tensor:
    """Return final predictor-minus-target errors as ``[image, patch, dim]``."""
    chunks: list[torch.Tensor] = []
    total_batches = math.ceil(len(samples) / batch_size)
    for batch_number, start in enumerate(range(0, len(samples), batch_size), start=1):
        batch_samples = samples[start : start + batch_size]
        observations = load_image_batch(batch_samples, adapter)
        context = adapter.context_encoder(observations, patch_ids=context_ids)
        prediction = adapter.predictor(context, context_ids, target_ids)
        target = adapter.target_encoder(observations)[:, target_ids, :]
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Predictor shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )
        chunks.append((prediction - target).to(device="cpu", dtype=torch.float32))
        print(f"  predictor errors: batch {batch_number}/{total_batches}", flush=True)
    return torch.cat(chunks, dim=0)


def fit_error_pca(
    errors: torch.Tensor,
    components: int,
    device: str | torch.device,
    power_iterations: int,
    ridge_fraction: float,
) -> dict[str, Any]:
    """Fit regularized PCA to calibration predictor-error vectors."""
    vectors = errors.reshape(-1, errors.shape[-1]).to(device=device, dtype=torch.float32)
    mean = vectors.mean(dim=0)
    centered = vectors - mean
    used = min(int(components), centered.shape[0], centered.shape[1])
    if used < 2:
        raise ValueError("Not enough predictor-error vectors for a two-component PCA")
    _, singular_values, directions = torch.pca_lowrank(
        centered, q=used, center=False, niter=int(power_iterations)
    )
    eigenvalues = singular_values.square() / max(1, centered.shape[0] - 1)
    total_variance = centered.var(dim=0, unbiased=True).sum()
    ridge = float(ridge_fraction) * eigenvalues.mean()
    return {
        "mean": mean,
        "components": directions,
        "eigenvalues": eigenvalues,
        "explained_ratio": eigenvalues / total_variance.clamp_min(1e-12),
        "ridge": float(ridge),
        "precision_denominator": eigenvalues + ridge,
        "components_used": used,
        "observations": int(vectors.shape[0]),
        "embedding_dim": int(vectors.shape[1]),
    }


@torch.no_grad()
def project_errors(
    errors: torch.Tensor, pca: dict[str, Any], batch_size: int, device: str | torch.device
) -> dict[str, torch.Tensor]:
    """Project errors and return minimal coordinates needed for later analysis."""
    mean = pca["mean"].to(device)
    components = pca["components"].to(device)
    zero_chunks: list[torch.Tensor] = []
    centered_chunks: list[torch.Tensor] = []
    mse_chunks: list[torch.Tensor] = []
    total_squared_chunks: list[torch.Tensor] = []
    centered_total_squared_chunks: list[torch.Tensor] = []
    for start in range(0, errors.shape[0], batch_size):
        batch = errors[start : start + batch_size].to(device=device, dtype=torch.float32)
        flat = batch.reshape(-1, batch.shape[-1])
        centered = flat - mean
        zero_chunks.append((flat @ components).reshape(batch.shape[0], batch.shape[1], -1).cpu())
        centered_chunks.append(
            (centered @ components).reshape(batch.shape[0], batch.shape[1], -1).cpu()
        )
        mse_chunks.append(batch.square().mean(dim=(1, 2)).cpu())
        total_squared_chunks.append(batch.square().sum(dim=(1, 2)).cpu())
        centered_total_squared_chunks.append(
            centered.square().reshape(batch.shape[0], batch.shape[1], -1).sum(dim=(1, 2)).cpu()
        )
    return {
        "zero_coordinates": torch.cat(zero_chunks),
        "centered_coordinates": torch.cat(centered_chunks),
        "predictor_mse": torch.cat(mse_chunks),
        "total_squared_error": torch.cat(total_squared_chunks),
        "total_centered_squared_error": torch.cat(centered_total_squared_chunks),
    }


def raw_metrics_from_projection(
    projected: dict[str, torch.Tensor], pca: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Return the three raw Task 7 scalar metrics, one value per image."""
    denominator = pca["precision_denominator"].detach().cpu()
    zero = projected["zero_coordinates"].square() / denominator
    centered = projected["centered_coordinates"].square() / denominator
    return {
        "predictor_mse": projected["predictor_mse"],
        "predictor_error_mahalanobis_zero": zero.sum(dim=2).mean(dim=1).sqrt(),
        "predictor_error_mahalanobis_centered": centered.sum(dim=2).mean(dim=1).sqrt(),
    }


def descending_ranks(values: dict[int, float]) -> dict[int, int]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
    return {label: rank for rank, (label, _) in enumerate(ordered, start=1)}


def ordinal_ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks so tied values do not receive arbitrary ordering."""
    return np.asarray(rankdata(values, method="average"), dtype=np.float64)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation for continuous values or precomputed rank vectors."""
    if left.size < 2:
        return float("nan")
    left_ranks = ordinal_ranks(left)
    right_ranks = ordinal_ranks(right)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])
