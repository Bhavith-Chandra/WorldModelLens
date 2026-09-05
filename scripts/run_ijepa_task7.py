#!/usr/bin/env python3
"""Task 7: I-JEPA prediction-error geometry under latent anisotropy.

For each ImageNet image, this experiment subtracts the target-encoder patch
embeddings from the corresponding predictor embeddings::

    error[image, patch, dim] = prediction - target

PCA is fitted to patch-level errors from a stratified calibration split. The
held-out errors are then scored with ordinary MSE and precision-weighted
Mahalanobis distance in the retained PCA subspace. Category rankings under the
two metrics reveal whether conclusions based on isotropic MSE are sensitive to
the directional geometry of I-JEPA's embedding space.

Two Mahalanobis scores are reported:

* ``mahalanobis_zero`` measures error relative to perfect prediction (zero).
* ``mahalanobis_centered`` measures atypicality relative to the model's average
  error, removing systematic prediction bias.

Edit CONFIG below, then run this file. Every setting and result is written to a
timestamped output directory.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from world_model_lens.data import load_imagenet_image, load_imagenet_subset
from world_model_lens.hub.model_hub import ModelHub


# ---------------------------------------------------------------------------
# Edit experiment arguments here. Every value is copied to config.yaml.
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    "MODEL_NAME": "ijepa-vit-h-in1k",
    "CHECKPOINT_PATH": None,
    "CACHE_DIR": None,
    "FORCE_DOWNLOAD": False,
    "IMAGENET_ROOT": "/content/imagenet/val",
    "OUTPUT_ROOT": "outputs/ijepa_task7",
    "NUM_SAMPLES": 1000,
    # Fifty classes gives 20 images per class. Half are used to fit the error
    # geometry and half to estimate category-level failure rankings.
    "NUM_CLASSES": 50,
    "CALIBRATION_FRACTION": 0.5,
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PRECISION": "fp16" if torch.cuda.is_available() else "fp32",
    "BATCH_SIZE": 16,
    # Every image predicts the same central target block, keeping patch
    # positions comparable across the dataset.
    "TARGET_PATCH_SIDE": 4,
    # Randomized low-rank PCA. Increase toward the embedding dimension for a
    # more complete covariance model at additional compute cost.
    "PCA_COMPONENTS": 256,
    "PCA_POWER_ITERATIONS": 4,
    # Added to every retained eigenvalue as RIDGE_FRACTION * mean(eigenvalue).
    "RIDGE_FRACTION": 1e-4,
    "TOP_K_CATEGORIES": 10,
    "ANNOTATE_RANK_SHIFTS": 8,
    "PLOT_DPI": 180,
    "SHOW_PLOTS": True,
}


METRIC_KEYS = (
    "prediction_mse",
    "prediction_rmse",
    "prediction_l2",
    "mahalanobis_zero",
    "mahalanobis_centered",
    "retained_centered_energy_fraction",
)


def validate_config() -> None:
    """Validate inexpensive configuration constraints before loading anything."""
    num_samples = int(CONFIG["NUM_SAMPLES"])
    num_classes = int(CONFIG["NUM_CLASSES"])
    if num_samples <= 0 or num_classes <= 1:
        raise ValueError("NUM_SAMPLES must be positive and NUM_CLASSES must exceed one")
    if num_samples % num_classes:
        raise ValueError("NUM_SAMPLES must be divisible by NUM_CLASSES")
    if num_samples // num_classes < 4:
        raise ValueError("Task 7 needs at least four images per class for its held-out split")
    fraction = float(CONFIG["CALIBRATION_FRACTION"])
    if not 0.0 < fraction < 1.0:
        raise ValueError("CALIBRATION_FRACTION must lie strictly between zero and one")
    if int(CONFIG["BATCH_SIZE"]) <= 0 or int(CONFIG["PCA_COMPONENTS"]) <= 1:
        raise ValueError("BATCH_SIZE must be positive and PCA_COMPONENTS must exceed one")
    if float(CONFIG["RIDGE_FRACTION"]) < 0.0:
        raise ValueError("RIDGE_FRACTION cannot be negative")
    if CONFIG["PRECISION"] == "fp16" and not str(CONFIG["DEVICE"]).startswith("cuda"):
        raise ValueError("fp16 requires a CUDA device; use fp32 on CPU")


def seed_everything() -> None:
    """Seed subset selection, splitting, and randomized PCA."""
    seed = int(CONFIG["SEED"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_directory() -> Path:
    """Create a timestamped output directory and record the exact configuration."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(CONFIG["OUTPUT_ROOT"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logged_config = dict(CONFIG)
    logged_config["RUN_ID"] = run_id
    logged_config["CREATED_AT_UTC"] = datetime.now(timezone.utc).isoformat()
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(logged_config, handle, sort_keys=False)
    return run_dir


def load_world_model() -> Any:
    """Load the official context encoder, target encoder, and predictor via ModelHub."""
    checkpoint_path = CONFIG.get("CHECKPOINT_PATH")
    if checkpoint_path:
        adapter = ModelHub.load_checkpoint(
            checkpoint_path, backend="ijepa", device=CONFIG["DEVICE"]
        )
    else:
        adapter = ModelHub.load(
            CONFIG["MODEL_NAME"],
            cache_dir=CONFIG.get("CACHE_DIR"),
            device=CONFIG["DEVICE"],
            force_download=bool(CONFIG["FORCE_DOWNLOAD"]),
        )
    if CONFIG["PRECISION"] == "fp16":
        adapter = adapter.half()
    adapter.eval()
    return adapter


def model_device_dtype(adapter: Any) -> tuple[torch.device, torch.dtype]:
    """Return the adapter parameter device and dtype."""
    parameter = next(adapter.context_encoder.parameters())
    return parameter.device, parameter.dtype


def build_fixed_masks(adapter: Any) -> tuple[list[int], list[int]]:
    """Construct complementary context indices and a central square target block."""
    num_patches = int(adapter.context_encoder.patch_embed.n_patches)
    grid = int(math.sqrt(num_patches))
    side = int(CONFIG["TARGET_PATCH_SIDE"])
    if grid * grid != num_patches or side <= 0 or side >= grid:
        raise ValueError("TARGET_PATCH_SIDE is invalid for the checkpoint patch grid")
    start = (grid - side) // 2
    target = {
        row * grid + column
        for row in range(start, start + side)
        for column in range(start, start + side)
    }
    context = [index for index in range(num_patches) if index not in target]
    return context, sorted(target)


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
) -> torch.Tensor:
    """Return prediction-minus-target errors shaped [images, patches, embedding]."""
    chunks: list[torch.Tensor] = []
    batch_size = int(CONFIG["BATCH_SIZE"])
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
        print(f"  prediction errors: batch {batch_number}/{total_batches}", flush=True)
    return torch.cat(chunks, dim=0)


def stratified_split(samples: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Split every class independently into PCA-calibration and evaluation images."""
    rng = random.Random(int(CONFIG["SEED"]))
    by_label: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        by_label.setdefault(int(sample["label"]), []).append(index)

    calibration: list[int] = []
    evaluation: list[int] = []
    fraction = float(CONFIG["CALIBRATION_FRACTION"])
    for indices in by_label.values():
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        calibration.extend(indices[:count])
        evaluation.extend(indices[count:])
    rng.shuffle(calibration)
    rng.shuffle(evaluation)
    return calibration, evaluation


def fit_error_pca(errors: torch.Tensor) -> dict[str, torch.Tensor | float | int]:
    """Fit regularized low-rank PCA to patch-level calibration errors."""
    device = torch.device(CONFIG["DEVICE"])
    vectors = errors.reshape(-1, errors.shape[-1]).to(device=device, dtype=torch.float32)
    mean = vectors.mean(dim=0)
    centered = vectors - mean
    max_components = min(centered.shape[0], centered.shape[1])
    components_requested = int(CONFIG["PCA_COMPONENTS"])
    components_used = min(components_requested, max_components)
    if components_used < 2:
        raise ValueError("Not enough error vectors to fit at least two PCA components")

    _, singular_values, components = torch.pca_lowrank(
        centered,
        q=components_used,
        center=False,
        niter=int(CONFIG["PCA_POWER_ITERATIONS"]),
    )
    eigenvalues = singular_values.square() / max(1, centered.shape[0] - 1)
    total_variance = centered.var(dim=0, unbiased=True).sum()
    explained_ratio = eigenvalues / total_variance.clamp_min(torch.finfo(torch.float32).eps)
    ridge = float(CONFIG["RIDGE_FRACTION"]) * eigenvalues.mean()
    precision_denominator = eigenvalues + ridge

    return {
        "mean": mean,
        "components": components,
        "eigenvalues": eigenvalues,
        "explained_ratio": explained_ratio,
        "precision_denominator": precision_denominator,
        "ridge": float(ridge),
        "components_used": components_used,
        "observations": int(vectors.shape[0]),
        "embedding_dim": int(vectors.shape[1]),
    }


def score_evaluation_errors(
    errors: torch.Tensor,
    pca: dict[str, torch.Tensor | float | int],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Score held-out images and return metrics plus mean per-PC contributions."""
    device = torch.device(CONFIG["DEVICE"])
    mean = pca["mean"]
    components = pca["components"]
    denominator = pca["precision_denominator"]
    if not all(isinstance(value, torch.Tensor) for value in (mean, components, denominator)):
        raise TypeError("PCA tensor state is incomplete")

    metric_chunks: dict[str, list[torch.Tensor]] = {key: [] for key in METRIC_KEYS}
    first_two_chunks: list[torch.Tensor] = []
    contribution_sum = torch.zeros(components.shape[1], device=device)
    patch_count = 0
    batch_size = int(CONFIG["BATCH_SIZE"])

    for start in range(0, errors.shape[0], batch_size):
        batch = errors[start : start + batch_size].to(device=device, dtype=torch.float32)
        flat = batch.reshape(-1, batch.shape[-1])
        centered_flat = flat - mean
        zero_coordinates = (flat @ components).reshape(
            batch.shape[0], batch.shape[1], -1
        )
        centered_coordinates = (centered_flat @ components).reshape(
            batch.shape[0], batch.shape[1], -1
        )
        zero_contributions = zero_coordinates.square() / denominator
        centered_contributions = centered_coordinates.square() / denominator

        mse = batch.square().mean(dim=(1, 2))
        metric_chunks["prediction_mse"].append(mse.cpu())
        metric_chunks["prediction_rmse"].append(mse.sqrt().cpu())
        metric_chunks["prediction_l2"].append(
            batch.square().sum(dim=2).sqrt().mean(dim=1).cpu()
        )
        metric_chunks["mahalanobis_zero"].append(
            zero_contributions.sum(dim=2).mean(dim=1).sqrt().cpu()
        )
        metric_chunks["mahalanobis_centered"].append(
            centered_contributions.sum(dim=2).mean(dim=1).sqrt().cpu()
        )
        retained_energy = centered_coordinates.square().sum(dim=(1, 2))
        total_centered_energy = centered_flat.square().reshape(
            batch.shape[0], batch.shape[1], -1
        ).sum(dim=(1, 2))
        metric_chunks["retained_centered_energy_fraction"].append(
            (retained_energy / total_centered_energy.clamp_min(1e-12)).cpu()
        )
        first_two_chunks.append(centered_coordinates[:, :, :2].mean(dim=1).cpu())
        contribution_sum += zero_contributions.sum(dim=(0, 1))
        patch_count += batch.shape[0] * batch.shape[1]

    metrics = {key: torch.cat(chunks) for key, chunks in metric_chunks.items()}
    metrics["pca_coordinates_2d"] = torch.cat(first_two_chunks)
    return metrics, (contribution_sum / patch_count).cpu()


def aggregate_categories(
    samples: list[dict[str, Any]],
    evaluation_indices: list[int],
    metrics: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-image records and category means/stds for held-out samples."""
    sample_rows: list[dict[str, Any]] = []
    for position, sample_index in enumerate(evaluation_indices):
        sample = samples[sample_index]
        row: dict[str, Any] = {
            "sample_index": sample_index,
            "path": sample["path"],
            "label": int(sample["label"]),
            "class_name": sample["class_name"],
        }
        for key in METRIC_KEYS:
            row[key] = float(metrics[key][position])
        coordinates = metrics["pca_coordinates_2d"][position]
        row["pca_1"] = float(coordinates[0])
        row["pca_2"] = float(coordinates[1])
        sample_rows.append(row)

    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in sample_rows:
        by_label.setdefault(row["label"], []).append(row)

    category_rows: list[dict[str, Any]] = []
    for label, rows in sorted(by_label.items()):
        category: dict[str, Any] = {
            "label": label,
            "class_name": rows[0]["class_name"],
            "n": len(rows),
        }
        for key in METRIC_KEYS:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            category[f"mean_{key}"] = float(values.mean())
            category[f"std_{key}"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
        category_rows.append(category)
    return sample_rows, category_rows


def descending_ranks(rows: list[dict[str, Any]], metric: str) -> dict[int, int]:
    """Return one-based descending rank positions keyed by remapped class label."""
    ordered = sorted(rows, key=lambda row: row[metric], reverse=True)
    return {int(row["label"]): rank for rank, row in enumerate(ordered, start=1)}


def rank_comparison(category_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare MSE and precision-weighted category failure rankings."""
    mse_ranks = descending_ranks(category_rows, "mean_prediction_mse")
    zero_ranks = descending_ranks(category_rows, "mean_mahalanobis_zero")
    centered_ranks = descending_ranks(category_rows, "mean_mahalanobis_centered")
    labels = sorted(mse_ranks)

    for row in category_rows:
        label = int(row["label"])
        row["mse_rank"] = mse_ranks[label]
        row["mahalanobis_zero_rank"] = zero_ranks[label]
        row["mahalanobis_centered_rank"] = centered_ranks[label]
        row["zero_rank_shift"] = zero_ranks[label] - mse_ranks[label]
        row["absolute_zero_rank_shift"] = abs(row["zero_rank_shift"])

    mse = np.asarray([mse_ranks[label] for label in labels], dtype=np.float64)
    zero = np.asarray([zero_ranks[label] for label in labels], dtype=np.float64)
    centered = np.asarray([centered_ranks[label] for label in labels], dtype=np.float64)

    def spearman(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.corrcoef(left, right)[0, 1])

    def kendall(left: np.ndarray, right: np.ndarray) -> float:
        concordant = 0
        discordant = 0
        for first in range(len(left)):
            for second in range(first + 1, len(left)):
                product = (left[first] - left[second]) * (right[first] - right[second])
                concordant += int(product > 0)
                discordant += int(product < 0)
        pairs = concordant + discordant
        return float((concordant - discordant) / pairs) if pairs else 0.0

    top_k = min(int(CONFIG["TOP_K_CATEGORIES"]), len(labels))
    top_mse = {label for label, rank in mse_ranks.items() if rank <= top_k}
    top_zero = {label for label, rank in zero_ranks.items() if rank <= top_k}
    overlap = len(top_mse & top_zero)

    return {
        "category_count": len(labels),
        "spearman_mse_vs_mahalanobis_zero": spearman(mse, zero),
        "kendall_mse_vs_mahalanobis_zero": kendall(mse, zero),
        "spearman_mse_vs_mahalanobis_centered": spearman(mse, centered),
        "kendall_mse_vs_mahalanobis_centered": kendall(mse, centered),
        "top_k": top_k,
        "top_k_overlap_count": overlap,
        "top_k_overlap_fraction": float(overlap / top_k),
        "mean_absolute_zero_rank_shift": float(np.abs(zero - mse).mean()),
        "max_absolute_zero_rank_shift": int(np.abs(zero - mse).max()),
    }


def finish_plot(fig: Any, output_path: Path) -> None:
    """Save a figure and optionally display it in an interactive environment."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(CONFIG["PLOT_DPI"]), bbox_inches="tight")
    if CONFIG["SHOW_PLOTS"]:
        plt.show()
    plt.close(fig)


def save_plots(
    run_dir: Path,
    pca: dict[str, torch.Tensor | float | int],
    metrics: dict[str, torch.Tensor],
    category_rows: list[dict[str, Any]],
    mean_pc_contribution: torch.Tensor,
) -> None:
    """Save the Task 7 PCA, distance, and category-ranking diagnostics."""
    explained = pca["explained_ratio"]
    if not isinstance(explained, torch.Tensor):
        raise TypeError("PCA explained-variance state is missing")
    explained_np = explained.detach().cpu().numpy()
    component_numbers = np.arange(1, len(explained_np) + 1)

    fig, (axis_variance, axis_cumulative) = plt.subplots(1, 2, figsize=(13, 4.8))
    shown = min(50, len(explained_np))
    axis_variance.bar(component_numbers[:shown], explained_np[:shown])
    axis_variance.set(xlabel="Principal component", ylabel="Explained variance ratio")
    axis_variance.set_title(f"First {shown} error PCs")
    axis_cumulative.plot(component_numbers, np.cumsum(explained_np), linewidth=2)
    axis_cumulative.set(xlabel="Retained components", ylabel="Cumulative variance")
    axis_cumulative.set_ylim(0, 1.02)
    axis_cumulative.grid(alpha=0.25)
    axis_cumulative.set_title("Retained error variance")
    finish_plot(fig, run_dir / "pca_explained_variance.png")

    coordinates = metrics["pca_coordinates_2d"].numpy()
    mse = metrics["prediction_mse"].numpy()
    fig, axis = plt.subplots(figsize=(7.5, 6))
    scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=mse, s=18, alpha=0.7)
    axis.set(xlabel="Error PC1", ylabel="Error PC2", title="Held-out image errors")
    fig.colorbar(scatter, ax=axis, label="Prediction MSE")
    finish_plot(fig, run_dir / "error_pca_scatter.png")

    category_mse = np.asarray(
        [row["mean_prediction_mse"] for row in category_rows], dtype=np.float64
    )
    category_maha = np.asarray(
        [row["mean_mahalanobis_zero"] for row in category_rows], dtype=np.float64
    )
    shifts = np.asarray(
        [row["absolute_zero_rank_shift"] for row in category_rows], dtype=np.float64
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(category_mse, category_maha, c=shifts, cmap="magma", s=45)
    axis.set(
        xlabel="Category mean prediction MSE",
        ylabel="Category mean zero-referenced Mahalanobis",
        title="Metric sensitivity to embedding anisotropy",
    )
    fig.colorbar(scatter, ax=axis, label="Absolute rank shift")
    annotate = min(int(CONFIG["ANNOTATE_RANK_SHIFTS"]), len(category_rows))
    for row in sorted(
        category_rows, key=lambda item: item["absolute_zero_rank_shift"], reverse=True
    )[:annotate]:
        axis.annotate(
            row["class_name"],
            (row["mean_prediction_mse"], row["mean_mahalanobis_zero"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    finish_plot(fig, run_dir / "mse_vs_mahalanobis.png")

    fig, axis = plt.subplots(figsize=(7, 7))
    mse_rank = np.asarray([row["mse_rank"] for row in category_rows])
    maha_rank = np.asarray([row["mahalanobis_zero_rank"] for row in category_rows])
    axis.scatter(mse_rank, maha_rank, c=shifts, cmap="magma", s=45)
    limit = len(category_rows) + 1
    axis.plot([1, limit], [1, limit], linestyle="--", color="gray")
    axis.set(
        xlabel="MSE failure rank (1 = worst)",
        ylabel="Mahalanobis failure rank (1 = worst)",
        title="ImageNet category ranking changes",
        xlim=(limit, 0),
        ylim=(limit, 0),
    )
    finish_plot(fig, run_dir / "category_rank_comparison.png")

    contribution = mean_pc_contribution.numpy()
    shown = min(50, len(contribution))
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(np.arange(1, shown + 1), contribution[:shown])
    axis.set(
        xlabel="Principal component",
        ylabel="Mean precision-weighted squared error",
        title=f"Contribution of first {shown} PCs to Mahalanobis distance",
    )
    finish_plot(fig, run_dir / "precision_weighted_pc_contributions.png")


def main() -> None:
    """Run the full Task 7 analysis."""
    validate_config()
    seed_everything()
    run_dir = create_run_directory()

    samples = load_imagenet_subset(
        CONFIG["IMAGENET_ROOT"],
        num_samples=int(CONFIG["NUM_SAMPLES"]),
        num_classes=int(CONFIG["NUM_CLASSES"]),
        seed=int(CONFIG["SEED"]),
    )
    calibration_indices, evaluation_indices = stratified_split(samples)
    manifest = []
    calibration_set = set(calibration_indices)
    for index, sample in enumerate(samples):
        manifest.append(
            {
                **sample,
                "sample_index": index,
                "split": "calibration" if index in calibration_set else "evaluation",
            }
        )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    adapter = load_world_model()
    context_ids, target_ids = build_fixed_masks(adapter)
    print(
        f"Collecting errors for {len(samples)} images and {len(target_ids)} target patches.",
        flush=True,
    )
    errors = collect_prediction_errors(adapter, samples, context_ids, target_ids)

    pca = fit_error_pca(errors[calibration_indices])
    evaluation_metrics, mean_pc_contribution = score_evaluation_errors(
        errors[evaluation_indices], pca
    )
    sample_rows, category_rows = aggregate_categories(
        samples, evaluation_indices, evaluation_metrics
    )
    ranking = rank_comparison(category_rows)

    explained = pca["explained_ratio"]
    eigenvalues = pca["eigenvalues"]
    if not isinstance(explained, torch.Tensor) or not isinstance(eigenvalues, torch.Tensor):
        raise TypeError("PCA result tensors are missing")
    pca_summary = {
        "calibration_images": len(calibration_indices),
        "evaluation_images": len(evaluation_indices),
        "patch_observations": int(pca["observations"]),
        "embedding_dim": int(pca["embedding_dim"]),
        "components_used": int(pca["components_used"]),
        "ridge": float(pca["ridge"]),
        "retained_variance_fraction": float(explained.sum()),
        "explained_variance_ratio": explained.detach().cpu().tolist(),
        "eigenvalues": eigenvalues.detach().cpu().tolist(),
        "mean_precision_weighted_pc_contribution": mean_pc_contribution.tolist(),
    }

    (run_dir / "per_sample_metrics.json").write_text(
        json.dumps(sample_rows, indent=2), encoding="utf-8"
    )
    (run_dir / "category_metrics.json").write_text(
        json.dumps(category_rows, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "mean": pca["mean"].detach().cpu(),
            "components": pca["components"].detach().cpu(),
            "eigenvalues": eigenvalues.detach().cpu(),
            "precision_denominator": pca["precision_denominator"].detach().cpu(),
        },
        run_dir / "pca_state.pt",
    )
    save_plots(run_dir, pca, evaluation_metrics, category_rows, mean_pc_contribution)

    results = {
        "config": CONFIG,
        "model": {
            "name": CONFIG["MODEL_NAME"],
            "context_patches": context_ids,
            "target_patches": target_ids,
            "target_patch_count": len(target_ids),
            "embedding_dim": int(errors.shape[-1]),
        },
        "error_definition": "predictor_embedding - target_encoder_embedding",
        "pca": pca_summary,
        "ranking_sensitivity": ranking,
        "category_metrics": category_rows,
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Saved Task 7 run to {run_dir.resolve()}", flush=True)
    print(f"Ranking sensitivity: {ranking}", flush=True)


if __name__ == "__main__":
    main()
