#!/usr/bin/env python3
"""Task 7 mainline: calibrated predictor-error geometry and DINO neighbours.

Only three raw scalar metrics are persisted per evaluation image:
``predictor_mse``, ``predictor_error_mahalanobis_zero``, and
``predictor_error_mahalanobis_centered``. Rankings and correlations are derived
summaries. Component-level storage lives in the subspace entry point.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import ijepa_task7_core as core
import matplotlib.pyplot as plt
import numpy as np
import torch

CONFIG: dict[str, Any] = core.base_config("outputs/ijepa_task7") | {
    "PCA_COMPONENTS": 256,
    "PCA_POWER_ITERATIONS": 4,
    "TOP_K_CATEGORIES": 10,
    "RUN_DINOV2": True,
    "DINOV2_MODEL": "facebook/dinov2-base",
    "DINO_NEIGHBORS": 3,
}


def validate_config() -> None:
    core.validate_base_config(CONFIG)
    if int(CONFIG["PCA_COMPONENTS"]) < 2:
        raise ValueError("PCA_COMPONENTS must exceed one")
    if int(CONFIG["PCA_POWER_ITERATIONS"]) < 0:
        raise ValueError("PCA_POWER_ITERATIONS cannot be negative")
    if int(CONFIG["DINO_NEIGHBORS"]) <= 0:
        raise ValueError("DINO_NEIGHBORS must be positive")
    if CONFIG["RUN_DINOV2"] and importlib.util.find_spec("transformers") is None:
        raise ValueError("RUN_DINOV2 needs transformers; install it or set RUN_DINOV2=False")


def build_sample_rows(
    samples: list[dict[str, Any]],
    evaluation_indices: list[int],
    metrics: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, sample_index in enumerate(evaluation_indices):
        sample = samples[sample_index]
        row: dict[str, Any] = {
            "sample_index": sample_index,
            "path": sample["path"],
            "label": int(sample["label"]),
            "class_name": sample["class_name"],
        }
        for key in core.RAW_METRIC_KEYS:
            row[key] = float(metrics[key][position])
        rows.append(row)
    return rows


def aggregate_categories(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in sample_rows:
        by_label.setdefault(int(row["label"]), []).append(row)
    categories: list[dict[str, Any]] = []
    for label, rows in sorted(by_label.items()):
        category: dict[str, Any] = {
            "label": label,
            "class_name": rows[0]["class_name"],
            "n": len(rows),
        }
        for key in core.RAW_METRIC_KEYS:
            values = np.asarray([float(row[key]) for row in rows])
            category[f"mean_{key}"] = float(values.mean())
            category[f"std_{key}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        categories.append(category)
    return categories


def ranking_analysis(categories: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in categories]
    fields = {
        "predictor_mse": "mean_predictor_mse",
        "mahalanobis_zero": "mean_predictor_error_mahalanobis_zero",
        "mahalanobis_centered": "mean_predictor_error_mahalanobis_centered",
    }
    ranks = {
        name: core.descending_ranks({int(row["label"]): float(row[field]) for row in categories})
        for name, field in fields.items()
    }
    for row in categories:
        label = int(row["label"])
        row["predictor_mse_difficulty_rank"] = ranks["predictor_mse"][label]
        row["mahalanobis_zero_difficulty_rank"] = ranks["mahalanobis_zero"][label]
        row["mahalanobis_centered_difficulty_rank"] = ranks["mahalanobis_centered"][label]
    mse = np.asarray([ranks["predictor_mse"][label] for label in labels], dtype=float)
    top_k = min(int(CONFIG["TOP_K_CATEGORIES"]), len(labels))
    top_mse = {label for label in labels if ranks["predictor_mse"][label] <= top_k}

    def comparison(name: str) -> dict[str, Any]:
        other = np.asarray([ranks[name][label] for label in labels], dtype=float)
        top_other = {label for label in labels if ranks[name][label] <= top_k}
        shifts = np.abs(other - mse)
        return {
            "spearman_vs_predictor_mse": float(np.corrcoef(mse, other)[0, 1]),
            "top_k": top_k,
            "top_k_overlap_count": len(top_mse & top_other),
            "top_k_overlap_fraction": float(len(top_mse & top_other) / top_k),
            "mean_absolute_rank_shift": float(shifts.mean()),
            "max_absolute_rank_shift": int(shifts.max()),
        }

    return {
        "category_count": len(labels),
        "mahalanobis_zero": comparison("mahalanobis_zero"),
        "mahalanobis_centered": comparison("mahalanobis_centered"),
    }


def raw_distance_correlations(
    sample_rows: list[dict[str, Any]], categories: list[dict[str, Any]]
) -> dict[str, Any]:
    def pair(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, float]:
        x = np.asarray([float(row[left]) for row in rows])
        y = np.asarray([float(row[right]) for row in rows])
        return {
            "pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": core.spearman(x, y),
        }

    return {
        "per_image": {
            "predictor_mse_vs_mahalanobis_zero": pair(
                sample_rows, "predictor_mse", "predictor_error_mahalanobis_zero"
            ),
            "predictor_mse_vs_mahalanobis_centered": pair(
                sample_rows, "predictor_mse", "predictor_error_mahalanobis_centered"
            ),
            "mahalanobis_zero_vs_centered": pair(
                sample_rows,
                "predictor_error_mahalanobis_zero",
                "predictor_error_mahalanobis_centered",
            ),
        },
        "per_category": {
            "predictor_mse_vs_mahalanobis_zero": pair(
                categories,
                "mean_predictor_mse",
                "mean_predictor_error_mahalanobis_zero",
            ),
            "predictor_mse_vs_mahalanobis_centered": pair(
                categories,
                "mean_predictor_mse",
                "mean_predictor_error_mahalanobis_centered",
            ),
        },
    }


@torch.no_grad()
def compute_dinov2_embeddings(paths: list[str]) -> np.ndarray:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device(CONFIG["DEVICE"])
    processor = AutoImageProcessor.from_pretrained(CONFIG["DINOV2_MODEL"])
    model = AutoModel.from_pretrained(CONFIG["DINOV2_MODEL"]).to(device).eval()
    features: list[torch.Tensor] = []
    batch_size = int(CONFIG["BATCH_SIZE"])
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        output = model(**inputs)
        pooled = getattr(output, "pooler_output", None)
        features.append((pooled if pooled is not None else output.last_hidden_state[:, 0]).cpu())
    embeddings = torch.cat(features).float()
    embeddings /= embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return embeddings.numpy()


def dino_neighbor_analysis(
    sample_rows: list[dict[str, Any]], embeddings: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Record top-three nearest/farthest DINO samples and their raw metrics."""
    similarity = embeddings @ embeddings.T
    count = len(sample_rows)
    k = min(int(CONFIG["DINO_NEIGHBORS"]), count - 1)
    nearest_similarity = similarity.copy()
    np.fill_diagonal(nearest_similarity, -np.inf)
    nearest = np.argsort(-nearest_similarity, axis=1)[:, :k]
    farthest_similarity = similarity.copy()
    np.fill_diagonal(farthest_similarity, np.inf)
    farthest = np.argsort(farthest_similarity, axis=1)[:, :k]

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_index": row["sample_index"],
            "path": row["path"],
            "label": row["label"],
            "class_name": row["class_name"],
            **{key: row[key] for key in core.RAW_METRIC_KEYS},
        }

    records: list[dict[str, Any]] = []
    for query_index, query in enumerate(sample_rows):
        item: dict[str, Any] = {"query": compact(query)}
        for name, indices in (
            ("nearest", nearest[query_index]),
            ("farthest", farthest[query_index]),
        ):
            group: list[dict[str, Any]] = []
            for neighbor_index in indices:
                neighbor = sample_rows[int(neighbor_index)]
                record = compact(neighbor)
                record["dino_cosine_similarity"] = float(similarity[query_index, neighbor_index])
                record["same_class"] = bool(query["label"] == neighbor["label"])
                record["absolute_metric_differences"] = {
                    key: abs(float(query[key]) - float(neighbor[key]))
                    for key in core.RAW_METRIC_KEYS
                }
                group.append(record)
            item[name] = group
        records.append(item)

    off_diagonal = ~np.eye(count, dtype=bool)
    summary: dict[str, Any] = {
        "model": CONFIG["DINOV2_MODEL"],
        "embedding_dim": int(embeddings.shape[1]),
        "neighbors_per_side": k,
        "same_class_fraction_nearest": float(
            np.mean(
                [
                    sample_rows[i]["label"] == sample_rows[int(j)]["label"]
                    for i in range(count)
                    for j in nearest[i]
                ]
            )
        ),
        "metrics": {},
    }
    for key in core.RAW_METRIC_KEYS:
        values = np.asarray([float(row[key]) for row in sample_rows])
        all_diff = np.abs(values[:, None] - values[None, :])[off_diagonal].mean()
        near_diff = np.mean(
            [abs(values[i] - values[int(j)]) for i in range(count) for j in nearest[i]]
        )
        far_diff = np.mean(
            [abs(values[i] - values[int(j)]) for i in range(count) for j in farthest[i]]
        )
        summary["metrics"][key] = {
            "mean_absolute_difference_nearest": float(near_diff),
            "mean_absolute_difference_farthest": float(far_diff),
            "mean_absolute_difference_all_pairs": float(all_diff),
            "nearest_to_all_ratio": float(near_diff / max(all_diff, 1e-12)),
            "farthest_to_all_ratio": float(far_diff / max(all_diff, 1e-12)),
        }
    return summary, records


def plot_category_bars(run_dir: Path, categories: list[dict[str, Any]]) -> None:
    ordered = sorted(categories, key=lambda row: row["mean_predictor_mse"], reverse=True)
    mse = np.asarray([row["mean_predictor_mse"] for row in ordered])
    maha = np.asarray([row["mean_predictor_error_mahalanobis_centered"] for row in ordered])
    mse /= max(float(mse.max()), 1e-12)
    maha /= max(float(maha.max()), 1e-12)
    x = np.arange(len(ordered))
    fig, axis = plt.subplots(figsize=(15, 5.5))
    axis.bar(x - 0.2, mse, 0.4, label="Predictor MSE / maximum")
    axis.bar(x + 0.2, maha, 0.4, label="Centered predictor-error Mahalanobis / maximum")
    axis.set_xticks(x, [row["class_name"] for row in ordered], rotation=90, fontsize=7)
    axis.set(xlabel="ImageNet class", ylabel="Within-metric normalized value")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "per_class_raw_metric_bars.png", dpi=int(CONFIG["PLOT_DPI"]))
    if CONFIG["SHOW_PLOTS"]:
        plt.show()
    plt.close(fig)


def main() -> None:
    validate_config()
    core.seed_everything(CONFIG)
    run_dir = core.create_run_directory(CONFIG)
    samples = core.load_samples(CONFIG)
    calibration_indices, evaluation_indices = core.split_from_config(samples, CONFIG)
    core.save_dataset_manifest(run_dir, samples, calibration_indices)

    adapter = core.load_world_model(CONFIG)
    context_ids, target_ids = core.build_fixed_masks(adapter, int(CONFIG["TARGET_PATCH_SIDE"]))
    errors = core.collect_prediction_errors(
        adapter, samples, context_ids, target_ids, int(CONFIG["BATCH_SIZE"])
    )
    pca = core.fit_error_pca(
        errors[calibration_indices],
        int(CONFIG["PCA_COMPONENTS"]),
        CONFIG["DEVICE"],
        int(CONFIG["PCA_POWER_ITERATIONS"]),
        float(CONFIG["RIDGE_FRACTION"]),
    )
    projected = core.project_errors(
        errors[evaluation_indices], pca, int(CONFIG["BATCH_SIZE"]), CONFIG["DEVICE"]
    )
    metrics = core.raw_metrics_from_projection(projected, pca)
    sample_rows = build_sample_rows(samples, evaluation_indices, metrics)
    categories = aggregate_categories(sample_rows)
    ranking = ranking_analysis(categories)
    correlations = raw_distance_correlations(sample_rows, categories)

    core.save_json(run_dir / "per_sample_metrics.json", sample_rows)
    core.save_json(run_dir / "category_metrics.json", categories)
    plot_category_bars(run_dir, categories)

    visual_similarity: dict[str, Any] = {}
    if CONFIG["RUN_DINOV2"]:
        embeddings = compute_dinov2_embeddings([row["path"] for row in sample_rows])
        visual_similarity, neighbor_rows = dino_neighbor_analysis(sample_rows, embeddings)
        core.save_json(run_dir / "dinov2_sample_neighbors.json", neighbor_rows)

    result = {
        "config": CONFIG,
        "model": {
            "name": CONFIG["MODEL_NAME"],
            "embedding_dim": int(errors.shape[-1]),
            "context_patch_count": len(context_ids),
            "target_patch_count": len(target_ids),
        },
        "split": {
            "calibration_images": len(calibration_indices),
            "evaluation_images": len(evaluation_indices),
            "calibration_patch_errors": len(calibration_indices) * len(target_ids),
        },
        "pca": {
            "components": int(pca["components_used"]),
            "ridge": float(pca["ridge"]),
            "calibration_variance_explained": float(pca["explained_ratio"].sum()),
            "mean_centered_predictor_squared_error_fraction_captured": float(
                projected["centered_coordinates"].square().sum()
                / projected["total_centered_squared_error"].sum().clamp_min(1e-12)
            ),
        },
        "raw_distance_correlations": correlations,
        "ranking_sensitivity": ranking,
        "dinov2_similarity": visual_similarity,
    }
    core.save_json(run_dir / "results.json", result)
    print(f"Saved Task 7 main run to {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
