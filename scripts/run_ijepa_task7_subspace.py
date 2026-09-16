#!/usr/bin/env python3
"""Task 7 subspace anatomy: raw-vs-whitened PCs and patch retrieval.

One calibration PCA is fitted at the largest requested dimension.  The run
stores minimal projected coordinates so every plot can be regenerated without
another I-JEPA pass, plus class-level predictor-squared-error and Mahalanobis
profiles and viewable patch montages for selected directions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import ijepa_task7_core as core
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

CONFIG: dict[str, Any] = core.base_config("outputs/ijepa_task7_subspace") | {
    "PCA_DIMS": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "MAX_PCA_COMPONENTS": 1024,
    "PCA_POWER_ITERATIONS": 4,
    "TOP_K_CATEGORIES": 10,
    "PC_PROFILE_TOP_K": 12,
    "HEATMAP_PCS": 30,
    "SPIDER_CLASSES": 4,
    # Retrieve 50 positive and 50 negative target patches for each selected PC.
    "PATCHES_PER_SIGN": 50,
    "HIGH_VARIANCE_PCS": 5,
    "LOW_VARIANCE_PCS": 5,
    "HIGH_MAHALANOBIS_PCS": 5,
    "PATCH_DISPLAY_SCALE": 6,
}


def validate_config() -> None:
    core.validate_base_config(CONFIG)
    dims = [int(value) for value in CONFIG["PCA_DIMS"]]
    if not dims or min(dims) < 2:
        raise ValueError("PCA_DIMS must contain dimensions greater than one")
    if int(CONFIG["MAX_PCA_COMPONENTS"]) < 2:
        raise ValueError("MAX_PCA_COMPONENTS must exceed one")
    for key in (
        "PC_PROFILE_TOP_K",
        "PATCHES_PER_SIGN",
        "HIGH_VARIANCE_PCS",
        "LOW_VARIANCE_PCS",
        "HIGH_MAHALANOBIS_PCS",
    ):
        if int(CONFIG[key]) <= 0:
            raise ValueError(f"{key} must be positive")


def category_means(labels: np.ndarray, values: np.ndarray) -> dict[int, float]:
    return {int(label): float(values[labels == label].mean()) for label in np.unique(labels)}


def precision_denominator(eigenvalues: np.ndarray) -> np.ndarray:
    return eigenvalues + float(CONFIG["RIDGE_FRACTION"]) * float(eigenvalues.mean())


def dimension_sweep(
    projected: dict[str, torch.Tensor], eigenvalues: np.ndarray, labels: np.ndarray
) -> list[dict[str, Any]]:
    zero_squared = projected["zero_coordinates"].square().numpy()
    predictor_mse = projected["predictor_mse"].numpy()
    total_squared = float(projected["total_squared_error"].sum())
    mse_ranks = core.descending_ranks(category_means(labels, predictor_mse))
    ordered_labels = sorted(mse_ranks)
    mse_rank_vector = np.asarray([mse_ranks[label] for label in ordered_labels], dtype=float)
    top_k = min(int(CONFIG["TOP_K_CATEGORIES"]), len(ordered_labels))
    top_mse = {label for label in ordered_labels if mse_ranks[label] <= top_k}

    rows: list[dict[str, Any]] = []
    for dim in sorted({int(value) for value in CONFIG["PCA_DIMS"] if value <= len(eigenvalues)}):
        contributions = zero_squared[:, :, :dim] / precision_denominator(eigenvalues[:dim])
        maha = np.sqrt(contributions.sum(axis=2).mean(axis=1))
        maha_ranks = core.descending_ranks(category_means(labels, maha))
        maha_rank_vector = np.asarray([maha_ranks[label] for label in ordered_labels], dtype=float)
        top_maha = {label for label in ordered_labels if maha_ranks[label] <= top_k}
        rows.append(
            {
                "pca_dimensions": dim,
                "predictor_squared_error_fraction_captured": float(
                    zero_squared[:, :, :dim].sum() / max(total_squared, 1e-12)
                ),
                "predictor_mse_mahalanobis_rank_spearman": float(
                    np.corrcoef(mse_rank_vector, maha_rank_vector)[0, 1]
                ),
                "top_k": top_k,
                "top_k_overlap_fraction": float(len(top_mse & top_maha) / top_k),
                "mean_absolute_rank_shift": float(
                    np.abs(mse_rank_vector - maha_rank_vector).mean()
                ),
            }
        )
    return rows


def pc_profiles(
    projected: dict[str, torch.Tensor],
    eigenvalues: np.ndarray,
    labels: np.ndarray,
    samples: list[dict[str, Any]],
    evaluation_indices: list[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    zero_squared = projected["zero_coordinates"].square().numpy()
    centered_squared = projected["centered_coordinates"].square().numpy()
    denominator = precision_denominator(eigenvalues)
    raw_per_image = zero_squared.mean(axis=1)
    zero_per_image = (zero_squared / denominator).mean(axis=1)
    centered_per_image = (centered_squared / denominator).mean(axis=1)
    unique_labels = sorted(int(label) for label in np.unique(labels))
    class_names = {
        int(samples[index]["label"]): samples[index]["class_name"] for index in evaluation_indices
    }
    matrices = {
        "predictor_squared_error_pc": np.stack(
            [raw_per_image[labels == label].mean(axis=0) for label in unique_labels]
        ),
        "mahalanobis_zero_pc": np.stack(
            [zero_per_image[labels == label].mean(axis=0) for label in unique_labels]
        ),
        "mahalanobis_centered_pc": np.stack(
            [centered_per_image[labels == label].mean(axis=0) for label in unique_labels]
        ),
    }
    top_k = min(int(CONFIG["PC_PROFILE_TOP_K"]), len(eigenvalues))
    records: list[dict[str, Any]] = []
    for row_index, label in enumerate(unique_labels):
        raw = matrices["predictor_squared_error_pc"][row_index]
        maha = matrices["mahalanobis_centered_pc"][row_index]
        raw_top = np.argsort(raw)[::-1][:top_k]
        maha_top = np.argsort(maha)[::-1][:top_k]
        intersection = sorted(set(raw_top) & set(maha_top))
        records.append(
            {
                "label": label,
                "class_name": class_names[label],
                "predictor_mse_top_pcs": [int(value) for value in raw_top],
                "mahalanobis_centered_top_pcs": [int(value) for value in maha_top],
                "common_top_pcs": [int(value) for value in intersection],
                "top_pc_overlap_count": len(intersection),
                "top_pc_overlap_fraction": float(len(intersection) / top_k),
                "profile_spearman": core.spearman(raw, maha),
            }
        )
    global_raw = raw_per_image.mean(axis=0)
    global_zero = zero_per_image.mean(axis=0)
    global_centered = centered_per_image.mean(axis=0)
    raw_top = np.argsort(global_raw)[::-1][:top_k]
    centered_top = np.argsort(global_centered)[::-1][:top_k]
    summary = {
        "reference_dimensions": len(eigenvalues),
        "top_k": top_k,
        "global": {
            "predictor_mse_top_pcs": [int(value) for value in raw_top],
            "mahalanobis_centered_top_pcs": [int(value) for value in centered_top],
            "common_top_pcs": [int(value) for value in sorted(set(raw_top) & set(centered_top))],
            "profile_spearman": core.spearman(global_raw, global_centered),
        },
        "per_class": records,
    }
    matrices.update(
        {
            "global_predictor_squared_error_pc": global_raw,
            "global_mahalanobis_zero_pc": global_zero,
            "global_mahalanobis_centered_pc": global_centered,
            "labels": np.asarray(unique_labels),
            "class_names": np.asarray([class_names[label] for label in unique_labels]),
        }
    )
    return summary, matrices


def save_projection_cache(
    run_dir: Path,
    projected: dict[str, torch.Tensor],
    pca: dict[str, Any],
    evaluation_indices: list[int],
    target_ids: list[int],
) -> None:
    """Save sufficient primitives to derive every per-PC metric later."""
    np.savez_compressed(
        run_dir / "evaluation_pc_coordinates.npz",
        evaluation_sample_indices=np.asarray(evaluation_indices, dtype=np.int32),
        target_patch_ids=np.asarray(target_ids, dtype=np.int16),
        zero_coordinates=projected["zero_coordinates"].numpy(),
        predictor_mse=projected["predictor_mse"].numpy(),
    )
    np.savez_compressed(
        run_dir / "pca_state.npz",
        mean=pca["mean"].detach().cpu().numpy(),
        mean_coordinates=(pca["mean"] @ pca["components"]).detach().cpu().numpy(),
        components=pca["components"].detach().cpu().numpy(),
        eigenvalues=pca["eigenvalues"].detach().cpu().numpy(),
        ridge=np.asarray([pca["ridge"]]),
    )


def save_profile_cache(run_dir: Path, matrices: dict[str, np.ndarray]) -> None:
    np.savez_compressed(run_dir / "class_pc_profiles.npz", **matrices)


def patch_crop(path: str, patch_id: int, grid: int, scale: int) -> Image.Image:
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    patch_size = 224 // grid
    row, column = divmod(patch_id, grid)
    crop = image.crop(
        (
            column * patch_size,
            row * patch_size,
            (column + 1) * patch_size,
            (row + 1) * patch_size,
        )
    )
    return crop.resize((patch_size * scale, patch_size * scale), Image.Resampling.NEAREST)


def save_patch_montage(path: Path, records: list[dict[str, Any]], grid: int) -> None:
    scale = int(CONFIG["PATCH_DISPLAY_SCALE"])
    tile = (224 // grid) * scale
    columns = 10
    rows = math.ceil(len(records) / columns)
    label_height = 14
    canvas = Image.new("RGB", (columns * tile, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        x = (index % columns) * tile
        y = (index // columns) * (tile + label_height)
        canvas.paste(patch_crop(record["path"], record["patch_id"], grid, scale), (x, y))
        draw.text((x + 1, y + tile), f"{record['class_name']}:{record['patch_id']}", fill="black")
    canvas.save(path)


def retrieve_pc_patches(
    run_dir: Path,
    projected: dict[str, torch.Tensor],
    eigenvalues: np.ndarray,
    samples: list[dict[str, Any]],
    evaluation_indices: list[int],
    target_ids: list[int],
    patch_grid_side: int,
    global_maha_zero: np.ndarray,
    global_maha_centered: np.ndarray,
) -> dict[str, Any]:
    coordinates = projected["zero_coordinates"].numpy()
    centered_coordinates = projected["centered_coordinates"].numpy()
    reference_dim = coordinates.shape[-1]
    high_count = min(int(CONFIG["HIGH_VARIANCE_PCS"]), reference_dim)
    low_count = min(int(CONFIG["LOW_VARIANCE_PCS"]), reference_dim)
    maha_count = min(int(CONFIG["HIGH_MAHALANOBIS_PCS"]), reference_dim)
    groups = {
        "highest_calibration_variance": list(range(high_count)),
        "lowest_retained_calibration_variance": list(
            range(reference_dim - low_count, reference_dim)
        ),
        "highest_evaluation_mahalanobis_zero_contribution": [
            int(value) for value in np.argsort(global_maha_zero)[::-1][:maha_count]
        ],
        "highest_evaluation_mahalanobis_centered_contribution": [
            int(value) for value in np.argsort(global_maha_centered)[::-1][:maha_count]
        ],
    }
    grid = int(patch_grid_side)
    per_sign = int(CONFIG["PATCHES_PER_SIGN"])
    output = run_dir / "pc_patch_montages"
    output.mkdir(parents=True, exist_ok=False)
    denominator = precision_denominator(eigenvalues)
    pc_groups: dict[int, list[str]] = {}
    for group_name, pcs in groups.items():
        for pc in pcs:
            pc_groups.setdefault(pc, []).append(group_name)
    artifacts: list[dict[str, Any]] = []
    for pc, selection_groups in sorted(pc_groups.items()):
        # PCA axes describe deviations from the calibration mean, so montage
        # extremes are selected by centred coordinates. Both zero- and
        # mean-referenced contributions remain in every patch record.
        flat = centered_coordinates[:, :, pc].reshape(-1)
        positive = np.argsort(flat)[::-1][:per_sign]
        negative = np.argsort(flat)[:per_sign]
        pc_record: dict[str, Any] = {
            "selection_groups": selection_groups,
            "pc": pc,
            "calibration_eigenvalue": float(eigenvalues[pc]),
            "precision_amplification": float(1.0 / denominator[pc]),
        }
        for sign, positions in (("positive", positive), ("negative", negative)):
            records: list[dict[str, Any]] = []
            for position in positions:
                image_position, target_position = divmod(int(position), len(target_ids))
                sample_index = evaluation_indices[image_position]
                sample = samples[sample_index]
                centered_coordinate = float(flat[position])
                zero_coordinate = float(coordinates[image_position, target_position, pc])
                records.append(
                    {
                        "sample_index": sample_index,
                        "path": sample["path"],
                        "label": int(sample["label"]),
                        "class_name": sample["class_name"],
                        "patch_id": int(target_ids[target_position]),
                        "patch_grid_row": int(target_ids[target_position] // grid),
                        "patch_grid_column": int(target_ids[target_position] % grid),
                        "zero_referenced_pc_coordinate": zero_coordinate,
                        "predictor_squared_error_pc_contribution": zero_coordinate**2,
                        "mahalanobis_zero_pc_contribution": zero_coordinate**2 / denominator[pc],
                        "centered_pc_coordinate": centered_coordinate,
                        "mahalanobis_centered_pc_contribution": (
                            centered_coordinate**2 / denominator[pc]
                        ),
                    }
                )
            filename = f"pc_{pc:04d}__{sign}.png"
            save_patch_montage(output / filename, records, grid)
            pc_record[f"{sign}_montage"] = str(Path("pc_patch_montages") / filename)
            pc_record[f"{sign}_patches"] = records
        artifacts.append(pc_record)
    result = {"patches_per_sign": per_sign, "groups": groups, "artifacts": artifacts}
    core.save_json(run_dir / "pc_patch_records.json", result)
    return result


def plot_pc_heatmaps(run_dir: Path, matrices: dict[str, np.ndarray]) -> None:
    raw = matrices["predictor_squared_error_pc"]
    maha = matrices["mahalanobis_centered_pc"]
    count = min(int(CONFIG["HEATMAP_PCS"]), raw.shape[1])
    selected = sorted(
        set(np.argsort(raw.mean(axis=0))[::-1][:count])
        | set(np.argsort(maha.mean(axis=0))[::-1][:count])
    )
    raw_share = raw[:, selected] / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
    maha_share = maha[:, selected] / np.maximum(maha.sum(axis=1, keepdims=True), 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=True)
    for axis, values, title in (
        (axes[0], raw_share, "Predictor squared-error share"),
        (axes[1], maha_share, "Centered Mahalanobis share"),
    ):
        image = axis.imshow(np.log10(values + 1e-12), aspect="auto", cmap="magma")
        axis.set_xticks(range(len(selected)), selected, rotation=90, fontsize=6)
        axis.set(xlabel="Actual PC index", title=title)
        fig.colorbar(image, ax=axis, label="log10 within-class share")
    axes[0].set_yticks(range(len(matrices["class_names"])), matrices["class_names"], fontsize=6)
    axes[0].set_ylabel("ImageNet class")
    fig.tight_layout()
    fig.savefig(run_dir / "per_class_pc_profile_heatmaps.png", dpi=int(CONFIG["PLOT_DPI"]))
    if CONFIG["SHOW_PLOTS"]:
        plt.show()
    plt.close(fig)


def plot_spider_profiles(run_dir: Path, matrices: dict[str, np.ndarray]) -> None:
    raw = matrices["predictor_squared_error_pc"]
    maha = matrices["mahalanobis_centered_pc"]
    divergence = np.abs(
        raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
        - maha / np.maximum(maha.sum(axis=1, keepdims=True), 1e-12)
    ).sum(axis=1)
    class_rows = np.argsort(divergence)[::-1][: int(CONFIG["SPIDER_CLASSES"])]
    fig, axes = plt.subplots(
        1,
        len(class_rows),
        figsize=(5 * len(class_rows), 5),
        subplot_kw={"projection": "polar"},
    )
    axes = np.atleast_1d(axes)
    top_k = min(int(CONFIG["PC_PROFILE_TOP_K"]), raw.shape[1])
    for axis, row in zip(axes, class_rows):
        pcs = sorted(
            set(np.argsort(raw[row])[::-1][:top_k]) | set(np.argsort(maha[row])[::-1][:top_k])
        )
        raw_values = raw[row, pcs] / max(float(raw[row].sum()), 1e-12)
        maha_values = maha[row, pcs] / max(float(maha[row].sum()), 1e-12)
        angles = np.linspace(0, 2 * np.pi, len(pcs), endpoint=False)
        closed_angles = np.r_[angles, angles[0]]
        axis.plot(closed_angles, np.r_[raw_values, raw_values[0]], label="Predictor MSE")
        axis.plot(closed_angles, np.r_[maha_values, maha_values[0]], label="Centered Maha")
        axis.set_xticks(angles, [str(pc) for pc in pcs], fontsize=6)
        axis.set_title(str(matrices["class_names"][row]))
        axis.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.15))
    fig.tight_layout()
    fig.savefig(run_dir / "per_class_pc_spider_profiles.png", dpi=int(CONFIG["PLOT_DPI"]))
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
    feasible = min(len(calibration_indices) * len(target_ids), errors.shape[-1])
    components = min(int(CONFIG["MAX_PCA_COMPONENTS"]), feasible)
    pca = core.fit_error_pca(
        errors[calibration_indices],
        components,
        CONFIG["DEVICE"],
        int(CONFIG["PCA_POWER_ITERATIONS"]),
        float(CONFIG["RIDGE_FRACTION"]),
    )
    projected = core.project_errors(
        errors[evaluation_indices], pca, int(CONFIG["BATCH_SIZE"]), CONFIG["DEVICE"]
    )
    eigenvalues = pca["eigenvalues"].detach().cpu().numpy()
    labels = np.asarray([int(samples[index]["label"]) for index in evaluation_indices])
    sweep = dimension_sweep(projected, eigenvalues, labels)
    profile_summary, matrices = pc_profiles(
        projected, eigenvalues, labels, samples, evaluation_indices
    )
    save_projection_cache(run_dir, projected, pca, evaluation_indices, target_ids)
    save_profile_cache(run_dir, matrices)
    patch_summary = retrieve_pc_patches(
        run_dir,
        projected,
        eigenvalues,
        samples,
        evaluation_indices,
        target_ids,
        int(round(math.sqrt(adapter.context_encoder.patch_embed.n_patches))),
        matrices["global_mahalanobis_zero_pc"],
        matrices["global_mahalanobis_centered_pc"],
    )
    plot_pc_heatmaps(run_dir, matrices)
    plot_spider_profiles(run_dir, matrices)
    result = {
        "config": CONFIG,
        "model": {
            "name": CONFIG["MODEL_NAME"],
            "embedding_dim": int(errors.shape[-1]),
            "target_patch_count": len(target_ids),
        },
        "split": {
            "calibration_images": len(calibration_indices),
            "evaluation_images": len(evaluation_indices),
        },
        "pca": {
            "reference_dimensions": components,
            "ridge": float(pca["ridge"]),
            "calibration_variance_explained": float(pca["explained_ratio"].sum()),
        },
        "dimension_sweep": sweep,
        "pc_profiles": profile_summary,
        "patch_retrieval": {
            "patches_per_sign": patch_summary["patches_per_sign"],
            "groups": patch_summary["groups"],
            "artifact_count": len(patch_summary["artifacts"]),
        },
    }
    core.save_json(run_dir / "results.json", result)
    print(f"Saved Task 7 subspace run to {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
