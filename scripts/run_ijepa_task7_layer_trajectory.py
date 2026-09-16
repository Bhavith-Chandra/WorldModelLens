#!/usr/bin/env python3
"""Task 7 component geometry at the final predictor/target MLP block.

Despite the historical filename, this entry point no longer claims to be a
predictor-layer trajectory.  It implements the deliberately narrow experiment
requested for V2: compare final predictor-output error with the error between
the last predictor block's projected MLP update and the last target-encoder
block's MLP update. Attention is intentionally excluded.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import ijepa_task7_core as core
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

CONFIG: dict[str, Any] = core.base_config("outputs/ijepa_task7_last_mlp") | {
    "PCA_COMPONENTS": 256,
    "PCA_POWER_ITERATIONS": 4,
}

SITE_FINAL = "final_predictor_output"
SITE_MLP = "last_block_mlp_output"


def validate_config() -> None:
    core.validate_base_config(CONFIG)
    if int(CONFIG["PCA_COMPONENTS"]) < 2:
        raise ValueError("PCA_COMPONENTS must exceed one")


@torch.no_grad()
def collect_final_and_mlp_errors(
    adapter: Any,
    samples: list[dict[str, Any]],
    context_ids: list[int],
    target_ids: list[int],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Collect aligned final-output and last-block MLP-update errors.

    Predictor MLP updates live in the predictor bottleneck space. Applying the
    trained ``predictor_project_back`` weight (without its output bias) places
    the update in the target encoder's 1280-D space before subtraction from the
    matching target MLP update.
    """
    captured: dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            captured[name] = output.detach()

        return hook

    predictor_layer = len(adapter.predictor.blocks) - 1
    target_layer = len(adapter.target_encoder.blocks) - 1
    handles = [
        adapter.predictor.blocks[predictor_layer].hook_mlp_out.register_forward_hook(
            save("predictor_mlp")
        ),
        adapter.target_encoder.blocks[target_layer].hook_mlp_out.register_forward_hook(
            save("target_mlp")
        ),
    ]
    chunks: dict[str, list[torch.Tensor]] = {SITE_FINAL: [], SITE_MLP: []}
    batch_size = int(CONFIG["BATCH_SIZE"])
    try:
        for batch_number, start in enumerate(range(0, len(samples), batch_size), start=1):
            observations = core.load_image_batch(samples[start : start + batch_size], adapter)
            context = adapter.context_encoder(observations, patch_ids=context_ids)
            prediction = adapter.predictor(context, context_ids, target_ids)
            target_all = adapter.target_encoder(observations)
            target = target_all[:, target_ids, :]
            predictor_mlp = captured["predictor_mlp"][:, len(context_ids) :, :]
            projection = adapter.predictor.predictor_project_back
            predictor_mlp_projected = functional.linear(predictor_mlp, projection.weight, bias=None)
            target_mlp = captured["target_mlp"][:, target_ids, :]
            if predictor_mlp_projected.shape != target_mlp.shape:
                raise RuntimeError(
                    "Projected predictor MLP and target MLP shapes differ: "
                    f"{tuple(predictor_mlp_projected.shape)} vs {tuple(target_mlp.shape)}"
                )
            chunks[SITE_FINAL].append((prediction - target).to(device="cpu", dtype=torch.float32))
            chunks[SITE_MLP].append(
                (predictor_mlp_projected - target_mlp).to(device="cpu", dtype=torch.float32)
            )
            total = math.ceil(len(samples) / batch_size)
            print(f"  final/MLP errors: batch {batch_number}/{total}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    metadata = {
        "predictor_last_block": predictor_layer,
        "target_encoder_last_block": target_layer,
        "predictor_mlp_dimension": int(captured["predictor_mlp"].shape[-1]),
        "target_mlp_dimension": int(captured["target_mlp"].shape[-1]),
        "comparison_dimension_after_projection": int(chunks[SITE_MLP][0].shape[-1]),
    }
    return {name: torch.cat(parts) for name, parts in chunks.items()}, metadata


def full_spectrum_statistics(errors: torch.Tensor) -> dict[str, float]:
    """Compute full covariance descriptors independently of PCA truncation."""
    device = torch.device(CONFIG["DEVICE"])
    vectors = errors.reshape(-1, errors.shape[-1]).to(device=device, dtype=torch.float32)
    centered = vectors - vectors.mean(dim=0)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum().clamp_min(1e-12)
    return {
        "top_eigenvalue_fraction": float(eigenvalues[-1] / total),
        "participation_ratio": float(total.square() / eigenvalues.square().sum().clamp_min(1e-12)),
    }


def fit_and_score_site(
    errors: torch.Tensor,
    calibration_indices: list[int],
    evaluation_indices: list[int],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
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
    summary = {
        **full_spectrum_statistics(errors[calibration_indices]),
        "pca_components": int(pca["components_used"]),
        "pca_calibration_variance_explained": float(pca["explained_ratio"].sum()),
        "ridge": float(pca["ridge"]),
        "metric_means": {key: float(value.mean()) for key, value in metrics.items()},
    }
    return summary, metrics


def build_rows(
    samples: list[dict[str, Any]],
    evaluation_indices: list[int],
    metrics: dict[str, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, sample_index in enumerate(evaluation_indices):
        sample = samples[sample_index]
        rows.append(
            {
                "sample_index": sample_index,
                "path": sample["path"],
                "label": int(sample["label"]),
                "class_name": sample["class_name"],
                "sites": {
                    site: {key: float(values[key][position]) for key in core.RAW_METRIC_KEYS}
                    for site, values in metrics.items()
                },
            }
        )
    return rows


def aggregate_categories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep component conclusions available per class, not only globally."""
    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(int(row["label"]), []).append(row)
    categories: list[dict[str, Any]] = []
    for label, members in sorted(by_label.items()):
        sites: dict[str, Any] = {}
        for site in (SITE_FINAL, SITE_MLP):
            sites[site] = {}
            for key in core.RAW_METRIC_KEYS:
                values = np.asarray([member["sites"][site][key] for member in members])
                sites[site][f"mean_{key}"] = float(values.mean())
                sites[site][f"std_{key}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        categories.append(
            {
                "label": label,
                "class_name": members[0]["class_name"],
                "n": len(members),
                "sites": sites,
            }
        )
    return categories


def site_association(metrics: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in core.RAW_METRIC_KEYS:
        mlp = metrics[SITE_MLP][key].numpy()
        final = metrics[SITE_FINAL][key].numpy()
        slope, intercept = np.polyfit(mlp, final, 1)
        prediction = slope * mlp + intercept
        r_squared = 1.0 - float(
            np.square(final - prediction).sum()
            / max(float(np.square(final - final.mean()).sum()), 1e-12)
        )
        result[key] = {
            "pearson": float(np.corrcoef(mlp, final)[0, 1]),
            "spearman": core.spearman(mlp, final),
            "linear_r_squared": r_squared,
        }
    return result


def plot_site_bars(run_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    metrics = list(core.RAW_METRIC_KEYS)
    x = np.arange(len(metrics))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5))
    for offset, site in ((-width / 2, SITE_MLP), (width / 2, SITE_FINAL)):
        values = np.asarray([summaries[site]["metric_means"][key] for key in metrics])
        values /= np.maximum(values.max(), 1e-12)
        axis.bar(x + offset, values, width, label=site)
    axis.set_xticks(x, ["Predictor MSE", "Maha zero", "Maha centered"])
    axis.set_ylabel("Within-site normalized mean")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "last_mlp_vs_final_metric_bars.png", dpi=int(CONFIG["PLOT_DPI"]))
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
    errors, component_metadata = collect_final_and_mlp_errors(
        adapter, samples, context_ids, target_ids
    )
    summaries: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, torch.Tensor]] = {}
    for site, site_errors in errors.items():
        summaries[site], metrics[site] = fit_and_score_site(
            site_errors, calibration_indices, evaluation_indices
        )
    rows = build_rows(samples, evaluation_indices, metrics)
    categories = aggregate_categories(rows)
    association = site_association(metrics)
    core.save_json(run_dir / "per_sample_component_metrics.json", rows)
    core.save_json(run_dir / "per_class_component_metrics.json", categories)
    plot_site_bars(run_dir, summaries)
    result = {
        "config": CONFIG,
        "model": {
            "name": CONFIG["MODEL_NAME"],
            "target_patch_count": len(target_ids),
            **component_metadata,
        },
        "split": {
            "calibration_images": len(calibration_indices),
            "evaluation_images": len(evaluation_indices),
        },
        "comparison_definition": {
            SITE_FINAL: "final predictor output - final target encoder output",
            SITE_MLP: (
                "weight-only project_back(last predictor block target-token MLP output) "
                "- last target-encoder block central-token MLP output"
            ),
        },
        "sites": summaries,
        "last_mlp_association_with_final_output": association,
        "caution": (
            "Association is descriptive, not an additive or causal decomposition; "
            "the transformer block is nonlinear and includes a residual path. The "
            "predictor output projection is a diagnostic lens and was not trained "
            "to align isolated MLP updates with target-encoder MLP updates."
        ),
    }
    core.save_json(run_dir / "results.json", result)
    print(f"Saved Task 7 last-MLP run to {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
