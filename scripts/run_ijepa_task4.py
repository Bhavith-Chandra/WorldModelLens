#!/usr/bin/env python3
"""Task 4: zero, dataset-mean, and cross-sample I-JEPA predictor ablations.

For every predictor layer we replace that layer's residual-stream activation
with one of three substitutes and measure the downstream effect on the
predicted target representation:

* ``zero``     -- replace with zeros (off-distribution; the reviewer's concern)
* ``mean``     -- replace with the dataset-average activation at that layer
                  (on-distribution in first moment: the cloud's centroid)
* ``resample`` -- replace with a different sample's activation at that layer
                  (a real point in the cloud, but semantically wrong image)

The reference "cloud" is simply the empirical set of real activations the model
produces at a layer over the sampled images. The library's full-covariance
Mahalanobis detector is fitted to token-pooled activations and predictions:

* ``substitution_maha``  -- how many std-devs the *substituted* activation sits
                            from that cloud. Answers reviewer Q2 directly:
                            zero >> 1, mean ~= 0, resample ~= 1.
* ``prediction_maha``    -- how far the *final* prediction lies from the clean
                            prediction cloud, i.e. whether the intervention
                            pushes the output off-distribution.

Forward passes are batched by calling the adapter's encoders / predictor
directly (mirroring ``IJEPAAdapter.compute_loss``) and interventions use native
``register_forward_hook`` on the model's exposed ``hook_resid_post`` points,
because ``HookedWorldModel.run_with_cache`` is sequential (its leading axis is
time, not batch) and cannot batch these passes.

Edit CONFIG below, then run this file.
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


# ---------------------------------------------------------------------------
# Edit experiment arguments here. Every value is copied to config.yaml.
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    # Use CHECKPOINT_PATH when the Meta checkpoint already exists locally.
    # Otherwise ModelHub downloads/loads MODEL_NAME into CACHE_DIR.
    "MODEL_NAME": "ijepa-vit-h-in1k",
    "CHECKPOINT_PATH": None,
    "CACHE_DIR": None,
    "FORCE_DOWNLOAD": False,
    "IMAGENET_ROOT": "/content/imagenet/val",
    "OUTPUT_ROOT": "outputs/ijepa_task4",
    "NUM_SAMPLES": 1000,
    # ImageNet has 1,000 classes. This pilot deliberately uses 50 classes so
    # 1,000 images give 20 examples per class, which is enough for stratified
    # train/test splits and cross-validation. To use every class, increase
    # NUM_SAMPLES as well (for example, 20,000 samples and 1,000 classes).
    "NUM_CLASSES": 50,
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PRECISION": "fp16" if torch.cuda.is_available() else "fp32",
    # Number of images per batched forward pass. Raise until GPU memory is the
    # limit; this is the main runtime lever.
    "BATCH_SIZE": 32,
    # None (or "all") sweeps every predictor layer. The official ViT-H
    # predictor has 12 layers; the ijepa_mini fallback has 4.
    "TARGET_LAYERS": None,
    "ABLATION_MODES": ["zero", "mean", "resample"],
    # A fixed central square is predicted for every image. Keeping masks fixed
    # makes dataset means and cross-sample activations token-aligned.
    "TARGET_PATCH_SIDE": 4,
    "PROBE_TEST_SPLIT": 0.2,
    "PROBE_USE_CV": True,
    "PLOT_DPI": 180,
    "SHOW_PLOTS": True,
}


REPO_ROOT = Path(__file__).resolve().parents[1]
from world_model_lens import LatentProber  # noqa: E402
from world_model_lens.analysis.ood_detection import MahalanobisOODDetector  # noqa: E402
from world_model_lens.data import load_imagenet_image, load_imagenet_subset  # noqa: E402
from world_model_lens.hub.model_hub import ModelHub  # noqa: E402


# Metric keys aggregated into per-(layer, mode) summaries and plotted.
METRIC_KEYS = (
    "prediction_mse",
    "prediction_mse_delta",
    "prediction_mse_ratio",
    "target_cosine",
    "prediction_shift_l2",
    "clean_prediction_cosine",
    "substitution_maha",
    "prediction_maha",
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def validate_config() -> None:
    """Fail fast on obviously inconsistent CONFIG values before any heavy work."""
    required = ("IMAGENET_ROOT", "OUTPUT_ROOT", "ABLATION_MODES")
    missing = [name for name in required if not CONFIG.get(name)]
    if missing:
        raise ValueError(f"Missing CONFIG values: {missing}")
    if CONFIG["NUM_SAMPLES"] % CONFIG["NUM_CLASSES"] != 0:
        raise ValueError("NUM_SAMPLES must be divisible by NUM_CLASSES")
    if CONFIG["NUM_SAMPLES"] // CONFIG["NUM_CLASSES"] < 2:
        raise ValueError(
            "The classification probe needs repeated examples per class; "
            "increase NUM_SAMPLES or reduce NUM_CLASSES"
        )
    unknown_modes = set(CONFIG["ABLATION_MODES"]) - {"zero", "mean", "resample"}
    if unknown_modes:
        raise ValueError(f"Unknown ABLATION_MODES: {sorted(unknown_modes)}")
    if int(CONFIG["BATCH_SIZE"]) <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    if CONFIG["PRECISION"] == "fp16" and not str(CONFIG["DEVICE"]).startswith("cuda"):
        raise ValueError("fp16 requires a CUDA device; use fp32 on CPU")


def seed_everything() -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible subsets and probes."""
    seed = int(CONFIG["SEED"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_directory() -> Path:
    """Create a timestamped run directory and persist the resolved CONFIG."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(CONFIG["OUTPUT_ROOT"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logged_config = dict(CONFIG)
    logged_config["RUN_ID"] = run_id
    logged_config["CREATED_AT_UTC"] = datetime.now(timezone.utc).isoformat()
    logged_config["REPO_ROOT"] = str(REPO_ROOT)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(logged_config, handle, sort_keys=False)
    return run_dir


def load_world_model() -> Any:
    """Load the I-JEPA adapter (encoders + pretrained predictor) via ModelHub.

    Returns the adapter in eval mode, cast to the configured precision, with the
    fused attention kernel enabled when requested.
    """
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


def resolve_layers(adapter: Any) -> list[int]:
    """Resolve CONFIG['TARGET_LAYERS'] into a validated list of layer indices."""
    depth = len(adapter.predictor.blocks)
    layers = CONFIG.get("TARGET_LAYERS")
    if layers in (None, "all"):
        return list(range(depth))
    resolved = [int(layer) for layer in layers]
    invalid = [layer for layer in resolved if layer < 0 or layer >= depth]
    if invalid:
        raise ValueError(
            f"TARGET_LAYERS contains invalid indices {invalid}; predictor depth is {depth}"
        )
    return resolved


def build_fixed_masks(adapter: Any) -> tuple[list[int], list[int]]:
    """Return (context_ids, target_ids) for a fixed central square target.

    A single fixed mask shared across all images is what makes dataset-mean and
    cross-sample (resample) activations token-aligned and comparable.
    """
    num_patches = int(adapter.context_encoder.patch_embed.n_patches)
    grid = int(math.sqrt(num_patches))
    side = int(CONFIG["TARGET_PATCH_SIDE"])
    if grid * grid != num_patches or side <= 0 or side >= grid:
        raise ValueError("TARGET_PATCH_SIDE is invalid for the checkpoint patch grid")
    start = (grid - side) // 2
    target = {
        row * grid + col
        for row in range(start, start + side)
        for col in range(start, start + side)
    }
    context = [patch for patch in range(num_patches) if patch not in target]
    return context, sorted(target)


# ---------------------------------------------------------------------------
# Batched forward passes + interventions
# ---------------------------------------------------------------------------


def model_device_dtype(adapter: Any) -> tuple[torch.device, torch.dtype]:
    """Return the device and dtype of the adapter's parameters."""
    parameter = next(adapter.context_encoder.parameters())
    return parameter.device, parameter.dtype


def load_image_batch(samples: list[dict[str, Any]], adapter: Any) -> torch.Tensor:
    """Load and preprocess a list of sample dicts into one [B, 3, H, W] batch."""
    device, dtype = model_device_dtype(adapter)
    tensors = [load_imagenet_image(sample["path"], image_size=224) for sample in samples]
    return torch.cat(tensors, dim=0).to(device=device, dtype=dtype)


@torch.no_grad()
def forward_capture(
    adapter: Any,
    obs: torch.Tensor,
    context_ids: list[int],
    target_ids: list[int],
    layers: list[int],
) -> dict[str, Any]:
    """Run a clean batched forward pass, capturing per-layer predictor activations.

    Returns a dict with the context latents, target-block prediction, target
    ground truth, and each requested layer's residual-stream activation
    (``hook_resid_post`` output, shape [B, seq, predictor_dim]).
    """
    store: dict[int, torch.Tensor] = {}

    def capture(layer: int):
        def hook(_module, _inputs, output):
            store[layer] = output.detach()

        return hook

    handles = [
        adapter.predictor.blocks[layer].hook_resid_post.register_forward_hook(capture(layer))
        for layer in layers
    ]
    try:
        context_latents = adapter.context_encoder(obs, patch_ids=context_ids)
        prediction = adapter.predictor(context_latents, context_ids, target_ids)
        target_full = adapter.target_encoder(obs)
    finally:
        for handle in handles:
            handle.remove()

    return {
        "context_latents": context_latents.detach(),
        "prediction": prediction.detach(),
        "target": target_full[:, target_ids, :].detach(),
        "activations": {layer: store[layer] for layer in layers},
    }


@torch.no_grad()
def forward_intervene(
    adapter: Any,
    context_latents: torch.Tensor,
    context_ids: list[int],
    target_ids: list[int],
    layer: int,
    mode: str,
    replacement: torch.Tensor | None,
) -> torch.Tensor:
    """Re-run only the predictor while replacing layer ``layer``'s activation.

    ``mode == 'zero'`` substitutes zeros; otherwise ``replacement`` (already
    shaped [B, seq, predictor_dim]) is inserted. The context encoder is not
    re-run because the intervention lives entirely inside the predictor.
    """

    def hook(_module, _inputs, output):
        if mode == "zero":
            return torch.zeros_like(output)
        if replacement is None:
            raise ValueError(f"mode '{mode}' requires a replacement activation")
        value = replacement.to(device=output.device, dtype=output.dtype)
        if value.shape != output.shape:
            raise ValueError(f"Replacement {tuple(value.shape)} != activation {tuple(output.shape)}")
        return value

    handle = adapter.predictor.blocks[layer].hook_resid_post.register_forward_hook(hook)
    try:
        prediction = adapter.predictor(context_latents, context_ids, target_ids)
    finally:
        handle.remove()
    return prediction.detach()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    clean_prediction: torch.Tensor,
    substitution: torch.Tensor,
    layer_detector: MahalanobisOODDetector,
    prediction_detector: MahalanobisOODDetector,
) -> dict[str, float]:
    """Compute one sample's effect metrics for an intervention.

    The library Mahalanobis detectors are fitted on token-pooled clean layer
    activations and token-pooled clean predictions, respectively.
    """
    pred = prediction.float().flatten()
    tgt = target.float().flatten()
    clean = clean_prediction.float().flatten()

    mse = torch.mean((pred - tgt) ** 2)
    clean_mse = torch.mean((clean - tgt) ** 2)

    substitution_feature = substitution.float().mean(dim=0, keepdim=True)
    prediction_feature = prediction.float().mean(dim=0, keepdim=True)

    return {
        "prediction_mse": float(mse),
        "prediction_mse_delta": float(mse - clean_mse),
        "prediction_mse_ratio": float(mse / clean_mse.clamp_min(1e-12)),
        "target_cosine": float(
            torch.nn.functional.cosine_similarity(pred, tgt, dim=0)
        ),
        "prediction_shift_l2": float((pred - clean).norm()),
        "clean_prediction_cosine": float(
            torch.nn.functional.cosine_similarity(pred, clean, dim=0)
        ),
        "substitution_maha": float(layer_detector.score(substitution_feature).item()),
        "prediction_maha": float(prediction_detector.score(prediction_feature).item()),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-sample rows into mean/std summaries per (layer, mode)."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["layer"], row["mode"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (layer, mode), group in sorted(grouped.items()):
        summary: dict[str, Any] = {"layer": layer, "mode": mode, "n": len(group)}
        for metric in METRIC_KEYS:
            values = np.asarray(
                [row[metric] for row in group if row.get(metric) is not None],
                dtype=np.float64,
            )
            if values.size == 0:
                summary[f"mean_{metric}"] = None
                summary[f"std_{metric}"] = None
                continue
            summary[f"mean_{metric}"] = float(values.mean())
            summary[f"std_{metric}"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summaries.append(summary)
    return summaries


# ---------------------------------------------------------------------------
# Linear probe (I-JEPA has no classification head; a probe reads class info)
# ---------------------------------------------------------------------------


def train_library_probe(
    features: list[list[float]], labels: list[int], name: str
) -> dict[str, Any]:
    """Fit a logistic probe on pooled prediction features to read class info.

    This does not train the backbone; it is a cheap linear read-out on frozen
    features that reports how much class information survives an intervention.
    """
    result = LatentProber(seed=int(CONFIG["SEED"])).train_probe(
        activations=torch.tensor(features, dtype=torch.float32),
        labels=np.asarray(labels, dtype=np.int64),
        concept_name="imagenet_class",
        activation_name=name,
        probe_type="logistic",
        test_split=float(CONFIG["PROBE_TEST_SPLIT"]),
        use_cv=bool(CONFIG["PROBE_USE_CV"]),
    )
    return {
        "accuracy": float(result.accuracy),
        "cv_mean": float(result.cv_mean),
        "cv_std": float(result.cv_std),
        "regularization_alpha": float(result.regularization_alpha),
        "training_samples": int(result.training_samples),
        "test_samples": int(result.test_samples),
    }


def add_probe_results(
    summaries: list[dict[str, Any]],
    feature_sets: dict[tuple[int, str], list[list[float]]],
    labels: list[int],
) -> dict[str, dict[str, Any]]:
    """Train one probe per (layer, mode) feature set and attach accuracy to summaries.

    The clean feature set is identical for every layer, so its probe is trained
    once and reused.
    """
    probe_results: dict[str, dict[str, Any]] = {}
    clean_result: dict[str, Any] | None = None
    for (layer, mode), features in feature_sets.items():
        name = f"predictor.layer_{layer}.{mode}"
        if mode == "clean" and clean_result is not None:
            probe_results[name] = dict(clean_result)
        else:
            probe_results[name] = train_library_probe(features, labels, name)
            if mode == "clean":
                clean_result = dict(probe_results[name])

    for summary in summaries:
        key = f"predictor.layer_{summary['layer']}.{summary['mode']}"
        summary["classification_accuracy"] = probe_results[key]["accuracy"]
        summary["classification_cv_mean"] = probe_results[key]["cv_mean"]
        summary["classification_cv_std"] = probe_results[key]["cv_std"]
    return probe_results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def plot_metric(
    summaries: list[dict[str, Any]],
    layers: list[int],
    modes: list[str],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Draw a grouped bar chart of ``metric`` across layers and ablation modes."""
    lookup = {(row["layer"], row["mode"]): row for row in summaries}
    width = 0.8 / max(1, len(modes))
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for mode_index, mode in enumerate(modes):
        values = [lookup.get((layer, mode), {}).get(metric) for layer in layers]
        values = [np.nan if value is None else value for value in values]
        offsets = x - 0.4 + width / 2 + mode_index * width
        ax.bar(offsets, values, width=width, label=mode)
    ax.set_xticks(x, [str(layer) for layer in layers])
    ax.set_xlabel("Predictor layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"I-JEPA Task 4: {ylabel}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(CONFIG["PLOT_DPI"]), bbox_inches="tight")
    if CONFIG["SHOW_PLOTS"]:
        plt.show()
    plt.close(fig)


def save_plots(run_dir: Path, summaries: list[dict[str, Any]], layers: list[int]) -> None:
    """Write the standard Task 4 figures for this run."""
    all_modes = ["clean", *CONFIG["ABLATION_MODES"]]
    ablation_modes = list(CONFIG["ABLATION_MODES"])
    plot_metric(
        summaries, layers, all_modes, "mean_prediction_mse",
        "Prediction MSE", run_dir / "prediction_mse.png",
    )
    plot_metric(
        summaries, layers, all_modes, "mean_target_cosine",
        "Target cosine", run_dir / "target_cosine.png",
    )
    plot_metric(
        summaries, layers, all_modes, "classification_accuracy",
        "Linear-probe accuracy", run_dir / "classification_accuracy.png",
    )
    # Mahalanobis metrics are undefined for the clean pass; plot ablations only.
    plot_metric(
        summaries, layers, ablation_modes, "mean_substitution_maha",
        "Substitution Mahalanobis (off-distribution)", run_dir / "substitution_maha.png",
    )
    plot_metric(
        summaries, layers, ablation_modes, "mean_prediction_maha",
        "Prediction Mahalanobis", run_dir / "prediction_maha.png",
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def batched_indices(total: int, batch_size: int) -> list[list[int]]:
    """Split ``range(total)`` into contiguous index batches of ``batch_size``."""
    return [list(range(start, min(start + batch_size, total))) for start in range(0, total, batch_size)]


def collect_clean(
    adapter: Any,
    samples: list[dict[str, Any]],
    context_ids: list[int],
    target_ids: list[int],
    layers: list[int],
) -> dict[str, Any]:
    """Pass 1: batched clean forward passes.

    Caches per-sample context latents, predictions, targets, and per-layer
    activations (on CPU, float16) so pass 2 only re-runs the predictor.
    """
    batch_size = int(CONFIG["BATCH_SIZE"])
    context_chunks: list[torch.Tensor] = []
    prediction_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    activation_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    batches = batched_indices(len(samples), batch_size)
    for batch_number, indices in enumerate(batches, start=1):
        obs = load_image_batch([samples[i] for i in indices], adapter)
        captured = forward_capture(adapter, obs, context_ids, target_ids, layers)
        context_chunks.append(captured["context_latents"].to("cpu", torch.float16))
        prediction_chunks.append(captured["prediction"].to("cpu", torch.float16))
        target_chunks.append(captured["target"].to("cpu", torch.float16))
        for layer in layers:
            activation_chunks[layer].append(
                captured["activations"][layer].to("cpu", torch.float16)
            )
        print(f"  clean pass: batch {batch_number}/{len(batches)}", flush=True)

    return {
        "context_latents": torch.cat(context_chunks, dim=0),
        "predictions": torch.cat(prediction_chunks, dim=0),
        "targets": torch.cat(target_chunks, dim=0),
        "activations": {layer: torch.cat(chunks, dim=0) for layer, chunks in activation_chunks.items()},
    }


def build_row(
    sample: dict[str, Any],
    sample_index: int,
    layer: int,
    mode: str,
    donor_index: int | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one per-sample result record."""
    return {
        "sample_index": sample_index,
        "label": sample["label"],
        "class_name": sample["class_name"],
        "layer": layer,
        "mode": mode,
        "donor_index": donor_index,
        **metrics,
    }


def run_interventions(
    adapter: Any,
    samples: list[dict[str, Any]],
    context_ids: list[int],
    target_ids: list[int],
    layers: list[int],
    clean: dict[str, Any],
    layer_means: dict[int, torch.Tensor],
    layer_detectors: dict[int, MahalanobisOODDetector],
    prediction_detector: MahalanobisOODDetector,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], list[list[float]]]]:
    """Pass 2: batched interventions for every (layer, mode).

    Returns per-sample metric rows and pooled probe feature sets. Only the
    predictor is re-run, with the target layer's activation replaced.
    """
    device, dtype = model_device_dtype(adapter)
    num_samples = len(samples)
    batch_size = int(CONFIG["BATCH_SIZE"])
    batches = batched_indices(num_samples, batch_size)

    rows: list[dict[str, Any]] = []
    feature_sets: dict[tuple[int, str], list[list[float]]] = {}

    for layer in layers:
        layer_mean = layer_means[layer]
        layer_detector = layer_detectors[layer]
        for mode in CONFIG["ABLATION_MODES"]:
            features: list[list[float]] = []
            for indices in batches:
                context_batch = clean["context_latents"][indices].to(device, dtype)

                replacement = None
                donor_indices = None
                if mode == "mean":
                    replacement = (
                        layer_mean.to(device, dtype).unsqueeze(0).expand(len(indices), -1, -1)
                    )
                    substitution_batch = layer_mean.unsqueeze(0).expand(len(indices), -1, -1)
                elif mode == "resample":
                    donor_indices = [(i + 1) % num_samples for i in indices]
                    donor = clean["activations"][layer][donor_indices]
                    replacement = donor.to(device, dtype)
                    substitution_batch = donor.float()
                else:  # zero
                    substitution_batch = torch.zeros(
                        len(indices), layer_mean.shape[0], layer_mean.shape[1]
                    )

                prediction = forward_intervene(
                    adapter, context_batch, context_ids, target_ids, layer, mode, replacement
                )
                prediction_cpu = prediction.to("cpu", torch.float32)
                features.extend(prediction_cpu.mean(dim=1).tolist())

                for position, sample_index in enumerate(indices):
                    metrics = prediction_metrics(
                        prediction_cpu[position],
                        clean["targets"][sample_index].float(),
                        clean["predictions"][sample_index].float(),
                        substitution_batch[position],
                        layer_detector,
                        prediction_detector,
                    )
                    donor_index = donor_indices[position] if donor_indices is not None else None
                    rows.append(
                        build_row(
                            samples[sample_index], sample_index, layer, mode, donor_index, metrics
                        )
                    )
            feature_sets[(layer, mode)] = features
        print(f"  interventions: finished predictor layer {layer}", flush=True)

    return rows, feature_sets


def build_clean_rows(
    samples: list[dict[str, Any]],
    layers: list[int],
    clean: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    """Build clean-baseline rows (mode='clean') and pooled clean probe features.

    Clean rows carry zeroed effect metrics (the prediction is its own baseline)
    and no Mahalanobis values.
    """
    rows: list[dict[str, Any]] = []
    clean_features = clean["predictions"].float().mean(dim=1).tolist()
    for layer in layers:
        for sample_index, sample in enumerate(samples):
            prediction = clean["predictions"][sample_index].float()
            target = clean["targets"][sample_index].float()
            pred = prediction.flatten()
            tgt = target.flatten()
            mse = float(torch.mean((pred - tgt) ** 2))
            rows.append(
                build_row(
                    sample,
                    sample_index,
                    layer,
                    "clean",
                    None,
                    {
                        "prediction_mse": mse,
                        "prediction_mse_delta": 0.0,
                        "prediction_mse_ratio": 1.0,
                        "target_cosine": float(
                            torch.nn.functional.cosine_similarity(pred, tgt, dim=0)
                        ),
                        "prediction_shift_l2": 0.0,
                        "clean_prediction_cosine": 1.0,
                        "substitution_maha": None,
                        "prediction_maha": None,
                    },
                )
            )
    return rows, clean_features


def main() -> None:
    """Run the full Task 4 pipeline and write results to a timestamped run dir."""
    validate_config()
    seed_everything()
    run_dir = create_run_directory()

    samples = load_imagenet_subset(
        CONFIG["IMAGENET_ROOT"],
        num_samples=int(CONFIG["NUM_SAMPLES"]),
        num_classes=int(CONFIG["NUM_CLASSES"]),
        seed=int(CONFIG["SEED"]),
    )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(samples, indent=2), encoding="utf-8"
    )

    adapter = load_world_model()

    layers = resolve_layers(adapter)
    context_ids, target_ids = build_fixed_masks(adapter)
    print(f"Sweeping predictor layers {layers} over {len(samples)} samples.", flush=True)

    # Pass 1: clean forward passes, cached for reuse.
    clean = collect_clean(adapter, samples, context_ids, target_ids, layers)

    # Fit the library's Mahalanobis detector to token-pooled clean features.
    layer_means: dict[int, torch.Tensor] = {}
    layer_detectors: dict[int, MahalanobisOODDetector] = {}
    for layer in layers:
        activations = clean["activations"][layer].float()
        layer_means[layer] = activations.mean(dim=0)
        layer_detectors[layer] = MahalanobisOODDetector().fit(activations.mean(dim=1))
    prediction_detector = MahalanobisOODDetector().fit(
        clean["predictions"].float().mean(dim=1)
    )

    # Pass 2: interventions.
    intervention_rows, feature_sets = run_interventions(
        adapter,
        samples,
        context_ids,
        target_ids,
        layers,
        clean,
        layer_means,
        layer_detectors,
        prediction_detector,
    )

    clean_rows, clean_features = build_clean_rows(samples, layers, clean)
    for layer in layers:
        feature_sets[(layer, "clean")] = clean_features
    rows = clean_rows + intervention_rows

    labels = [int(sample["label"]) for sample in samples]
    summaries = aggregate(rows)
    probe_results = add_probe_results(summaries, feature_sets, labels)

    (run_dir / "summary_metrics.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    (run_dir / "per_sample_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    save_plots(run_dir, summaries, layers)

    results = {
        "config": CONFIG,
        "model": {
            "name": CONFIG["MODEL_NAME"],
            "predictor_depth": len(adapter.predictor.blocks),
            "layers": layers,
            "context_patches": context_ids,
            "target_patches": target_ids,
        },
        "probe_results": probe_results,
        "summary": summaries,
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Saved Task 4 run to {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
