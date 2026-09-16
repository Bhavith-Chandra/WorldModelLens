#!/usr/bin/env python3
"""Task 4 (next step): activation interventions inside the I-JEPA CONTEXT ENCODER.

The original Task 4 sweep intervened only on the *predictor*. This script mirrors
the same whole-state methodology one stage earlier: it overwrites the residual
stream at each of the context encoder's transformer blocks
(``context_encoder.blocks[N].hook_resid_post``), re-runs the encoder *and* the
predictor, and measures how far the resulting target-block prediction drifts.

Substitution modes (identical semantics to the predictor sweep):
  * ``zero``     -- blank the whole layer activation (off-distribution by design)
  * ``mean``     -- the dataset-average activation at that layer (cloud centre)
  * ``resample`` -- another image's real activation at that layer (rolled donor)

Because the intervention lives *inside* the encoder, every (layer, mode) pass
must re-run the full 32-layer encoder, so activations are cached one layer at a
time (storing all 32 layers at once would need terabytes). Probes use a single
fit (no CV) -- with 32 layers a CV sweep would take hours; accuracy is still
reported. Metrics, aggregation, Mahalanobis detectors, probes and plots are
reused verbatim from ``run_ijepa_task4.py`` so results are directly comparable.

Run: ``PYTHONPATH=<repo> MPLBACKEND=Agg python scripts/run_ijepa_task4_context_encoder.py``
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_ijepa_task4 as t4  # noqa: E402  (reuse the whole Task-4 toolbox)

from world_model_lens.data import load_imagenet_subset  # noqa: E402

# ---------------------------------------------------------------------------
# Reconfigure the shared Task-4 CONFIG for the context-encoder sweep.
# ---------------------------------------------------------------------------
CONFIG = t4.CONFIG
CONFIG["OUTPUT_ROOT"] = os.environ.get(
    "WML_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "ijepa_task4_context_encoder")
)
CONFIG["PROBE_USE_CV"] = False  # 32 layers x CV would take hours
CONFIG["SHOW_PLOTS"] = False
CONFIG["TARGET_LAYERS"] = None  # None -> sweep every encoder block


@torch.no_grad()
def capture_layer_activation(
    adapter: Any,
    obs_all: torch.Tensor,
    context_ids: list[int],
    layer: int,
) -> torch.Tensor:
    """Clean pass over the whole set capturing one encoder block's residual output.

    Returns [num_samples, n_context_tokens, embed_dim] on CPU float16.
    """
    device, dtype = t4.model_device_dtype(adapter)
    batch_size = int(CONFIG["BATCH_SIZE"])
    store: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        store.append(output.detach().to("cpu", torch.float16))

    handle = adapter.context_encoder.blocks[layer].hook_resid_post.register_forward_hook(hook)
    try:
        for start in range(0, obs_all.shape[0], batch_size):
            obs = obs_all[start : start + batch_size].to(device=device, dtype=dtype)
            adapter.context_encoder(obs, patch_ids=context_ids)
    finally:
        handle.remove()
    return torch.cat(store, dim=0)


@torch.no_grad()
def forward_intervene_encoder(
    adapter: Any,
    obs: torch.Tensor,
    context_ids: list[int],
    target_ids: list[int],
    layer: int,
    mode: str,
    replacement: torch.Tensor | None,
) -> torch.Tensor:
    """Re-run encoder+predictor while replacing one encoder block's activation."""

    def hook(_module, _inputs, output):
        if mode == "zero":
            return torch.zeros_like(output)
        if replacement is None:
            raise ValueError(f"mode '{mode}' requires a replacement activation")
        value = replacement.to(device=output.device, dtype=output.dtype)
        if value.shape != output.shape:
            raise ValueError(
                f"Replacement {tuple(value.shape)} != activation {tuple(output.shape)}"
            )
        return value

    handle = adapter.context_encoder.blocks[layer].hook_resid_post.register_forward_hook(hook)
    try:
        context_latents = adapter.context_encoder(obs, patch_ids=context_ids)
        prediction = adapter.predictor(context_latents, context_ids, target_ids)
    finally:
        handle.remove()
    return prediction.detach()


@torch.no_grad()
def collect_clean_baseline(
    adapter: Any,
    obs_all: torch.Tensor,
    context_ids: list[int],
    target_ids: list[int],
) -> dict[str, torch.Tensor]:
    """One clean pass: cache predictions and targets (CPU float16)."""
    device, dtype = t4.model_device_dtype(adapter)
    batch_size = int(CONFIG["BATCH_SIZE"])
    preds, tgts = [], []
    for start in range(0, obs_all.shape[0], batch_size):
        obs = obs_all[start : start + batch_size].to(device=device, dtype=dtype)
        context_latents = adapter.context_encoder(obs, patch_ids=context_ids)
        prediction = adapter.predictor(context_latents, context_ids, target_ids)
        target_full = adapter.target_encoder(obs)
        preds.append(prediction.to("cpu", torch.float16))
        tgts.append(target_full[:, target_ids, :].to("cpu", torch.float16))
    return {"predictions": torch.cat(preds, 0), "targets": torch.cat(tgts, 0)}


def main() -> None:
    t4.validate_config()
    t4.seed_everything()
    run_dir = t4.create_run_directory()

    samples = load_imagenet_subset(
        CONFIG["IMAGENET_ROOT"],
        num_samples=int(CONFIG["NUM_SAMPLES"]),
        num_classes=int(CONFIG["NUM_CLASSES"]),
        seed=int(CONFIG["SEED"]),
    )
    (run_dir / "dataset_manifest.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    adapter = t4.load_world_model()
    device, dtype = t4.model_device_dtype(adapter)
    context_ids, target_ids = t4.build_fixed_masks(adapter)
    depth = len(adapter.context_encoder.blocks)
    layers = list(range(depth))
    num_samples = len(samples)
    modes = list(CONFIG["ABLATION_MODES"])
    print(
        f"Context-encoder sweep: {depth} layers x {modes} over {num_samples} samples.", flush=True
    )

    # Preload & cache all preprocessed images once (reused for every pass).
    t0 = time.time()
    obs_all = torch.cat(
        [t4.load_image_batch([s], adapter).to("cpu", torch.float16) for s in samples], dim=0
    )
    print(f"  preloaded {obs_all.shape[0]} images in {time.time()-t0:.1f}s", flush=True)

    # Clean baseline (predictions + targets) and the prediction detector.
    clean = collect_clean_baseline(adapter, obs_all, context_ids, target_ids)
    prediction_detector = t4.MahalanobisOODDetector().fit(clean["predictions"].float().mean(dim=1))
    clean_features = clean["predictions"].float().mean(dim=1).tolist()

    rows: list[dict[str, Any]] = []
    feature_sets: dict[tuple[int, str], list[list[float]]] = {}
    batches = t4.batched_indices(num_samples, int(CONFIG["BATCH_SIZE"]))

    for layer in layers:
        t_layer = time.time()
        act = capture_layer_activation(adapter, obs_all, context_ids, layer)  # [N, T, D] cpu fp16
        layer_mean = act.float().mean(dim=0)  # [T, D]
        layer_detector = t4.MahalanobisOODDetector().fit(act.float().mean(dim=1))

        for mode in modes:
            features: list[list[float]] = []
            for indices in batches:
                obs = obs_all[indices].to(device=device, dtype=dtype)
                replacement = None
                donor_indices = None
                if mode == "mean":
                    replacement = (
                        layer_mean.to(device, dtype).unsqueeze(0).expand(len(indices), -1, -1)
                    )
                    substitution_batch = layer_mean.unsqueeze(0).expand(len(indices), -1, -1)
                elif mode == "resample":
                    donor_indices = [(i + 1) % num_samples for i in indices]
                    donor = act[donor_indices]
                    replacement = donor.to(device, dtype)
                    substitution_batch = donor.float()
                else:  # zero
                    substitution_batch = torch.zeros(
                        len(indices), layer_mean.shape[0], layer_mean.shape[1]
                    )

                prediction = forward_intervene_encoder(
                    adapter, obs, context_ids, target_ids, layer, mode, replacement
                )
                prediction_cpu = prediction.to("cpu", torch.float32)
                features.extend(prediction_cpu.mean(dim=1).tolist())

                for position, sample_index in enumerate(indices):
                    metrics = t4.prediction_metrics(
                        prediction_cpu[position],
                        clean["targets"][sample_index].float(),
                        clean["predictions"][sample_index].float(),
                        substitution_batch[position],
                        layer_detector,
                        prediction_detector,
                    )
                    donor_index = donor_indices[position] if donor_indices is not None else None
                    rows.append(
                        t4.build_row(
                            samples[sample_index], sample_index, layer, mode, donor_index, metrics
                        )
                    )
            feature_sets[(layer, mode)] = features

        del act
        print(f"  layer {layer:2d}/{depth-1} done in {time.time()-t_layer:.1f}s", flush=True)

    # GPU work is done; the rest of the pipeline (probe training, aggregation,
    # plotting) is CPU-only, so release the model and its cached CUDA memory
    # instead of holding it idle for the remainder of the run.
    predictor_depth = len(adapter.predictor.blocks)
    n_patches = adapter.context_encoder.patch_embed.n_patches
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clean-baseline rows (mode='clean') for every layer, using cached predictions.
    for layer in layers:
        for sample_index, sample in enumerate(samples):
            pred = clean["predictions"][sample_index].float().flatten()
            tgt = clean["targets"][sample_index].float().flatten()
            mse = float(torch.mean((pred - tgt) ** 2))
            rows.append(
                t4.build_row(
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
        feature_sets[(layer, "clean")] = clean_features

    labels = [int(s["label"]) for s in samples]
    summaries = t4.aggregate(rows)
    print("  fitting probes (no CV) ...", flush=True)
    probe_results = t4.add_probe_results(summaries, feature_sets, labels)

    (run_dir / "summary_metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (run_dir / "per_sample_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    t4.save_plots(run_dir, summaries, layers)

    results = {
        "config": CONFIG,
        "model": {
            "name": CONFIG["MODEL_NAME"],
            "intervention_locus": "context_encoder",
            "context_encoder_depth": depth,
            "predictor_depth": predictor_depth,
            "layers": layers,
            "context_patches": context_ids,
            "target_patches": target_ids,
            "embedding_dim": int(n_patches and layer_mean.shape[1]),
        },
        "probe_results": probe_results,
        "summary": summaries,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved context-encoder Task 4 run to {run_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
