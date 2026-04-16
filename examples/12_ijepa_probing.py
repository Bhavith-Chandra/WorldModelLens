"""Example 12: I-JEPA Probing.

This example mirrors the general structure of examples/02_probing.py while
using the I-JEPA adapter configuration exercised in tests/backends/test_ijepa_adapter.py.

It demonstrates:
1. Building a small probing dataset from cached I-JEPA context patches
2. Creating patch-position labels from the sampled context mask
3. Training the currently available linear probes
4. Printing cross-validation statistics for each concept
"""

from __future__ import annotations

import numpy as np
import torch

from world_model_lens import HookedWorldModel, LatentProber, WorldModelConfig
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter


def _build_patch_labels(
    patch_ids: list[int],
    grid_size: int,
    sample_index: int,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Build simple patch-level labels from linear patch indices."""
    patch_ids_np = np.asarray(patch_ids, dtype=np.int64)
    rows = patch_ids_np // grid_size
    cols = patch_ids_np % grid_size

    center_low = grid_size // 4
    center_high = grid_size - center_low
    sample_offset = sample_index / max(n_samples * grid_size * grid_size, 1)

    return {
        "top_half": (rows < grid_size // 2).astype(np.int64),
        "left_half": (cols < grid_size // 2).astype(np.int64),
        "center_band": ((rows >= center_low) & (rows < center_high)).astype(np.int64),
        "patch_index_norm": patch_ids_np.astype(np.float32) / max(grid_size * grid_size - 1, 1)
        + sample_offset,
    }


def _collect_probe_dataset(
    wm: HookedWorldModel,
    adapter: IJEPAAdapter,
    n_samples: int = 4,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Collect context-patch activations and aligned synthetic labels."""
    all_activations = []
    label_buckets: dict[str, list[np.ndarray]] = {
        "top_half": [],
        "left_half": [],
        "center_band": [],
        "patch_index_norm": [],
    }

    grid_size = adapter.context_encoder.patch_embed.grid_size

    for sample_index in range(n_samples):
        obs = torch.randn(1, 3, 64, 64)
        _, cache = wm.run_with_cache(obs)

        context_latents = cache["z_posterior", 0]
        if context_latents.dim() != 2:
            context_latents = context_latents.reshape(context_latents.shape[0], -1)

        patch_labels = _build_patch_labels(adapter.last_context_ids, grid_size, sample_index, n_samples)

        all_activations.append(context_latents)
        for name, values in patch_labels.items():
            label_buckets[name].append(values)

    activations = torch.cat(all_activations, dim=0)
    labels_dict = {name: np.concatenate(values, axis=0) for name, values in label_buckets.items()}
    return activations, labels_dict


def main() -> None:
    print("=" * 60)
    print("World Model Lens - I-JEPA Probing Example")
    print("=" * 60)

    config = WorldModelConfig(
        backend="ijepa",
        d_embed=32,
        n_heads=4,
        predictor_embed_dim=64,
        predictor_heads=4,
        n_layers=1,
        predictor_depth=1,
    )
    config.img_size = 64
    config.patch_size = 16

    adapter = IJEPAAdapter(config)
    wm = HookedWorldModel(adapter=adapter, config=config)
    prober = LatentProber(seed=42, n_folds=3)

    print("\n[1] Collecting cached context-patch activations...")
    activations, labels_dict = _collect_probe_dataset(wm, adapter, n_samples=4)
    print(f"    Collected {activations.shape[0]} patch activations with shape {tuple(activations.shape)}")

    print("\n[2] Training probes...")

    probe_specs = [
        ("top_half", "logistic"),
        ("left_half", "logistic"),
        ("center_band", "logistic"),
        ("patch_index_norm", "ridge"),
    ]

    results = {}
    for concept_name, probe_type in probe_specs:
        results[concept_name] = prober.train_probe(
            activations=activations,
            labels=labels_dict[concept_name],
            concept_name=concept_name,
            activation_name="z_posterior",
            probe_type=probe_type,
        )

    print("\n[3] Probe results:")
    for concept_name, result in results.items():
        metric_name = "r2" if result.r2 is not None else "accuracy"
        metric_value = result.r2 if result.r2 is not None else result.accuracy
        print(
            "    "
            f"{concept_name:<16} "
            f"probe={result.probe_type:<8} "
            f"{metric_name}={metric_value:.3f} "
            f"cv_mean={result.cv_mean:.3f} "
            f"cv_std={result.cv_std:.3f} "
            f"alpha={result.regularization_alpha:.3g}"
        )

    print("\n[4] Label summary:")
    for name, labels in labels_dict.items():
        unique_count = len(np.unique(labels))
        print(f"    {name:<16} samples={len(labels):<4} unique_values={unique_count}")

    print("\n" + "=" * 60)
    print("I-JEPA probing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
