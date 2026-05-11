"""Example 02: Probing - Train linear probes on cached activations.

This example demonstrates:
1. Collecting activations from multiple trajectories
2. Creating synthetic labels for concepts
3. Training linear probes
4. Analyzing probe results
5. Visualizing latent structure
"""

import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("assets/examples")

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "assets" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from world_model_lens import HookedWorldModel, LatentProber, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.visualization import plot_probing_dashboard


def main():
    print("=" * 60)
    print("World Model Lens - Linear Probing Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    print("\n[1] Collecting activations from 5 trajectories...")

    all_activations = []
    all_labels = []

    for ep in range(5):
        obs_seq = torch.randn(20, 3, 64, 64)
        action_seq = torch.randn(20, cfg.d_action)

        traj, cache = wm.run_with_cache(obs_seq, action_seq)

        z_posterior = cache["z_posterior"]
        z_flat = z_posterior.flatten(1) if z_posterior.dim() > 2 else z_posterior
        all_activations.append(z_flat)

        labels = np.random.randint(0, 3, size=len(z_flat))
        all_labels.append(labels)

    activations = torch.cat(all_activations, dim=0)
    labels = np.concatenate(all_labels)

    print(f"    Collected {activations.shape[0]} activations, shape: {activations.shape}")

    print("\n[2] Training linear probes...")

    prober = LatentProber(seed=42)

    concepts = {
        "reward_region": (labels == 0).astype(np.float32),
        "novel_state": (labels == 1).astype(np.float32),
        "high_value": (labels == 2).astype(np.float32),
    }

    sweep_result = prober.sweep(
        cache={"z_posterior": activations},
        activation_names=["z_posterior"],
        labels_dict=concepts,
        probe_type="linear",
    )

    print("\n[3] Probe Results:")
    for key, result in sweep_result.results.items():
        print(f"    {key}: accuracy={result.accuracy:.3f}")

    print("\n[4] Building visualization dashboard...")
    plot_probing_dashboard(
        sweep_result=sweep_result,
        activations=activations,
        labels=labels,
        concepts=concepts,
        cache=cache,
        output_path=OUTPUT_DIR / "probing_dashboard.png",
    )
    print("    Saved probing_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Probing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
