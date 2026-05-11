"""Example 12: AOPC Faithfulness Analysis

This example demonstrates how to use the AOPC (Area Over Perturbation Curve)
metric to evaluate the faithfulness of latent representations in a world model.

Steps:
1. Set up a world model
2. Generate observations
3. Compute AOPC for different components
4. Visualize results
"""

import pathlib

import torch
import matplotlib.pyplot as plt

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.analysis import FaithfulnessAnalyzer

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("AOPC Faithfulness Analysis Example")
    print("=" * 60)

    # Setup world model
    cfg = WorldModelConfig(d_h=256, n_cat=32, n_cls=32, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg, name="aopc_example")

    # Generate data
    T, C, H, W = 20, 3, 64, 64
    obs_seq = torch.randn(T, C, H, W)
    action_seq = torch.randn(T, cfg.d_action)

    print(f"Created data: obs={obs_seq.shape}, actions={action_seq.shape}")

    # Initialize analyzer
    analyzer = FaithfulnessAnalyzer(wm)

    # Compute AOPC for z_posterior
    print("\nComputing AOPC for z_posterior...")
    result_z = analyzer.aopc(obs_seq, actions=action_seq, target_component="z_posterior", max_k=10)

    print(f"AOPC Score for z_posterior: {result_z.aopc_score:.4f}")

    # Compute for h
    print("\nComputing AOPC for h...")
    result_h = analyzer.aopc(obs_seq, actions=action_seq, target_component="h", max_k=10)

    print(f"AOPC Score for h: {result_h.aopc_score:.4f}")

    # Plot curves
    print("\nGenerating plots...")
    fig_z = result_z.plot()
    fig_z.savefig(OUTPUT_DIR / "aopc_z_posterior.png")
    plt.show()

    fig_h = result_h.plot()
    fig_h.savefig(OUTPUT_DIR / "aopc_h.png")
    plt.show()

    print(f"Saved plots to {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("AOPC example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
