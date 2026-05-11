"""Example 06: Disentanglement Analysis.

This example demonstrates:
1. Computing disentanglement metrics
2. Analyzing factor representations
3. Visualizing factor-dimension assignments
"""

import pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer
from world_model_lens.visualization import plot_disentanglement_dashboard


def main():
    print("=" * 60)
    print("World Model Lens - Disentanglement Analysis Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)
    analyzer = BeliefAnalyzer(wm)

    print("\n[1] Collecting activations...")

    obs_seq = torch.randn(50, 3, 64, 64)
    action_seq = torch.randn(50, cfg.d_action)

    traj, cache = wm.run_with_cache(obs_seq, action_seq)

    print("\n[2] Creating synthetic factors...")

    factors = {
        "speed": torch.tensor([float(i % 10) / 10 for i in range(50)]),
        "direction": torch.tensor([float((i * 7) % 10) / 10 for i in range(50)]),
        "reward_level": torch.tensor([1.0 if i < 25 else 0.0 for i in range(50)]),
    }

    print(f"    Factors: {list(factors.keys())}")

    print("\n[3] Computing disentanglement metrics...")

    disentanglement_result = analyzer.disentanglement_score(
        cache=cache,
        factors=factors,
        metrics=["MIG", "DCI", "SAP"],
        component="z_posterior",
    )

    print(f"    MIG score: {disentanglement_result.scores.get('MIG', 0):.4f}")
    print(f"    DCI score: {disentanglement_result.scores.get('DCI', 0):.4f}")
    print(f"    SAP score: {disentanglement_result.scores.get('SAP', 0):.4f}")
    print(f"    Total score: {disentanglement_result.total_score:.4f}")

    print("\n[4] Factor assignments:")
    for factor, dims in disentanglement_result.factor_dim_assignment.items():
        print(f"    {factor}: dims {dims[:5]}...")

    print("\n[5] Building visualization dashboard...")
    plot_disentanglement_dashboard(
        disentanglement_result=disentanglement_result,
        cache=cache,
        factors=factors,
        output_path=OUTPUT_DIR / "disentanglement_dashboard.png",
    )
    print("    Saved disentanglement_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Disentanglement analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
