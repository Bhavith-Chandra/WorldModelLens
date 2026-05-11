"""Example 05: Belief Analysis - Surprise, concepts, saliency.

This example demonstrates:
1. Computing surprise timeline
2. Searching for concept alignment
3. Computing saliency maps
4. Detecting hallucinations
5. Visualizing belief analysis results
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
from world_model_lens.visualization import plot_belief_dashboard


def main():
    print("=" * 60)
    print("World Model Lens - Belief Analysis Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)
    analyzer = BeliefAnalyzer(wm)

    T = 20
    n_cat, n_cls = cfg.n_cat, cfg.n_cls

    print("\n[1] Running forward pass...")

    obs_seq = torch.randn(T, 3, 64, 64)
    action_seq = torch.randn(T, cfg.d_action)

    traj, cache = wm.run_with_cache(obs_seq, action_seq)

    print("\n[2] Computing surprise timeline...")

    surprise_result = analyzer.surprise_timeline(cache)
    print(f"    Mean surprise: {surprise_result.mean_surprise:.4f}")
    print(f"    Max surprise at t={surprise_result.max_surprise_timestep}")
    print(f"    Peak count: {len(surprise_result.peaks)}")

    print("\n[3] Searching for concept alignment...")

    pos_t = [0, 1, 2, 3, 4]
    neg_t = [10, 11, 12, 13, 14]

    concept_result = analyzer.concept_search(
        concept_name="early_vs_late",
        positive_timesteps=pos_t,
        negative_timesteps=neg_t,
        cache=cache,
        component="z_posterior",
    )

    print(f"    Top dims: {concept_result.top_dims[:5]}")
    print(f"    Method: {concept_result.method}")

    print("\n[4] Computing saliency...")

    saliency_result = analyzer.latent_saliency(
        traj=traj,
        cache=cache,
        timestep=5,
        target="reward_pred",
    )

    print(f"    h_saliency shape: {saliency_result.h_saliency.shape}")
    print(f"    z_saliency shape: {saliency_result.z_saliency.shape}")

    print("\n[5] Detecting hallucinations...")

    imagined = wm.imagine(start_state=traj.states[0], horizon=20)

    hallucination_result = analyzer.detect_hallucinations(
        real_traj=traj,
        imagined_traj=imagined,
        method="latent_distance",
        threshold=0.5,
    )

    print(f"    Severity score: {hallucination_result.severity_score:.4f}")
    print(f"    Hallucination timesteps: {hallucination_result.hallucination_timesteps}")

    print("\n[6] Building visualization dashboard...")
    plot_belief_dashboard(
        wm=wm,
        traj=traj,
        cache=cache,
        saliency_result=saliency_result,
        hallucination_result=hallucination_result,
        n_cat=n_cat,
        n_cls=n_cls,
        imagined_traj=imagined,
        output_path=OUTPUT_DIR / "belief_analysis_dashboard.png",
    )
    print("    Saved belief_analysis_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Belief analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
