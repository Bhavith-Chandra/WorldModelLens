"""Counterfactual engine: interventions, divergence, and branch comparison.

Demonstrates `CounterfactualEngine` from `world_model_lens.causal.counterfactual`:
baseline rollout, latent ablation, trajectory comparison, divergence tracing,
branch tree, and side-by-side intervention metrics.
"""

import pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.causal import (
    CounterfactualEngine,
    Intervention,
    rollout_comparison,
)
from world_model_lens.visualization import plot_causal_engine_dashboard

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("World Model Lens — CounterfactualEngine walkthrough")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    T = 15
    obs_seq = torch.randn(T, 3, 64, 64)
    action_seq = torch.randn(T, cfg.d_action)

    engine = CounterfactualEngine(wm)

    # --- Baseline trajectory (no hooks)
    baseline_traj, cache = wm.run_with_cache(obs_seq, action_seq)
    print(f"\nBaseline trajectory: {len(baseline_traj.states)} states")
    print(f"  State dim: {baseline_traj.states[0].state.shape}")

    # --- Single intervention: zero selected posterior dimensions at t=5
    ablate_z = Intervention(
        target_timestep=5,
        target_type="dimension",
        target_indices=[0, 1, 2],
        description="Ablate z dims 0–2 at t=5",
    )
    cf_traj = engine.intervene(
        observations=obs_seq,
        intervention=ablate_z,
        actions=action_seq,
    )
    print("\nCounterfactual (dimension ablation on z_posterior): done.")

    # --- Compare trajectories
    metrics = rollout_comparison(baseline_traj, cf_traj)
    print("\nrollout_comparison (baseline vs counterfactual):")
    for k, v in metrics.to_dict().items():
        print(f"  {k}: {v:.6f}")

    # --- Cumulative divergence over time
    div_curve = engine.trace_divergence(baseline_traj, cf_traj)
    sample_ts = [0, T // 2, T - 1]  # 0, 7, 14
    print("\ntrace_divergence (cumulative MSE-style drift) at sample timesteps:")
    for t in sample_ts:
        if t in div_curve:
            print(f"  t={t}: cumulative = {div_curve[t]:.6f}")

    # --- Branch tree: several interventions at the same fork
    interventions = [
        Intervention(
            target_timestep=5,
            target_type="dimension",
            target_indices=[0],
            description="Ablate dim 0",
        ),
        Intervention(
            target_timestep=5,
            target_type="dimension",
            target_indices=[1],
            description="Ablate dim 1",
        ),
        Intervention(
            target_timestep=5,
            target_type="action",
            target_indices=None,
            description="Zero action at t=5",
        ),
    ]
    tree = engine.build_branch_tree(
        observations=obs_seq,
        interventions=interventions,
        base_actions=action_seq,
    )
    print(f"\nBranchTree: {len(tree.branches)} branches from baseline")
    for i, br in enumerate(tree.branches):
        print(
            f"  [{i}] fork={br.fork_point} "
            f"divergence(trajectory_distance)={br.divergence:.6f} "
            f"— {br.intervention.description}"
        )

    # --- Tabular comparison of multiple interventions
    compared = engine.compare_interventions(
        observations=obs_seq,
        interventions=interventions,
        base_actions=action_seq,
        target_metric="reward_pred",
    )
    print("\ncompare_interventions (reward_pred outcome + divergence metrics):")
    for idx, row in compared.items():
        print(f"  intervention {idx}:")
        for key in (
            "intervention_description",
            "target_timestep",
            "baseline_outcome",
            "counterfactual_outcome",
            "outcome_delta",
            "l2_distance",
            "trajectory_distance",
        ):
            if key in row:
                print(f"    {key}: {row[key]}")

    # --- Build visualization dashboard ---
    print("\nBuilding visualization dashboard...")
    plot_causal_engine_dashboard(
        wm=wm,
        div_curve=div_curve,
        tree=tree,
        cache=cache,
        intervention_t=5,
        output_path=OUTPUT_DIR / "causal_engine_dashboard.png",
    )
    print("    Saved causal_engine_dashboard.png")
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
