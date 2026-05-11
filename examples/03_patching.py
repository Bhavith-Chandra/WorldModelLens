"""Example 03: Activation Patching - Test causal roles of components.

This example demonstrates:
1. Setting up clean and corrupted runs
2. Patching specific components
3. Measuring recovery rate
4. Full patching sweep
5. Visualizing recovery heatmap and divergence
"""

import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from world_model_lens import HookContext, HookedWorldModel, HookPoint, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.patching.patcher import TemporalPatcher
from world_model_lens.visualization import plot_patching_dashboard


def main():
    print("=" * 60)
    print("World Model Lens - Activation Patching Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    print("\n[1] Creating clean and corrupted runs...")

    obs_seq = torch.randn(15, 3, 64, 64)
    action_seq = torch.randn(15, cfg.d_action)

    clean_traj, clean_cache = wm.run_with_cache(obs_seq, action_seq)

    obs_corrupted = obs_seq.clone()
    obs_corrupted[5:] = torch.randn_like(obs_corrupted[5:])

    corrupted_traj, corrupted_cache = wm.run_with_cache(obs_corrupted, action_seq)

    print(f"    Clean cache timesteps: {len(clean_cache.timesteps)}")
    print(f"    Corrupted cache timesteps: {len(corrupted_cache.timesteps)}")

    print("\n[2] Running patching experiment...")

    patcher = TemporalPatcher(wm)

    components = ["h", "z_posterior", "z_prior"]
    timesteps = [5, 6, 7, 8, 9]

    def reward_metric(pred) -> float:
        return pred.mean().item() if pred is not None else 0.0

    sweep_result = patcher.full_sweep(
        clean_cache=clean_cache,
        corrupted_cache=corrupted_cache,
        components=components,
        metric_fn=reward_metric,
        t_range=timesteps,
        parallel=False,
        clean_obs_seq=obs_seq,
        clean_action_seq=action_seq,
    )

    print("\n[3] Top patches by recovery rate:")
    top_patches = sweep_result.top_k_patches(k=5)
    for patch in top_patches:
        print(f"    {patch.component}@t={patch.timestep}: recovery={patch.recovery_rate:.3f}")

    print("\n[4] Building visualization dashboard...")
    plot_patching_dashboard(
        wm=wm,
        sweep_result=sweep_result,
        clean_traj=clean_traj,
        corrupted_traj=corrupted_traj,
        clean_cache=clean_cache,
        corrupted_cache=corrupted_cache,
        corruption_start_t=5,
        output_path=OUTPUT_DIR / "patching_dashboard.png",
    )
    print("    Saved patching_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Patching complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
