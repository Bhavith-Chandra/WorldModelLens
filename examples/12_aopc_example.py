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
from world_model_lens.envs import GymnasiumAdapter
from world_model_lens.analysis import FaithfulnessAnalyzer

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("AOPC Faithfulness Analysis Example")
    print("=" * 60)

    # Setup world model
    # Pendulum-v1 has a 3-dim obs and 1-dim continuous action matching these config values.
    cfg = WorldModelConfig(d_h=256, n_cat=32, n_cls=32, d_action=1, d_obs=3)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg, name="aopc_example")

    # AOPC measures faithfulness of latent representations — perturbation curves computed on
    # random noise have no ground truth to be faithful to.
    T = 20
    env = GymnasiumAdapter("Pendulum-v1")
    obs, _ = env.reset(seed=42)
    obs_list, action_list = [], []
    for _ in range(T):
        action = env.action_space.sample()
        obs_list.append(torch.from_numpy(obs).float())
        action_list.append(torch.from_numpy(action).float())
        result = env.step(action)
        obs = result.observation
        if result.done:
            break
    env.close()
    obs_seq = torch.stack(obs_list)
    action_seq = torch.stack(action_list)
    print(f"Collected {len(obs_list)} steps from Pendulum-v1: obs={obs_seq.shape}, actions={action_seq.shape}")

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
