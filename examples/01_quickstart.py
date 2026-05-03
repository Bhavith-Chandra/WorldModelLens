"""Example 01: Quickstart - Load model, run cache, train probe.

This example demonstrates the basic workflow:
1. Create a world model adapter
2. Wrap it in HookedWorldModel
3. Run forward pass with caching
4. Train a linear probe on cached activations
"""

import pathlib

import torch
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.envs import GymnasiumAdapter
from world_model_lens.visualization import plot_quickstart_dashboard


def main():
    print("=" * 60)
    print("World Model Lens - Quickstart Example")
    print("=" * 60)

    cfg = WorldModelConfig(
        d_h=256,
        n_cat=32,
        n_cls=32,
        d_action=1,
        d_obs=3,
    )
    print(f"\n[1] Config created: d_h={cfg.d_h}, n_cat={cfg.n_cat}, n_cls={cfg.n_cls}")

    adapter = DreamerV3Adapter(cfg)
    print("\n[2] DreamerV3Adapter initialized")

    wm = HookedWorldModel(adapter=adapter, config=cfg, name="quickstart")
    print("\n[3] HookedWorldModel wrapper created")

    env = GymnasiumAdapter("Pendulum-v1")
    obs, _ = env.reset(seed=42)
    obs_list, action_list = [], []
    T = 10
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
    print(f"\n[4] Collected {len(obs_list)} steps from Pendulum-v1: obs={obs_seq.shape}, actions={action_seq.shape}")

    world_traj, latent_traj, cache = wm.run_with_cache(obs_seq, action_seq)
    print(f"\n[5] Forward pass complete!")
    print(f"    Trajectory length: {traj.length}")
    print(f"    Cache keys: {cache.component_names}")

    h_t = cache["h", 0]
    z_posterior = cache["z_posterior", 0]
    print(f"\n[6] Sample activations:")
    print(f"    h_t shape: {h_t.shape}")
    print(f"    z_posterior shape: {z_posterior.shape}")

    imagined_world, imagined_latent = wm.imagine(start_state=latent_traj.states[5], horizon=20)
    print(f"\n[7] Imagination complete: {imagined.length} steps")

    print("\n[8] Building visualization dashboard...")
    plot_quickstart_dashboard(
        traj=traj,
        cache=cache,
        imagined_traj=imagined,
        n_cat=cfg.n_cat,
        n_cls=cfg.n_cls,
        wm=wm,
        output_path=OUTPUT_DIR / "quickstart_dashboard.png",
    )
    print("    Saved quickstart_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
