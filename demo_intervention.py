import torch
import numpy as np
import plotly.graph_objects as go
from world_model_lens import HookedWorldModel, WorldModelConfig, HookPoint
from world_model_lens.backends.toy_scientific_model import ToyScientificAdapter
from world_model_lens.visualization.intervention_plots import InterventionVisualizer

def main():
    print("=== Intervention Visualization Demo ===")

    # 1. Initialize a World Model (using ToyScientificAdapter)
    cfg = WorldModelConfig(d_h=16, n_cat=16, n_cls=16, d_action=4, d_obs=10)
    adapter = ToyScientificAdapter(cfg, obs_dim=10, latent_dim=16)
    wm = HookedWorldModel(adapter=adapter, config=cfg)
    
    # 2. Setup inputs
    obs_seq = torch.randn(15, 10)
    action_seq = torch.randn(15, cfg.d_action)

    # 3. Get the "before" trajectory  (clean run)
    print("Running clean trajectory...")
    clean_traj, _ = wm.run_with_cache(obs_seq, action_seq)

    # 4. Create an intervention: we will "corrupt" the observations starting at t=5
    print("Running intervened trajectory (corrupted obs from t=5)...")
    obs_corrupted = obs_seq.clone()
    obs_corrupted[5:] = torch.randn_like(obs_corrupted[5:])
    cf_traj, _ = wm.run_with_cache(obs_corrupted, action_seq)

    # 5. Use InterventionVisualizer
    print("\n--- Visualizing Intervention ---")
    viz = InterventionVisualizer(wm)
    
    # Get high-level summary
    summary = viz.intervention_summary(clean_traj, cf_traj)
    print(f"Total Cumulative Divergence: {summary['total_divergence']:.4f}")
    print(f"Number of timesteps affected: {summary['affected_timesteps']}")

    # Get the detailed divergence curve over time
    curve = viz.divergence_curve(clean_traj, cf_traj)
    print("\nDivergence over time:")
    for t in range(4, 8):  # print a few timesteps around the intervention
        print(f"  t={t}: {curve.get(t, 0):.4f}")

    # Generate a heatmap array (Timesteps x Latent Dimensions)
    heatmap = viz.intervention_heatmap(clean_traj, cf_traj)
    print("\nHeatmap array shape [Timesteps, Latent Dims]:", heatmap.shape)
    # Show the effect at timestep 6 for the first 5 dimensions
    print("Heatmap values at t=6 (first 5 dims):", heatmap[6, :5])
    
    fig = go.Figure(data=go.Heatmap(z=heatmap[6, :5]))
    fig.show()

if __name__ == "__main__":
    main()
