"""Example 09: Toy Scientific Latent Dynamics Analysis.

This example demonstrates using WorldModelLens with a non-RL scientific dynamics model.
The ToyScientificAdapter implements a simple latent dynamics model with:
- Encoder from observation vectors to latent states
- Latent dynamics MLP (no actions, nonlinear dynamics)
- No decoder, no rewards, no actions

This example shows:
1. Creating a toy scientific dynamics model adapter
2. Using synthetic trajectories (Lorenz attractor, pendulum)
3. Analyzing latent geometry and temporal memory
4. Understanding scientific model dynamics

No RL-specific features are used - this is pure scientific modeling.
"""

import torch

from world_model_lens import HookedWorldModel
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer
from world_model_lens.probing.geometry import GeometryAnalyzer
from world_model_lens.probing.temporal_memory import TemporalMemoryProber
from world_model_lens.backends.toy_scientific_model import (
    ToyScientificAdapter,
    generate_lorenz_trajectory,
    generate_pendulum_trajectory,
)
from world_model_lens.visualization import plot_scientific_dynamics_dashboard
import pathlib
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_scientific_model(obs_dim: int = 3, latent_dim: int = 16):
    """Create a toy scientific dynamics model."""
    config = type(
        "Config",
        (),
        {
            "obs_dim": obs_dim,
            "latent_dim": latent_dim,
            "hidden_dim": 64,
        },
    )()

    adapter = ToyScientificAdapter(
        config=config,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=64,
    )

    return HookedWorldModel(adapter=adapter, config=config, name="toy_scientific")


def main():
    print("=" * 60)
    print("WorldModelLens: Toy Scientific Dynamics Analysis")
    print("=" * 60)

    print("\n[1] Creating toy scientific world model...")
    wm = create_scientific_model(obs_dim=3, latent_dim=16)
    print(f"    Model name: {wm.name}")
    print(f"    Has decoder: {wm.capabilities.has_decoder}")
    print(f"    Has reward head: {wm.capabilities.has_reward_head}")
    print(f"    Uses actions: {wm.capabilities.uses_actions}")
    print(f"    Is RL model: {wm.capabilities.is_rl_model()}")

    print("\n[2] Generating Lorenz attractor trajectory...")
    lorenz_traj = generate_lorenz_trajectory(n_steps=100, dt=0.02)
    print(f"    Trajectory shape: {lorenz_traj.shape}")
    print(f"    X range: [{lorenz_traj[:, 0].min():.2f}, {lorenz_traj[:, 0].max():.2f}]")
    print(f"    Y range: [{lorenz_traj[:, 1].min():.2f}, {lorenz_traj[:, 1].max():.2f}]")
    print(f"    Z range: [{lorenz_traj[:, 2].min():.2f}, {lorenz_traj[:, 2].max():.2f}]")

    print("\n[3] Running forward pass with Lorenz trajectory...")
    traj, cache = wm.run_with_cache(lorenz_traj)
    print(f"    Trajectory states: {len(traj.states)}")
    print(f"    Cache keys: {list(cache.keys())[:5]}...")

    print("\n[4] Belief Analysis (surprise timeline)...")
    belief_analyzer = BeliefAnalyzer(wm)
    surprise_result = belief_analyzer.surprise_timeline(cache)
    print(f"    Mean surprise: {surprise_result.mean_surprise:.4f}")
    print(f"    Max surprise: {surprise_result.max_surprise_value:.4f}")
    print(f"    Surprise peaks: {len(surprise_result.peaks)}")

    print("\n[5] Geometry Analysis...")
    geometry_analyzer = GeometryAnalyzer(wm)

    pca_result = geometry_analyzer.pca_projection(cache, component="z_posterior")
    if pca_result.pca_components is not None:
        print(f"    PCA components shape: {pca_result.pca_components.shape}")
        if pca_result.pca_explained_variance is not None:
            print(
                f"    Variance explained (top 3): {pca_result.pca_explained_variance[:3].tolist()}"
            )

    traj_metrics = geometry_analyzer.trajectory_metrics(cache, component="z_posterior")
    print(f"    Mean trajectory distance: {traj_metrics.mean_trajectory_distance:.4f}")
    print(f"    Temporal coherence: {traj_metrics.temporal_coherence:.4f}")
    if traj_metrics.trajectory_curvature is not None:
        print(f"    Mean curvature: {traj_metrics.trajectory_curvature.mean():.4f}")

    cluster_result = geometry_analyzer.clustering(cache, n_clusters=3, component="z_posterior")
    print(f"    Clusters found: {len(torch.unique(cluster_result['clusters']))}")

    manifold_result = geometry_analyzer.manifold_analysis(cache, component="z_posterior")
    print(
        f"    Intrinsic dimensionality estimate: {manifold_result['intrinsic_dimensionality_estimate']:.2f}"
    )
    print(f"    Local linearity: {manifold_result['local_linearity']:.4f}")

    print("\n[6] Temporal Memory Analysis...")
    temporal_prober = TemporalMemoryProber(wm)

    mem_result = temporal_prober.memory_retention(cache, component="z_posterior", max_lag=15)
    print(f"    Memory capacity: {mem_result.memory_capacity:.2f}")
    print(f"    Working memory estimate: {mem_result.working_memory_estimate:.2f}")
    print(f"    Short-term retention: {mem_result.temporal_dependencies.get('short_term', 0):.4f}")
    print(f"    Long-term retention: {mem_result.temporal_dependencies.get('long_term', 0):.4f}")

    dep_result = temporal_prober.temporal_dependencies(cache, component="z_posterior", max_lag=10)
    print(f"    Dominant period: {dep_result['dominant_period']}")

    pattern_result = temporal_prober.sequential_patterns(
        cache, component="z_posterior", pattern_length=3
    )
    print(f"    Top patterns found: {len(pattern_result['patterns'])}")

    print("\n[7] Testing imagination (no actions needed)...")
    start_state = traj.states[0]
    imagined = wm.imagine(start_state, actions=None, horizon=20)
    print(f"    Imagined states: {len(imagined.states)}")

    print("\n[8] Testing RL-specific analysis (should be skipped)...")
    reward_result = belief_analyzer.reward_attribution(traj, cache)
    print(f"    Reward attribution available: {reward_result.is_available}")

    value_result = belief_analyzer.value_analysis(cache)
    print(f"    Value analysis available: {value_result.get('is_available', True)}")

    print("\n[9] Comparing with pendulum dynamics...")
    pendulum_traj = generate_pendulum_trajectory(n_steps=100)
    # Pad pendulum trajectory to match model input dimension (3)
    pendulum_traj = torch.cat([pendulum_traj, torch.zeros(pendulum_traj.shape[0], 1)], dim=1)
    traj2, cache2 = wm.run_with_cache(pendulum_traj)

    surprise2 = belief_analyzer.surprise_timeline(cache2)
    print(f"    Pendulum mean surprise: {surprise2.mean_surprise:.4f}")

    traj_metrics2 = geometry_analyzer.trajectory_metrics(cache2, component="z_posterior")
    print(f"    Pendulum trajectory distance: {traj_metrics2.mean_trajectory_distance:.4f}")
    
    print("\n[10] Plotting scientific dynamics dashboard...")
    fig = plot_scientific_dynamics_dashboard(
        lorenz_traj_raw=lorenz_traj,
        traj_lorenz=traj,
        traj_pendulum=traj2,
        dep_result=dep_result,
        mem_result=mem_result,
        output_path=OUTPUT_DIR / "toy_scientific_dynamics_dashboard.png",
    )
    plt.show()
    print("\n" + "=" * 60)
    print("Summary: Scientific dynamics analysis complete!")
    print("=" * 60)
    print("\nThis demonstrates WorldModelLens working with a scientific model:")
    print("  - Lorenz attractor dynamics analyzed")
    print("  - Pendulum dynamics analyzed")
    print("  - Latent geometry reveals system structure")
    print("  - Temporal memory shows dynamics predictability")
    print("  - RL-specific analysis gracefully skipped")
    print()
    


if __name__ == "__main__":
    main()
