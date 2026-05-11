"""Example 08: Toy Video World Model Analysis.

This example demonstrates using WorldModelLens with a non-RL video prediction model.
The ToyVideoAdapter implements a simple video world model with:
- CNN encoder from frames to latent
- Latent dynamics MLP (no actions)
- CNN decoder to reconstruct frames

This example shows:
1. Creating a toy video world model adapter
2. Running forward passes with HookedWorldModel
3. Analyzing latent representations without rewards/actions
4. Visualizing latent trajectories and surprise timelines

No RL-specific features are used - this works entirely with latent geometry analysis.
"""

import torch
import numpy as np

from world_model_lens import HookedWorldModel
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer
from world_model_lens.probing.geometry import GeometryAnalyzer
from world_model_lens.probing.temporal_memory import TemporalMemoryProber
from world_model_lens.visualization import plot_toy_video_dashboard
import pathlib

OUTPUT_DIR = pathlib.Path("assets/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_video_model():
    """Create a toy video world model."""
    from world_model_lens.backends.toy_video_model import ToyVideoAdapter

    config = type(
        "Config",
        (),
        {
            "latent_dim": 32,
            "hidden_dim": 128,
            "obs_channels": 3,
            "frame_size": 64,
        },
    )()

    adapter = ToyVideoAdapter(
        config=config,
        latent_dim=32,
        hidden_dim=128,
        obs_channels=3,
    )

    return HookedWorldModel(adapter=adapter, config=config, name="toy_video")


def generate_moving_pattern_video(n_frames: int = 30, frame_size: int = 64) -> torch.Tensor:
    """Generate a video with a moving pattern.

    Creates a video where a bright square moves diagonally across the frame.
    """
    frames = []
    for t in range(n_frames):
        frame = torch.zeros(3, frame_size, frame_size)

        x = int((t / n_frames) * (frame_size - 10))
        y = int((t / n_frames) * (frame_size - 10))

        frame[:, y : y + 10, x : x + 10] = 1.0

        frames.append(frame)

    return torch.stack(frames)


def main():
    print("=" * 60)
    print("WorldModelLens: Toy Video World Model Analysis")
    print("=" * 60)

    print("\n[1] Creating toy video world model...")
    wm = create_video_model()
    print(f"    Model name: {wm.name}")
    print(f"    Has decoder: {wm.capabilities.has_decoder}")
    print(f"    Uses actions: {wm.capabilities.uses_actions}")
    print(f"    Is RL model: {wm.capabilities.is_rl_model()}")

    print("\n[2] Generating synthetic video...")
    frames = generate_moving_pattern_video(n_frames=30, frame_size=64)
    print(f"    Video shape: {frames.shape}")

    print("\n[3] Running forward pass with caching...")
    traj, cache = wm.run_with_cache(frames)
    print(f"    Trajectory states: {len(traj.states)}")
    print(f"    Cache keys: {list(cache.keys())[:5]}...")

    print("\n[4] Belief Analysis (surprise timeline)...")
    belief_analyzer = BeliefAnalyzer(wm)
    surprise_result = belief_analyzer.surprise_timeline(cache)
    print(f"    Mean surprise: {surprise_result.mean_surprise:.4f}")
    print(
        f"    Max surprise: {surprise_result.max_surprise_value:.4f} at t={surprise_result.max_surprise_timestep}"
    )
    print(f"    Surprise peaks: {len(surprise_result.peaks)}")

    print("\n[5] Geometry Analysis...")
    geometry_analyzer = GeometryAnalyzer(wm)

    pca_result = geometry_analyzer.pca_projection(cache, component="z_posterior")
    print(
        f"    PCA components shape: {pca_result.pca_components.shape if pca_result.pca_components is not None else 'N/A'}"
    )

    traj_metrics = geometry_analyzer.trajectory_metrics(cache, component="z_posterior")
    print(f"    Mean trajectory distance: {traj_metrics.mean_trajectory_distance:.4f}")
    print(f"    Temporal coherence: {traj_metrics.temporal_coherence:.4f}")

    cluster_result = geometry_analyzer.clustering(cache, n_clusters=3, component="z_posterior")
    print(f"    Clusters: {len(torch.unique(cluster_result['clusters']))}")

    manifold_result = geometry_analyzer.manifold_analysis(cache, component="z_posterior")
    print(
        f"    Intrinsic dimensionality: {manifold_result['intrinsic_dimensionality_estimate']:.2f}"
    )

    print("\n[6] Temporal Memory Analysis...")
    temporal_prober = TemporalMemoryProber(wm)

    mem_result = temporal_prober.memory_retention(cache, component="z_posterior", max_lag=10)
    print(f"    Memory capacity: {mem_result.memory_capacity:.2f}")
    print(f"    Short-term retention: {mem_result.temporal_dependencies.get('short_term', 0):.4f}")
    print(f"    Long-term retention: {mem_result.temporal_dependencies.get('long_term', 0):.4f}")

    dep_result = temporal_prober.temporal_dependencies(cache, component="z_posterior", max_lag=5)
    print(f"    Autocorrelations computed: {len(dep_result['autocorrelations'])}")

    print("\n[7] Testing imagination (no actions needed)...")
    start_state = traj.states[0]
    imagined = wm.imagine(start_state, actions=None, horizon=10)
    print(f"    Imagined states: {len(imagined.states)}")

    print("\n[8] Testing RL-specific analysis (should be skipped)...")
    reward_result = belief_analyzer.reward_attribution(traj, cache)
    print(f"    Reward attribution available: {reward_result.is_available}")

    value_result = belief_analyzer.value_analysis(cache)
    print(f"    Value analysis available: {value_result.get('is_available', True)}")

    print("\n" + "=" * 60)
    print("Summary: Toy video model analysis complete!")
    print("=" * 60)
    print("\nThis demonstrates WorldModelLens working with a non-RL model:")
    print("  - Latent geometry analysis works")
    print("  - Temporal memory analysis works")
    print("  - Surprise timeline works")
    print("  - RL-specific analysis gracefully skipped")
    print()

    print("\n[9] Generating visualization dashboard...")
    plot_toy_video_dashboard(
        traj=traj,
        cache=cache,
        mem_result=mem_result,
        # surprise_result=surprise_result,
        # traj_metrics=traj_metrics,
        # manifold_result=manifold_result,
        # dep_result=dep_result,
        output_path=OUTPUT_DIR / "toy_video_dashboard.png",
    )
    print("    Saved toy_video_dashboard.png")


if __name__ == "__main__":
    main()
